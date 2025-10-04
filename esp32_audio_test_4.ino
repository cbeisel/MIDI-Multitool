
#include <WiFi.h>
#include <WiFiUdp.h>
#include <PubSubClient.h>
#include "driver/i2s.h"
#include <ArduinoJson.h>

// Wifi settings
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT settings
const char* mqtt_server = "192.168.0.99";   // Raspberry Pi IP
WiFiClient espClient;
PubSubClient client(espClient);

// UDP settings
WiFiUDP udp;
const char* udp_host = "192.168.0.129";  // PC or Raspberry Pi IP
const int udp_port = 5005;               // Port for audio streaming

// Audio settings
#define SAMPLE_RATE     16000
#define CHUNK_SIZE      512

// I2S pins (ICS43434 mic)
#define I2S_SD   10   // DOUT
#define I2S_SCK  11   // BCLK
#define I2S_WS   12   // LRCLK

// Start/Stop Button
#define BUTTON_PIN 13
bool recording = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Buffers
int32_t i2sBuffer[CHUNK_SIZE]; // raw 32-bit
int16_t chunk[CHUNK_SIZE];     // downscaled 16-bit

// Setup
void setup_wifi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("WiFi connected");
}

// MQTT messaging
void callbCallbackack(char* topic, byte* payload, unsigned int length) {
  // Copy payload into a null-terminated string
  String msg;
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }

  Serial.print("[MQTT] ");
  Serial.print(topic);
  Serial.print(" -> ");
  Serial.println(msg);

  // Parse JSON
  StaticJsonDocument<512> doc;  // adjust size if needed
  DeserializationError error = deserializeJson(doc, msg);

  if (error) {
    Serial.print("[ERROR] JSON parse failed: ");
    Serial.println(error.f_str());
    return;
  }

  // Example expected JSON: {"type":"feedback","status":"processed","tempo":120,"loop_length":4}
  const char* type = doc["type"];
  const char* status = doc["status"];
  int tempo = doc["tempo"] | -1;
  int loopLength = doc["loop_length"] | -1;

  Serial.println("[INFO] Parsed feedback:");
  if (type) Serial.printf("  type: %s\n", type);
  if (status) Serial.printf("  status: %s\n", status);
  if (tempo != -1) Serial.printf("  tempo: %d bpm\n", tempo);
  if (loopLength != -1) Serial.printf("  loop length: %d bars\n", loopLength);
}

// MQTT Setup
void reconnect() {
  while (!client.connected()) {
    if (client.connect("ESP32S3")) {
      client.publish("esp32/status", "{\"type\":\"status\",\"device\":\"esp32-s3\",\"state\":\"connected\"}");

      // Subscribe to feedback from RPi
      client.subscribe("rpi/feedback");
      Serial.println("Subscribed to rpi/feedback");
    } else {
      delay(1000);
    }
  }
}

// I2S microphone setup. Might need to modify this if using an I2S audio codec.
void setupI2SMic() {
  const i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config = {
    .bck_io_num = I2S_SCK,
    .ws_io_num = I2S_WS,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = I2S_SD
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);
}

// Setup code
void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);

  setup_wifi();
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  reconnect();

  setupI2SMic();
  udp.begin(udp_port);
}

// Loop code
void loop() {
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Button handling
  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    static bool buttonPressed = false;

    if (reading == LOW && !buttonPressed) {
      buttonPressed = true;
      recording = !recording;
      
      // This is to start and stop recording via button press. Change it to record while holding button?
      // Tap in/out is more similar to looping pedals, might be more intuitive to keep it this way.
      // Need to add voice commands as well?
      if (recording) {
        client.publish("esp32/audio_control", "{\"type\":\"control\",\"command\":\"start_recording\"}");
        Serial.println("Recording started");
      } else {
        client.publish("esp32/audio_control", "{\"type\":\"control\",\"command\":\"stop_recording\"}");
        Serial.println("Recording stopped");
      }
    }

    if (reading == HIGH && buttonPressed) {
      buttonPressed = false;
    }
  }

  lastButtonState = reading;

  // Stream audio over UDP. Can change to TCP later if losing packets. Do we dedicate one core to this?
  if (recording) {
    size_t bytes_read = 0;
    i2s_read(I2S_NUM_0, (void*)i2sBuffer, sizeof(i2sBuffer), &bytes_read, portMAX_DELAY);

    int samples = bytes_read / 4;
    for (int i = 0; i < samples && i < CHUNK_SIZE; i++) {
      chunk[i] = (int16_t)(i2sBuffer[i] >> 14); // downscale 24→16 bit, might just shift to 44.1khz, 24 bit if everything can keep up.
    }

    udp.beginPacket(udp_host, udp_port);
    udp.write((uint8_t*)chunk, CHUNK_SIZE * sizeof(int16_t));
    udp.endPacket();
    Serial.println("Sent UDP packet");
  }
}