// already in platformio.ini #define MQTT_MAX_PACKET_SIZE 32768 // Increase MQTT packet size for larger JSON messages
#include <Arduino.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <PubSubClient.h>
#include "driver/i2s.h"
#include <ArduinoJson.h>
#include <AppleMIDI.h>
#include <HardwareSerial.h>
#include <math.h>

USING_NAMESPACE_APPLEMIDI;

// WiFi settings - Replace with your network credentials
const char* ssid = "ssid";
const char* password = "password";

// MQTT settings
const char* mqtt_server = "192.168.0.150";   // Jetson IP
WiFiClient espClient;
PubSubClient client(espClient);

// UDP settings
WiFiUDP udp;
const char* udp_host = "192.168.0.150";  // Jetson IP
const int udp_port = 5006;               // UDP port for audio streaming
const int applemidi_port = 5004;         // AppleMIDI standard port

// MIDI settings
static const int MIDI_TX_PIN = GPIO_NUM_39; // USB TX
static const int MIDI_RX_PIN = -1;          // Not used
HardwareSerial MidiUart(1);                 // Use UART1 for MIDI output

APPLEMIDI_CREATE_INSTANCE(WiFiUDP, AppleMIDI, "ESP32S3_MMT", applemidi_port);

// DC-block filter state
static int32_t x_prev = 0;
static float   y_prev = 0.0f;

// For level monitoring
static int16_t local_min = 32767;
static int16_t local_max = -32768;

// Filter coefficient (you already have R = 0.995f in your code)
static const float R = 0.995f;

// Select analog audio source
// 0 = ICS43434 on I2S0
// 1 = WM8782 on I2S1
int currentSource  = 1;
i2s_port_t activeI2SPort = I2S_NUM_1;
bool i2sInitialized = false;

// Audio settings
#define SAMPLE_RATE     22050
#define CHUNK_SIZE      512

// I2S pins (ICS43434 mic)
#define ICS_DOUT_PIN   10   // DOUT
#define ICS_BCLK_PIN   11   // BCLK
#define ICS_LRCLK_PIN  12   // LRCLK

// I2S pins (WM8782 ADC)
#define WM8782_BCLK_PIN   16   // BCLK
#define WM8782_LRCLK_PIN  17   // LRCLK
#define WM8782_DOUT_PIN   18   // DOUT
#define WM8782_MCLK_PIN   3   // MCLK
// Start/Stop Recording Button
#define BUTTON_PIN 13
bool recording = false;
bool lastButtonState = HIGH;
unsigned long lastDebounceTime = 0;
const unsigned long debounceDelay = 50;

// Loop Button
#define LOOP_PIN 9
bool loopLastButtonState = HIGH;
unsigned long loopLastDebounceTime = 0;
const unsigned long loopDebounceDelay = 50;

// Buffers
int32_t i2sBuffer[CHUNK_SIZE * 2]; // stereo buffer
int16_t chunk[CHUNK_SIZE];         // mono buffer for UDP


// Note event structure
struct NoteEvent {
  uint32_t t_ms;    // timestamp in milliseconds
  uint8_t status;   // MIDI status byte (includes channel)
  uint8_t d1;       // first data byte (note number)
  uint8_t d2;       // second data byte (velocity)
};

// Pattern storage
static const int MAX_PATTERN_EVENTS = 512;
NoteEvent patternEvents[MAX_PATTERN_EVENTS];
int patternEventCount = 0;

// Pattern playback state
bool patternReady = false;        // A complete pattern has been received
bool patternPlaying = false;      // Pattern is being played back
uint32_t patternStartMillis = 0;  // When playback started
int currentEventIndex = 0;
uint32_t patternLengthMs = 0;     // Total length of the pattern in ms
float patternTempoBPM = 120.0f;   // Tempo of the pattern

// Loop counter
uint32_t patternLoopCount = 0;    // Number of times the pattern has looped

// MIDI handlers
void sendMIDIMessage(byte command, byte data1, byte data2) {
  MidiUart.write(command);
  MidiUart.write(data1);
  MidiUart.write(data2);
}

// Note On handler
void sendNoteOnRaw(byte channel, byte note, byte velocity) {
  uint8_t command = 0x90 | ((channel - 1) & 0x0F); // channel 1-16
  sendMIDIMessage(command, note, velocity);
}

// Note Off handler
void sendNoteOffRaw(byte channel, byte note, byte velocity) {
  uint8_t command = 0x80 | ((channel - 1) & 0x0F); // channel 1-16
  sendMIDIMessage(command, note, velocity);
};

// MIDI Note On handler
void OnNoteOn(byte channel, byte note, byte velocity) {
  Serial.printf("[MIDI] Note On  - Channel: %d, Note: %d, Velocity: %d\n", channel, note, velocity);
  sendNoteOnRaw(channel, note, velocity);
}

// MIDI Note Off handler
void OnNoteOff(byte channel, byte note, byte velocity) {
  Serial.printf("[MIDI] Note Off - Channel: %d, Note: %d, Velocity: %d\n", channel, note, velocity);
  sendNoteOffRaw(channel, note, velocity);
}

// AppleMIDI note on handler
void rtpHandleNoteOn(byte channel, byte note, byte velocity) {
  Serial.printf("[RTP-MIDI] Note On  ch=%d note=%d vel=%d\n", channel, note, velocity);
}

// AppleMIDI note off handler
void rtpHandleNoteOff(byte channel, byte note, byte velocity) {
  Serial.printf("[RTP-MIDI] Note Off ch=%d note=%d vel=%d\n", channel, note, velocity);
}

// Setup WiFi
void setup_wifi() {
  Serial.print("Connecting to WiFi");
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("\nWiFi connection failed!");
  }
}

// MQTT callback
void callback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) {
    msg += (char)payload[i];
  }

  Serial.print("[MQTT] ");
  Serial.print(topic);
  Serial.print(" -> ");
  Serial.println(msg);

  // Parse JSON
  DynamicJsonDocument doc(4096);
  DeserializationError error = deserializeJson(doc, msg);

  if (error) {
    Serial.print("[ERROR] JSON parse failed: ");
    Serial.println(error.f_str());
    return;
  }

  const char* type = doc["type"] | "";
  if (type[0] == '\0') {
    Serial.println("[WARN] 'type' field missing in JSON");
    return;
  } 

  // Handle feedback messages
  
  if (strcmp(type, "feedback") == 0) {
    const char* status = doc["status"];
    int tempo = doc["tempo"] | -1;
    int loopLength = doc["loop_length"] | -1;
    int notesCount = doc["notes_count"] | -1;

    Serial.println("[INFO] Received feedback:");
    if (status) Serial.printf("  status: %s\n", status);
    if (notesCount != -1) Serial.printf("  notes detected: %d\n", notesCount);
    if (tempo != -1) Serial.printf("  tempo: %d bpm\n", tempo);
    if (loopLength != -1) Serial.printf("  loop length: %d bars\n", loopLength);
  }
  else if (strcmp(type, "analysis_result") == 0) {
    const char* sourceFile = doc["source_file"];
    const char* midiFile = doc["midi_file"];
    
    Serial.println("[INFO] Analysis complete:");
    if (sourceFile) Serial.printf("  source: %s\n", sourceFile);
    if (midiFile) Serial.printf("  midi: %s\n", midiFile);
  }
  else if (strcmp(type, "note_events") == 0) {

    // Stop current playback
    patternPlaying = false;
    patternReady = false;
    patternEventCount = 0;
    patternLoopCount = 0;

    // Handle incoming note events for pattern playback
    JsonArray evArray = doc["events"].as<JsonArray>();
    if (evArray.isNull()) {
      Serial.println("[WARN] note_events has no 'events' array");
      return;
    } 

    patternTempoBPM = doc["tempo"] | 120.0f;
    float patternLengthSec = doc["pattern_length"] | 0.0f;

    // Parse events
    for (JsonObject ev : evArray) {
      if (patternEventCount >= MAX_PATTERN_EVENTS) {
        Serial.println("[WARN] Max pattern events reached, ignoring extra events");
        break;
      }

      float t_sec = ev["t"] | 0.0f; //
      uint32_t t_ms = (uint32_t)(t_sec * 1000.0f + 0.5f); // convert to ms

      // MIDI event data
      uint8_t status = ev["status"] | 0;
      uint8_t d1 = ev["d1"] | 0;
      uint8_t d2 = ev["d2"] | 0;

      // Store MIDI event
      patternEvents[patternEventCount].t_ms = t_ms;
      patternEvents[patternEventCount].status = status;
      patternEvents[patternEventCount].d1 = d1;
      patternEvents[patternEventCount].d2 = d2;
      patternEventCount++;
    }
    
    // Finalize pattern setup
    if (patternEventCount > 0) {
      patternLengthMs = (uint32_t)(patternLengthSec * 1000.0f + 0.5f);
      if (patternLengthMs == 0) {
        // Infer length from last event
        patternLengthMs = patternEvents[patternEventCount - 1].t_ms;
      }
      
      // Reset playback state
      currentEventIndex = 0;
      patternLoopCount = 0;
      patternReady = true;
      patternPlaying = true;
      patternStartMillis = millis();

      Serial.printf("[MMT] Loaded pattern: %d events, length=%.2f sec, tempo=%.2f bpm\n",
                    patternEventCount,
                    patternLengthMs / 1000.0f,
                    patternTempoBPM);
  } else {
      Serial.println("[WARN] No note events received");
      patternReady = false;
      patternPlaying = false;
    }
  }
}

// MQTT reconnection
void reconnect() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection...");
    
    if (client.connect("ESP32S3_Audio")) {
      Serial.println("connected");
      
      // Publish connection status
      client.publish("mmt/status", "{\"type\":\"status\",\"device\":\"esp32-s3\",\"state\":\"connected\"}");

      // Subscribe to correct topics that Jetson publishes to
      client.subscribe("mmt/feedback");
      client.subscribe("mmt/analysis_result");
      client.subscribe("mmt/note_events");

      Serial.println("Subscribed to mmt/feedback, mmt/analysis_result, and mmt/note_events");

      return;
    } else {
      Serial.print("failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 2 seconds");
      delay(2000);
    }
  }
}

// MIDI loop scheduler
void handlePatternPlayback() {
  if (!patternPlaying || !patternReady || patternEventCount == 0) {
    return;
  }

  if (patternLengthMs == 0) {
    patternLengthMs = patternEvents[patternEventCount - 1].t_ms;
    if (patternLengthMs == 0) {
      return;
    }
  }

  // Calculate elapsed time
  uint32_t now = millis();
 
  for (int safety = 0; safety < patternEventCount; ++safety) {
    uint32_t eventOffsetMs =
      patternEvents[currentEventIndex].t_ms +
      (patternLengthMs * patternLoopCount);    
    uint32_t targetTime = patternStartMillis + eventOffsetMs;

    int32_t timeDiff = (int32_t)(now - targetTime);
    if (timeDiff < 0) {
      break; // next event not yet due
    }
  
    // Time to send this event
    const NoteEvent& ev = patternEvents[currentEventIndex];
    sendMIDIMessage(ev.status, ev.d1, ev.d2);

    // Advance to next event
    currentEventIndex++;
    if (currentEventIndex >= patternEventCount) {
      // Loop back to start
      currentEventIndex = 0;
      patternLoopCount++;
    }

    // Update now for next iteration
    now = millis();
  }
}

// I2S microphone setup
void setupI2S_ICS43434() {
  const i2s_config_t i2s_config_ics = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 4,
    .dma_buf_len = 256,
    .use_apll = false,
    .tx_desc_auto_clear = false,
    .fixed_mclk = 0
  };

  const i2s_pin_config_t pin_config_ics = {
    .bck_io_num = ICS_BCLK_PIN,
    .ws_io_num = ICS_LRCLK_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = ICS_DOUT_PIN
  };

  esp_err_t result = i2s_driver_install(I2S_NUM_0,
                                        &i2s_config_ics,
                                        0,
                                        NULL);
  if (result != ESP_OK) {
    Serial.printf("I2S0 (ICS43434) driver install failed: %d\n", result);
  }
  
  result = i2s_set_pin(I2S_NUM_0, &pin_config_ics);
  if (result != ESP_OK) {
    Serial.printf("I2S0 (ICS43434) pin config failed: %d\n", result);
  } else {
    Serial.println("I2S0 (ICS43434) microphone initialized successfully");
  }
}

// I2S WM8782 setup
void setupI2S_WM8782() {
  const i2s_config_t i2s_config_pcm = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
  .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = 8,
    .dma_buf_len = 256,
    .use_apll = true,
    .tx_desc_auto_clear = false,
    .fixed_mclk = SAMPLE_RATE * 256  // Set MCLK to 256x sample rate
  };

  const i2s_pin_config_t pin_config_pcm = {
    .mck_io_num = WM8782_MCLK_PIN,
    .bck_io_num = WM8782_BCLK_PIN,
    .ws_io_num = WM8782_LRCLK_PIN,
    .data_out_num = I2S_PIN_NO_CHANGE,
    .data_in_num = WM8782_DOUT_PIN
  };

  // Install and start I2S driver
  esp_err_t result = i2s_driver_install(I2S_NUM_1,
                                        &i2s_config_pcm,
                                        0,
                                        NULL);

  if (result != ESP_OK) {
    Serial.printf("I2S1 (WM8782) driver install failed: %d\n", result);
  }
  result = i2s_set_pin(I2S_NUM_1, &pin_config_pcm);
  if (result != ESP_OK) {
    Serial.printf("I2S1 (WM8782) pin config failed: %d\n", result);
  } else {
    Serial.println("I2S1 (WM8782) initialized successfully");
  }
}

// Function to switch audio source
void switchAudioSource(int newSource) {
  if (newSource != 0 && newSource != 1) return;

  // Stop recording before switching
  bool prevRecording = recording;
  recording = false;

  // Delay a moment to let last read finish
  delay(50);

  // Uninstall existing I2S driver if initialized
  if (i2sInitialized) {
    i2s_driver_uninstall(activeI2SPort);
  }

  // Install the new one
  if (newSource == 0) {
    setupI2S_ICS43434();
    activeI2SPort = I2S_NUM_0;
    Serial.println("[AUDIO] Switched to ICS43434 (I2S0)");
  } 
  else {
    setupI2S_WM8782();
    activeI2SPort = I2S_NUM_1;
    Serial.println("[AUDIO] Switched to WM8782 (I2S1)");
  }

  currentSource = newSource;
  i2sInitialized = true;

  // Resume recording if previously active
  recording = prevRecording;
}

void setup() {
  Serial.begin(115200);
  delay(3000);
  
  Serial.println("\n-=[ ESP32-S3 MIDI Multitool ]=-");
  
  // Setup WiFi and MQTT
  setup_wifi();
  WiFi.setSleep(WIFI_PS_NONE); // Disable WiFi sleep
  client.setServer(mqtt_server, 1883);
  client.setCallback(callback);
  reconnect();

  // Print WiFi signal strength
  Serial.print("RSSI: ");
  Serial.println(WiFi.RSSI());

  //
  AppleMIDI.begin();
  AppleMIDI.setHandleNoteOn(rtpHandleNoteOn);
  AppleMIDI.setHandleNoteOff(rtpHandleNoteOff);

  // Setup MIDI UART
  MidiUart.begin(31250, SERIAL_8N1, MIDI_RX_PIN, MIDI_TX_PIN); // Standard MIDI baud rate

  // Initialize I2S analog audio sources
  switchAudioSource(currentSource);

  udp.begin(udp_port);

  // Setup buttons
  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LOOP_PIN, INPUT_PULLUP);
 
  Serial.println("System ready!");
}

void loop() {
  // Maintain MQTT connection
  if (!client.connected()) {
    reconnect();
  }
  client.loop();

  // Handle AppleMIDI
  AppleMIDI.read();

  // Handle pattern playback
  handlePatternPlayback();

  // Button handling with debouncing
  bool reading = digitalRead(BUTTON_PIN);

  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  if ((millis() - lastDebounceTime) > debounceDelay) {
    static bool buttonPressed = false;

    if (reading == LOW && !buttonPressed) {
      buttonPressed = true;
      recording = !recording;
      
      if (recording) {
        client.publish("mmt/audio_control", "{\"type\":\"control\",\"command\":\"start_recording\"}");
        Serial.println("[BUTTON] Recording started");
      } else {
        client.publish("mmt/audio_control", "{\"type\":\"control\",\"command\":\"stop_recording\"}");
        Serial.println("[BUTTON] Recording stopped");
      }
    }

    if (reading == HIGH && buttonPressed) {
      buttonPressed = false;
    }
  }

  lastButtonState = reading;

  // Loop button handling with debouncing
  bool loopReading = digitalRead(LOOP_PIN);

  if (loopReading != loopLastButtonState) {
    loopLastDebounceTime = millis();
  }
  if ((millis() - loopLastDebounceTime) > loopDebounceDelay) {
    static bool loopButtonPressed = false;
    if (loopReading == LOW && !loopButtonPressed) {
      loopButtonPressed = true;
      if (!patternReady) {
        // No pattern loaded
        Serial.println("[LOOP] No pattern loaded");
      } else {
        if (patternPlaying) {
          // Stop pattern playback
          patternPlaying = false;
          Serial.println("[LOOP] Pattern playback stopped");
        } else {
          // Restart pattern playback from beginning
          currentEventIndex = 0;
          patternLoopCount = 0;
          patternStartMillis = millis();
          patternPlaying = true;
          Serial.println("[LOOP] Pattern playback started");
        }
      }
    }

    if (loopReading == HIGH && loopButtonPressed) {
      loopButtonPressed = false;
    }
  }
  loopLastButtonState = loopReading;

  // Stream audio over UDP when recording
  if (recording) {
    size_t bytes_read = 0;
    esp_err_t result = i2s_read(activeI2SPort,
                                (void*)i2sBuffer,
                                sizeof(i2sBuffer),
                                &bytes_read,
                                portMAX_DELAY);
    
    
    if (result == ESP_OK && bytes_read > 0) {
      // Process audio data: DC-block, gain, limiting
      const float R = 0.995f;  // DC-block filter coefficient
      int total_words = bytes_read / 4;   // 32-bit words (L,R,L,R,...)
      int frames = total_words / 2;       // stereo frames
      if (frames > CHUNK_SIZE) frames = CHUNK_SIZE;

      // Process audio: extract left channel, DC-block, gain, limiting
      for (int i = 0; i < frames; i++) {
        int32_t s32 = (int32_t)i2sBuffer[2 * i];   // left channel only

        // 24->16 bit conversion
        int16_t s16 = (int16_t)(s32 >> 8);

        if (s16 < local_min) local_min = s16;
        if (s16 > local_max) local_max = s16;

        // DC-block: y[n] = x[n] - x[n-1] + R*y[n-1]
        int32_t x = s16;
        float y = (float)(x - x_prev) + R * y_prev;
        x_prev = x;
        y_prev = y;

        // Apply gain
        const float gain = .5f;
        float boosted = y * gain;

        // Soft limiting
        const float limit_thresh = 24000.0f;
        float abs_boosted = fabsf(boosted);
        if (abs_boosted > limit_thresh) {
          float exceed = abs_boosted - limit_thresh;
          boosted -= copysignf(exceed * 0.75f, boosted);
        }

        // Hard limit to int16 range
        if (boosted >  32767.0f) boosted =  32767.0f;
        if (boosted < -32768.0f) boosted = -32768.0f;

        // Store in chunk buffer
        chunk[i] = (int16_t)boosted;

        //chunk[i] = s16; // Bypass processing for testing
      }

      // Send UDP packet
      if (frames > 0) {    
        int udpResult = udp.beginPacket(udp_host, udp_port);
        if (udpResult == 1) {
          size_t bytes_to_send = frames * sizeof(int16_t);
          size_t bytes_sent = udp.write((uint8_t*)chunk, bytes_to_send); // frames * 2 bytes per sample
          if (bytes_sent != bytes_to_send) {
            Serial.printf("[UDP] Short write: sent %d of %d bytes\n",
                          bytes_sent, bytes_to_send);
          }

          if (udp.endPacket() != 1) {
            Serial.println("[UDP] endPacket failed");
          }
        } else {
          Serial.println("[UDP] beginPacket failed");
        }
        
        delayMicroseconds(500); // Throttle to avoid flooding

        // Print status every 50 packets (~1 second at 22050Hz)
        static int packetCount = 0;
        packetCount++;
        if (packetCount % 50 == 0) {
          Serial.printf("[UDP] Sent %d packets (%d bytes each), min/max=%d/%d\n",
                        packetCount,
                        frames * 2,
                        local_min,
                        local_max);
          local_min = 32767;
          local_max = -32768;
        }
      }
        
    } else if (result != ESP_OK) {
      Serial.printf("[ERROR] I2S read failed: %d\n", result);
    } else {
      Serial.println("[WARN] I2S read returned 0 bytes");
    }
  }

  // Heartbeat publishing every 5 seconds
  static unsigned long lastHeartbeat = 0;
  if (millis() - lastHeartbeat > 5000) {
    lastHeartbeat = millis();
    if (client.connected()) {
      client.publish("mmt/status", "{\"type\":\"heartbeat\",\"state\":\"alive\"}");
    }
  }
}