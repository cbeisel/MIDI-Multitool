# MIDI Multitool

IoT device based on ESP32-S3 with edge processing (either with Raspberry Pi or Jetson Orin Nano).

Portable audio interface device with controls and a microphone and/or audio interface that takes audio, sends it to an edge computing device for automatic music transcription (AMT) with MIDI and other potential information sent back the IoT device.

main.cpp is the ESP32-S3 code. An Adafruit ESP32-S3 Feather with 4MB Flash and 2MB PSRAM was used. Built with PlatformIO in VS Code.

mmt_jetson51.py is the Python script to run the edge processing on an Nvidia Jetson Orin Nano. The Jetson Orin Nano is also hosting MQTT.



