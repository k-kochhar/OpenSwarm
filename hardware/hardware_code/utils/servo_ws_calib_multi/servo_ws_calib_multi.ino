#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ESP32Servo.h>

using namespace websockets;

Servo servo;

const int SERVO_PIN = 4;

const char* WIFI_SSID = "Aditya";
const char* WIFI_PASS = "123456789";

const char* WS_URL = "ws://10.96.28.112:8765";

// 🔴 CHANGE THIS FOR EACH ESP32
const char* DEVICE_ID = "ESP2";

WebsocketsClient client;

void onMessageCallback(WebsocketsMessage message) {
  String msg = message.data();
  msg.trim();

  int value = msg.toInt();

  if (value >= 0 && value <= 180) {
    servo.write(value);
    Serial.print("Servo set to: ");
    Serial.println(value);
  }
}

void onEventsCallback(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    Serial.println("WS Connected");

    // 🔹 Send ID immediately after connect
    client.send(String("ID:") + DEVICE_ID);
  }
}

void setup() {
  Serial.begin(115200);

  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2500);
  servo.write(90);

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }

  Serial.println("\nWiFi connected");

  client.onMessage(onMessageCallback);
  client.onEvent(onEventsCallback);

  client.connect(WS_URL);
}

void loop() {
  client.poll();

  // Auto reconnect
  if (WiFi.status() == WL_CONNECTED && !client.available()) {
    static unsigned long lastTry = 0;
    if (millis() - lastTry > 3000) {
      lastTry = millis();
      client.connect(WS_URL);
    }
  }
}
