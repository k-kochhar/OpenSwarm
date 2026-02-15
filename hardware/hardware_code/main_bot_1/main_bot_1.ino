#include <WiFi.h>
#include <ArduinoWebsockets.h>
#include <ESP32Servo.h>

using namespace websockets;

const char* WIFI_SSID = "Aditya";
const char* WIFI_PASS = "123456789";
const char* WS_URL    = "ws://192.168.137.178:8765";

// CHANGE PER ROBOT
const char* DEVICE_ID = "ESP1";

// Servo pins
const int LEFT_PIN  = 3;
const int RIGHT_PIN = 6;

Servo leftServo;
Servo rightServo;

WebsocketsClient client;

// --- Your calibration ---
// Left: neutral ~90, forward higher, backward lower
const int L_NEUTRAL = 90;
const int L_FWD     = 96;  // neutral + 6
const int L_BACK    = 88;  // neutral - 4

// Right: neutral ~88, reversed direction
const int R_NEUTRAL = 88;
const int R_FWD     = 83;  // neutral - 4 (forward)
const int R_BACK    = 91;  // neutral + 5 (backward)

// ===== Tick (nudge) tuning =====
// Forward/back nudges
const unsigned long MOVE_ON_MS  = 100;   // move pulse duration
const unsigned long MOVE_OFF_MS = 20;  // pause between pulses

// Left/right nudges
const unsigned long TURN_ON_MS  = 100;   // turn pulse duration
const unsigned long TURN_OFF_MS = 20;  // pause between pulses

// Current command from websocket
volatile char currentCmd = 'S';

// Nudge state
bool pulseOn = false;
unsigned long phaseStart = 0;

// ---- motor helpers ----
void stopMotors() {
  leftServo.write(L_NEUTRAL);
  rightServo.write(R_NEUTRAL);
}

void applyForward() {
  leftServo.write(L_FWD);
  rightServo.write(R_FWD);
}

void applyBackward() {
  leftServo.write(L_BACK);
  rightServo.write(R_BACK);
}

void applyLeft() {   // spin left
  leftServo.write(L_BACK);
  rightServo.write(R_FWD);
}

void applyRight() {  // spin right
  leftServo.write(L_FWD);
  rightServo.write(R_BACK);
}

// Call this continuously from loop() (non-blocking)
void updateMotion() {
  char c = currentCmd;
  unsigned long now = millis();

  // Stop / default
  if (c != 'F' && c != 'B' && c != 'L' && c != 'R') {
    pulseOn = false;
    phaseStart = 0;
    stopMotors();
    return;
  }

  // Init when command starts
  if (phaseStart == 0) {
    pulseOn = true;
    phaseStart = now;
  }

  // Choose timings based on motion type
  const bool isMove = (c == 'F' || c == 'B');
  unsigned long onTime  = isMove ? MOVE_ON_MS  : TURN_ON_MS;
  unsigned long offTime = isMove ? MOVE_OFF_MS : TURN_OFF_MS;

  if (pulseOn) {
    // ON phase
    if (c == 'F') applyForward();
    else if (c == 'B') applyBackward();
    else if (c == 'L') applyLeft();
    else if (c == 'R') applyRight();

    if (now - phaseStart >= onTime) {
      pulseOn = false;
      phaseStart = now;
    }
  } else {
    // OFF phase
    stopMotors();

    if (now - phaseStart >= offTime) {
      pulseOn = true;
      phaseStart = now;
    }
  }
}

void onMessageCallback(WebsocketsMessage message) {
  String msg = message.data();
  msg.trim();
  if (msg.length() == 0) return;

  // Expect single-letter commands: F/B/L/R/S
  char c = msg.charAt(0);
  currentCmd = c;

  // Reset timing when switching commands (gives clean ticking)
  pulseOn = false;
  phaseStart = 0;

  Serial.print("CMD: ");
  Serial.println(c);
}

void onEventsCallback(WebsocketsEvent event, String data) {
  if (event == WebsocketsEvent::ConnectionOpened) {
    Serial.println("WS Connected");
    client.send(String("ID:") + DEVICE_ID);

    // Start stopped
    currentCmd = 'S';
    stopMotors();
  } else if (event == WebsocketsEvent::ConnectionClosed) {
    Serial.println("WS Disconnected");
    currentCmd = 'S';
    stopMotors(); // safety stop
  }
}

void setup() {
  Serial.begin(115200);
  delay(300);

  leftServo.setPeriodHertz(50);
  rightServo.setPeriodHertz(50);

  // Typical continuous servo pulse range (adjust if needed)
  leftServo.attach(LEFT_PIN, 500, 2500);
  rightServo.attach(RIGHT_PIN, 500, 2500);

  stopMotors();

  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(300);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("WiFi connected, IP: ");
  Serial.println(WiFi.localIP());

  client.onMessage(onMessageCallback);
  client.onEvent(onEventsCallback);

  Serial.print("Connecting WS: ");
  Serial.println(WS_URL);

  bool ok = client.connect(WS_URL);
  Serial.println(ok ? "WS connect OK" : "WS connect FAILED");
}

void loop() {
  client.poll();
  updateMotion();

  // reconnect logic
  if (WiFi.status() == WL_CONNECTED && !client.available()) {
    static unsigned long lastTry = 0;
    if (millis() - lastTry > 3000) {
      lastTry = millis();
      Serial.println("Trying WS reconnect...");
      client.connect(WS_URL);
    }
  }
}
