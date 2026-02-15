#include <ESP32Servo.h>

Servo servo;

const int SERVO_PIN = 3;   // Change to your GPIO pin

void setup() {
  Serial.begin(115200);
  delay(1000);

  servo.setPeriodHertz(50);       // 50Hz servo signal
  servo.attach(SERVO_PIN, 500, 2500);

  Serial.println("Enter value from 0 to 180:");
}

void loop() {
  if (Serial.available()) {
    int value = Serial.parseInt();   // Read number

    if (value >= 0 && value <= 180) {
      servo.write(value);
      Serial.print("Servo set to: ");
      Serial.println(value);
    } else {
      Serial.println("Enter number between 0 and 180");
    }

    // Clear remaining serial buffer
    while (Serial.available()) Serial.read();
  }
}
