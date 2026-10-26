// MLX90614 live stream + pan/tilt servo control for the fusion console
// (12_WebApp/console.html, also 05_Code/fusion_alarm_demo.py for the sensor part).
//
// Wiring:  MLX90614 VIN->3V3, GND->GND, SDA->D21, SCL->D22
//          Pan servo signal -> GPIO18, tilt servo signal -> GPIO19
//          Servo red/brown from the SEPARATE 5V supply, its GND tied to an ESP32 GND
//          (servos on the ESP32's own 5V pin brown it out and it reboots).
//
// Output lines:
//   T,<ambient C>,<object C>     every 100 ms
//   S,<pan>,<tilt>,<mode>        every 200 ms   (mode: WAIT / PATROL / HOLD / AIM)
// Input commands (one per line):
//   P            patrol: slow sweep
//   H            hold the current position
//   C            centre (90, 90) and hold
//   A<pan>,<tilt>  glide to that angle and hold, e.g. A75,95
// At boot the servos are not driven at all (WAIT): they stay exactly where they
// are until the console sends its first command, which it only does after
// Calibrate. After CMD_TIMEOUT_MS without a command the gimbal stops where it is
// and goes back to WAIT, so a closed browser tab leaves it still, not sweeping.
//
// Libraries: "Adafruit MLX90614 Library" (+ Adafruit BusIO), "ESP32Servo".

#include <Wire.h>
#include <Adafruit_MLX90614.h>
#include <ESP32Servo.h>

const int SDA_PIN = 21, SCL_PIN = 22;
const int PAN_PIN = 18, TILT_PIN = 19;

// Same window as servo_bringup_test, known not to hit the bracket's stops.
const int PAN_MIN = 60, PAN_MAX = 120;
const int TILT_MIN = 70, TILT_MAX = 110;

const unsigned long TEMP_PERIOD_MS = 100;
const unsigned long STATUS_PERIOD_MS = 200;
const unsigned long SERVO_TICK_MS = 20;
const float MAX_SPEED_DPS = 90.0;      // glide speed for aim / centre moves
const float PATROL_SPEED_DPS = 25.0;   // slow, readable sweep
const unsigned long CMD_TIMEOUT_MS = 3000;

Adafruit_MLX90614 mlx = Adafruit_MLX90614();
Servo panServo, tiltServo;
bool sensorOk = false;

enum Mode { WAIT, PATROL, HOLD, AIM };
Mode mode = WAIT;
bool servosAttached = false;
float pan = 90, tilt = 90;             // current commanded angle
float panTarget = 90, tiltTarget = 90;
int patrolDir = 1;
unsigned long lastCmdAt = 0, lastTemp = 0, lastStatus = 0, lastTick = 0;
char cmd[24];
int cmdLen = 0;

const char *modeName() {
  return mode == WAIT ? "WAIT" : mode == PATROL ? "PATROL" : mode == HOLD ? "HOLD" : "AIM";
}

// Attached once, on the first command, never detached: before this the servo
// pins carry no signal, so the servos don't jump or move at power-up.
void attachServos() {
  if (servosAttached) return;
  panServo.attach(PAN_PIN);
  tiltServo.attach(TILT_PIN);
  panServo.write(90);
  tiltServo.write(90);
  servosAttached = true;
  Serial.println("Servos armed (pan GPIO18, tilt GPIO19)");
}

float stepToward(float cur, float target, float maxStep) {
  float d = target - cur;
  if (d > maxStep) return cur + maxStep;
  if (d < -maxStep) return cur - maxStep;
  return target;
}

void handleCommand(const char *c) {
  if (!strchr("PpHhCcAa", c[0])) return;
  lastCmdAt = millis();
  attachServos();
  switch (c[0]) {
    case 'P': case 'p': mode = PATROL; break;
    case 'H': case 'h': mode = HOLD; panTarget = pan; tiltTarget = tilt; break;
    case 'C': case 'c': mode = AIM; panTarget = 90; tiltTarget = 90; break;
    case 'A': case 'a': {
      int p, t;
      if (sscanf(c + 1, "%d,%d", &p, &t) == 2) {
        mode = AIM;
        panTarget = constrain(p, PAN_MIN, PAN_MAX);
        tiltTarget = constrain(t, TILT_MIN, TILT_MAX);
      }
      break;
    }
  }
}

void readCommands() {
  while (Serial.available()) {
    char ch = Serial.read();
    if (ch == '\n' || ch == '\r') {
      if (cmdLen > 0) { cmd[cmdLen] = 0; handleCommand(cmd); cmdLen = 0; }
    } else if (cmdLen < (int)sizeof(cmd) - 1) {
      cmd[cmdLen++] = ch;
    }
  }
}

void updateServos(unsigned long now) {
  float dt = (now - lastTick) / 1000.0;
  lastTick = now;
  if (mode != WAIT && now - lastCmdAt > CMD_TIMEOUT_MS) mode = WAIT;
  if (mode == WAIT) return;              // not driven, or stopped where it is

  if (mode == PATROL) {
    pan += patrolDir * PATROL_SPEED_DPS * dt;
    if (pan >= PAN_MAX) { pan = PAN_MAX; patrolDir = -1; }
    if (pan <= PAN_MIN) { pan = PAN_MIN; patrolDir = 1; }
    tilt = stepToward(tilt, 90, MAX_SPEED_DPS * dt);
  } else {
    pan = stepToward(pan, panTarget, MAX_SPEED_DPS * dt);
    tilt = stepToward(tilt, tiltTarget, MAX_SPEED_DPS * dt);
  }
  panServo.write((int)(pan + 0.5));
  tiltServo.write((int)(tilt + 0.5));
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Printed before any I2C or servo call, so a blank Serial Monitor points at
  // serial/port/baud rather than the sensor or the servos.
  Serial.println();
  Serial.println("=== BOOT OK - mlx90614_stream + servos ===");

  // Servos are attached later, by attachServos(), with plain attach() and no
  // manual timer allocation: that froze the servos before.
  Serial.println("Servos waiting - press Calibrate in the console to start them");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setTimeOut(200); // don't block forever if an I2C line is stuck low
  sensorOk = mlx.begin();
  // The servos keep working without the sensor, so this is no longer fatal.
  Serial.println(sensorOk ? "MLX90614 OK - streaming T,ambient,object"
                          : "MLX90614 NOT FOUND - check SDA->D21, SCL->D22, VIN->3V3. Servos still run.");
  lastTick = millis();
}

void loop() {
  unsigned long now = millis();
  readCommands();

  if (now - lastTick >= SERVO_TICK_MS) updateServos(now);

  if (sensorOk && now - lastTemp >= TEMP_PERIOD_MS) {
    lastTemp = now;
    float ambient = mlx.readAmbientTempC();
    float object = mlx.readObjectTempC();
    if (isnan(ambient) || isnan(object)) Serial.println("ERR,read failed (NaN)");
    else Serial.printf("T,%.2f,%.2f\n", ambient, object);
  }

  if (now - lastStatus >= STATUS_PERIOD_MS) {
    lastStatus = now;
    Serial.printf("S,%d,%d,%s\n", (int)(pan + 0.5), (int)(tilt + 0.5), modeName());
  }
}
