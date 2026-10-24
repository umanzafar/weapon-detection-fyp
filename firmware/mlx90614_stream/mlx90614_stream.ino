// MLX90614 live stream for the fusion alarm demo (05_Code/fusion_alarm_demo.py).
// Same wiring as mlx90614_test: VIN->3V3, GND->GND, SDA->D21, SCL->D22.
//
// Prints one machine-readable line every PERIOD_MS:
//   T,<ambient C>,<object C>
// Anything else on the line (boot messages, errors) is ignored by the laptop side.
//
// Library needed: "Adafruit MLX90614 Library" (accept the Adafruit BusIO dependency).

#include <Wire.h>
#include <Adafruit_MLX90614.h>

const int SDA_PIN = 21;
const int SCL_PIN = 22;
const unsigned long PERIOD_MS = 100; // 10 Hz - fast enough for an "instant" alarm

Adafruit_MLX90614 mlx = Adafruit_MLX90614();

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Printed before any I2C call, so a blank Serial Monitor rules the sensor out
  // and points at serial/port/baud instead.
  Serial.println();
  Serial.println("=== BOOT OK - mlx90614_stream ===");

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setTimeOut(200); // don't block forever if an I2C line is stuck low

  if (!mlx.begin()) {
    Serial.println("MLX90614 NOT FOUND - check SDA->D21, SCL->D22, VIN->3V3.");
    while (true) delay(5000);
  }
  Serial.println("MLX90614 OK - streaming T,ambient,object");
}

void loop() {
  float ambient = mlx.readAmbientTempC();
  float object = mlx.readObjectTempC();

  if (isnan(ambient) || isnan(object)) {
    Serial.println("ERR,read failed (NaN)");
  } else {
    Serial.printf("T,%.2f,%.2f\n", ambient, object);
  }
  delay(PERIOD_MS);
}
