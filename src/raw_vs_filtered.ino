/*
 * File:     raw_vs_filtered.ino
 *
 * Purpose:  Log the raw and EMA-filtered sensor signals side by side, to
 *           show that the filter removes transient glitches without
 *           erasing the difference between classes.
 *
 * How it works:
 *   1. Read UV (GUVA-S12SD, analog ADC) and IR (TSL2591 IR channel, I2C).
 *   2. Apply the same EMA filter as the data-collection and inference
 *      sketches (alpha = 0.2, same gain and integration time).
 *   3. While a label is active, print the raw and filtered values together.
 *
 * Serial commands (115200 baud):
 *   L:<label>   start a run for that class, e.g. L:led
 *               (also resets the filter and the sample index)
 *   L:          (empty label) stop logging
 *
 * Output rows:  idx,t_ms,label,raw_uv,raw_ir,filt_uv,filt_ir
 *   Lines starting with '#' are status messages, not data.
 *
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// Pin assignments
const int UV_PIN = 34;             // analog UV sensor output
const int SDA_PIN = 21;            // I2C data for the TSL2591
const int SCL_PIN = 22;            // I2C clock for the TSL2591


const float ALPHA = 0.2f;                          // EMA smoothing factor
const unsigned long SAMPLE_INTERVAL_MS = 100;      // minimum time between samples

// Filter state
float uvFiltered = -1;             // -1 = "not yet initialized"
float irFiltered = -1;

String currentLabel = "";          // empty = not logging
unsigned long sampleIndex = 0;     // row number within the current run
unsigned long lastSampleTime = 0;

// Reads "L:<label>" commands from the serial monitor to start or stop a run
void handleSerialCommands() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    line.trim();
    if (line.startsWith("L:")) {
      currentLabel = line.substring(2);
      currentLabel.trim();
      // Restart the filter and index for each run, so every run is self-contained
      uvFiltered = -1;
      irFiltered = -1;
      sampleIndex = 0;
      Serial.println(currentLabel.length() ? "# Logging started" : "# Logging paused");
    }
  }
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);                          // 12-bit ADC, 0-4095
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!tsl.begin()) {
    Serial.println("# ERROR: TSL2591 not found.");
    while (1) delay(1000);                           // halt: nothing to log without the sensor
  }
  tsl.setGain(TSL2591_GAIN_MED);                     // same as logging and inference
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
  Serial.println("# Send L:<label> to start a run, L: to stop.");
  Serial.println("idx,t_ms,label,raw_uv,raw_ir,filt_uv,filt_ir");   // CSV header
}

void loop() {
  handleSerialCommands();   // check for a new label command every pass

  // Rate limit: do nothing until the next sample is due
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // Read both sensors
  int rawUV = analogRead(UV_PIN);
  uint32_t lum = tsl.getFullLuminosity();
  uint16_t rawIR = lum >> 16;                        // upper 16 bits = IR channel

  // Same EMA as the training data: y_n = alpha*x_n + (1-alpha)*y_(n-1)
  if (uvFiltered < 0) { uvFiltered = rawUV; irFiltered = rawIR; }   // seed with the first reading
  else {
    uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
    irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
  }

  // Print one row only while a label is active
  if (currentLabel.length() > 0) {
    Serial.print(sampleIndex++); Serial.print(",");
    Serial.print(now);           Serial.print(",");   // timestamp in ms since boot
    Serial.print(currentLabel);  Serial.print(",");
    Serial.print(rawUV);         Serial.print(",");
    Serial.print(rawIR);         Serial.print(",");
    Serial.print(uvFiltered, 2); Serial.print(",");
    Serial.println(irFiltered, 2);
  }
}
