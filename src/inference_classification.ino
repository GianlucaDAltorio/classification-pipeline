/*
 * File:     inference_classification.ino
 * Purpose:  Live on-device classification of the lighting condition
 *           (LED, fluorescent, shade, sun) using a table-lookup model.
 *
 * How it works:
 *   1. Read UV (GUVA-S12SD, ADC) and IR (TSL2591, I2C).
 *   2. Smooth both channels with an EMA filter (alpha = 0.2).
 *   3. Scale with the training bounds (IR goes through log1p first).
 *   4. Use the scaled point to pick one cell of the 16x16 grid.
 *   5. Read one stored byte per class from that cell. The highest wins,
 *      unless it is below REJECT_TAU, in which case the result is UNKNOWN.
 */

#include <Wire.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_TSL2591.h>
#include "model.h"                 // CLASS_TABLE, SCALE_BOUNDS, CLASS_NAMES, REJECT_TAU, GRID_N, NUM_CLASSES

Adafruit_TSL2591 tsl = Adafruit_TSL2591(2591);

// Pin assignments
const int UV_PIN = 34;             // analog UV sensor output
const int SDA_PIN = 21;            // I2C data for the TSL2591
const int SCL_PIN = 22;            // I2C clock for the TSL2591


const float ALPHA = 0.2f;                          // EMA smoothing factor
const unsigned long SAMPLE_INTERVAL_MS = 100;      // minimum time between samples (the TSL2591 read adds
                                                   // to this, giving about 122 ms per sample in practice)

// Filter and bookkeeping state
float uvFiltered = -1;             // -1 means "not initialised yet"
float irFiltered = -1;
unsigned long lastSampleTime = 0;
unsigned long sampleIndex = 0;

// Returns class index, or -1 for UNKNOWN. Also reports the cell and winning value.
int classify(float uv, float ir, int &row, int &col, float &bestVal) {
  // Same scaling as training: UV raw, IR through log1p, both min-max with the shipped bounds
  float x = (uv - SCALE_BOUNDS[0]) / (SCALE_BOUNDS[1] - SCALE_BOUNDS[0]);
  float y = (log1pf(ir) - SCALE_BOUNDS[2]) / (SCALE_BOUNDS[3] - SCALE_BOUNDS[2]);

  // Pick the grid cell; constrain() clips out-of-range readings to the edge cells
  col = constrain((int)(x * GRID_N), 0, GRID_N - 1);   // col = scaled UV
  row = constrain((int)(y * GRID_N), 0, GRID_N - 1);   // row = scaled log(IR)

  // Read one byte per class from that cell and keep the highest
  int best = -1;
  bestVal = -1.0f;
  for (int c = 0; c < NUM_CLASSES; c++) {
    float v = pgm_read_byte(&CLASS_TABLE[c][row][col]) / 255.0f;   // table lives in flash (PROGMEM)
    if (v > bestVal) { bestVal = v; best = c; }
  }
  // Below the reject threshold: no learned class is close enough, report UNKNOWN
  return (bestVal < REJECT_TAU) ? -1 : best;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);                            // 12-bit ADC, 0-4095
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!tsl.begin()) {
    Serial.println("# ERROR: TSL2591 not found.");
    while (1) delay(1000);                             // halt: nothing to classify without the sensor
  }
  tsl.setGain(TSL2591_GAIN_MED);                       // same as logging
  tsl.setTiming(TSL2591_INTEGRATIONTIME_100MS);
  Serial.println("idx,uv,ir,row,col,class,latency_us");   // CSV header
}

void loop() {
  // Rate limit: do nothing until the next sample is due
  unsigned long now = millis();
  if (now - lastSampleTime < SAMPLE_INTERVAL_MS) return;
  lastSampleTime = now;

  // Read both sensors
  int rawUV = analogRead(UV_PIN);
  uint32_t lum = tsl.getFullLuminosity();
  uint16_t rawIR = lum >> 16;                          // upper 16 bits = IR channel

  // EMA filter: seed with the first reading, then smooth
  if (uvFiltered < 0) { uvFiltered = rawUV; irFiltered = rawIR; }
  else {
    uvFiltered = ALPHA * rawUV + (1 - ALPHA) * uvFiltered;
    irFiltered = ALPHA * rawIR + (1 - ALPHA) * irFiltered;
  }

  // Classify and time just the lookup
  int row, col;
  float val;
  unsigned long t0 = micros();
  int cls = classify(uvFiltered, irFiltered, row, col, val);   // lookup only, no sensor read
  unsigned long dt = micros() - t0;

  // Print one CSV line: idx,uv,ir,row,col,class,latency_us
  Serial.print(sampleIndex++); Serial.print(",");
  Serial.print(uvFiltered, 2); Serial.print(",");
  Serial.print(irFiltered, 2); Serial.print(",");
  Serial.print(row); Serial.print(",");
  Serial.print(col); Serial.print(",");
  Serial.print(cls < 0 ? "UNKNOWN" : CLASS_NAMES[cls]); Serial.print(",");
  Serial.println(dt);
}
