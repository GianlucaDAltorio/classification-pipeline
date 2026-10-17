# classification-pipeline
Repository for Lab 3 and Assignment 1 in the Project I course.

A density-based classification pipeline for memory-constrained devices: DBSCAN finds the classes, kernel density estimates become small byte tables, and the device classifies with table lookups only.

- **Lab 3:** the pipeline built in Python on synthetic data.
- **Assignment 1 (release v2.0):** the same pipeline on real sensor data, running on an ESP32.

## What Assignment 1 adds

An ESP32 with a GUVA-S12SD UV sensor and a TSL2591 IR sensor classifies four lighting conditions (LED, fluorescent, shade, sun). The report covers:

- **Data and clustering:** 600 labelled samples per class, EMA-filtered (α = 0.2). DBSCAN (ε = 0.02, min_samples = 8) recovers all four classes without labels (confusion table in the paper).
- **Model:** one 16×16 byte table per class, 1,024 B plus 16 B of scaling bounds, exported as `model.h`. IR is log-scaled before the lookup.
- **On-device results:** the ESP32 agrees with the desktop model on all 2,324 replayed samples. Measured cost is 1,060 B of flash, no extra SRAM and 4–8 µs per lookup.
- **Filtering evidence:** raw-versus-filtered plot for LED and fluorescent.
- **Limitations:** overcast sun is rejected as UNKNOWN, classes are brightness bands, and time-series data makes validation optimistic.

## Layout

- `src/`: Python pipeline scripts, `generate_model_header.py` and the generated `model.h`, labelled data (`data_sensors.csv`), `dbscan_confusion.csv`, fresh test batches (`run_*.csv`), and the agreement script and output.
- `src/*.ino`: `data_collection.ino` (labelled logging), `raw_vs_filtered.ino` (filter evidence), `inference_classification.ino` (on-device classifier).
- `figures/generated/`: Figures used in the report, including the proof-of-concept photo.
- `paper/`: LaTeX source and PDF of the report.
- `dataset.csv` and `generate_dataset.py` are the synthetic data from Lab 3.
