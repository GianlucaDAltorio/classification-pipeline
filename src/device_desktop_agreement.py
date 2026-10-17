"""
device_desktop_agreement.py

Replays logged sensor runs through a Python copy of the on-device classifier
and checks that the two agree.

For each run CSV (uv, ir readings plus the class the device reported), this
script:
  1. Rebuilds the classifier from the saved artifacts (labels.csv, surfaces.npz)
     using the quantized per-class lookup tables.
  2. Re-predicts every sample in Python.
  3. Reports device-vs-Python agreement, accuracy against the known lighting
     condition, and the share of UNKNOWN predictions, listing any disagreements.

Inputs:  labels.csv, surfaces.npz, run_*.csv, plus the pipeline_common module
Output:  console, mirrored to device_desktop_agreement.txt
"""

import numpy as np
import pandas as pd
import pipeline_common as pc
import sys

class Tee:
    # Print to the screen and to a file at the same time.
    def __init__(self, path):
        self.file = open(path, "w")
        self.stdout = sys.stdout
    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
    def flush(self):
        self.stdout.flush()
        self.file.flush()

sys.stdout = Tee("device_desktop_agreement.txt")
data = pc.load_split_scaled()
lo, hi = data["bounds"]
labels = np.loadtxt("labels.csv", dtype=int)
names = pc.cluster_names(labels, data["y_train"])
surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)
# One quantized lookup table per class, as stored on the device
tables = {c: pc.quantize(surfaces[f"c{c}"]) for c in classes}

def predict(uv, ir):
    p = (np.array([uv, np.log1p(ir)]) - lo) / (hi - lo)   # same scaling as training
    r, c = pc.cell_of(p)
    v = np.array([pc.dequantize(tables[k][r, c]) for k in classes])
    best = int(np.argmax(v))
    # Reject if even the best class scores below the threshold
    return "UNKNOWN" if v[best] < pc.REJECT_TAU else names[classes[best]]

RUNS = [("run_led.csv", "LED"), ("run_fluorescent.csv", "fluorescent"),
        ("run_shade.csv", "shade"), ("run_sun.csv", "sun")]

for f, truth in RUNS:
    # Skip the first 20 samples of each run
    d = pd.read_csv(f, comment="#").iloc[20:].copy()
    d["py"] = [predict(u, i) for u, i in zip(d["uv"], d["ir"])]
    print(f"{truth}: n={len(d)}  device-vs-python agreement {(d['py'] == d['class']).mean():.1%}  "
          f"correct vs truth {(d['class'] == truth).mean():.1%}  "
          f"UNKNOWN {(d['class'] == 'UNKNOWN').mean():.1%}")
    diff = d[d["py"] != d["class"]]
    print("  disagreements:", "none" if diff.empty else "")
    if not diff.empty:
        print(diff[["idx", "uv", "ir", "row", "col", "class", "py"]])
