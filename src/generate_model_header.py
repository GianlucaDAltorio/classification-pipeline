"""Part 3 -- emit model.h from surfaces.npz and the scaling bounds."""
import numpy as np
from pipeline_common import load_split_scaled, quantize, GRID_N, REJECT_TAU

surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)
K = len(classes)

data = load_split_scaled()
lo, hi = data["bounds"]               # per-feature min / max, IR already log1p'd
y = data["y_train"]
labels = np.loadtxt("labels.csv", dtype=int)   # must come from the eps=0.02 run

names = []
for c in classes:                     # same order as the table
    vals, counts = np.unique(y[labels == c], return_counts=True)
    names.append(str(vals[counts.argmax()]))

with open("model.h", "w") as f:
    f.write("#pragma once\n#include <Arduino.h>\n\n")
    f.write(f"#define GRID_N {GRID_N}\n#define NUM_CLASSES {K}\n\n")
    f.write("// index as CLASS_TABLE[class][row][col]; row = scaled log(IR), col = scaled UV\n")
    f.write("const uint8_t CLASS_TABLE[NUM_CLASSES][GRID_N][GRID_N] PROGMEM = {\n")
    for c in classes:
        q = quantize(surfaces[f"c{c}"])
        f.write("  {\n")
        for row in q:
            f.write("    {" + ", ".join(str(int(v)) for v in row) + "},\n")
        f.write("  },\n")
    f.write("};\n\n")
    f.write("// xmin, xmax, ymin, ymax  (x = UV, y = log1p(IR))\n")
    f.write(f"const float SCALE_BOUNDS[4] = {{{lo[0]:.6f}f, {hi[0]:.6f}f, "
            f"{lo[1]:.6f}f, {hi[1]:.6f}f}};\n\n")
    f.write("const char* CLASS_NAMES[NUM_CLASSES] = {"
            + ", ".join(f'"{n}"' for n in names) + "};\n\n")
    f.write(f"#define REJECT_TAU {REJECT_TAU}f\n")

print(f"classes in table order: {names}")
print(f"table = {K} * {GRID_N}^2 = {K * GRID_N**2} bytes, plus 16 bytes of bounds")