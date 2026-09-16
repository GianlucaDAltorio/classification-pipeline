""" Part 4, creating class label for a raw point
    or none for any that are unknown """

import numpy as np
from pipeline_common import (load_split_scaled, scale_features,
                             quantize, dequantize,
                             GRID_N, REJECT_TAU)

# Training bounds ship with the model.
bounds = load_split_scaled()["bounds"]
surfaces = np.load("surfaces.npz")
classes = sorted(int(k[1:]) for k in surfaces.files)

# The shipped model.
# Code only reads these and "bounds".
tables = {c: quantize(surfaces[f"c{c}"]) for c in classes}

# Give raw point a class label,
# otherwise give "None" for an unknown.
def classify(raw_point):
    point, _= scale_features(np.asarray(raw_point), bounds)

    row = min(max(int(point[1] * GRID_N), 0), GRID_N - 1)
    col = min(max(int(point[0] * GRID_N), 0), GRID_N - 1)

    values = {c: dequantize(tables[c][row, col]) for c in classes}
    best_class = max(values, key=values.get)
    if values[best_class] < REJECT_TAU:
        return None
    return best_class

if __name__ == "__main__":
    # One raw point on each shape, then the ring's hollow
    # centre (unknown).
    # [5.99958893, 3.17868816] corresponds to 
    for q in ([0.0, -1.8], [0.0, 5.0], [6.8, 1.6],
              [5.99958893, 3.17868816], [4.8, 1.6]):
        print(q, "->", classify(q))