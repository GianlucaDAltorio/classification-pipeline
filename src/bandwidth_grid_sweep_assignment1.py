"""Part 2 -- choose (h, G) by end-to-end validation on the stored byte tables.

The log-likelihood sweep (bandwidth_sweep.py),this script scores the shipped
model itself: for every (grid size, bandwidth) pair, build the uint8 tables from
the training split, classify every validation point by table lookup, and report
the share classified correctly and the share wrongly rejected as UNKNOWN. The test
set is never read.
"""
import numpy as np
import pipeline_common as pc
from sklearn.cluster import DBSCAN

data = pc.load_split_scaled()
Xtr, Xval = data["train"], data["val"]
ytr, yval = data["y_train"], data["y_val"]

# Discover classes on the training split only, then name each cluster by the
# majority true label inside it (ground truth is used only for scoring).
labels = DBSCAN(eps=0.02, min_samples=8).fit_predict(Xtr)
classes = sorted(set(labels) - {-1})
names = pc.cluster_names(labels, ytr)

for G in [16, 32, 64]:
    pc.GRID_N = G                      # make_grid/cell_of read this global
    xx, yy, gp = pc.make_grid()
    for h in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        # The stored model: one peak-normalised uint8 table per class.
        tables = {c: pc.quantize(pc.normalise_peak(
                      pc.evaluate_on_grid(pc.fit_density(Xtr[labels == c], h), gp, xx)))
                  for c in classes}
        correct = rejected = 0
        for p, true in zip(Xval, yval):
            # Same lookup the device will do: one byte per class from the cell.
            r, col = pc.cell_of(p)
            v = np.array([pc.dequantize(tables[k][r, col]) for k in classes])
            best = int(np.argmax(v))
            if v[best] < pc.REJECT_TAU:
                rejected += 1          # winner too weak: answer UNKNOWN
            elif names[classes[best]] == true:
                correct += 1
        n = len(Xval)
        # Flash cost is K * G^2 bytes (one byte per cell, per class).
        print(f"G={G:3d} h={h:<6} correct={correct/n:6.1%} "
              f"rejected={rejected/n:6.1%} flash={len(classes)*G*G} B")