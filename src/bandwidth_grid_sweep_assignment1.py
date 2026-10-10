import numpy as np
import pipeline_common as pc
from sklearn.cluster import DBSCAN

data = pc.load_split_scaled()
Xtr, Xval = data["train"], data["val"]
ytr, yval = data["y_train"], data["y_val"]

labels = DBSCAN(eps=0.02, min_samples=8).fit_predict(Xtr)
classes = sorted(set(labels) - {-1})
vals_list = lambda c: list(ytr[labels == c])
names = {c: max(set(vals_list(c)), key=vals_list(c).count) for c in classes}

for G in [16, 32, 64]:
    pc.GRID_N = G                      # make_grid/cell_of read this global
    xx, yy, gp = pc.make_grid()
    for h in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        tables = {c: pc.quantize(pc.normalise_peak(
                      pc.evaluate_on_grid(pc.fit_density(Xtr[labels == c], h), gp, xx)))
                  for c in classes}
        correct = rejected = 0
        for p, true in zip(Xval, yval):
            r, col = pc.cell_of(p)
            v = np.array([pc.dequantize(tables[k][r, col]) for k in classes])
            best = int(np.argmax(v))
            if v[best] < pc.REJECT_TAU:
                rejected += 1
            elif names[classes[best]] == true:
                correct += 1
        n = len(Xval)
        print(f"G={G:3d} h={h:<6} correct={correct/n:6.1%} "
              f"rejected={rejected/n:6.1%} flash={len(classes)*G*G} B")