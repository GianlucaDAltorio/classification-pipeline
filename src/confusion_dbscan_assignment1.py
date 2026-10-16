"""Part 2 -- score DBSCAN against the ground-truth labels."""
import numpy as np
import pandas as pd
from pipeline_common import load_split_scaled

data = load_split_scaled()
y = data["y_train"]                       # true labels (training split)
labels = np.loadtxt("labels.csv", dtype=int)   # DBSCAN output from cluster_dbscan.py

table = pd.crosstab(pd.Series(y, name="true label"),
                    pd.Series(labels, name="DBSCAN cluster"))
print(table)
table.to_csv("dbscan_confusion.csv")

# cluster -> label mapping by majority vote (needed for CLASS_NAMES in model.h)
for c in sorted(set(labels) - {-1}):
    members = y[labels == c]
    names, counts = np.unique(members, return_counts=True)
    print(f"cluster {c} -> {names[counts.argmax()]} "
          f"({counts.max()}/{len(members)} points)")