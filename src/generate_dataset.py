"""Part 0: Write the dataset to disk so every
   later stage reads the same file, and so the file itself
   can be committed as evidence.
    """

import numpy as np
from pipeline_common import load_dataset

# Synthetic by default.
X = load_dataset()

np.savetxt("dataset.csv", X, delimiter=",", fmt="%.6f")
print(f"Wrote dataset.csv: {X.shape[0]} points, {X.shape[1]} features")