""""
Raw vs. Filtered IR Signal Comparison (TSL2591, LED vs. Fluorescent).
Plots raw and EMA-filtered IR counts side by side against the
model.h row boundaries, and prints per-light-source statistics.
"""

import pandas as pd
import matplotlib.pyplot as plt

files = ["raw_vs_filtered_led.csv", "raw_vs_filtered_fluorescent.csv"]   # your file names
BOUNDS = [185.4, 261.6]     # model.h row edges in IR counts

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax, f in zip(axes, files):
    d = pd.read_csv(f, comment="#").iloc[20:]       # skip the EMA startup
    lab = d["label"].iloc[0]
    ax.plot(d["idx"], d["raw_ir"], color="0.65", lw=0.8, label="raw")
    ax.plot(d["idx"], d["filt_ir"], color="black", lw=1.4, label=r"filtered (EMA, $\alpha=0.2$)")
    for b in BOUNDS:
        ax.axhline(b, color="black", ls="--", lw=0.6)
    ax.set_title(lab)
    ax.set_xlabel("sample index")

    med = d["filt_ir"].median()
    n_raw = int((d["raw_ir"] < BOUNDS[1]).sum()) if lab == "fluorescent" else None
    print(f"{lab}: median {med:.1f}, "
          f"largest raw excursion {(d['raw_ir'] - med).abs().max():.1f}, "
          f"largest filtered excursion {(d['filt_ir'] - med).abs().max():.1f}, "
          f"filtered range {d['filt_ir'].min():.1f}-{d['filt_ir'].max():.1f}"
          + (f", raw samples below LED boundary: {n_raw}" if n_raw is not None else ""))
axes[0].set_ylabel("IR (TSL2591 counts)")
handles, labels_ = axes[0].get_legend_handles_labels()
fig.legend(handles, labels_, fontsize=8, loc="upper center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.92))   # leave room at the top for the legend
fig.savefig("../figures/generated/raw_vs_filtered.pdf", bbox_inches="tight")