import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
OUT = ROOT / "work_c" / "results" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
ARMS = [
    ("SP", sorted(glob.glob(str(RES / "train_seed[0-9]*.npz")))),
    ("FCP-scratch", sorted(glob.glob(str(RES / "fcp_seed*.npz")))),
    ("Arm P", sorted(glob.glob(str(RES / "fcp2" / "fcp_seed*.npz")))),
    ("Arm S", sorted(glob.glob(str(RES / "sp_ref" / "fcp_seed*.npz")))),
]
def smooth(y, k=24):
    return np.convolve(y, np.ones(k)/k, mode="valid") if len(y) > k else y

fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=160, sharey=True)
collapse = []
for ax, (name, files) in zip(axes.ravel(), ARMS):
    c = 0
    for f in files:
        d = np.load(f); x = d["timestep"]/1e6; y = d["soups_delivered"]
        ax.plot(x, y, lw=.5, alpha=.12, color="tab:blue")
        ys = smooth(y); ax.plot(x[len(y)-len(ys):], ys, lw=1.6)
        tail = y[d["timestep"] >= d["timestep"].max()*0.5]
        if len(tail) and np.median(tail) < 1: c += 1
    collapse.append((name, c, len(files)))
    ax.set_title(name); ax.set_xlabel("step (M)"); ax.grid(alpha=.3); ax.set_ylim(-.5, 12.5)
axes[0,0].set_ylabel("soups (training)"); axes[1,0].set_ylabel("soups (training)")
fig.suptitle("Training-time soups, four arms")
fig.tight_layout(); fig.savefig(OUT / "F5_training_curves.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.4, 4), dpi=160)
ax.bar([c[0] for c in collapse], [100*c[1]/max(c[2],1) for c in collapse], color="#E07541")
for i, (n, c, t) in enumerate(collapse):
    ax.annotate(f"{c}/{t}", (i, 100*c/max(t,1)), textcoords="offset points", xytext=(0, 6), ha="center")
ax.set_ylabel("seeds collapsed in 2nd half (%)"); ax.set_title("Single-learner arms collapse")
ax.set_ylim(0, 108); ax.grid(axis="y", alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "F6_collapse.png"); plt.close(fig)
print("saved F5/F6", collapse)
