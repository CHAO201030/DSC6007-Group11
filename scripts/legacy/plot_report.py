import csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES = Path(__file__).resolve().parents[2] / "work_c" / "results"
OUT = RES / "figures"; OUT.mkdir(parents=True, exist_ok=True)
BUCKETS = [(0,0),(0.5,1.0),(1.5,2.0),(2.5,3.0),(3.5,4.0),(4.5,5.0),(5.5,6.0),(6.5,7.0)]

def step(s): return int(s.split("_")[-1])
def read(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))

pts = []
for f in sorted(glob.glob(str(RES / "decay_s*.csv"))):
    for r in read(f):
        pts.append(((step(r["ego"].split("/")[-1]) - step(r["partner"].split("/")[-1]))/1e6,
                    float(r["soups_median"])))
xs, ys = [p[0] for p in pts], [p[1] for p in pts]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
ax = axes[0]; cx, cy = [], []
for lo, hi in BUCKETS:
    v = [y for x, y in pts if lo-1e-9 <= x <= hi+1e-9]
    if not v: continue
    cx.append((lo+hi)/2); cy.append(100*sum(1 for y in v if y >= 8)/len(v))
ax.plot(cx, cy, marker="o", color="#1a365d")
for a, b in zip(cx, cy):
    ax.annotate(str(int(b))+"%", (a, b), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
ax.set_xlabel("checkpoint distance (M steps)"); ax.set_ylabel("compatible partners (%)")
ax.set_title("Compatibility decays with policy distance"); ax.set_ylim(-5, 108); ax.grid(alpha=.3)
ax = axes[1]
ax.scatter(xs, ys, s=22, color="#E07541", alpha=.85, edgecolor="none")
ax.set_xlabel("checkpoint distance (M steps)"); ax.set_ylabel("soups delivered")
ax.set_title("Pairs are bimodal"); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(OUT / "F1_decay.png"); plt.close(fig)

rows = read(RES / "transfer_stoch.csv")
egos = sorted({r["ego"] for r in rows}); pars = sorted({r["partner"] for r in rows})
M = np.full((len(egos), len(pars)), np.nan)
for r in rows:
    M[egos.index(r["ego"]), pars.index(r["partner"])] = float(r["soups_median"])
fig, ax = plt.subplots(figsize=(9.5, 6), dpi=160)
im = ax.imshow(M, cmap="viridis", vmin=0, vmax=11, aspect="auto")
ax.set_xticks(range(len(pars))); ax.set_xticklabels([p.split("/")[1] for p in pars], rotation=90, fontsize=6)
ax.set_yticks(range(len(egos)))
ax.set_yticklabels([e.split("/")[0].replace("ppo_","").replace("_env","")+"/"+e.split("/")[1] for e in egos], fontsize=7)
ax.set_title("ego x partner (soups, stochastic)"); fig.colorbar(im, ax=ax, label="soups")
fig.tight_layout(); fig.savefig(OUT / "F2_matrix.png"); plt.close(fig)

ft = read(RES / "final_table.csv")
COL = {"sp":"#1a365d","p":"#E07541","s":"#2f855a","f":"#805ad5"}
NAME = {"sp":"SP","p":"Arm P","s":"Arm S","f":"FCP-scratch"}
import collections
groups = collections.defaultdict(list)
for r in ft:
    a = r["checkpoint"].split("_")[0]
    groups[a].append((float(r["seen_soups_med"]), float(r["heldout_soups_med"])))

fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=160)
rng = np.random.default_rng(0)
for a, ps in groups.items():
    x = np.array([p[0] for p in ps]) + rng.normal(0, 0.12, len(ps))
    y = np.array([p[1] for p in ps]) + rng.normal(0, 0.08, len(ps))
    ax.scatter(x, y, s=80, color=COL.get(a, "gray"), alpha=.85,
               edgecolor="white", label=NAME.get(a, a) + f" (n={len(ps)})")
ax.annotate("SP: 11 -> 0", (11, 0), xytext=(8.6, 1.2), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.annotate("FCP-scratch: 8 -> 7", (8, 7), xytext=(4.6, 6.4), fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=.8))
ax.set_xlabel("self-play soups (capability)")
ax.set_ylabel("held-out soups (generality)")
ax.set_title("Capability vs partner generality")
ax.set_xlim(-0.6, 12); ax.set_ylim(-0.8, 8)
ax.grid(alpha=.3); ax.legend(fontsize=8, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "F3_tradeoff.png"); plt.close(fig)

arms = ["sp","p","s","f"]; lab = ["SP","Arm P","Arm S","FCP-scratch"]
sv = [np.mean([float(r["seen_soups_med"]) for r in ft if r["checkpoint"].split("_")[0]==a]) for a in arms]
hv = [np.mean([float(r["heldout_soups_med"]) for r in ft if r["checkpoint"].split("_")[0]==a]) for a in arms]
x = np.arange(4); w = .38
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=160)
ax.bar(x-w/2, sv, w, label="self-play", color="#1a365d")
ax.bar(x+w/2, hv, w, label="held-out", color="#E07541")
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylabel("soups delivered")
ax.set_title("Self-play vs held-out (5 seeds, mean)"); ax.grid(axis="y", alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "F4_bars.png"); plt.close(fig)
print("saved F1-F4")
