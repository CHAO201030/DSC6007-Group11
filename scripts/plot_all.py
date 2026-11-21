import csv, glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results" / "eval"
TRES = RES / "curves"
OUT = RES / "figures"; OUT.mkdir(parents=True, exist_ok=True)

C = {"sp": "#1f3864", "p": "#c55a11", "s": "#2e7d32", "f": "#6a3d9a"}
N = {"sp": "SP", "p": "Arm P", "s": "Arm S", "f": "FCP-scratch"}
N2 = {"ppo_single_env": "SP", "fcp2": "Arm P", "sp_ref": "Arm S", "fcp": "FCP-scratch"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": .25, "axes.axisbelow": True})

def read(p): return list(csv.DictReader(open(p, encoding="utf-8-sig")))
def step(s): return int(s.split("_")[-1])

# ---------- F1 ----------
pts = []
for f in sorted(glob.glob(str(RES / "decay_s*.csv"))):
    for r in read(f):
        pts.append(((step(r["ego"].split("/")[-1]) - step(r["partner"].split("/")[-1]))/1e6,
                    float(r["soups_median"])))
BUCK = [(0,0),(.5,1),(1.5,2),(2.5,3),(3.5,4),(4.5,5),(5.5,6),(6.5,7)]
fig, ax = plt.subplots(1, 2, figsize=(11, 4), dpi=180)
cx, cy = [], []
for lo, hi in BUCK:
    v = [y for x, y in pts if lo-1e-9 <= x <= hi+1e-9]
    if v: cx.append((lo+hi)/2); cy.append(100*sum(1 for y in v if y >= 8)/len(v))
ax[0].plot(cx, cy, "-o", color=C["sp"], ms=6, lw=1.8)
for a, b in zip(cx, cy):
    ax[0].annotate(f"{b:.0f}%", (a, b), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=C["sp"])
ax[0].set(xlabel="checkpoint distance (M steps)", ylabel="compatible partners (%)", ylim=(-5, 110))
ax[0].set_title("(a) Compatibility decays with policy distance")
ax[1].scatter([p[0] for p in pts], [p[1] for p in pts], s=24, color=C["p"], alpha=.8, edgecolor="white", lw=.5)
ax[1].set(xlabel="checkpoint distance (M steps)", ylabel="soups delivered", ylim=(-.6, 12))
ax[1].set_title("(b) Individual pairs are bimodal")
fig.tight_layout(); fig.savefig(OUT / "F1_decay.png"); plt.close(fig)

# ---------- F2 ----------
rows = read(RES / "transfer_stoch.csv")
egos = sorted({r["ego"] for r in rows}); pars = sorted({r["partner"] for r in rows})
M = np.full((len(egos), len(pars)), np.nan)
for r in rows: M[egos.index(r["ego"]), pars.index(r["partner"])] = float(r["soups_median"])
def tag(p):
    d = p.split("/")
    return ("vec/" if "vec" in d[0] else "single/") + d[1]
fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=180)
im = ax.imshow(M, cmap="magma", vmin=0, vmax=11, aspect="auto")
ax.set_xticks(range(len(pars))); ax.set_xticklabels([tag(p) for p in pars], rotation=90, fontsize=7)
ax.set_yticks(range(len(egos))); ax.set_yticklabels([f"{N2.get(e.split('/')[0], e.split('/')[0])}/{e.split('/')[1]}" for e in egos], fontsize=7)
ax.axvline(4.5, color="white", lw=1.2, ls="--")
ax.set_title("ego x partner: soups delivered (stochastic evaluator)")
ax.grid(False); fig.colorbar(im, ax=ax, label="soups", fraction=.04)
fig.tight_layout(); fig.savefig(OUT / "F2_matrix.png"); plt.close(fig)

# ---------- F3 ----------
ft = read(RES / "final_table.csv")
fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=180)
rng = np.random.default_rng(0)
for a in ("sp","p","s","f"):
    ps = [(float(r["seen_soups_med"]), float(r["heldout_soups_med"])) for r in ft if r["checkpoint"].split("_")[0] == a]
    ax.scatter([p[0] for p in ps] + rng.normal(0,.1,len(ps)), [p[1] for p in ps],
               s=80, color=C[a], alpha=.85, edgecolor="white", lw=.8, label=f"{N[a]} (n={len(ps)})")
ax.annotate("SP: 11 → 0", (11, 0), xytext=(8.3, 1.1), fontsize=8, color=C["sp"],
            arrowprops=dict(arrowstyle="->", lw=.8, color=C["sp"]))
ax.annotate("FCP: 8 → 7", (8, 7), xytext=(4.3, 6.3), fontsize=8, color=C["f"],
            arrowprops=dict(arrowstyle="->", lw=.8, color=C["f"]))
ax.set(xlabel="self-play soups (capability)", ylabel="held-out soups (generality)",
       xlim=(-.7, 12), ylim=(-.8, 8))
ax.set_title("Capability vs partner generality (5 seeds per arm)")
ax.legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(OUT / "F3_tradeoff.png"); plt.close(fig)

# ---------- F4 斜率图 ----------
fig, ax = plt.subplots(figsize=(6.6, 4.8), dpi=180)
for i, a in enumerate(("sp","p","s","f")):
    for r in [x for x in ft if x["checkpoint"].split("_")[0] == a]:
        j = rng.normal(0, .035)
        ax.plot([i-.18+j, i+.18+j], [float(r["seen_soups_med"]), float(r["heldout_soups_med"])],
                color=C[a], alpha=.45, lw=1.0, marker="o", ms=3.5)
ax.set_xticks(range(4)); ax.set_xticklabels([N[a] for a in ("sp","p","s","f")])
ax.set(ylabel="soups delivered", ylim=(-.8, 12))
ax.set_title("Every seed: self-play → held-out (lines = seeds)")
fig.tight_layout(); fig.savefig(OUT / "F4_bars.png"); plt.close(fig)

# ---------- F5 ----------
ARMS = [("sp", sorted([f for f in glob.glob(str(ROOT/"results"/"train_seed*.npz")) if "_vec" not in f])),
        ("f", sorted(glob.glob(str(TRES/"f_seed*.npz")))),
        ("p", sorted(glob.glob(str(TRES/"p_seed*.npz")))),
        ("s", sorted(glob.glob(str(TRES/"s_seed*.npz"))))]
fig, axs = plt.subplots(2, 2, figsize=(10, 6.4), dpi=170, sharey=True)
for ax, (a, files) in zip(axs.ravel(), ARMS):
    for f in files:
        d = np.load(f); x = d["timestep"]/1e6; y = d["soups_delivered"]
        ax.plot(x, y, lw=.5, alpha=.10, color=C[a])
        k = 24
        if len(y) > k:
            ys = np.convolve(y, np.ones(k)/k, "valid"); ax.plot(x[len(y)-len(ys):], ys, lw=1.7, color=C[a])
    ax.set_title(N[a]); ax.set_xlabel("step (M)"); ax.set_ylim(-.5, 12.5)
axs[0,0].set_ylabel("soups (training)"); axs[1,0].set_ylabel("soups (training)")
fig.suptitle("Training-time soups (thin = raw, bold = 24-window mean)", y=1.0)
fig.tight_layout(); fig.savefig(OUT / "F5_training_curves.png"); plt.close(fig)

# ---------- F6 塌缩率（正确数据源：final_table 的 seen） ----------
cnt = {}
for a in ("sp","p","s","f"):
    v = [float(r["seen_soups_med"]) for r in ft if r["checkpoint"].split("_")[0] == a]
    cnt[a] = (sum(1 for x in v if x < 1), len(v))
fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=180)
ks = list(cnt); vals = [100*cnt[a][0]/cnt[a][1] for a in ks]
ax.bar([N[a] for a in ks], vals, color=[C[a] for a in ks], width=.55)
for i, a in enumerate(ks):
    ax.annotate(f"{cnt[a][0]}/{cnt[a][1]}", (i, vals[i]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=9)
ax.set(ylabel="seeds collapsed in self-play (%)", ylim=(0, 115))
ax.set_title("Self-play collapse by arm (final checkpoint)")
fig.tight_layout(); fig.savefig(OUT / "F6_collapse.png"); plt.close(fig)
print("saved F1-F6")
