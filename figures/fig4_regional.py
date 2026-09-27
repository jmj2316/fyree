"""Fig.4: GFED 14개 지역 x 리드 — 짝지은 ΔAUPRC 히트맵 (Ours−TeleViT, TeleViT−Clim, Ours−Clim).
지역별로 47개 타깃 시점 블록 bootstrap(500회), CI가 0을 포함하면 해당 칸을 약하게 표시."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch

from common import LEADS, OUT, ROOT, panel_label, save, style
from mapdata import TILE, load, preds_on_tiles

sys.path.insert(0, str(ROOT))
from pyree.bootstrap import ap_from_counts, block_hist  # noqa: E402

REG = ["BONA", "TENA", "CEAM", "NHSA", "SHSA", "EURO", "MIDE", "NHAF", "SHAF", "BOAS", "CEAS",
       "SEAS", "EQAS", "AUST"]
CACHE = OUT / "work" / "regional_bootstrap.json"
N_BOOT, N_BINS = 500, 5000


def compute():
    md = load()
    rng = np.random.default_rng(0)
    res = {}
    for h in LEADS:
        tiles, po, c, y = preds_on_tiles("v4_noanchor", h, "test", md)
        _, pt, _, _ = preds_on_tiles(f"televit_orig_h{h}", h, "test", md)
        reg = np.stack([md["region"][i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
                        for _, i, j in tiles])
        Tpx = np.repeat(tiles[:, 0], TILE * TILE)
        reg, y, po, pt, c = (a.reshape(-1) for a in (reg, y, po, pt, c))
        for r in range(1, 15):
            m = reg == r
            if y[m].sum() < 50:
                continue
            blocks = np.unique(Tpx[m])
            preds = {"ours": po[m], "televit": pt[m], "clim": c[m]}
            edges = np.unique(np.quantile(np.concatenate(list(preds.values())),
                                          np.linspace(0, 1, N_BINS + 1)[1:-1]))
            hist = {k: block_hist(v, y[m].astype(np.float64), Tpx[m], blocks, edges)
                    for k, v in preds.items()}
            W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)
            W = np.vstack([np.ones(len(blocks)), W])
            ap = {k: ap_from_counts(W @ a, W @ b) for k, (a, b) in hist.items()}
            d = {"n_pos": int(y[m].sum()), "n_blocks": int(len(blocks))}
            for k in preds:
                d[k] = float(ap[k][0])
            for u, v in [("ours", "televit"), ("televit", "clim"), ("ours", "clim")]:
                diff = ap[u] - ap[v]
                d[f"{u}-{v}"] = float(diff[0])
                d[f"{u}-{v}_ci"] = [float(q) for q in np.percentile(diff[1:], [2.5, 97.5])]
            res[f"{REG[r-1]}|{h}"] = d
            print(h, REG[r - 1], {k: round(v, 3) for k, v in d.items() if isinstance(v, float)},
                  flush=True)
    CACHE.write_text(json.dumps(res, indent=1))
    return res


def main():
    res = json.loads(CACHE.read_text()) if CACHE.exists() else compute()
    md = load()
    style()
    fig = plt.figure(figsize=(7.2, 6.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.78, 1.0], hspace=0.28, wspace=0.08)

    # (a) 지역 지도
    ax0 = fig.add_subplot(gs[0, :])
    reg = md["region"].astype(float)
    reg[reg == 0] = np.nan
    cmap = ListedColormap(plt.get_cmap("tab20").colors[:14])
    ax0.imshow(reg, cmap=cmap, vmin=0.5, vmax=14.5, extent=[-180, 180, -90, 90],
               interpolation="nearest")
    ax0.contour(md["lon"], md["lat"], md["lsm"], levels=[0.5], colors="k", linewidths=0.25)
    for r in range(1, 15):
        ys, xs = np.where(md["region"] == r)
        ax0.text(md["lon"][int(np.median(xs))], md["lat"][int(np.median(ys))], REG[r - 1],
                 ha="center", va="center", fontsize=6.5, fontweight="bold",
                 bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax0.set_xlim(-180, 180)
    ax0.set_ylim(-58, 80)
    ax0.set_xticks([-120, -60, 0, 60, 120], ["120°W", "60°W", "0°", "60°E", "120°E"])
    ax0.set_yticks([-40, 0, 40, 80], ["40°S", "0°", "40°N", "80°N"])
    ax0.tick_params(labelsize=7)
    panel_label(ax0, "a", x=-0.06)

    # (b-d) 히트맵
    comps = [("ours-televit", r"Pyree − TeleViT$_{i,g}$"),
             ("televit-clim", r"TeleViT$_{i,g}$ − climatology"),
             ("ours-clim", "Pyree − climatology")]
    rows = [r for r in REG if all(f"{r}|{h}" in res for h in LEADS)]
    vmax = max(abs(res[f"{r}|{h}"][k]) for r in rows for h in LEADS for k, _ in comps)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax * 100, vmax=vmax * 100)
    for col, (key, lab) in enumerate(comps):
        ax = fig.add_subplot(gs[1, col])
        M = np.array([[res[f"{r}|{h}"][key] * 100 for h in LEADS] for r in rows])
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
        for a_, r in enumerate(rows):
            for b_, h in enumerate(LEADS):
                d = res[f"{r}|{h}"]
                lo, hi = d[f"{key}_ci"]
                sig = lo > 0 or hi < 0
                ax.text(b_, a_, f"{M[a_, b_]:+.1f}", ha="center", va="center", fontsize=6,
                        color=("white" if abs(M[a_, b_]) > 0.6 * vmax * 100 else "k") if sig else "0.55",
                        fontstyle="normal" if sig else "italic")
                if not sig:
                    ax.add_patch(plt.Rectangle((b_ - 0.5, a_ - 0.5), 1, 1, fill=False, hatch="////",
                                               ec="0.75", lw=0))
        ax.set_xticks(range(len(LEADS)), [str(h) for h in LEADS])
        ax.set_xlabel("Lead time (× 8 days)")
        if col == 0:
            n1 = {r: res[f"{r}|1"]["n_pos"] for r in rows}
            ax.set_yticks(range(len(rows)), [f"{r} ({n1[r]/1e3:.0f}k)" for r in rows])
        else:
            ax.set_yticks(range(len(rows)), [])
        ax.text(0.5, 1.02, lab, transform=ax.transAxes, ha="center", va="bottom", fontsize=8)
        panel_label(ax, "bcd"[col], x=-0.42 if col == 0 else -0.08, y=1.07)
    cax = fig.add_axes([0.92, 0.11, 0.013, 0.36])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"Paired $\Delta$AUPRC (percentage points)")
    fig.legend(handles=[Patch(facecolor="white", edgecolor="0.6", hatch="////",
                              label="95% CI includes 0")],
               loc="upper center", bbox_to_anchor=(0.5, 0.035), frameon=False, fontsize=7.5)
    save(fig, "fig4_regional_breakdown")


if __name__ == "__main__":
    main()
