"""Fig.7: 실제 산불 사건 케이스 스터디 (2019 test + 2020 추가 held-out)."""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle

from case_studies import CACHE, EVENTS
from common import COLORS, LEADS, NAME, OUT, panel_label, save, style
from mapdata import load

HIGHLIGHT = ["Amazon 2019", "Indonesia 2019", "SE Australia 2019–20", "Siberian Arctic 2020"]
TS_LEAD = 4


def main():
    res = json.loads(CACHE.read_text())
    md = load()
    style()
    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.95, 1.0, 1.0], hspace=0.62, wspace=0.30)
    SHORT = {"Amazon 2019": "Amazon '19", "Siberia 2019": "Siberia '19", "Alaska 2019": "Alaska '19",
             "Indonesia 2019": "Indonesia '19", "SE Australia 2019–20": "SE Aus. '19–20",
             "Siberian Arctic 2020": "Sib. Arctic '20", "US West Coast 2020": "US West '20",
             "Pantanal 2020": "Pantanal '20"}

    top = gs[0, :].subgridspec(1, 3, width_ratios=[1.0, 0.45, 0.95], wspace=0.0)
    # (a) 사건 위치
    ax = fig.add_subplot(top[0, 0])
    ax.contour(md["lon"], md["lat"], md["lsm"], levels=[0.5], colors="0.35", linewidths=0.3)
    for k, (name, (la0, la1, lo0, lo1), _) in enumerate(EVENTS):
        col = plt.get_cmap("tab10")(k)
        ax.add_patch(Rectangle((lo0, la0), lo1 - lo0, la1 - la0, fill=False, ec=col, lw=1.3))
        below = name.startswith("Pantanal")  # 아마존 상자와 라벨이 겹치지 않게 아래에
        ax.text(lo0 + (lo1 - lo0) / 2, (la0 - 3) if below else (la1 + 3), str(k + 1), color=col,
                ha="center", va="top" if below else "bottom", fontsize=7, fontweight="bold")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-58, 80)
    ax.set_xticks([-120, -60, 0, 60, 120], ["120°W", "60°W", "0°", "60°E", "120°E"], fontsize=6.5)
    ax.set_yticks([-40, 0, 40, 80], ["40°S", "0°", "40°N", "80°N"], fontsize=6.5)
    ax.set_aspect("equal")
    panel_label(ax, "a", x=-0.12)

    # (b) ΔAUPRC 사건 x 리드
    bx = fig.add_subplot(top[0, 2])
    names = [n for n, _, _ in EVENTS]
    M = np.full((len(names), len(LEADS)), np.nan)
    sig = np.zeros_like(M, dtype=bool)
    for a_, n in enumerate(names):
        for b_, h in enumerate(LEADS):
            d = res.get(f"{n}|{h}")
            if d is None:
                continue
            M[a_, b_] = d["ours-televit"] * 100
            lo, hi = d["ours-televit_ci"]
            sig[a_, b_] = lo > 0 or hi < 0
    vmax = np.nanmax(np.abs(M))
    im = bx.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(0, -vmax, vmax), aspect="auto")
    for a_ in range(M.shape[0]):
        for b_ in range(M.shape[1]):
            if np.isnan(M[a_, b_]):
                continue
            s = sig[a_, b_]
            bx.text(b_, a_, f"{M[a_, b_]:+.1f}", ha="center", va="center", fontsize=6,
                    color=("white" if abs(M[a_, b_]) > 0.6 * vmax else "k") if s else "0.5",
                    fontstyle="normal" if s else "italic")
            if not s:
                bx.add_patch(Rectangle((b_ - 0.5, a_ - 0.5), 1, 1, fill=False, hatch="////",
                                       ec="0.75", lw=0))
    bx.set_xticks(range(len(LEADS)), [str(h) for h in LEADS])
    bx.set_yticks(range(len(names)), [f"{k + 1} {SHORT[n]}" for k, n in enumerate(names)], fontsize=6.3)
    bx.set_xlabel("Lead time (× 8 days)")
    cb = fig.colorbar(im, ax=bx, fraction=0.05, pad=0.03)
    cb.set_label(r"$\Delta$AUPRC, Pyree − TeleViT$_{i,g}$ (pp)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.3)
    panel_label(bx, "b", x=-0.55)

    # (c-f) 시계열
    for q, n in enumerate(HIGHLIGHT):
        cx = fig.add_subplot(gs[1 + q // 2, q % 2])
        d = res.get(f"{n}|{TS_LEAD}")
        if d is None:
            cx.axis("off")
            continue
        ts = d["timeseries"]
        x = np.arange(len(ts["dates"]))
        cx.bar(x, ts["obs"], color="0.8", width=0.8, label="Observed burned pixels")
        cx.plot(x, ts["clim"], color=COLORS["clim_recency"], ls="-.", lw=1.2, label=NAME["clim_recency"])
        cx.plot(x, ts["televit"], color=COLORS["televit_repro"], lw=1.4, marker="s", ms=2.5,
                label=NAME["televit_repro"])
        cx.plot(x, ts["ours"], color=COLORS["ours"], lw=1.8, marker="o", ms=2.8, label=NAME["ours"])
        step = max(1, len(x) // 5)
        cx.set_xticks(x[::step], [s[5:] for s in ts["dates"][::step]], fontsize=6.3)
        cx.set_ylabel("Burned pixels (0.25°)", fontsize=7)
        k = names.index(n)
        cx.text(0.02, 0.97, f"{k + 1}. {n}\nAUPRC Pyree {d['ours_auprc']:.2f} · TeleViT "
                f"{d['televit_auprc']:.2f} · clim {d['clim_auprc']:.2f}",
                transform=cx.transAxes, va="top", fontsize=6.2,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        cx.set_ylim(0, max(max(ts["obs"]), max(ts["ours"]), max(ts["televit"]), max(ts["clim"])) * 1.45)
        cx.grid(axis="y", lw=0.4, alpha=0.4)
        panel_label(cx, "cdef"[q], x=-0.2)
        if q == 0:
            handles, labels = cx.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=4,
               frameon=False, fontsize=6.8,
               title=f"Expected burned pixels = sum of forecast probabilities, lead {TS_LEAD} × 8 days",
               title_fontsize=6.8)
    save(fig, "fig7_case_studies")

    with open(OUT / "table_case_studies.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["event", "lead", "n_dates", "n_burned_px", "AUPRC_ours", "AUPRC_televit",
                    "AUPRC_clim", "BSS_ours", "BSS_televit", "dAUPRC_ours_minus_televit",
                    "CI_low", "CI_high", "dAUPRC_ours_minus_clim", "CI_low", "CI_high"])
        for n in names:
            for h in LEADS:
                d = res.get(f"{n}|{h}")
                if d is None:
                    continue
                w.writerow([n, h, d["n_dates"], d["n_burned"], f"{d['ours_auprc']:.4f}",
                            f"{d['televit_auprc']:.4f}", f"{d['clim_auprc']:.4f}",
                            f"{d['ours_bss']:+.3f}", f"{d['televit_bss']:+.3f}",
                            f"{d['ours-televit']:+.4f}", f"{d['ours-televit_ci'][0]:+.4f}",
                            f"{d['ours-televit_ci'][1]:+.4f}", f"{d['ours-clim']:+.4f}",
                            f"{d['ours-clim_ci'][0]:+.4f}", f"{d['ours-clim_ci'][1]:+.4f}"])


if __name__ == "__main__":
    main()
