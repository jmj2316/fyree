"""Fig.9: 전체(육지) 패치 평가 — 불 유무와 무관한 모든 육지 타일, 육지 픽셀만 채점 (2019, 2020)."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import COLORS, LEADS, NAME, OUT, panel_label, save, style

d = json.loads((OUT / "work" / "land_scores.json").read_text())
style()
x = np.log2(LEADS)
fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), gridspec_kw={"wspace": 0.55})
YEARS = [("test", "2019", "-"), ("test2020", "2020", "--")]

ax = axes[0]
for split, yr, ls in YEARS:
    for k, col in [("ours", COLORS["ours"]), ("televit", COLORS["televit_repro"])]:
        ax.plot(x, [d[f"{split}|{h}"][f"{k}_auprc"] for h in LEADS], color=col, ls=ls, marker="o", ms=3, lw=1.5)
    ax.axhline(d[f"{split}|1"]["clim_auprc"], color=COLORS["clim_recency"], ls=ls, lw=0.9, alpha=0.8)
ax.set_ylabel("AUPRC (all land patches)")
panel_label(ax, "a", x=-0.3)

bx = axes[1]
for split, yr, ls in YEARS:
    for k, col in [("ours", COLORS["ours"]), ("televit", COLORS["televit_repro"])]:
        bx.plot(x, [d[f"{split}|{h}"][f"{k}_bss"] for h in LEADS], color=col, ls=ls, marker="o", ms=3, lw=1.5)
bx.axhline(0, color=COLORS["clim_recency"], lw=0.9)
bx.set_ylabel("BSS vs. climatology")
panel_label(bx, "b", x=-0.3)

cx = axes[2]
for (split, yr, ls), dx in zip(YEARS, (-0.07, 0.07)):
    m = np.array([d[f"{split}|{h}"]["ours-televit"] for h in LEADS])
    ci = np.array([d[f"{split}|{h}"]["ours-televit_ci"] for h in LEADS])
    cx.errorbar(x + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o", ls=ls, color=COLORS["televit_repro"],
                ms=3.5, lw=1.2, capsize=2)
cx.axhline(0, color="k", lw=0.7)
cx.text(0.97, 0.95, r"Fyree − TeleViT$_{i,g}$", transform=cx.transAxes, ha="right", va="top", fontsize=7)
cx.set_ylabel(r"Paired $\Delta$AUPRC (95% CI)")
panel_label(cx, "c", x=-0.3)

for a in axes:
    a.set_xticks(x, [str(h) for h in LEADS])
    a.set_xlabel("Lead time (× 8 days)")
    a.grid(axis="y", lw=0.4, alpha=0.4)

handles = [Line2D([], [], color=COLORS["ours"], lw=1.8, label=NAME["ours"]),
           Line2D([], [], color=COLORS["televit_repro"], lw=1.8, label=NAME["televit_repro"]),
           Line2D([], [], color=COLORS["clim_recency"], lw=0.9, alpha=0.8,
                  label=NAME["clim_recency"] + " (gray lines in a, b)"),
           Line2D([], [], color="0.3", ls="-", lw=1.2, label="2019 (test)"),
           Line2D([], [], color="0.3", ls="--", lw=1.2, label="2020 (second held-out year)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False,
           fontsize=7, columnspacing=1.3)
save(fig, "fig9_all_land_patches")
