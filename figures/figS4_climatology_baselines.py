"""Fig.S4: TeleViT1.0(저널판, 전체 패치) 공개값 vs 기후값 정의별 AUPRC vs Fyree (전체 육지, 2019)."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from common import COLORS, LEADS, NAME, OUT, panel_label, save, style

# TeleViT1.0 (Prapas et al., arXiv 2512.00089) 표, 리드 0,1,2,4,8,16
PUB = {"ViT": [0.6173, 0.6114, 0.6102, 0.6065, 0.6014, 0.5824],
       "U-Net++": [0.6202, 0.6097, 0.6018, 0.5911, 0.5907, 0.5782],
       "TeleViT_i": [0.6217, 0.6160, 0.6104, 0.6095, 0.6000, 0.5854],
       "TeleViT_g": [0.6229, 0.6140, 0.6113, 0.6119, 0.6079, 0.6031],
       "TeleViT_ig": [0.6302, 0.6226, 0.6160, 0.6155, 0.6081, 0.6016]}
PUB_CLIM = 0.5716
# clim_definitions.py 결과 (우리 파이프라인, 2019 전체 육지 패치, 육지 픽셀)
CLIM = [("Unsmoothed, 2003–2017\n(TeleViT1.0 definition)", 0.5697),
        ("Unsmoothed, 2002–2017", 0.5708),
        ("Uniform, ±1 slot", 0.6046),
        ("Uniform, ±2 slots", 0.6117),
        ("Uniform, ±3 slots", 0.6137),
        ("Recency-weighted, ±3 slots\n(this study)", 0.6292)]
(OUT / "work" / "clim_variants.json").write_text(json.dumps({"published_televit1": PUB, "published_clim": PUB_CLIM,
                                                              "clim_variants": CLIM}, indent=1))
land = json.loads((OUT / "work" / "land_scores.json").read_text())

style()
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.75})
L = [0] + LEADS
x = np.arange(len(L))
pal = {"ViT": COLORS["paper_ViT"], "U-Net++": COLORS["paper_U-Net++"], "TeleViT_i": COLORS["paper_TeleViT_i"],
       "TeleViT_g": COLORS["paper_TeleViT_g"], "TeleViT_ig": COLORS["paper_TeleViT_ig"]}
for k, v in PUB.items():
    ax.plot(x, v, color=pal[k], lw=1.0, ls="--", marker="s", ms=2.8, alpha=0.9)
ax.plot(x[1:], [land[f"test|{h}"]["televit_auprc"] for h in LEADS], color=COLORS["televit_repro"], lw=1.4,
        marker="o", ms=3.2)
ax.plot(x[1:], [land[f"test|{h}"]["ours_auprc"] for h in LEADS], color=COLORS["ours"], lw=2.0, marker="o", ms=4)
ax.axhline(PUB_CLIM, color="0.6", lw=1.0, ls=":")
ax.axhline(CLIM[4][1], color=COLORS["clim_uniform"], lw=1.0)
ax.axhline(CLIM[5][1], color=COLORS["clim_recency"], lw=1.0)
ax.set_xticks(x, [str(h) for h in L])
ax.set_xlabel("Lead time (× 8 days)")
ax.set_ylabel("Test AUPRC, 2019 (all patches)")
ax.set_ylim(0.56, 0.735)
ax.grid(axis="y", lw=0.4, alpha=0.4)
panel_label(ax, "a", x=-0.2)

ys = np.arange(len(CLIM))[::-1]
cols = ["0.6", "0.6", COLORS["clim_uniform"], COLORS["clim_uniform"], COLORS["clim_uniform"], COLORS["clim_recency"]]
bx.barh(ys, [v for _, v in CLIM], color=cols, height=0.62)
for y_, (_, v) in zip(ys, CLIM):
    bx.text(0.5525, y_, f"{v:.3f}", va="center", ha="left", fontsize=6.8, color="white")
bx.axvline(PUB["TeleViT_ig"][1], color=COLORS["paper_TeleViT_ig"], lw=1.0, ls="--")
bx.axvline(PUB["TeleViT_ig"][5], color=COLORS["paper_TeleViT_ig"], lw=1.0, ls=":")
bx.set_yticks(ys, [n for n, _ in CLIM], fontsize=6.6)
bx.set_xlim(0.55, 0.645)
bx.set_xlabel("Climatology AUPRC\n(2019, all land patches)")
panel_label(bx, "b", x=-0.95)

LAB = {"ViT": "ViT", "U-Net++": "U-Net++", "TeleViT_i": r"TeleViT$_{i}$", "TeleViT_g": r"TeleViT$_{g}$",
       "TeleViT_ig": r"TeleViT$_{i,g}$"}
handles = [Line2D([], [], color=COLORS["ours"], lw=2.0, marker="o", ms=4, label=NAME["ours"] + " (all land patches)"),
           Line2D([], [], color=COLORS["televit_repro"], lw=1.4, marker="o", ms=3,
                  label=r"TeleViT$_{i,g}$ reproduced (all land patches)"),
           Line2D([], [], color="0.6", lw=1.0, ls=":", label="Climatology as published in TeleViT1.0 (a)"),
           Line2D([], [], color=COLORS["clim_uniform"], lw=1.0, label="Climatology, uniform ±3 slots (a)"),
           Line2D([], [], color=COLORS["clim_recency"], lw=1.0, label="Climatology, recency-weighted (a)")]
handles += [Line2D([], [], color=pal[k], lw=1.0, ls="--", marker="s", ms=2.8,
                   label=LAB[k] + " as published in TeleViT1.0 (a)") for k in PUB]
handles += [Line2D([], [], color=COLORS["paper_TeleViT_ig"], lw=1.0, ls="--",
                   label=r"Published TeleViT$_{i,g}$, lead 1 (b)"),
            Line2D([], [], color=COLORS["paper_TeleViT_ig"], lw=1.0, ls=":",
                   label=r"Published TeleViT$_{i,g}$, lead 16 (b)")]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False, fontsize=6.4,
           columnspacing=1.0, handlelength=2.6)
save(fig, "figS4_climatology_baselines")
