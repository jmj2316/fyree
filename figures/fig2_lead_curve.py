"""Fig.2: 리드별 test AUPRC (a) + 짝지은 bootstrap 격차 (b)."""
import matplotlib.pyplot as plt
import numpy as np

from common import (COLORS, LEADS, NAME, bootstrap_table, panel_label, paper_digitized, save,
                    style, test_auprc)

style()
ours = test_auprc("v4_noanchor")
ours_a = test_auprc("v4_anchor")
tv = {h: test_auprc(f"televit_orig_h{h}")[h] for h in LEADS}
paper = paper_digitized()
bt = bootstrap_table()
clim_rec = ours[1]["clim_auprc"]
clim_uni = test_auprc("v2_full")[1]["clim_auprc"]

x = np.log2(LEADS)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) 절대 성능
for key, lab in [("U-Net++", "U-Net++"), ("ViT", "ViT"), ("TeleViT_g", r"TeleViT$_{g}$"),
                 ("TeleViT_i", r"TeleViT$_{i}$"), ("TeleViT_ig", r"TeleViT$_{i,g}$")]:
    ax.plot(x, [paper[key][h] for h in LEADS], ls="--", lw=0.9, marker="x", ms=4, alpha=0.75,
            color=COLORS[f"paper_{key}"], label=f"{lab} (published)")
ax.axhline(clim_uni, color=COLORS["clim_uniform"], ls=":", lw=1.2, label=NAME["clim_uniform"])
ax.axhline(clim_rec, color=COLORS["clim_recency"], ls="-.", lw=1.2, label=NAME["clim_recency"])
ax.plot(x, [tv[h]["auprc"] for h in LEADS], color=COLORS["televit_repro"], lw=1.8, marker="s",
        ms=4, label=NAME["televit_repro"])
ax.plot(x, [ours_a[h]["auprc"] for h in LEADS], color=COLORS["ours_anchor"], lw=1.4, marker="^",
        ms=4, label=NAME["ours_anchor"])
ax.plot(x, [ours[h]["auprc"] for h in LEADS], color=COLORS["ours"], lw=2.2, marker="o", ms=5,
        label=NAME["ours"])
ax.set_xticks(x, [str(h) for h in LEADS])
ax.set_xlabel("Lead time (× 8 days)")
ax.set_ylabel("Test AUPRC (2019)")
ax.set_ylim(0.570, 0.735)
ax.grid(axis="y", lw=0.4, alpha=0.4)
panel_label(ax, "a", x=-0.16)

# (b) 짝지은 격차 + 95% CI
pairs = [("A-B", "Pyree − TeleViT$_{i,g}$ (reproduced)", COLORS["televit_repro"], -0.12),
         ("A-clim", "Pyree − climatology", COLORS["clim_recency"], 0.0),
         ("B-clim", "TeleViT$_{i,g}$ (reproduced) − climatology", COLORS["televit_info"], 0.12)]
for key, lab, col, dx in pairs:
    m = np.array([bt[h][key] for h in LEADS])
    ci = np.array([bt[h][f"{key}_ci"] for h in LEADS])
    bx.errorbar(x + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o-", color=col, ms=4, lw=1.3,
                capsize=2.5, label=lab)
bx.axhline(0, color="k", lw=0.8)
bx.set_xticks(x, [str(h) for h in LEADS])
bx.set_xlabel("Lead time (× 8 days)")
bx.set_ylabel(r"Paired $\Delta$AUPRC (95% block-bootstrap CI)")
bx.grid(axis="y", lw=0.4, alpha=0.4)
panel_label(bx, "b", x=-0.2)

# 범례는 두 패널 아래 한 줄 블록으로 (축과 겹치지 않게)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = bx.get_legend_handles_labels()
fig.legend(h1[::-1], l1[::-1], loc="upper center", bbox_to_anchor=(0.33, -0.02), ncol=2,
           frameon=False, columnspacing=1.2, handlelength=2.2)
fig.legend(h2, l2, loc="upper center", bbox_to_anchor=(0.80, -0.02), ncol=1, frameon=False,
           handlelength=2.2)
fig.tight_layout(w_pad=2.0)
save(fig, "fig2_lead_time_auprc")
