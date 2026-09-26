"""Fig.6: (a) 단일 요소 절제 (변형 − 전체 모델, 짝지은 bootstrap CI), (b) 파라미터 vs AUPRC, (c) 지연시간."""
import json

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, LEADS, NAME, RUNS, panel_label, save, style, test_auprc

VARIANTS = [
    ("abl_nofire", "− fire\nhistory"),
    ("abl_noclim", "− climatology\nchannels"),
    ("abl_noanom", "− accumulated\nanomalies"),
    ("abl_nooci", "− teleconnection\nindices"),
    ("abl_notau", "− recency\nweighting"),
    ("v4_anchor", "+ logit\nanchoring"),
    ("televit_info", "TeleViT$_{i,g}$ backbone,\nsame inputs"),
]
LEAD_COL = ["#fdd49e", "#fdbb84", "#fc8d59", "#e34a33", "#b30000"]

style()
full = test_auprc("v4_noanchor")
fig = plt.figure(figsize=(7.2, 5.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.62, wspace=0.42)

# (a) 절제
ax = fig.add_subplot(gs[0, :])
avail = [(r, l) for r, l in VARIANTS if (RUNS / r / "results.json").exists()]
w = 0.16
for k, (run, lab) in enumerate(avail):
    res = test_auprc(run)
    bfile = RUNS / f"bootstrap_full_vs_{run}.json"
    bt = json.loads(bfile.read_text()) if bfile.exists() else None
    for j, h in enumerate(LEADS):
        d = (res[h]["auprc"] - full[h]["auprc"]) * 100
        xpos = k + (j - 2) * w
        err = None
        if bt is not None:
            lo, hi = bt[str(h)]["A-B_ci"]
            # 변형(B) − 전체(A) 의 CI = −(A−B 의 CI)
            err = [[abs(d - (-hi * 100))], [abs((-lo * 100) - d)]]
        ax.bar(xpos, d, width=w, color=LEAD_COL[j], edgecolor="0.3", lw=0.3,
               yerr=err, capsize=1.2, error_kw={"lw": 0.6},
               label=f"lead {h} × 8 d" if k == 0 else None)
ax.axhline(0, color="k", lw=0.7)
ax.set_xticks(range(len(avail)), [l for _, l in avail], fontsize=6.2)
ax.tick_params(axis="x", length=0, pad=4)
ax.set_ylabel("Test ΔAUPRC vs. full model\n(percentage points)")
ax.grid(axis="y", lw=0.4, alpha=0.4)
ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), frameon=False, fontsize=6.8)
panel_label(ax, "a", x=-0.09)

# (b) 파라미터 vs AUPRC
lat = json.loads((RUNS / "latency.json").read_text())
bx = fig.add_subplot(gs[1, 0])
pts = [("ours", lat["ours_params_M"], full),
       ("ours_anchor", lat["ours_params_M"] * 1.04, test_auprc("v4_anchor")),
       ("televit_repro", lat["televit_params_M"], {h: test_auprc(f"televit_orig_h{h}")[h] for h in LEADS}),
       ("televit_info", lat["televit_info_params_M"], test_auprc("televit_info"))]
for key, p, r in pts:
    bx.scatter(p, r[1]["auprc"], s=36, facecolors="none", edgecolors=COLORS[key], lw=1.3)
    bx.scatter(p, r[16]["auprc"], s=36, color=COLORS[key], label=NAME[key])
    bx.plot([p, p], [r[16]["auprc"], r[1]["auprc"]], color=COLORS[key], lw=0.8, alpha=0.6)
bx.axhline(full[1]["clim_auprc"], color=COLORS["clim_recency"], ls="-.", lw=1.0,
           label=NAME["clim_recency"] + " (0 params)")
bx.set_xscale("log")
bx.set_xticks([20, 40, 80], ["20", "40", "80"])
bx.xaxis.set_minor_formatter(plt.NullFormatter())
bx.set_xlim(14, 120)
bx.set_xlabel("Parameters (millions, log scale)")
bx.set_ylabel("Test AUPRC")
bx.grid(lw=0.4, alpha=0.4)
bx.scatter([], [], s=30, facecolors="none", edgecolors="0.3", label="open: lead 1 × 8 d")
bx.scatter([], [], s=30, color="0.3", label="filled: lead 16 × 8 d")
panel_label(bx, "b", x=-0.25)

# (c) 지연시간
cx = fig.add_subplot(gs[1, 1])
lab_ms = [("ours", lat["ours_ms_per_batch64"]), ("televit_repro", lat["televit_ms_per_batch64"])]
cx.barh([1, 0], [v for _, v in lab_ms], color=[COLORS[k] for k, _ in lab_ms], height=0.55)
for yv, (_, v) in zip([1, 0], lab_ms):
    cx.text(v + 0.8, yv, f"{v:.1f} ms", va="center", fontsize=7)
cx.set_yticks([1, 0], ["Fyree", "TeleViT$_{i,g}$\n(reproduced)"], fontsize=7)
cx.set_xlabel("Inference time per 64 tiles (ms, A40, fp16)")
cx.set_xlim(0, max(v for _, v in lab_ms) * 1.3)
panel_label(cx, "c", x=-0.42)

h1, l1 = bx.get_legend_handles_labels()
fig.legend(h1, l1, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=False,
           fontsize=6.8, columnspacing=1.2)
save(fig, "fig6_ablation_efficiency")
print("variants plotted:", [r for r, _ in avail])
