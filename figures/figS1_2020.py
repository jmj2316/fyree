"""Fig.S1: 두 번째 held-out 연도(2020) — 리드별 AUPRC (a) + 짝지은 격차 (b). 표도 함께."""
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from common import COLORS, LEADS, NAME, OUT, RUNS, panel_label, save, style


def ev(run):
    return {int(h): v for h, v in json.loads((RUNS / run / "test2020_eval.json").read_text()).items()}


style()
ours, ours_a, info = ev("v4_noanchor"), ev("v4_anchor"), ev("televit_info")
tv = {h: ev(f"televit_orig_h{h}")[h] for h in LEADS}
clim_rec = ours[1]["clim_auprc"]
clim_uni = ev("abl_notau")[1]["clim_auprc"]
bt = {h: json.loads((RUNS / f"bootstrap2020_v4_vs_televit_h{h}.json").read_text())[str(h)] for h in LEADS}

x = np.log2(LEADS)
fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 2.9), gridspec_kw={"width_ratios": [1.25, 1]})
ax.axhline(clim_uni, color=COLORS["clim_uniform"], ls=":", lw=1.2, label=NAME["clim_uniform"])
ax.axhline(clim_rec, color=COLORS["clim_recency"], ls="-.", lw=1.2, label=NAME["clim_recency"])
ax.plot(x, [info[h]["auprc"] for h in LEADS], color=COLORS["televit_info"], lw=1.4, marker="D", ms=3.5,
        label=NAME["televit_info"])
ax.plot(x, [tv[h]["auprc"] for h in LEADS], color=COLORS["televit_repro"], lw=1.8, marker="s", ms=4,
        label=NAME["televit_repro"])
ax.plot(x, [ours_a[h]["auprc"] for h in LEADS], color=COLORS["ours_anchor"], lw=1.4, marker="^", ms=4,
        label=NAME["ours_anchor"])
ax.plot(x, [ours[h]["auprc"] for h in LEADS], color=COLORS["ours"], lw=2.2, marker="o", ms=5,
        label=NAME["ours"])
ax.set_xticks(x, [str(h) for h in LEADS])
ax.set_xlabel("Lead time (× 8 days)")
ax.set_ylabel("Test AUPRC (2020, held out)")
ax.grid(axis="y", lw=0.4, alpha=0.4)
panel_label(ax, "a", x=-0.16)

for key, lab, col, dx in [("A-B", "Pyree − TeleViT$_{i,g}$ (reproduced)", COLORS["televit_repro"], -0.12),
                          ("A-clim", "Pyree − climatology", COLORS["clim_recency"], 0.0),
                          ("B-clim", "TeleViT$_{i,g}$ (reproduced) − climatology", COLORS["televit_info"], 0.12)]:
    m = np.array([bt[h][key] for h in LEADS])
    ci = np.array([bt[h][f"{key}_ci"] for h in LEADS])
    bx.errorbar(x + dx, m, yerr=[m - ci[:, 0], ci[:, 1] - m], fmt="o-", color=col, ms=4, lw=1.3,
                capsize=2.5, label=lab)
bx.axhline(0, color="k", lw=0.8)
bx.set_xticks(x, [str(h) for h in LEADS])
bx.set_xlabel("Lead time (× 8 days)")
bx.set_ylabel(r"Paired $\Delta$AUPRC (95% CI)")
bx.grid(axis="y", lw=0.4, alpha=0.4)
panel_label(bx, "b", x=-0.2)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = bx.get_legend_handles_labels()
fig.legend(h1[::-1] + h2, l1[::-1] + l2, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
           frameon=False, columnspacing=1.0, handlelength=2.0, fontsize=6.8)
fig.tight_layout(w_pad=2.0)
save(fig, "figS1_2020_heldout")

# 2020 표
rows = [("Climatology, 2002–2017 mean", {h: (clim_uni, None) for h in LEADS}),
        ("Climatology, recency-weighted", {h: (clim_rec, ours[h]["clim_brier"]) for h in LEADS}),
        ("TeleViT_i,g, reproduced", {h: (tv[h]["auprc"], tv[h]["brier"]) for h in LEADS})]
for run, lab in [("televit_info", "TeleViT_i,g backbone + Pyree inputs"), ("v4_anchor", "Pyree, logit-anchored"),
                 ("v4_noanchor", "Pyree (main)"), ("abl_nofire", "Ablation: − fire history"),
                 ("abl_noclim", "Ablation: − climatology channels"), ("abl_noanom", "Ablation: − anomalies"),
                 ("abl_nooci", "Ablation: − OCIs"), ("abl_notau", "Ablation: − recency weighting")]:
    r = ev(run)
    rows.append((lab, {h: (r[h]["auprc"], r[h]["brier"]) for h in LEADS}))
CB = {h: ours[h]["clim_brier"] for h in LEADS}
with open(OUT / "table_test2020_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model"] + [f"{m}_h{h}" for h in LEADS for m in ("AUPRC", "Brier", "BSS_vs_recency_clim")])
    for lab, d in rows:
        line = [lab]
        for h in LEADS:
            a_, b_ = d[h]
            line += [f"{a_:.4f}", "" if b_ is None else f"{b_:.5f}",
                     "" if b_ is None else f"{1 - b_ / CB[h]:+.3f}"]
        w.writerow(line)
print("rows", len(rows))
