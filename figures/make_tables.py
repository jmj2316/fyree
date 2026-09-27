"""결과 표: test 2019, 리드별 AUPRC / Brier / BSS(최근가중 기후값 기준). CSV + Markdown."""
import csv

from common import LEADS, OUT, RUNS, paper_digitized, test_auprc

ref = test_auprc("v4_noanchor")  # 최근가중 기후값의 AUPRC·Brier는 이 런의 clim 필드
CLIM_B = {h: ref[h]["clim_brier"] for h in LEADS}
uni = test_auprc("v2_full")

rows = []
rows.append(("Climatology, 2002–2017 mean (no training)",
             {h: (uni[h]["clim_auprc"], uni[h]["clim_brier"]) for h in LEADS}))
rows.append(("Climatology, recency-weighted tau=4 yr (no training)",
             {h: (ref[h]["clim_auprc"], ref[h]["clim_brier"]) for h in LEADS}))
paper = paper_digitized()["TeleViT_ig"]
rows.append(("TeleViT_i,g, published (digitized from Fig. 2)", {h: (paper[h], None) for h in LEADS}))
rows.append(("TeleViT_i,g, reproduced (one model per lead)",
             {h: (test_auprc(f"televit_orig_h{h}")[h]["auprc"], test_auprc(f"televit_orig_h{h}")[h]["brier"])
              for h in LEADS}))
named = [("televit_info", "TeleViT_i,g backbone + Pyree inputs"),
         ("v2_full", "Pyree development v2: logit anchor + fire memory (uniform climatology)"),
         ("v3_anom", "Pyree development v3: + accumulated anomalies + 1-yr burn frequency"),
         ("v4_anchor", "Pyree, logit-anchored"),
         ("v4_noanchor", "Pyree (main model)"),
         ("abl_nofire", "Ablation: Pyree − fire history"),
         ("abl_noclim", "Ablation: Pyree − climatology channels"),
         ("abl_noanom", "Ablation: Pyree − accumulated anomalies"),
         ("abl_nooci", "Ablation: Pyree − teleconnection indices"),
         ("abl_notau", "Ablation: Pyree − recency weighting")]
for run, lab in named:
    if (RUNS / run / "results.json").exists():
        r = test_auprc(run)
        rows.append((lab, {h: (r[h]["auprc"], r[h]["brier"]) for h in LEADS}))

with open(OUT / "table_test_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["model"] + [f"{m}_h{h}" for h in LEADS for m in ("AUPRC", "Brier", "BSS_vs_recency_clim")])
    for lab, d in rows:
        line = [lab]
        for h in LEADS:
            a, b = d[h]
            line += [f"{a:.4f}", "" if b is None else f"{b:.5f}",
                     "" if b is None else f"{1 - b / CLIM_B[h]:+.3f}"]
        w.writerow(line)

md = ["| Model | " + " | ".join(f"h={h} AUPRC (BSS)" for h in LEADS) + " |",
      "|---|" + "---|" * len(LEADS)]
for lab, d in rows:
    cells = []
    for h in LEADS:
        a, b = d[h]
        cells.append(f"{a:.3f}" + ("" if b is None else f" ({1 - b / CLIM_B[h]:+.3f})"))
    md.append(f"| {lab} | " + " | ".join(cells) + " |")
(OUT / "table_test_results.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
