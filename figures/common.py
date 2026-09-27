"""그림 공통: 스타일, 색, 결과 로더. 모든 그림은 제목 없음, 영어 라벨, 범례는 축 밖."""
import os
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

ROOT = Path(os.environ.get("PYREE_ROOT", Path(__file__).resolve().parents[1]))
RUNS = Path(os.environ.get("PYREE_RUNS", ROOT / "runs"))
OUT = ROOT / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
LEADS = [1, 2, 4, 8, 16]

COLORS = {
    "ours": "#c0392b",
    "ours_anchor": "#8c510a",
    "televit_repro": "#6a1b9a",
    "televit_info": "#9e7cc1",
    "clim_recency": "#4d4d4d",
    "clim_uniform": "#9e9e9e",
    # 논문 Fig.2 팔레트 (tab10 + purple) 그대로
    "paper_U-Net++": "#1f77b4",
    "paper_ViT": "#ff7f0e",
    "paper_TeleViT_i": "#2ca02c",
    "paper_TeleViT_g": "#17becf",
    "paper_TeleViT_ig": "#800080",
}

NAME = {
    "ours": "Pyree",
    "ours_anchor": "Pyree, logit-anchored",
    "televit_repro": r"TeleViT$_{i,g}$ (reproduced)",
    "televit_info": r"TeleViT$_{i,g}$ backbone + Pyree inputs",
    "clim_recency": "Climatology (recency-weighted)",
    "clim_uniform": "Climatology (2002–2017 mean)",
}


def style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    print("saved", OUT / f"{name}.png")


def panel_label(ax, s, x=-0.12, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")


def test_auprc(run):
    r = json.loads((RUNS / run / "results.json").read_text())["test"]
    return {int(h): v for h, v in r.items()}


def paper_digitized():
    """TeleViT 논문 Fig.2 (raster) 픽셀 디지타이즈 결과, AUPRC 비율(0-1)."""
    raw = json.loads((OUT / "work" / "televit_fig2_digitized_raw.json").read_text())
    return {k: {d["h"]: d["v_box"] / 100 for d in v} for k, v in raw.items()}


def bootstrap_table():
    """v4_noanchor(A) vs TeleViT 재현(B) vs 기후값(recency) — 리드별 짝지은 bootstrap."""
    rows = {}
    for h in LEADS:
        d = json.loads((RUNS / f"bootstrap_v4_vs_televit_h{h}.json").read_text())[str(h)]
        rows[h] = d
    return rows
