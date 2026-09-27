"""Fig.1: Pyree 구조도 + TeleViT 대비 인셋. 제목 없음, 영어."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from common import panel_label, save, style

style()
fig = plt.figure(figsize=(7.2, 4.6))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 100)
ax.set_ylim(0, 64)
ax.axis("off")

C = {"met": "#d6eaf8", "anom": "#d1f2eb", "fire": "#fadbd8", "clim": "#e5e7e9", "pos": "#f9f9f9",
     "cond": "#fdebd0", "slow": "#d5dbdb", "fast": "#f5cba7", "out": "#f2d7d5", "tv": "#ebdef0"}


def box(x, y, w, h, text, fc, fs=6.6, bold=False, ec="0.35", lw=0.8, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.15)


def arrow(x0, y0, x1, y1, ls="-", color="0.25", rad=0.0, lw=0.9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1))


# ---------------- 입력 (왼쪽) ----------------
ax.text(61.25, 62.2, "Pyree: slow–fast transformer (21.0 M parameters)\none model for all lead times", ha="center", va="center",
        fontsize=7.2, fontweight="bold", color="#c0392b")
ax.text(10.5, 62.2, "Inputs at issue time t − h\n(80 × 80 local patch, 0.25°)", ha="center",
        va="top", fontsize=7, fontweight="bold")
inputs = [
    (52.0, "Fire-driver snapshot (10 variables)\nLST, MSLP, NDVI, SST, soil moisture,\nT2m, precipitation, VPD, SSRD, pop.", "met"),
    (43.6, "Accumulated anomalies\n6 variables × {1, 4, 12, 24} steps", "anom"),
    (35.2, "Fire memory\nrecent burns (1, 4, 12, 46 steps),\nsame slot last year, time since last fire", "fire"),
    (26.8, "Fire climatology prior\nrecency-weighted slot frequency\n(target and issue slot)", "clim"),
    (20.6, "Position (sin/cos lat, lon)", "pos"),
]
for y, t, k in inputs:
    h = 4.6 if k == "pos" else 7.0
    box(0.8, y, 21.0, h, t, C[k], fs=5.5)

box(0.8, 9.4, 21.0, 7.4, "Teleconnection indices\n10 OCIs × 10 months\n(ENSO, PDO, NAO, SOI, ...)", C["cond"], fs=5.9)
box(0.8, 2.2, 21.0, 5.2, "Lead time h\n(1–16 × 8 days, one model)", C["cond"], fs=5.9)

# ---------------- 변수 인지 분할 ----------------
box(25.2, 43.0, 13.6, 9.0, "Slow group\n32 channels\n(land state, memory,\nclimatology, ≥1-month\nanomalies)", C["slow"], fs=5.5)
box(25.2, 27.5, 13.6, 9.0, "Fast group\n14 channels\n(weather snapshot,\nrecent burns,\n8-day anomalies)", C["fast"], fs=5.5)
for y, _, k in inputs:
    yc = y + (2.3 if k == "pos" else 3.5)
    to_slow = k in ("clim", "pos", "anom", "fire", "met")
    arrow(22.2, yc, 25.2, 47.5 if yc > 38 else 44.5, color="0.55", lw=0.6)
    if k in ("met", "fire", "anom"):
        arrow(22.2, yc, 25.2, 33.0, color="0.55", lw=0.6)
ax.text(32.0, 40.0, "variable-aware\nsplit", ha="center", va="center", fontsize=5.8, style="italic",
        color="0.3")

# ---------------- 패치 임베딩 + Step1/Step2 ----------------
box(41.3, 44.2, 10.4, 6.6, "Patch embed\n4 × 4 patches\n400 tokens × 192", "white", fs=5.5)
box(41.3, 28.7, 10.4, 6.6, "Patch embed\n4 × 4 patches\n400 tokens × 192", "white", fs=5.5)
arrow(39.1, 47.5, 41.2, 47.5)
arrow(39.1, 32.0, 41.2, 32.0)

box(54.5, 42.2, 13.5, 10.6, "Step 1: slow path\n6 × transformer block\nwidth 192\n(narrow, deep\nfor slow drivers)", C["slow"], fs=5.9, bold=False)
arrow(51.8, 47.5, 54.2, 47.5)
box(54.5, 26.2, 13.5, 11.6, "Step 2: full width\nconcat[Step 1 ‖ fast]\n6 × transformer block\nwidth 384", C["fast"], fs=5.9)
arrow(51.8, 32.0, 54.2, 32.0)
arrow(61.25, 42.0, 61.25, 38.1)

# ---------------- AdaLN 조건 ----------------
box(25.5, 5.8, 26.0, 9.0, "Conditioning vector c\nc = MLP(OCIs) + MLP(h)\ninjected into every block by\nAdaLN-Zero (scale, shift, gate)", C["cond"], fs=5.9)
arrow(22.2, 13.1, 25.2, 11.5)
arrow(22.2, 4.8, 25.2, 8.5)
OR = "#b9770e"
ax.plot([51.9, 53.0, 53.0], [10.3, 10.3, 44.0], ls="--", color=OR, lw=0.9)
arrow(53.0, 44.0, 54.3, 44.0, ls="--", color=OR)
arrow(53.0, 28.0, 54.3, 28.0, ls="--", color=OR)
ax.text(53.6, 20.0, "AdaLN to\nevery block", fontsize=5.6, color=OR, style="italic", va="center")

# ---------------- 출력 ----------------
box(71.5, 29.0, 11.5, 6.2, "Per-token head\n+ unpatchify", "white", fs=5.9)
arrow(68.2, 32.0, 71.2, 32.0)
box(86.0, 28.3, 13.0, 7.6, "P(burned area)\nat t, 80 × 80 pixels\n(each token → its own\n4 × 4 pixels)", C["out"], fs=5.8, bold=False)
arrow(83.2, 32.0, 85.7, 32.0)
ax.text(79.0, 38.8, "spatially resolved output:\npixel-level fire memory reaches\nthe matching output pixels",
        ha="center", va="center", fontsize=5.6, color="#922b21", style="italic")

# ---------------- TeleViT 대비 인셋 ----------------
ax.add_patch(FancyBboxPatch((63.0, 1.2), 36.3, 20.4, boxstyle="round,pad=0.3,rounding_size=1.0",
                            fc="#fbfaff", ec="#7d3c98", lw=0.8, ls="--"))
ax.text(81.15, 20.2, r"Contrast: TeleViT$_{i,g}$ (Prapas et al., 2023)", ha="center", va="center",
        fontsize=6.3, fontweight="bold", color="#6c3483")
tv = [(64.2, "[cls]"), (70.0, "local\n25 tok"), (76.0, "OCI\n100 tok"), (82.0, "global 1°\n72 tok")]
for x, t in tv:
    box(x, 12.4, 4.6 if t == "[cls]" else 5.0, 4.6, t, C["tv"], fs=5.1)
box(88.6, 12.4, 9.8, 4.6, "ViT ×8\nwidth 768", C["tv"], fs=5.4)
arrow(87.6, 14.7, 88.4, 14.7)
box(87.2, 3.2, 11.2, 5.4, "cls token only\n(1 × 768)", C["tv"], fs=5.4)
box(75.6, 3.2, 8.8, 5.4, "Linear\n768 → 6400", C["tv"], fs=5.4)
box(64.2, 3.2, 8.6, 5.4, "80 × 80\npixels", C["tv"], fs=5.4)
arrow(93.5, 12.2, 93.5, 8.8)
arrow(87.0, 5.9, 84.6, 5.9)
arrow(75.4, 5.9, 73.0, 5.9)
ax.text(76.0, 10.3, "all 6400 pixels decoded from one token\n(spatial bottleneck)",
        ha="center", va="center", fontsize=5.3, color="#6c3483", style="italic")

save(fig, "fig1_architecture")
