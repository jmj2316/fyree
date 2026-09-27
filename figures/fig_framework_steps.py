"""Pyree 구조도 (Sonny 2-Steps Network 스타일): 범례 줄 + 가로 파이프라인. 제목 없음, 영어."""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from common import save, style

style()
W, H = 100, 47
fig = plt.figure(figsize=(7.2, 7.2 * H / W))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.set_aspect("equal")
ax.axis("off")

BLUE, BLUE_F, BLUE_L = "#1f5f99", "#5b9bd5", "#dce9f6"
RED, RED_F = "#b03a2e", "#e8756b"
PUR, PUR_F, PUR_L = "#6c3483", "#a569bd", "#efe4f4"
ORA = "#c77c11"
GREY_F, GREY_E = "#9a9a9a", "#4d4d4d"
FS = 6.4


def rbox(x, y, w, h, fc, ec="0.3", lw=0.8, ls="-", r=0.9, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def text(x, y, s, fs=FS, color="black", **kw):
    ax.text(x, y, s, fontsize=fs, color=color, zorder=6, **kw)


def arrow(x0, y0, x1, y1, color="0.2", lw=0.9, ls="-", rad=0.0, style_="-|>", z=3):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style_, mutation_scale=7, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0,
                                 zorder=z))


def line(xs, ys, color="0.2", lw=0.9, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="round")


def block(x, y, w, h, n, width):
    """회색 ViT 블록 + 점선 잔차 화살표 + × n."""
    rbox(x - 0.35, y - 0.35, w + 0.7, h + 0.7, "none", ec="0.2", lw=0.6, r=0.9, z=4)
    rbox(x, y, w, h, GREY_F, ec=GREY_E, lw=0.9, r=0.7, z=4)
    text(x + w / 2, y + h / 2, "Block", fs=6.2, ha="center", va="center", rotation=90)
    ax.add_patch(FancyArrowPatch((x + w + 0.8, y + 1.0), (x + w + 0.8, y + h - 1.0), arrowstyle="<|-|>",
                                 mutation_scale=5, color=RED, lw=0.7, ls=(0, (2, 1.5)),
                                 connectionstyle="arc3,rad=-0.45", zorder=5))
    text(x + w + 2.9, y + h / 2 + 0.9, rf"$\times$ {n}", fs=7.2, va="center")
    text(x + w + 2.9, y + h / 2 - 1.6, f"width {width}", fs=5.6, va="center", color="0.3")


def circ(x, y, s, fc, r=1.45):
    ax.add_patch(Circle((x, y), r, fc=fc, ec="0.25", lw=0.7, zorder=5))
    text(x, y - 0.05, s, fs=6.6, ha="center", va="center", color="white", fontweight="bold")


# ---------------- 범례 줄 ----------------
ly = 43.6
rbox(0.8, ly - 1.4, 5.2, 2.8, GREY_F, ec=GREY_E, r=0.6)
rbox(0.45, ly - 1.75, 5.9, 3.5, "none", ec="0.2", lw=0.6, r=0.8)
text(3.4, ly, "Block", fs=5.6, ha="center", va="center")
text(7.2, ly, ": transformer block with\n  residual connection", va="center", fs=5.9)
rbox(24.0, ly - 1.6, 3.6, 3.2, "#f4f4f4", ec="0.4", lw=0.7, ls=(0, (3, 2)), r=0.6)
text(28.3, ly, ": residual sub-network", va="center", fs=5.9)
circ(43.0, ly, "S", "0.45", r=1.25)
circ(46.0, ly, "C", PUR_F, r=1.25)
text(47.8, ly, ": split / concatenate", va="center", fs=5.9)
leg = [(BLUE_F, "Slow channels (32): climatology, long fire memory,\nland state, anomalies ≥ 1 month, position"),
       (RED_F, "Fast channels (14): weather snapshot,\nrecent burns, 8-day anomalies")]
for k, (c, s) in enumerate(leg):
    rbox(64.5, ly + 0.3 - k * 4.1, 2.4, 1.9, c, ec="0.3", r=0.4)
    text(67.4, ly + 1.25 - k * 4.1, s, va="center", fs=5.6)
rbox(64.5, ly - 7.9, 2.4, 1.9, PUR_F, ec="0.3", r=0.4)
text(67.4, ly - 6.95, "Coupled slow + fast tokens (full width 384)", va="center", fs=5.6)
rbox(64.5, ly - 11.0, 2.4, 1.9, "#f6d7a7", ec=ORA, r=0.4)
text(67.4, ly - 10.05, "Conditioning (AdaLN-Zero)", va="center", fs=5.6)

# ---------------- 파이프라인 ----------------
yc = 21.0  # 중심선
# 입력
text(4.6, yc + 1.6, "Inputs at\nissue time", fs=6.2, ha="center", va="center")
text(4.6, yc - 2.2, r"$t-h$" + "\n46 ch × 80 × 80", fs=5.8, ha="center", va="center")
arrow(9.0, yc, 11.0, yc)
rbox(11.0, yc - 3.2, 7.4, 6.4, "#cfc9dd", ec="0.35")
text(14.7, yc + 0.9, "Patch", fs=6.4, ha="center", va="center")
text(14.7, yc - 1.1, "embed 4×4", fs=5.8, ha="center", va="center")
arrow(18.4, yc, 20.2, yc)
# 토큰 점 + 회색 x
for k in range(9):
    for j in range(2):
        ax.add_patch(Circle((20.9 + j * 0.9, yc - 7.0 + k * 1.75), 0.33, fc=RED_F, ec="none", zorder=4))
arrow(23.3, yc + 7.4, 23.3, yc - 7.4, color=RED_F, lw=0.8, style_="<|-|>")
rbox(24.3, yc - 8.0, 3.2, 16.0, GREY_F, ec=GREY_E, r=0.7)
text(25.9, yc, r"$x$", fs=8.5, ha="center", va="center")
text(25.9, yc - 9.6, "400 tokens", fs=5.5, ha="center", va="center", color="0.3")
arrow(27.5, yc, 29.4, yc)
circ(30.9, yc, "S", "0.45")

# Step 1: 느린 경로
yb = yc + 7.0
line([30.9, 30.9], [yc + 1.45, yb], color=BLUE)
arrow(30.9, yb, 35.4, yb, color=BLUE)
text(31.4, yb + 0.9, "slow", fs=5.8, color=BLUE, va="bottom")
rbox(35.4, yb - 6.4, 16.0, 12.8, BLUE_L, ec=BLUE, lw=0.9, ls=(0, (4, 2.5)), r=1.2)
text(43.4, yb + 7.4, "Step 1: slow path", fs=6.8, color=BLUE, ha="center", va="bottom", fontweight="bold")
block(37.6, yb - 4.8, 3.4, 9.6, 6, 192)
text(50.0, yb - 5.6, r"$\mathcal{F}_1$", fs=7.5, color=BLUE, ha="center", va="bottom")
arrow(51.4, yb, 53.2, yb, color=BLUE)
rbox(53.2, yb - 3.6, 3.8, 7.2, BLUE_F, ec=BLUE)
text(55.1, yb, r"$y_1$", fs=8, ha="center", va="center", color="white")

# 빠른 경로
yf = yc - 9.0
line([30.9, 30.9], [yc - 1.45, yf], color=RED)
arrow(30.9, yf, 37.4, yf, color=RED)
text(31.4, yf - 0.9, "fast", fs=5.8, color=RED, va="top")
rbox(37.4, yf - 2.4, 4.4, 4.8, RED_F, ec=RED)
text(39.6, yf, r"$x_2$", fs=8, ha="center", va="center", color="white")
text(45.4, yf + 0.7, "fast tokens (width 192)", fs=5.6, color=RED, ha="left", va="bottom")

# 결합
xc = 60.3
line([57.0, 58.8], [yb, yb], color=BLUE)
arrow(58.8, yb, xc, yc + 1.5, color=BLUE, rad=-0.35)
line([41.8, 58.8], [yf, yf], color=RED)
arrow(58.8, yf, xc, yc - 1.5, color=RED, rad=0.35)
circ(xc, yc, "C", PUR_F)

# Step 2
arrow(xc + 1.45, yc, 64.4, yc, color=PUR)
rbox(64.4, yc - 7.5, 15.0, 15.0, PUR_L, ec=PUR, lw=0.9, ls=(0, (4, 2.5)), r=1.2)
text(71.9, yc + 8.3, "Step 2: full-width coupled blocks", fs=6.8, color=PUR, ha="center", va="bottom",
     fontweight="bold")
block(66.6, yc - 5.4, 3.4, 10.8, 6, 384)
text(78.2, yc - 6.7, r"$\mathcal{F}_2$", fs=7.5, color=PUR, ha="center", va="bottom")
arrow(79.4, yc, 81.0, yc, color=PUR)
rbox(81.0, yc - 5.5, 3.6, 11.0, PUR_F, ec=PUR)
text(82.8, yc, r"$y_2$", fs=8, ha="center", va="center", color="white")
arrow(84.6, yc, 86.2, yc, color=PUR)
rbox(86.2, yc - 6.2, 13.4, 12.4, "#f6e3e1", ec="0.35")
text(92.9, yc + 3.3, "Per-token head", fs=6.3, ha="center", va="center", fontweight="bold")
text(92.9, yc - 1.6, "each token → its own\n4 × 4 pixels\n" + r"$P$" + "(burned) at " + r"$t$" + ", 80 × 80",
     fs=5.6, ha="center", va="center", linespacing=1.2)

# ---------------- 조건 벡터 (AdaLN-Zero) ----------------
ycd = 4.2
rbox(0.8, ycd - 2.9, 26.0, 5.8, "#fdf1dc", ec=ORA, lw=0.8)
text(13.8, ycd + 1.0, "Teleconnection indices (10 × 10 months)", fs=5.9, ha="center", va="center")
text(13.8, ycd - 1.3, r"+ lead time $h$ (1–16 × 8 days, one model)", fs=5.9, ha="center", va="center")
arrow(26.8, ycd, 30.0, ycd, color=ORA)
rbox(30.0, ycd - 2.6, 10.2, 5.2, "#f6d7a7", ec=ORA, lw=0.8)
text(35.1, ycd, r"$c$ = MLP + MLP", fs=6.0, ha="center", va="center")
line([40.2, 71.9], [ycd, ycd], color=ORA, ls=(0, (3, 2)))
arrow(43.4, ycd, 43.4, yb - 6.4, color=ORA, ls=(0, (3, 2)))
arrow(71.9, ycd, 71.9, yc - 7.5, color=ORA, ls=(0, (3, 2)))
text(57.6, ycd - 0.8, "AdaLN-Zero (scale, shift, gate) in every block", fs=5.8, color=ORA, ha="center",
     va="top", style="italic")

save(fig, "fig_framework_steps")
