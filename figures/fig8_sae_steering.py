"""Fig.8: Step-1(slow path) SAE 해석과 인과적 steering 검증."""
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D

from common import OUT, ROOT, RUNS, panel_label, save, style

sys.path.insert(0, str(ROOT))
from pyree.sae.interpret import load_sae  # noqa: E402

SDIR = RUNS / "v4_noanchor" / "sae"
SAES = [("topk", "TopK"), ("jumprelu_l01.0", "JumpReLU"), ("kan", "KAN (B-spline)")]
SCOL = {"topk": "#7f8c8d", "jumprelu_l01.0": "#2e86c1", "kan": "#c0392b"}
CONCEPTS = [("clim_target", "Climatology\n(target slot)"), ("fire_tslf", "Time since\nlast fire"),
            ("anom_tp_w24", "Precip. anomaly\n(6 months)"), ("anom_vpd_w4", "VPD anomaly\n(1 month)")]
HEAT_ROWS = [("clim_target", "Climatology (target)"), ("fire_tslf", "Time since last fire"),
             ("fire_issue", "Burned at issue time"), ("fire_recent46", "Burn freq., 1 yr"),
             ("anom_tp_w24", "Precip. anomaly, 6 mo"), ("anom_vpd_w4", "VPD anomaly, 1 mo"),
             ("ndvi", "NDVI"), ("swvl1", "Soil moisture"), ("gmsst", "GMSST (OCI)")]


def main():
    style()
    interp = json.loads((SDIR / "interpret.json").read_text())
    steer = json.loads((SDIR / "steer_eval_rand20.json").read_text())
    logs = {s: json.loads((SDIR / f"sae_{s}_log.json").read_text())[-1] for s, _ in SAES}

    fig = plt.figure(figsize=(7.2, 8.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.9, 1.05, 0.85], hspace=0.62, wspace=0.55)

    # (a) SAE 비교
    ax = fig.add_subplot(gs[0, 0])
    mets = [("EV (test)", lambda s: logs[s]["test_ev"]),
            ("Dead", lambda s: logs[s]["test_dead"]),
            ("Interp.\n(|r|>0.5)", lambda s: interp[s]["summary"]["n_r>0.5"] / 100)]
    w = 0.26
    for k, (s, lab) in enumerate(SAES):
        vals = [f(s) for _, f in mets]
        ax.bar(np.arange(3) + (k - 1) * w, vals, width=w, color=SCOL[s], label=lab)
    ax.set_xticks(range(3), [m for m, _ in mets], fontsize=6.6)
    ax.set_ylabel("Fraction (interp.: count / 100)", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    panel_label(ax, "a", x=-0.35)

    # (b) 개념-feature 상관 히트맵
    bx = fig.add_subplot(gs[0, 1:])
    M = np.array([[interp[s]["concept_best"][c]["r"] if c in interp[s]["concept_best"] else np.nan
                   for s, _ in SAES] for c, _ in HEAT_ROWS])
    im = bx.imshow(np.abs(M).T, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            bx.text(i, j, f"{M[i, j]:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(M[i, j]) > 0.6 else "k")
    bx.set_yticks(range(len(SAES)), [l for _, l in SAES], fontsize=6.8)
    bx.set_xticks(range(len(HEAT_ROWS)), [l for _, l in HEAT_ROWS], rotation=35, ha="right",
                  fontsize=6.3)
    cb = fig.colorbar(im, ax=bx, fraction=0.035, pad=0.02)
    cb.set_label("|r| of best-matching feature", fontsize=6.6)
    cb.ax.tick_params(labelsize=6)
    panel_label(bx, "b", x=-0.2)

    # (c) steering 특이성 (리드 1, 16)
    cx = fig.add_subplot(gs[1, :2])
    if True:
        for ci, (c, _) in enumerate(CONCEPTS):
            for k, (s, _) in enumerate(SAES):
                for hh, dx in [(1, -0.13), (16, 0.13)]:
                    d = steer[f"{s}|{c}|{hh}"]
                    xpos = ci * 4 + k + dx
                    rc = np.array(d["random_corrs"])
                    cx.scatter(np.full(len(rc), xpos), rc, s=4, color="0.75", zorder=1)
                    sig = d["steer_corr"] > rc.max()  # 무작위 20개 모두 초과 = 경험적 p <= 1/21
                    cx.scatter([xpos], [d["steer_corr"]], s=26, facecolor=SCOL[s] if sig else "white",
                               edgecolor=SCOL[s] if not sig else "k", lw=0.9 if not sig else 0.4,
                               marker="o" if hh == 1 else "D", zorder=3)
        cx.axhline(0, color="k", lw=0.6)
        cx.set_xticks([ci * 4 + 1 for ci in range(len(CONCEPTS))], [l for _, l in CONCEPTS], fontsize=6.6)
        for ci in range(1, len(CONCEPTS)):
            cx.axvline(ci * 4 - 0.5, color="0.85", lw=0.6)
        cx.set_ylabel("Corr(steering effect,\ninput-counterfactual effect)", fontsize=7)
        cx.grid(axis="y", lw=0.4, alpha=0.4)
        panel_label(cx, "c", x=-0.1)
    # (d) 단일 feature ablation
    dx_ = fig.add_subplot(gs[1, 2])
    for k, (s, lab) in enumerate(SAES):
        for ci, (c, _) in enumerate(CONCEPTS[:2]):
            for hh, off in [(1, -0.18), (16, 0.18)]:
                d = steer[f"{s}|{c}|{hh}"]
                xpos = ci * 4 + k + off
                ra = np.array(d["random_ablation_dAUPRC"]) * 100
                dx_.scatter(np.full(len(ra), xpos), ra, s=4, color="0.75", zorder=1)
                dx_.bar(xpos, d["ablation_dAUPRC"] * 100, width=0.34, color=SCOL[s],
                        alpha=1.0 if hh == 16 else 0.55, edgecolor="k", lw=0.3, zorder=2)
    dx_.axhline(0, color="k", lw=0.6)
    dx_.set_xticks([1, 5], ["Climatology\nfeature", "Time since\nlast fire feature"], fontsize=6.5)
    dx_.set_ylabel("ΔAUPRC when the single\nfeature is zeroed (pp)", fontsize=7)
    dx_.grid(axis="y", lw=0.4, alpha=0.4)
    panel_label(dx_, "d", x=-0.45)

    # (e) KAN 반응곡선
    te = torch.load(SDIR / "step1_test.pt")
    X = te["acts"].reshape(-1, te["acts"].shape[-1])
    X = X[torch.randperm(len(X), generator=torch.Generator().manual_seed(0))[:200000]].float()
    kan = load_sae(SDIR, "kan", "cpu")
    with torch.no_grad():
        F = kan.encode(X)
    ex_axes = [fig.add_subplot(gs[2, j]) for j in range(3)]
    for ci, (c, lab) in enumerate(CONCEPTS[:3]):
        ex = ex_axes[ci]
        fi = interp["kan"]["concept_best"][c]["feature"]
        act = F[:, fi]
        hi = float(torch.quantile(act[act > 0], 0.99)) if (act > 0).any() else 1.0
        grid = torch.linspace(0, hi, 200)
        f = torch.zeros(len(grid), F.shape[1])
        f[:, fi] = grid
        with torch.no_grad():
            phi = kan.phi(f)[:, fi]
        # 디코더 방향의 부호는 임의 -> 활성 구간 기울기가 양수가 되도록 부호 정렬
        on = act[act > 0]
        lo_q = float(torch.quantile(on, 0.05))
        m = grid >= lo_q
        slope = np.polyfit(grid[m].numpy(), phi[m].numpy(), 1)[0]
        sgn = 1.0 if slope >= 0 else -1.0
        g_, p_ = grid.numpy(), sgn * phi.numpy()
        coef = np.polyfit(g_[m.numpy()], p_[m.numpy()], 1)
        fit = np.polyval(coef, g_)
        ss_res = ((p_[m.numpy()] - fit[m.numpy()]) ** 2).sum()
        ss_tot = ((p_[m.numpy()] - p_[m.numpy()].mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        ex.plot(g_, p_, color=SCOL["kan"], lw=1.8)
        ex.plot(g_[m.numpy()], fit[m.numpy()], color="0.4", ls="--", lw=0.9)
        ax2 = ex.twinx()
        ax2.hist(on.numpy(), bins=40, range=(0, hi), color="0.85", zorder=0)
        ax2.set_yticks([])
        ex.set_zorder(ax2.get_zorder() + 1)
        ex.patch.set_visible(False)
        ex.text(0.97, 0.05, f"linear fit R² = {r2:.3f}", transform=ex.transAxes, ha="right",
                fontsize=6.3, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        ex.set_xlabel(f"Feature activation (f{fi})", fontsize=6.8)
        if ci == 0:
            ex.set_ylabel("Decoder gain ±φ(a)\n(sign-aligned)", fontsize=7)
        ex.text(0.03, 0.97, lab.replace("\n", " "), transform=ex.transAxes, va="top", fontsize=6.4)
        ex.grid(lw=0.4, alpha=0.4)
        if ci == 0:
            panel_label(ex, "e", x=-0.32)

    handles = [Line2D([], [], marker="s", ls="", color=SCOL[s], label=l) for s, l in SAES]
    handles += [Line2D([], [], marker="o", ls="", color="0.3", label="lead 1 × 8 d (c) / lighter bar (d)"),
                Line2D([], [], marker="D", ls="", color="0.3", label="lead 16 × 8 d (c) / darker bar (d)"),
                Line2D([], [], marker="o", ls="", color="0.75", ms=3, label="20 random features (control)"),
                Line2D([], [], marker="o", ls="", mfc="white", mec="0.3", label="open: not above all random controls"),
                Line2D([], [], color=SCOL["kan"], lw=1.8, label="KAN learned gain (e)"),
                Line2D([], [], color="0.4", ls="--", lw=0.9, label="linear fit, active range (e)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False,
               fontsize=6.5, columnspacing=1.0)
    save(fig, "fig8_sae_steering")


if __name__ == "__main__":
    main()
