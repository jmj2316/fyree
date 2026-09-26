"""Fig.3: 정성 비교 + 오차지도. 임계값은 val(2018)에서 F1 최대가 되는 값(모델·리드별)."""
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, PowerNorm
from matplotlib.patches import Patch
from sklearn.metrics import precision_recall_curve

from common import OUT, RUNS, panel_label, save, style
from mapdata import TILE, load, preds_on_tiles

THR_CACHE = OUT / "work" / "f1_thresholds_val.json"


def f1_threshold(run, h, md):
    _, p, _, y = preds_on_tiles(run, h, "val", md)
    pr, rc, th = precision_recall_curve(y.reshape(-1), p.reshape(-1))
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    k = int(np.nanargmax(f1[:-1]))
    return float(th[k]), float(f1[k])


def thresholds(md, needs):
    thr = json.loads(THR_CACHE.read_text()) if THR_CACHE.exists() else {}
    for run, h in needs:
        key = f"{run}|{h}"
        if key not in thr:
            thr[key] = f1_threshold(run, h, md)
            print("val F1-opt threshold", key, thr[key], flush=True)
    THR_CACHE.write_text(json.dumps(thr, indent=1))
    return thr


def scene(md, run, h, T, ii, jj):
    """(T, 타일 행 ii, 타일 열 jj) 영역의 확률/기후값/정답 맵. 평가 안 된 타일은 NaN."""
    tiles, p, c, y = preds_on_tiles(run, h, "test", md)
    H, W = len(ii) * TILE, len(jj) * TILE
    P = np.full((H, W), np.nan, np.float32)
    Cm = np.full((H, W), np.nan, np.float32)
    for k, (t, i, j) in enumerate(tiles):
        if t == T and i in ii and j in jj:
            r, q = ii.index(i) * TILE, jj.index(j) * TILE
            P[r:r + TILE, q:q + TILE] = p[k]
            Cm[r:r + TILE, q:q + TILE] = c[k]
    Y = md["fire"][T, ii[0] * TILE:(ii[-1] + 1) * TILE, jj[0] * TILE:(jj[-1] + 1) * TILE]
    return P, Cm, Y.astype(float)


def pick_T(md, ii, jj, month_from, month_to):
    t_test = sorted(set(md["tiles_test"][:, 0]))
    best, bestT = -1, None
    for T in t_test:
        d = md["times"][T]
        if not (month_from <= d[:7] <= month_to):
            continue
        n = md["fire"][T, ii[0] * TILE:(ii[-1] + 1) * TILE, jj[0] * TILE:(jj[-1] + 1) * TILE].sum()
        if n > best:
            best, bestT = n, T
    return bestT


def main():
    md = load()
    style()
    lat, lon = md["lat"], md["lon"]
    # 장면: (라벨, 리드, 타일행, 타일열, 기간)
    rows = [("Southern Africa", 1, [4, 5], [9, 10], ("2019-06", "2019-10")),
            ("Southern Africa", 16, [4, 5], [9, 10], ("2019-06", "2019-10")),
            ("Eastern Australia", 1, [5, 6], [15, 16], ("2019-11", "2020-01"))]
    rows = [(lab, h, ii, jj, pick_T(md, ii, jj, *per)) for lab, h, ii, jj, per in rows]
    rows[1] = (rows[1][0], rows[1][1], rows[1][2], rows[1][3], rows[0][4])  # 같은 날짜
    needs = [("v4_noanchor", h) for _, h, *_ in rows] + [(f"televit_orig_h{h}", h) for _, h, *_ in rows]
    thr = thresholds(md, needs)

    cols = ["Observed\nburned area", "Climatology\nprior", "TeleViT$_{i,g}$\n(reproduced)",
            "Fyree", "TeleViT$_{i,g}$\nerrors", "Fyree\nerrors"]
    fig, axes = plt.subplots(len(rows), 6, figsize=(7.2, 4.35), gridspec_kw={"wspace": 0.04,
                                                                            "hspace": 0.1})
    pcmap = plt.get_cmap("YlOrRd").copy()
    pcmap.set_bad("#e6e6e6")
    ecmap = ListedColormap(["#ffffff", "#1a9641", "#2c7bb6", "#d7191c"])  # TN, TP, FP, FN
    for r_, (lab, h, ii, jj, T) in enumerate(rows):
        Po, Cm, Y = scene(md, "v4_noanchor", h, T, ii, jj)
        Pt, _, _ = scene(md, f"televit_orig_h{h}", h, T, ii, jj)
        la = lat[ii[0] * TILE:(ii[-1] + 1) * TILE]
        lo = lon[jj[0] * TILE:(jj[-1] + 1) * TILE]
        ext = [lo[0] - 0.125, lo[-1] + 0.125, la[-1] - 0.125, la[0] + 0.125]
        lsm = md["lsm"][ii[0] * TILE:(ii[-1] + 1) * TILE, jj[0] * TILE:(jj[-1] + 1) * TILE]
        evalmask = np.isnan(Po)
        panels = []
        Yshow = np.where(evalmask, np.nan, Y)
        panels.append(("prob", Yshow))
        panels.append(("prob", Cm))
        panels.append(("prob", Pt))
        panels.append(("prob", Po))
        f1s = []
        for run, P in [(f"televit_orig_h{h}", Pt), ("v4_noanchor", Po)]:
            t = thr[f"{run}|{h}"][0]
            B = P >= t
            E = np.where(B & (Y > 0), 1, np.where(B & (Y == 0), 2, np.where(~B & (Y > 0), 3, 0)))
            E = np.where(evalmask, np.nan, E).astype(float)
            ev = ~evalmask
            tp, fp, fn = ((E == 1) & ev).sum(), ((E == 2) & ev).sum(), ((E == 3) & ev).sum()
            f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
            panels.append(("err", E))
        for c_, (kind, M) in enumerate(panels):
            ax = axes[r_, c_]
            if kind == "prob":
                im = ax.imshow(M, cmap=pcmap, norm=PowerNorm(0.5, vmin=0, vmax=1), extent=ext,
                               interpolation="nearest")
            else:
                Mm = np.ma.masked_invalid(M)
                cm = ecmap.copy()
                cm.set_bad("#e6e6e6")
                ax.imshow(Mm, cmap=cm, vmin=-0.5, vmax=3.5, extent=ext, interpolation="nearest")
                ax.text(0.03, 0.04, f"F1 = {f1s[c_ - 4]:.2f}", transform=ax.transAxes, fontsize=6.2,
                        va="bottom", bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6",
                                               lw=0.4, alpha=0.9))
            ax.contour(lo, la, lsm, levels=[0.5], colors="0.25", linewidths=0.3)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.5)
            if r_ == 0:
                ax.text(0.5, 1.04, cols[c_], transform=ax.transAxes, ha="center", va="bottom",
                        fontsize=6.3)
        date = md["times"][T]
        axes[r_, 0].text(-0.08, 0.5, f"{lab}\n{date}\nlead {h} × 8 d", transform=axes[r_, 0].transAxes,
                         rotation=90, ha="right", va="center", fontsize=6.3)
        panel_label(axes[r_, 0], "abc"[r_], x=-0.42, y=0.9)

    cax = fig.add_axes([0.20, 0.035, 0.36, 0.016])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_ticks([0, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0])
    cb.set_label("Burn probability (square-root scale; observed: 0/1)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.3)
    fig.legend(handles=[Patch(color="#1a9641", label="Hit (TP)"),
                        Patch(color="#2c7bb6", label="False alarm (FP)"),
                        Patch(color="#d7191c", label="Miss (FN)")],
               loc="upper center", bbox_to_anchor=(0.78, 0.075), ncol=1, frameon=False,
               fontsize=6.3, handlelength=1.2, columnspacing=0.8)
    save(fig, "fig3_qualitative_error_maps")
    lab, _, ii, jj, T = rows[2]
    thr_all = thresholds(md, [("v4_noanchor", h) for h in (1, 4, 16)] +
                         [(f"televit_orig_h{h}", h) for h in (1, 4, 16)])
    for h in (1, 4, 16):
        f = {}
        for run in ("v4_noanchor", f"televit_orig_h{h}"):
            P, _, Y = scene(md, run, h, T, ii, jj)
            B = P >= thr_all[f"{run}|{h}"][0]
            ev = ~np.isnan(P)
            tp = (B & (Y > 0) & ev).sum(); fp = (B & (Y == 0) & ev).sum(); fn = (~B & (Y > 0) & ev).sum()
            f[run] = 2 * tp / max(2 * tp + fp + fn, 1)
        print(f"{lab} {md['times'][T]} lead {h}: F1", {k: round(v, 3) for k, v in f.items()})
    (OUT / "work" / "fig3_scenes.json").write_text(json.dumps(
        [{"label": lab, "lead": h, "tile_rows": ii, "tile_cols": jj, "date": str(md["times"][T])}
         for lab, h, ii, jj, T in rows], indent=1))


if __name__ == "__main__":
    main()
