"""전체(육지) 패치 평가 채점: 불 유무와 무관한 모든 육지 타일, 육지 픽셀(lsm>0.5)만 채점.
리드별 AUPRC / Brier / BSS(최근가중 기후값 기준) + 날짜 블록 bootstrap 짝지은 차이 (2019, 2020)."""
import json
import sys

import numpy as np

from common import LEADS, OUT, ROOT, RUNS
from mapdata import TILE, load

sys.path.insert(0, str(ROOT))
from pyree.bootstrap import ap_from_counts, block_hist  # noqa: E402

CACHE = OUT / "work" / "land_scores.json"
N_BOOT = 1000


def land_pixels(run, h, split, lsm):
    z = np.load(RUNS / run / f"{split}_land_preds.npz")
    T, ij = z[f"T_h{h}"], z[f"ij_h{h}"]
    p, c, y = z[f"p_h{h}"], z[f"c_h{h}"], z[f"y_h{h}"]
    masks = np.stack([(lsm[(k // 100) * TILE:(k // 100 + 1) * TILE,
                           (k % 100) * TILE:(k % 100 + 1) * TILE] > 0.5).ravel() for k in ij])
    return T, ij, p[masks], c[masks], y[masks].astype(np.float64), np.repeat(T, masks.sum(1))


def main():
    md = load()
    lsm = md["lsm"]
    rng = np.random.default_rng(0)
    out = {}
    for split in ("test", "test2020"):
        for h in LEADS:
            T, ij, po, c, y, Tpx = land_pixels("v4_noanchor", h, split, lsm)
            T2, ij2, pt, _, y2, _ = land_pixels(f"televit_orig_h{h}", h, split, lsm)
            assert np.array_equal(T, T2) and np.array_equal(ij, ij2) and np.array_equal(y, y2)
            preds = {"ours": po, "televit": pt, "clim": c}
            blocks = np.unique(Tpx)
            edges = np.unique(np.quantile(np.concatenate(list(preds.values())),
                                          np.linspace(0, 1, 20001)[1:-1]))
            hist = {k: block_hist(v, y, Tpx, blocks, edges) for k, v in preds.items()}
            W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)
            W = np.vstack([np.ones(len(blocks)), W])
            ap = {k: ap_from_counts(W @ a, W @ b) for k, (a, b) in hist.items()}
            cb = float(np.mean((c - y) ** 2))
            d = {"n_tiles": int(len(T)), "n_land_px": int(len(y)), "pos_rate": float(y.mean())}
            for k, v in preds.items():
                d[f"{k}_auprc"] = float(ap[k][0])
                d[f"{k}_brier"] = float(np.mean((v - y) ** 2))
                d[f"{k}_bss"] = 1 - d[f"{k}_brier"] / cb
            for u, v in [("ours", "televit"), ("ours", "clim"), ("televit", "clim")]:
                diff = ap[u] - ap[v]
                d[f"{u}-{v}"] = float(diff[0])
                d[f"{u}-{v}_ci"] = [float(q) for q in np.percentile(diff[1:], [2.5, 97.5])]
            out[f"{split}|{h}"] = d
            print(f"[{split} h{h}] tiles={d['n_tiles']} base={d['pos_rate']:.4f} | AUPRC ours "
                  f"{d['ours_auprc']:.3f} televit {d['televit_auprc']:.3f} clim {d['clim_auprc']:.3f} | "
                  f"BSS ours {d['ours_bss']:+.3f} televit {d['televit_bss']:+.3f} | Δ(ours−tv) "
                  f"{d['ours-televit']:+.3f} [{d['ours-televit_ci'][0]:+.3f},{d['ours-televit_ci'][1]:+.3f}]",
                  flush=True)
    CACHE.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
