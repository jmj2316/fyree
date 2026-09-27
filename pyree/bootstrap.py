"""
시간 블록 bootstrap: test 타깃 시점(T)을 블록으로 복원추출해 AUPRC와 짝지은 차이의 95% CI.

속도를 위해 블록별 (확률 구간 x 양성/음성) 히스토그램을 미리 만들고, 재표본은 블록 가중합으로
AUPRC를 계산한다 (같은 구간 안의 동률 처리 = sklearn의 동일 threshold 처리와 같음).
구간은 비교 대상 전체 예측의 합동 분위수 20k개라, 점추정 오차는 출력에서 exact와 함께 확인.

사용:
  python -m pyree.bootstrap --a runs/v4_noanchor/test_preds.npz \
      --b runs/televit_orig_h1/test_preds.npz --leads 1
"""
import argparse
import json

import numpy as np
from sklearn.metrics import average_precision_score


def load(path, h):
    z = np.load(path)
    return (z[f"p_h{h}"].reshape(-1).astype(np.float32), z[f"y_h{h}"].reshape(-1).astype(np.uint8),
            z[f"c_h{h}"].reshape(-1).astype(np.float32), z[f"T_h{h}"], z[f"p_h{h}"].shape[1])


def block_hist(p, y, block_of_px, blocks, edges):
    nb = len(edges) + 1
    idx = np.searchsorted(edges, p, side="right")
    pos = np.zeros((len(blocks), nb))
    neg = np.zeros((len(blocks), nb))
    for k, b in enumerate(blocks):
        m = block_of_px == b
        pos[k] = np.bincount(idx[m], weights=y[m], minlength=nb)
        neg[k] = np.bincount(idx[m], weights=1 - y[m], minlength=nb)
    return pos, neg


def ap_from_counts(pos, neg):
    """pos, neg: (B, nb) 오름차순 구간 -> (B,) average precision."""
    tp = np.cumsum(pos[:, ::-1], axis=1)
    fp = np.cumsum(neg[:, ::-1], axis=1)
    denom = np.maximum(tp + fp, 1e-12)
    prec = tp / denom
    rec = tp / np.maximum(tp[:, -1:], 1e-12)
    drec = np.diff(np.concatenate([np.zeros((rec.shape[0], 1)), rec], axis=1), axis=1)
    return (drec * prec).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="주 모델 preds npz")
    ap.add_argument("--b", default=None, help="비교 모델 preds npz (선택)")
    ap.add_argument("--leads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_bins", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = {}
    for h in args.leads:
        pa, y, c, T, npx = load(args.a, h)
        preds = {"A": pa, "clim": c}
        if args.b:
            pb, yb, _, Tb, _ = load(args.b, h)
            assert np.array_equal(T, Tb) and np.array_equal(y, yb), "A/B 샘플 순서 불일치"
            preds["B"] = pb
        block_of_px = np.repeat(T, npx)
        blocks = np.unique(T)
        pooled = np.concatenate(list(preds.values()))
        edges = np.unique(np.quantile(pooled, np.linspace(0, 1, args.n_bins + 1)[1:-1]))
        hists = {k: block_hist(v, y, block_of_px, blocks, edges) for k, v in preds.items()}
        W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=args.n_boot)
        W = np.vstack([np.ones(len(blocks)), W])  # 0행 = 원표본(점추정)
        aps = {k: ap_from_counts(W @ ps, W @ ng) for k, (ps, ng) in hists.items()}
        r = {"n_blocks": int(len(blocks))}
        for k, v in preds.items():
            r[f"{k}_exact"] = float(average_precision_score(y, v))
            r[f"{k}_hist"] = float(aps[k][0])
            r[f"{k}_ci"] = [float(x) for x in np.percentile(aps[k][1:], [2.5, 97.5])]
        pairs = [("A", "clim")] + ([("A", "B"), ("B", "clim")] if args.b else [])
        for u, v in pairs:
            d = aps[u] - aps[v]
            r[f"{u}-{v}"] = float(d[0])
            r[f"{u}-{v}_ci"] = [float(x) for x in np.percentile(d[1:], [2.5, 97.5])]
            r[f"{u}-{v}_p_le0"] = float(np.mean(d[1:] <= 0))
        out[h] = r
        print(f"h={h}: " + json.dumps(r), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
