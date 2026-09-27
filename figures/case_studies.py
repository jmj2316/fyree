"""실제 산불 사건 케이스 스터디: 우리 모델 vs TeleViT 재현 vs 기후값.
사건 = (위경도 상자, 기간). 평가 픽셀 = 기간 내 타깃 시점의 평가 타일(양성 타일) 안에서 상자에 속한 픽셀.
리드별 AUPRC, 기후값 대비 BSS, 짝지은 블록 bootstrap(블록=타깃 날짜) ΔAUPRC,
그리고 날짜별 관측 화재 픽셀 수 vs 예상 화재 픽셀 수(확률 합) 시계열."""
import json
import sys

import numpy as np

from common import LEADS, OUT, ROOT, RUNS
from mapdata import TILE, load

sys.path.insert(0, str(ROOT))
from pyree.bootstrap import ap_from_counts, block_hist  # noqa: E402

EVENTS = [
    # 이름, (위도 최소, 최대, 경도 최소, 최대), (시작, 끝)
    ("Amazon 2019", (-13, -2, -68, -48), ("2019-07-15", "2019-10-15")),
    ("Siberia 2019", (52, 68, 88, 135), ("2019-06-01", "2019-09-15")),
    ("Alaska 2019", (58, 70, -165, -140), ("2019-06-01", "2019-08-31")),
    ("Indonesia 2019", (-6, 3, 98, 118), ("2019-08-01", "2019-10-31")),
    ("SE Australia 2019–20", (-39, -25, 140, 154), ("2019-09-01", "2020-02-28")),
    ("Siberian Arctic 2020", (60, 73, 110, 160), ("2020-05-15", "2020-09-15")),
    ("US West Coast 2020", (32, 49, -125, -114), ("2020-07-15", "2020-10-31")),
    ("Pantanal 2020", (-22, -15, -60, -54), ("2020-07-01", "2020-10-31")),
]
MODELS = {"ours": lambda h: "v4_noanchor", "televit": lambda h: f"televit_orig_h{h}"}
N_BOOT = 1000
CACHE = OUT / "work" / "case_studies.json"


def load_split_preds(run, h, split, md):
    f = RUNS / run / f"{split}_preds.npz"
    z = np.load(f)
    T = z[f"T_h{h}"]
    tiles = md[f"tiles_{split}"]
    assert np.array_equal(T, tiles[:, 0]), f"{run} {split} h{h} 순서 불일치"
    return tiles, z[f"p_h{h}"], z[f"c_h{h}"], z[f"y_h{h}"]


def event_pixels(md, box, window, h):
    """사건의 (T 블록, y, 모델별 p, clim) 픽셀 벡터."""
    la0, la1, lo0, lo1 = box
    lat, lon = md["lat"], md["lon"]
    out = {"T": [], "y": [], "clim": [], "ours": [], "televit": []}
    for split in ("test", "test2020"):
        tiles, po, c, y = load_split_preds(MODELS["ours"](h), h, split, md)
        _, pt, _, yt = load_split_preds(MODELS["televit"](h), h, split, md)
        assert np.array_equal(y, yt)
        for k, (T, i, j) in enumerate(tiles):
            d = md["times"][T]
            if not (window[0] <= d <= window[1]):
                continue
            la = lat[i * TILE:(i + 1) * TILE]
            lo = lon[j * TILE:(j + 1) * TILE]
            m = ((la >= la0) & (la <= la1))[:, None] & ((lo >= lo0) & (lo <= lo1))[None, :]
            if not m.any():
                continue
            m = m.ravel()
            out["T"].append(np.full(m.sum(), T))
            out["y"].append(y[k][m])
            out["clim"].append(c[k][m])
            out["ours"].append(po[k][m])
            out["televit"].append(pt[k][m])
    return {k: (np.concatenate(v) if v else np.array([])) for k, v in out.items()}


def analyse(md):
    rng = np.random.default_rng(0)
    res = {}
    for name, box, window in EVENTS:
        for h in LEADS:
            ev = event_pixels(md, box, window, h)
            if len(ev["y"]) == 0 or ev["y"].sum() < 20:
                continue
            y = ev["y"].astype(np.float64)
            blocks = np.unique(ev["T"])
            preds = {k: ev[k].astype(np.float32) for k in ("ours", "televit", "clim")}
            edges = np.unique(np.quantile(np.concatenate(list(preds.values())),
                                          np.linspace(0, 1, 5001)[1:-1]))
            hist = {k: block_hist(v, y, ev["T"], blocks, edges) for k, v in preds.items()}
            W = rng.multinomial(len(blocks), np.full(len(blocks), 1 / len(blocks)), size=N_BOOT)
            W = np.vstack([np.ones(len(blocks)), W])
            ap = {k: ap_from_counts(W @ a, W @ b) for k, (a, b) in hist.items()}
            clim_brier = float(np.mean((preds["clim"] - y) ** 2))
            d = {"n_dates": int(len(blocks)), "n_px": int(len(y)), "n_burned": int(y.sum())}
            for k in preds:
                d[f"{k}_auprc"] = float(ap[k][0])
                d[f"{k}_bss"] = 1 - float(np.mean((preds[k] - y) ** 2)) / clim_brier
            for u, v in [("ours", "televit"), ("ours", "clim"), ("televit", "clim")]:
                diff = ap[u] - ap[v]
                d[f"{u}-{v}"] = float(diff[0])
                d[f"{u}-{v}_ci"] = [float(q) for q in np.percentile(diff[1:], [2.5, 97.5])]
            # 날짜별 관측 vs 예상 화재 픽셀 수
            ts = {"dates": [str(md["times"][T]) for T in blocks], "obs": [], "ours": [],
                  "televit": [], "clim": []}
            for T in blocks:
                m = ev["T"] == T
                ts["obs"].append(float(y[m].sum()))
                for k in ("ours", "televit", "clim"):
                    ts[k].append(float(preds[k][m].sum()))
            d["timeseries"] = ts
            res[f"{name}|{h}"] = d
            print(f"{name} h{h}: n_dates={d['n_dates']} burned={d['n_burned']} | AUPRC ours "
                  f"{d['ours_auprc']:.3f} televit {d['televit_auprc']:.3f} clim {d['clim_auprc']:.3f} | "
                  f"Δ(ours−tv) {d['ours-televit']:+.3f} [{d['ours-televit_ci'][0]:+.3f},"
                  f"{d['ours-televit_ci'][1]:+.3f}]", flush=True)
    CACHE.write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    analyse(load())
