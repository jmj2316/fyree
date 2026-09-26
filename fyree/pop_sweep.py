"""인구밀도 반사실 스윕 (부분의존): test 패치의 pop_dens 채널만 고정값으로 바꿔 예측 확률 변화를 본다.
재학습 없음. 육지이면서 원래 인구값이 있는 픽셀만 개입 (바다·결측 픽셀은 그대로).
바이옴 그룹별 평균 예측 확률(부분의존 곡선), 리드 1 / 16.

사용: python -m fyree.pop_sweep --run_dir runs/v4_noanchor --n_per_lead 1000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from fyree.dataset_v2 import OCI_VARS, TILE, ZARR, SeasFireStore, SeasFireV2, variables
from fyree.eval_ckpt import DEFAULTS
from fyree.model import WildfireStepsNet

POP_GRID = [0, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000]  # 명/km²
BIOME_GROUPS = {
    "Tropical savanna & grassland": [1, 2],
    "Tropical forest": [11, 12, 13],
    "Temperate grassland & montane": [6, 14],
    "Temperate forest": [7, 15],
    "Mediterranean": [3],
    "Boreal forest & tundra": [9, 10],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_per_lead", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run = Path(a.run_dir)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**(DEFAULTS | ck["args"]))
    store = SeasFireStore(prior_sigma=args.prior_sigma, prior_w=args.prior_w,
                          anomaly=bool(args.anom_windows), recency_tau=args.recency_tau)
    ds = SeasFireV2(store, "test", leads=args.leads, anom_windows=args.anom_windows,
                    long_fire=args.long_fire, tslf=args.tslf)
    names = variables(args.anom_windows, args.long_fire, args.tslf)
    model = WildfireStepsNet(variables=names, oci_vars=OCI_VARS, img_size=(80, 80), oci_lag=10,
                             patch_size=args.patch_size, hidden_size=args.hidden_size,
                             depth=args.depth, num_heads=args.num_heads, use_lead=True)
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()

    cube = xr.open_zarr(ZARR)
    biome = np.nan_to_num(cube["biomes"].values, nan=0).astype(int)
    pop_raw = cube["pop_dens"].sel(time="2019-06-01", method="nearest").values
    valid = (store.lsm > 0.5) & ~np.isnan(pop_raw)
    group = np.full(biome.shape, -1, dtype=int)
    gnames = list(BIOME_GROUPS)
    for g, codes in enumerate(BIOME_GROUPS.values()):
        group[np.isin(biome, codes) & valid] = g
    mu, sd = store.stats["pop_dens"]
    ch = names.index("pop_dens")
    grid_norm = [(np.log1p(v) - mu) / sd for v in POP_GRID]

    rng = np.random.default_rng(0)
    hs = np.array([s[3] for s in ds.samples])
    out = {"pop_grid": POP_GRID, "groups": gnames, "stats": {"mu": mu, "sd": sd}}
    with torch.no_grad():
        for h in (1, 16):
            idx = rng.choice(np.where(hs == h)[0], size=min(a.n_per_lead, int((hs == h).sum())), replace=False)
            G = len(gnames)
            psum = np.zeros((len(POP_GRID), G))
            base_sum, ysum, cnt = np.zeros(G), np.zeros(G), np.zeros(G)
            popsum_obs = [[] for _ in range(G)]
            for s in range(0, len(idx), a.batch):
                sel = idx[s:s + a.batch]
                items = [ds[int(k)] for k in sel]
                x = torch.stack([it["x_local"] for it in items]).to(a.device)
                o = torch.stack([it["x_oci"] for it in items]).to(a.device)
                ld = torch.stack([it["lead"] for it in items]).to(a.device)
                y = torch.stack([it["y_local"] for it in items]).numpy()[:, 0]
                gtile = np.stack([group[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
                                  for (_, i, j, _) in (ds.samples[int(k)] for k in sel)])
                prawt = np.stack([pop_raw[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE]
                                  for (_, i, j, _) in (ds.samples[int(k)] for k in sel)])
                vmask = torch.from_numpy(gtile >= 0).to(a.device)
                p0 = torch.sigmoid(model(x, o, lead=ld)).cpu().numpy()[:, 0]
                for g in range(G):
                    m = gtile == g
                    base_sum[g] += p0[m].sum()
                    ysum[g] += y[m].sum()
                    cnt[g] += m.sum()
                    popsum_obs[g].append(prawt[m])
                for vi, vn in enumerate(grid_norm):
                    xv = x.clone()
                    xv[:, ch] = torch.where(vmask, torch.full_like(xv[:, ch], float(vn)), xv[:, ch])
                    pv = torch.sigmoid(model(xv, o, lead=ld)).cpu().numpy()[:, 0]
                    for g in range(G):
                        psum[vi, g] += pv[gtile == g].sum()
            res = {}
            for g, gn in enumerate(gnames):
                if cnt[g] == 0:
                    continue
                po = np.concatenate(popsum_obs[g])
                res[gn] = {"n_px": int(cnt[g]), "pd_mean_prob": (psum[:, g] / cnt[g]).tolist(),
                           "baseline_mean_prob": float(base_sum[g] / cnt[g]),
                           "observed_burn_frac": float(ysum[g] / cnt[g]),
                           "pop_q01_q50_q99": [float(q) for q in np.nanpercentile(po, [1, 50, 99])]}
                pd = res[gn]["pd_mean_prob"]
                print(f"h{h} {gn:32s} n={int(cnt[g]):8d} base={res[gn]['baseline_mean_prob']:.4f} "
                      f"PD: " + " ".join(f"{v:.4f}" for v in pd) +
                      f" | pop q01/50/99 = {res[gn]['pop_q01_q50_q99']}", flush=True)
            out[str(h)] = res
    (run / "pop_sweep.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
