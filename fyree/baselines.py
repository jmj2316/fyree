"""
모델 없는 기준선: TeleViT와 동일한 샘플 집합(80x80 비중첩 타일, 타깃 시점 T에 화재가 있는
타일만, xarray inclusive 연도 슬라이스)에서 아래 예측자의 픽셀 풀링 AUPRC를 계산한다.

  clim      : train 연도(2002-2017)로 계산한 연중 슬롯별 화재 발생 빈도 (타깃 슬롯 기준)
  persist   : 예보 발행 시점 T-h의 화재 여부 (0/1)
  recent_k  : T-h 이전 k스텝 동안의 화재 빈도
  lastyear  : 1년 전 같은 슬롯(T-46)의 화재 여부 (T-46 <= T-h 이므로 h<=46이면 발행 시점에 관측됨)

사용: python -m fyree.baselines --split val
"""
import os
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.metrics import average_precision_score

ZARR = os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr")
SPLITS = {
    "train": ("2002-01-01", "2018-01-01"),
    "val": ("2018-01-01", "2019-01-01"),
    "test": ("2019-01-01", "2020-01-01"),
}
STEPS_PER_YEAR = 46
TILE = 80


def load_fire(zarr_path=ZARR, target="gwis_ba"):
    ds = xr.open_zarr(zarr_path)
    times = pd.to_datetime(ds.time.values)
    fire = (ds[target].fillna(0).values > 0).astype(np.uint8)  # (T, 720, 1440)
    return fire, times


def climatology(fire, times, start="2002-01-01", end="2017-12-31"):
    idx = np.where((times >= start) & (times <= end))[0]
    slots = idx % STEPS_PER_YEAR
    clim = np.zeros((STEPS_PER_YEAR,) + fire.shape[1:], dtype=np.float32)
    for s in range(STEPS_PER_YEAR):
        clim[s] = fire[idx[slots == s]].mean(axis=0)
    return clim


def split_indices(times, split):
    a, b = SPLITS[split]
    return np.where((times >= a) & (times <= b))[0]  # xarray slice는 양끝 포함


def evaluate(fire, times, clim, split, h, k_recent=(4, 12)):
    t_idx = split_indices(times, split)
    nlat, nlon = fire.shape[1] // TILE, fire.shape[2] // TILE
    y_all, preds = [], {"clim": [], "persist": [], "lastyear": []}
    for k in k_recent:
        preds[f"recent_{k}"] = []
    n_pos = 0
    for T in t_idx:
        issue = T - h
        if issue - max(k_recent) + 1 < 0:
            continue
        clim_T = clim[T % STEPS_PER_YEAR]
        for i in range(nlat):
            for j in range(nlon):
                sl = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))
                y = fire[T][sl]
                if y.sum() == 0:
                    continue
                n_pos += 1
                y_all.append(y.ravel())
                preds["clim"].append(clim_T[sl].ravel())
                preds["persist"].append(fire[issue][sl].ravel().astype(np.float32))
                preds["lastyear"].append(fire[T - STEPS_PER_YEAR][sl].ravel().astype(np.float32))
                for k in k_recent:
                    preds[f"recent_{k}"].append(
                        fire[issue - k + 1: issue + 1][(slice(None),) + sl].mean(axis=0).ravel())
    y_all = np.concatenate(y_all)
    out = {"split": split, "h": h, "n_patches": n_pos, "pos_rate": float(y_all.mean())}
    for name, p in preds.items():
        out[name] = float(average_precision_score(y_all, np.concatenate(p)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--leads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--out", default="runs/baselines.json")
    args = ap.parse_args()

    fire, times = load_fire()
    clim = climatology(fire, times)
    rows = []
    for split in args.splits:
        for h in args.leads:
            r = evaluate(fire, times, clim, split, h)
            print(json.dumps(r))
            rows.append(r)
    Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
