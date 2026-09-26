"""지도용 데이터 캐시: 화재(이진), GFED 지역, 해륙마스크, 좌표, 그리고 split별 positive 타일 순서.
SeasFireV2의 샘플 순서(시점 T -> 타일 i -> 타일 j, 리드는 안쪽 루프)와 같은 순서로 타일 목록을
만들고, preds npz의 T 배열과 일치하는지 검증해서 예측을 (T, i, j) 위치에 되돌려 놓는다."""
import os
import json

import numpy as np
import pandas as pd
import xarray as xr

from common import OUT, RUNS

ZARR = os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr")
CACHE = OUT / "work" / "mapdata.npz"
TILE = 80
SPLITS = {"val": ("2018-01-01", "2019-01-01"), "test": ("2019-01-01", "2020-01-01"),
          "test2020": ("2020-01-02", "2020-12-31")}


def build():
    ds = xr.open_zarr(ZARR)
    times = pd.DatetimeIndex(ds.time.values)
    n_t = int(np.searchsorted(times, np.datetime64("2020-12-31"), side="right"))
    fire = (ds["gwis_ba"].isel(time=slice(0, n_t)).fillna(0).values > 0).astype(np.uint8)
    region = ds["gfed_region"].values.astype(np.int16)
    lsm = ds["lsm"].values.astype(np.float32)
    out = {"fire": fire, "region": region, "lsm": lsm,
           "lat": ds.latitude.values, "lon": ds.longitude.values,
           "times": times[:n_t].values.astype("datetime64[D]").astype(str)}
    for split, (a, b) in SPLITS.items():
        tiles = []
        for T in np.where((times[:n_t] >= a) & (times[:n_t] <= b))[0]:
            for i in range(720 // TILE):
                for j in range(1440 // TILE):
                    if fire[T, i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE].any():
                        tiles.append((T, i, j))
        out[f"tiles_{split}"] = np.array(tiles, dtype=np.int32)
    np.savez_compressed(CACHE, **out)
    attrs = {k: str(v) for k, v in ds["gfed_region"].attrs.items()}
    (OUT / "work" / "gfed_region_attrs.json").write_text(json.dumps(attrs, indent=1))
    print("cached", CACHE, {k: v.shape for k, v in out.items()})
    print(attrs)


def load():
    z = np.load(CACHE, allow_pickle=False)
    return {k: z[k] for k in z.files}


def preds_on_tiles(run, h, split, md):
    """run의 split 예측(npz)을 (tiles, p(N,80,80), c, y)로. 타일 순서 검증 포함."""
    z = np.load(RUNS / run / f"{split}_preds.npz")
    T = z[f"T_h{h}"]
    tiles = md[f"tiles_{split}"]
    assert len(T) == len(tiles) and np.array_equal(T, tiles[:, 0]), f"{run} h{h}: 타일 순서 불일치"
    p = z[f"p_h{h}"].reshape(-1, TILE, TILE)
    c = z[f"c_h{h}"].reshape(-1, TILE, TILE)
    y = z[f"y_h{h}"].reshape(-1, TILE, TILE)
    return tiles, p, c, y


if __name__ == "__main__":
    build()
