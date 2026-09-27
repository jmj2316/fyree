"""기후값 prior의 평활 하이퍼파라미터(sigma, w)를 val AUPRC로 고르고 test에 보고."""
import itertools
import json

import numpy as np
from sklearn.metrics import average_precision_score

from pyree.baselines import TILE, load_fire, split_indices
from pyree.fire_prior import FirePrior


def positive_tiles(fire, times, split):
    tiles = []
    nlat, nlon = fire.shape[1] // TILE, fire.shape[2] // TILE
    for T in split_indices(times, split):
        for i in range(nlat):
            for j in range(nlon):
                if fire[T, i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE].any():
                    tiles.append((T, i, j))
    return tiles


def score(fire, prior, tiles):
    ys, ps = [], []
    for T, i, j in tiles:
        sl = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))
        ys.append(fire[T][sl].ravel())
        ps.append(prior.clim(T)[sl].ravel())
    return float(average_precision_score(np.concatenate(ys), np.concatenate(ps)))


def main():
    fire, times = load_fire()
    tiles = {s: positive_tiles(fire, times, s) for s in ("val", "test")}
    rows = []
    for sigma, w in itertools.product([0, 1, 2, 3, 5], [0, 1, 2, 3]):
        prior = FirePrior(fire, times, sigma=sigma, w=w)
        r = {"sigma": sigma, "w": w,
             "val": score(fire, prior, tiles["val"]),
             "test": score(fire, prior, tiles["test"])}
        print(json.dumps(r), flush=True)
        rows.append(r)
    best = max(rows, key=lambda r: r["val"])
    print("BEST_BY_VAL", json.dumps(best))


if __name__ == "__main__":
    main()
