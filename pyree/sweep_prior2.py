"""prior 2차 스윕: 슬롯 윈도우 w 확장 + 최근 연도 가중(recency_tau).
A) train 연도 2002-2017 고정 prior (val/test 모두 동일 prior) — 선택은 val로.
B) 참고용 '운영형' prior: test 예보 시 이미 관측된 2018까지 포함 (2002-2018)."""
import json

from pyree.baselines import load_fire
from pyree.fire_prior import FirePrior
from pyree.sweep_prior import positive_tiles, score


def main():
    fire, times = load_fire()
    tiles = {s: positive_tiles(fire, times, s) for s in ("val", "test")}
    rows = []
    for w in [3, 4, 6]:
        for tau in [None, 16, 8, 4, 2]:
            pr = FirePrior(fire, times, sigma=0, w=w, recency_tau=tau)
            r = {"w": w, "tau": tau, "val": score(fire, pr, tiles["val"]),
                 "test": score(fire, pr, tiles["test"])}
            pr_op = FirePrior(fire, times, sigma=0, w=w, recency_tau=tau, train_end="2018-12-31")
            r["test_operational_2002_2018"] = score(fire, pr_op, tiles["test"])
            print(json.dumps(r), flush=True)
            rows.append(r)
    best = max(rows, key=lambda r: r["val"])
    print("BEST_BY_VAL", json.dumps(best))


if __name__ == "__main__":
    main()
