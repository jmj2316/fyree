"""
화재 기후값 prior (Dual-Clock의 slow clock 역할).

clim[s] = train 연도들의 "연중 슬롯 s 주변(±w 슬롯) + 공간 가우시안(sigma 픽셀)"
평활 화재 빈도. train 샘플용으로는 해당 연도를 뺀 leave-one-year-out 버전을 제공해서,
학습 중 기후값 feature에 타깃 자신이 섞이는 in-sample 낙관을 막는다.
"""
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

STEPS_PER_YEAR = 46


class FirePrior:
    def __init__(self, fire, times, sigma=0.0, w=0, train_start="2002-01-01",
                 train_end="2017-12-31", recency_tau=None):
        """fire: (T, H, W) uint8 이진 화재, times: DatetimeIndex (46스텝/년, 1/1 시작 정렬).
        recency_tau(년): 주어지면 연도 가중 w_y = exp(-(마지막 train 연도 - y)/tau) —
        장기 화재 감소 추세(Andela+ 2017) 때문에 균등 기후값이 최근 연도를 과대평가하는 문제 대응."""
        self.sigma, self.w = sigma, w
        self.times = pd.DatetimeIndex(times)
        self.year0 = self.times[0].year
        tr = np.where((self.times >= train_start) & (self.times <= train_end))[0]
        self.train_years = sorted(set(self.times[tr].year))
        n_t = fire.shape[0]

        smooth = np.empty(fire.shape, dtype=np.float32)
        for t in range(n_t):
            f = fire[t].astype(np.float32)
            smooth[t] = gaussian_filter(f, sigma, mode=("nearest", "wrap")) if sigma > 0 else f

        # 연도 y, 슬롯 s의 윈도우 평균 W_y[s] (윈도우는 연도 경계를 넘어 전역 인덱스로)
        self.per_year = {}
        for y in self.train_years:
            arr = np.zeros((STEPS_PER_YEAR,) + fire.shape[1:], dtype=np.float32)
            base = (y - self.year0) * STEPS_PER_YEAR
            for s in range(STEPS_PER_YEAR):
                lo, hi = max(0, base + s - w), min(n_t, base + s + w + 1)
                arr[s] = smooth[lo:hi].mean(axis=0)
            self.per_year[y] = arr
        last = max(self.train_years)
        self.wy = {y: (1.0 if recency_tau is None else float(np.exp(-(last - y) / recency_tau)))
                   for y in self.train_years}
        self.total = np.sum([self.wy[y] * self.per_year[y] for y in self.train_years], axis=0)
        self.wsum = sum(self.wy.values())
        self.full = self.total / self.wsum  # (46, H, W)

    def clim(self, t_idx, exclude_year=None):
        """전역 시간 인덱스 t_idx의 슬롯에 대한 기후값 맵. exclude_year가 train 연도면 LOO."""
        s = t_idx % STEPS_PER_YEAR
        if exclude_year is not None and exclude_year in self.per_year:
            wy = self.wy[exclude_year]
            return (self.total[s] - wy * self.per_year[exclude_year][s]) / (self.wsum - wy)
        return self.full[s]

    def year_of(self, t_idx):
        return self.year0 + t_idx // STEPS_PER_YEAR
