"""
v2 데이터셋: 기후값-앵커 + fire memory + 멀티-리드.

샘플 집합은 TeleViT와 동일 (80x80 비중첩 타일, 타깃 시점 T에 화재가 있는 타일만,
xarray inclusive 연도 슬라이스) — baselines.py에서 val 2,426개로 일치 검증됨.
한 샘플 = (T, 타일 i, j, 리드 h). 모든 입력은 발행 시점 issue = T - h 기준으로만 구성.

입력 채널 (x_local, 20ch):
  기상/지표 10ch (issue 시점, train 통계로 정규화, NaN -> -1) + 위치 4ch
  + fire memory 6ch:
    fire_issue    : issue 시점 화재 여부
    fire_recent4  : issue 포함 직전 4스텝 화재 빈도
    fire_recent12 : issue 포함 직전 12스텝 화재 빈도
    fire_lastyear : 1년 전 타깃 슬롯(T-46) 화재 여부 (h<=46이면 issue 시점에 관측됨)
    clim_target   : 타깃 슬롯 기후값 (train 샘플은 타깃 연도 제외 LOO)
    clim_issue    : issue 슬롯 기후값 (train 샘플은 issue 연도 제외 LOO)
prior_logit: logit(clim_target) — 모델 출력에 더해지는 slow-clock 앵커.
x_oci: issue 날짜 이전에 완결된 최근 10개월의 월평균 원격상관 지수 (10변수 x 10개월).
"""
import os
import warnings

import numpy as np
import pandas as pd
import torch
import xarray as xr
from torch.utils.data import Dataset

from pyree.fire_prior import STEPS_PER_YEAR, FirePrior

ZARR = os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr")
INPUT_VARS = ["lst_day", "mslp", "ndvi", "pop_dens", "ssrd", "sst", "swvl1",
              "t2m_mean", "tp", "vpd"]
LOG_VARS = {"tp", "pop_dens"}
POS_VARS = ["cos_lat", "sin_lat", "cos_lon", "sin_lon"]
MEMORY_VARS = ["fire_issue", "fire_recent4", "fire_recent12", "fire_lastyear",
               "clim_target", "clim_issue"]
OCI_VARS = ["oci_censo", "oci_ea", "oci_epo", "oci_gmsst", "oci_nao",
            "oci_nina34_anom", "oci_pdo", "oci_pna", "oci_soi", "oci_wp"]
# v3: 픽셀·슬롯별 기후값을 뺀 anomaly를 누적 윈도우로 제공할 변수 (가뭄·연료·열 스트레스)
ANOM_VARS = ["tp", "swvl1", "ndvi", "vpd", "t2m_mean", "lst_day"]
SPLITS = {
    "train": ("2002-01-01", "2018-01-01"),
    "val": ("2018-01-01", "2019-01-01"),
    "test": ("2019-01-01", "2020-01-01"),
    # 두 번째 held-out 연도 (모델은 2002-2017 학습, 2018 선택 — 2020은 완전 미사용).
    # 2020-01-01은 TeleViT 관례상 test에 포함되므로 제외. GWIS 2020-12-18/26은 무효(NaN→화재 없음→타깃 제외).
    "test2020": ("2020-01-02", "2020-12-31"),
}
TILE = 80
OCI_LAG = 10
PRIOR_EPS = 1e-4


class SeasFireStore:
    """전 split이 공유하는 in-memory 데이터 (프로세스당 한 번 로드, fork worker와 CoW 공유)."""

    def __init__(self, zarr_path=ZARR, target="gwis_ba", t_end="2020-01-01",
                 prior_sigma=0.0, prior_w=0, anomaly=False, recency_tau=None,
                 with_global=False, global_factor=4, verbose=True):
        ds = xr.open_zarr(zarr_path)
        times = pd.DatetimeIndex(ds.time.values)
        self.n_t = int(np.searchsorted(times, np.datetime64(t_end), side="right"))
        self.times = times[: self.n_t]
        tr_a, tr_b = SPLITS["train"]
        tr_mask = (self.times >= tr_a) & (self.times <= tr_b)

        self.inputs = np.empty((len(INPUT_VARS), self.n_t, 720, 1440), dtype=np.float16)
        gf = global_factor
        self.gcoarse = (np.empty((len(INPUT_VARS), self.n_t, 720 // gf, 1440 // gf), dtype=np.float16)
                        if with_global else None)
        self.stats = {}
        for k, v in enumerate(INPUT_VARS):
            a = ds[v].isel(time=slice(0, self.n_t)).values.astype(np.float32)
            if v in LOG_VARS:
                a = np.log1p(a)
            mu, sd = float(np.nanmean(a[tr_mask])), float(np.nanstd(a[tr_mask]))
            self.stats[v] = (mu, sd)
            a = (a - mu) / sd
            if with_global:
                # TeleViT의 1도 coarsened 큐브처럼 결측 무시 평균 후 -1 채움
                blk = a.reshape(self.n_t, 720 // gf, gf, 1440 // gf, gf)
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    self.gcoarse[k] = np.nan_to_num(np.nanmean(blk, axis=(2, 4)), nan=-1.0)
            self.inputs[k] = np.nan_to_num(a, nan=-1.0)
            del a
            if verbose:
                print(f"  loaded {v} mean={mu:.4g} std={sd:.4g}", flush=True)

        lat, lon = ds.latitude.values, ds.longitude.values
        lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
        self.pos = np.stack([np.cos(np.deg2rad(lat2)), np.sin(np.deg2rad(lat2)),
                             np.cos(np.deg2rad(lon2)), np.sin(np.deg2rad(lon2))]).astype(np.float32)
        self.gpos = (self.pos.reshape(4, 720 // gf, gf, 1440 // gf, gf).mean(axis=(2, 4))
                     if with_global else None)

        self.fire = (ds[target].isel(time=slice(0, self.n_t)).fillna(0).values > 0).astype(np.uint8)
        self.lsm = ds["lsm"].values.astype(np.float32)
        vm = f"{target}_valid_mask"
        self.fire_valid = (ds[vm].isel(time=slice(0, self.n_t)).values > 0) if vm in ds else \
            np.ones(self.n_t, dtype=bool)
        self.prior = FirePrior(self.fire, self.times, sigma=prior_sigma, w=prior_w,
                               recency_tau=recency_tau)
        self.oci = self._build_oci(ds, times, tr_b)
        self.varclim = self._build_varclim() if anomaly else None
        # 마지막 화재 이후 경과 스텝 (연료 회복 시간), 255로 캡. 관측 시작(2001) 이전은 모름 -> 캡 취급
        self.tslf = np.empty(self.fire.shape, dtype=np.uint8)
        prev = np.full(self.fire.shape[1:], 255, dtype=np.uint16)
        for t in range(self.n_t):
            prev = np.where(self.fire[t] > 0, 0, np.minimum(prev + 1, 255))
            self.tslf[t] = prev
        if verbose:
            print("  store ready", flush=True)

    def _build_varclim(self):
        """ANOM_VARS의 픽셀·슬롯별 기후값 (정규화 단위, train 연도 2002-2017만)."""
        tr = np.where((self.times >= "2002-01-01") & (self.times <= "2017-12-31"))[0]
        slots = tr % STEPS_PER_YEAR
        out = np.empty((len(ANOM_VARS), STEPS_PER_YEAR, 720, 1440), dtype=np.float16)
        for a, v in enumerate(ANOM_VARS):
            k = INPUT_VARS.index(v)
            for s in range(STEPS_PER_YEAR):
                out[a, s] = self.inputs[k, tr[slots == s]].astype(np.float32).mean(axis=0)
        return out

    def _build_oci(self, ds, all_times, train_end):
        series = {}
        for v in OCI_VARS:
            s = pd.Series(ds[v].values, index=all_times)
            if v == "oci_pdo":
                s = s.where(s > -9)
            if v == "oci_epo":
                s = s.where(s > -90)
            series[v] = s.ffill().fillna(0.0)
        monthly = pd.DataFrame(series).resample("MS").mean()  # 월 시작 라벨
        sd = monthly[monthly.index <= train_end].std()
        monthly = monthly / sd
        out = np.zeros((self.n_t, len(OCI_VARS), OCI_LAG), dtype=np.float32)
        month_starts = monthly.index
        vals = monthly.values  # (n_months, n_vars)
        for t in range(self.n_t):
            d = self.times[t]
            cur_month = pd.Timestamp(d.year, d.month, 1)
            # issue 날짜가 속한 달은 미완결이므로 제외, 그 이전 완결된 OCI_LAG개월
            end = int(np.searchsorted(month_starts, cur_month, side="left"))
            start = max(0, end - OCI_LAG)
            chunk = vals[start:end].T  # (n_vars, <=OCI_LAG)
            if chunk.shape[1] > 0:
                out[t, :, OCI_LAG - chunk.shape[1]:] = chunk
        return out

    def split_t(self, split):
        a, b = SPLITS[split]
        return np.where((self.times >= a) & (self.times <= b))[0]

    def land_tiles(self, split):
        """전체 패치 평가용: 육지 픽셀이 하나라도 있는 모든 타일 (불 유무 무관).
        타깃 GWIS가 무효인 날짜는 제외 (NaN->0이 '불 없음'으로 잘못 채점되지 않게)."""
        tiles = []
        land = [[bool((self.lsm[i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE] > 0.5).any())
                 for j in range(1440 // TILE)] for i in range(720 // TILE)]
        for T in self.split_t(split):
            if not self.fire_valid[T]:
                continue
            for i in range(720 // TILE):
                for j in range(1440 // TILE):
                    if land[i][j]:
                        tiles.append((int(T), i, j))
        return tiles

    def positive_tiles(self, split):
        tiles = []
        for T in self.split_t(split):
            for i in range(720 // TILE):
                for j in range(1440 // TILE):
                    if self.fire[T, i * TILE:(i + 1) * TILE, j * TILE:(j + 1) * TILE].any():
                        tiles.append((int(T), i, j))
        return tiles


class SeasFireV2(Dataset):
    def __init__(self, store, split, leads=(1, 2, 4, 8, 16), max_lead=16,
                 anom_windows=(), long_fire=False, tslf=False, with_global=False,
                 tile_mode="positive"):
        self.store, self.split = store, split
        self.loo = split == "train"
        self.with_global = with_global
        if with_global:
            assert store.gcoarse is not None, "with_global은 SeasFireStore(with_global=True)가 필요"
        self.anom_windows = tuple(anom_windows)
        self.long_fire = long_fire
        self.use_tslf = tslf
        if self.anom_windows:
            assert store.varclim is not None, "anom_windows는 SeasFireStore(anomaly=True)가 필요"
        need = max([12] + list(self.anom_windows) + ([STEPS_PER_YEAR] if long_fire else []))
        tiles = store.land_tiles(split) if tile_mode == "land" else store.positive_tiles(split)
        self.samples = [(T, i, j, h) for (T, i, j) in tiles for h in leads if T - h - need + 1 >= 0]
        self.max_lead = max_lead

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.store
        T, i, j, h = self.samples[idx]
        issue = T - h
        sl = (slice(i * TILE, (i + 1) * TILE), slice(j * TILE, (j + 1) * TILE))

        met = s.inputs[:, issue][(slice(None),) + sl].astype(np.float32)
        pos = s.pos[(slice(None),) + sl]
        fire = s.fire
        ex_T = s.prior.year_of(T) if self.loo else None
        ex_I = s.prior.year_of(issue) if self.loo else None
        clim_T = s.prior.clim(T, exclude_year=ex_T)[sl]
        clim_I = s.prior.clim(issue, exclude_year=ex_I)[sl]
        mem = np.stack([
            fire[issue][sl].astype(np.float32),
            fire[issue - 3: issue + 1][(slice(None),) + sl].mean(axis=0),
            fire[issue - 11: issue + 1][(slice(None),) + sl].mean(axis=0),
            fire[T - STEPS_PER_YEAR][sl].astype(np.float32),
            clim_T,
            clim_I,
        ]).astype(np.float32)
        parts = [met, pos, mem]
        if self.long_fire:
            parts.append(fire[issue - STEPS_PER_YEAR + 1: issue + 1][(slice(None),) + sl]
                         .mean(axis=0, dtype=np.float32)[None])
        if self.use_tslf:
            parts.append((np.log1p(s.tslf[issue][sl].astype(np.float32)) / np.log(256.0))[None])
        if self.anom_windows:
            wmax = max(self.anom_windows)
            t0 = issue - wmax + 1
            slots = np.arange(t0, issue + 1) % STEPS_PER_YEAR
            anoms = []
            for a, v in enumerate(ANOM_VARS):
                k = INPUT_VARS.index(v)
                raw = s.inputs[k, t0: issue + 1][(slice(None),) + sl].astype(np.float32)
                clim = s.varclim[a][(slice(None),) + sl][slots].astype(np.float32)
                d = raw - clim  # (wmax, 80, 80), 마지막이 issue 시점
                for w in self.anom_windows:
                    anoms.append(d[-w:].mean(axis=0))
            parts.append(np.stack(anoms).astype(np.float32))
        x_local = np.concatenate(parts, axis=0)
        p = np.clip(clim_T, PRIOR_EPS, 1 - PRIOR_EPS)
        prior_logit = np.log(p / (1 - p)).astype(np.float32)[None]

        item = {}
        if self.with_global:
            item["x_global"] = torch.from_numpy(
                np.concatenate([s.gcoarse[:, issue].astype(np.float32), s.gpos]).astype(np.float32))
        return item | {
            "x_local": torch.from_numpy(x_local),
            "x_oci": torch.from_numpy(s.oci[issue]),
            "lead": torch.tensor([h / self.max_lead], dtype=torch.float32),
            "prior_logit": torch.from_numpy(prior_logit),
            "clim": torch.from_numpy(clim_T.astype(np.float32)[None]),
            "y_local": torch.from_numpy(fire[T][sl].astype(np.float32)[None]),
            "h": h,
            "T": T,
            "ij": i * 100 + j,
        }


def variables(anom_windows=(), long_fire=False, tslf=False):
    names = INPUT_VARS + POS_VARS + MEMORY_VARS
    if long_fire:
        names = names + ["fire_recent46"]
    if tslf:
        names = names + ["fire_tslf"]
    names = names + [f"anom_{v}_w{w}" for v in ANOM_VARS for w in anom_windows]
    return names
