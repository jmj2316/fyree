"""Fig.10: 인구밀도 반사실 스윕 (부분의존) vs 관측된 단순 관계, 바이옴 그룹별."""
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from common import COLORS, OUT, RUNS, panel_label, save, style
from mapdata import load

ZARR = os.environ.get("SEASFIRE_ZARR", "data/SeasFireCube_v3.zarr")
GROUPS = {"Tropical savanna & grassland": [1, 2], "Tropical forest": [11, 12, 13],
          "Temperate grassland & montane": [6, 14], "Temperate forest": [7, 15],
          "Mediterranean": [3], "Boreal forest & tundra": [9, 10]}


def observed_relation(pop_grid):
    """2002-2017 픽셀별 화재 빈도를 인구 구간별로 평균 (바이옴 그룹 내, 교란 포함 단순 관계)."""
    md = load()
    cube = xr.open_zarr(ZARR)
    biome = np.nan_to_num(cube["biomes"].values, nan=0).astype(int)
    pop = cube["pop_dens"].sel(time="2010-06-01", method="nearest").values
    valid = (md["lsm"] > 0.5) & ~np.isnan(pop)
    tr = [k for k, d in enumerate(md["times"]) if "2002-01-01" <= d <= "2017-12-31"]
    freq = np.zeros(md["fire"].shape[1:], dtype=np.float64)
    for k in tr:
        freq += md["fire"][k]
    freq /= len(tr)
    g = np.array(pop_grid, dtype=float)
    mids = np.sqrt(np.maximum(g[:-1], 0.1) * g[1:])
    edges = np.concatenate([[-1], mids, [np.inf]])
    out = {}
    for name, codes in GROUPS.items():
        m = valid & np.isin(biome, codes)
        f, p = freq[m], pop[m]
        rel = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (p > lo) & (p <= hi)
            rel.append(f[sel].mean() / f.mean() - 1 if sel.sum() > 200 else np.nan)
        out[name] = np.array(rel) * 100
    return out


def main():
    d = json.loads((RUNS / "v4_noanchor" / "pop_sweep.json").read_text())
    grid = np.array(d["pop_grid"], dtype=float)
    obs = observed_relation(grid)
    style()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.9), gridspec_kw={"hspace": 0.62, "wspace": 0.55})
    xs = np.where(grid == 0, 0.1, grid)
    for k, (name, ax) in enumerate(zip(GROUPS, axes.ravel())):
        q01, _, q99 = d["1"][name]["pop_q01_q50_q99"]
        for h, ls in (("1", "-"), ("16", "--")):
            r = d[h][name]
            rel = (np.array(r["pd_mean_prob"]) / r["baseline_mean_prob"] - 1) * 100
            ax.plot(xs, rel, color=COLORS["ours"], ls=ls, marker="o", ms=2.8, lw=1.5)
        ax.axhline(0, color="0.4", lw=0.6)
        ax.axvspan(max(q99, 0.1), 5000, color="0.9", zorder=0)
        if q01 > 0.1:
            ax.axvspan(0.07, q01, color="0.9", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.07, 5000)
        ax.set_xticks([0.1, 1, 10, 100, 1000], ["0", "1", "10", "100", "1000"])
        ax.tick_params(labelsize=6.5)
        ax2 = ax.twinx()
        ax2.plot(xs, obs[name], color="0.45", ls=":", marker="s", ms=2.2, lw=1.1)
        ax2.tick_params(labelsize=6, colors="0.4")
        ax2.spines["right"].set_visible(True)
        ax2.spines["right"].set_color("0.6")
        ax.text(0.5, 1.03, name, transform=ax.transAxes, fontsize=7, ha="center", va="bottom")
        if k % 3 == 0:
            ax.set_ylabel("Model: change in mean\nburn probability (%)", fontsize=7, color=COLORS["ours"])
        if k % 3 == 2:
            ax2.set_ylabel("Observed 2002–2017: burn\nfrequency vs. biome mean (%)", fontsize=6.6, color="0.4")
        if k >= 3:
            ax.set_xlabel("Population density (persons km$^{-2}$)", fontsize=7)
        panel_label(ax, "abcdef"[k], x=-0.35, y=1.03)
    handles = [Line2D([], [], color=COLORS["ours"], lw=1.5, ls="-",
                      label="Model partial dependence, lead 1 × 8 d"),
               Line2D([], [], color=COLORS["ours"], lw=1.5, ls="--",
                      label="Model partial dependence, lead 16 × 8 d"),
               Line2D([], [], color="0.45", ls=":", marker="s", ms=3,
                      label="Observed burn frequency by population bin (right axis)"),
               Patch(color="0.9", label="Outside the biome's 1–99th percentile of population")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=2, frameon=False,
               fontsize=6.8, handlelength=3.5)
    save(fig, "fig10_population_counterfactual")


if __name__ == "__main__":
    main()
