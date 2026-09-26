"""Fig.5: Integrated Gradients (타깃 = 타일 내 탄 픽셀 평균 logit − 안 탄 픽셀 평균 logit).
부호 있는 IG를 픽셀 합산한 '순기여'(완전성: 합 = 판별력 f(x) − f(기준점)). 위치 채널은 고정(대상 아님).
(a) 입력 그룹별 순기여 vs 리드, (b) OCI 변수 x 지연 개월 순기여 (lead 1, 16), (c) 상위 채널."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from common import LEADS, RUNS, panel_label, save, style

z = np.load(RUNS / "v4_noanchor" / "ig_attributions_disc.npz")
names = [str(n) for n in z["names"]]
oci_vars = [str(v).replace("oci_", "") for v in z["oci_vars"]]
OCI_LABEL = {"censo": "BEST ENSO", "ea": "EA/WR", "epo": "EP/NP", "gmsst": "GMSST", "nao": "NAO",
             "nina34_anom": "Niño 3.4", "pdo": "PDO", "pna": "PNA", "soi": "SOI", "wp": "WP"}
MET = ["lst_day", "mslp", "ndvi", "pop_dens", "ssrd", "sst", "swvl1", "t2m_mean", "tp", "vpd"]
POS = ("cos_lat", "sin_lat", "cos_lon", "sin_lon")
GROUPS = [
    ("Fire-driver snapshot", lambda n: n in MET, "#5dade2"),
    ("Anomalies, 8 days", lambda n: n.startswith("anom_") and n.endswith("_w1"), "#48c9b0"),
    ("Anomalies, 1–6 months", lambda n: n.startswith("anom_") and not n.endswith("_w1"), "#117a65"),
    ("Recent burns (≤ 3 months)", lambda n: n in ("fire_issue", "fire_recent4", "fire_recent12"), "#e74c3c"),
    ("Long fire memory (1 yr, time since last fire)",
     lambda n: n in ("fire_lastyear", "fire_recent46", "fire_tslf"), "#922b21"),
    ("Climatology prior", lambda n: n in ("clim_target", "clim_issue"), "#566573"),
]
PRETTY = {"lst_day": "LST", "mslp": "MSLP", "ndvi": "NDVI", "pop_dens": "Population density",
          "ssrd": "SSRD", "sst": "SST", "swvl1": "Soil moisture", "t2m_mean": "T2m", "tp": "Precipitation",
          "vpd": "VPD", "fire_issue": "Burned at issue time", "fire_recent4": "Burn freq., 1 month",
          "fire_recent12": "Burn freq., 3 months", "fire_lastyear": "Burned, same slot 1 yr ago",
          "fire_recent46": "Burn freq., 1 year", "fire_tslf": "Time since last fire",
          "clim_target": "Climatology, target slot", "clim_issue": "Climatology, issue slot"}


def pretty(n):
    if n.startswith("anom_"):
        v, w = n[5:].rsplit("_w", 1)
        return f"{PRETTY.get(v, v)} anomaly, {int(w) * 8} d"
    return PRETTY.get(n, n)


style()
net, chan, oci_hm, total = {}, {}, {}, {}
for h in LEADS:
    s = z[f"local_sgn_h{h}"].mean(0)          # 채널별 순기여 (logit)
    o = z[f"oci_sgn_h{h}"].mean(0)            # (10 변수, 10 개월)
    net[h] = [sum(s[k] for k, n in enumerate(names) if f(n)) for _, f, _ in GROUPS] + [o.sum()]
    chan[h] = s
    oci_hm[h] = o
    total[h] = sum(net[h])

fig = plt.figure(figsize=(7.2, 7.4))
gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.95], width_ratios=[1, 1, 1.05],
                      hspace=0.75, wspace=0.35)

# (a) 그룹별 순기여 vs 리드 (범례는 오른쪽 빈 칸)
ax = fig.add_subplot(gs[0, :2])
x = np.log2(LEADS)
labels = [g for g, _, _ in GROUPS] + ["Teleconnection indices (OCIs)"]
cols = [c for _, _, c in GROUPS] + ["#f39c12"]
for k, (lab, col) in enumerate(zip(labels, cols)):
    ax.plot(x, [net[h][k] for h in LEADS], marker="o", ms=3.5, lw=1.5, color=col, label=lab)
ax.plot(x, [total[h] for h in LEADS], color="k", ls="--", lw=1.0, marker="s", ms=3,
        label="Total discrimination gain")
ax.axhline(0, color="0.5", lw=0.6)
ax.set_xticks(x, [str(h) for h in LEADS])
ax.set_xlabel("Lead time (× 8 days)")
ax.set_ylabel("Net IG contribution to\ndiscrimination (logit)")
ax.grid(axis="y", lw=0.4, alpha=0.4)
ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5), frameon=False, fontsize=6.6)
panel_label(ax, "a", x=-0.15, y=1.04)

# (b) OCI 변수 x 지연 개월
vmax = max(np.abs(oci_hm[1]).max(), np.abs(oci_hm[16]).max())
norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
for c_, h in enumerate([1, 16]):
    bx = fig.add_subplot(gs[1, c_])
    im = bx.imshow(oci_hm[h], cmap="PuOr_r", norm=norm, aspect="auto")
    bx.set_yticks(range(len(oci_vars)), [OCI_LABEL.get(v, v) for v in oci_vars] if c_ == 0 else [])
    bx.set_xticks([0, 3, 6, 9], ["−10", "−7", "−4", "−1"])
    bx.set_xlabel("Month before issue")
    bx.text(0.5, 1.03, f"lead {h} × 8 d", transform=bx.transAxes, ha="center", va="bottom",
            fontsize=7.5)
    if c_ == 0:
        panel_label(bx, "b", x=-0.36, y=1.05)
pos_b = bx.get_position()
cax = fig.add_axes([pos_b.x1 + 0.015, pos_b.y0, 0.012, pos_b.height])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Net IG (logit)", fontsize=7)
cb.ax.tick_params(labelsize=6.5)

# (c) 상위 채널 (세로 막대, 전체 폭)
cx = fig.add_subplot(gs[2, :])
cand = [k for k, n in enumerate(names) if n not in POS]
rank = sorted(cand, key=lambda k: -(abs(chan[1][k]) + abs(chan[16][k])))[:14]
xx = np.arange(len(rank))
cx.bar(xx - 0.2, [chan[1][k] for k in rank], width=0.38, color="#5d6d7e", label="lead 1 × 8 d")
cx.bar(xx + 0.2, [chan[16][k] for k in rank], width=0.38, color="#c0392b", label="lead 16 × 8 d")
cx.axhline(0, color="0.4", lw=0.6)
cx.set_xticks(xx, [pretty(names[k]) for k in rank], rotation=35, ha="right", fontsize=6.5)
cx.set_ylabel("Net IG (logit)")
cx.grid(axis="y", lw=0.4, alpha=0.4)
cx.legend(loc="upper right", frameon=False, fontsize=6.8)
panel_label(cx, "c", x=-0.07)

save(fig, "fig5_attribution")
for h in LEADS:
    print(f"h={h} total={total[h]:+.2f} | " + ", ".join(f"{l}={v:+.2f}" for l, v in zip(labels, net[h])))
