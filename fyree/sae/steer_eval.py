"""SAE feature steering 검증 — '입력 반사실 vs feature steering' 일치도 + feature ablation.

개념(slow path에만 들어가는 입력): clim_target(x1.5), fire_tslf(+0.15, 로그스케일=경과시간 약 2배),
anom_tp_w24(−1σ, 6개월 더 건조), anom_vpd_w4(+1σ, 1개월 VPD 증가).
입력 개입 효과 ΔL_in 은 전부 Step-1(y1)을 통해서만 출력에 간다(fast path 불변).
개념 feature c 의 토큰별 활성 변화 δ_t = f_c(y1') − f_c(y1) 만큼 c만 steering 한 효과 ΔL_st 와 비교:
  corr = 픽셀 풀링 Pearson(ΔL_st, ΔL_in), frac = <ΔL_st, ΔL_in> / ||ΔL_in||²
대조군: 비슷한 빈도의 무작위 살아있는 feature 5개에 같은 δ_t 로 steering.
ablation: 개념 feature를 모든 토큰에서 0으로 → ΔAUPRC (리드 1, 16), 무작위 feature ablation 대조.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from fyree.eval_ckpt import DEFAULTS
from fyree.model import WildfireStepsNet
from fyree.sae.interpret import load_sae
from fyree.sae.tools import finish_forward, step1_forward, steer

CONCEPTS = {  # (연산, 값, 0-1 범위로 자를지)
    "clim_target": ("mul", 1.5, True),
    "fire_tslf": ("add", 0.15, True),
    "anom_tp_w24": ("add", -1.0, False),
    "anom_vpd_w4": ("add", 1.0, False),
}


def intervene(x, ch, how, v, clip01):
    x2 = x.clone()
    x2[:, ch] = x2[:, ch] * v if how == "mul" else x2[:, ch] + v
    if clip01:
        x2[:, ch] = x2[:, ch].clamp(0.0, 1.0)
    return x2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/v4_noanchor")
    ap.add_argument("--saes", nargs="+", default=["jumprelu_l01.0", "kan", "topk"])
    ap.add_argument("--n_per_lead", type=int, default=400)
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run = Path(a.run_dir)
    sdir = run / "sae"
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
    for c in CONCEPTS:
        assert c in model.group1_vars, f"{c} 는 slow path 입력이어야 함"
    interp = json.loads((sdir / "interpret.json").read_text())
    te = torch.load(sdir / "step1_test.pt")
    idx_all = te["sample_idx"].numpy()
    rng = np.random.default_rng(0)
    samples = {1: rng.choice(idx_all[:len(idx_all) // 2], a.n_per_lead, replace=False),
               16: rng.choice(idx_all[len(idx_all) // 2:], a.n_per_lead, replace=False)}
    saes = {n: load_sae(sdir, n, a.device) for n in a.saes}

    results = {}
    with torch.no_grad():
        for h, idx in samples.items():
            # 배치별로 기준 y1과 개입 후 y1', 기준 로짓 저장
            batches = []
            for s in range(0, len(idx), a.batch):
                items = [ds[int(k)] for k in idx[s:s + a.batch]]
                x = torch.stack([it["x_local"] for it in items]).to(a.device)
                o = torch.stack([it["x_oci"] for it in items]).to(a.device)
                ld = torch.stack([it["lead"] for it in items]).to(a.device)
                y = torch.stack([it["y_local"] for it in items])
                y1, x2, c, hw = step1_forward(model, x, o, ld)
                L0 = finish_forward(model, y1, x2, c, hw)
                batches.append({"x": x, "o": o, "ld": ld, "y": y, "y1": y1, "x2": x2, "c": c,
                                "hw": hw, "L0": L0})
            yy = torch.cat([b["y"] for b in batches]).numpy().ravel()
            base_ap = average_precision_score(yy, torch.cat(
                [torch.sigmoid(b["L0"]).cpu() for b in batches]).numpy().ravel())
            for sname, sae in saes.items():
                feats = interp[sname]["features"]
                freq = {f["id"]: f["freq"] for f in feats}
                for concept, (how, v, clip01) in CONCEPTS.items():
                    ch = names.index(concept)
                    fc = interp[sname]["concept_best"][concept]["feature"]
                    pool = [i for i, fr in freq.items() if i != fc and 0.5 * freq[fc] <= fr <= 2 * freq[fc]]
                    rand_feats = list(rng.choice(pool, size=min(a.n_random, len(pool)), replace=False))
                    din, dst, drn = [], [], [[] for _ in rand_feats]
                    P_abl, P_rabl = [], [[] for _ in rand_feats]
                    for b in batches:
                        xp = intervene(b["x"], ch, how, v, clip01)
                        y1p, _, _, _ = step1_forward(model, xp, b["o"], b["ld"])
                        Lin = finish_forward(model, y1p, b["x2"], b["c"], b["hw"]) - b["L0"]
                        B, Lt, d = b["y1"].shape
                        delta = (sae.encode(y1p.reshape(-1, d))[:, fc] -
                                 sae.encode(b["y1"].reshape(-1, d))[:, fc]).reshape(B, Lt)
                        Lst = finish_forward(model, steer(sae, b["y1"], fc, delta=delta), b["x2"],
                                             b["c"], b["hw"]) - b["L0"]
                        din.append(Lin.cpu())
                        dst.append(Lst.cpu())
                        for r, rf in enumerate(rand_feats):
                            Lr = finish_forward(model, steer(sae, b["y1"], int(rf), delta=delta),
                                                b["x2"], b["c"], b["hw"]) - b["L0"]
                            drn[r].append(Lr.cpu())
                        # feature ablation
                        La = finish_forward(model, steer(sae, b["y1"], fc, set_to=0.0), b["x2"],
                                            b["c"], b["hw"])
                        P_abl.append(torch.sigmoid(La).cpu())
                        for r, rf in enumerate(rand_feats):
                            Lra = finish_forward(model, steer(sae, b["y1"], int(rf), set_to=0.0),
                                                 b["x2"], b["c"], b["hw"])
                            P_rabl[r].append(torch.sigmoid(Lra).cpu())
                    din = torch.cat(din).numpy().ravel()
                    dst = torch.cat(dst).numpy().ravel()

                    def align(u):
                        return (float(np.corrcoef(u, din)[0, 1]) if u.std() > 0 else 0.0,
                                float((u * din).sum() / (din ** 2).sum()))
                    corr, frac = align(dst)
                    rc = [align(torch.cat(r).numpy().ravel()) for r in drn]
                    abl = average_precision_score(yy, torch.cat(P_abl).numpy().ravel()) - base_ap
                    rabl = [average_precision_score(yy, torch.cat(r).numpy().ravel()) - base_ap
                            for r in P_rabl]
                    burned = yy > 0
                    key = f"{sname}|{concept}|{h}"
                    results[key] = {
                        "feature": fc, "feature_r": interp[sname]["concept_best"][concept]["r"],
                        "input_effect_mean_logit_burned": float(din[burned].mean()),
                        "input_effect_mean_logit_unburned": float(din[~burned].mean()),
                        "steer_corr": corr, "steer_frac": frac,
                        "random_corr_mean": float(np.mean([x[0] for x in rc])),
                        "random_frac_mean": float(np.mean([x[1] for x in rc])),
                        "random_corr_max": float(np.max([abs(x[0]) for x in rc])),
                        "random_corrs": [float(x[0]) for x in rc],
                        "random_fracs": [float(x[1]) for x in rc],
                        "random_ablation_dAUPRC": [float(v) for v in rabl],
                        "ablation_dAUPRC": float(abl),
                        "random_ablation_dAUPRC_mean": float(np.mean(rabl)),
                        "random_ablation_dAUPRC_minabs": float(np.min(np.abs(rabl))),
                        "base_auprc": float(base_ap),
                    }
                    r_ = results[key]
                    print(f"[h{h}] {sname:15s} {concept:12s} f{fc:4d} | input Δlogit burned "
                          f"{r_['input_effect_mean_logit_burned']:+.3f} | steer corr {corr:+.3f} "
                          f"frac {frac:+.2f} (rand corr {r_['random_corr_mean']:+.3f} frac "
                          f"{r_['random_frac_mean']:+.2f}) | ablate ΔAUPRC {abl*100:+.2f}pp "
                          f"(rand {np.mean(rabl)*100:+.2f})", flush=True)
    (sdir / f"steer_eval_rand{a.n_random}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
