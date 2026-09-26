"""
Integrated Gradients: v4 모델의 test 샘플에서 입력 채널(46)과 OCI(10변수 x 10개월)별 기여도.
타깃(기본 disc) = 타일 안 탄 픽셀 평균 logit − 안 탄 픽셀 평균 logit (AUPRC가 보는 판별력),
기준점 = 0 (정규화 입력의 평균, 화재 없음, 기후값 0). 위치(sin/cos)와 리드는 실제값 고정(기여 대상 아님).
완전성(기여 합 = f(x) − f(기준점)) 확인.

사용: python -m fyree.attribution --run_dir runs/v4_noanchor --n_per_lead 384
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from fyree import train_v2 as TR
from fyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from fyree.eval_ckpt import DEFAULTS
from fyree.model import WildfireStepsNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_per_lead", type=int, default=384)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--target", choices=["mean", "disc"], default="disc",
                    help="mean: 타일 평균 logit / disc: 탄 픽셀 평균 logit − 안 탄 픽셀 평균 logit")
    ap.add_argument("--out_name", default="ig_attributions_disc.npz")
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
    for p in model.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(0)
    hs = np.array([s[3] for s in ds.samples])
    out = {"names": np.array(names), "oci_vars": np.array(OCI_VARS)}
    alphas = (torch.arange(a.steps, device=a.device, dtype=torch.float32) + 0.5) / a.steps
    # 위치(sin/cos)는 기준점 0이 임의의 지점(적도·본초자오선)이라 기여 대상에서 제외하고 실제값 고정
    pos_idx = [names.index(v) for v in ("cos_lat", "sin_lat", "cos_lon", "sin_lon")]
    attr_mask = torch.ones(len(names), device=a.device)
    attr_mask[pos_idx] = 0.0
    attr_mask = attr_mask.view(1, -1, 1, 1)

    def target(logits, y):
        if a.target == "mean":
            return logits.mean(dim=(1, 2, 3))
        pos = (logits * y).sum(dim=(1, 2, 3)) / y.sum(dim=(1, 2, 3)).clamp_min(1)
        neg = (logits * (1 - y)).sum(dim=(1, 2, 3)) / (1 - y).sum(dim=(1, 2, 3)).clamp_min(1)
        return pos - neg
    for h in args.leads:
        idx = rng.choice(np.where(hs == h)[0], size=min(a.n_per_lead, int((hs == h).sum())),
                         replace=False)
        loc_abs, loc_sgn, oci_sgn, oci_abs, compl = [], [], [], [], []
        for s in range(0, len(idx), a.batch):
            items = [ds[int(k)] for k in idx[s:s + a.batch]]
            x = torch.stack([it["x_local"] for it in items]).to(a.device)
            o = torch.stack([it["x_oci"] for it in items]).to(a.device)
            ld = torch.stack([it["lead"] for it in items]).to(a.device)
            yy = torch.stack([it["y_local"] for it in items]).to(a.device)
            x_fix = x * (1 - attr_mask)   # 고정 부분(위치)
            x_var = x * attr_mask         # 기여 대상
            gx = torch.zeros_like(x)
            go = torch.zeros_like(o)
            for al in alphas:
                xa = (al * x_var).requires_grad_(True)
                oa = (al * o).requires_grad_(True)
                f = target(model(xa + x_fix, oa, lead=ld), yy).sum()
                g1, g2 = torch.autograd.grad(f, [xa, oa])
                gx += g1 / a.steps
                go += g2 / a.steps
            ax = gx * x_var
            ao = go * o
            with torch.no_grad():
                fx = target(model(x, o, lead=ld), yy)
                f0 = target(model(x_fix, torch.zeros_like(o), lead=ld), yy)
            compl.append(((ax.sum(dim=(1, 2, 3)) + ao.sum(dim=(1, 2))) - (fx - f0)).abs().cpu()
                         / (fx - f0).abs().clamp_min(1e-6).cpu())
            loc_abs.append(ax.abs().sum(dim=(2, 3)).cpu())
            loc_sgn.append(ax.sum(dim=(2, 3)).cpu())
            oci_abs.append(ao.abs().cpu())
            oci_sgn.append(ao.cpu())
        out[f"local_abs_h{h}"] = torch.cat(loc_abs).numpy()
        out[f"local_sgn_h{h}"] = torch.cat(loc_sgn).numpy()
        out[f"oci_abs_h{h}"] = torch.cat(oci_abs).numpy()
        out[f"oci_sgn_h{h}"] = torch.cat(oci_sgn).numpy()
        c = torch.cat(compl).numpy()
        out[f"completeness_relerr_h{h}"] = c
        print(f"h={h}: n={len(idx)} completeness rel.err median={np.median(c):.3f}", flush=True)
    out["target"] = np.array(a.target)
    np.savez_compressed(run / a.out_name, **out)
    print("saved", run / a.out_name)


if __name__ == "__main__":
    main()
