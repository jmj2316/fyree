"""v4 모델의 Step-1(slow path) 출력 토큰 수집 + 토큰별 입력 메타데이터.
토큰 하나 = 4x4 픽셀 패치. 메타: 46개 입력 채널의 패치 평균, 탄 비율, 샘플의 리드/시점/GMSST.

사용: python -m pyree.sae.collect_step1 --run_dir runs/v4_noanchor --n_train 6000 --n_test 2000
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from pyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from pyree.eval_ckpt import DEFAULTS
from pyree.model import WildfireStepsNet
from pyree.sae.tools import finish_forward, step1_forward


@torch.no_grad()
def collect(model, ds, idx, device, batch=64, check=False):
    acts, meta, ybar, lead, T, gm = [], [], [], [], [], []
    g_idx = OCI_VARS.index("oci_gmsst")
    for s in range(0, len(idx), batch):
        items = [ds[int(k)] for k in idx[s:s + batch]]
        x = torch.stack([it["x_local"] for it in items]).to(device)
        o = torch.stack([it["x_oci"] for it in items]).to(device)
        ld = torch.stack([it["lead"] for it in items]).to(device)
        y = torch.stack([it["y_local"] for it in items]).to(device)
        y1, x2, c, hw = step1_forward(model, x, o, ld)
        if check and s == 0:
            ref = model(x, o, lead=ld)
            out = finish_forward(model, y1, x2, c, hw)
            print("split-forward max|Δlogit| =", float((ref - out).abs().max()), flush=True)
        acts.append(y1.half().cpu())
        meta.append(F.avg_pool2d(x, 4).flatten(2).transpose(1, 2).half().cpu())  # (B, 400, 46)
        ybar.append(F.avg_pool2d(y, 4).flatten(1).half().cpu())                 # (B, 400)
        lead.append(torch.tensor([it["h"] for it in items]))
        T.append(torch.tensor([it["T"] for it in items]))
        gm.append(o[:, g_idx, -1].cpu())
    return {"acts": torch.cat(acts), "meta": torch.cat(meta), "ybar": torch.cat(ybar),
            "lead": torch.cat(lead), "T": torch.cat(T), "gmsst": torch.cat(gm)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--n_train", type=int, default=6000)
    ap.add_argument("--n_test", type=int, default=2000, help="리드 1, 16 각각")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    run = Path(a.run_dir)
    out_dir = run / "sae"
    out_dir.mkdir(exist_ok=True)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**(DEFAULTS | ck["args"]))
    store = SeasFireStore(prior_sigma=args.prior_sigma, prior_w=args.prior_w,
                          anomaly=bool(args.anom_windows), recency_tau=args.recency_tau)
    kw = dict(leads=args.leads, anom_windows=args.anom_windows, long_fire=args.long_fire,
              tslf=args.tslf)
    ds_tr, ds_te = SeasFireV2(store, "train", **kw), SeasFireV2(store, "test", **kw)
    names = variables(args.anom_windows, args.long_fire, args.tslf)
    model = WildfireStepsNet(variables=names, oci_vars=OCI_VARS, img_size=(80, 80), oci_lag=10,
                             patch_size=args.patch_size, hidden_size=args.hidden_size,
                             depth=args.depth, num_heads=args.num_heads, use_lead=True)
    model.load_state_dict(ck["model"])
    model.to(a.device).eval()
    rng = np.random.default_rng(0)
    tr_idx = rng.choice(len(ds_tr), size=a.n_train, replace=False)
    d = collect(model, ds_tr, tr_idx, a.device, check=True)
    torch.save(d | {"names": names}, out_dir / "step1_train.pt")
    print("train tokens", tuple(d["acts"].shape), flush=True)
    hs = np.array([s[3] for s in ds_te.samples])
    te_idx = np.concatenate([rng.choice(np.where(hs == h)[0], size=a.n_test, replace=False)
                             for h in (1, 16)])
    d = collect(model, ds_te, te_idx, a.device)
    torch.save(d | {"names": names, "sample_idx": torch.tensor(te_idx)}, out_dir / "step1_test.pt")
    print("test tokens", tuple(d["acts"].shape), flush=True)


if __name__ == "__main__":
    main()
