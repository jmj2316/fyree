"""Step-1 활성에 SAE 학습: TopK(SAE-Xplainers 기준선) / NoClamp B-spline KAN-SAE / JumpReLU.
평가: 설명분산(EV), L0, dead feature 비율 — 학습 분포 밖인 test(2019) 토큰에서도.

사용: python -m pyree.sae.train_sae --run_dir runs/v4_noanchor --variant topk
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from pyree.sae.sae_models import JumpReLUSAE, LinearSAE, NoClampBSplineSAE


def build(variant, d, dict_size, k, l0_coeff):
    if variant == "topk":
        return LinearSAE(d, dict_size, k)
    if variant == "kan":
        return NoClampBSplineSAE(d, dict_size, k, grid_size=8)
    if variant == "jumprelu":
        return JumpReLUSAE(d, dict_size, k, bandwidth=0.5, l0_coeff=l0_coeff)
    raise ValueError(variant)


def optimizer(sae, lr):
    if not hasattr(sae, "phi"):
        return torch.optim.Adam(sae.parameters(), lr=lr)
    back = [p for n, p in sae.named_parameters() if not n.startswith("phi.")]
    return torch.optim.Adam([{"params": back, "lr": lr},
                             {"params": list(sae.phi.parameters()), "lr": lr * 10}])


@torch.no_grad()
def metrics(sae, X, bs=16384):
    sae.eval()
    se, var_num, l0, active = 0.0, 0.0, 0.0, torch.zeros(sae.dict_size, device=X.device)
    mu = X.mean(0)
    for s in range(0, len(X), bs):
        x = X[s:s + bs]
        f = sae.encode(x)
        xh = sae.decode(f)
        se += float(((x - xh) ** 2).sum())
        var_num += float(((x - mu) ** 2).sum())
        l0 += float((f > 0).sum())
        active += (f > 0).float().sum(0)
    sae.train()
    return {"ev": 1 - se / var_num, "l0": l0 / len(X), "dead": float((active == 0).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--variant", choices=["topk", "kan", "jumprelu"], required=True)
    ap.add_argument("--dict_size", type=int, default=768)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--l0_coeff", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    sdir = Path(a.run_dir) / "sae"
    tr = torch.load(sdir / "step1_train.pt")
    te = torch.load(sdir / "step1_test.pt")
    X = tr["acts"].reshape(-1, tr["acts"].shape[-1]).float().to(a.device)
    Xte = te["acts"].reshape(-1, te["acts"].shape[-1]).float().to(a.device)
    perm = torch.randperm(len(X), device=a.device)
    n_val = len(X) // 20
    Xval, Xtr = X[perm[:n_val]], X[perm[n_val:]]
    d = X.shape[1]
    sae = build(a.variant, d, a.dict_size, a.k, a.l0_coeff).to(a.device)
    with torch.no_grad():  # 데이터 중심으로 b_pre 초기화
        sae.b_pre.copy_(Xtr.mean(0))
    opt = optimizer(sae, a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    name = f"{a.variant}{a.tag}"
    log = []
    for ep in range(1, a.epochs + 1):
        idx = torch.randperm(len(Xtr), device=a.device)
        for s in range(0, len(Xtr), a.batch):
            x = Xtr[idx[s:s + a.batch]]
            opt.zero_grad()
            mse, l0 = sae.loss(x)
            mse.backward()
            nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            opt.step()
            sae._normalize_decoder()
        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == a.epochs:
            mv, mt = metrics(sae, Xval), metrics(sae, Xte)
            row = {"epoch": ep, **{f"val_{k}": v for k, v in mv.items()},
                   **{f"test_{k}": v for k, v in mt.items()}}
            log.append(row)
            print(f"[{name}] ep {ep:2d} | val EV {mv['ev']:.3f} L0 {mv['l0']:.1f} dead {mv['dead']:.1%}"
                  f" | test EV {mt['ev']:.3f} dead {mt['dead']:.1%}", flush=True)
    torch.save({"state_dict": sae.state_dict(), "args": vars(a)}, sdir / f"sae_{name}.pt")
    (sdir / f"sae_{name}_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
