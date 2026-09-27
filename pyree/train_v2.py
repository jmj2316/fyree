"""
v2 학습: 기후값-앵커 잔차 + fire memory + 리드 조건부 단일 모델.

체크포인트 선택은 val(2018) 리드 평균 AUPRC, 최종 보고는 test(2019) — TeleViT와 동일.
리드별로 AUPRC, Brier, 기후값 대비 BSS, 같은 샘플에서의 기후값 AUPRC를 함께 기록.

사용:
  python -m pyree.train_v2 --out_dir runs/v2_anchor --device cuda:0
"""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, RandomSampler

from torch import nn

from pyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from pyree.model import WildfireStepsNet
from pyree.model_televit import TeleViT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--leads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--prior_sigma", type=float, default=0.0)
    p.add_argument("--prior_w", type=int, default=0)
    p.add_argument("--no_anchor", action="store_true", help="prior_logit을 출력에 더하지 않음 (ablation)")
    p.add_argument("--no_memory", action="store_true", help="fire memory 채널 0으로 (ablation)")
    p.add_argument("--anom_windows", nargs="*", type=int, default=[],
                   help="v3: 누적 anomaly 윈도우(스텝), 예: 1 4 12 24")
    p.add_argument("--long_fire", action="store_true", help="v3: 지난 46스텝(1년) 화재 빈도 채널")
    p.add_argument("--tslf", action="store_true", help="v4: 마지막 화재 이후 경과 시간(연료 회복) 채널")
    p.add_argument("--recency_tau", type=float, default=None, help="prior 최근 연도 가중 (년)")
    p.add_argument("--ema", type=float, default=0.0, help="EMA decay (0이면 미사용), 평가·저장은 EMA 가중치")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--samples_per_epoch", type=int, default=60000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--hidden_size", type=int, default=384)
    p.add_argument("--depth", type=int, default=12)
    p.add_argument("--num_heads", type=int, default=6)
    p.add_argument("--patch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--arch", choices=["stepsnet", "televit"], default="stepsnet")
    p.add_argument("--drop", nargs="*", default=[],
                   help="절제: 0으로 만들 입력 그룹 (fire_history, clim, oci, anom, met)")
    p.add_argument("--basic_inputs", action="store_true",
                   help="로컬 입력을 기상10+위치4 (TeleViT 원본)로 제한")
    return p.parse_args()


MEMORY_IDX = None  # main()에서 variables() 기준으로 채움
DROP_IDX = []  # main()에서 --drop 그룹 기준으로 채움
DROP_GROUPS = {
    "fire_history": ("fire_issue", "fire_recent4", "fire_recent12", "fire_lastyear",
                     "fire_recent46", "fire_tslf"),
    "clim": ("clim_target", "clim_issue"),
    "met": ("lst_day", "mslp", "ndvi", "pop_dens", "ssrd", "sst", "swvl1", "t2m_mean", "tp", "vpd"),
}


def drop_indices(var_names, groups):
    idx = []
    for g in groups:
        if g in ("oci",):
            continue
        if g == "anom":
            idx += [k for k, v in enumerate(var_names) if v.startswith("anom_")]
        else:
            idx += [var_names.index(v) for v in DROP_GROUPS[g] if v in var_names]
    return sorted(set(idx))
N_BASIC = 14  # 기상 10 + 위치 4 = TeleViT 원본 로컬 입력


def forward(model, batch, device, args):
    """배치를 옮기고 아키텍처에 맞게 호출. (logits, y) 반환."""
    x = batch["x_local"].to(device, non_blocking=True)
    if args.no_memory:
        x = x.clone()
        x[:, MEMORY_IDX] = 0.0
    if DROP_IDX:
        x = x.clone()
        x[:, DROP_IDX] = 0.0
    if args.basic_inputs:
        x = x[:, :N_BASIC]
    oci = batch["x_oci"].to(device, non_blocking=True)
    if "oci" in getattr(args, "drop", []):
        oci = torch.zeros_like(oci)
    lead = batch["lead"].to(device, non_blocking=True)
    prior = None if args.no_anchor else batch["prior_logit"].to(device, non_blocking=True)
    y = batch["y_local"].to(device, non_blocking=True)
    if args.arch == "televit":
        xg = batch["x_global"].to(device, non_blocking=True)
        out = model(x, oci, xg, lead=lead, prior_logit=prior)
    else:
        out = model(x, oci, lead=lead, prior_logit=prior)
    return out, y


@torch.no_grad()
def evaluate(model, loader, device, args, leads, clim_cache=None, save_preds=None):
    model.eval()
    buf = {h: {"p": [], "y": [], "c": [], "T": [], "ij": []} for h in leads}
    for batch in loader:
        with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
            out, y = forward(model, batch, device, args)
        prob = torch.sigmoid(out.float()).cpu().numpy()
        yy = y.cpu().numpy()
        cc = batch["clim"].numpy()
        for k, h in enumerate(batch["h"].tolist()):
            buf[h]["p"].append(prob[k].ravel())
            buf[h]["y"].append(yy[k].ravel().astype(np.uint8))
            buf[h]["c"].append(cc[k].ravel())
            buf[h]["T"].append(int(batch["T"][k]))
            if "ij" in batch:
                buf[h]["ij"].append(int(batch["ij"][k]))
    if save_preds is not None:  # 저장만 float32 확률 그대로 (AUPRC 동률 방지), y는 uint8
        np.savez_compressed(save_preds, **{
            f"{key}_h{h}": (np.stack(buf[h][key]) if key not in ("T", "ij") else np.array(buf[h][key]))
            for h in leads for key in ("p", "y", "c", "T", "ij") if buf[h][key]})
    res = {}
    for h in leads:
        p = np.concatenate(buf[h]["p"]).astype(np.float32)
        y = np.concatenate(buf[h]["y"]).astype(np.float32)
        c = np.concatenate(buf[h]["c"]).astype(np.float32)
        r = {"auprc": float(average_precision_score(y, p)),
             "brier": float(np.mean((p - y) ** 2))}
        if clim_cache is not None and h in clim_cache:
            r.update(clim_cache[h])
        else:
            r["clim_auprc"] = float(average_precision_score(y, c))
            r["clim_brier"] = float(np.mean((c - y) ** 2))
        r["bss_vs_clim"] = 1.0 - r["brier"] / r["clim_brier"]
        res[h] = r
    model.train()
    return res


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = open(out_dir / "train_log.jsonl", "a")

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading store ...", flush=True)
    use_global = args.arch == "televit"
    store = SeasFireStore(prior_sigma=args.prior_sigma, prior_w=args.prior_w,
                          anomaly=bool(args.anom_windows), recency_tau=args.recency_tau,
                          with_global=use_global)
    dkw = dict(leads=args.leads, anom_windows=args.anom_windows, long_fire=args.long_fire,
               tslf=args.tslf, with_global=use_global)
    ds_tr = SeasFireV2(store, "train", **dkw)
    ds_va = SeasFireV2(store, "val", **dkw)
    ds_te = SeasFireV2(store, "test", **dkw)
    var_names = variables(args.anom_windows, args.long_fire, args.tslf)
    global MEMORY_IDX, DROP_IDX
    MEMORY_IDX = [var_names.index(v) for v in
                  ("fire_issue", "fire_recent4", "fire_recent12", "fire_lastyear",
                   "clim_target", "clim_issue")]
    DROP_IDX = drop_indices(var_names, args.drop)
    print(f"[{time.strftime('%H:%M:%S')}] store loaded in {time.time()-t0:.0f}s; "
          f"train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}", flush=True)

    sampler = RandomSampler(ds_tr, replacement=False,
                            num_samples=min(args.samples_per_epoch, len(ds_tr)))
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, sampler=sampler,
                       num_workers=args.num_workers, drop_last=True, persistent_workers=True)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers)
    dl_te = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers)

    device = args.device if torch.cuda.is_available() else "cpu"
    n_in = N_BASIC if args.basic_inputs else len(var_names)
    if args.arch == "televit":
        model = TeleViT(in_channels=n_in, global_in_channels=N_BASIC,
                        use_lead=len(args.leads) > 1).to(device)
        if not args.no_anchor:  # 앵커 모드면 StepsNet처럼 시작점 = 기후값
            nn.init.zeros_(model.head.weight)
            nn.init.zeros_(model.head.bias)
        grouping = "-"
    else:
        model = WildfireStepsNet(
            variables=var_names[:n_in], oci_vars=OCI_VARS, img_size=(80, 80), oci_lag=10,
            patch_size=args.patch_size, hidden_size=args.hidden_size, depth=args.depth,
            num_heads=args.num_heads, use_lead=True,
        ).to(device)
        grouping = f"slow={len(model.group1_vars)} fast={len(model.group2_vars)}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} params {n_params/1e6:.2f}M | anchor={not args.no_anchor} "
          f"memory={not args.no_memory} basic_inputs={args.basic_inputs} n_in={n_in} "
          f"prior(sigma={args.prior_sigma}, w={args.prior_w}, tau={args.recency_tau}) "
          f"anom={args.anom_windows} long_fire={args.long_fire} tslf={args.tslf} ema={args.ema} "
          f"leads={args.leads} drop={args.drop}(n_ch={len(DROP_IDX)}) {grouping}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(dl_tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total_steps,
                                                pct_start=0.05)
    crit = torch.nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler(enabled=device.startswith("cuda"))
    log.write(json.dumps({"event": "start", "args": vars(args), "n_params": n_params}) + "\n")

    # 학습 전(=순수 기후값 앵커) 성능도 기록: anchor 모드면 이게 정확히 기후값이어야 함
    clim_cache = None
    r0 = evaluate(model, dl_va, device, args, args.leads)
    clim_cache = {h: {"clim_auprc": r0[h]["clim_auprc"], "clim_brier": r0[h]["clim_brier"]}
                  for h in args.leads}
    print("[init] " + " ".join(f"h{h}:{r0[h]['auprc']:.4f}(clim {r0[h]['clim_auprc']:.4f})"
                               for h in args.leads), flush=True)
    log.write(json.dumps({"event": "init_val", "res": r0}) + "\n")

    ema = None
    if args.ema > 0:
        ema = copy.deepcopy(model).eval()
        for p in ema.parameters():
            p.requires_grad_(False)
    eval_model = ema if ema is not None else model

    best, since = -1.0, 0
    for ep in range(args.epochs):
        t1 = time.time()
        run, n = 0.0, 0
        for batch in dl_tr:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=device.startswith("cuda")):
                out, y = forward(model, batch, device, args)
                loss = crit(out.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if ema is not None:
                with torch.no_grad():
                    for pe, pm in zip(ema.parameters(), model.parameters()):
                        pe.mul_(args.ema).add_(pm.detach(), alpha=1 - args.ema)
            run += loss.item() * y.shape[0]
            n += y.shape[0]
        r = evaluate(eval_model, dl_va, device, args, args.leads, clim_cache)
        mean_auprc = float(np.mean([r[h]["auprc"] for h in args.leads]))
        print(f"[ep {ep}] loss={run/n:.4f} val_mean={mean_auprc:.4f} | " +
              " ".join(f"h{h}:{r[h]['auprc']:.4f}/bss{r[h]['bss_vs_clim']:+.3f}" for h in args.leads) +
              f" ({time.time()-t1:.0f}s)", flush=True)
        log.write(json.dumps({"event": "epoch", "epoch": ep, "loss": run / n,
                              "val_mean_auprc": mean_auprc, "val": r}) + "\n")
        log.flush()
        if mean_auprc > best:
            best, since = mean_auprc, 0
            torch.save({"model": eval_model.state_dict(), "args": vars(args), "epoch": ep,
                        "val": r, "variables": var_names}, out_dir / "best.pt")
        else:
            since += 1
            if since >= args.patience:
                print(f"early stop at epoch {ep}", flush=True)
                break

    ck = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    rt = evaluate(model, dl_te, device, args, args.leads, save_preds=out_dir / "test_preds.npz")
    summary = {"best_epoch": ck["epoch"], "val": ck["val"], "test": rt}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1))
    print("[TEST] " + " | ".join(
        f"h{h}: auprc {rt[h]['auprc']:.4f} (clim {rt[h]['clim_auprc']:.4f}) "
        f"brier {rt[h]['brier']:.5f} bss {rt[h]['bss_vs_clim']:+.3f}" for h in args.leads), flush=True)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
