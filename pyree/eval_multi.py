"""여러 체크포인트를 공유 store 하나로 평가(예측 저장). 2020(두 번째 held-out 연도) 재평가용.
store는 모든 런의 상위 집합(anomaly+global)으로 한 번만 로드하고, 런마다 자기 입력 설정으로
데이터셋을 만든다. prior 설정(sigma, w, tau)이 store와 다른 런은 건너뛴다.

사용: python -m pyree.eval_multi --split test2020 --runs v4_noanchor televit_orig_h1 ...
"""
import os
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pyree import train_v2 as TR
from pyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from pyree.eval_ckpt import DEFAULTS
from pyree.model import WildfireStepsNet
from pyree.model_televit import TeleViT

RUNS = Path(os.environ.get("PYREE_RUNS", "runs"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test2020")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--t_end", default="2020-12-31")
    ap.add_argument("--prior_w", type=int, default=3)
    ap.add_argument("--recency_tau", type=float, default=4.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tile_mode", choices=["positive", "land"], default="positive")
    a = ap.parse_args()
    if a.recency_tau is not None and a.recency_tau < 0:
        a.recency_tau = None  # 균등 기후값 (abl_notau용)
    store = SeasFireStore(t_end=a.t_end, prior_sigma=0.0, prior_w=a.prior_w, anomaly=True,
                          recency_tau=a.recency_tau, with_global=True)
    for name in a.runs:
        run = RUNS / name
        ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        args = argparse.Namespace(**(DEFAULTS | ck["args"]))
        if (args.prior_sigma, args.prior_w, args.recency_tau) != (0.0, a.prior_w, a.recency_tau):
            print(f"skip {name}: prior settings differ", flush=True)
            continue
        use_global = args.arch == "televit"
        ds = SeasFireV2(store, a.split, leads=args.leads, anom_windows=args.anom_windows,
                        long_fire=args.long_fire, tslf=args.tslf, with_global=use_global,
                        tile_mode=a.tile_mode)
        var_names = variables(args.anom_windows, args.long_fire, args.tslf)
        TR.MEMORY_IDX = [var_names.index(v) for v in ("fire_issue", "fire_recent4", "fire_recent12",
                                                       "fire_lastyear", "clim_target", "clim_issue")]
        TR.DROP_IDX = TR.drop_indices(var_names, args.drop)
        n_in = TR.N_BASIC if args.basic_inputs else len(var_names)
        if use_global:
            model = TeleViT(in_channels=n_in, global_in_channels=TR.N_BASIC,
                            use_lead=len(args.leads) > 1)
        else:
            model = WildfireStepsNet(variables=var_names[:n_in], oci_vars=OCI_VARS, img_size=(80, 80),
                                     oci_lag=10, patch_size=args.patch_size,
                                     hidden_size=args.hidden_size, depth=args.depth,
                                     num_heads=args.num_heads, use_lead=True)
        model.load_state_dict(ck["model"])
        model.to(a.device)
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
        tag = a.split if a.tile_mode == "positive" else f"{a.split}_land"
        res = TR.evaluate(model, dl, a.device, args, args.leads, save_preds=run / f"{tag}_preds.npz")
        (run / f"{tag}_eval.json").write_text(json.dumps(res, indent=1))
        print(f"{name} [{tag}] n={len(ds)} " + " ".join(
            f"h{h}:{r['auprc']:.4f}(clim {r['clim_auprc']:.4f}, bss {r['bss_vs_clim']:+.3f})"
            for h, r in res.items()), flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
