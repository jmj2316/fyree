"""학습된 best.pt를 test에 다시 평가하고 예측(test_preds.npz)을 저장.
사용: python -m fyree.eval_ckpt --run_dir runs/v4_noanchor --device cuda:0"""
import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from fyree import train_v2 as TR
from fyree.dataset_v2 import OCI_VARS, SeasFireStore, SeasFireV2, variables
from fyree.model import WildfireStepsNet
from fyree.model_televit import TeleViT

DEFAULTS = {"arch": "stepsnet", "basic_inputs": False, "tslf": False, "recency_tau": None,
            "anom_windows": [], "long_fire": False, "ema": 0.0, "drop": []}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    run = Path(a.run_dir)
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**(DEFAULTS | ck["args"]))
    use_global = args.arch == "televit"
    store = SeasFireStore(prior_sigma=args.prior_sigma, prior_w=args.prior_w,
                          anomaly=bool(args.anom_windows), recency_tau=args.recency_tau,
                          with_global=use_global)
    ds = SeasFireV2(store, a.split, leads=args.leads, anom_windows=args.anom_windows,
                    long_fire=args.long_fire, tslf=args.tslf, with_global=use_global)
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
    res = TR.evaluate(model, dl, a.device, args, args.leads,
                      save_preds=run / f"{a.split}_preds.npz")
    print(json.dumps({str(h): r for h, r in res.items()}, indent=1))
    (run / f"{a.split}_reeval.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
