"""추론 지연시간/파라미터: WildfireStepsNet(v4) vs TeleViT 재현. 배치 64 타일(80x80), A40, fp16 autocast."""
import os
import json
import time

import torch

from fyree.dataset_v2 import OCI_VARS, variables
from fyree.model import WildfireStepsNet
from fyree.model_televit import TeleViT


def bench(fn, n_warm=10, n=50):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def main():
    dev = "cuda:0"
    B = 64
    names = variables((1, 4, 12, 24), True, True)
    ours = WildfireStepsNet(variables=names, oci_vars=OCI_VARS, img_size=(80, 80), oci_lag=10,
                            patch_size=4, hidden_size=384, depth=12, num_heads=6,
                            use_lead=True).to(dev).eval()
    tv = TeleViT(in_channels=14, global_in_channels=14, use_lead=False).to(dev).eval()
    x = torch.randn(B, len(names), 80, 80, device=dev)
    o = torch.randn(B, 10, 10, device=dev)
    ld = torch.rand(B, 1, device=dev)
    xg = torch.randn(B, 14, 180, 360, device=dev)
    res = {}
    with torch.no_grad(), torch.autocast("cuda"):
        res["ours_ms_per_batch64"] = bench(lambda: ours(x, o, lead=ld))
        res["televit_ms_per_batch64"] = bench(lambda: tv(x[:, :14], o, xg))
    res["ours_params_M"] = sum(p.numel() for p in ours.parameters()) / 1e6
    res["televit_params_M"] = sum(p.numel() for p in tv.parameters()) / 1e6
    tvi = TeleViT(in_channels=len(names), global_in_channels=14, use_lead=True)
    res["televit_info_params_M"] = sum(p.numel() for p in tvi.parameters()) / 1e6
    print(json.dumps(res, indent=1))
    with open(os.path.join(os.environ.get("FYREE_RUNS", "runs"), "latency.json"), "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
