"""SAE feature 해석: test 토큰에서 feature 활성과 토큰 입력 통계의 상관.
설명변수 = 46개 입력 채널의 패치 평균 + GMSST(샘플) + 리드 + 실제 탄 비율(ybar).
출력: 변형별 (feature -> 최고 상관 변수, r), 개념별 최고 feature, 요약 지표."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pyree.sae.train_sae import build

CONCEPTS = ["fire_tslf", "clim_target", "fire_recent46", "fire_issue", "gmsst",
            "anom_tp_w24", "anom_swvl1_w12", "anom_vpd_w4", "ndvi", "swvl1", "ybar"]


def load_sae(sdir, name, device):
    ck = torch.load(sdir / f"sae_{name}.pt", map_location="cpu", weights_only=False)
    a = ck["args"]
    sae = build(a["variant"], 192, a["dict_size"], a["k"], a["l0_coeff"])
    sae.load_state_dict(ck["state_dict"])
    return sae.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="runs/v4_noanchor")
    ap.add_argument("--saes", nargs="+", default=["topk", "jumprelu_l01.0", "kan"])
    ap.add_argument("--n_tokens", type=int, default=400000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    sdir = Path(a.run_dir) / "sae"
    te = torch.load(sdir / "step1_test.pt")
    names = list(te["names"])
    N, L, d = te["acts"].shape
    rng = np.random.default_rng(0)
    tok = rng.choice(N * L, size=min(a.n_tokens, N * L), replace=False)
    s_idx = tok // L
    X = te["acts"].reshape(-1, d)[tok].float()
    meta = te["meta"].reshape(-1, te["meta"].shape[-1])[tok].float().numpy()
    extra = {"gmsst": te["gmsst"].numpy()[s_idx], "lead": te["lead"].numpy()[s_idx].astype(float),
             "ybar": te["ybar"].reshape(-1)[tok].float().numpy()}
    var_names = names + list(extra)
    V = np.concatenate([meta, np.stack([extra[k] for k in extra], 1)], 1)  # (n, n_var)
    Vz = (V - V.mean(0)) / (V.std(0) + 1e-8)
    out = {"var_names": var_names}
    for name in a.saes:
        sae = load_sae(sdir, name, a.device)
        with torch.no_grad():
            F = torch.cat([sae.encode(X[s:s + 50000].to(a.device)).cpu()
                           for s in range(0, len(X), 50000)]).numpy()
        freq = (F > 0).mean(0)
        alive = freq > 1e-4
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)
        R = (Fz.T @ Vz) / len(Fz)               # (dict, n_var) Pearson r
        R[~alive] = 0.0
        best_var = np.abs(R).argmax(1)
        best_r = R[np.arange(len(R)), best_var]
        feats = [{"id": int(i), "freq": float(freq[i]), "var": var_names[best_var[i]],
                  "r": float(best_r[i])} for i in np.where(alive)[0]]
        concept_best = {}
        for c in CONCEPTS:
            j = var_names.index(c)
            i = int(np.abs(R[:, j]).argmax())
            concept_best[c] = {"feature": i, "r": float(R[i, j]), "freq": float(freq[i]),
                               "top3": [(int(k), float(R[k, j])) for k in np.argsort(-np.abs(R[:, j]))[:3]]}
        summ = {"alive": int(alive.sum()), "dict": int(len(freq)),
                "n_r>0.5": int((np.abs(best_r) > 0.5).sum()),
                "n_r>0.7": int((np.abs(best_r) > 0.7).sum()),
                "median_max|r|_alive": float(np.median(np.abs(best_r[alive])))}
        out[name] = {"summary": summ, "concept_best": concept_best, "features": feats}
        print(f"== {name}: {summ}")
        for c, v in concept_best.items():
            print(f"   {c:15s} best f{v['feature']:4d} r={v['r']:+.3f} freq={v['freq']:.3f}")
    (sdir / "interpret.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
