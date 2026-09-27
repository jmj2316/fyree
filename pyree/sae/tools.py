"""WildfireStepsNet을 Step-1(slow path) 출력에서 둘로 나눠 실행: 활성 수집과 steering 공통 도구."""
import torch
import torch.nn.functional as F


def step1_forward(model, x_local, x_oci, lead):
    """Step-1 끝까지 실행. 반환: y1 (B, L, d1), x2 (B, L, d2), c (B, hidden), 원래 H, W."""
    orig_h, orig_w = x_local.shape[-2:]
    x_local = F.pad(x_local, model.pad)
    emb1 = model.embed1(x_local[:, model.idx1])
    emb2 = model.embed2(x_local[:, model.idx2])
    x = model.embed_norm(torch.cat([emb1, emb2], dim=2))
    c = model.oci_embedder(x_oci)
    if model.lead_embedder is not None:
        c = c + model.lead_embedder(lead)
    c1 = model.c_proj_step1(c)
    y1 = x[:, :, :model.d1]
    for blk in model.step1_blocks:
        y1 = blk(y1, c1)
    return y1, x[:, :, model.d1:], c, (orig_h, orig_w)


def finish_forward(model, y1, x2, c, hw):
    """Step-2 + head. (앵커 없는 모델 기준 — prior_logit은 호출자가 필요하면 더함)"""
    y2 = torch.cat([y1, x2], dim=2)
    for blk in model.step2_blocks:
        y2 = blk(y2, c)
    return model.unpatchify(model.head(y2, c), *hw)


def steer(sae, y1, feat, delta=None, set_to=None, token_mask=None):
    """y1 (B, L, d)의 feature `feat` 활성을 바꾼 효과를 활성 공간에 더한다.
    delta: 스칼라 또는 토큰별 (B, L) 증분 / set_to: 활성을 이 값으로 고정(0이면 ablation).
    Δ = decode(f') − decode(f): 재구성 오차는 보존, 비선형(KAN) 디코더에도 동일하게 적용."""
    B, L, d = y1.shape
    flat = y1.reshape(-1, d)
    f = sae.encode(flat)
    f2 = f.clone()
    if set_to is not None:
        f2[:, feat] = set_to
    else:
        dl = delta.reshape(-1) if torch.is_tensor(delta) else delta
        f2[:, feat] = (f2[:, feat] + dl).clamp_min(0.0)
    diff = (sae.decode(f2) - sae.decode(f)).reshape(B, L, d)
    if token_mask is not None:
        diff = diff * token_mask.unsqueeze(-1)
    return y1 + diff
