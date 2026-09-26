"""
TeleViT 재구현 (원 코드 televit/src/models/components/{televit,vit_base}.py 구조를 따름).

  로컬 (C,80,80) -> 16x16 패치 25토큰
  OCI (10변수 x 10개월) -> 스칼라 하나당 1토큰 (1x1 conv = 공유 선형사상) 100토큰
  전지구 (C,180,360) -> 30x30 패치 72토큰
  [cls] + 위 토큰 + 학습형 위치임베딩 -> timm ViT Block x8 (D=768, 12헤드, pre-norm) -> norm
  cls 토큰 -> Linear -> 80x80 logit   (원본은 2클래스 CE, 여기선 동치인 1-logit BCE)

비교 실험용 옵션 (원본엔 없음): use_lead(리드 토큰 1개 추가), forward의 prior_logit(기후값 앵커).
"""
import torch
import torch.nn as nn
from timm.layers import trunc_normal_
from timm.models.vision_transformer import Block


class TeleViT(nn.Module):
    def __init__(self, in_channels, global_in_channels, local_size=80, patch_size=16,
                 n_oci=10, oci_lag=10, global_shape=(180, 360), global_patch_size=30,
                 embed_dim=768, depth=8, num_heads=12, use_lead=False):
        super().__init__()
        self.local_size = local_size
        self.local_embed = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        n_local = (local_size // patch_size) ** 2
        self.oci_embed = nn.Conv2d(1, embed_dim, 1, 1)
        n_oci_tok = n_oci * oci_lag
        self.global_embed = nn.Conv2d(global_in_channels, embed_dim, global_patch_size,
                                      global_patch_size)
        n_global = (global_shape[0] // global_patch_size) * (global_shape[1] // global_patch_size)
        self.lead_embed = nn.Linear(1, embed_dim) if use_lead else None
        n_tok = 1 + n_local + n_oci_tok + n_global + (1 if use_lead else 0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, n_tok, embed_dim) * 0.02)
        self.blocks = nn.Sequential(*[Block(embed_dim, num_heads, mlp_ratio=4.0, qkv_bias=True)
                                      for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, local_size * local_size)
        nn.init.normal_(self.cls_token, std=1e-6)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x_local, x_oci, x_global, lead=None, prior_logit=None):
        b = x_local.shape[0]
        toks = [self.cls_token.expand(b, -1, -1),
                self.local_embed(x_local).flatten(2).transpose(1, 2),
                self.oci_embed(x_oci.unsqueeze(1)).flatten(2).transpose(1, 2),
                self.global_embed(x_global).flatten(2).transpose(1, 2)]
        if self.lead_embed is not None:
            toks.append(self.lead_embed(lead).unsqueeze(1))
        x = torch.cat(toks, dim=1) + self.pos_embed
        x = self.norm(self.blocks(x))
        out = self.head(x[:, 0]).view(b, 1, self.local_size, self.local_size)
        if prior_logit is not None:
            out = out + prior_logit
        return out
