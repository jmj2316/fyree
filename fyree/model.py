"""
WildfireStepsNet — Sonny(StormBreaker5)의 StepsNet 백본을 SeasFire 산불 예측에 이식.

원 Sonny (mj/sonny/networks/sonny.py)의 핵심 아이디어를 그대로 가져옴:
  - 변수를 두 그룹(slow/fast)으로 나눠 서로 다른 폭(d1/d2)의 sub-transformer에 태움
  - Step1(slow, narrow) 먼저 처리 -> Step2(fast, full width)에서 합류
  - AdaLN-Zero 컨디셔닝으로 global 조건을 모든 토큰에 주입

원본과 다른 점:
  - group1/group2 분류를 dynamics/thermodynamics 대신 teleconnection-coupled(slow)
    vs local fire-weather(fast)로 재정의 (see `classify_wildfire_variables`)
  - AdaLN 컨디셔닝 c 는 time_interval이 아니라 원격상관 지수(oci_*, oci_lag개월 시계열)에서 옴
    -> TeleViT의 "coarsened global grid + 별도 global 브랜치" 대신, 원격상관을 스칼라
    컨디셔닝으로 전 토큰에 주입하는 단순한 메커니즘 (TeleViT의 asymmetric tokenization 대체)
  - xformers 의존 제거, 표준 torch attention 사용 (CPU 스모크테스트 가능)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_
from timm.layers.mlp import Mlp


# ---- slow(teleconnection-coupled) / fast(local fire-weather) 변수 분류 ----
# TeleViT train.yaml의 로컬 입력 변수 기준. 필요시 확장 가능.
SLOW_VARS = {"sst", "swvl1", "swvl2", "swvl3", "swvl4", "ndvi", "lai", "pop_dens",
             "biomes", "gfed_region", "lsm", "area"}
FAST_VARS = {"t2m_max", "t2m_mean", "t2m_min", "tp", "vpd", "ws10", "rel_hum",
             "mslp", "skt", "ssr", "ssrd", "lst_day", "fwi_max", "fwi_mean",
             "drought_code_max", "drought_code_mean",
             # 발행 시점의 최근 화재 상태는 빠르게 변하는 신호 -> fast
             "fire_issue", "fire_recent4"}
# v3 anomaly: 8일(w1) 윈도우는 fast, 1/3/6개월 누적(w4/w12/w24)은 slow
FAST_VARS |= {f"anom_{v}_w1" for v in ("tp", "swvl1", "ndvi", "vpd", "t2m_mean", "lst_day")}
# fire_recent12 / fire_lastyear / clim_target / clim_issue 는 느린 신호 -> slow (기본값)
# 위치 인코딩(cos/sin lat/lon)은 시간불변 컨텍스트이므로 slow 취급
POSITIONAL_VARS = {"cos_lat", "sin_lat", "cos_lon", "sin_lon"}


def classify_wildfire_variables(variables):
    group1, group2 = [], []  # group1=slow, group2=fast
    for v in variables:
        if v in FAST_VARS:
            group2.append(v)
        else:
            # SLOW_VARS + POSITIONAL_VARS + 미지 변수는 보수적으로 slow
            group1.append(v)
    if not group1 or not group2:
        raise ValueError(f"slow/fast 그룹 중 하나가 비었음: slow={group1}, fast={group2}")
    return group1, group2


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class GroupPatchEmbed(nn.Module):
    """channel 서브셋(group) -> patch conv -> (B, L, d) 토큰"""

    def __init__(self, n_vars, img_size, patch_size, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(n_vars, embed_dim, kernel_size=patch_size, stride=patch_size)
        h, w = img_size
        self.grid = (h // patch_size, w // patch_size)
        n_tokens = self.grid[0] * self.grid[1]
        self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, n_vars, H, W)
        x = self.proj(x)  # (B, d, h, w)
        x = x.flatten(2).transpose(1, 2)  # (B, L, d)
        return x + self.pos_embed


class OciEmbedder(nn.Module):
    """(B, n_oci_vars, oci_lag) 원격상관 시계열 -> (B, hidden_size) 컨디셔닝 벡터"""

    def __init__(self, n_oci_vars, oci_lag, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_oci_vars * oci_lag, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_oci):
        b = x_oci.shape[0]
        return self.mlp(x_oci.reshape(b, -1))


class GlobalContextEmbedder(nn.Module):
    """
    coarsened 전지구 필드(같은 fire-driver 변수, 1도 정도로 뭉갠 (B,C,Hg,Wg)) ->
    (B, hidden_size) 컨디셔닝 벡터.

    TeleViT의 TeleViT_g처럼 별도 토큰 시퀀스로 넣는 대신, 작은 conv 스택 +
    global average pooling으로 "지금 지구 반대편은 어떤 상태인가"를 스칼라 벡터로
    압축해서 oci 컨디셔닝과 같은 통로(AdaLN)에 합류시킨다. TeleViT_g가 오는 신호
    (원격지 공간 맥락)를 OciEmbedder가 놓치는 부분(스칼라 원격상관 지수만으론
    "어디서" 무슨 일이 나는지 모름)에 대한 보완.
    """

    def __init__(self, n_vars, hidden_size, base_channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(n_vars, base_channels, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(base_channels * 2, hidden_size)

    def forward(self, x_global):
        # x_global: (B, n_vars, Hg, Wg)
        h = self.net(x_global).flatten(1)  # (B, base_channels*2)
        return self.proj(h)


class GlobalTokenEmbed(nn.Module):
    """coarsened 전지구 필드 -> (B, Ng, hidden_size) 토큰 시퀀스 (공간 대응관계 유지).

    GlobalContextEmbedder(전역평균풀링, 위치정보 완전 소실)의 대안: 위치별 토큰을
    그대로 유지해서 CrossAttnBlock이 로컬 패치별로 "관련있는 원격 지역"에 다르게
    attend할 수 있게 함 — TeleViT_g가 주는 이점(공간 특정적 원격 신호)의 핵심.
    """

    def __init__(self, n_vars, img_size, patch_size, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(n_vars, hidden_size, kernel_size=patch_size, stride=patch_size)
        h, w = img_size
        gh, gw = h // patch_size, w // patch_size
        self.pos_embed = nn.Parameter(torch.zeros(1, gh * gw, hidden_size))
        trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x_global):
        x = self.proj(x_global).flatten(2).transpose(1, 2)  # (B, Ng, hidden)
        return x + self.pos_embed


class CrossAttnBlock(nn.Module):
    """로컬 토큰(query)이 글로벌 coarse 토큰(key/value)에 cross-attend.
    AdaLN-Zero(oci 조건 c)로 게이팅 -> 초기엔 identity, 학습되며 필요한 만큼만 반영."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm_kv = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                        act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, g, c):
        shift_ca, scale_ca, gate_ca, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        q = modulate(self.norm_q(x), shift_ca, scale_ca)
        kv = self.norm_kv(g)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        x = x + gate_ca.unsqueeze(1) * attn_out
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """adaLN-Zero 컨디셔닝 트랜스포머 블록 (표준 torch attention, xformers 미사용)"""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=attn_drop,
                                           batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden,
                        act_layer=lambda: nn.GELU(approximate="tanh"), drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * self.proj_drop(attn_out)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.Identity()
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class WildfireStepsNet(nn.Module):
    """
    트렁크(결정론적 slow/fast baseline). 입력:
      x_local: (B, C, H, W)  -- input_vars + positional_vars, 채널 순서는 `variables`와 일치
      x_oci:   (B, n_oci_vars, oci_lag)
    출력:
      logits: (B, out_channels, H, W)
    """

    def __init__(self, variables, oci_vars, img_size, oci_lag=10, patch_size=4,
                 hidden_size=384, depth=8, num_heads=6, mlp_ratio=4.0,
                 step_ratio=0.5, depth_step1=None, out_channels=1,
                 global_vars=None, global_mode="adaln", global_img_size=(180, 360),
                 global_patch_size=30, n_cross_blocks=2, use_lead=False):
        """
        global_mode: None(global 컨텍스트 미사용) | "adaln"(전역평균풀링 -> AdaLN 벡터,
          위치정보 소실, 저렴) | "cross_attn"(coarse 토큰 유지 -> 로컬이 cross-attend,
          위치별 원격 대응관계 보존, TeleViT_g에 더 가까운 메커니즘)
        """
        super().__init__()
        h, w = img_size
        pad_h = (patch_size - h % patch_size) % patch_size
        pad_w = (patch_size - w % patch_size) % patch_size
        self.pad = (0, pad_w, 0, pad_h)
        self.img_size = (h + pad_h, w + pad_w)
        self.patch_size = patch_size
        self.variables = list(variables)
        self.out_channels = out_channels

        self.group1_vars, self.group2_vars = classify_wildfire_variables(self.variables)
        self.idx1 = [self.variables.index(v) for v in self.group1_vars]
        self.idx2 = [self.variables.index(v) for v in self.group2_vars]

        self.d1 = int(hidden_size * step_ratio)
        self.d2 = hidden_size - self.d1

        self.embed1 = GroupPatchEmbed(len(self.group1_vars), self.img_size, patch_size, self.d1)
        self.embed2 = GroupPatchEmbed(len(self.group2_vars), self.img_size, patch_size, self.d2)
        self.embed_norm = nn.LayerNorm(hidden_size)

        self.oci_embedder = OciEmbedder(len(oci_vars), oci_lag, hidden_size)
        # Sonny의 time_interval 임베딩과 같은 역할: 리드 h를 AdaLN 조건에 넣어 한 모델로 전 리드 학습
        self.lead_embedder = nn.Sequential(
            nn.Linear(1, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size)
        ) if use_lead else None
        self.global_vars = list(global_vars) if global_vars else None
        self.global_mode = global_mode if self.global_vars else None
        self.global_embedder = None
        self.global_token_embed = None
        self.cross_blocks = None
        if self.global_mode == "adaln":
            self.global_embedder = GlobalContextEmbedder(len(self.global_vars), hidden_size)
        elif self.global_mode == "cross_attn":
            self.global_token_embed = GlobalTokenEmbed(
                len(self.global_vars), global_img_size, global_patch_size, hidden_size)
            self.cross_blocks = nn.ModuleList(
                [CrossAttnBlock(hidden_size, num_heads, mlp_ratio) for _ in range(n_cross_blocks)])
        self.c_proj_step1 = nn.Linear(hidden_size, self.d1)

        head_dim = hidden_size // num_heads
        num_heads_1 = max(1, self.d1 // head_dim)

        depth_1 = depth // 2 if depth_step1 is None else int(depth_step1)
        depth_2 = depth - depth_1
        assert depth_1 >= 1 and depth_2 >= 1, f"depth_1={depth_1}, depth_2={depth_2}"

        self.step1_blocks = nn.ModuleList(
            [Block(self.d1, num_heads_1, mlp_ratio) for _ in range(depth_1)])
        self.step2_blocks = nn.ModuleList(
            [Block(hidden_size, num_heads, mlp_ratio) for _ in range(depth_2)])

        self.head = FinalLayer(hidden_size, patch_size, out_channels)
        self.apply(self._init_weights)
        block_groups = [self.step1_blocks, self.step2_blocks]
        if self.cross_blocks is not None:
            block_groups.append(self.cross_blocks)
        for blocks in block_groups:
            for blk in blocks:
                nn.init.constant_(blk.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(blk.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.head.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.head.linear.weight, 0)
        nn.init.constant_(self.head.linear.bias, 0)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def unpatchify(self, x, orig_h, orig_w):
        p = self.patch_size
        gh, gw = self.img_size[0] // p, self.img_size[1] // p
        v = self.out_channels
        x = x.reshape(x.shape[0], gh, gw, p, p, v)
        x = torch.einsum("nhwpqv->nvhpwq", x)
        imgs = x.reshape(x.shape[0], v, gh * p, gw * p)
        return imgs[:, :, :orig_h, :orig_w]

    def forward(self, x_local, x_oci, x_global=None, lead=None, prior_logit=None):
        orig_h, orig_w = x_local.shape[-2:]
        x_local = F.pad(x_local, self.pad)

        x1_in = x_local[:, self.idx1]
        x2_in = x_local[:, self.idx2]
        emb1 = self.embed1(x1_in)  # (B, L, d1)
        emb2 = self.embed2(x2_in)  # (B, L, d2)
        x = self.embed_norm(torch.cat([emb1, emb2], dim=2))

        c = self.oci_embedder(x_oci)  # (B, hidden)
        if self.lead_embedder is not None:
            assert lead is not None, "use_lead=True 모델은 lead가 필요함"
            c = c + self.lead_embedder(lead)
        if self.global_embedder is not None:
            assert x_global is not None, "global_vars가 설정된 모델은 x_global이 필요함"
            c = c + self.global_embedder(x_global)
        c1 = self.c_proj_step1(c)

        x1 = x[:, :, :self.d1]
        x2 = x[:, :, self.d1:]
        y1 = x1
        for blk in self.step1_blocks:
            y1 = blk(y1, c1)

        y2 = torch.cat([y1, x2], dim=2)
        for blk in self.step2_blocks:
            y2 = blk(y2, c)

        if self.cross_blocks is not None:
            assert x_global is not None, "global_mode=cross_attn인 모델은 x_global이 필요함"
            g = self.global_token_embed(x_global)  # (B, Ng, hidden)
            for blk in self.cross_blocks:
                y2 = blk(y2, g, c)

        out = self.unpatchify(self.head(y2, c), orig_h, orig_w)
        if prior_logit is not None:
            # slow-clock 앵커: head가 zero-init이라 학습 시작 시 출력 = 기후값 그대로,
            # 모델은 평년 대비 편차(anomaly)만 학습
            out = out + prior_logit
        return out


# Public name used in the paper.
Fyree = WildfireStepsNet
