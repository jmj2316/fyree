"""
SAE variants for Sonny residual stream interpretability.

Three decoders:
  LinearSAE     — standard: x_hat = W_dec @ f + b_pre
  FourierKANSAE — factored: x_hat = sum_i phi_i(f_i) * v_i + b_pre
  BSplineKANSAE — factored: same structure, B-spline phi_i

phi_i: R -> R is a learned scalar nonlinearity per feature.
v_i: unit-norm decoder direction in R^D (same as LinearSAE column).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def topk_activation(x: torch.Tensor, k: int) -> torch.Tensor:
    """TopK sparsity: keep top-k activations per token, zero the rest."""
    topk_vals, topk_idx = torch.topk(x, k, dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(-1, topk_idx, topk_vals)
    return out


class LinearSAE(nn.Module):
    """Standard TopK Sparse Autoencoder (linear decoder, baseline)."""

    def __init__(self, d_model: int, dict_size: int, k: int):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_activation(F.relu(self.W_enc @ (x - self.b_pre).T).T + self.b_enc, self.k)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss(self, x: torch.Tensor):
        x_hat, f = self(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        l0 = (f > 0).float().sum(-1).mean()
        return mse, l0


class _FourierPhi(nn.Module):
    """Per-feature Fourier scalar nonlinearity: phi_i(a) = a + Σ_k [c_k cos(kπa/s) + s_k sin(kπa/s)]."""

    def __init__(self, dict_size: int, n_freqs: int = 8):
        super().__init__()
        self.n_freqs = n_freqs
        # [M, n_freqs, 2]: last dim = (cos_coeff, sin_coeff)
        self.coeffs = nn.Parameter(torch.zeros(dict_size, n_freqs, 2))
        # Learnable input scale (log-parameterized, initialized to log(1.0))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [..., M]
        scale = self.log_scale.exp().clamp(min=0.1)
        a_norm = f / scale  # normalize activations
        k = torch.arange(1, self.n_freqs + 1, device=f.device, dtype=f.dtype)
        # args: [..., M, n_freqs]
        args = a_norm.unsqueeze(-1) * k * math.pi
        cos_part = torch.cos(args) * self.coeffs[..., 0]
        sin_part = torch.sin(args) * self.coeffs[..., 1]
        # phi(f) = f + fourier_correction
        return f + (cos_part + sin_part).sum(-1)


class FourierKANSAE(nn.Module):
    """Factored KAN-SAE with Fourier scalar nonlinearity per feature."""

    def __init__(self, d_model: int, dict_size: int, k: int, n_freqs: int = 8):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        # Decoder directions: columns of W_dec are unit-norm feature directions
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()

        self.phi = _FourierPhi(dict_size, n_freqs)

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_activation(F.relu(self.W_enc @ (x - self.b_pre).T).T + self.b_enc, self.k)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        # f_phi: nonlinearly scaled codes [..., M]
        f_phi = self.phi(f)
        # x_hat = f_phi @ W_dec.T + b_pre
        return f_phi @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss(self, x: torch.Tensor):
        x_hat, f = self(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        l0 = (f > 0).float().sum(-1).mean()
        return mse, l0


class _BSplinePhi(nn.Module):
    """Per-feature B-spline scalar nonlinearity using a uniform grid over [0, scale]."""

    def __init__(self, dict_size: int, grid_size: int = 8):
        super().__init__()
        self.grid_size = grid_size
        # Control points per feature: [M, grid_size + 1]
        self.ctrl = nn.Parameter(torch.zeros(dict_size, grid_size + 1))
        # Initialize control points as identity (phi(a) ≈ a)
        t = torch.linspace(0, 1, grid_size + 1)
        self.ctrl.data = t.unsqueeze(0).expand(dict_size, -1).clone()
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [..., M]  (activations, >= 0 after TopK+ReLU)
        scale = self.log_scale.exp().clamp(min=0.1)
        t = (f / scale).clamp(0.0, 1.0)  # [..., M], in [0,1]

        G = self.grid_size
        # Find bin index for each activation
        idx = (t * G).long().clamp(0, G - 1)  # [..., M]
        t_local = t * G - idx.float()  # fractional part in [0,1]

        c0 = self.ctrl[:, :G]   # [M, G]
        c1 = self.ctrl[:, 1:]   # [M, G]

        # Gather control points: c0_sel[..., i] = c0[i, idx[..., i]]
        # idx: [..., M], c0: [M, G]
        orig_shape = f.shape
        idx_flat = idx.reshape(-1, idx.shape[-1])          # [B, M]
        B = idx_flat.shape[0]
        c0_exp = c0.unsqueeze(0).expand(B, -1, -1)        # [B, M, G]
        c1_exp = c1.unsqueeze(0).expand(B, -1, -1)
        idx_g  = idx_flat.clamp(0, G - 1).unsqueeze(-1)   # [B, M, 1]
        c0_sel = c0_exp.gather(2, idx_g).squeeze(-1).reshape(orig_shape)
        c1_sel = c1_exp.gather(2, idx_g).squeeze(-1).reshape(orig_shape)

        # Interpolated value in [0,1] range, then rescale
        phi_norm = c0_sel * (1 - t_local) + c1_sel * t_local
        return phi_norm * scale


class BSplineKANSAE(nn.Module):
    """Factored KAN-SAE with B-spline scalar nonlinearity per feature."""

    def __init__(self, d_model: int, dict_size: int, k: int, grid_size: int = 8):
        super().__init__()
        self.d_model = d_model
        self.dict_size = dict_size
        self.k = k

        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()

        self.phi = _BSplinePhi(dict_size, grid_size)

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return topk_activation(F.relu(self.W_enc @ (x - self.b_pre).T).T + self.b_enc, self.k)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        f_phi = self.phi(f)
        return f_phi @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        x_hat = self.decode(f)
        return x_hat, f

    def loss(self, x: torch.Tensor):
        x_hat, f = self(x)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        l0 = (f > 0).float().sum(-1).mean()
        return mse, l0


# ═══════════════════════════════════════════════════════════════════════════
# Encoder-side variants (added 2026-08-24 for the revision).
#
# The three classes above all share the SAME encoder --- topk(relu(W_enc x)) ---
# and differ only in the decoder.  The two classes below instead replace the
# encoder nonlinearity, which is what the paper's Eq. (5) describes.
# ═══════════════════════════════════════════════════════════════════════════


class _EncBSplinePhi(nn.Module):
    """Per-feature piecewise-linear spline over a symmetric grid [-s, s].

    Unlike _BSplinePhi (which acts on already-positive TopK codes in the
    decoder) this acts on raw pre-activations, so it must handle h < 0.
    Control points are initialised to ReLU so training starts exactly at the
    Lin-SAE baseline and any departure is learned.
    """

    def __init__(self, dict_size: int, grid_size: int = 16, scale_init: float = 10.0):
        super().__init__()
        self.grid_size = grid_size
        grid = torch.linspace(-1.0, 1.0, grid_size + 1)
        self.ctrl = nn.Parameter(
            grid.clamp(min=0.0).unsqueeze(0).expand(dict_size, -1).clone()
        )
        self.log_scale = nn.Parameter(torch.tensor(float(math.log(scale_init))))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        G = self.grid_size
        scale = self.log_scale.exp().clamp(min=0.1)
        t = ((h / scale).clamp(-1.0, 1.0) + 1.0) * 0.5      # -> [0, 1]
        idx = (t * G).long().clamp(0, G - 1)
        t_loc = t * G - idx.float()

        orig = h.shape
        idx_f = idx.reshape(-1, idx.shape[-1])
        B = idx_f.shape[0]
        c0 = self.ctrl[:, :G].unsqueeze(0).expand(B, -1, -1)
        c1 = self.ctrl[:, 1:].unsqueeze(0).expand(B, -1, -1)
        g = idx_f.unsqueeze(-1)
        a = c0.gather(2, g).squeeze(-1).reshape(orig)
        b = c1.gather(2, g).squeeze(-1).reshape(orig)
        return (a * (1 - t_loc) + b * t_loc) * scale


class EncBSplineSAE(nn.Module):
    """KAN-SAE as described in the paper: spline replaces ReLU in the ENCODER.

    z = TopK( phi(W_enc (x - b_pre) + b_enc) ),   x_hat = W_dec z + b_pre
    """

    def __init__(self, d_model: int, dict_size: int, k: int, grid_size: int = 16):
        super().__init__()
        self.d_model, self.dict_size, self.k = d_model, dict_size, k
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()
        self.phi = _EncBSplinePhi(dict_size, grid_size)

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = (self.W_enc @ (x - self.b_pre).T).T + self.b_enc
        return topk_activation(self.phi(h), self.k)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor):
        x_hat, f = self(x)
        return (x - x_hat).pow(2).sum(-1).mean(), (f > 0).float().sum(-1).mean()


class _JumpReLU(torch.autograd.Function):
    """JumpReLU with a rectangle straight-through estimator for the threshold.

    Forward:  h * 1[h > theta]
    Backward: dL/dh    passes through where h > theta
              dL/dtheta uses the rectangle kernel of Rajamanoharan et al. (2024)
    """

    @staticmethod
    def forward(ctx, h, theta, bandwidth):
        ctx.save_for_backward(h, theta)
        ctx.bandwidth = bandwidth
        return h * (h > theta).to(h.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        h, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        gate = (h > theta).to(h.dtype)
        grad_h = grad_out * gate
        rect = ((h - theta).abs() < bw * 0.5).to(h.dtype) / bw
        grad_theta = -(theta / bw) * rect * grad_out * bw   # = -theta * rect * grad_out
        grad_theta = grad_theta.reshape(-1, theta.shape[0]).sum(0)
        return grad_h, grad_theta, None


class _HeavisideSTE(torch.autograd.Function):
    """1[h > theta] with the same rectangle STE, used for the L0 penalty."""

    @staticmethod
    def forward(ctx, h, theta, bandwidth):
        ctx.save_for_backward(h, theta)
        ctx.bandwidth = bandwidth
        return (h > theta).to(h.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        h, theta = ctx.saved_tensors
        bw = ctx.bandwidth
        rect = ((h - theta).abs() < bw * 0.5).to(h.dtype) / bw
        grad_theta = (-rect * grad_out).reshape(-1, theta.shape[0]).sum(0)
        return None, grad_theta, None


class JumpReLUSAE(nn.Module):
    """Standard JumpReLU SAE: per-feature learnable threshold, L0 penalty.

        z = h * 1[h > theta],   h = W_enc (x - b_pre) + b_enc
        L = ||x - x_hat||^2 + l0_coeff * ||z||_0

    No TopK.  Stacking JumpReLU under TopK is degenerate: the selected entries
    sit far above theta, so the rectangle STE never fires and theta receives no
    gradient.  Sparsity here is therefore controlled by l0_coeff, which also
    gives the (L0, EV) frontier that a fixed-K model cannot trace.
    """

    def __init__(self, d_model: int, dict_size: int, k: int,
                 bandwidth: float = 0.5, l0_coeff: float = 1.0):
        super().__init__()
        self.d_model, self.dict_size, self.k = d_model, dict_size, k
        self.bandwidth, self.l0_coeff = bandwidth, l0_coeff
        self.b_pre = nn.Parameter(torch.zeros(d_model))
        self.W_enc = nn.Parameter(torch.empty(dict_size, d_model))
        self.b_enc = nn.Parameter(torch.zeros(dict_size))
        self.W_dec = nn.Parameter(torch.empty(d_model, dict_size))
        self.log_theta = nn.Parameter(torch.full((dict_size,), math.log(0.1)))
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        self.W_dec.data = F.normalize(self.W_dec.data, dim=0)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = (self.W_enc @ (x - self.b_pre).T).T + self.b_enc
        return _JumpReLU.apply(h, self.log_theta.exp(), self.bandwidth)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec.T + self.b_pre

    def forward(self, x: torch.Tensor):
        f = self.encode(x)
        return self.decode(f), f

    def loss(self, x: torch.Tensor):
        h = (self.W_enc @ (x - self.b_pre).T).T + self.b_enc
        theta = self.log_theta.exp()
        f = _JumpReLU.apply(h, theta, self.bandwidth)
        x_hat = self.decode(f)
        mse = (x - x_hat).pow(2).sum(-1).mean()
        l0 = _HeavisideSTE.apply(h, theta, self.bandwidth).sum(-1).mean()
        self.last_mse = mse.detach()
        return mse + self.l0_coeff * l0, l0



class _NoClampBSplinePhi(_BSplinePhi):
    """B-spline gain with the saturation removed (linear extrapolation above s).

    Isolates the mechanism hypothesis: if what suppresses dead features is the
    gain saturation imposed by clipping the code to [0, s] -- rather than
    nonlinearity as such -- then removing the clip should restore the collapse
    that Lin-SAE and the Fourier variant both show.
    """

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        G = self.grid_size
        scale = self.log_scale.exp().clamp(min=0.1)
        t_raw = f / scale                       # NOT clipped
        t = t_raw.clamp(0.0, 1.0)
        idx = (t * G).long().clamp(0, G - 1)
        t_local = t * G - idx.float()
        orig = f.shape
        idx_f = idx.reshape(-1, idx.shape[-1])
        B = idx_f.shape[0]
        c0 = self.ctrl[:, :G].unsqueeze(0).expand(B, -1, -1)
        c1 = self.ctrl[:, 1:].unsqueeze(0).expand(B, -1, -1)
        g = idx_f.clamp(0, G - 1).unsqueeze(-1)
        a = c0.gather(2, g).squeeze(-1).reshape(orig)
        b = c1.gather(2, g).squeeze(-1).reshape(orig)
        phi = (a * (1 - t_local) + b * t_local) * scale
        # linear extrapolation beyond the last knot, using the final segment slope
        slope = (self.ctrl[:, G] - self.ctrl[:, G - 1]) * G
        over = (t_raw - 1.0).clamp(min=0.0) * scale
        return phi + slope.unsqueeze(0).expand(B, -1).reshape(orig) * over


class NoClampBSplineSAE(BSplineKANSAE):
    """BSplineKANSAE with the gain saturation removed."""

    def __init__(self, d_model: int, dict_size: int, k: int, grid_size: int = 8):
        super().__init__(d_model, dict_size, k, grid_size)
        self.phi = _NoClampBSplinePhi(dict_size, grid_size)
