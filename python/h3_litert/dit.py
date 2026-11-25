"""The H3 Omni-Transformer as PyTorch modules, cut as `docs/graphs.md` says.
`H3Dit` composes them into the uncut forward the tests compare the graphs
against (and `tests/test_parity.py` compares to Diffusers' model)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import H3Config

SHIFT_MSA, SCALE_MSA, GATE_MSA, SHIFT_MLP, SCALE_MLP, GATE_MLP = range(6)


def rms_norm(x: torch.Tensor, w: torch.Tensor | None, eps: float) -> torch.Tensor:
    y = x * torch.rsqrt((x * x).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def rotate_half(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, rot: int) -> torch.Tensor:
    """`x [rows, heads, hd]`, `cos/sin [rows, rot/2]`: rotate-half on the first `rot` channels."""
    half = rot // 2
    xr, xp = x[..., :rot], x[..., rot:]
    x1, x2 = xr[..., :half], xr[..., half:]
    c, s = cos[:, None, :], sin[:, None, :]
    y = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], dim=-1)
    return torch.cat([y, xp], dim=-1)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """`[rows, G, hd]` tensors → `[rows_q, G·hd]`; `mask [rows_k]` additive."""
    hd = q.shape[-1]
    qh, kh, vh = q.permute(1, 0, 2), k.permute(1, 0, 2), v.permute(1, 0, 2)
    s = torch.matmul(qh, kh.transpose(1, 2)) * (1.0 / math.sqrt(hd))
    if mask is not None:
        s = s + mask[None, None, :]
    p = torch.softmax(s, dim=-1)
    return torch.matmul(p, vh).permute(1, 0, 2).reshape(q.shape[0], -1)


class AttnFf(nn.Module):
    """One attention + SwiGLU unit without its modulation (a refiner block, or a transformer block's weights)."""

    def __init__(self, cfg: H3Config):
        super().__init__()
        d, inner = cfg.hidden, cfg.dim
        self.cfg = cfg
        self.to_q = nn.Linear(d, inner, bias=False)
        self.to_k = nn.Linear(d, inner, bias=False)
        self.to_v = nn.Linear(d, inner, bias=False)
        self.to_out = nn.Linear(inner, d, bias=False)
        self.norm_q = nn.Parameter(torch.ones(cfg.head_dim))
        self.norm_k = nn.Parameter(torch.ones(cfg.head_dim))
        self.norm1 = nn.Parameter(torch.ones(d))
        self.norm2 = nn.Parameter(torch.ones(d))
        self.ff_in = nn.Linear(d, 2 * cfg.ffn_dim, bias=False)
        self.ff_out = nn.Linear(cfg.ffn_dim, d, bias=False)

    def qkv(self, n: torch.Tensor, cos: torch.Tensor | None, sin: torch.Tensor | None):
        c = self.cfg
        rows = n.shape[0]
        q = rms_norm(self.to_q(n).reshape(rows, c.heads, c.head_dim), self.norm_q, c.qk_norm_eps)
        k = rms_norm(self.to_k(n).reshape(rows, c.heads, c.head_dim), self.norm_k, c.qk_norm_eps)
        v = self.to_v(n).reshape(rows, c.heads, c.head_dim)
        if cos is not None:
            q = rotate_half(q, cos, sin, c.rotary_dim())
            k = rotate_half(k, cos, sin, c.rotary_dim())
        return q, k, v

    def ff(self, n: torch.Tensor) -> torch.Tensor:
        value, gate = self.ff_in(n).chunk(2, dim=-1)
        return self.ff_out(value * F.silu(gate))


class TextRefine(nn.Module):
    """`text_refine_T`: context embedder, refiner blocks (masked attention), final norm."""

    def __init__(self, cfg: H3Config):
        super().__init__()
        self.cfg = cfg
        self.context_embedder = nn.Linear(cfg.text_dim, cfg.hidden)
        self.blocks = nn.ModuleList(AttnFf(cfg) for _ in range(cfg.refiner_layers))
        self.final_norm = nn.Parameter(torch.ones(cfg.hidden))

    def forward(self, embeds: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        h = self.context_embedder(embeds)
        for b in self.blocks:
            q, k, v = b.qkv(rms_norm(h, b.norm1, c.norm_eps), None, None)
            h = h + b.to_out(attention(q, k, v, mask))
            h = h + b.ff(rms_norm(h, b.norm2, c.norm_eps))
        return rms_norm(h, self.final_norm, c.final_norm_eps)


class Embed(nn.Module):
    """`embed_video` / `embed_audio`."""

    def __init__(self, cfg: H3Config, features: int):
        super().__init__()
        self.proj = nn.Linear(features, cfg.hidden)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.proj(rows)


class BlockQkv(nn.Module):
    def __init__(self, unit: AttnFf):
        super().__init__()
        self.unit = unit

    def forward(self, h: torch.Tensor, mods: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        u = self.unit
        n = rms_norm(h, u.norm1, u.cfg.norm_eps) * (1 + mods[SCALE_MSA]) + mods[SHIFT_MSA]
        return u.qkv(n, cos, sin)


class Attention(nn.Module):
    """`attn_sS_qCq_hG`: dense attention over one head group with an additive key mask."""

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return attention(q, k, v, mask)


class BlockPost(nn.Module):
    def __init__(self, unit: AttnFf):
        super().__init__()
        self.unit = unit

    def forward(self, h: torch.Tensor, o: torch.Tensor, mods: torch.Tensor) -> torch.Tensor:
        u = self.unit
        h = h + u.to_out(o) * mods[GATE_MSA]
        n = rms_norm(h, u.norm2, u.cfg.norm_eps) * (1 + mods[SCALE_MLP]) + mods[SHIFT_MLP]
        return h + u.ff(n) * mods[GATE_MLP]


class Head(nn.Module):
    """`head_video` / `head_audio`: modulated final RMSNorm, then the projection
    (or the `pdd_steps` fused projections with per-step weights)."""

    def __init__(self, cfg: H3Config, out: int, norm_w: nn.Parameter):
        super().__init__()
        self.cfg, self.out = cfg, out
        self.norm_w = norm_w
        self.pdd = cfg.pdd_steps
        self.proj = nn.Linear(cfg.hidden, out * (self.pdd or 1))

    def forward(self, h: torch.Tensor, mods: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
        n = rms_norm(h, self.norm_w, self.cfg.final_norm_eps) * (1 + mods[1]) + mods[0]
        y = self.proj(n)
        if self.pdd:
            y = (y.reshape(h.shape[0], self.pdd, self.out) * weights[None, :, None]).sum(1)
        return y


class TimeEmbed(nn.Module):
    """`time_embedder` (+ `adaln_basis`): what the exporter feeds to the AdaLN projections."""

    def __init__(self, cfg: H3Config):
        super().__init__()
        self.cfg = cfg
        self.linear_1 = nn.Linear(cfg.freq_dim, cfg.time_embed_hidden)
        self.linear_2 = nn.Linear(cfg.time_embed_hidden, cfg.time_embed_dim)
        self.basis = nn.Parameter(torch.zeros(cfg.adaln_rank, cfg.time_embed_dim)) if cfg.adaln_rank else None

    def temb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.cfg.freq_dim // 2
        f = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32) / half)
        a = t.reshape(-1, 1) * f[None]
        emb = torch.cat([torch.cos(a), torch.sin(a)], dim=-1)
        return self.linear_2(F.silu(self.linear_1(emb)))

    def adaln_input(self, t: torch.Tensor) -> torch.Tensor:
        s = F.silu(self.temb(t))
        return s @ self.basis.T if self.basis is not None else s


class H3Dit(nn.Module):
    """The whole DiT, for the uncut reference forward and for the exporter."""

    def __init__(self, cfg: H3Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden
        self.time = TimeEmbed(cfg)
        self.text = TextRefine(cfg)
        self.embed_video = Embed(cfg, cfg.video_patch_dim())
        self.embed_audio = Embed(cfg, cfg.audio_in_channels)
        self.blocks = nn.ModuleList(AttnFf(cfg) for _ in range(cfg.layers))
        self.adaln = nn.ModuleList(nn.Linear(cfg.adaln_in(), 18 * d) for _ in range(cfg.layers))
        self.norm_out = nn.Parameter(torch.ones(d))
        self.norm_out_linear = nn.Linear(cfg.adaln_in(), 2 * d)
        self.head_video = Head(cfg, cfg.video_patch_dim(), self.norm_out)
        self.head_audio = Head(cfg, cfg.audio_in_channels, self.norm_out)

    def qkv_graph(self, l: int) -> BlockQkv:
        return BlockQkv(self.blocks[l])

    def post_graph(self, l: int) -> BlockPost:
        return BlockPost(self.blocks[l])

    @torch.no_grad()
    def adaln_tables(self, timesteps: torch.Tensor):
        """`blocks [U, layers, 3, 6, d]`, `head [U, 2, d]` for clean times `timesteps [U]`."""
        x = self.time.adaln_input(timesteps)  # [U, adaln_in]
        d = self.cfg.hidden
        blocks = torch.stack([lin(x).reshape(-1, 3, 6, d) for lin in self.adaln], dim=1)
        head = self.norm_out_linear(x).reshape(-1, 2, d)
        return blocks, head

    def forward(self, video_rows, audio_rows, text, unique_t, t_idx, tags, positions, video_idx, audio_idx, text_idx,
                pdd_weights: tuple[torch.Tensor, torch.Tensor] | None = None):
        """Uncut forward over a packed layout (all index tensors int64, positions `[S, 3]` f32)."""
        from .rope import h3_rope

        c = self.cfg
        s, d = positions.shape[0], c.hidden
        cos, sin = (torch.from_numpy(a) for a in h3_rope(positions.numpy(), c.rope_freq_dim, c.rope_theta))
        blocks_tab, head_tab = self.adaln_tables(unique_t)
        idx = t_idx * 3 + tags
        txt = self.text(text, torch.zeros(text.shape[0]))
        h = torch.zeros(s, d)
        h[text_idx] = txt
        h[video_idx] = self.embed_video(video_rows)
        h[audio_idx] = self.embed_audio(audio_rows)
        for l, u in enumerate(self.blocks):
            tab = blocks_tab[:, l].reshape(-1, 6, d)[idx]  # [S, 6, d]
            n = rms_norm(h, u.norm1, c.norm_eps) * (1 + tab[:, SCALE_MSA]) + tab[:, SHIFT_MSA]
            q, k, v = u.qkv(n, cos, sin)
            h = h + u.to_out(attention(q, k, v)) * tab[:, GATE_MSA]
            n = rms_norm(h, u.norm2, c.norm_eps) * (1 + tab[:, SCALE_MLP]) + tab[:, SHIFT_MLP]
            h = h + u.ff(n) * tab[:, GATE_MLP]
        ht = head_tab[t_idx]  # [S, 2, d]
        n = rms_norm(h, self.norm_out, c.final_norm_eps) * (1 + ht[:, 1]) + ht[:, 0]
        wv, wa = pdd_weights if pdd_weights else (None, None)
        vid = self._head(self.head_video, n[video_idx], wv)
        aud = self._head(self.head_audio, n[audio_idx], wa)
        return vid, aud

    @staticmethod
    def _head(head: Head, n: torch.Tensor, w):
        y = head.proj(n)
        if head.pdd:
            y = (y.reshape(n.shape[0], head.pdd, head.out) * w[None, :, None]).sum(1)
        return y
