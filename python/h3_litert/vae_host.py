"""The video VAE's host side (h3-rs `vae.rs`, Diffusers `AutoencoderKLMiniMaxH3`):
spatial tiles and their stitch, the temporal chunk plan and frame blend, the
posterior, pixel and latent normalisation. Pure numpy; graphs are called
through a callback so the reference (PyTorch) and the LiteRT runtime share it."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .config import PIXEL_MEAN, PIXEL_STD, VideoVaeConfig

TILE, OVERLAP = 256, 64


def split_tiles(length: int, tile: int, min_overlap: int, ratio: int) -> tuple[list[int], list[int], list[int]]:
    """`_split_tiles`: starts, lengths, overlaps along one pixel axis."""
    if tile >= length:
        return [0], [length], []
    n = -(-length // tile)
    while tile * n - min_overlap * (n - 1) - length < 0:
        n += 1
    overlaps = [min_overlap] * (n - 1)
    remaining = tile * n - sum(overlaps) - length
    for i in range(remaining // ratio):
        overlaps[i % (n - 1)] += ratio
    starts = [0]
    for i in range(n - 1):
        starts.append(starts[-1] + tile - overlaps[i])
    return starts, [tile] * n, overlaps


def blend(a: np.ndarray, b: np.ndarray, extent: int, axis: int) -> np.ndarray:
    """Linear cross-fade of `a`'s last `extent` with `b`'s first `extent` along `axis`, then the rest of `b`."""
    extent = min(a.shape[axis], b.shape[axis], extent)
    if extent == 0:
        return b
    w = (np.arange(extent, dtype=np.float32) / np.float32(extent))
    shape = [1] * a.ndim
    shape[axis] = extent
    wa, wb = (1 - w).reshape(shape), w.reshape(shape)
    sa = [slice(None)] * a.ndim
    sa[axis] = slice(a.shape[axis] - extent, None)
    sb = [slice(None)] * b.ndim
    sb[axis] = slice(0, extent)
    out = a[tuple(sa)] * wa + b[tuple(sb)] * wb
    if extent == b.shape[axis]:
        return out.astype(np.float32)
    rest = [slice(None)] * b.ndim
    rest[axis] = slice(extent, None)
    return np.concatenate([out, b[tuple(rest)]], axis=axis).astype(np.float32)


def stitch_tiles(tiles: list[list[np.ndarray]], h_overlaps: list[int], w_overlaps: list[int]) -> np.ndarray:
    rows = []
    for i, row in enumerate(tiles):
        out_row = []
        for j, tile in enumerate(row):
            if i > 0:
                tile = blend(tiles[i - 1][j], tile, h_overlaps[i - 1], -2)
            if j > 0:
                tile = blend(row[j - 1], tile, w_overlaps[j - 1], -1)
            if i < len(tiles) - 1:
                tile = tile[..., : tile.shape[-2] - h_overlaps[i], :]
            if j < len(row) - 1:
                tile = tile[..., :, : tile.shape[-1] - w_overlaps[j]]
            out_row.append(tile)
        rows.append(np.concatenate(out_row, axis=-1))
    return np.concatenate(rows, axis=-2)


def tiled(x: np.ndarray, ratio_in: int, ratio_out: int, tile_px: int, overlap_px: int, run: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """Run `run` on tiles of `x [C, T, H, W]` laid out in pixel space (`H·ratio_in` pixels), stitch in the
    output's units (`ratio_out` pixels per output cell)."""
    hp, wp = x.shape[-2] * ratio_in, x.shape[-1] * ratio_in
    ys, yl, yo = split_tiles(hp, tile_px, overlap_px, 16)
    xs, xl, xo = split_tiles(wp, tile_px, overlap_px, 16)
    rows = []
    for y0, yn in zip(ys, yl):
        row = []
        for x0, xn in zip(xs, xl):
            t = x[..., y0 // ratio_in : (y0 + yn) // ratio_in, x0 // ratio_in : (x0 + xn) // ratio_in]
            row.append(run(np.ascontiguousarray(t)))
        rows.append(row)
    if len(rows) == 1 and len(rows[0]) == 1:
        return rows[0][0]
    return stitch_tiles(rows, [o // ratio_out for o in yo], [o // ratio_out for o in xo])


def temporal_plan(cfg: VideoVaeConfig, latent_frames: int) -> tuple[int, int, int]:
    """`(pad latent frames, chunks, output frames)` of `_decode`."""
    if latent_frames == 0:
        raise ValueError("no latent frames to decode")
    drop, chunk, tr = cfg.token_drop, cfg.tokens_chunk(), cfg.temporal_ratio()
    n = latent_frames + drop
    pad = (chunk - n % chunk) % chunk
    chunks = (n + pad) // chunk - int(drop > 0)
    if chunks < 1:
        pad += chunk
        chunks = 1
    frames = chunks * (chunk * tr - cfg.frame_pre_padding())
    if drop > 0:
        frames += cfg.frame_overlap()
    if pad > 0:
        intra = cfg.clip_length % tr
        pad_frames = sum(intra if intra and (latent_frames + off) % chunk == 0 else tr for off in range(pad))
        frames -= pad_frames
    if frames <= 0:
        raise ValueError(f"the decode plan yields no frames for {latent_frames} latent frames")
    return pad, chunks, frames


def decode(cfg: VideoVaeConfig, z: np.ndarray, decode_clip: Callable[[np.ndarray], np.ndarray], on_chunk=None) -> np.ndarray:
    """Denormalised latents `[lz, T, h, w]` → pixels `[3, F, H, W]` in `[0, 1]`; `decode_clip` decodes one
    `[lz, 7, h, w]` (already tiled by the caller or not) into ImageNet-normalised `[3, 28, 16h, 16w]`."""
    pad, chunks, total = temporal_plan(cfg, z.shape[1])
    if pad:
        z = np.concatenate([z, np.repeat(z[:, -1:], pad, axis=1)], axis=1)
    chunk_tok, tr, pre = cfg.tokens_chunk(), cfg.temporal_ratio(), cfg.frame_pre_padding()
    chunk_frames = chunk_tok * tr
    out, written, overlap = [], 0, None
    for ci in range(chunks):
        start = ci * chunk_tok
        n = min(chunk_tok + cfg.token_overlap(), z.shape[1] - start)
        clip = decode_clip(np.ascontiguousarray(z[:, start : start + n]))  # [3, 4n, H, W]
        chunk = clip[:, pre:chunk_frames]
        if overlap is not None:
            chunk = blend(overlap, chunk, cfg.frame_overlap(), 1)
        overlap = clip[:, chunk_frames + pre :] if cfg.token_drop > 0 and clip.shape[1] > chunk_frames + pre else None
        keep = min(chunk.shape[1], total - written)
        px = denormalize_pixels(chunk[:, :keep])
        if keep > 0:
            out.append(px)
            written += keep
            if on_chunk:
                on_chunk(px)
    if overlap is not None and written < total:
        keep = min(total - written, overlap.shape[1])
        px = denormalize_pixels(overlap[:, :keep])
        out.append(px)
        written += keep
        if on_chunk:
            on_chunk(px)
    if written != total:
        raise ValueError(f"decoded {written} frames, the plan expected {total}")
    return np.concatenate(out, axis=1)


def denormalize_pixels(x: np.ndarray) -> np.ndarray:
    m = np.array(PIXEL_MEAN, np.float32).reshape(3, 1, 1, 1)
    s = np.array(PIXEL_STD, np.float32).reshape(3, 1, 1, 1)
    return np.clip(x * s + m, 0.0, 1.0).astype(np.float32)


def normalize_pixels(x: np.ndarray) -> np.ndarray:
    m = np.array(PIXEL_MEAN, np.float32).reshape(3, 1, 1, 1)
    s = np.array(PIXEL_STD, np.float32).reshape(3, 1, 1, 1)
    return ((x - m) / s).astype(np.float32)


def normalize_latents(cfg: VideoVaeConfig, z: np.ndarray) -> np.ndarray:
    m = np.array(cfg.latents_mean, np.float32).reshape(-1, 1, 1, 1)
    s = np.array(cfg.latents_std, np.float32).reshape(-1, 1, 1, 1)
    return ((z - m) / s).astype(np.float32)


def denormalize_latents(cfg: VideoVaeConfig, z: np.ndarray) -> np.ndarray:
    m = np.array(cfg.latents_mean, np.float32).reshape(-1, 1, 1, 1)
    s = np.array(cfg.latents_std, np.float32).reshape(-1, 1, 1, 1)
    return (z * s + m).astype(np.float32)


def posterior_sample(moments: np.ndarray, noise: np.ndarray) -> np.ndarray:
    """`mean + exp(logvar/2)·ε` from `[2·lz, …]` moments (logvar clamped to [−30, 20])."""
    lz = moments.shape[0] // 2
    mean, logvar = moments[:lz], np.clip(moments[lz:], -30.0, 20.0)
    return (mean + np.exp(0.5 * logvar) * noise).astype(np.float32)


def encode(cfg: VideoVaeConfig, pixels: np.ndarray, encode_clip: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """ImageNet-normalised pixels `[3, F, H, W]` → moments `[2·lz, T, h, w]` (`_encode`: 17-frame clips, the trailing
    `token_drop` latent frames dropped; a single frame goes through alone)."""
    f = pixels.shape[1]
    if f == 1:
        return encode_clip(pixels)
    cl = cfg.clip_length
    if f % cl:
        pixels = np.concatenate([pixels, np.repeat(pixels[:, -1:], (-f) % cl, axis=1)], axis=1)
    moments = np.concatenate([encode_clip(np.ascontiguousarray(pixels[:, i * cl : (i + 1) * cl])) for i in range(pixels.shape[1] // cl)], axis=1)
    if cfg.token_drop > 0:
        moments = moments[:, : -cfg.token_drop]
    return moments
