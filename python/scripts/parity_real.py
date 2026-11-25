"""Graph pipeline against the uncut PyTorch model at real weights (T2VA): the
per-forward error on the reference trajectory, the end-to-end latent / audio
drift, and the decoders on the reference latents. `docs/runs/` records the numbers.

    python scripts/parity_real.py --variant fasth3-v2 --model-dir /workspace/weights/h3-8step \
        --graphs graphs/h3-f16 --prompt gallery-h3/fox.safetensors --size 256x256x22
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from h3_litert import vae_host
from h3_litert.audio import AudioDecoder
from h3_litert.cli import _load_prompt
from h3_litert.config import AudioVaeConfig, VideoVaeConfig
from h3_litert.dit import H3Dit
from h3_litert.packing import Geometry, row_timesteps, unpatch_rows
from h3_litert.pipeline import Pipeline
from h3_litert.request import Request
from h3_litert.rng import Rng
from h3_litert.runtime import GraphStore, Manifest
from h3_litert.sampler import flow_step, needs_noise
from h3_litert.vae import Decoder
from h3_litert.variants import by_name
from h3_litert.weights import SafetensorsDir, load_audio_decoder, load_dit, load_vae_decoder


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fasth3-v2")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--base-dir")
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--size", default="256x256x22")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--accelerator", default="cpu")
    ap.add_argument("--out")
    a = ap.parse_args()

    variant = by_name(a.variant)
    root = Path(a.model_dir)
    tdir = root / "transformer" if (root / "transformer").is_dir() else root
    base = Path(a.base_dir) if a.base_dir else root
    vae_cfg = VideoVaeConfig.fasth3_trim() if variant.name == "fasth3-trim" else VideoVaeConfig.h3()
    audio_cfg = AudioVaeConfig.h3()
    emb, tags = _load_prompt(a.prompt)
    g = Geometry.parse(a.size)
    req = Request(emb, tags, g, a.seed)

    m = Manifest(a.graphs)
    pipe = Pipeline(GraphStore(m, accelerator=a.accelerator))
    sampler = m.sampler
    rule = sampler["rule"]
    assert rule != "pdd", "this script covers the Euler / renoise samplers"

    # ---- the graph pipeline on its own ---------------------------------------------------------
    t0 = time.time()
    lat_g, aud_g = pipe.denoise(req)
    t_den_g = time.time() - t0
    print(f"graphs: denoise {t_den_g:.1f} s ({len(sampler['steps'])} forwards)", flush=True)

    # ---- the PyTorch model, the graphs fed its rows at every step ----------------------------------
    t0 = time.time()
    src = SafetensorsDir(tdir)
    dit = H3Dit(variant.config).eval()
    load_dit(dit, src)
    print(f"torch: DiT loaded in {time.time() - t0:.0f} s", flush=True)
    c = dit.cfg
    ratio = vae_cfg.spatial_ratio()
    lt, lh, lw = g.latent(ratio)
    na = g.audio_latents()
    p = pipe.prepare(req, pipe.encoder_tile_runner())
    attn = m.attention_for(p.layout.sequence_length + p.pad)
    layout = p.layout
    video_rows, audio_rows = p.video_rows.copy(), p.audio_rows.copy()  # T2VA: the same draws as reference.denoise
    renoise = Rng(a.seed + 1)
    off_v = layout.num_condition_video_rows
    L = lambda x: torch.tensor(x, dtype=torch.long)  # noqa: E731
    pos = torch.tensor(layout.position_ids, dtype=torch.float32)
    emb_t = torch.from_numpy(np.ascontiguousarray(emb, dtype=np.float32))
    step_err_v, step_err_a, t_fwd_r, t_fwd_g = [], [], [], []
    with torch.no_grad():
        for st in sampler["steps"]:
            unique, inv = row_timesteps(layout, st["t_video"], st["t_audio"], max(st["t_video"], sampler["cond_video_t"]), sampler["cond_audio_t"])
            t0 = time.time()
            v, aa = dit(torch.from_numpy(video_rows), torch.from_numpy(audio_rows), emb_t, torch.tensor(unique), L(inv), L(layout.token_tags), pos,
                        L(layout.video_indices), L(layout.audio_indices), L(layout.text_indices), None)
            v, aa = v.numpy(), aa.numpy()
            t_fwd_r.append(time.time() - t0)
            p.video_rows[:] = video_rows
            p.audio_rows[:] = audio_rows
            t0 = time.time()
            v_g, a_g = pipe.forward(p, st, attn)
            t_fwd_g.append(time.time() - t0)
            step_err_v.append(rel(v_g, v))
            step_err_a.append(rel(a_g, aa))
            print(f"  t_video {st['t_video']:.1f}: video rel {step_err_v[-1]:.4f}, audio rel {step_err_a[-1]:.4f}  (torch {t_fwd_r[-1]:.1f} s, graphs {t_fwd_g[-1]:.1f} s)", flush=True)
            nz = renoise.normal(video_rows[off_v:].size).reshape(video_rows[off_v:].shape) if needs_noise(rule, st["sigma_video_next"]) else None
            video_rows[off_v:] = flow_step(video_rows[off_v:], v[off_v:], st["sigma_video"], st["sigma_video_next"], rule, nz)
            nz = renoise.normal(audio_rows.size).reshape(audio_rows.shape) if needs_noise(rule, st["sigma_audio_next"]) else None
            audio_rows[:] = flow_step(audio_rows, aa, st["sigma_audio"], st["sigma_audio_next"], rule, nz)
    lat_r = unpatch_rows(video_rows[off_v:], c.in_channels, lt, lh, lw, c.patch)
    aud_r = audio_rows
    del dit, src

    # ---- decoders on the reference latents ------------------------------------------------------
    bsrc = SafetensorsDir(base / "vae")
    dec = Decoder(vae_cfg).eval()
    load_vae_decoder(dec, bsrc)
    with torch.no_grad():
        tile_run = lambda z: dec(torch.from_numpy(z)).numpy()  # noqa: E731
        t0 = time.time()
        frames = vae_host.decode(vae_cfg, vae_host.denormalize_latents(vae_cfg, lat_r), lambda z: vae_host.tiled(z, ratio, 1, vae_host.TILE, vae_host.OVERLAP, tile_run))
        t_dec_r = time.time() - t0
    rgb_r = (np.clip(frames, 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 3, 0)
    t0 = time.time()
    rgb_g = pipe.decode_video(lat_r)
    t_dec_g = time.time() - t0
    adec = AudioDecoder(audio_cfg).eval()
    load_audio_decoder(adec, SafetensorsDir(base / "audio_vae"))
    mean = np.array(audio_cfg.latents_mean, np.float32).reshape(-1, 1)
    std = np.array(audio_cfg.latents_std, np.float32).reshape(-1, 1)
    with torch.no_grad():
        wave_r = np.stack([adec(torch.from_numpy(np.ascontiguousarray(aud_r[ch * na : (ch + 1) * na].T * std + mean))).numpy() for ch in range(2)]).astype(np.float32)
    wave_g = np.asarray(pipe.decode_audio(aud_r, na), np.float32).reshape(wave_r.shape)
    dec_diff = np.abs(rgb_g.astype(np.int16) - rgb_r.astype(np.int16))
    rep = {
        "variant": variant.name, "size": a.size, "seed": a.seed, "weights": m.raw.get("weights"), "accelerator": a.accelerator,
        "rows": int(layout.sequence_length), "forwards": len(sampler["steps"]),
        "step_video_rel_err": step_err_v, "step_audio_rel_err": step_err_a,
        "latent_rel_err_end_to_end": rel(lat_g, lat_r), "audio_rows_rel_err_end_to_end": rel(aud_g, aud_r),
        "vae_decode_max_abs_8bit": int(dec_diff.max()), "vae_decode_mean_abs_8bit": float(dec_diff.mean()),
        "audio_wave_max_abs": float(np.abs(wave_g - wave_r).max()), "audio_wave_rel_err": rel(wave_g, wave_r),
        "torch_forward_s": t_fwd_r, "graphs_forward_s": t_fwd_g, "graphs_denoise_s": t_den_g, "torch_vae_decode_s": t_dec_r, "graphs_vae_decode_s": t_dec_g,
        "latent_shape": list(lat_r.shape), "frames": list(rgb_r.shape), "audio_samples": int(wave_r.shape[-1]),
    }
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
