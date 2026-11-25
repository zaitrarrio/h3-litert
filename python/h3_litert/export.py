"""Export: checkpoint (or synthetic weights) → the graph directory of
`docs/graphs.md`, the AdaLN table and the manifest."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .audio import AudioDecoder, AudioEncoder
from .config import AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig
from .dit import Attention, H3Dit
from .packing import Geometry
from .runtime import FORMAT
from .tae import TaeH3
from .vae import DecIn, DecOut, Decoder, Encoder, EncoderGraph
from .variants import Variant
from .vae_host import OVERLAP, TILE
from .weights import (Synthetic, WeightSource, load_audio_decoder, load_audio_encoder, load_dit, load_tae,
                      load_vae_decoder, load_vae_encoder)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
WEIGHT_FORMATS = ("f32", "f16", "int8", "int4")
TEXT_BUCKETS = (128, 256, 512, 1024)


def _convert(module: torch.nn.Module, kwargs: dict, out: Path, f16: bool) -> None:
    import litert_torch

    flags = None
    if f16:
        import tensorflow as tf

        # nested keys: litert-torch walks the dict (`converter.target_spec.supported_types`); a dotted
        # key sets a stray attribute and `Optimize.DEFAULT` alone quantizes to dynamic-range int8
        flags = {"optimizations": [tf.lite.Optimize.DEFAULT], "target_spec": {"supported_types": [tf.float16]}}
    edge = litert_torch.convert(module.eval(), sample_kwargs=kwargs, strict_export=True, _ai_edge_converter_flags=flags)
    edge.export(str(out))


def _block_job(args) -> dict:
    """Worker: export one block's `qkv` and `post` graphs (and its AdaLN rows) from a fresh source."""
    cfg_json, src_spec, l, out_dir, chunk, weights, ts = args
    import warnings

    warnings.filterwarnings("ignore")
    from .dit import AttnFf, BlockPost, BlockQkv
    from .weights import load_adaln, load_attn_ff, open_source

    cfg = H3Config.from_json(cfg_json)
    src = open_source(src_spec)
    out_dir = Path(out_dir)
    d, inner = cfg.hidden, cfg.dim
    u = AttnFf(cfg).eval()
    load_attn_ff(u, src, f"transformer_blocks.{l}")
    f16_block = weights == "f16"
    entry = {}
    for kind, module, kw in (
        ("qkv", BlockQkv(u), {"h": torch.randn(chunk, d), "mods": torch.randn(6, d), "cos": torch.rand(chunk, cfg.rotary_dim() // 2), "sin": torch.rand(chunk, cfg.rotary_dim() // 2)}),
        ("post", BlockPost(u), {"h": torch.randn(chunk, d), "o": torch.randn(chunk, inner), "mods": torch.randn(6, d)}),
    ):
        name = f"block_{l:02d}_{kind}.tflite"
        t0 = time.time()
        _convert(module, kw, out_dir / name, f16_block)
        if weights in ("int8", "int4"):
            _quantize(out_dir / name, weights)
        entry[kind] = name
        entry[f"{kind}_secs"] = round(time.time() - t0, 1)
    # The block's AdaLN rows for the ladder's timesteps: [U, 3, 6, d] (the input x is tiny; the linear is 1 GB).
    lin = torch.nn.Linear(cfg.adaln_in(), 18 * d)
    load_adaln(lin, src, cfg, l)
    x = torch.tensor(ts["x"], dtype=torch.float32)  # [U, adaln_in]
    with torch.no_grad():
        rows = lin(x).reshape(-1, 3, 6, d).numpy().astype(np.float16)
    np.save(out_dir / f".adaln_{l:02d}.npy", rows)
    return {"l": l, **entry}


def _quantize(path: Path, weights: str) -> None:
    from ai_edge_quantizer import quantizer, recipe

    rec = recipe.weight_only_wi8_afp32() if weights == "int8" else recipe.weight_only_wi4_afp32()
    res = quantizer.Quantizer(str(path), rec).quantize()
    tmp = path.with_suffix(".q.tflite")
    res.export_model(str(tmp))
    os.replace(tmp, path)


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class TaeGraph(torch.nn.Module):
    def __init__(self, tae: TaeH3):
        super().__init__()
        self.tae = tae

    def forward(self, z, mem_0, mem_1, mem_2, mem_3, mem_4, mem_5, mem_6, mem_7, mem_8):
        return self.tae(z, mem_0, mem_1, mem_2, mem_3, mem_4, mem_5, mem_6, mem_7, mem_8)


class HeadGraph(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, h, mods, weights=None):
        return self.head(h, mods, weights)


class HeadGraphPdd(HeadGraph):
    def forward(self, h, mods, weights):
        return self.head(h, mods, weights)


def export(
    out_dir: Path,
    variant: Variant,
    *,
    dit_src: WeightSource,
    src_spec,
    vae_src: WeightSource | None,
    audio_src: WeightSource | None,
    tae_src: WeightSource | None,
    vae_cfg: VideoVaeConfig,
    audio_cfg: AudioVaeConfig,
    tae_cfg: TaeH3Config | None,
    sizes: list[Geometry],
    model_name: str,
    weights: str = "f16",
    chunk: int = 1024,
    q_chunk: int = 128,
    head_group: int = 8,
    text_buckets=TEXT_BUCKETS,
    tae_frames: int = 5,
    tile: int = TILE,
    with_encoders: bool = True,
    extra_rows: tuple[int, ...] = (),
    jobs: int = 1,
    log=print,
) -> dict:
    if weights not in WEIGHT_FORMATS:
        raise ValueError(f"--weights must be one of {WEIGHT_FORMATS}")
    cfg = variant.config
    if cfg.heads % head_group:
        raise ValueError(f"head_group {head_group} must divide heads {cfg.heads}")
    ratio = vae_cfg.spatial_ratio()
    for g in sizes:
        g.validate(ratio)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    f16 = weights != "f32"
    files: dict[str, dict] = {}

    def done(name: str, t0: float):
        p = out_dir / name
        files[name] = {"bytes": p.stat().st_size, "sha256": _sha(p)}
        log(f"  {name:34s} {p.stat().st_size / 1e6:9.1f} MB  {time.time() - t0:6.1f} s")

    def conv(name: str, module, kw, is_block: bool):
        """Block graphs: f16 for an f16 export, f32 then weight-only quantized for int8 / int4.
        Everything else: f16 unless the export is all-f32."""
        t0 = time.time()
        _convert(module, kw, out_dir / name, (weights == "f16") if is_block else f16)
        if is_block and weights in ("int8", "int4"):
            _quantize(out_dir / name, weights)
        done(name, t0)
        return name

    # ---- DiT (everything but the blocks) ------------------------------------------------
    dit = H3Dit(cfg, with_blocks=False).eval()
    load_dit(dit, dit_src)
    d, H, hd, inner = cfg.hidden, cfg.heads, cfg.head_dim, cfg.dim
    sampler = variant.sampler_json()
    graphs: dict = {}
    graphs["text_refine"] = []
    for t in text_buckets:
        name = conv(f"text_refine_{t}.tflite", dit.text, {"embeds": torch.randn(t, cfg.text_dim), "mask": torch.zeros(t)}, False)
        graphs["text_refine"].append({"rows": t, "file": name})
    graphs["embed_video"] = conv("embed_video.tflite", dit.embed_video, {"rows": torch.randn(chunk, cfg.video_patch_dim())}, False)
    graphs["embed_audio"] = conv("embed_audio.tflite", dit.embed_audio, {"rows": torch.randn(chunk, cfg.audio_in_channels)}, False)
    head_kw = {"h": torch.randn(chunk, d), "mods": torch.randn(2, d)}
    if cfg.pdd_steps:
        head_kw["weights"] = torch.rand(cfg.pdd_steps)
        graphs["head_video"] = conv("head_video.tflite", HeadGraphPdd(dit.head_video), head_kw, False)
        graphs["head_audio"] = conv("head_audio.tflite", HeadGraphPdd(dit.head_audio), head_kw, False)
    else:
        graphs["head_video"] = conv("head_video.tflite", HeadGraph(dit.head_video), head_kw, False)
        graphs["head_audio"] = conv("head_audio.tflite", HeadGraph(dit.head_audio), head_kw, False)

    # ---- the ladder's timesteps and the AdaLN input -----------------------------------------
    ts = sorted({v for st in sampler["steps"] for v in (st["t_video"], st["t_audio"])} | {sampler["cond_video_t"], sampler["cond_audio_t"]})
    ts = [float(np.float32(t)) for t in ts]
    with torch.no_grad():
        x_adaln = dit.time.adaln_input(torch.tensor(ts, dtype=torch.float32))  # [U, adaln_in]
        head = dit.norm_out_linear(x_adaln).reshape(-1, 2, d).numpy().astype(np.float16)

    # ---- blocks, one at a time, `jobs` at once ------------------------------------------------
    job_args = [(cfg.to_json(), src_spec, l, str(out_dir), chunk, weights, {"x": x_adaln.numpy().tolist()}) for l in range(cfg.layers)]
    results = []
    if jobs > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(jobs) as pool:
            for r in pool.imap_unordered(_block_job, job_args):
                results.append(r)
                log(f"  block {r['l']:02d}: qkv {r['qkv_secs']} s, post {r['post_secs']} s")
    else:
        for a in job_args:
            r = _block_job(a)
            results.append(r)
            log(f"  block {r['l']:02d}: qkv {r['qkv_secs']} s, post {r['post_secs']} s")
    results.sort(key=lambda r: r["l"])
    blocks = []
    for r in results:
        for k in ("qkv", "post"):
            pth = out_dir / r[k]
            files[r[k]] = {"bytes": pth.stat().st_size, "sha256": _sha(pth)}
        blocks.append({"qkv": r["qkv"], "post": r["post"], "bytes": files[r["qkv"]]["bytes"] + files[r["post"]]["bytes"]})
    graphs["blocks"] = blocks

    # ---- AdaLN table --------------------------------------------------------------------------
    t0 = time.time()
    tab = np.stack([np.load(out_dir / f".adaln_{l:02d}.npy") for l in range(cfg.layers)], axis=1)  # [U, L, 3, 6, d]
    for l in range(cfg.layers):
        (out_dir / f".adaln_{l:02d}.npy").unlink()
    from safetensors.numpy import save_file

    save_file({"timesteps": np.array(ts, np.float32), "blocks": np.ascontiguousarray(tab), "head": np.ascontiguousarray(head)},
              str(out_dir / "adaln.safetensors"))
    done("adaln.safetensors", t0)

    # ---- attention per size, TAEH3, audio ----------------------------------------------------
    attention, taeh3_list, audio_list = [], [], []
    seen_s, seen_hw, seen_a = set(), set(), set()
    from .packing import build_layout

    for g in sizes:
        lt, lh, lw = g.latent(ratio)
        na = g.audio_latents()
        # Rows of the T2VA layout with the largest text bucket: the attention graph's S must match the request's
        # rows exactly, so one graph per (size, prompt rows) bucket is exported.
        for t in text_buckets:
            for anchors in ([], ["first"], ["first", "last"]):
                s = build_layout([1] * t, lt, lh, lw, na, cfg.patch, anchors).sequence_length
                if s in seen_s:
                    continue
                seen_s.add(s)
                cq = min(q_chunk, s)
                name = conv(f"attn_s{s}_q{cq}_h{head_group}.tflite", Attention(),
                            {"q": torch.randn(cq, head_group, hd), "k": torch.randn(s, head_group, hd), "v": torch.randn(s, head_group, hd), "mask": torch.zeros(s)}, False)
                attention.append({"tokens": s, "q_chunk": cq, "heads": head_group, "file": name})
        if tae_cfg is not None and tae_src is not None and (lh, lw) not in seen_hw:
            seen_hw.add((lh, lw))
            tae = TaeH3(tae_cfg).eval()
            load_tae(tae, tae_src)
            f = min(tae_frames, lt)
            kw = {"z": torch.randn(f, tae_cfg.latent_channels, lh, lw)}
            for i, shape in enumerate(tae.memory_shapes(lh, lw)):
                kw[f"mem_{i}"] = torch.zeros(shape)
            name = conv(f"taeh3_f{f}_{lh}x{lw}.tflite", TaeGraph(tae), kw, False)
            taeh3_list.append({"frames": f, "latent_h": lh, "latent_w": lw, "file": name})
        if audio_src is not None and na not in seen_a:
            seen_a.add(na)
            dec = AudioDecoder(audio_cfg).eval()
            load_audio_decoder(dec, audio_src)
            entry = {"latents": na, "decoder": conv(f"avae_dec_{na}.tflite", dec, {"z": torch.randn(audio_cfg.latent_channels, na)}, False)}
            if with_encoders:
                enc = AudioEncoder(audio_cfg).eval()
                load_audio_encoder(enc, audio_src)
                entry["encoder"] = conv(f"avae_enc_{na}.tflite", enc, {"wave": torch.randn(audio_cfg.hop() * na) * 0.1}, False)
            audio_list.append(entry)

    for s in extra_rows:
        if s not in seen_s:
            seen_s.add(s)
            cq = min(q_chunk, s)
            name = conv(f"attn_s{s}_q{cq}_h{head_group}.tflite", Attention(),
                        {"q": torch.randn(cq, head_group, hd), "k": torch.randn(s, head_group, hd), "v": torch.randn(s, head_group, hd), "mask": torch.zeros(s)}, False)
            attention.append({"tokens": s, "q_chunk": cq, "heads": head_group, "file": name})

    # ---- video VAE ----------------------------------------------------------------------------
    video_vae = None
    if vae_src is not None:
        dec = Decoder(vae_cfg).eval()
        load_vae_decoder(dec, vae_src)
        # The decoder's token count is fixed by the tile (256 px → 16×16 latent) or the canvas when smaller,
        # and by the 7-latent-frame chunk.
        g0 = sizes[0]
        _, lh0, lw0 = g0.latent(ratio)
        th = min(tile // ratio, lh0)
        tw = min(tile // ratio, lw0)
        lt_chunk = vae_cfg.tokens_chunk() + vae_cfg.token_overlap()
        rows = dec.rows(lt_chunk, th, tw)
        d_in = conv("vae_dec_in.tflite", DecIn(dec, lt_chunk, th, tw), {"z": torch.randn(vae_cfg.latent_channels, lt_chunk, th, tw)}, False)
        dim = vae_cfg.decoder_dim()
        half = vae_cfg.decoder_rotary_dim() // 2
        dblocks = [conv(f"vae_dec_block_{l:02d}.tflite", dec.blocks[l], {"h": torch.randn(rows, dim), "cos": torch.rand(rows, half), "sin": torch.rand(rows, half)}, True) for l in range(vae_cfg.decoder_layers)]
        d_out = conv("vae_dec_out.tflite", DecOut(dec, lt_chunk, th, tw), {"h": torch.randn(rows, dim)}, False)
        video_vae = {"decoder": {"latent_t": lt_chunk, "latent_h": th, "latent_w": tw, "rows": rows, "in": d_in, "blocks": dblocks, "out": d_out}, "tile": tile, "overlap": OVERLAP}
        if with_encoders:
            enc = Encoder(vae_cfg).eval()
            load_vae_encoder(enc, vae_src)
            px_h, px_w = th * ratio, tw * ratio
            video_vae["encoder_clip"] = conv("vae_enc_clip.tflite", EncoderGraph(enc), {"x": torch.randn(3, vae_cfg.clip_length, px_h, px_w)}, False)
            video_vae["encoder_image"] = conv("vae_enc_image.tflite", EncoderGraph(enc), {"x": torch.randn(3, 1, px_h, px_w)}, False)

    manifest = {
        "format": FORMAT, "model": model_name, "variant": variant.name,
        "config": cfg.to_json(), "vae": vae_cfg.to_json(), "audio_vae": audio_cfg.to_json(), "taeh3": tae_cfg.to_json() if tae_cfg else None,
        "sampler": sampler, "adaln": {"file": "adaln.safetensors", "timesteps": ts},
        "weights": weights, "chunk": chunk, "head_group": head_group,
        "graphs": graphs, "attention": attention, "video_vae": video_vae, "audio": audio_list, "taeh3_graphs": taeh3_list,
        "sizes": [g.to_json() for g in sizes], "files": files,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    log(f"wrote {out_dir / 'manifest.json'}: {len(files)} files, {sum(f['bytes'] for f in files.values()) / 1e6:.1f} MB")
    return manifest


# ---- the tiny fixture ---------------------------------------------------------------------------

TINY_GEOMETRY = Geometry(96, 64, 22)  # latent (7, 4, 6), 37 audio latents
TINY_SEED = 7
TINY_TEXT_TAGS = [1, 1, 0, 1, 1, 1]


def tiny_prompt(cfg: H3Config) -> np.ndarray:
    from .rng import Rng

    return Rng(1234).normal(len(TINY_TEXT_TAGS) * cfg.text_dim).reshape(len(TINY_TEXT_TAGS), cfg.text_dim)


def export_tiny(out_dir: Path, weights: str = "f32", log=print) -> dict:
    from safetensors.numpy import save_file

    from . import reference
    from .variants import by_name

    v = by_name("tiny")
    src = Synthetic(0)
    vae_cfg, audio_cfg, tae_cfg = VideoVaeConfig.tiny(), AudioVaeConfig.tiny(), TaeH3Config.tiny()
    manifest = export(out_dir, v, dit_src=src, src_spec=("synthetic", 0), vae_src=src, audio_src=src, tae_src=src, vae_cfg=vae_cfg, audio_cfg=audio_cfg, tae_cfg=tae_cfg,
                      sizes=[TINY_GEOMETRY], model_name="tiny", weights=weights, chunk=16, q_chunk=32, head_group=1, text_buckets=(8, 16),
                      tae_frames=5, log=log)
    embeds = tiny_prompt(v.config)
    exp = reference.generate(v, src, vae_cfg, audio_cfg, embeds, TINY_TEXT_TAGS, TINY_GEOMETRY, TINY_SEED)
    save_file({"prompt_embeds": embeds, "text_tags": np.array(TINY_TEXT_TAGS, np.float32), **exp}, str(Path(out_dir) / "expected.safetensors"),
              metadata={"seed": str(TINY_SEED), "geometry": str(TINY_GEOMETRY)})
    log("wrote expected.safetensors: " + ", ".join(f"{k} {list(t.shape)}" for k, t in exp.items()))
    return manifest
