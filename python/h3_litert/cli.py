"""`h3-litert`: export, generate, check, plan, variants."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .config import AudioVaeConfig, TaeH3Config, VideoVaeConfig
from .packing import Geometry, H3_768P


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def cmd_export(a) -> int:
    from .export import export, export_tiny
    from .variants import by_name

    if a.variant == "tiny":
        # The self-test fixture is f32 unless asked otherwise: the tiny weights exceed the converter's
        # f16 threshold, and the check compares against an f32 PyTorch reference.
        export_tiny(Path(a.out), weights=a.weights or "f32", log=_log)
        return 0
    from .weights import SafetensorsDir, read_h3_config

    v = by_name(a.variant)
    root = Path(a.model_dir)
    tdir = root / v.transformer_dir if (root / v.transformer_dir).is_dir() else root
    if (tdir / "config.json").exists():
        from dataclasses import replace

        v = replace(v, config=read_h3_config(tdir))
    base = Path(a.base_dir) if a.base_dir else root
    vae_cfg = VideoVaeConfig.fasth3_trim() if v.name == "fasth3-trim" else VideoVaeConfig.h3()
    sizes = [Geometry.parse(s) for s in a.size] if a.size else [H3_768P]
    weights = a.weights or "f16"
    export(Path(a.out), v, dit_src=SafetensorsDir(tdir), vae_src=SafetensorsDir(base / "vae") if (base / "vae").is_dir() else None,
           audio_src=SafetensorsDir(base / "audio_vae") if (base / "audio_vae").is_dir() else None,
           tae_src=SafetensorsDir(Path(a.tae)) if a.tae else None, vae_cfg=vae_cfg, audio_cfg=AudioVaeConfig.h3(),
           tae_cfg=TaeH3Config.taeh3() if a.tae else None, sizes=sizes, model_name="h3", weights=weights, chunk=a.chunk,
           q_chunk=a.q_chunk, head_group=a.head_group, with_encoders=not a.no_encoders, extra_rows=tuple(a.rows or ()), log=_log)
    return 0


def _open(a):
    from .pipeline import Pipeline
    from .runtime import GraphStore, Manifest

    return Pipeline(GraphStore(Manifest(a.graphs), accelerator=a.accelerator, resident=a.resident, threads=a.threads))


def _load_prompt(path: str):
    from safetensors.numpy import load_file

    t = load_file(path)
    emb = t.get("prompt_embeds", t.get("embeds"))
    if emb is None:
        raise SystemExit(f"{path}: no prompt_embeds")
    tags = t["text_tags"].astype(int).tolist() if "text_tags" in t else [1] * emb.shape[0]
    return emb.astype(np.float32), tags


def cmd_generate(a) -> int:
    from .media import read_png_rgb, write_wav, write_y4m
    from .request import Keyframe, Request

    pipe = _open(a)
    emb, tags = _load_prompt(a.prompt)
    g = Geometry.parse(a.size)
    kfs = []
    for anchor, path in (("first", a.first_frame), ("last", a.last_frame)):
        if path:
            rgb = read_png_rgb(Path(path))
            if rgb.shape[:2] != (g.height, g.width):
                raise SystemExit(f"{path} is {rgb.shape[1]}x{rgb.shape[0]}, the canvas is {g.width}x{g.height}")
            kfs.append(Keyframe(anchor, (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[:, None]))
    req = Request(emb, tags, g, a.seed, kfs, a.preview)

    def on(ev):
        if ev["type"] == "step":
            _log(f"step {ev['index'] + 1}/{ev['total']}: {ev['ms'] / 1000:.1f} s")
        elif ev["type"] == "frames":
            _log(f"frames {ev['first']}..{ev['first'] + ev['count']} {ev['width']}x{ev['height']}")
        elif ev["type"] == "prepared":
            _log(f"prepared {ev['tokens']} rows in {ev['ms'] / 1000:.1f} s")

    frames, audio, tm = pipe.generate(req, on)
    out = Path(a.out)
    write_y4m(out, frames, 24)
    write_wav(out.with_suffix(".wav"), audio, pipe.m.audio_vae.sample_rate)
    _log(f"{out}: {frames.shape[0]} frames {frames.shape[2]}x{frames.shape[1]}, {out.with_suffix('.wav')}; denoise {sum(tm.steps_s):.1f} s, decode {tm.decode_s:.1f} s, audio {tm.audio_s:.1f} s")
    _log("mux: ffmpeg -i out.y4m -i out.wav -c:v libx264 -c:a aac out.mp4")
    return 0


def cmd_check(a) -> int:
    from safetensors.numpy import load_file

    from .request import Request

    pipe = _open(a)
    d = Path(a.graphs)
    exp = load_file(str(d / "expected.safetensors"))
    size = pipe.m.raw["sizes"][0]
    g = Geometry(size["width"], size["height"], size["frames"])
    req = Request(exp["prompt_embeds"], exp["text_tags"].astype(int).tolist(), g, 7)
    t0 = time.time()
    latent, audio_rows = pipe.denoise(req)
    rel_l = float(np.abs(latent - exp["latent"]).max() / np.abs(exp["latent"]).max())
    rel_a = float(np.abs(audio_rows - exp["audio_rows"]).max() / np.abs(exp["audio_rows"]).max())
    frames = pipe.decode_video(latent)
    fd = int(np.abs(frames.astype(int) - exp["frames"].astype(int)).max())
    audio = pipe.decode_audio(audio_rows, g.audio_latents())
    rel_w = float(np.abs(audio - exp["audio"]).max() / max(np.abs(exp["audio"]).max(), 1e-6))
    # f16 / int8 / int4 exports are judged against the f32 reference at a looser tolerance
    # (the contract's GPU budget); the fixture itself is f32.
    tol = a.tol if pipe.m.weights == "f32" else max(a.tol, 2e-2)
    fmax = 2 if pipe.m.weights == "f32" else 16
    ok = rel_l <= tol and rel_a <= tol and fd <= fmax and rel_w <= 10 * tol
    _log(f"[{pipe.m.weights}] video latent rel {rel_l:.2e}, audio rows rel {rel_a:.2e}, frames max |Δ| {fd}/255, waveform rel {rel_w:.2e} (tol {tol:.0e}) in {time.time() - t0:.1f} s: {'OK' if ok else 'FAIL'}")
    return 0 if ok else 1


def cmd_variants(a) -> int:
    from .variants import all_variants

    for v in all_variants():
        print(f"{v.name:12s} {v.forwards():3d} forwards  {v.rule():8s} {v.summary}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="h3-litert", description="MiniMax H3 on LiteRT")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--variant", default="fasth3-v2")
    e.add_argument("--model-dir", help="the variant's download (transformer/ inside)")
    e.add_argument("--base-dir", help="MiniMaxAI/MiniMax-H3 download with vae/ and audio_vae/ (default: --model-dir)")
    e.add_argument("--tae", help="taeh3.safetensors for the preview decoder")
    e.add_argument("--out", required=True)
    e.add_argument("--weights", choices=["f32", "f16", "int8", "int4"], help="block weight format (default f16; the tiny fixture f32)")
    e.add_argument("--chunk", type=int, default=1024)
    e.add_argument("--q-chunk", type=int, default=128)
    e.add_argument("--head-group", type=int, default=8)
    e.add_argument("--size", action="append", help="WxHxF (repeatable; default 1344x768x124)")
    e.add_argument("--rows", type=int, action="append", help="extra packed-sequence lengths to export attention graphs for (Ref2VA)")
    e.add_argument("--no-encoders", action="store_true")
    e.set_defaults(fn=cmd_export)

    def rt(p):
        p.add_argument("--graphs", required=True)
        p.add_argument("--accelerator", default="cpu", choices=["cpu", "gpu"])
        p.add_argument("--resident", type=int, default=2)
        p.add_argument("--threads", type=int)

    gen = sub.add_parser("generate")
    rt(gen)
    gen.add_argument("--prompt", required=True, help="safetensors with prompt_embeds [rows, 5120] (+ text_tags)")
    gen.add_argument("--size", default="1344x768x124")
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--first-frame", help="PNG at the canvas size (FL2VA)")
    gen.add_argument("--last-frame")
    gen.add_argument("--preview", action="store_true", help="decode with TAEH3")
    gen.add_argument("--out", default="out.y4m")
    gen.set_defaults(fn=cmd_generate)
    ch = sub.add_parser("check")
    rt(ch)
    ch.add_argument("--tol", type=float, default=1e-3)
    ch.set_defaults(fn=cmd_check)
    sub.add_parser("variants").set_defaults(fn=cmd_variants)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
