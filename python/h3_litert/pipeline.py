"""Prompt embeddings → packed layout → denoising loop → decoded video and audio,
driving the graphs exactly as `docs/graphs.md`'s host loop says. T2VA and FL2VA;
Euler, renoise and PDD samplers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import vae_host
from .config import AUDIO_TAG, H3Config, TEXT_TAG, VIDEO_TAG
from .packing import Layout, build_layout, mod_runs, patch_rows, row_timesteps, runs, unpatch_rows
from .request import KEYFRAME_ENCODE_SEED, Request
from .rng import Rng
from .rope import h3_rope
from .runtime import GraphStore, Manifest
from .sampler import flow_step, needs_noise
from .schedule import KEYFRAME_NOISE_AUG

OnEvent = Callable[[dict], None]
MASK_PAD = -1e9


@dataclass
class Timings:
    prepare_s: float = 0.0
    steps_s: list[float] = field(default_factory=list)
    decode_s: float = 0.0
    audio_s: float = 0.0


@dataclass
class Prepared:
    layout: Layout
    text: np.ndarray  # [T, d] (bucket rows)
    n_text: int
    pad: int
    video_rows: np.ndarray
    audio_rows: np.ndarray
    cos: np.ndarray
    sin: np.ndarray
    mask: np.ndarray
    video_idx: list[int]  # packed indices after the text padding shift
    audio_idx: list[int]
    tags: list[int]  # packed tags after the shift


class AdalnTables:
    def __init__(self, path):
        from safetensors.numpy import load_file

        t = load_file(str(path))
        self.timesteps = t["timesteps"].astype(np.float32)
        self.blocks = t["blocks"]  # [U, L, 3, 6, d] f16
        self.head = t["head"]  # [U, 2, d] f16

    def index_of(self, t: float) -> int:
        i = np.where(np.isclose(self.timesteps, np.float32(t), rtol=0, atol=1e-6))[0]
        if len(i) == 0:
            raise ValueError(f"no AdaLN table row for timestep {t}; this export covers {self.timesteps.tolist()}")
        return int(i[0])

    def mods(self, u: int, layer: int, tag: int) -> np.ndarray:
        return self.blocks[u, layer, tag].astype(np.float32)

    def head_mods(self, u: int) -> np.ndarray:
        return self.head[u].astype(np.float32)


class Pipeline:
    def __init__(self, store: GraphStore):
        self.store = store
        self.m: Manifest = store.m
        self.cfg: H3Config = self.m.cfg
        self.adaln = AdalnTables(self.m.path(self.m.raw["adaln"]["file"]))

    # ---- helpers ------------------------------------------------------------------------
    def _rows(self, graph, chunk: int, rows: dict[str, np.ndarray], fixed: dict[str, np.ndarray], n_real: int) -> list[np.ndarray]:
        """Run a per-row graph over `rows` arrays in chunks of `chunk` (zero-padded); outputs trimmed to `n_real`."""
        outs: list[list[np.ndarray]] = []
        for r in range(0, n_real, chunk):
            n = min(chunk, n_real - r)
            feed = {}
            for k, v in rows.items():
                blk = np.zeros((chunk,) + v.shape[1:], np.float32)
                blk[:n] = v[r : r + n]
                feed[k] = blk
            outs.append(graph.run(**feed, **fixed))
        return [np.concatenate([o[i] for o in outs], axis=0)[:n_real] for i in range(len(outs[0]))]

    # ---- prompt + layout --------------------------------------------------------------------
    def prepare(self, req: Request, encode_keyframe) -> Prepared:
        c, m = self.cfg, self.m
        ratio = m.vae.spatial_ratio()
        g = req.geometry
        g.validate(ratio)
        lt, lh, lw = g.latent(ratio)
        na = g.audio_latents()
        n_text = len(req.text_tags)
        if req.embeds.shape != (n_text, c.text_dim):
            raise ValueError(f"prompt embeds are {req.embeds.shape}, expected ({n_text}, {c.text_dim})")
        bucket = m.text_refine_for(n_text)
        T = int(bucket["rows"])
        pad = T - n_text
        emb = np.zeros((T, c.text_dim), np.float32)
        emb[:n_text] = req.embeds
        tmask = np.zeros(T, np.float32)
        tmask[n_text:] = MASK_PAD
        (text,) = self.store.fixed(bucket["file"]).run(embeds=emb, mask=tmask)
        anchors = [k.anchor for k in req.keyframes]
        layout = build_layout(req.text_tags, lt, lh, lw, na, c.patch, anchors)
        rng = Rng(req.seed)
        cond = np.zeros((0, c.video_patch_dim()), np.float32)
        if req.keyframes:
            rows = [self._encode_keyframe(k.pixels, encode_keyframe) for k in req.keyframes]
            cond = np.concatenate(rows, axis=0)
            noise = rng.normal(cond.size).reshape(cond.shape)
            cond = (np.float32(KEYFRAME_NOISE_AUG) * cond + np.float32(1.0 - KEYFRAME_NOISE_AUG) * noise).astype(np.float32)
        noise = rng.normal(c.in_channels * lt * lh * lw).reshape(c.in_channels, lt, lh, lw)
        video_rows = np.concatenate([cond, patch_rows(noise, c.patch)], axis=0)
        audio_rows = rng.normal(na * 2 * c.audio_in_channels).reshape(na * 2, c.audio_in_channels)
        shift = lambda i: i if i < n_text else i + pad  # noqa: E731
        s_padded = layout.sequence_length + pad
        pos = np.zeros((s_padded, 3), np.float32)
        tags = [TEXT_TAG] * s_padded
        for i, p in enumerate(layout.position_ids):
            pos[shift(i)] = p
            tags[shift(i)] = layout.token_tags[i]
        cos, sin = h3_rope(pos, c.rope_freq_dim, c.rope_theta)
        mask = np.zeros(s_padded, np.float32)
        mask[n_text:T] = MASK_PAD
        return Prepared(layout, text, n_text, pad, video_rows, audio_rows, cos, sin, mask,
                        [shift(i) for i in layout.video_indices], [shift(i) for i in layout.audio_indices], tags)

    def _encode_keyframe(self, pixels: np.ndarray, encode_keyframe) -> np.ndarray:
        vc, ratio = self.m.vae, self.m.vae.spatial_ratio()
        moments = vae_host.encode(vc, vae_host.normalize_pixels(pixels), encode_keyframe)
        lz = moments.shape[0] // 2
        rng = Rng(KEYFRAME_ENCODE_SEED)
        z = vae_host.posterior_sample(moments, rng.normal(lz * int(np.prod(moments.shape[1:]))).reshape(lz, *moments.shape[1:]))
        z = vae_host.normalize_latents(vc, z.astype(np.float16).astype(np.float32))
        return patch_rows(z, self.cfg.patch)

    def encoder_tile_runner(self):
        """`encode_clip` over the encoder graphs with the manifest's tiling, for `prepare`."""
        vv = self.m.raw["video_vae"]
        ratio = self.m.vae.spatial_ratio()

        def run(x: np.ndarray) -> np.ndarray:
            name = vv.get("encoder_image" if x.shape[1] == 1 else "encoder_clip") if vv else None
            if name is None:
                raise ValueError("this export has no VAE encoder graphs (exported with --no-encoders)")
            return vae_host.tiled(x, 1, ratio, vv["tile"], vv["overlap"], lambda t: self.store.fixed(name).run(x=t)[0])

        return run

    # ---- one forward ---------------------------------------------------------------------------
    def forward(self, p: Prepared, step: dict, attn: dict):
        c, m, ad = self.cfg, self.m, self.adaln
        C, G, d, inner, hd = m.chunk, m.head_group, c.hidden, c.dim, c.head_dim
        s = p.layout.sequence_length + p.pad
        unique, inv = row_timesteps(p.layout, step["t_video"], step["t_audio"], max(step["t_video"], m.sampler["cond_video_t"]), m.sampler["cond_audio_t"])
        u_of = [ad.index_of(t) for t in unique]
        shift = lambda i: i if i < p.n_text else i + p.pad  # noqa: E731
        t_idx = [0] * s
        for i, ti in enumerate(inv):
            t_idx[shift(i)] = ti
        for i in range(p.n_text, p.n_text + p.pad):
            t_idx[i] = inv[0]
        h = np.zeros((s, d), np.float32)
        h[: p.n_text + p.pad] = p.text
        emb_v, emb_a = self.store.fixed(m.graphs["embed_video"]), self.store.fixed(m.graphs["embed_audio"])
        for r in runs(p.video_idx):
            (h[r.start : r.start + r.rows],) = self._rows(emb_v, C, {"rows": p.video_rows[r.offset : r.offset + r.rows]}, {}, r.rows)
        for r in runs(p.audio_idx):
            (h[r.start : r.start + r.rows],) = self._rows(emb_a, C, {"rows": p.audio_rows[r.offset : r.offset + r.rows]}, {}, r.rows)
        mruns = mod_runs(t_idx, p.tags)
        attn_graph = self.store.fixed(attn["file"])
        cq = int(attn["q_chunk"])
        q = np.zeros((s, c.heads, hd), np.float32)
        k = np.zeros_like(q)
        v = np.zeros_like(q)
        o = np.zeros((s, inner), np.float32)
        for l, b in enumerate(m.graphs["blocks"]):
            gq = self.store.block(b["qkv"])
            for start, n, ti, tag in mruns:
                mods = ad.mods(u_of[ti], l, tag)
                sl = slice(start, start + n)
                q[sl], k[sl], v[sl] = self._rows(gq, C, {"h": h[sl], "cos": p.cos[sl], "sin": p.sin[sl]}, {"mods": mods}, n)
            for g in range(c.heads // G):
                hs = slice(g * G, (g + 1) * G)
                kg, vg = np.ascontiguousarray(k[:, hs]), np.ascontiguousarray(v[:, hs])
                (og,) = self._rows(attn_graph, cq, {"q": np.ascontiguousarray(q[:, hs])}, {"k": kg, "v": vg, "mask": p.mask}, s)
                o[:, g * G * hd : (g + 1) * G * hd] = og
            gp = self.store.block(b["post"])
            for start, n, ti, tag in mruns:
                mods = ad.mods(u_of[ti], l, tag)
                sl = slice(start, start + n)
                (h[sl],) = self._rows(gp, C, {"h": h[sl], "o": o[sl]}, {"mods": mods}, n)
        hv, ha = self.store.fixed(m.graphs["head_video"]), self.store.fixed(m.graphs["head_audio"])
        extra = {}
        if m.sampler["rule"] == "pdd":
            extra = {"video": {"weights": np.array(step["pdd_video_weights"], np.float32)}, "audio": {"weights": np.array(step["pdd_audio_weights"], np.float32)}}
        v_out = np.zeros((len(p.video_idx), c.video_patch_dim()), np.float32)
        a_out = np.zeros((len(p.audio_idx), c.audio_in_channels), np.float32)
        for graph, idx, out, key in ((hv, p.video_idx, v_out, "video"), (ha, p.audio_idx, a_out, "audio")):
            for r in runs(idx):
                # a run of one modality may still span two timesteps (condition rows vs target rows)
                start = r.start
                while start < r.start + r.rows:
                    ti = t_idx[start]
                    end = start
                    while end < r.start + r.rows and t_idx[end] == ti:
                        end += 1
                    n = end - start
                    off = r.offset + (start - r.start)
                    (out[off : off + n],) = self._rows(graph, C, {"h": h[start:end]}, {"mods": ad.head_mods(u_of[ti]), **extra.get(key, {})}, n)
                    start = end
        return v_out, a_out

    # ---- denoise ------------------------------------------------------------------------------
    def denoise(self, req: Request, on: OnEvent | None = None, tm: Timings | None = None):
        c, m = self.cfg, self.m
        tm = tm or Timings()
        t0 = time.perf_counter()
        p = self.prepare(req, self.encoder_tile_runner())
        attn = m.attention_for(p.layout.sequence_length + p.pad)
        tm.prepare_s = time.perf_counter() - t0
        if on:
            on({"type": "prepared", "ms": tm.prepare_s * 1000.0, "tokens": p.layout.sequence_length})
        rule = m.sampler["rule"]
        steps = m.sampler["steps"]
        off_v = p.layout.num_condition_video_rows
        if rule == "pdd":
            p.video_rows[off_v:] *= np.float32(steps[0]["sigma_video"])
            p.audio_rows *= np.float32(steps[0]["sigma_audio"])
        renoise = Rng(req.seed + 1)
        for i, st in enumerate(steps):
            ts = time.perf_counter()
            v, a = self.forward(p, st, attn)
            tgt = p.video_rows[off_v:]
            nz = renoise.normal(tgt.size).reshape(tgt.shape) if needs_noise(rule, st["sigma_video_next"]) else None
            p.video_rows[off_v:] = flow_step(tgt, v[off_v:], st["sigma_video"], st["sigma_video_next"], rule, nz)
            nz = renoise.normal(p.audio_rows.size).reshape(p.audio_rows.shape) if needs_noise(rule, st["sigma_audio_next"]) else None
            p.audio_rows[:] = flow_step(p.audio_rows, a, st["sigma_audio"], st["sigma_audio_next"], rule, nz)
            tm.steps_s.append(time.perf_counter() - ts)
            if on:
                on({"type": "step", "index": i, "total": len(steps), "ms": tm.steps_s[-1] * 1000.0})
        lt, lh, lw = req.geometry.latent(m.vae.spatial_ratio())
        latent = unpatch_rows(p.video_rows[off_v:], c.in_channels, lt, lh, lw, c.patch)
        return latent, p.audio_rows

    # ---- decode -------------------------------------------------------------------------------
    def decode_video(self, latent: np.ndarray, preview: bool = False, on: OnEvent | None = None, tm: Timings | None = None) -> np.ndarray:
        """Normalised latent `[C, T, h, w]` → `[F, H, W, 3]` uint8."""
        tm = tm or Timings()
        t0 = time.perf_counter()
        m = self.m
        if preview:
            frames = self._decode_taeh3(latent, on)
        else:
            vv = m.raw["video_vae"]
            if vv is None:
                raise ValueError("this export has no video VAE decoder")
            dec = vv["decoder"]
            ratio = m.vae.spatial_ratio()
            g_in, g_out = self.store.fixed(dec["in"]), self.store.fixed(dec["out"])
            blocks = [self.store.fixed(f) for f in dec["blocks"]]

            def clip_tile(z: np.ndarray) -> np.ndarray:
                if z.shape[1:] != (dec["latent_t"], dec["latent_h"], dec["latent_w"]):
                    raise ValueError(f"decoder tile {z.shape[1:]} but the export has graphs for {(dec['latent_t'], dec['latent_h'], dec['latent_w'])}")
                h, cos, sin = g_in.run(z=z)
                for b in blocks:
                    (h,) = b.run(h=h, cos=cos, sin=sin)
                return g_out.run(h=h)[0]

            total = 0

            def on_chunk(px):
                nonlocal total
                if on:
                    rgb = (np.clip(px, 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 3, 0)
                    on({"type": "frames", "first": total, "count": rgb.shape[0], "width": rgb.shape[2], "height": rgb.shape[1], "rgb": np.ascontiguousarray(rgb)})
                total += px.shape[1]

            px = vae_host.decode(m.vae, vae_host.denormalize_latents(m.vae, latent),
                                 lambda z: vae_host.tiled(z, ratio, 1, vv["tile"], vv["overlap"], clip_tile), on_chunk)
            frames = np.ascontiguousarray((np.clip(px, 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(1, 2, 3, 0))
        tm.decode_s = time.perf_counter() - t0
        return frames

    def _decode_taeh3(self, latent: np.ndarray, on: OnEvent | None) -> np.ndarray:
        m = self.m
        tc = m.taeh3
        if tc is None:
            raise ValueError("this export has no TAEH3 graph")
        c, t, h, w = latent.shape
        entry = m.taeh3_for(h, w)
        graph = self.store.fixed(entry["file"])
        f = int(entry["frames"])
        mems = [np.zeros((1, tc.widths[s], h << s, w << s), np.float32) for s in range(3) for _ in range(tc.mem_blocks)]
        out, raw_index, total = [], 0, 0
        for f0 in range(0, t, f):
            n = min(f, t - f0)
            z = np.zeros((f, c, h, w), np.float32)
            z[:n] = latent[:, f0 : f0 + n].transpose(1, 0, 2, 3)
            res = graph.run(z=z, **{f"mem_{i}": mems[i] for i in range(len(mems))})
            rgb, mems = res[0][: n * tc.t_upscale()], list(res[1:])
            keep = [i for i in range(rgb.shape[0]) if tc.keeps_raw_frame(raw_index + i)]
            raw_index += rgb.shape[0]
            rgb8 = np.ascontiguousarray((np.clip(rgb[keep], 0, 1) * 255.0 + 0.5).astype(np.uint8).transpose(0, 2, 3, 1))
            if rgb8.shape[0]:
                out.append(rgb8)
                if on:
                    on({"type": "frames", "first": total, "count": rgb8.shape[0], "width": rgb8.shape[2], "height": rgb8.shape[1], "rgb": rgb8})
                total += rgb8.shape[0]
        return np.concatenate(out, axis=0)

    def decode_audio(self, audio_rows: np.ndarray, na: int, tm: Timings | None = None) -> np.ndarray:
        """Normalised target audio rows `[2A, A_c]` → stereo `[2, 800·A]`."""
        tm = tm or Timings()
        t0 = time.perf_counter()
        ac = self.m.audio_vae
        entry = self.m.audio_for(na)
        g = self.store.fixed(entry["decoder"])
        mean = np.array(ac.latents_mean, np.float32).reshape(-1, 1)
        std = np.array(ac.latents_std, np.float32).reshape(-1, 1)
        out = []
        for ch in range(2):
            z = np.ascontiguousarray(audio_rows[ch * na : (ch + 1) * na].T * std + mean)
            out.append(g.run(z=z)[0])
        tm.audio_s = time.perf_counter() - t0
        return np.stack(out).astype(np.float32)

    def generate(self, req: Request, on: OnEvent | None = None):
        tm = Timings()
        latent, audio_rows = self.denoise(req, on, tm)
        frames = self.decode_video(latent, req.preview, on, tm)
        audio = self.decode_audio(audio_rows, req.geometry.audio_latents(), tm)
        if on:
            on({"type": "audio", "sample_rate": self.m.audio_vae.sample_rate, "stereo": audio})
        return frames, audio, tm
