"""Pixel ↔ latent ↔ packed-token geometry of one H3 request (h3-core
`packing.rs`; FastVideo `packing.py`). Positions are computed in f64 exactly as
the reference does; the rotary tables are then formed in f32."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .config import AUDIO_TAG, TEXT_TAG, VIDEO_TAG

FPS = 24
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344
CANVAS_MULTIPLE = 32
MIN_ASPECT, MAX_ASPECT = 0.25, 4.0
FRAMES_PER_CHUNK = 17
LATENTS_PER_CHUNK = 5
AUDIO_LATENTS_PER_SECOND = 40
AUDIO_CHANNELS = 2
MIN_ALIGNED_FRAMES, MAX_ALIGNED_FRAMES = 124, 362
CONTINUATION_FRAMES = (5, 22, 39)
_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1.0, 4.0, 4.0, 4.0, 4.0)
_ROPE_SPATIAL_SCALE = 32.0


def canvas_size(aspect_w: float, aspect_h: float, short_edge: int = SHORT_EDGE, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Canvas `(height, width)` for an aspect ratio, snapped to the 32-pixel lattice."""
    if aspect_w <= 0 or aspect_h <= 0:
        raise ValueError(f"aspect ratio must be positive, got {aspect_w}:{aspect_h}")
    ratio = aspect_w / aspect_h
    if not (MIN_ASPECT <= ratio <= MAX_ASPECT):
        raise ValueError(f"aspect ratios run from 1:4 to 4:1, got {aspect_w}:{aspect_h}")
    w, h = (short_edge * ratio, float(short_edge)) if ratio >= 1.0 else (float(short_edge), short_edge / ratio)
    area = w * h
    if area > max_pixels:
        s = math.sqrt(max_pixels / area)
        w, h = w * s, h * s
    m = float(CANVAS_MULTIPLE)
    snap = lambda v: int(max(round(v / m) * m, m))  # noqa: E731
    return snap(h), snap(w)


def align_num_frames(n: int) -> int:
    n = max(n, 1)
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        n += 1
    return n


def resolve_num_frames(n: int) -> int:
    a = align_num_frames(n)
    if not (MIN_ALIGNED_FRAMES <= a <= MAX_ALIGNED_FRAMES):
        raise ValueError(f"H3 generates 5-15 s at 24 fps; aligned num_frames {a} is outside 124-362")
    return a


def video_latent_frames(n: int) -> int:
    if n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK:
        raise ValueError(f"num_frames must be 17n + 5, got {n}")
    return (n - LATENTS_PER_CHUNK) // FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2


def audio_latents(frames: int) -> int:
    """`round(frames / 24 · 40)` with Python's half-even rounding."""
    return int(round(frames / FPS * AUDIO_LATENTS_PER_SECOND))


@dataclass(frozen=True)
class Geometry:
    width: int
    height: int
    frames: int

    def validate(self, spatial_ratio: int) -> None:
        if self.width <= 0 or self.height <= 0 or self.width % CANVAS_MULTIPLE or self.height % CANVAS_MULTIPLE:
            raise ValueError(f"{self.width}x{self.height} is not a positive multiple of 32")
        if self.width % spatial_ratio or self.height % spatial_ratio:
            raise ValueError(f"{self.width}x{self.height} is not divisible by the VAE ratio {spatial_ratio}")
        video_latent_frames(self.frames)

    def latent(self, spatial_ratio: int) -> tuple[int, int, int]:
        return video_latent_frames(self.frames), self.height // spatial_ratio, self.width // spatial_ratio

    def audio_latents(self) -> int:
        return audio_latents(self.frames)

    def to_json(self) -> dict:
        return {"width": self.width, "height": self.height, "frames": self.frames}

    @classmethod
    def parse(cls, s: str) -> "Geometry":
        try:
            w, h, f = (int(v) for v in s.lower().split("x"))
        except ValueError as e:
            raise ValueError(f"geometry `{s}` is not WIDTHxHEIGHTxFRAMES") from e
        return cls(w, h, f)

    def __str__(self) -> str:
        return f"{self.width}x{self.height}x{self.frames}"


H3_768P = Geometry(1344, 768, 124)
H3_480P = Geometry(832, 480, 124)


@dataclass
class RefGeom:
    kind: str  # "image" | "video" | "audio"
    latent_t: int = 0
    latent_h: int = 0
    latent_w: int = 0
    audio_latents: int = 0


@dataclass
class Layout:
    sequence_length: int
    position_ids: list[tuple[float, float, float]]
    token_tags: list[int]
    video_indices: list[int]
    audio_indices: list[int]
    text_indices: list[int]
    num_condition_video_rows: int
    num_condition_audio_rows: int
    num_video_latent_frames: int
    latent_height: int
    latent_width: int
    num_audio_latents: int
    reference_segments: list[tuple[str, int, tuple[int, int, int]]] = field(default_factory=list)

    def num_text_tokens(self) -> int:
        return len(self.text_indices)

    def num_target_video_rows(self) -> int:
        return len(self.video_indices) - self.num_condition_video_rows

    def num_target_audio_rows(self) -> int:
        return len(self.audio_indices) - self.num_condition_audio_rows


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> list[float]:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    n = dim // patch
    step = ratio / n
    return [(left + i * step) * _ROPE_SPATIAL_SCALE for i in range(n)]


def temporal_position_grid(num_latent_frames: int, origin: float) -> list[float]:
    out, acc = [], 0.0
    for i in range(num_latent_frames):
        out.append(origin + acc)
        acc += _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5]
    return out


def _temporal_span(n: int) -> float:
    return sum(_ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % 5] for i in range(n))


def _frame_grid(lh: int, lw: int, ph: int, pw: int):
    sqrt_area = math.sqrt(lh * lw)
    hg = _spatial_position_grid(lh, ph, sqrt_area)
    wg = _spatial_position_grid(lw, pw, sqrt_area)
    return [(h, w) for h in hg for w in wg], wg


def _fill_audio(pos, start: int, n: int, time0: float, wg):
    for ch in range(AUDIO_CHANNELS):
        wpos = wg[0] if ch == 0 else wg[-1]
        for i in range(n):
            pos[start + ch * n + i] = (time0 + i, 0.0, wpos)


def _check_text_tags(tags):
    if any(t not in (TEXT_TAG, VIDEO_TAG) for t in tags):
        raise ValueError("text token tags may be text or vision only")


def build_layout(text_tags: list[int], lt: int, lh: int, lw: int, na: int, patch: tuple[int, int, int], anchors: list[str]) -> Layout:
    """T2VA / FL2VA: `[text | condition keyframes | audio | video]`; anchors are "first" / "last"."""
    _check_text_tags(text_tags)
    pt, ph, pw = patch
    if lt % pt or lh % ph or lw % pw:
        raise ValueError(f"latent ({lt}, {lh}, {lw}) is not divisible by the patch {patch}")
    rpf = (lh // ph) * (lw // pw)
    n_text, n_cond, n_audio, n_video = len(text_tags), len(anchors) * rpf, na * AUDIO_CHANNELS, (lt // pt) * rpf
    total = n_text + n_cond + n_audio + n_video
    cond_start, audio_start, video_start = n_text, n_text + n_cond, n_text + n_cond + n_audio
    pos = [(0.0, 0.0, 0.0)] * total
    for i in range(n_text):
        pos[i] = (float(i), 0.0, 0.0)
    grid, wg = _frame_grid(lh, lw, ph, pw)
    for k, a in enumerate(anchors):
        if a == "first":
            t = float(n_text)
        elif a == "last":
            t = n_text + _temporal_span(lt) - _ROPE_FRAME_RESCALE
        else:
            raise ValueError(f"anchor {a!r} is not first or last")
        for j, g in enumerate(grid):
            pos[cond_start + k * rpf + j] = (t, g[0], g[1])
    _fill_audio(pos, audio_start, na, float(n_text), wg)
    for f, t in enumerate(temporal_position_grid(lt, float(n_text))):
        for j, g in enumerate(grid):
            pos[video_start + f * rpf + j] = (t, g[0], g[1])
    tags = list(text_tags) + [VIDEO_TAG] * n_cond + [AUDIO_TAG] * n_audio + [VIDEO_TAG] * n_video
    return Layout(total, pos, tags, list(range(cond_start, audio_start)) + list(range(video_start, total)),
                  list(range(audio_start, video_start)), list(range(n_text)), n_cond, 0, lt, lh, lw, na, [])


def build_ref2va_layout(text_tags, refs: list[RefGeom], lt, lh, lw, na, patch, continuation: tuple[int, int] | None = None) -> Layout:
    """Ref2VA: `[text | ordered references | continuation clip | target audio | target video]`."""
    if tuple(patch) != (1, 2, 2):
        raise ValueError(f"Ref2VA requires patch (1, 2, 2), got {patch}")
    _check_text_tags(text_tags)
    if not refs:
        raise ValueError("Ref2VA requires at least one reference")
    if lt == 0 or lh % 2 or lw % 2 or na == 0:
        raise ValueError("Ref2VA target geometry must be positive and patch-aligned")
    n_text = len(text_tags)

    def rows_of(t, h, w):
        if t == 0 or h == 0 or w == 0 or h % 2 or w % 2:
            raise ValueError(f"visual reference geometry ({t}, {h}, {w}) is not patch-aligned")
        return t * (h // 2) * (w // 2)

    rptf = (lh // 2) * (lw // 2)
    n_target_video, n_target_audio = lt * rptf, na * AUDIO_CHANNELS
    n_ref_video = n_ref_audio = 0
    for r in refs:
        if r.kind == "image":
            n_ref_video += rows_of(1, r.latent_h, r.latent_w)
        elif r.kind == "video":
            n_ref_video += rows_of(r.latent_t, r.latent_h, r.latent_w)
            n_ref_audio += r.audio_latents * AUDIO_CHANNELS
        elif r.kind == "audio":
            if r.audio_latents == 0:
                raise ValueError("an audio reference has no latents")
            n_ref_audio += r.audio_latents * AUDIO_CHANNELS
        else:
            raise ValueError(f"reference kind {r.kind!r}")
    if continuation is not None:
        ct, ca = continuation
        if ct == 0 or ct >= lt:
            raise ValueError(f"continuation clip of {ct} latent frames does not fit a target of {lt}")
        n_ref_video += ct * rptf
        n_ref_audio += ca * AUDIO_CHANNELS
    total = n_text + n_ref_video + n_ref_audio + n_target_audio + n_target_video
    pos = [(0.0, 0.0, 0.0)] * total
    for i in range(n_text):
        pos[i] = (float(i), 0.0, 0.0)
    target_grid, target_wg = _frame_grid(lh, lw, 2, 2)
    video_indices, audio_indices, segments = [], [], []
    cursor, clock = n_text, float(n_text)
    for r in refs:
        if r.kind == "image":
            n = rows_of(1, r.latent_h, r.latent_w)
            segments.append(("image", n, (1, r.latent_h, r.latent_w)))
            grid, _ = _frame_grid(r.latent_h, r.latent_w, 2, 2)
            for j, g in enumerate(grid):
                pos[cursor + j] = (clock, g[0], g[1])
            video_indices += list(range(cursor, cursor + n))
            cursor += n
            clock += 1.0
        elif r.kind == "audio":
            n = r.audio_latents * AUDIO_CHANNELS
            segments.append(("audio", n, (0, 0, 0)))
            _fill_audio(pos, cursor, r.audio_latents, clock, target_wg)
            audio_indices += list(range(cursor, cursor + n))
            cursor += n
            clock += float(r.audio_latents)
        else:
            na_r, nv = r.audio_latents * AUDIO_CHANNELS, rows_of(r.latent_t, r.latent_h, r.latent_w)
            if na_r:
                segments.append(("audio", na_r, (0, 0, 0)))
            segments.append(("video", nv, (r.latent_t, r.latent_h, r.latent_w)))
            grid, wg = _frame_grid(r.latent_h, r.latent_w, 2, 2)
            if na_r:
                _fill_audio(pos, cursor, r.audio_latents, clock, wg)
                audio_indices += list(range(cursor, cursor + na_r))
                cursor += na_r
            for f, tt in enumerate(temporal_position_grid(r.latent_t, clock)):
                for j, g in enumerate(grid):
                    pos[cursor + f * len(grid) + j] = (tt, g[0], g[1])
            video_indices += list(range(cursor, cursor + nv))
            cursor += nv
            clock += max(float(r.audio_latents), _temporal_span(r.latent_t))
    if continuation is not None:
        ct, ca = continuation
        na_c, nv = ca * AUDIO_CHANNELS, ct * rptf
        if na_c:
            segments.append(("audio", na_c, (0, 0, 0)))
        segments.append(("continuation", nv, (ct, lh, lw)))
        clock += max(float(ca), _temporal_span(ct))
        if na_c:
            _fill_audio(pos, cursor, ca, clock, target_wg)
            audio_indices += list(range(cursor, cursor + na_c))
            cursor += na_c
        for f, tt in enumerate(temporal_position_grid(ct, clock)):
            for j, g in enumerate(target_grid):
                pos[cursor + f * rptf + j] = (tt, g[0], g[1])
        video_indices += list(range(cursor, cursor + nv))
        cursor += nv
    audio_start = cursor
    video_start = audio_start + n_target_audio
    _fill_audio(pos, audio_start, na, clock, target_wg)
    for f, tt in enumerate(temporal_position_grid(lt, clock)):
        for j, g in enumerate(target_grid):
            pos[video_start + f * rptf + j] = (tt, g[0], g[1])
    audio_indices += list(range(audio_start, video_start))
    video_indices += list(range(video_start, total))
    assert video_start + n_target_video == total
    tags = [VIDEO_TAG] * total
    tags[:n_text] = list(text_tags)
    for i in audio_indices:
        tags[i] = AUDIO_TAG
    for i in video_indices:
        tags[i] = VIDEO_TAG
    return Layout(total, pos, tags, video_indices, audio_indices, list(range(n_text)), n_ref_video, n_ref_audio, lt, lh, lw, na, segments)


def row_timesteps(layout: Layout, video_t: float, audio_t: float, cond_video_t: float, cond_audio_t: float) -> tuple[list[float], list[int]]:
    """Per-row clean times of one step (f32 values); the sorted unique times and each row's index into them."""
    import numpy as np

    f = lambda x: float(np.float32(x))  # noqa: E731
    rows = [f(video_t)] * layout.sequence_length
    for i in layout.video_indices[: layout.num_condition_video_rows]:
        rows[i] = f(cond_video_t)
    for i in layout.audio_indices[layout.num_condition_audio_rows :]:
        rows[i] = f(audio_t)
    for i in layout.audio_indices[: layout.num_condition_audio_rows]:
        rows[i] = f(cond_audio_t)
    unique = sorted(set(rows))
    pos = {u: k for k, u in enumerate(unique)}
    return unique, [pos[r] for r in rows]


@dataclass(frozen=True)
class Run:
    """A contiguous run of packed rows: `(packed start, rows, offset in the modality's own row buffer)`."""

    start: int
    rows: int
    offset: int


def runs(indices: list[int]) -> list[Run]:
    out: list[Run] = []
    for off, i in enumerate(indices):
        if out and out[-1].start + out[-1].rows == i:
            out[-1] = Run(out[-1].start, out[-1].rows + 1, out[-1].offset)
        else:
            out.append(Run(i, 1, off))
    return out


def mod_runs(t_idx: list[int], tags: list[int]) -> list[tuple[int, int, int, int]]:
    """Maximal runs of rows sharing `(timestep index, tag)`: `(start, rows, t_idx, tag)`."""
    out = []
    for i, (t, g) in enumerate(zip(t_idx, tags)):
        if out and out[-1][2] == t and out[-1][3] == g:
            s, n, _, _ = out[-1]
            out[-1] = (s, n + 1, t, g)
        else:
            out.append((i, 1, t, g))
    return out


def patch_rows(z, patch: tuple[int, int, int]):
    """`[C, T, H, W]` latent → `[rows, C·pt·ph·pw]` patch rows (channel-major features)."""
    import numpy as np

    c, t, h, w = z.shape
    pt, ph, pw = patch
    x = z.reshape(c, t // pt, pt, h // ph, ph, w // pw, pw)
    x = x.transpose(1, 3, 5, 0, 2, 4, 6)  # tt, hh, ww, c, kt, kh, kw
    return np.ascontiguousarray(x.reshape((t // pt) * (h // ph) * (w // pw), c * pt * ph * pw))


def unpatch_rows(rows, c: int, t: int, h: int, w: int, patch: tuple[int, int, int]):
    import numpy as np

    pt, ph, pw = patch
    x = rows.reshape(t // pt, h // ph, w // pw, c, pt, ph, pw)
    x = x.transpose(3, 0, 4, 1, 5, 2, 6)
    return np.ascontiguousarray(x.reshape(c, t, h, w))
