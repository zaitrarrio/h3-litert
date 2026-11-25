"""Frames and sound out without codecs: `.y4m` (ffmpeg reads it), `.wav`, `.png`."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def rgb_to_yuv444(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float32) for i in range(3))
    y = 16 + (65.481 * r + 128.553 * g + 24.966 * b) / 255.0
    u = 128 + (-37.797 * r - 74.203 * g + 112.0 * b) / 255.0
    v = 128 + (112.0 * r - 93.786 * g - 18.214 * b) / 255.0
    return np.clip(np.stack([y, u, v]) + 0.5, 0, 255).astype(np.uint8)


def write_y4m(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    f, h, w, _ = frames.shape
    with open(path, "wb") as out:
        out.write(f"YUV4MPEG2 W{w} H{h} F{fps}:1 Ip A1:1 C444\n".encode())
        for i in range(f):
            out.write(b"FRAME\n")
            out.write(rgb_to_yuv444(frames[i]).tobytes())


def write_wav(path: Path, stereo: np.ndarray, sample_rate: int) -> None:
    """`[2, samples]` in [−1, 1] → 16-bit PCM stereo."""
    pcm = (np.clip(stereo, -1.0, 1.0) * 32767.0).astype("<i2").T.reshape(-1).tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 2, sample_rate, sample_rate * 4, 4, 16) + b"data" + struct.pack("<I", len(pcm))
    Path(path).write_bytes(header + pcm)


def write_png(path: Path, rgb: np.ndarray) -> None:
    h, w, _ = rgb.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def read_png_rgb(path: Path) -> np.ndarray:
    """A minimal 8-bit RGB / RGBA PNG reader (no interlace) → `[H, W, 3]` uint8."""
    data = Path(path).read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    pos, idat, w = 8, b"", 0
    while pos < len(data):
        n = struct.unpack(">I", data[pos : pos + 4])[0]
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + n]
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            assert depth == 8 and ctype in (2, 6) and interlace == 0, "only 8-bit RGB/RGBA non-interlaced PNGs"
            ch = 3 if ctype == 2 else 4
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    raw = zlib.decompress(idat)
    stride = w * ch
    out = np.zeros((h, stride), np.uint8)
    prev = np.zeros(stride, np.int32)
    p = 0
    for y in range(h):
        flt = raw[p]
        line = np.frombuffer(raw[p + 1 : p + 1 + stride], np.uint8).astype(np.int32)
        p += 1 + stride
        if flt == 0:
            cur = line
        elif flt == 2:
            cur = (line + prev) & 255
        else:
            cur = line.copy()
            for i in range(stride):
                a = cur[i - ch] if i >= ch else 0
                b = prev[i]
                c = prev[i - ch] if i >= ch else 0
                if flt == 1:
                    cur[i] = (line[i] + a) & 255
                elif flt == 3:
                    cur[i] = (line[i] + (a + b) // 2) & 255
                else:
                    pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                    pred = a if pa <= pb and pa <= pc else (b if pb <= pc else c)
                    cur[i] = (line[i] + pred) & 255
        out[y] = cur
        prev = cur
    return out.reshape(h, w, ch)[..., :3]
