"""Component parity at real weights, within a small memory budget: every graph of an export
against the PyTorch module it was cut from, one block at a time (the whole DiT in f32 is
56 GB; a RunPod pod's cgroup allows 61 GB). Inputs are drawn once per graph: normal rows for
`h`, the export's own AdaLN rows for `mods`, a real RoPE table for `cos`/`sin`.

    python scripts/parity_blocks.py --variant fasth3-v2 --model-dir /workspace/weights/h3-8step \
        --graphs graphs/h3-f16 [--blocks 0,1,24,49]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from h3_litert.dit import AttnFf, BlockPost, BlockQkv, H3Dit
from h3_litert.pipeline import AdalnTables
from h3_litert.rope import h3_rope
from h3_litert.runtime import Graph, Manifest
from h3_litert.variants import by_name
from h3_litert.weights import SafetensorsDir, load_attn_ff, load_dit


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="fasth3-v2")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--blocks", help="comma list (default: all)")
    ap.add_argument("--rows", type=int, default=512, help="real rows fed per graph (≤ chunk)")
    ap.add_argument("--out")
    a = ap.parse_args()
    variant = by_name(a.variant)
    cfg = variant.config
    root = Path(a.model_dir)
    tdir = root / "transformer" if (root / "transformer").is_dir() else root
    src = SafetensorsDir(tdir)
    m = Manifest(a.graphs)
    ad = AdalnTables(Path(a.graphs) / m.raw["adaln"]["file"])
    C, d, inner = m.chunk, cfg.hidden, cfg.dim
    n = min(a.rows, C)
    rng = np.random.default_rng(0)
    pos = np.stack([rng.integers(0, 8, n), rng.integers(0, 16, n), rng.integers(0, 16, n)], axis=1).astype(np.float32)
    cos, sin = h3_rope(pos, cfg.rope_freq_dim, cfg.rope_theta)
    pad = lambda x: np.concatenate([x, np.zeros((C - n,) + x.shape[1:], np.float32)], axis=0) if x.shape[0] < C else x  # noqa: E731
    rep: dict = {"variant": variant.name, "weights": m.raw.get("weights"), "chunk": C, "rows": n, "blocks": {}}

    # ---- the non-block graphs ---------------------------------------------------------------------
    dit = H3Dit(cfg, with_blocks=False).eval()
    load_dit(dit, src)
    with torch.no_grad():
        ts = torch.tensor(ad.timesteps, dtype=torch.float32)
        x = dit.time.adaln_input(ts)
        head_t = dit.norm_out_linear(x).reshape(-1, 2, d).numpy()
        rep["adaln_head_rows_rel"] = rel(ad.head.astype(np.float32), head_t)
        # text refiner, every bucket
        rep["text_refine"] = {}
        for entry in m.graphs["text_refine"]:
            T = int(entry["rows"])
            emb = rng.standard_normal((T, cfg.text_dim)).astype(np.float32)
            mask = np.zeros(T, np.float32)
            mask[T // 2 :] = -1e9
            want = dit.text(torch.from_numpy(emb), torch.from_numpy(mask)).numpy()
            (got,) = Graph(m.path(entry["file"])).run(embeds=emb, mask=mask)
            rep["text_refine"][str(T)] = rel(got[: T // 2], want[: T // 2])  # the real rows; padded rows are don't-care
        for name, mod, width in (("embed_video", dit.embed_video, cfg.video_patch_dim()), ("embed_audio", dit.embed_audio, cfg.audio_in_channels)):
            rows = rng.standard_normal((C, width)).astype(np.float32)
            want = mod(torch.from_numpy(rows)).numpy()
            (got,) = Graph(m.path(m.graphs[name])).run(rows=rows)
            rep[name] = rel(got, want)
        for name, mod in (("head_video", dit.head_video), ("head_audio", dit.head_audio)):
            h = rng.standard_normal((C, d)).astype(np.float32)
            mods = ad.head_mods(0)
            want = mod(torch.from_numpy(h), torch.from_numpy(mods)).numpy()
            (got,) = Graph(m.path(m.graphs[name])).run(h=h, mods=mods)
            rep[name] = rel(got, want)
    del dit
    print(json.dumps({k: v for k, v in rep.items() if k != "blocks"}, indent=1), flush=True)

    # ---- the blocks, one at a time ------------------------------------------------------------------
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else list(range(cfg.layers))
    h = rng.standard_normal((C, d)).astype(np.float32)
    o = (0.5 * rng.standard_normal((C, inner))).astype(np.float32)
    cos_c, sin_c = pad(cos), pad(sin)
    worst = 0.0
    for l in blocks:
        t0 = time.time()
        u = AttnFf(cfg).eval()
        load_attn_ff(u, src, f"transformer_blocks.{l}")
        mods = ad.mods(0, l, 0)
        with torch.no_grad():
            q, k, v = (t.numpy() for t in BlockQkv(u)(torch.from_numpy(h), torch.from_numpy(mods), torch.from_numpy(cos_c), torch.from_numpy(sin_c)))
            hp = BlockPost(u)(torch.from_numpy(h), torch.from_numpy(o), torch.from_numpy(mods)).numpy()
        b = m.graphs["blocks"][l]
        gq, gp = Graph(m.path(b["qkv"])), Graph(m.path(b["post"]))
        q_g, k_g, v_g = gq.run(h=h, mods=mods, cos=cos_c, sin=sin_c)
        (hp_g,) = gp.run(h=h, o=o, mods=mods)
        e = {"q": rel(q_g[:n], q[:n]), "k": rel(k_g[:n], k[:n]), "v": rel(v_g[:n], v[:n]), "post": rel(hp_g[:n], hp[:n])}
        e["secs"] = round(time.time() - t0, 1)
        worst = max(worst, e["q"], e["k"], e["v"], e["post"])
        rep["blocks"][str(l)] = e
        print(f"block {l:02d}: q {e['q']:.2e} k {e['k']:.2e} v {e['v']:.2e} post {e['post']:.2e}  ({e['secs']} s)", flush=True)
        del u, gq, gp
    rep["worst_block_rel"] = worst
    print(f"worst block rel error {worst:.2e} over {len(blocks)} blocks")
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
