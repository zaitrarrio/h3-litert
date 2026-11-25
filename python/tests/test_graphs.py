"""The graph cut against the uncut PyTorch pipeline through ai-edge-litert: the
tiny export run by `Pipeline` equals `reference.generate` (video latent, audio
rows, frames, waveform); FL2VA through the encoder graphs likewise."""

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

litert_torch = pytest.importorskip("litert_torch")

from h3_litert import reference  # noqa: E402
from h3_litert.config import AudioVaeConfig, VideoVaeConfig  # noqa: E402
from h3_litert.export import TINY_GEOMETRY, TINY_SEED, TINY_TEXT_TAGS, export_tiny, tiny_prompt  # noqa: E402
from h3_litert.pipeline import Pipeline  # noqa: E402
from h3_litert.request import Keyframe, Request  # noqa: E402
from h3_litert.rng import Rng  # noqa: E402
from h3_litert.runtime import GraphStore, Manifest  # noqa: E402
from h3_litert.variants import by_name  # noqa: E402
from h3_litert.weights import Synthetic  # noqa: E402


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    d = tmp_path_factory.mktemp("tiny")
    export_tiny(d, weights="f32", log=lambda *a: None)
    return d


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-6))


def test_manifest_is_the_contract(tiny):
    m = json.loads((tiny / "manifest.json").read_text())
    assert m["format"] == "h3-litert/1" and m["variant"] == "tiny"
    assert m["chunk"] == 16 and m["head_group"] == 1 and len(m["graphs"]["blocks"]) == 2
    assert m["sampler"]["rule"] == "euler" and len(m["sampler"]["steps"]) == 2
    assert {t["rows"] for t in m["graphs"]["text_refine"]} == {8, 16}
    assert m["video_vae"]["decoder"]["rows"] == 7 * 4 * 6 + 2 + 1
    assert m["audio"][0]["latents"] == 37 and m["taeh3_graphs"][0]["frames"] == 5
    tab = load_file(str(tiny / "adaln.safetensors"))
    assert tab["blocks"].shape == (len(m["adaln"]["timesteps"]), 2, 3, 6, 32) and tab["blocks"].dtype == np.float16


def test_t2va_matches_reference(tiny):
    exp = load_file(str(tiny / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    events = []
    req = Request(exp["prompt_embeds"], exp["text_tags"].astype(int).tolist(), TINY_GEOMETRY, TINY_SEED)
    latent, audio_rows = pipe.denoise(req, events.append)
    assert rel(latent, exp["latent"]) < 1e-4
    assert rel(audio_rows, exp["audio_rows"]) < 1e-4
    frames = pipe.decode_video(latent, on=events.append)
    assert frames.shape == exp["frames"].shape == (22, 64, 96, 3)
    assert np.abs(frames.astype(int) - exp["frames"].astype(int)).max() <= 2
    audio = pipe.decode_audio(audio_rows, TINY_GEOMETRY.audio_latents())
    assert audio.shape == (2, 370) and rel(audio, exp["audio"]) < 1e-3
    kinds = [e["type"] for e in events]
    assert kinds[:3] == ["prepared", "step", "step"] and all(k == "frames" for k in kinds[3:]) and len(kinds) > 3


def test_taeh3_preview_matches_module(tiny):
    import torch

    from h3_litert.config import TaeH3Config
    from h3_litert.tae import TaeH3
    from h3_litert.weights import load_tae

    exp = load_file(str(tiny / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    frames = pipe.decode_video(exp["latent"], preview=True)
    tae = TaeH3(TaeH3Config.tiny()).eval()
    load_tae(tae, Synthetic(0))
    z = torch.from_numpy(exp["latent"])
    with torch.no_grad():
        mems = tae.zero_memory(4, 6)
        out = tae(z[:, :5].permute(1, 0, 2, 3), *mems)
        rgb_a, mems = out[0], list(out[1:])
        z2 = torch.zeros(5, 4, 4, 6)
        z2[:2] = z[:, 5:7].permute(1, 0, 2, 3)
        rgb_b = tae(z2, *mems)[0][:8]
    raw = torch.cat([rgb_a, rgb_b]).numpy()
    keep = [i for i in range(raw.shape[0]) if TaeH3Config.keeps_raw_frame(i)]
    want = (np.clip(raw[keep], 0, 1) * 255 + 0.5).astype(np.uint8).transpose(0, 2, 3, 1)
    assert frames.shape == want.shape == (22, 64, 96, 3)
    assert np.abs(frames.astype(int) - want.astype(int)).max() <= 2


def test_fl2va_keyframe_matches_reference(tiny):
    v = by_name("tiny")
    src = Synthetic(0)
    px = (Rng(99).normal(3 * 64 * 96).reshape(3, 1, 64, 96) * 0.2 + 0.5).clip(0, 1).astype(np.float32)
    kf = Keyframe("first", px)
    exp = reference.generate(v, src, VideoVaeConfig.tiny(), AudioVaeConfig.tiny(), tiny_prompt(v.config), TINY_TEXT_TAGS, TINY_GEOMETRY, TINY_SEED, [kf])
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    latent, audio_rows = pipe.denoise(Request(tiny_prompt(v.config), TINY_TEXT_TAGS, TINY_GEOMETRY, TINY_SEED, [kf]))
    assert rel(latent, exp["latent"]) < 1e-3 and rel(audio_rows, exp["audio_rows"]) < 1e-3


def test_seed_changes_the_clip(tiny):
    exp = load_file(str(tiny / "expected.safetensors"))
    pipe = Pipeline(GraphStore(Manifest(tiny)))
    other, _ = pipe.denoise(Request(exp["prompt_embeds"], exp["text_tags"].astype(int).tolist(), TINY_GEOMETRY, TINY_SEED + 1))
    assert np.abs(other - exp["latent"]).max() > 0.1
