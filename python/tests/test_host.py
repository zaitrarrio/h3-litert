"""Host math against the values h3-rs's tests pin down."""

import numpy as np

from h3_litert.config import H3Config, TEXT_TAG, TaeH3Config, VIDEO_TAG, VideoVaeConfig
from h3_litert.packing import (H3_768P, RefGeom, align_num_frames, audio_latents, build_layout, build_ref2va_layout,
                               canvas_size, mod_runs, patch_rows, resolve_num_frames, row_timesteps, runs, unpatch_rows,
                               video_latent_frames)
from h3_litert.rng import Rng
from h3_litert.rope import h3_rope, vae_decoder_rope
from h3_litert.schedule import (FASTH3_V2_RUNGS, PIKA_KNOTS, PLUG4_KNOTS, PddPlan, SHIFTS_H3, Schedule, shift_sigma)
from h3_litert.timestep import sinusoid


def bits(x: float) -> int:
    return int(np.array([x], np.float32).view(np.uint32)[0])


def test_v2_ladder():
    s = Schedule.from_rungs(FASTH3_V2_RUNGS, 10.0)
    assert s.steps == 8
    assert abs(s.sigmas[0] - shift_sigma(0.999, 10.0)) < 1e-7
    assert abs(s.sigmas[4] - 10.0 * 0.5 / 5.5) < 1e-6
    assert s.sigmas[-1] == 0.0
    assert abs(s.timestep(4) - (1.0 - 10.0 * 0.5 / 5.5)) < 1e-6
    a = Schedule.from_rungs(FASTH3_V2_RUNGS, 3.0)
    assert 0.74 < a.sigmas[4] < 0.76


def test_uniform_and_pika():
    s = Schedule.uniform(50, 12.0)
    assert s.steps == 49 and s.sigmas[0] == 1.0
    p = Schedule.from_rungs(PIKA_KNOTS, 12.0, "renoise")
    assert p.steps == 6 and p.sigmas[0] == 1.0
    assert abs(p.sigmas[5] - shift_sigma(0.04, 12.0)) < 1e-7


def test_plug4_grid_matches_diffusers_bits():
    want = {12.0: [0x3F800000, 0x3F7914C2, 0x3F6C4EC5, 0x3F4CCCCD, 0], 3.0: [0x3F800000, 0x3F666666, 0x3F400000, 0x3F000000, 0]}
    for shift, b in want.items():
        s = Schedule.from_rungs(PLUG4_KNOTS, shift, "renoise")
        assert [bits(x) for x in s.sigmas] == b, shift
        assert Schedule.uniform(5, shift).sigmas == s.sigmas
        for i in range(4):
            assert 1.0 - s.timestep(i) == s.sigmas[i]


def test_pdd_omni_plan():
    p = PddPlan.new([0, 4, 8, 12, 16, 20, 24, 28, 32], SHIFTS_H3)
    assert len(p.fine_grid) - 1 == 32 and p.num_steps() == 8 and p.block(0) == (0, 4)
    assert abs(p.video_sigmas[0] - 0.999) < 1e-6 and p.video_sigmas[-1] == 0.0
    assert all(w < 0 for w in p.video_weights)
    assert abs(sum(PddPlan.fused_weights(p.video_weights, 0, 4)) - 1.0) < 1e-6
    assert abs(sum(p.video_weights[4:8]) - (p.video_sigmas[2] - p.video_sigmas[1])) < 1e-6


def test_frame_and_latent_counts():
    assert (align_num_frames(120), align_num_frames(240), align_num_frames(360)) == (124, 243, 362)
    assert (video_latent_frames(124), video_latent_frames(362)) == (37, 107)
    assert (audio_latents(124), audio_latents(243)) == (207, 405)
    assert resolve_num_frames(5 * 24) == 124


def test_canvas_rules():
    assert canvas_size(16, 9) == (768, 1344)
    assert canvas_size(9, 16) == (1344, 768)
    assert canvas_size(1, 1) == (768, 768)
    h, w = canvas_size(21, 9)
    assert h * w <= 768 * 1344 + 32 * 32 and h % 32 == 0 and w % 32 == 0 and w > h


def test_t2va_layout_768p():
    t, h, w = H3_768P.latent(16)
    assert (t, h, w) == (37, 48, 84)
    l = build_layout([TEXT_TAG] * 10, t, h, w, H3_768P.audio_latents(), (1, 2, 2), [])
    video = 37 * 24 * 42
    assert l.sequence_length == 10 + 2 * 207 + video and len(l.audio_indices) == 414
    assert l.position_ids[3] == (3.0, 0.0, 0.0)
    a0 = l.position_ids[l.audio_indices[0]]
    assert a0[0] == 10.0 and a0[1] == 0.0
    a_r = l.position_ids[l.audio_indices[207]]
    assert a_r[0] == a0[0] and a_r[2] > a0[2]
    v0, v1, v2 = (l.position_ids[l.video_indices[k * 24 * 42]] for k in range(3))
    assert v0[0] == 10.0 and abs(v1[0] - (10 + 5 / 3)) < 1e-9 and abs(v2[0] - (10 + 5 / 3 + 4 * 5 / 3)) < 1e-9


def test_fl2va_anchors_and_row_timesteps():
    l = build_layout([TEXT_TAG, TEXT_TAG, VIDEO_TAG], 7, 4, 6, 20, (1, 2, 2), ["first", "last"])
    assert l.num_condition_video_rows == 12
    assert l.position_ids[l.video_indices[0]][0] == 3.0
    assert abs(l.position_ids[l.video_indices[6]][0] - (3.0 + 5.0 / 3.0 * 22.0 - 5.0 / 3.0)) < 1e-9
    u, inv = row_timesteps(l, 0.3, 0.6, 0.999, 1.0)
    assert [round(x, 6) for x in u] == [0.3, 0.6, 0.999]
    assert inv[l.video_indices[0]] == 2 and inv[l.audio_indices[0]] == 1 and inv[0] == 0
    mr = mod_runs(inv, l.token_tags)
    # text(0,text) | vision placeholder(0,video) | cond video (2, video) | audio (1, audio) | video (0, video)
    assert [(n, t, g) for _, n, t, g in mr] == [(2, 0, 1), (1, 0, 0), (12, 2, 0), (40, 1, 2), (42, 0, 0)]


def test_ref2va_layout_orders_segments():
    refs = [RefGeom("image", 0, 4, 4), RefGeom("video", 2, 4, 6, 3), RefGeom("audio", audio_latents=5)]
    l = build_ref2va_layout([TEXT_TAG] * 4, refs, 7, 4, 6, 20, (1, 2, 2))
    assert l.num_condition_video_rows == 4 + 12 and l.num_condition_audio_rows == 16
    assert [s[0] for s in l.reference_segments] == ["image", "audio", "video", "audio"]
    assert l.sequence_length == 4 + 4 + 6 + 12 + 10 + 40 + 42


def test_runs_and_patch_rows():
    assert [(r.start, r.rows, r.offset) for r in runs([3, 4, 5, 9, 10])] == [(3, 3, 0), (9, 2, 3)]
    z = np.random.default_rng(0).standard_normal((4, 3, 4, 6)).astype(np.float32)
    rows = patch_rows(z, (1, 2, 2))
    assert rows.shape == (3 * 2 * 3, 16)
    assert rows[0, 1 * 4 + 1 * 2 + 0] == z[1, 0, 1, 0]
    assert np.array_equal(unpatch_rows(rows, 4, 3, 4, 6, (1, 2, 2)), z)


def test_rope_tables():
    cos, _ = h3_rope([(0, 0, 0), (1, 2, 3)], 16, 10000.0)
    assert cos.shape == (2, 48) and np.all(cos[0] == 1.0)
    assert abs(cos[1, 0] - np.cos(1.0)) < 1e-6 and abs(cos[1, 16] - np.cos(2.0)) < 1e-6 and abs(cos[1, 32] - np.cos(3.0)) < 1e-6
    c, _ = vae_decoder_rope((1, 2, 2), 48, 100.0, 5)
    assert c.shape == (9, 24)
    assert np.allclose(c[0, :8], 1.0, atol=1e-6) and abs(c[0, 8] + 1.0) < 1e-5
    assert np.allclose(c[4:], 1.0, atol=1e-6)


def test_configs_and_tae_rule():
    assert H3Config.h3().rotary_dim() == 96 and H3Config.tiny().rotary_dim() == 12
    assert H3Config.fasth3_trim().layers == 42 and H3Config.fasth3_omni().pdd_steps == 32
    v = VideoVaeConfig.h3()
    assert (v.spatial_ratio(), v.temporal_ratio(), v.frame_pre_padding(), v.tokens_chunk(), v.token_overlap(), v.frame_overlap()) == (16, 4, 3, 5, 2, 5)
    kept = lambda n: sum(TaeH3Config.keeps_raw_frame(i) for i in range(n * 4))  # noqa: E731
    assert (kept(5), kept(37), kept(2)) == (17, 124, 5)


def test_rng_and_sinusoid():
    r = Rng(0x9E3779B97F4A7C15)
    assert r.next_u64() == 0xE220A8397B1DCDAF
    e = sinusoid(0.5, 8)
    assert abs(e[0] - np.cos(np.float32(0.5))) < 1e-7 and abs(e[4] - np.sin(np.float32(0.5))) < 1e-7
