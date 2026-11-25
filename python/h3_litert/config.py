"""Architecture constants (h3-core `config.rs`; each checkpoint's `config.json`,
FastVideo's `configs/models/*/minimax_h3*.py`)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

MOD_SLOTS = 6
MODALITIES = 3
VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2


@dataclass(frozen=True)
class H3Config:
    hidden: int
    heads: int
    head_dim: int
    layers: int
    refiner_layers: int
    ffn_dim: int
    in_channels: int
    audio_in_channels: int
    patch: tuple[int, int, int]
    text_dim: int
    freq_dim: int
    time_embed_hidden: int
    time_embed_dim: int
    rope_freq_dim: int
    rope_theta: float
    norm_eps: float
    qk_norm_eps: float
    final_norm_eps: float
    adaln_rank: int | None = None
    pdd_steps: int | None = None
    kept_block_indices: tuple[int, ...] | None = None

    @property
    def dim(self) -> int:
        """`heads · head_dim` (7168: larger than `hidden`)."""
        return self.heads * self.head_dim

    def rotary_dim(self) -> int:
        return 6 * self.rope_freq_dim

    def video_patch_dim(self) -> int:
        return self.in_channels * self.patch[0] * self.patch[1] * self.patch[2]

    def adaln_in(self) -> int:
        return self.adaln_rank or self.time_embed_dim

    def to_json(self) -> dict:
        d = asdict(self)
        d["patch"] = list(self.patch)
        d["kept_block_indices"] = list(self.kept_block_indices) if self.kept_block_indices else None
        return d

    @classmethod
    def from_json(cls, v: dict) -> "H3Config":
        v = dict(v)
        v["patch"] = tuple(v["patch"])
        if v.get("kept_block_indices"):
            v["kept_block_indices"] = tuple(v["kept_block_indices"])
        return cls(**{k: v[k] for k in cls.__dataclass_fields__ if k in v})

    @classmethod
    def h3(cls) -> "H3Config":
        return cls(5376, 56, 128, 50, 2, 14336, 24, 32, (1, 2, 2), 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5)

    @classmethod
    def fasth3_trim(cls) -> "H3Config":
        kept = (0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35,
                36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49)
        return cls(5376, 56, 128, 42, 2, 14336, 24, 32, (1, 2, 2), 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5, adaln_rank=16, kept_block_indices=kept)

    @classmethod
    def fasth3_omni(cls) -> "H3Config":
        return cls(5376, 56, 128, 50, 2, 14336, 24, 32, (1, 2, 2), 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5, pdd_steps=32)

    @classmethod
    def tiny(cls) -> "H3Config":
        """h3-core's tiny: head_dim 16 with rope_freq_dim 2 rotates 12 of 16."""
        return cls(32, 2, 16, 2, 1, 48, 4, 6, (1, 2, 2), 24, 16, 32, 24, 2, 10000.0, 1e-5, 1e-5, 1e-5)

    @classmethod
    def from_diffusers_json(cls, v: dict) -> "H3Config":
        g = lambda *ks, d=None: next((v[k] for k in ks if k in v), d)  # noqa: E731
        patch = tuple(v.get("patch_size", (1, 2, 2)))
        c = cls(
            hidden=g("hidden_size"), heads=g("num_attention_heads"), head_dim=g("attention_head_dim"), layers=g("num_layers"),
            refiner_layers=g("num_refiner_layers", "token_refiner_num_layers"), ffn_dim=g("ffn_dim", "ffn_hidden_size"),
            in_channels=g("in_channels", "latents_dim"), audio_in_channels=g("audio_in_channels", "audio_latents_dim"),
            patch=patch, text_dim=g("text_dim"), freq_dim=g("freq_dim", "timestep_input_dim"),
            time_embed_hidden=g("time_embed_hidden_dim", "time_embed_hidden_size"), time_embed_dim=g("time_embed_dim"),
            rope_freq_dim=g("rope_freq_dim", "rope_inv_freq_len"), rope_theta=float(v.get("rope_theta", 10000.0)),
            norm_eps=float(v.get("norm_eps", 1e-5)), qk_norm_eps=float(v.get("qk_norm_eps", 1e-5)),
            final_norm_eps=float(v.get("final_norm_eps", 1e-5)), adaln_rank=v.get("adaln_rank"), pdd_steps=v.get("pdd_steps"),
            kept_block_indices=tuple(v["kept_block_indices"]) if v.get("kept_block_indices") else None,
        )
        if c.rotary_dim() > c.head_dim or c.rotary_dim() % 2:
            raise ValueError(f"rotary width {c.rotary_dim()} must be even and <= head_dim {c.head_dim}")
        return c


H3_LATENTS_MEAN = [0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949,
                   -1.3675414323806763, -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736,
                   0.6657370328903198, 0.19690127670764923, -0.5460608005523682, -0.4035342037677765, -0.23683024942874908,
                   0.25928452610969543, -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
                   -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264]
H3_LATENTS_STD = [1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833,
                  2.194143533706665, 0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114,
                  1.8955937623977661, 0.946841835975647, 0.7996809482574463, 0.44988900423049927, 0.7197399735450745,
                  0.6936293244361877, 2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
                  3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523]
PIXEL_MEAN = [0.485, 0.456, 0.406]
PIXEL_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class VideoVaeConfig:
    in_channels: int
    latent_channels: int
    block_out_channels: tuple[int, ...]
    layers_per_block: int
    spatial_down: tuple[int, ...]
    temporal_down: tuple[int, ...]
    norm_groups: int
    norm_eps: float
    decoder_layers: int
    decoder_heads: int
    decoder_head_dim: int
    decoder_register_tokens: int
    decoder_ffn_mult: int
    decoder_rope_theta: float
    decoder_rope_dim_ratio: float
    decoder_norm_eps: float
    clip_length: int
    token_drop: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    def spatial_ratio(self) -> int:
        r = 1
        for s in self.spatial_down:
            r *= s
        return r

    def temporal_ratio(self) -> int:
        r = 1
        for s in self.temporal_down:
            r *= s
        return r

    def decoder_dim(self) -> int:
        return self.decoder_heads * self.decoder_head_dim

    def decoder_rotary_dim(self) -> int:
        return int(self.decoder_head_dim * self.decoder_rope_dim_ratio)

    def frame_pre_padding(self) -> int:
        tr = self.temporal_ratio()
        return (tr - self.clip_length % tr) % tr

    def tokens_chunk(self) -> int:
        return -(-self.clip_length // self.temporal_ratio())

    def token_overlap(self) -> int:
        c = self.tokens_chunk()
        return (c - self.token_drop % c) % c

    def frame_overlap(self) -> int:
        return max(self.token_overlap() * self.temporal_ratio() - self.frame_pre_padding(), 0)

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("block_out_channels", "spatial_down", "temporal_down", "latents_mean", "latents_std"):
            d[k] = list(getattr(self, k))
        return d

    @classmethod
    def from_json(cls, v: dict) -> "VideoVaeConfig":
        v = dict(v)
        for k in ("block_out_channels", "spatial_down", "temporal_down", "latents_mean", "latents_std"):
            v[k] = tuple(v[k])
        return cls(**v)

    @classmethod
    def h3(cls) -> "VideoVaeConfig":
        return cls(3, 24, (128, 256, 256, 512, 512, 1024), 2, (2, 2, 2, 2, 1, 1), (1, 2, 2, 1, 1, 1), 32, 1e-6, 36, 32, 64, 4, 4,
                   100.0, 0.75, 1e-5, 17, 3, tuple(H3_LATENTS_MEAN), tuple(H3_LATENTS_STD))

    @classmethod
    def fasth3_trim(cls) -> "VideoVaeConfig":
        c = cls.h3()
        return cls(**{**asdict(c), "decoder_layers": 26})

    @classmethod
    def tiny(cls) -> "VideoVaeConfig":
        return cls(3, 4, (8, 8, 16, 16, 16, 16), 1, (2, 2, 2, 2, 1, 1), (1, 2, 2, 1, 1, 1), 4, 1e-6, 2, 2, 16, 2, 2, 100.0, 0.75,
                   1e-5, 17, 3, (0.1, -0.2, 0.3, 0.0), (1.1, 0.9, 1.3, 1.0))


AUDIO_LATENTS_MEAN = [-0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671,
                      -0.35782641352446604, 0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091,
                      0.2941778783780663, -0.07901167601970885, -0.056815072777201, -0.3699028221860095, -0.31616315591624855,
                      0.5905951377425391, -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
                      -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344, 0.02868032184767538,
                      -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464,
                      -0.0502599202236253, 0.3979258376211797]
AUDIO_LATENTS_STD = [1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453,
                     2.7788298348882177, 1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833,
                     2.5056497896915633, 1.811019237886178, 1.9579657790720237, 1.6685498243529284, 1.4922469314453364,
                     3.298670198067373, 1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
                     1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912, 1.8318420762504692,
                     1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705,
                     1.8661226051202384, 1.5613768203168363]


@dataclass(frozen=True)
class AudioVaeConfig:
    encoder_dim: int
    encoder_rates: tuple[int, ...]
    latent_dim: int
    latent_channels: int
    decoder_dim: int
    decoder_rates: tuple[int, ...]
    decoder_kernels: tuple[int, ...]
    heads: int
    resblock_kernels: tuple[int, ...]
    resblock_dilations: tuple[tuple[int, ...], ...]
    sample_rate: int
    latents_mean: tuple[float, ...]
    latents_std: tuple[float, ...]

    def hop(self) -> int:
        h = 1
        for r in self.encoder_rates:
            h *= r
        return h

    def to_json(self) -> dict:
        d = asdict(self)
        for k in ("encoder_rates", "decoder_rates", "decoder_kernels", "resblock_kernels", "latents_mean", "latents_std"):
            d[k] = list(getattr(self, k))
        d["resblock_dilations"] = [list(x) for x in self.resblock_dilations]
        return d

    @classmethod
    def from_json(cls, v: dict) -> "AudioVaeConfig":
        v = dict(v)
        for k in ("encoder_rates", "decoder_rates", "decoder_kernels", "resblock_kernels", "latents_mean", "latents_std"):
            v[k] = tuple(v[k])
        v["resblock_dilations"] = tuple(tuple(x) for x in v["resblock_dilations"])
        return cls(**v)

    @classmethod
    def h3(cls) -> "AudioVaeConfig":
        return cls(64, (2, 4, 4, 5, 5), 2048, 32, 1024, (5, 5, 2, 2, 2, 2, 2), (9, 9, 4, 4, 4, 4, 4), 8, (3, 7, 11),
                   ((1, 3, 5), (1, 3, 5), (1, 3, 5)), 32000, tuple(AUDIO_LATENTS_MEAN), tuple(AUDIO_LATENTS_STD))

    @classmethod
    def tiny(cls) -> "AudioVaeConfig":
        return cls(4, (2, 5), 24, 6, 8, (5, 2), (9, 4), 2, (3, 7), ((1, 3), (1, 3)), 320, (0.0, 0.1, -0.1, 0.2, 0.05, -0.3),
                   (1.0, 1.5, 0.8, 1.2, 0.9, 1.1))


@dataclass(frozen=True)
class TaeH3Config:
    latent_channels: int
    widths: tuple[int, int, int, int]
    tgrow: tuple[int, int, int]
    mem_blocks: int
    patch: int

    def t_upscale(self) -> int:
        return self.tgrow[0] * self.tgrow[1] * self.tgrow[2]

    def spatial_scale(self) -> int:
        return 8 * self.patch

    def out_channels(self) -> int:
        return 3 * self.patch * self.patch

    @staticmethod
    def keeps_raw_frame(raw_index: int) -> bool:
        """H3's 17 frames per 5 latents are the raw frames `>= 3 (mod 20)`."""
        return raw_index % 20 >= 3

    def to_json(self) -> dict:
        d = asdict(self)
        d["widths"], d["tgrow"] = list(self.widths), list(self.tgrow)
        return d

    @classmethod
    def from_json(cls, v: dict) -> "TaeH3Config":
        return cls(v["latent_channels"], tuple(v["widths"]), tuple(v["tgrow"]), v["mem_blocks"], v["patch"])

    @classmethod
    def taeh3(cls) -> "TaeH3Config":
        return cls(24, (256, 128, 64, 64), (1, 2, 2), 3, 2)

    @classmethod
    def tiny(cls) -> "TaeH3Config":
        return cls(4, (8, 6, 4, 4), (1, 2, 2), 3, 2)
