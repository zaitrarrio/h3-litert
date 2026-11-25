/**
 * Architecture constants (h3-core `config.rs`; each checkpoint's `config.json`;
 * port of `python/h3_litert/config.py`): the DiT, the ViT video VAE, the audio
 * VAE and the TAEH3 preview decoder.
 */

export const MOD_SLOTS = 6;
export const MODALITIES = 3;
export const VIDEO_TAG = 0;
export const TEXT_TAG = 1;
export const AUDIO_TAG = 2;

export type Patch = readonly [number, number, number];

export interface H3ConfigJson {
  hidden: number;
  heads: number;
  head_dim: number;
  layers: number;
  refiner_layers: number;
  ffn_dim: number;
  in_channels: number;
  audio_in_channels: number;
  patch: number[];
  text_dim: number;
  freq_dim: number;
  time_embed_hidden: number;
  time_embed_dim: number;
  rope_freq_dim: number;
  rope_theta: number;
  norm_eps: number;
  qk_norm_eps: number;
  final_norm_eps: number;
  adaln_rank?: number | null;
  pdd_steps?: number | null;
  kept_block_indices?: number[] | null;
}

/** The MiniMax H3 Omni-Transformer. Batch is always 1; the packed sequence carries every modality. */
export class H3Config {
  constructor(
    readonly hidden: number,
    readonly heads: number,
    readonly headDim: number,
    readonly layers: number,
    readonly refinerLayers: number,
    readonly ffnDim: number,
    readonly inChannels: number,
    readonly audioInChannels: number,
    readonly patch: Patch,
    readonly textDim: number,
    readonly freqDim: number,
    readonly timeEmbedHidden: number,
    readonly timeEmbedDim: number,
    readonly ropeFreqDim: number,
    readonly ropeTheta: number,
    readonly normEps: number,
    readonly qkNormEps: number,
    readonly finalNormEps: number,
    readonly adalnRank: number | null = null,
    readonly pddSteps: number | null = null,
    readonly keptBlockIndices: readonly number[] | null = null,
  ) {
    if (this.rotaryDim() > headDim || this.rotaryDim() % 2 !== 0) {
      throw new Error(`rotary width ${this.rotaryDim()} must be even and <= head_dim ${headDim}`);
    }
  }

  /** `heads · head_dim` (7168: larger than `hidden`). */
  get dim(): number {
    return this.heads * this.headDim;
  }

  /** Rotated channels per head: `6 · rope_freq_dim` (96 of 128). */
  rotaryDim(): number {
    return 6 * this.ropeFreqDim;
  }

  /** `P`: `in_channels · pt · ph · pw` (96). */
  videoPatchDim(): number {
    return this.inChannels * this.patch[0] * this.patch[1] * this.patch[2];
  }

  adalnIn(): number {
    return this.adalnRank ?? this.timeEmbedDim;
  }

  /** Dense parameters of one block (qkv/out projections, the two norms, the SwiGLU FFN). */
  blockParams(): number {
    const d = this.hidden;
    const inner = this.dim;
    return 3 * d * inner + inner * d + 2 * d + 2 * this.headDim + 3 * d * this.ffnDim;
  }

  toJson(): H3ConfigJson {
    return {
      hidden: this.hidden,
      heads: this.heads,
      head_dim: this.headDim,
      layers: this.layers,
      refiner_layers: this.refinerLayers,
      ffn_dim: this.ffnDim,
      in_channels: this.inChannels,
      audio_in_channels: this.audioInChannels,
      patch: [...this.patch],
      text_dim: this.textDim,
      freq_dim: this.freqDim,
      time_embed_hidden: this.timeEmbedHidden,
      time_embed_dim: this.timeEmbedDim,
      rope_freq_dim: this.ropeFreqDim,
      rope_theta: this.ropeTheta,
      norm_eps: this.normEps,
      qk_norm_eps: this.qkNormEps,
      final_norm_eps: this.finalNormEps,
      adaln_rank: this.adalnRank,
      pdd_steps: this.pddSteps,
      kept_block_indices: this.keptBlockIndices ? [...this.keptBlockIndices] : null,
    };
  }

  static fromJson(v: H3ConfigJson): H3Config {
    if (v.patch.length !== 3) throw new Error(`config.patch has ${v.patch.length} entries, expected 3`);
    return new H3Config(
      v.hidden,
      v.heads,
      v.head_dim,
      v.layers,
      v.refiner_layers,
      v.ffn_dim,
      v.in_channels,
      v.audio_in_channels,
      [v.patch[0], v.patch[1], v.patch[2]],
      v.text_dim,
      v.freq_dim,
      v.time_embed_hidden,
      v.time_embed_dim,
      v.rope_freq_dim,
      v.rope_theta,
      v.norm_eps,
      v.qk_norm_eps,
      v.final_norm_eps,
      v.adaln_rank ?? null,
      v.pdd_steps ?? null,
      v.kept_block_indices && v.kept_block_indices.length ? [...v.kept_block_indices] : null,
    );
  }

  /** `MiniMaxAI/MiniMax-H3` and the FastH3 V1 / V2 / Omni students. */
  static h3(): H3Config {
    return new H3Config(5376, 56, 128, 50, 2, 14336, 24, 32, [1, 2, 2], 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5);
  }

  /** FastH3-Trim: 42 kept blocks, rank-16 AdaLN. */
  static fasth3Trim(): H3Config {
    const kept = [
      0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
      40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
    ];
    return new H3Config(5376, 56, 128, 42, 2, 14336, 24, 32, [1, 2, 2], 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5, 16, null, kept);
  }

  /** FastH3-Omni: Parallel Decoding Distillation with 32 fused heads. */
  static fasth3Omni(): H3Config {
    return new H3Config(5376, 56, 128, 50, 2, 14336, 24, 32, [1, 2, 2], 5120, 256, 5376, 2688, 16, 10000.0, 1e-5, 1e-5, 1e-5, null, 32);
  }

  /** h3-core's tiny: head_dim 16 with rope_freq_dim 2 rotates 12 of 16. */
  static tiny(): H3Config {
    return new H3Config(32, 2, 16, 2, 1, 48, 4, 6, [1, 2, 2], 24, 16, 32, 24, 2, 10000.0, 1e-5, 1e-5, 1e-5);
  }
}

export const H3_LATENTS_MEAN: readonly number[] = [
  0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075, -0.2727581858634949, -1.3675414323806763,
  -0.2553254961967468, -0.26907554268836975, -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
  -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543, -0.30133944749832153, 0.211341992020607,
  -1.1206848621368408, 0.3581933379173279, -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264,
];
export const H3_LATENTS_STD: readonly number[] = [
  1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037, 1.5636216402053833, 2.194143533706665,
  0.9653137922286987, 1.0569885969161987, 0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
  0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877, 2.961095094680786, 2.7694199085235596,
  3.0496184825897217, 2.1088054180145264, 3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523,
];
export const PIXEL_MEAN: readonly [number, number, number] = [0.485, 0.456, 0.406];
export const PIXEL_STD: readonly [number, number, number] = [0.229, 0.224, 0.225];

export interface VideoVaeConfigJson {
  in_channels: number;
  latent_channels: number;
  block_out_channels: number[];
  layers_per_block: number;
  spatial_down: number[];
  temporal_down: number[];
  norm_groups: number;
  norm_eps: number;
  decoder_layers: number;
  decoder_heads: number;
  decoder_head_dim: number;
  decoder_register_tokens: number;
  decoder_ffn_mult: number;
  decoder_rope_theta: number;
  decoder_rope_dim_ratio: number;
  decoder_norm_eps: number;
  clip_length: number;
  token_drop: number;
  latents_mean: number[];
  latents_std: number[];
}

/** `AutoencoderKLMiniMaxH3`: causal 3-D CNN encoder, ViT decoder. */
export class VideoVaeConfig {
  constructor(
    readonly inChannels: number,
    readonly latentChannels: number,
    readonly blockOutChannels: readonly number[],
    readonly layersPerBlock: number,
    readonly spatialDown: readonly number[],
    readonly temporalDown: readonly number[],
    readonly normGroups: number,
    readonly normEps: number,
    readonly decoderLayers: number,
    readonly decoderHeads: number,
    readonly decoderHeadDim: number,
    readonly decoderRegisterTokens: number,
    readonly decoderFfnMult: number,
    readonly decoderRopeTheta: number,
    readonly decoderRopeDimRatio: number,
    readonly decoderNormEps: number,
    readonly clipLength: number,
    readonly tokenDrop: number,
    readonly latentsMean: readonly number[],
    readonly latentsStd: readonly number[],
  ) {
    if (latentsMean.length !== latentChannels || latentsStd.length !== latentChannels) {
      throw new Error(`vae.latents_mean/std have ${latentsMean.length}/${latentsStd.length} entries for ${latentChannels} channels`);
    }
  }

  spatialRatio(): number {
    return this.spatialDown.reduce((a, b) => a * b, 1);
  }

  temporalRatio(): number {
    return this.temporalDown.reduce((a, b) => a * b, 1);
  }

  decoderDim(): number {
    return this.decoderHeads * this.decoderHeadDim;
  }

  decoderRotaryDim(): number {
    return Math.trunc(this.decoderHeadDim * this.decoderRopeDimRatio);
  }

  /** Frames the first latent of a clip stands for beyond the ratio: `(tr − clip % tr) % tr` (3). */
  framePrePadding(): number {
    const tr = this.temporalRatio();
    return (tr - (this.clipLength % tr)) % tr;
  }

  /** Latent frames per decoder chunk: `ceil(clip / tr)` (5). */
  tokensChunk(): number {
    return Math.ceil(this.clipLength / this.temporalRatio());
  }

  tokenOverlap(): number {
    const c = this.tokensChunk();
    return (c - (this.tokenDrop % c)) % c;
  }

  frameOverlap(): number {
    return Math.max(this.tokenOverlap() * this.temporalRatio() - this.framePrePadding(), 0);
  }

  toJson(): VideoVaeConfigJson {
    return {
      in_channels: this.inChannels,
      latent_channels: this.latentChannels,
      block_out_channels: [...this.blockOutChannels],
      layers_per_block: this.layersPerBlock,
      spatial_down: [...this.spatialDown],
      temporal_down: [...this.temporalDown],
      norm_groups: this.normGroups,
      norm_eps: this.normEps,
      decoder_layers: this.decoderLayers,
      decoder_heads: this.decoderHeads,
      decoder_head_dim: this.decoderHeadDim,
      decoder_register_tokens: this.decoderRegisterTokens,
      decoder_ffn_mult: this.decoderFfnMult,
      decoder_rope_theta: this.decoderRopeTheta,
      decoder_rope_dim_ratio: this.decoderRopeDimRatio,
      decoder_norm_eps: this.decoderNormEps,
      clip_length: this.clipLength,
      token_drop: this.tokenDrop,
      latents_mean: [...this.latentsMean],
      latents_std: [...this.latentsStd],
    };
  }

  static fromJson(v: VideoVaeConfigJson): VideoVaeConfig {
    return new VideoVaeConfig(
      v.in_channels,
      v.latent_channels,
      [...v.block_out_channels],
      v.layers_per_block,
      [...v.spatial_down],
      [...v.temporal_down],
      v.norm_groups,
      v.norm_eps,
      v.decoder_layers,
      v.decoder_heads,
      v.decoder_head_dim,
      v.decoder_register_tokens,
      v.decoder_ffn_mult,
      v.decoder_rope_theta,
      v.decoder_rope_dim_ratio,
      v.decoder_norm_eps,
      v.clip_length,
      v.token_drop,
      [...v.latents_mean],
      [...v.latents_std],
    );
  }

  static h3(): VideoVaeConfig {
    return new VideoVaeConfig(3, 24, [128, 256, 256, 512, 512, 1024], 2, [2, 2, 2, 2, 1, 1], [1, 2, 2, 1, 1, 1], 32, 1e-6, 36, 32, 64, 4, 4, 100.0, 0.75, 1e-5, 17, 3, H3_LATENTS_MEAN, H3_LATENTS_STD);
  }

  /** The Trim checkpoint's 26-block decoder. */
  static fasth3Trim(): VideoVaeConfig {
    const c = VideoVaeConfig.h3();
    return VideoVaeConfig.fromJson({ ...c.toJson(), decoder_layers: 26 });
  }

  static tiny(): VideoVaeConfig {
    return new VideoVaeConfig(3, 4, [8, 8, 16, 16, 16, 16], 1, [2, 2, 2, 2, 1, 1], [1, 2, 2, 1, 1, 1], 4, 1e-6, 2, 2, 16, 2, 2, 100.0, 0.75, 1e-5, 17, 3, [0.1, -0.2, 0.3, 0.0], [1.1, 0.9, 1.3, 1.0]);
  }
}

export const AUDIO_LATENTS_MEAN: readonly number[] = [
  -0.020211687488382354, 0.3876466479950502, -0.04398279799186767, -0.28591514936373, 0.08179686214561671, -0.35782641352446604,
  0.040623809960919084, -0.01552534501956604, -0.223362481667332, 0.1821006842509091, 0.2941778783780663, -0.07901167601970885,
  -0.056815072777201, -0.3699028221860095, -0.31616315591624855, 0.5905951377425391, -0.052139568068853864, 0.013673160263486295,
  -0.03691647864630577, 0.09732660653298163, -0.3394662328788498, -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
  0.02868032184767538, -0.21217779266454084, -0.1678263169941987, 0.3221287889040614, -0.1223055851554907, 0.4356604928128464,
  -0.0502599202236253, 0.3979258376211797,
];
export const AUDIO_LATENTS_STD: readonly number[] = [
  1.6895524230479284, 2.76263727217653, 1.7945344281264435, 1.6801681847309828, 1.6390226546605453, 2.7788298348882177,
  1.7659090095747236, 1.6199757612137327, 2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
  1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373, 1.9491804496832168, 1.8720003270431442,
  1.8334080103291832, 1.6488070416529093, 1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
  1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855, 1.718045989838149, 1.6307219190837705,
  1.8661226051202384, 1.5613768203168363,
];

export interface AudioVaeConfigJson {
  encoder_dim: number;
  encoder_rates: number[];
  latent_dim: number;
  latent_channels: number;
  decoder_dim: number;
  decoder_rates: number[];
  decoder_kernels: number[];
  heads: number;
  resblock_kernels: number[];
  resblock_dilations: number[][];
  sample_rate: number;
  latents_mean: number[];
  latents_std: number[];
}

/** `AutoencoderKLMiniMaxH3Audio`: DAC encoder, BigVGAN decoder, 32 latent channels at 40 Hz. */
export class AudioVaeConfig {
  constructor(
    readonly encoderDim: number,
    readonly encoderRates: readonly number[],
    readonly latentDim: number,
    readonly latentChannels: number,
    readonly decoderDim: number,
    readonly decoderRates: readonly number[],
    readonly decoderKernels: readonly number[],
    readonly heads: number,
    readonly resblockKernels: readonly number[],
    readonly resblockDilations: readonly (readonly number[])[],
    readonly sampleRate: number,
    readonly latentsMean: readonly number[],
    readonly latentsStd: readonly number[],
  ) {
    if (latentsMean.length !== latentChannels || latentsStd.length !== latentChannels) {
      throw new Error(`audio_vae.latents_mean/std have ${latentsMean.length}/${latentsStd.length} entries for ${latentChannels} channels`);
    }
  }

  /** Samples per latent (800 at 32 kHz / 40 Hz). */
  hop(): number {
    return this.encoderRates.reduce((a, b) => a * b, 1);
  }

  toJson(): AudioVaeConfigJson {
    return {
      encoder_dim: this.encoderDim,
      encoder_rates: [...this.encoderRates],
      latent_dim: this.latentDim,
      latent_channels: this.latentChannels,
      decoder_dim: this.decoderDim,
      decoder_rates: [...this.decoderRates],
      decoder_kernels: [...this.decoderKernels],
      heads: this.heads,
      resblock_kernels: [...this.resblockKernels],
      resblock_dilations: this.resblockDilations.map((d) => [...d]),
      sample_rate: this.sampleRate,
      latents_mean: [...this.latentsMean],
      latents_std: [...this.latentsStd],
    };
  }

  static fromJson(v: AudioVaeConfigJson): AudioVaeConfig {
    return new AudioVaeConfig(
      v.encoder_dim,
      [...v.encoder_rates],
      v.latent_dim,
      v.latent_channels,
      v.decoder_dim,
      [...v.decoder_rates],
      [...v.decoder_kernels],
      v.heads,
      [...v.resblock_kernels],
      v.resblock_dilations.map((d) => [...d]),
      v.sample_rate,
      [...v.latents_mean],
      [...v.latents_std],
    );
  }

  static h3(): AudioVaeConfig {
    return new AudioVaeConfig(64, [2, 4, 4, 5, 5], 2048, 32, 1024, [5, 5, 2, 2, 2, 2, 2], [9, 9, 4, 4, 4, 4, 4], 8, [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], 32000, AUDIO_LATENTS_MEAN, AUDIO_LATENTS_STD);
  }

  static tiny(): AudioVaeConfig {
    return new AudioVaeConfig(4, [2, 5], 24, 6, 8, [5, 2], [9, 4], 2, [3, 7], [[1, 3], [1, 3]], 320, [0.0, 0.1, -0.1, 0.2, 0.05, -0.3], [1.0, 1.5, 0.8, 1.2, 0.9, 1.1]);
  }
}

export interface TaeH3ConfigJson {
  latent_channels: number;
  widths: number[];
  tgrow: number[];
  mem_blocks: number;
  patch: number;
}

/** TAEH3 (madebyollin/taehv): TAEHV with 24 latent channels and the ×2 pixel-shuffle head. */
export class TaeH3Config {
  constructor(
    readonly latentChannels: number,
    readonly widths: readonly [number, number, number, number],
    readonly tgrow: readonly [number, number, number],
    readonly memBlocks: number,
    readonly patch: number,
  ) {}

  tUpscale(): number {
    return this.tgrow[0] * this.tgrow[1] * this.tgrow[2];
  }

  spatialScale(): number {
    return 8 * this.patch;
  }

  outChannels(): number {
    return 3 * this.patch * this.patch;
  }

  /** H3's 17 frames per 5 latents are the raw frames `>= 3 (mod 20)`. */
  static keepsRawFrame(rawIndex: number): boolean {
    return rawIndex % 20 >= 3;
  }

  toJson(): TaeH3ConfigJson {
    return { latent_channels: this.latentChannels, widths: [...this.widths], tgrow: [...this.tgrow], mem_blocks: this.memBlocks, patch: this.patch };
  }

  static fromJson(v: TaeH3ConfigJson): TaeH3Config {
    if (v.widths.length !== 4) throw new Error(`taeh3.widths has ${v.widths.length} entries, expected 4`);
    if (v.tgrow.length !== 3) throw new Error(`taeh3.tgrow has ${v.tgrow.length} entries, expected 3`);
    return new TaeH3Config(v.latent_channels, [v.widths[0], v.widths[1], v.widths[2], v.widths[3]], [v.tgrow[0], v.tgrow[1], v.tgrow[2]], v.mem_blocks, v.patch);
  }

  static taeh3(): TaeH3Config {
    return new TaeH3Config(24, [256, 128, 64, 64], [1, 2, 2], 3, 2);
  }

  static tiny(): TaeH3Config {
    return new TaeH3Config(4, [8, 6, 4, 4], [1, 2, 2], 3, 2);
  }
}
