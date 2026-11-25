/**
 * Deterministic standard-normal noise (h3-core `rng.rs` (the same generator as fastwan-rs)): SplitMix64 →
 * Box–Muller, bit for bit the same stream on every platform, so the same seed
 * gives the same latent in Rust, Python, TypeScript, Kotlin and Swift.
 *
 * The 64-bit state is kept as two unsigned 32-bit halves so the hot loop
 * never allocates a BigInt; `nextU64()` returns a BigInt for callers that
 * want the raw stream (the tests pin the canonical SplitMix64 values).
 */

const GOLDEN_HI = 0x9e3779b9;
const GOLDEN_LO = 0x7f4a7c15;
const M1_HI = 0xbf58476d;
const M1_LO = 0x1ce4e5b9;
const M2_HI = 0x94d049bb;
const M2_LO = 0x133111eb;

const TWO32 = 4294967296;
const TWO53 = 9007199254740992;

export class Rng {
  private hi: number;
  private lo: number;

  constructor(seed: number | bigint) {
    const s = BigInt.asUintN(64, BigInt(seed));
    const hi = Number(s >> 32n) >>> 0;
    const lo = Number(s & 0xffffffffn) >>> 0;
    this.hi = (hi ^ GOLDEN_HI) >>> 0;
    this.lo = (lo ^ GOLDEN_LO) >>> 0;
  }

  /** Advance the state and mix; the result is left in `outHi`/`outLo`. */
  private step(): void {
    // state += GOLDEN (wrapping)
    let lo = this.lo + GOLDEN_LO;
    let hi = this.hi + GOLDEN_HI + (lo >= TWO32 ? 1 : 0);
    lo = lo >>> 0;
    hi = hi >>> 0;
    this.hi = hi;
    this.lo = lo;
    // z = (z ^ (z >> 30)) * M1
    let xlo = (lo ^ ((lo >>> 30) | (hi << 2))) >>> 0;
    let xhi = (hi ^ (hi >>> 30)) >>> 0;
    [xhi, xlo] = mul64(xhi, xlo, M1_HI, M1_LO);
    // z = (z ^ (z >> 27)) * M2
    let ylo = (xlo ^ ((xlo >>> 27) | (xhi << 5))) >>> 0;
    let yhi = (xhi ^ (xhi >>> 27)) >>> 0;
    [yhi, ylo] = mul64(yhi, ylo, M2_HI, M2_LO);
    // z ^ (z >> 31)
    this.outLo = (ylo ^ ((ylo >>> 31) | (yhi << 1))) >>> 0;
    this.outHi = (yhi ^ (yhi >>> 31)) >>> 0;
  }

  private outHi = 0;
  private outLo = 0;

  /** The next raw SplitMix64 output. */
  nextU64(): bigint {
    this.step();
    return (BigInt(this.outHi) << 32n) | BigInt(this.outLo);
  }

  /** Uniform in `(0, 1]`: `((next >> 11) + 1) / 2^53`. */
  unit(): number {
    this.step();
    // (z >> 11) = outHi * 2^21 + (outLo >>> 11); exact in a double (< 2^53).
    const top = this.outHi * 2097152 + (this.outLo >>> 11);
    return (top + 1) / TWO53;
  }

  /** `n` standard normals as f32 (Box–Muller in f64, pairs). */
  normal(n: number): Float32Array {
    const pairs = Math.floor((n + 1) / 2);
    const out = new Float32Array(2 * pairs);
    for (let p = 0; p < pairs; p++) {
      const u1 = this.unit();
      const u2 = this.unit();
      const r = Math.sqrt(-2.0 * Math.log(u1));
      const th = 2.0 * Math.PI * u2;
      out[2 * p] = r * Math.cos(th);
      out[2 * p + 1] = r * Math.sin(th);
    }
    return pairs * 2 === n ? out : out.subarray(0, n);
  }

  /** Uniform in `[-a, a)`; for synthetic weights. */
  uniform(n: number, a: number): Float32Array {
    const out = new Float32Array(n);
    const af = Math.fround(a);
    for (let i = 0; i < n; i++) {
      const u = this.unit();
      out[i] = Math.fround(Math.fround(u * 2.0 - 1.0) * af);
    }
    return out;
  }
}

/** `(ahi:alo) · (bhi:blo) mod 2^64` as `[hi, lo]`, both unsigned 32-bit. */
export function mul64(ahi: number, alo: number, bhi: number, blo: number): [number, number] {
  const a0 = alo & 0xffff;
  const a1 = alo >>> 16;
  const b0 = blo & 0xffff;
  const b1 = blo >>> 16;
  const p00 = a0 * b0;
  const mid = a0 * b1 + a1 * b0; // < 2^33, exact
  const p11 = a1 * b1;
  const loFull = p00 + (mid % 65536) * 65536; // < 2^33, exact
  const lo = loFull >>> 0;
  const carry = Math.floor(loFull / TWO32);
  const hiFromLo = p11 + Math.floor(mid / 65536) + carry; // < 2^34, exact
  const hi = (hiFromLo + Math.imul(ahi, blo) + Math.imul(alo, bhi)) >>> 0;
  return [hi, lo];
}
