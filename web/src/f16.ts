/** IEEE 754 binary16 ↔ binary32 (round to nearest even), for the AdaLN table and the keyframe latent rounding. */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

/** The f32 value of one half-precision bit pattern. */
export function f16BitsToF32(h: number): number {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >>> 10) & 0x1f;
  const mant = h & 0x3ff;
  if (exp === 0) return sign * mant * 2 ** -24;
  if (exp === 31) return mant ? NaN : sign * Infinity;
  return sign * (1 + mant / 1024) * 2 ** (exp - 15);
}

/** The half-precision bit pattern nearest to `x` (ties to even; overflow to ±inf). */
export function f32ToF16Bits(x: number): number {
  f32[0] = x;
  const b = u32[0];
  const sign = (b >>> 16) & 0x8000;
  const exp = (b >>> 23) & 0xff;
  let mant = b & 0x7fffff;
  if (exp === 0xff) return sign | 0x7c00 | (mant ? 0x200 : 0);
  const e = exp - 127 + 15;
  if (e >= 31) return sign | 0x7c00;
  if (e <= 0) {
    if (e < -10) return sign;
    mant |= 0x800000;
    const shift = 14 - e;
    let half = mant >>> shift;
    const rem = mant & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    if (rem > halfway || (rem === halfway && (half & 1) === 1)) half += 1;
    return sign | half;
  }
  let half = sign | (e << 10) | (mant >>> 13);
  const rem = mant & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (half & 1) === 1)) half += 1;
  return half;
}

/** `x.astype(f16).astype(f32)`. */
export function roundF16(x: number): number {
  return f16BitsToF32(f32ToF16Bits(x));
}

export function f16ToF32Array(src: Uint16Array, start = 0, end = src.length): Float32Array {
  const out = new Float32Array(end - start);
  for (let i = start; i < end; i++) out[i - start] = f16BitsToF32(src[i]);
  return out;
}
