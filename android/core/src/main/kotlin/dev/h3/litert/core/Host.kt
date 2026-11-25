package dev.h3.litert.core

/**
 * Host-side tensor plumbing shared by the pipeline and the decoders: chunk padding, the RGB
 * quantisation and the f16 round trip of the keyframe latents. Single precision in the reference's
 * order.
 */
public object Host {
    public class Padded(public val tensor: Tensor, public val realRows: Int)

    /** Zero-pad the first axis to a multiple of `chunk`. */
    public fun padRows(x: Tensor, chunk: Int): Padded {
        require(chunk > 0) { "chunk must be positive" }
        val n = x.rows
        val full = (n + chunk - 1) / chunk * chunk
        if (full == n) return Padded(x, n)
        val rs = x.rowSize
        val data = FloatArray(full * rs)
        System.arraycopy(x.data, 0, data, 0, x.data.size)
        return Padded(Tensor(data, intArrayOf(full) + x.shape.copyOfRange(1, x.shape.size)), n)
    }

    private fun q8(v: Float): Byte {
        val c = if (v < 0.0f) 0.0f else if (v > 1.0f) 1.0f else v
        return (c * 255.0f + 0.5f).toInt().toByte()
    }

    /** `[3, F, H, W]` in `[0, 1]` → `[F, H, W, 3]` uint8 (`(clip(v)·255 + 0.5) as u8`). */
    public fun toRgb8(frames: Tensor): ByteArray = toBytes(frames, 3)

    /** As [toRgb8] but `[F, H, W, 4]` with alpha 255: the byte order of an `ARGB_8888` `Bitmap`. */
    public fun toRgba8(frames: Tensor): ByteArray = toBytes(frames, 4)

    private fun toBytes(frames: Tensor, channels: Int): ByteArray {
        require(frames.rank == 4 && frames.shape[0] == 3) { "wants [3, F, H, W], got ${frames.shapeString()}" }
        val f = frames.shape[1]
        val h = frames.shape[2]
        val w = frames.shape[3]
        val plane = f * h * w
        val out = ByteArray(plane * channels)
        val src = frames.data
        for (i in 0 until plane) {
            val o = i * channels
            out[o] = q8(src[i])
            out[o + 1] = q8(src[i + plane])
            out[o + 2] = q8(src[i + 2 * plane])
            if (channels == 4) out[o + 3] = -1 // 0xFF
        }
        return out
    }

    /** `[F, 3, H, W]` (the TAEH3 order) in `[0, 1]` → `[F, H, W, 4]` uint8, alpha 255. */
    public fun framesToRgba8(frames: Tensor): ByteArray {
        require(frames.rank == 4 && frames.shape[1] == 3) { "framesToRgba8 wants [F, 3, H, W], got ${frames.shapeString()}" }
        val f = frames.shape[0]
        val h = frames.shape[2]
        val w = frames.shape[3]
        val plane = h * w
        val out = ByteArray(f * plane * 4)
        val src = frames.data
        for (i in 0 until f) for (p in 0 until plane) {
            val o = (i * plane + p) * 4
            val s = i * 3 * plane + p
            out[o] = q8(src[s])
            out[o + 1] = q8(src[s + plane])
            out[o + 2] = q8(src[s + 2 * plane])
            out[o + 3] = -1
        }
        return out
    }

    /** `[F, H, W, 3]` uint8 → `[F, H, W, 4]` uint8, alpha 255. */
    public fun rgbToRgba(rgb: ByteArray): ByteArray {
        require(rgb.size % 3 == 0) { "rgb byte count ${rgb.size} is not a multiple of 3" }
        val px = rgb.size / 3
        val out = ByteArray(px * 4)
        for (i in 0 until px) {
            out[i * 4] = rgb[i * 3]
            out[i * 4 + 1] = rgb[i * 3 + 1]
            out[i * 4 + 2] = rgb[i * 3 + 2]
            out[i * 4 + 3] = -1
        }
        return out
    }

    /** IEEE 754 binary16 of `f`, round-to-nearest-even (numpy's `astype(float16)`). */
    public fun floatToHalf(f: Float): Short {
        val bits = f.toRawBits()
        val sign = (bits ushr 16) and 0x8000
        val exp = (bits ushr 23) and 0xFF
        val mant = bits and 0x7FFFFF
        if (exp == 0xFF) return (sign or 0x7C00 or (if (mant != 0) 0x200 else 0)).toShort() // inf / nan
        val e = exp - 127 + 15
        if (e >= 0x1F) return (sign or 0x7C00).toShort() // overflow → inf
        if (e <= 0) {
            if (e < -10) return sign.toShort() // underflow → ±0
            // subnormal half: shift the (implicit-1) mantissa right by 1 − e, rounding to nearest even
            val m = mant or 0x800000
            val shift = 14 - e
            var half = m ushr shift
            val rem = m and ((1 shl shift) - 1)
            val mid = 1 shl (shift - 1)
            if (rem > mid || (rem == mid && (half and 1) == 1)) half++
            return (sign or half).toShort()
        }
        var half = (e shl 10) or (mant ushr 13)
        val rem = mant and 0x1FFF
        if (rem > 0x1000 || (rem == 0x1000 && (half and 1) == 1)) half++ // may carry into the exponent: correct
        return (sign or half).toShort()
    }

    /** `x.astype(np.float16).astype(np.float32)` element-wise. */
    public fun roundThroughHalf(x: Tensor): Tensor =
        Tensor(FloatArray(x.size) { SafeTensors.halfToFloat(floatToHalf(x.data[it])) }, x.shape.copyOf())
}
