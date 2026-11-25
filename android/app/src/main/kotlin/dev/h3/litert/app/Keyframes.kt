package dev.h3.litert.app

import android.content.ContentResolver
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import dev.h3.litert.core.Tensor

/** Keyframe images for FL2VA: a picked picture → `[3, 1, H, W]` pixels in `[0, 1]` at the canvas size. */
object Keyframes {
    /** Decode `uri` and scale it (ignoring its aspect ratio) to `width × height`. */
    fun load(resolver: ContentResolver, uri: Uri, width: Int, height: Int): Tensor {
        val decoded = resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it) }
            ?: throw IllegalArgumentException("cannot decode the keyframe image $uri")
        val scaled = if (decoded.width == width && decoded.height == height) decoded else Bitmap.createScaledBitmap(decoded, width, height, true)
        val argb = IntArray(width * height)
        scaled.getPixels(argb, 0, width, 0, 0, width, height)
        return fromArgb(argb, width, height)
    }

    /** `[H·W]` packed ARGB ints → `[3, 1, H, W]` floats in `[0, 1]`. */
    fun fromArgb(argb: IntArray, width: Int, height: Int): Tensor {
        val plane = width * height
        val out = FloatArray(3 * plane)
        for (i in 0 until plane) {
            val p = argb[i]
            out[i] = ((p ushr 16) and 0xFF) / 255f
            out[plane + i] = ((p ushr 8) and 0xFF) / 255f
            out[2 * plane + i] = (p and 0xFF) / 255f
        }
        return Tensor(out, intArrayOf(3, 1, height, width))
    }
}
