package dev.h3.litert.core

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs

/**
 * `adaln.safetensors` (`docs/graphs.md`): `timesteps [U]` f32, `blocks [U, layers, 3, 6, d]` f16
 * and `head [U, 2, d]` f16. The f16 tables stay as bytes; a `[6, d]` / `[2, d]` row is decoded to
 * f32 when a run asks for it, so the 170 MB table of an 8-step student is never doubled in memory.
 */
public class AdalnTables(st: SafeTensors) {
    public val timesteps: FloatArray
    public val layers: Int
    public val hidden: Int
    public val entries: Int get() = timesteps.size

    private val blocks: ByteBuffer
    private val blocksShape: IntArray
    private val head: ByteBuffer
    private val headShape: IntArray

    init {
        timesteps = st.tensor("timesteps").data
        blocksShape = st.shape("blocks")
        headShape = st.shape("head")
        require(blocksShape.size == 5 && blocksShape[2] == MODALITIES && blocksShape[3] == MOD_SLOTS) {
            "adaln blocks must be [U, layers, 3, 6, d], got ${blocksShape.contentToString()}"
        }
        require(headShape.size == 3 && headShape[1] == 2) { "adaln head must be [U, 2, d], got ${headShape.contentToString()}" }
        require(blocksShape[0] == timesteps.size && headShape[0] == timesteps.size) {
            "adaln tables cover ${blocksShape[0]} / ${headShape[0]} timesteps but `timesteps` lists ${timesteps.size}"
        }
        layers = blocksShape[1]
        hidden = blocksShape[4]
        require(headShape[2] == hidden) { "adaln head width ${headShape[2]} vs blocks ${hidden}" }
        require(st.entry("blocks").dtype == "F16" && st.entry("head").dtype == "F16") {
            "adaln blocks / head must be F16 (got ${st.entry("blocks").dtype} / ${st.entry("head").dtype})"
        }
        blocks = ByteBuffer.wrap(st.bytes("blocks")).order(ByteOrder.LITTLE_ENDIAN)
        head = ByteBuffer.wrap(st.bytes("head")).order(ByteOrder.LITTLE_ENDIAN)
    }

    /** The table must match the model it modulates. */
    public fun validate(cfg: H3Config, manifestTimesteps: List<Float>) {
        require(layers == cfg.layers && hidden == cfg.hidden) {
            "adaln table is for $layers layers of $hidden, the model has ${cfg.layers} of ${cfg.hidden}"
        }
        require(manifestTimesteps.size == timesteps.size && manifestTimesteps.indices.all { abs(manifestTimesteps[it] - timesteps[it]) <= Manifest.ADALN_ATOL }) {
            "adaln.safetensors timesteps ${timesteps.toList()} differ from the manifest's $manifestTimesteps"
        }
    }

    /** Index of the clean time `t` (within 1e-6), or an error naming what the table covers. */
    public fun indexOf(t: Float): Int {
        for (i in timesteps.indices) if (abs(timesteps[i] - t) <= Manifest.ADALN_ATOL) return i
        throw IllegalArgumentException("no AdaLN table row for timestep $t; this export covers ${timesteps.toList()}")
    }

    private fun decode(buf: ByteBuffer, elementOffset: Int, n: Int, shape: IntArray): Tensor {
        val out = FloatArray(n)
        var p = elementOffset * 2
        for (i in 0 until n) { out[i] = SafeTensors.halfToFloat(buf.getShort(p)); p += 2 }
        return Tensor(out, shape)
    }

    /** `blocks[u, layer, tag]` as f32 `[6, d]`. */
    public fun mods(u: Int, layer: Int, tag: Int): Tensor {
        require(u in 0 until entries && layer in 0 until layers && tag in 0 until MODALITIES) { "adaln mods($u, $layer, $tag) out of range" }
        val off = ((u * layers + layer) * MODALITIES + tag) * MOD_SLOTS * hidden
        return decode(blocks, off, MOD_SLOTS * hidden, intArrayOf(MOD_SLOTS, hidden))
    }

    /** `head[u]` as f32 `[2, d]` (shift, scale). */
    public fun headMods(u: Int): Tensor {
        require(u in 0 until entries) { "adaln headMods($u) out of range" }
        return decode(head, u * 2 * hidden, 2 * hidden, intArrayOf(2, hidden))
    }

    public companion object {
        public fun load(file: File): AdalnTables = AdalnTables(SafeTensors.read(file))

        public fun read(bytes: ByteArray): AdalnTables = AdalnTables(SafeTensors.read(bytes))
    }
}
