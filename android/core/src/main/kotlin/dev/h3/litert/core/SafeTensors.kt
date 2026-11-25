package dev.h3.litert.core

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * A tiny safetensors reader: 8-byte little-endian header length, a JSON header
 * (`name → {dtype, shape, data_offsets}` plus `__metadata__`), then the raw little-endian data.
 * Enough for prompt galleries (`prompt_embeds [rows, 5120]`, `text_tags [rows]`), `adaln.safetensors` (F16) and the tiny fixture.
 */
public class SafeTensors private constructor(
    private val entries: Map<String, Entry>,
    public val metadata: Map<String, String>,
    private val data: ByteBuffer,
) {
    public class Entry(public val dtype: String, public val shape: IntArray, public val begin: Long, public val end: Long) {
        public val byteCount: Long get() = end - begin
    }

    public val names: Set<String> get() = entries.keys

    public operator fun contains(name: String): Boolean = name in entries

    public fun entry(name: String): Entry =
        entries[name] ?: throw IllegalArgumentException("safetensors has no tensor `$name` (it has ${entries.keys.sorted()})")

    public fun shape(name: String): IntArray = entry(name).shape.copyOf()

    /** The raw bytes of a tensor (any dtype). */
    public fun bytes(name: String): ByteArray {
        val e = entry(name)
        val out = ByteArray(e.byteCount.toInt())
        val dup = data.duplicate()
        dup.position(e.begin.toInt())
        dup.get(out)
        return out
    }

    /** An `F32`, `F16` or `BF16` tensor as float32. */
    public fun tensor(name: String): Tensor {
        val e = entry(name)
        val n = Tensor.numel(e.shape)
        val buf = data.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        buf.position(e.begin.toInt())
        val out = FloatArray(n)
        when (e.dtype) {
            "F32" -> {
                require(e.byteCount == 4L * n) { "`$name`: ${e.byteCount} bytes for $n F32 values" }
                buf.asFloatBuffer().get(out)
            }
            "F16" -> {
                require(e.byteCount == 2L * n) { "`$name`: ${e.byteCount} bytes for $n F16 values" }
                for (i in 0 until n) out[i] = halfToFloat(buf.getShort())
            }
            "BF16" -> {
                require(e.byteCount == 2L * n) { "`$name`: ${e.byteCount} bytes for $n BF16 values" }
                for (i in 0 until n) out[i] = Float.fromBits((buf.getShort().toInt() and 0xFFFF) shl 16)
            }
            else -> throw IllegalArgumentException("`$name` is ${e.dtype}; tensor() reads F32, F16 or BF16 (use bytes() for ${e.dtype})")
        }
        return Tensor(out, e.shape.copyOf())
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = true }

        public fun read(file: File): SafeTensors = file.inputStream().buffered().use { read(it) }

        public fun read(input: InputStream): SafeTensors = read(input.readBytes())

        public fun read(bytes: ByteArray): SafeTensors {
            require(bytes.size >= 8) { "safetensors: ${bytes.size} bytes is too short for a header" }
            val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val headerLen = bb.getLong(0)
            require(headerLen in 0..(bytes.size - 8).toLong()) { "safetensors: header length $headerLen does not fit in ${bytes.size} bytes" }
            val headerText = String(bytes, 8, headerLen.toInt(), Charsets.UTF_8)
            val root = json.parseToJsonElement(headerText).jsonObject
            val dataStart = 8 + headerLen.toInt()
            val dataLen = bytes.size - dataStart
            val entries = LinkedHashMap<String, Entry>()
            var metadata: Map<String, String> = emptyMap()
            for ((k, v) in root) {
                if (k == "__metadata__") {
                    metadata = (v as JsonObject).mapValues { it.value.jsonPrimitive.content }
                    continue
                }
                val o = v.jsonObject
                val dtype = o["dtype"]!!.jsonPrimitive.content
                val shape = o["shape"]!!.jsonArray.map { it.jsonPrimitive.content.toInt() }.toIntArray()
                val offs = o["data_offsets"]!!.jsonArray.map { it.jsonPrimitive.content.toLong() }
                require(offs.size == 2 && offs[0] in 0..offs[1] && offs[1] <= dataLen) {
                    "safetensors: `$k` data_offsets $offs outside the $dataLen data bytes"
                }
                entries[k] = Entry(dtype, shape, offs[0], offs[1])
            }
            val data = ByteBuffer.wrap(bytes, dataStart, dataLen).slice().order(ByteOrder.LITTLE_ENDIAN)
            return SafeTensors(entries, metadata, data)
        }

        /** IEEE 754 binary16 → float32 (Android has no `Float.float16ToFloat` below API 34 / JDK 20). */
        public fun halfToFloat(h: Short): Float {
            val bits = h.toInt() and 0xFFFF
            val sign = (bits ushr 15) and 1
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x3FF
            val f = when (exp) {
                0 -> if (mant == 0) 0.0f else mant.toFloat() * (1.0f / (1 shl 24))
                0x1F -> if (mant == 0) Float.POSITIVE_INFINITY else Float.NaN
                else -> Float.fromBits(((exp + 112) shl 23) or (mant shl 13))
            }
            return if (sign == 1) -f else f
        }
    }
}
