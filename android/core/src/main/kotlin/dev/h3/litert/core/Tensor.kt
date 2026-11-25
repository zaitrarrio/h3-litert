package dev.h3.litert.core

/**
 * A dense row-major float32 tensor: the only tensor type that crosses the graph boundary
 * (`docs/graphs.md`: every tensor is float32, shapes are exact, no batch dimension).
 */
public class Tensor(public val data: FloatArray, public val shape: IntArray) {
    init {
        require(shape.isNotEmpty()) { "a tensor needs at least one dimension" }
        require(shape.all { it >= 0 }) { "negative dimension in ${shape.contentToString()}" }
        require(data.size == numel(shape)) {
            "tensor data has ${data.size} floats but shape ${shape.contentToString()} needs ${numel(shape)}"
        }
    }

    public constructor(shape: IntArray) : this(FloatArray(numel(shape)), shape)

    public val size: Int get() = data.size
    public val rank: Int get() = shape.size

    /** Rows along the first axis. */
    public val rows: Int get() = shape[0]

    /** Floats per row along the first axis. */
    public val rowSize: Int get() = if (shape.size == 1) 1 else numel(shape, 1)

    public fun shapeString(): String = shape.joinToString(prefix = "[", postfix = "]")

    /** The same data viewed with another shape of the same element count. */
    public fun reshape(vararg dims: Int): Tensor = Tensor(data, dims)

    /** Rows `[from, to)` of the first axis, copied. */
    public fun sliceRows(from: Int, to: Int): Tensor {
        require(from in 0..to && to <= rows) { "rows $from..$to outside 0..$rows" }
        val rs = rowSize
        val out = data.copyOfRange(from * rs, to * rs)
        return Tensor(out, intArrayOf(to - from) + shape.copyOfRange(1, shape.size))
    }

    public fun copy(): Tensor = Tensor(data.copyOf(), shape.copyOf())

    override fun toString(): String = "Tensor${shapeString()}"

    public companion object {
        public fun numel(shape: IntArray, from: Int = 0): Int {
            var n = 1
            for (i in from until shape.size) n *= shape[i]
            return n
        }

        public fun zeros(vararg shape: Int): Tensor = Tensor(shape)

        public fun scalar(v: Float): Tensor = Tensor(floatArrayOf(v), intArrayOf(1))

        /** Concatenate along the first axis; the trailing dimensions must agree. */
        public fun concatRows(parts: List<Tensor>): Tensor {
            require(parts.isNotEmpty()) { "nothing to concatenate" }
            val tail = parts[0].shape.copyOfRange(1, parts[0].shape.size)
            var rows = 0
            for (p in parts) {
                require(p.shape.copyOfRange(1, p.shape.size).contentEquals(tail)) {
                    "cannot concatenate ${p.shapeString()} with ${parts[0].shapeString()}"
                }
                rows += p.rows
            }
            val out = FloatArray(rows * parts[0].rowSize)
            var o = 0
            for (p in parts) {
                System.arraycopy(p.data, 0, out, o, p.data.size)
                o += p.data.size
            }
            return Tensor(out, intArrayOf(rows) + tail)
        }
    }
}
