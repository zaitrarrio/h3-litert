package dev.h3.litert.core

/**
 * One compiled `.tflite` graph. Inputs are bound **by name** (the signature input names of
 * `docs/graphs.md`), outputs are returned **by position** (`output_0`, `output_1`, …).
 * Implementations must reject inputs whose names or shapes do not match the graph, with the
 * mismatch in the message.
 */
public interface Graph : AutoCloseable {
    /** The file name this graph was opened from. */
    public val name: String

    public fun run(inputs: Map<String, Tensor>): List<Tensor>

    override fun close() {}
}

/**
 * Graphs of one manifest, opened on demand (`runtime.py` `GraphStore`). The fixed graphs
 * (text refiner, embedders, heads, attention, the VAEs, TAEH3) stay open; block graphs beyond [resident] are closed
 * least-recently-used first, so a 50-block model streams through memory a few blocks at a time.
 *
 * Subclasses open a file into a [Graph]; the Android app does this with LiteRT's `CompiledModel`,
 * tests with fakes.
 */
public abstract class GraphStore(
    public val manifest: Manifest,
    resident: Int = 4,
) : AutoCloseable {
    public val resident: Int = maxOf(1, resident)

    private val fixed = HashMap<String, Graph>()
    private val blocks = LinkedHashMap<String, Graph>(16, 0.75f, true) // access order = LRU

    /** Open the graph stored in `file` (a name relative to the manifest directory). */
    protected abstract fun open(file: String): Graph

    /** A graph that stays resident for the life of the store. */
    @Synchronized
    public fun fixed(file: String): Graph = fixed.getOrPut(file) { open(file) }

    /** A block graph under the LRU residency limit. */
    @Synchronized
    public fun block(file: String): Graph {
        blocks[file]?.let { return it }
        while (blocks.size >= resident) {
            val eldest = blocks.keys.first()
            blocks.remove(eldest)?.close()
        }
        val g = open(file)
        blocks[file] = g
        return g
    }

    /** Block graph files currently open, least recently used first. */
    @Synchronized
    public fun residentBlocks(): List<String> = blocks.keys.toList()

    @Synchronized
    public fun dropBlocks() {
        for (g in blocks.values) g.close()
        blocks.clear()
    }

    @Synchronized
    override fun close() {
        dropBlocks()
        for (g in fixed.values) g.close()
        fixed.clear()
    }
}
