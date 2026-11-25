package dev.h3.litert.core

import java.io.File

/** The tiny export's manifest, AdaLN table and expected tensors, copied from `fixtures/tiny`. */
object Fixture {
    fun text(name: String): String =
        Fixture::class.java.getResourceAsStream("/tiny/$name")!!.use { String(it.readBytes(), Charsets.UTF_8) }

    fun bytes(name: String): ByteArray = Fixture::class.java.getResourceAsStream("/tiny/$name")!!.use { it.readBytes() }

    fun manifest(dir: File? = null): Manifest = Manifest.parse(text("manifest.json"), dir)

    fun adaln(): AdalnTables = AdalnTables.read(bytes("adaln.safetensors"))

    fun expected(): SafeTensors = SafeTensors.read(bytes("expected.safetensors"))

    /** The tiny T2VA request of `export_tiny`: the 6-row prompt tagged `[1, 1, 0, 1, 1, 1]`, 96×64×22, seed 7. */
    fun request(seed: Long = 7, keyframes: List<Keyframe> = emptyList(), preview: Boolean = false): Request {
        val e = expected()
        return Request(e.tensor("prompt_embeds"), e.tensor("text_tags").data.map { it.toInt() }, Geometry.TINY, seed, keyframes, preview)
    }

    fun pipeline(store: FakeGraphStore) = Pipeline(store, store.manifest, adaln())
}
