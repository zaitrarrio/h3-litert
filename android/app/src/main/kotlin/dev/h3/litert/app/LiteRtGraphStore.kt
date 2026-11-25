package dev.h3.litert.app

import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.LiteRtException
import com.google.ai.edge.litert.TensorBuffer
import dev.h3.litert.core.Graph
import dev.h3.litert.core.GraphStore
import dev.h3.litert.core.Manifest
import dev.h3.litert.core.Tensor
import java.io.File

/**
 * `:core`'s [GraphStore] over LiteRT Next's `CompiledModel`, reading graphs from the manifest's
 * directory on external storage.
 *
 * API surface used here was checked against the published `litert-api` 2.1.6 classes:
 *   CompiledModel.create(path: String, options: CompiledModel.Options, env: Environment)
 *   CompiledModel.Options(vararg Accelerator) { gpuOptions = CompiledModel.GpuOptions(precision = …) }
 *   createInputBuffer(inputName: String, signature: String = "")
 *   createOutputBuffers(signature: String)                         // positional: output_0, output_1, …
 *   getInputTensorType(name, signature = "").layout.dimensions    // the graph's shape, to check ours
 *   run(inputs: Map<String, TensorBuffer>, outputs: Map<String, TensorBuffer>, signature: String = "")
 *   TensorBuffer.writeFloat(FloatArray) / readFloat() / close()
 * Behaviour on a device (GPU compile time, fp16 accuracy, buffer reuse across runs) is NOT verified.
 */
class LiteRtGraphStore(
    manifest: Manifest,
    val accelerator: Accelerator = Accelerator.GPU,
    /** Where the GPU accelerator may cache compiled programs; null disables caching. */
    private val cacheDir: File? = null,
    resident: Int = 4,
    /** Fall back to CPU when a graph fails to compile for [accelerator]; reported through [onFallback]. */
    private val cpuFallback: Boolean = true,
    private val onFallback: ((file: String, error: Throwable) -> Unit)? = null,
) : GraphStore(manifest, resident) {

    private val dir: File = manifest.directory
        ?: throw IllegalArgumentException("the manifest must be loaded from a directory (Manifest.load)")

    // One LiteRT environment for every graph of the store.
    private val env: Environment = Environment.create()

    private fun options(acc: Accelerator, file: String): CompiledModel.Options {
        val o = CompiledModel.Options(acc)
        if (acc == Accelerator.GPU) {
            o.gpuOptions = CompiledModel.GpuOptions(
                // Activations are f32 in the graph; fp16 on the GPU is docs/design.md.
                precision = CompiledModel.GpuOptions.Precision.FP16,
                // Block graphs are reopened under the LRU; let the backend cache compiled programs.
                // TODO verify against LiteRT 2.1.6 docs: cache-key semantics of serializationDir/modelCacheKey.
                serializationDir = cacheDir?.absolutePath,
                modelCacheKey = if (cacheDir != null) file else null,
                serializeProgramCache = if (cacheDir != null) true else null,
            )
        }
        return o
    }

    override fun open(file: String): Graph {
        val path = File(dir, file)
        if (!path.isFile) throw IllegalArgumentException("graph `$file` is missing from $dir (adb push it; see android/README.md)")
        val model = try {
            CompiledModel.create(path.absolutePath, options(accelerator, file), env)
        } catch (e: LiteRtException) {
            if (!cpuFallback || accelerator == Accelerator.CPU) throw e
            Log.w(TAG, "$file: ${accelerator.name} compile failed (${e.message}); using CPU", e)
            onFallback?.invoke(file, e)
            CompiledModel.create(path.absolutePath, options(Accelerator.CPU, file), env)
        }
        return LiteRtGraph(file, model)
    }

    override fun close() {
        super.close()
        env.close()
    }

    /** One compiled graph; inputs by name, outputs by position (`output_i`). */
    private class LiteRtGraph(override val name: String, private val model: CompiledModel) : Graph {
        private val inputBuffers = HashMap<String, TensorBuffer>()
        private val inputShapes = HashMap<String, List<Int>>()
        private val outputs: List<TensorBuffer> = model.createOutputBuffers()
        private val outputNames: List<String> = List(outputs.size) { "output_$it" }
        private val outputMap: Map<String, TensorBuffer> = outputNames.zip(outputs).toMap()
        private val outputShapes: List<IntArray> = outputNames.map { n ->
            // `layout` is nullable in the Kotlin API (an unranked tensor); every graph of the contract has a fixed shape.
            val layout = model.getOutputTensorType(n).layout
                ?: throw IllegalStateException("$name: output `$n` has no static shape")
            layout.dimensions.toIntArray()
        }

        private fun input(n: String): TensorBuffer = inputBuffers.getOrPut(n) {
            try {
                inputShapes[n] = model.getInputTensorType(n).layout?.dimensions
                    ?: throw IllegalStateException("$name: input `$n` has no static shape")
                model.createInputBuffer(n)
            } catch (e: LiteRtException) {
                throw IllegalArgumentException("$name has no input `$n` (${e.message})", e)
            }
        }

        @Synchronized
        override fun run(inputs: Map<String, Tensor>): List<Tensor> {
            val bound = HashMap<String, TensorBuffer>(inputs.size)
            for ((n, t) in inputs) {
                val buf = input(n)
                val want = inputShapes[n]!!
                val got = t.shape.toList()
                if (want != got) {
                    throw IllegalArgumentException("$name: input `$n` is $got in the host but $want in the graph (export/manifest mismatch)")
                }
                buf.writeFloat(t.data)
                bound[n] = buf
            }
            if (bound.size != inputBuffers.size) {
                // A previously bound input that this call left out would run on stale data.
                throw IllegalArgumentException("$name: this call bound ${bound.keys.sorted()}, earlier calls also used ${inputBuffers.keys.sorted()}")
            }
            model.run(bound, outputMap)
            return List(outputs.size) { i -> Tensor(outputs[i].readFloat(), outputShapes[i].copyOf()) }
        }

        override fun close() {
            for (b in inputBuffers.values) b.close()
            for (b in outputs) b.close()
            model.close()
        }
    }

    companion object {
        private const val TAG = "LiteRtGraphStore"
    }
}
