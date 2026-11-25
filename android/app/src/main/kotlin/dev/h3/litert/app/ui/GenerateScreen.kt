package dev.h3.litert.app.ui

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.input.KeyboardOptions
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import android.net.Uri
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import dev.h3.litert.app.GenerateViewModel
import dev.h3.litert.app.PromptFile
import dev.h3.litert.app.UiState
import dev.h3.litert.core.Geometry
import dev.h3.litert.core.Packing
import kotlinx.coroutines.delay
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GenerateScreen(vm: GenerateViewModel) {
    val s by vm.state.collectAsStateWithLifecycle()
    Scaffold(topBar = { TopAppBar(title = { Text("H3 LiteRT") }) }) { padding ->
        Column(
            Modifier.fillMaxSize().padding(padding).verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            FramePlayer(s)
            StatusCard(s, onPlay = vm::playAudio, onStop = vm::stopAudio)
            GraphsCard(s, onRefresh = vm::refresh)
            PromptPicker(s, onPick = vm::selectPrompt)
            SizeAndSeed(s, onSize = vm::selectSize, onSeed = vm::setSeed)
            KeyframeRow(s, onKeyframe = vm::setKeyframe)
            Switches(s, onGpu = vm::setUseGpu, onPreview = vm::setPreview)
            PlanCard(s)
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                Button(onClick = vm::generate, enabled = s.canGenerate, modifier = Modifier.weight(1f)) { Text("Generate") }
                OutlinedButton(onClick = vm::cancel, enabled = s.running, modifier = Modifier.weight(1f)) { Text("Cancel") }
            }
            LogCard(s)
            Spacer(Modifier.height(24.dp))
        }
    }
}

/** Plays the frames that have arrived so far at 24 fps, looping; shows the last one while decoding. */
@Composable
private fun FramePlayer(s: UiState) {
    var index by remember { mutableIntStateOf(0) }
    val frames = s.frames
    LaunchedEffect(frames.size, s.running) {
        if (frames.isEmpty()) { index = 0; return@LaunchedEffect }
        while (true) {
            delay(1000L / Packing.FPS)
            index = if (s.running) frames.lastIndex else (index + 1) % frames.size
        }
    }
    val aspect = if (s.frameWidth > 0 && s.frameHeight > 0) s.frameWidth.toFloat() / s.frameHeight
    else s.size?.let { it.width.toFloat() / it.height } ?: (16f / 9f)
    Box(
        Modifier.fillMaxWidth().aspectRatio(aspect).background(MaterialTheme.colorScheme.surfaceVariant),
        contentAlignment = Alignment.Center,
    ) {
        if (frames.isNotEmpty()) {
            val i = index.coerceIn(0, frames.lastIndex)
            Image(
                bitmap = frames[i].asImageBitmap(),
                contentDescription = "frame ${i + 1} of ${frames.size}",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
            Text(
                "${i + 1}/${frames.size}  ${s.frameWidth}x${s.frameHeight}",
                Modifier.align(Alignment.BottomEnd).padding(6.dp),
                style = MaterialTheme.typography.labelSmall,
            )
        } else {
            Text(if (s.running) "generating…" else "no frames yet", color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
    }
}

@Composable
private fun StatusCard(s: UiState, onPlay: () -> Unit, onStop: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            if (s.running) LinearProgressIndicator(Modifier.fillMaxWidth())
            Text(s.status.ifEmpty { "idle" }, style = MaterialTheme.typography.bodyMedium)
            if (s.prepareMs != null || s.stepMs.isNotEmpty() || s.decodeMs != null) {
                Row(horizontalArrangement = Arrangement.spacedBy(6.dp), modifier = Modifier.fillMaxWidth().horizontalScroll(rememberScrollState())) {
                    s.prepareMs?.let { AssistChip(onClick = {}, label = { Text("prepare ${fmtMs(it)}") }) }
                    s.stepMs.forEachIndexed { i, ms -> AssistChip(onClick = {}, label = { Text("step ${i + 1} ${fmtMs(ms)}") }) }
                    s.decodeMs?.let { AssistChip(onClick = {}, label = { Text("decode ${fmtMs(it)}") }) }
                    s.audioMs?.let { AssistChip(onClick = {}, label = { Text("audio ${fmtMs(it)}") }) }
                }
            }
            val audio = s.audio
            if (audio != null) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text(
                        "audio: ${audio.size} ch × ${audio[0].size} samples @ ${s.sampleRate} Hz (${String.format(Locale.ROOT, "%.2f", audio[0].size / s.sampleRate.toDouble())} s)",
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.weight(1f),
                    )
                    OutlinedButton(onClick = if (s.playing) onStop else onPlay) { Text(if (s.playing) "Stop" else "Play") }
                }
            }
            s.error?.let { Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall) }
        }
    }
}

@Composable
private fun GraphsCard(s: UiState, onRefresh: () -> Unit) {
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Graphs", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                OutlinedButton(onClick = onRefresh, enabled = !s.running) { Text("Rescan") }
            }
            Text(s.graphsDir.path, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            val m = s.manifest
            if (m != null) {
                Text(
                    "${m.model} (${m.variant}) · ${m.weights} weights · chunk ${m.chunk} · head group ${m.headGroup} · " +
                        "${m.graphs.blocks.size} blocks · ${m.sampler.steps.size} ${m.sampler.rule} steps · " +
                        "${m.attention.size} attention / ${m.taeh3Graphs.size} TAEH3 / ${m.audio.size} audio graphs" +
                        (m.videoVae?.let { " · VAE decoder ${it.decoder.blocks.size} blocks" } ?: " · no VAE decoder"),
                    style = MaterialTheme.typography.bodySmall,
                )
            } else {
                Text(s.manifestError ?: "no manifest", color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
                Text(
                    "adb push <export dir>/. ${s.graphsDir.path}/  (see android/README.md)",
                    style = MaterialTheme.typography.bodySmall,
                    fontFamily = FontFamily.Monospace,
                )
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun PromptPicker(s: UiState, onPick: (PromptFile) -> Unit) {
    var open by remember { mutableStateOf(false) }
    ExposedDropdownMenuBox(expanded = open, onExpandedChange = { if (!s.running) open = it }) {
        OutlinedTextField(
            value = s.prompt?.let { "${it.name}  (${it.rows} rows${if (it.visionRows > 0) ", ${it.visionRows} vision" else ""})" }
                ?: if (s.gallery.isEmpty()) "no *.safetensors in ${s.galleryDir.path}" else "",
            onValueChange = {},
            readOnly = true,
            label = { Text("Prompt (gallery/: prompt_embeds + text_tags)") },
            trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = open) },
            modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryNotEditable),
            singleLine = true,
        )
        ExposedDropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            for (p in s.gallery) {
                DropdownMenuItem(text = { Text("${p.name}  (${p.rows} rows)") }, onClick = { onPick(p); open = false })
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun SizeAndSeed(s: UiState, onSize: (Geometry) -> Unit, onSeed: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically) {
        ExposedDropdownMenuBox(expanded = open, onExpandedChange = { if (!s.running) open = it }, modifier = Modifier.weight(1f)) {
            OutlinedTextField(
                value = s.size?.toString() ?: "",
                onValueChange = {},
                readOnly = true,
                label = { Text("Size (manifest sizes)") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = open) },
                modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryNotEditable),
                singleLine = true,
            )
            ExposedDropdownMenu(expanded = open, onDismissRequest = { open = false }) {
                for (g in s.sizes) {
                    val lat = s.manifest?.let { m -> g.latent(m.vae.spatialRatio()) }
                    val detail = lat?.let { "  latent ${it.t}×${it.h}×${it.w}, ${g.audioLatents()} audio latents" } ?: ""
                    DropdownMenuItem(text = { Text("$g$detail") }, onClick = { onSize(g); open = false })
                }
            }
        }
        OutlinedTextField(
            value = s.seed,
            onValueChange = onSeed,
            label = { Text("Seed") },
            singleLine = true,
            enabled = !s.running,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.width(120.dp),
        )
    }
}

/** Optional first / last keyframe pictures (FL2VA); scaled to the canvas, encoded through the VAE encoder graphs. */
@Composable
private fun KeyframeRow(s: UiState, onKeyframe: (String, Uri?) -> Unit) {
    val pickFirst = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> if (uri != null) onKeyframe("first", uri) }
    val pickLast = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri -> if (uri != null) onKeyframe("last", uri) }
    val hasEncoders = s.manifest?.videoVae?.encoderImage != null
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text("Keyframes (FL2VA, optional)", style = MaterialTheme.typography.titleSmall)
            if (!hasEncoders && s.manifest != null) {
                Text("this export has no VAE encoder graphs; text-to-video-and-audio only", style = MaterialTheme.typography.bodySmall)
            }
            KeyframeLine("first", s.firstKeyframe, enabled = hasEncoders && !s.running,
                onPick = { pickFirst.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                onClear = { onKeyframe("first", null) })
            KeyframeLine("last", s.lastKeyframe, enabled = hasEncoders && !s.running && s.firstKeyframe != null,
                onPick = { pickLast.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                onClear = { onKeyframe("last", null) })
            if (s.lastKeyframe != null && s.firstKeyframe == null) {
                Text("a last keyframe needs a first one (the exporter writes attention graphs for 0, 1 or 2 anchors in that order)",
                    color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun KeyframeLine(anchor: String, uri: Uri?, enabled: Boolean, onPick: () -> Unit, onClear: () -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            "$anchor: ${uri?.lastPathSegment ?: "none"}",
            style = MaterialTheme.typography.bodySmall,
            modifier = Modifier.weight(1f),
            fontFamily = FontFamily.Monospace,
        )
        OutlinedButton(onClick = onPick, enabled = enabled) { Text("Pick") }
        OutlinedButton(onClick = onClear, enabled = enabled && uri != null) { Text("Clear") }
    }
}

@Composable
private fun Switches(s: UiState, onGpu: (Boolean) -> Unit, onPreview: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text("GPU (LiteRT CompiledModel, fp16)", modifier = Modifier.weight(1f), style = MaterialTheme.typography.bodyMedium)
        Switch(checked = s.useGpu, onCheckedChange = onGpu, enabled = !s.running)
    }
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text("Preview decoder (TAEH3)", style = MaterialTheme.typography.bodyMedium)
            Text("fast, approximate frames instead of the ViT VAE decoder", style = MaterialTheme.typography.bodySmall)
        }
        Switch(checked = s.preview, onCheckedChange = onPreview, enabled = !s.running && (s.manifest?.taeh3Graphs?.isNotEmpty() ?: false))
    }
}

@Composable
private fun PlanCard(s: UiState) {
    val plan = s.plan
    val err = s.planError
    if (plan == null && err == null) return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            val fits = plan?.fits == true
            Text(
                when {
                    plan == null -> "Memory plan: cannot be computed"
                    fits -> "Memory plan: fits the free memory"
                    else -> "Memory plan: does not fit — Generate is disabled"
                },
                style = MaterialTheme.typography.titleSmall,
                color = if (fits) MaterialTheme.colorScheme.onSurface else MaterialTheme.colorScheme.error,
            )
            Text(plan?.toString() ?: err.orEmpty(), style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            if (plan != null && !fits) {
                Text(
                    "H3 is a 33 B-parameter model: its block graphs alone are 11 GB at int4 and the 768p sequence is ~38 k rows. " +
                        "No phone holds that; this app exists to run the same contract, and only the tiny self-test export " +
                        "(fixtures/tiny, 96×64×22) fits here. The plan says no, and the app does not start what will not fit.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.error,
                )
            }
        }
    }
}

@Composable
private fun LogCard(s: UiState) {
    if (s.log.isEmpty()) return
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            for (line in s.log.takeLast(30)) {
                Text(line, style = MaterialTheme.typography.bodySmall, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

private fun fmtMs(ms: Double): String =
    if (ms >= 1000) String.format(Locale.ROOT, "%.1f s", ms / 1000) else String.format(Locale.ROOT, "%.0f ms", ms)
