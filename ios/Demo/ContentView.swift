// Prompt, size, seed, optional keyframes and the decoder in; the plan, step
// timings, frames and audio out. Frames show as they arrive and play at 24 fps
// with the audio when the clip is done. Generate is enabled only when the
// plan fits: a phone cannot hold the 33 B model; the tiny export runs.

import PhotosUI
import SwiftUI
import H3LiteRTCore

struct ContentView: View {
    @StateObject private var model = GenerationModel()
    @State private var firstItem: PhotosPickerItem?
    @State private var lastItem: PhotosPickerItem?

    var body: some View {
        NavigationStack {
            Form {
                Section("Export") {
                    TextField("Documents folder of the graphs", text: $model.graphsFolder)
                        .autocorrectionDisabled()
                        .onSubmit { model.refresh() }
                    Text(model.status).font(.footnote).foregroundStyle(.secondary)
                    Button("Rescan Documents") { model.refresh() }
                }
                Section("Prompt") {
                    if model.gallery.isEmpty {
                        Text("No prompt embeddings in Documents/gallery (h3-litert encode writes them).")
                            .font(.footnote).foregroundStyle(.secondary)
                    } else {
                        Picker("Embedding", selection: $model.prompt) {
                            ForEach(model.gallery) { Text("\($0.name) (\($0.rows) rows)").tag($0.name) }
                        }
                        .onChange(of: model.prompt) { _ in model.updatePlan() }
                    }
                    Picker("Size", selection: $model.size) {
                        ForEach(model.sizes, id: \.self) { Text($0.description).tag($0) }
                    }
                    .onChange(of: model.size) { _ in model.updatePlan() }
                    HStack {
                        Text("Seed")
                        Spacer()
                        TextField("seed", text: $model.seedText)
                            .keyboardType(.numberPad)
                            .multilineTextAlignment(.trailing)
                    }
                    Toggle("Preview decoder (TAEH3)", isOn: $model.preview)
                        .onChange(of: model.preview) { _ in model.updatePlan() }
                    Toggle("Metal delegate (GPU)", isOn: $model.useMetal)
                }
                Section("Keyframes (FL2VA)") {
                    keyframeRow("First frame", item: $firstItem, image: model.firstFrame) { model.firstFrame = $0 }
                    keyframeRow("Last frame", item: $lastItem, image: model.lastFrame) { model.lastFrame = $0 }
                    Text("Photos are centre-cropped to the canvas, encoded with the VAE encoder graphs and packed as condition rows.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
                if !model.plan.isEmpty {
                    Section(model.planFits ? "Plan" : "Plan (does not fit)") {
                        Text(model.plan).font(.system(.footnote, design: .monospaced))
                            .foregroundStyle(model.planFits ? Color.primary : Color.red)
                    }
                }
                Section {
                    if model.running {
                        HStack {
                            ProgressView()
                            Text(model.status).padding(.leading, 8)
                            Spacer()
                            Button("Cancel") { model.cancel() }
                        }
                    } else {
                        Button("Generate") { model.generate() }
                            .disabled(model.manifest == nil || model.prompt.isEmpty || !model.planFits)
                    }
                }
                if !model.timings.isEmpty {
                    Section("Timings") {
                        ForEach(model.timings) { t in
                            HStack {
                                Text(t.label)
                                Spacer()
                                Text(String(format: "%.0f ms", t.milliseconds)).monospacedDigit()
                            }
                        }
                    }
                }
                if !model.frames.isEmpty {
                    Section(model.done ? "Clip (\(model.frames.count) frames, 24 fps)" : "Frames so far (\(model.frames.count))") {
                        let index = min(model.frameIndex, model.frames.count - 1)
                        Image(decorative: model.frames[index], scale: 1.0)
                            .resizable()
                            .interpolation(.none)
                            .aspectRatio(contentMode: .fit)
                            .frame(maxWidth: .infinity)
                        if model.done {
                            HStack {
                                Button(model.playing ? "Pause" : "Play") {
                                    if model.playing { model.stopPlayback() } else { model.startPlayback() }
                                }
                                Spacer()
                                Text("frame \(index + 1)/\(model.frames.count)").monospacedDigit().foregroundStyle(.secondary)
                            }
                        }
                    }
                }
                if let a = model.audio {
                    Section("Audio (\(a.channels) ch, \(a.samples) samples, \(a.sampleRate) Hz)") {
                        HStack {
                            Button(model.audioPlaying ? "Stop" : "Play audio") {
                                if model.audioPlaying { model.stopAudio() } else { model.playAudio() }
                            }
                            Spacer()
                            Button("Save WAV") { model.saveWav() }
                        }
                    }
                }
            }
            .navigationTitle("H3 LiteRT")
            .onAppear { model.refresh() }
        }
    }

    @ViewBuilder
    private func keyframeRow(_ title: String, item: Binding<PhotosPickerItem?>, image: CGImage?, set: @escaping (CGImage?) -> Void) -> some View {
        HStack {
            PhotosPicker(selection: item, matching: .images) {
                Text(title)
            }
            Spacer()
            if let img = image {
                Image(decorative: img, scale: 1.0)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(height: 44)
                Button("Clear") {
                    set(nil)
                    model.updatePlan()
                }
            } else {
                Text("none").foregroundStyle(.secondary)
            }
        }
        .onChange(of: item.wrappedValue) { newItem in
            guard let newItem = newItem else { return }
            Task {
                let data = try? await newItem.loadTransferable(type: Data.self)
                let cg = data.flatMap { FrameImage.cgImage(from: $0) }
                await MainActor.run {
                    set(cg)
                    model.updatePlan()
                }
            }
        }
    }
}
