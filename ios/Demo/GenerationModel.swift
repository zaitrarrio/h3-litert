// The demo's state: the export and gallery in Documents, the chosen prompt,
// size, seed, keyframes and decoder, the memory plan (shown and enforced),
// and one generation at a time running off the main actor.

import Combine
import CoreGraphics
import Foundation
import H3LiteRTCore
#if canImport(TensorFlowLite)
import H3LiteRTRuntime
#endif

struct GalleryEntry: Identifiable, Hashable {
    var id: String { name }
    let name: String
    /// Prompt rows (`prompt_embeds` [rows, text_dim]).
    let rows: Int
}

@MainActor
final class GenerationModel: ObservableObject {
    struct StepTiming: Identifiable {
        let id: Int
        let label: String
        let milliseconds: Double
    }

    /// Documents/<graphs>: the export the user copied with the Files app or Finder.
    @Published var graphsFolder = "graphs"
    @Published var manifest: Manifest?
    @Published var gallery: [GalleryEntry] = []
    @Published var prompt: String = ""
    @Published var size: Geometry = .tiny
    @Published var seedText: String = "7"
    @Published var useMetal = true
    @Published var preview = false
    @Published var firstFrame: CGImage?
    @Published var lastFrame: CGImage?
    @Published var status = "Copy an export (manifest.json, adaln.safetensors, the .tflite graphs) into Documents/graphs and prompt embeddings (.safetensors) into Documents/gallery."
    @Published var plan: String = ""
    @Published var planFits = false
    @Published var timings: [StepTiming] = []
    @Published var frames: [CGImage] = []
    @Published var audio: Audio?
    @Published var running = false
    @Published var done = false
    @Published var playing = false
    @Published var audioPlaying = false
    @Published var frameIndex = 0

    private let audioPlayer = AudioPlayer()
    private var playTimer: Timer?
    private var task: Task<Void, Never>?

    var seed: UInt64 { UInt64(seedText.trimmingCharacters(in: .whitespaces)) ?? 7 }

    var documents: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    var graphsURL: URL { documents.appendingPathComponent(graphsFolder) }
    var galleryURL: URL { documents.appendingPathComponent("gallery") }

    var sizes: [Geometry] {
        if let m = manifest, !m.sizes.isEmpty { return m.sizes }
        return [.tiny]
    }

    /// What an iOS app can count on: half the device's memory, at most 3 GiB.
    var budgetBytes: Int {
        min(Int(ProcessInfo.processInfo.physicalMemory / 2), 3 << 30)
    }

    var promptRows: Int {
        gallery.first(where: { $0.name == prompt })?.rows ?? 0
    }

    var keyframeCount: Int {
        (firstFrame == nil ? 0 : 1) + (lastFrame == nil ? 0 : 1)
    }

    func refresh() {
        do {
            let m = try Manifest(directory: graphsURL)
            manifest = m
            if !m.sizes.contains(size) {
                size = m.sizes.first ?? .tiny
            }
            status = "\(m.model) export (\(m.variant ?? "")), \(m.weights) weights, \(m.graphs.blocks.count) blocks, chunk \(m.chunk), \(m.sampler.steps.count) \(m.sampler.rule.rawValue) steps."
        } catch {
            manifest = nil
            status = "\(error)"
        }
        let files = (try? FileManager.default.contentsOfDirectory(atPath: galleryURL.path)) ?? []
        gallery = files.filter { $0.hasSuffix(".safetensors") }.sorted().compactMap { file in
            let name = String(file.dropLast(".safetensors".count))
            guard let st = try? SafeTensors(url: galleryURL.appendingPathComponent(file)),
                let e = st.entries["prompt_embeds"] ?? st.entries["embeds"], e.shape.count == 2
            else { return nil }
            return GalleryEntry(name: name, rows: e.shape[0])
        }
        if prompt.isEmpty || !gallery.contains(where: { $0.name == prompt }) {
            prompt = gallery.first?.name ?? ""
        }
        updatePlan()
    }

    func updatePlan() {
        guard let m = manifest else {
            plan = ""
            planFits = false
            return
        }
        do {
            let p = try Plan.make(
                manifest: m, geometry: size, textRows: max(promptRows, 1), keyframes: keyframeCount, preview: preview,
                budgetBytes: budgetBytes, residentBlocks: 2
            )
            plan = p.description
            planFits = p.fits
        } catch {
            plan = "\(error)"
            planFits = false
        }
    }

    /// The prompt embeddings and tags of the chosen gallery entry.
    nonisolated static func loadPrompt(_ url: URL) throws -> (embeds: Tensor, tags: [Int]) {
        let st = try SafeTensors(url: url)
        let embeds = try st.float32(st.contains("prompt_embeds") ? "prompt_embeds" : "embeds")
        guard embeds.rank == 2 else {
            throw H3Error.shape("\(url.lastPathComponent): prompt_embeds is \(embeds.shape), expected [rows, text_dim]")
        }
        var tags = [Int](repeating: textTag, count: embeds.rows)
        if st.contains("text_tags") {
            tags = try st.float32("text_tags").data.map { Int($0) }
        }
        return (embeds, tags)
    }

    func keyframes() throws -> [Keyframe] {
        var out: [Keyframe] = []
        for (anchor, image) in [("first", firstFrame), ("last", lastFrame)] {
            guard let img = image else { continue }
            guard let kf = FrameImage.keyframe(anchor: anchor, image: img, width: size.width, height: size.height) else {
                throw H3Error.io("cannot render the \(anchor) keyframe at \(size.width)x\(size.height)")
            }
            out.append(kf)
        }
        return out
    }

    func generate() {
        guard !running else { return }
        guard let m = manifest else {
            status = "No export at \(graphsURL.path)."
            return
        }
        guard !prompt.isEmpty else {
            status = "Pick a prompt embedding from Documents/gallery."
            return
        }
        updatePlan()
        guard planFits else {
            status = "This request does not fit the memory budget (see the plan); a phone cannot hold the 33 B model. The tiny export runs."
            return
        }
        let kfs: [Keyframe]
        do {
            kfs = try keyframes()
        } catch {
            status = "\(error)"
            return
        }
        stopPlayback()
        stopAudio()
        running = true
        done = false
        frames = []
        timings = []
        audio = nil
        frameIndex = 0
        status = "Opening graphs…"
        let promptURL = galleryURL.appendingPathComponent("\(prompt).safetensors")
        let graphsURL = self.graphsURL
        let size = self.size
        let seed = self.seed
        let useMetal = self.useMetal
        let preview = self.preview
        task = Task.detached(priority: .userInitiated) { [weak self] in
            do {
                let (embeds, tags) = try GenerationModel.loadPrompt(promptURL)
                let (store, adaln) = try GenerationModel.openStore(manifest: m, directory: graphsURL, useMetal: useMetal)
                let pipe = Pipeline(store: store, adaln: adaln)
                let req = Request(embeds: embeds, textTags: tags, geometry: size, seed: seed, keyframes: kfs, preview: preview)
                let (video, audio, tm) = try pipe.generate(req) { event in
                    Task { @MainActor [weak self] in self?.handle(event) }
                }
                store.dropBlocks()
                await MainActor.run { [weak self] in
                    guard let self = self else { return }
                    self.timings.append(StepTiming(id: 1000, label: preview ? "decode (TAEH3)" : "decode (VAE)", milliseconds: tm.decodeSeconds * 1000))
                    self.timings.append(StepTiming(id: 1001, label: "audio", milliseconds: tm.audioSeconds * 1000))
                    let total = tm.prepareSeconds + tm.stepSeconds.reduce(0, +) + tm.decodeSeconds + tm.audioSeconds
                    self.status = "\(video.frames) frames \(video.width)x\(video.height), \(audio.samples) samples at \(audio.sampleRate) Hz in \(String(format: "%.1f", total)) s"
                    self.audio = audio
                    self.running = false
                    self.done = true
                    self.startPlayback()
                    self.playAudio()
                }
            } catch {
                await MainActor.run { [weak self] in
                    self?.status = "Failed: \(error)"
                    self?.running = false
                }
            }
        }
    }

    nonisolated static func openStore(manifest: Manifest, directory: URL, useMetal: Bool) throws -> (GraphStore, AdalnTables) {
        #if canImport(TensorFlowLite)
        var options = LiteRTOptions()
        options.useMetal = useMetal
        options.residentBlocks = 2
        let store = LiteRTGraphStore(manifest: manifest, directory: directory, options: options)
        let missing = store.missingFiles()
        guard missing.isEmpty else {
            throw H3Error.io("missing files in \(directory.lastPathComponent): \(missing.prefix(5).joined(separator: ", "))\(missing.count > 5 ? ", …" : "")")
        }
        return (store, try store.adalnTables())
        #else
        throw H3Error.graph("this build has no LiteRT backend: link the TensorFlowLiteSwift pod (see ios/README.md)")
        #endif
    }

    private func handle(_ event: PipelineEvent) {
        switch event {
        case .prepared(let secs, let tokens):
            timings.append(StepTiming(id: -1, label: "prepare (\(tokens) rows)", milliseconds: secs * 1000))
            status = "Denoising…"
        case .step(let index, let total, let secs):
            timings.append(StepTiming(id: index, label: "step \(index + 1)/\(total)", milliseconds: secs * 1000))
            status = index + 1 < total ? "Denoising… step \(index + 1)/\(total) done" : "Decoding…"
        case .frames(let f):
            frames.append(contentsOf: FrameImage.images(of: f))
            frameIndex = max(0, frames.count - 1)
        case .audio:
            break  // delivered with the result
        }
    }

    func cancel() {
        task?.cancel()  // the pipeline has no cancellation points; this only drops the result
        stopPlayback()
        stopAudio()
        running = false
        status = "Cancelled (the current step runs to its end)."
    }

    func startPlayback() {
        guard frames.count > 1 else { return }
        playing = true
        frameIndex = 0
        playTimer?.invalidate()
        playTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / Double(fps), repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self = self, self.playing, !self.frames.isEmpty else { return }
                self.frameIndex = (self.frameIndex + 1) % self.frames.count
            }
        }
    }

    func stopPlayback() {
        playing = false
        playTimer?.invalidate()
        playTimer = nil
    }

    func playAudio() {
        guard let a = audio else { return }
        do {
            try audioPlayer.play(a, loop: true)
            audioPlaying = true
        } catch {
            status = "Audio: \(error)"
            audioPlaying = false
        }
    }

    func stopAudio() {
        audioPlayer.stop()
        audioPlaying = false
    }

    /// The waveform as a WAV file in Documents, for sharing.
    func saveWav() {
        guard let a = audio else { return }
        let url = documents.appendingPathComponent("h3-\(seed).wav")
        do {
            try a.wav().write(to: url)
            status = "Saved \(url.lastPathComponent) to Documents."
        } catch {
            status = "Cannot save the WAV: \(error)"
        }
    }
}
