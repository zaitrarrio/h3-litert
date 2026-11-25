import Foundation

/// Every error the core raises. Messages name the mismatch; nothing falls back silently.
public enum H3Error: Error, CustomStringConvertible, LocalizedError {
    case invalidGeometry(String)
    case manifest(String)
    case noAttentionGraph(tokens: Int, have: [Int])
    case noTextRefineGraph(rows: Int, largest: Int)
    case noAudioGraph(latents: Int, have: [Int])
    case noTaeGraph(latentH: Int, latentW: Int, have: [String])
    case noAdalnRow(timestep: Float, have: [Float])
    case layout(String)
    case shape(String)
    case graph(String)
    case safetensors(String)
    case io(String)

    public var description: String {
        switch self {
        case .invalidGeometry(let m): return m
        case .manifest(let m): return m
        case .noAttentionGraph(let tokens, let have):
            return "this export has no attention graph for \(tokens) rows (it has \(have)); re-export with --size"
        case .noTextRefineGraph(let rows, let largest):
            return "the prompt has \(rows) rows; this export's largest text_refine bucket is \(largest)"
        case .noAudioGraph(let latents, let have):
            return "this export has no audio VAE graphs for \(latents) latents (it has \(have))"
        case .noTaeGraph(let h, let w, let have):
            return "this export has no TAEH3 graph for latent \(h)x\(w) (it has \(have))"
        case .noAdalnRow(let t, let have):
            return "no AdaLN table row for timestep \(t); this export covers \(have)"
        case .layout(let m): return m
        case .shape(let m): return m
        case .graph(let m): return m
        case .safetensors(let m): return m
        case .io(let m): return m
        }
    }

    public var errorDescription: String? { description }
}
