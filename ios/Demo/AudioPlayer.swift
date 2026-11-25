// Plays the pipeline's stereo waveform with AVAudioEngine once the clip is
// done. The tiny fixture's audio VAE runs at 320 Hz, below what the engine
// accepts, so a low rate is zero-order-held up to at least 8 kHz first.

import AVFoundation
import Foundation
import H3LiteRTCore

final class AudioPlayer {
    private let engine = AVAudioEngine()
    private let player = AVAudioPlayerNode()
    private var attached = false

    /// Play `audio` from the start; `loop` repeats it until `stop()`.
    func play(_ audio: Audio, loop: Bool = false) throws {
        stop()
        let hold = max(1, (8000 + audio.sampleRate - 1) / audio.sampleRate)
        let rate = Double(audio.sampleRate * hold)
        guard let format = AVAudioFormat(standardFormatWithSampleRate: rate, channels: AVAudioChannelCount(audio.channels)) else {
            throw H3Error.io("cannot make a \(rate) Hz \(audio.channels)-channel audio format")
        }
        let frames = AVAudioFrameCount(audio.samples * hold)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames), let channels = buffer.floatChannelData else {
            throw H3Error.io("cannot allocate an audio buffer of \(frames) frames")
        }
        for c in 0..<audio.channels {
            let src = audio.channel(c)
            let dst = channels[c]
            var i = 0
            for v in src {
                for _ in 0..<hold {
                    dst[i] = v
                    i += 1
                }
            }
        }
        buffer.frameLength = frames
        if !attached {
            engine.attach(player)
            attached = true
        }
        engine.connect(player, to: engine.mainMixerNode, format: format)
        if !engine.isRunning {
            try engine.start()
        }
        player.scheduleBuffer(buffer, at: nil, options: loop ? [.loops] : [], completionHandler: nil)
        player.play()
    }

    func stop() {
        if player.isPlaying {
            player.stop()
        }
    }

    var isPlaying: Bool { player.isPlaying }
}
