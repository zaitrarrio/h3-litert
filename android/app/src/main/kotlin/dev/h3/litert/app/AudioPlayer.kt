package dev.h3.litert.app

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/**
 * Plays the decoded stereo waveform (`Event.Audio`: `[2][samples]` floats in `[−1, 1]`) through a
 * static-mode `AudioTrack` in `ENCODING_PCM_FLOAT`.
 *
 * `AudioTrack` accepts sample rates of roughly 4 kHz to 192 kHz; the tiny self-test model's audio
 * VAE runs at 320 Hz, so a rate below [MIN_RATE] is raised by repeating each sample an integer
 * number of times (a zero-order hold), which keeps the clip's duration.
 */
class AudioPlayer {
    private var track: AudioTrack? = null

    /** The rate actually handed to the device (after the integer upsampling). */
    var playbackRate: Int = 0
        private set

    val isPlaying: Boolean get() = track?.playState == AudioTrack.PLAYSTATE_PLAYING

    @Synchronized
    fun play(stereo: Array<FloatArray>, sampleRate: Int) {
        stop()
        require(stereo.size == 2) { "expected 2 channels, got ${stereo.size}" }
        val n = stereo[0].size
        require(stereo[1].size == n) { "channel lengths differ" }
        var repeat = 1
        while (sampleRate * repeat < MIN_RATE) repeat++
        playbackRate = sampleRate * repeat
        val interleaved = FloatArray(n * repeat * 2)
        var o = 0
        for (i in 0 until n) {
            val l = stereo[0][i].coerceIn(-1f, 1f)
            val r = stereo[1][i].coerceIn(-1f, 1f)
            for (k in 0 until repeat) {
                interleaved[o++] = l
                interleaved[o++] = r
            }
        }
        val format = AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
            .setSampleRate(playbackRate)
            .setChannelMask(AudioFormat.CHANNEL_OUT_STEREO)
            .build()
        val attrs = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_MEDIA)
            .setContentType(AudioAttributes.CONTENT_TYPE_MOVIE)
            .build()
        val t = AudioTrack.Builder()
            .setAudioAttributes(attrs)
            .setAudioFormat(format)
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setBufferSizeInBytes(interleaved.size * 4)
            .build()
        val written = t.write(interleaved, 0, interleaved.size, AudioTrack.WRITE_BLOCKING)
        if (written < 0) {
            t.release()
            throw IllegalStateException("AudioTrack.write failed ($written)")
        }
        t.play()
        track = t
    }

    @Synchronized
    fun stop() {
        track?.let {
            try { it.stop() } catch (_: IllegalStateException) {}
            it.release()
        }
        track = null
    }

    companion object {
        const val MIN_RATE = 8000
    }
}
