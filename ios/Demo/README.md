# Demo app

SwiftUI sources for a small iOS app around the package: pick a prompt
embedding from a gallery, a size, a seed, optionally a first and a last
keyframe photo (FL2VA), the VAE or the TAEH3 preview decoder; see the
memory plan; run `Pipeline.generate` off the main actor; watch the step
timings and the frames as the decoder chunks finish; play the clip at 24 fps
with its audio when done.

The plan is enforced: **Generate** is disabled while `Plan.make` says the
request does not fit the app's budget (half the device's memory, at most
3 GiB). A phone cannot hold the 33 B model, and the plan says so for every
real export; the tiny fixture runs.

The sources are here, not an `.xcodeproj`: create the project in Xcode.

1. Xcode → File → New → Project → iOS → App. Interface SwiftUI, language
   Swift. Name it `H3Demo`. Minimum deployment iOS 16.
2. Delete the generated `H3DemoApp.swift` and `ContentView.swift`; add the
   five Swift files of this folder to the app target (`H3DemoApp.swift`,
   `ContentView.swift`, `GenerationModel.swift`, `FrameImage.swift`,
   `AudioPlayer.swift`).
3. File → Add Package Dependencies… → Add Local… → choose the `ios/` folder
   of this repository. Add the `H3LiteRTCore` and `H3LiteRTRuntime` products
   to the app target.
4. The LiteRT backend. The TensorFlow Lite Swift API is published as a
   CocoaPod, so close Xcode and in the project folder:

   ```ruby
   # Podfile
   platform :ios, '16.0'
   use_frameworks!
   target 'H3Demo' do
     pod 'TensorFlowLiteSwift', '~> 2.17.0', :subspecs => ['Metal']
   end
   ```

   `pod install`, open the `.xcworkspace`. The `H3LiteRTRuntime` sources
   compile their LiteRT code only where `import TensorFlowLite` resolves. A
   Swift package target built by Xcode does not see the pod's framework
   search paths, so if `LiteRTGraphStore` is reported missing, add
   `ios/Sources/H3LiteRTRuntime/LiteRTGraph.swift` directly to the app
   target instead of (or as well as) the package product: the file then
   compiles inside the app, where the pod is linked. `// TODO verify` on a
   real Xcode: this is the integration we could not run here.
5. Info.plist: add `UIFileSharingEnabled = YES` and
   `LSSupportsOpeningDocumentsInPlace = YES` so the app's Documents folder
   appears in the Files app and in Finder (device → Files tab). The keyframe
   picker is `PhotosPicker`, which needs no photo-library usage string.
6. Copy an export to `Documents/graphs/` (the whole directory:
   `manifest.json`, `adaln.safetensors` and every `.tflite`;
   `LiteRTGraphStore.missingFiles()` lists what is absent) and prompt
   embeddings to `Documents/gallery/<name>.safetensors` (`prompt_embeds`
   `[rows, 5120]` F32 plus `text_tags` `[rows]`, as `h3-litert encode` /
   `h3 encode` write them; without `text_tags` every row is text). "Rescan
   Documents" in the app.

The Metal delegate is on by default (`MetalDelegate`,
`isPrecisionLossAllowed = true`: the contract judges the f16 GPU path at
2e-2). Turn it off to run on the CPU (XNNPACK).

Audio plays through `AVAudioEngine` from an `AVAudioPCMBuffer` of the
pipeline's `[2, 800·A]` floats. The tiny fixture's audio VAE runs at 320 Hz,
below what the engine accepts, so rates under 8 kHz are zero-order-held up
(`AudioPlayer.swift`); the real model's 32 kHz plays as is. "Save WAV"
writes `Documents/h3-<seed>.wav`.

For the tiny fixture (`fixtures/tiny`): copy its directory to
`Documents/graphs` and its `expected.safetensors` to
`Documents/gallery/tiny.safetensors` (the reader picks `prompt_embeds` and
`text_tags` by name), size 96×64×22, seed 7. `FixtureCheck.run(store:directory:)`
compares the latent, the audio rows, the frames and the waveform against
`expected.safetensors` the way `h3-litert check` does.
