// swift-tools-version: 5.9
// H3LiteRT: MiniMax H3 (video + audio) on LiteRT, the iOS runtime.
//
// - H3LiteRTCore      pure Swift (Foundation only): the host math of docs/graphs.md (packing, RoPE,
//                     schedules, samplers, the VAE host side), the manifest, the AdaLN table and the
//                     pipeline over `Graph`.
// - H3LiteRTRuntime   `Graph` / `GraphStore` over the TensorFlow Lite Swift API (LiteRT for iOS).
//                     Its sources are behind `#if canImport(TensorFlowLite)`: link the
//                     `TensorFlowLiteSwift` CocoaPod (with the `Metal` subspec) into the app, see
//                     README.md. Without it the target compiles to a stub, so `swift test` of the
//                     core runs on any Mac.
// - H3LiteRTCoreTests XCTest: every pinned value of python/tests/test_host.py and the pipeline
//                     orchestration over a fake store on the tiny manifest.

import PackageDescription

let package = Package(
    name: "H3LiteRT",
    platforms: [
        .iOS(.v16),
        .macOS(.v13),
    ],
    products: [
        .library(name: "H3LiteRTCore", targets: ["H3LiteRTCore"]),
        .library(name: "H3LiteRTRuntime", targets: ["H3LiteRTRuntime"]),
    ],
    dependencies: [
        // TODO verify: TensorFlow Lite Swift is published as the CocoaPod `TensorFlowLiteSwift`
        // (2.17.0, subspec `Metal` for the GPU delegate), not as a Swift package, and the newer
        // `LiteRTSwift` pod (nightly) is the successor. If a Swift package of either appears,
        // add it here and to H3LiteRTRuntime's dependencies, e.g.
        //   .package(url: "https://github.com/<mirror>/TensorFlowLiteSwift.git", from: "2.17.0"),
    ],
    targets: [
        .target(
            name: "H3LiteRTCore",
            dependencies: [],
            path: "Sources/H3LiteRTCore"
        ),
        .target(
            name: "H3LiteRTRuntime",
            dependencies: ["H3LiteRTCore"],
            path: "Sources/H3LiteRTRuntime"
        ),
        .testTarget(
            name: "H3LiteRTCoreTests",
            dependencies: ["H3LiteRTCore"],
            path: "Tests/H3LiteRTCoreTests",
            resources: [
                .copy("Resources/manifest.json"),
                .copy("Resources/adaln.safetensors"),
            ]
        ),
    ]
)
