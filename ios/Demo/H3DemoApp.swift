// The demo app's entry point. Create an iOS App project in Xcode (SwiftUI,
// Swift), delete its generated App/ContentView files, add the files of this
// folder, add the `H3LiteRT` package (Core + Runtime products) and the
// `TensorFlowLiteSwift` pod; see Demo/README.md.

import SwiftUI

@main
struct H3DemoApp: App {
    var body: some Scene {
        WindowGroup {
            ContentView()
        }
    }
}
