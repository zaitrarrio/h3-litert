// h3-litert Android runtime.
//
//   :core  pure Kotlin/JVM: host math, manifest, pipeline over a Graph interface. Builds and tests anywhere.
//   :app   Android application (LiteRT CompiledModel on the GPU, Compose UI). Needs the Android SDK.
//
// `:app` is only included when an Android SDK can be found (ANDROID_HOME / ANDROID_SDK_ROOT, or
// `sdk.dir` in local.properties), so `gradle :core:test` works on a machine without the SDK.
pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "h3-litert-android"

include(":core")

val localProps = java.util.Properties().apply {
    val f = file("local.properties")
    if (f.isFile) f.inputStream().use { load(it) }
}
val sdkDir: String? = localProps.getProperty("sdk.dir")
    ?: System.getenv("ANDROID_HOME")
    ?: System.getenv("ANDROID_SDK_ROOT")
val forceApp = providers.gradleProperty("h3.app").orNull?.toBoolean()

if (forceApp == true || (forceApp == null && sdkDir != null && file(sdkDir).isDirectory)) {
    include(":app")
} else {
    logger.lifecycle("h3-litert: no Android SDK found (ANDROID_HOME / sdk.dir); building :core only. Pass -Ph3.app=true to force :app.")
}
