// Root build. Only the Kotlin plugins are declared here (apply false) so that a `:core`-only build
// never resolves the Android Gradle Plugin; `:app` declares AGP and the Compose compiler plugin
// itself, with versions from gradle/libs.versions.toml.
plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}
