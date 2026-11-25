// :core — the host side of h3-litert in pure Kotlin/JVM. No Android imports; the Android app
// (and any JVM host) plugs a GraphStore in.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
    `java-library`
}

group = "dev.h3.litert"
version = "0.1.0"

// JVM 17 bytecode (Android's desugaring target and the JDK a CI box has); builds with any JDK >= 17.
java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
    explicitApi()
}

dependencies {
    api(libs.kotlinx.serialization.json)
    testImplementation(kotlin("test"))
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform()
    maxHeapSize = "1g"
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = false
    }
}
