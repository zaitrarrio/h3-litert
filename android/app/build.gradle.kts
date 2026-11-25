// :app — the Android host: LiteRT CompiledModel (GPU) behind :core's GraphStore, a ViewModel that
// runs Pipeline.generate off the main thread, and a Compose screen. Needs the Android SDK
// (compileSdk 36); it is only included by settings.gradle.kts when one is found.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

kotlin {
    compilerOptions { jvmTarget.set(JvmTarget.JVM_17) }
}

android {
    namespace = "dev.h3.litert.app"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.h3.litert"
        minSdk = 26
        targetSdk = 36
        versionCode = 1
        versionName = "0.1.0"
        // LiteRT ships arm64-v8a, armeabi-v7a and x86_64 natives; the phone tier is arm64.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures { compose = true }

    packaging {
        // .tflite graphs are never bundled (they are adb-pushed / downloaded); keep the natives uncompressed.
        jniLibs { useLegacyPackaging = false }
    }
}

dependencies {
    implementation(project(":core"))

    // LiteRT Next: CompiledModel / TensorBuffer / Accelerator. The `litert` AAR depends on
    // `litert-api` (the Kotlin API) and bundles libLiteRt.so + libLiteRtClGlAccelerator.so (GPU).
    implementation(libs.litert)

    implementation(libs.kotlinx.coroutines.android)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.lifecycle.viewmodel.compose)
    implementation(libs.androidx.lifecycle.runtime.compose)

    implementation(platform(libs.compose.bom))
    implementation(libs.compose.ui)
    implementation(libs.compose.ui.tooling.preview)
    implementation(libs.compose.foundation)
    implementation(libs.compose.material3)
}
