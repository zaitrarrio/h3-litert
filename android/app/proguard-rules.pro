# LiteRT's Kotlin API reaches its natives through JNI; keep it whole if minification is enabled.
-keep class com.google.ai.edge.litert.** { *; }
# kotlinx.serialization: keep the generated serializers of :core's manifest classes.
-keep,includedescriptorclasses class dev.h3.litert.core.**$$serializer { *; }
-keepclassmembers class dev.h3.litert.core.** { *** Companion; }
-keepclasseswithmembers class dev.h3.litert.core.** { kotlinx.serialization.KSerializer serializer(...); }
