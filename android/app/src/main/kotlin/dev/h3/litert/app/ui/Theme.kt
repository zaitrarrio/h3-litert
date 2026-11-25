package dev.h3.litert.app.ui

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Scheme = darkColorScheme(
    primary = Color(0xFF9CD3FF),
    onPrimary = Color(0xFF00334D),
    secondary = Color(0xFFB8C8D6),
    background = Color(0xFF101214),
    onBackground = Color(0xFFE4E7EA),
    surface = Color(0xFF17191C),
    onSurface = Color(0xFFE4E7EA),
    surfaceVariant = Color(0xFF22262A),
    onSurfaceVariant = Color(0xFFB8C0C8),
    error = Color(0xFFFF9A8A),
)

@Composable
fun H3Theme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = Scheme, content = content)
}
