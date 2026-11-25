package dev.h3.litert.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.lifecycle.viewmodel.compose.viewModel
import dev.h3.litert.app.ui.GenerateScreen
import dev.h3.litert.app.ui.H3Theme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            H3Theme {
                val vm: GenerateViewModel = viewModel()
                GenerateScreen(vm)
            }
        }
    }
}
