package pl.scrablozaur.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import pl.scrablozaur.R
import pl.scrablozaur.ScrablozaurApplication
import pl.scrablozaur.ui.benchmark.BenchmarkScreen
import pl.scrablozaur.ui.benchmark.BenchmarkViewModel
import pl.scrablozaur.ui.game.GameScreen
import pl.scrablozaur.ui.game.GameViewModel
import pl.scrablozaur.ui.setup.SetupScreen
import pl.scrablozaur.ui.theme.ScrablozaurTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as ScrablozaurApplication
        setContent { ScrablozaurTheme { App(app) } }
    }
}

private enum class Screen { Setup, Game, Benchmark, Scan }

@Composable
private fun App(app: ScrablozaurApplication) {
    val game: GameViewModel = viewModel()
    val bench: BenchmarkViewModel = viewModel()
    val premiums = app.engine.premiums
    // Saveable, so a rotation or a process restart comes back to the same
    // screen. A game in progress is resumed straight into on launch.
    var screen by rememberSaveable { mutableStateOf(if (app.saves.hasUnfinishedGame()) Screen.Game else Screen.Setup) }
    var previous by rememberSaveable { mutableStateOf(Screen.Setup) }

    fun go(to: Screen) {
        previous = screen
        screen = to
    }

    LaunchedEffect(screen) {
        when (screen) {
            // A no-op when the game is already loaded; after process death it
            // reloads the save.
            Screen.Game -> game.resumeIfNeeded()
            // A run does not survive the process; its screen would wait forever.
            Screen.Benchmark -> if (!bench.started) screen = Screen.Setup
            else -> {}
        }
    }

    when (screen) {
        Screen.Setup -> {
            BackHandler(enabled = game.hasGame) { go(Screen.Game) }
            SetupScreen(
                languages = app.engine.languages,
                defaultLanguage = app.engine.defaultLanguage,
                canContinue = app.saves.hasUnfinishedGame(),
                onContinue = { go(Screen.Game) },
                onStart = { config ->
                    game.forgetGame()
                    game.start(config)
                    go(Screen.Game)
                },
                onBenchmark = { request ->
                    bench.start(request)
                    go(Screen.Benchmark)
                },
                onScan = { go(Screen.Scan) },
            )
        }
        Screen.Game -> {
            BackHandler { go(Screen.Setup) }
            GameScreen(game, premiums, onNewGame = { go(Screen.Setup) }, onScan = { go(Screen.Scan) })
        }
        Screen.Benchmark -> {
            BackHandler { bench.cancel(); go(Screen.Setup) }
            BenchmarkScreen(bench, premiums, onBack = { go(Screen.Setup) }, onRerun = { go(Screen.Setup) })
        }
        Screen.Scan -> {
            BackHandler { go(previous) }
            ScannerPlaceholder { go(previous) }
        }
    }
}

/** The live scanner lands in a later phase; until then the tab says so. */
@Composable
private fun ScannerPlaceholder(onBack: () -> Unit) {
    Column(
        Modifier.fillMaxSize().systemBarsPadding().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text(stringResource(R.string.scan_coming_soon), textAlign = TextAlign.Center)
        Button(onClick = onBack) { Text(stringResource(R.string.back)) }
    }
}
