package pl.scrablozaur.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.listSaver
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.saveable.rememberSaveableStateHolder
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalView
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import pl.scrablozaur.BuildConfig
import pl.scrablozaur.ScrablozaurApplication
import pl.scrablozaur.core.AppSettings
import pl.scrablozaur.core.GameRecord
import pl.scrablozaur.core.me
import pl.scrablozaur.core.playerKey
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.PlayerSetup
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.ui.benchmark.BenchmarkScreen
import pl.scrablozaur.ui.benchmark.BenchmarkViewModel
import pl.scrablozaur.ui.game.GameScreen
import pl.scrablozaur.ui.game.GameViewModel
import pl.scrablozaur.ui.game.NewGame
import pl.scrablozaur.ui.review.ReviewScreen
import pl.scrablozaur.ui.scan.ScanScreen
import pl.scrablozaur.ui.scan.ScanViewModel
import pl.scrablozaur.ui.settings.SettingsScreen
import pl.scrablozaur.ui.setup.SetupActions
import pl.scrablozaur.ui.setup.SetupScreen
import pl.scrablozaur.ui.stats.StatsScreen
import pl.scrablozaur.ui.stats.StatsTab
import pl.scrablozaur.ui.theme.ScrablozaurTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as ScrablozaurApplication
        // Debug builds only: `adb shell am start -n pl.scrablozaur/.ui.MainActivity
        // --el bench_seed 2823 --ei bench_level 10` opens the automatic game
        // `scrablozaur bot-bench` plays with that seed, so the app's engine speed
        // can be compared with the CLI's move for move on the same phone.
        val requested = intent.getLongExtra("bench_seed", 0L).takeIf { BuildConfig.DEBUG && savedInstanceState == null }
        val benchGame = requested?.takeIf { it != 0L }?.let { seed ->
            val level = intent.getIntExtra("bench_level", 10).toUByte()
            NewGame(
                language = app.engine.defaultLanguage,
                mode = Mode.SANDBOX_AUTO,
                players = listOf(PlayerSetup("A", true, level, 100u), PlayerSetup("B", true, level, 100u)),
                seed = seed.toULong(),
            )
        }
        // Debug builds only: `--es scan_replay <frames.bin>` opens the scanner on
        // recorded frames (`frame-bench`'s SCRBFRM1) instead of the camera.
        val replay = intent.getStringExtra("scan_replay").takeIf { BuildConfig.DEBUG && savedInstanceState == null }
        setContent { ScrablozaurTheme { App(app, benchGame, replay) } }
    }
}

private enum class Screen { Setup, Game, Benchmark, Scan }

/** The screens opened over the others -- statistics, a game's review, the settings -- which Back closes in turn. */
private enum class Page { Stats, Review, Settings }

/** An open page, and what it shows: the stats tab, the reviewed game's id. */
private data class Overlay(val page: Page, val arg: String = "")

private val OverlaysSaver = listSaver<List<Overlay>, String>(
    save = { list -> list.map { "${it.page.name}:${it.arg}" } },
    restore = { saved -> saved.map { Overlay(Page.valueOf(it.substringBefore(':')), it.substringAfter(':')) } },
)

@Composable
private fun App(app: ScrablozaurApplication, benchGame: NewGame?, replay: String?) {
    val game: GameViewModel = viewModel()
    val bench: BenchmarkViewModel = viewModel()
    val premiums = app.engine.premiums
    val settings by app.settings.state.collectAsStateWithLifecycle()
    val records by app.history.records.collectAsStateWithLifecycle()
    // Saveable, so a rotation or a process restart comes back to the same
    // screen. A game in progress is resumed straight into on launch.
    var screen by rememberSaveable {
        mutableStateOf(if (replay != null) Screen.Scan else if (app.saves.hasUnfinishedGame()) Screen.Game else Screen.Setup)
    }
    var previous by rememberSaveable { mutableStateOf(Screen.Setup) }
    var overlays by rememberSaveable(stateSaver = OverlaysSaver) { mutableStateOf(emptyList<Overlay>()) }
    // A page covered by another keeps its tab, filters and scroll for when it shows again.
    val pageStates = rememberSaveableStateHolder()
    fun key(depth: Int, o: Overlay) = "$depth:${o.page}:${o.arg}"

    fun go(to: Screen) {
        previous = screen
        screen = to
    }

    fun open(page: Page, arg: String = "") {
        overlays = overlays + Overlay(page, arg)
    }

    fun close() {
        overlays.lastOrNull()?.let { pageStates.removeState(key(overlays.lastIndex, it)) }
        overlays = overlays.dropLast(1)
    }

    // While a game, a scan or a benchmark is on screen, if the player asked.
    val view = LocalView.current
    val keepOn = settings.keepScreenOn && overlays.isEmpty() && screen != Screen.Setup
    DisposableEffect(view, keepOn) {
        view.keepScreenOn = keepOn
        onDispose { view.keepScreenOn = false }
    }

    LaunchedEffect(benchGame) {
        if (benchGame != null) {
            game.forgetGame()
            game.start(benchGame)
            go(Screen.Game)
        }
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

    val top = overlays.lastOrNull()
    if (top != null) {
        BackHandler { close() }
        pageStates.SaveableStateProvider(key(overlays.lastIndex, top)) {
            Page(app, top, records, settings, premiums, onOpenGame = { open(Page.Review, it) }, onClose = ::close)
        }
        return
    }

    when (screen) {
        Screen.Setup -> {
            BackHandler(enabled = game.hasGame) { go(Screen.Game) }
            // Everyone who played here, the latest first.
            val knownNames = remember(records) {
                records.orEmpty().sortedByDescending { it.finishedAt }.map { it.me.name }.distinctBy(::playerKey)
            }
            SetupScreen(
                languages = app.engine.languages,
                defaultLanguage = app.engine.defaultLanguage,
                choices = settings.setup,
                playerName = settings.playerName,
                knownNames = knownNames,
                canContinue = app.saves.hasUnfinishedGame(),
                actions = SetupActions(
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
                    onRemember = { choices, name -> app.settings.update { it.copy(setup = choices, playerName = name) } },
                    onStats = { open(Page.Stats, StatsTab.Overview.name) },
                    onRanking = { open(Page.Stats, StatsTab.Ranking.name) },
                    onSettings = { open(Page.Settings) },
                ),
            )
        }
        Screen.Game -> {
            BackHandler { go(Screen.Setup) }
            GameScreen(
                game,
                premiums,
                onNewGame = { go(Screen.Setup) },
                onScan = { go(Screen.Scan) },
                onReview = { open(Page.Review, it) },
            )
        }
        Screen.Benchmark -> {
            BackHandler { bench.cancel(); go(Screen.Setup) }
            BenchmarkScreen(bench, premiums, onBack = { go(Screen.Setup) }, onRerun = { go(Screen.Setup) })
        }
        Screen.Scan -> {
            // Made on first use, then kept with the activity: the confirmed board
            // and the loaded scanner outlive a trip to the game and back.
            val scan: ScanViewModel = viewModel()
            LaunchedEffect(replay) { if (replay != null) scan.replay(replay) }
            BackHandler { go(previous) }
            ScanScreen(scan, premiums, onBack = { go(previous) })
        }
    }
}

/** One of the pages opened over the main screens. */
@Composable
private fun Page(
    app: ScrablozaurApplication,
    page: Overlay,
    records: List<GameRecord>?,
    settings: AppSettings,
    premiums: List<Premium>,
    onOpenGame: (String) -> Unit,
    onClose: () -> Unit,
) {
    when (page.page) {
        Page.Stats -> StatsScreen(
            records = records,
            languages = app.engine.languages,
            playerName = settings.playerName,
            initialTab = StatsTab.entries.firstOrNull { it.name == page.arg } ?: StatsTab.Overview,
            onOpenGame = onOpenGame,
            onBack = onClose,
        )
        Page.Review -> {
            val record = records?.firstOrNull { it.id == page.arg }
            ReviewScreen(
                record = record,
                loaded = records != null,
                language = record?.let { app.engine.language(it.language) },
                premiums = premiums,
                replayMs = settings.playback.replayMs,
                definitions = app.definitions.takeIf { settings.onlineDefinitions },
                onDelete = {
                    app.history.delete(page.arg)
                    onClose()
                },
                onBack = onClose,
            )
        }
        Page.Settings -> SettingsScreen(
            settings = settings,
            historySize = records?.size,
            version = BuildConfig.VERSION_NAME,
            onChange = app.settings::update,
            onClearHistory = app.history::clear,
            onBack = onClose,
        )
    }
}
