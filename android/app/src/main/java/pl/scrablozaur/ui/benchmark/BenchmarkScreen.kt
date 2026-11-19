package pl.scrablozaur.ui.benchmark

import android.app.Application
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringArrayResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.scrablozaur.R
import pl.scrablozaur.ScrablozaurApplication
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.engine.Benchmark
import pl.scrablozaur.engine.BenchmarkReport
import pl.scrablozaur.engine.Outcome
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.engine.SeatStats
import pl.scrablozaur.ui.components.BoardOverlay
import pl.scrablozaur.ui.components.BoardView
import pl.scrablozaur.ui.components.Mark
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.game.DefinitionCard
import pl.scrablozaur.ui.game.DefinitionUi
import pl.scrablozaur.ui.game.Grid
import pl.scrablozaur.ui.game.wordThrough
import pl.scrablozaur.ui.setup.BenchmarkRequest
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors

data class BenchmarkUi(
    val done: Int = 0,
    val total: Int = 0,
    val report: BenchmarkReport? = null,
    val language: LanguageMeta? = null,
    val points: Map<String, UInt> = emptyMap(),
    /** Replay position: moves applied so far, 0 being the empty board. */
    val step: Int = 0,
    val playing: Boolean = false,
    val definition: DefinitionUi? = null,
)

class BenchmarkViewModel(app: Application) : AndroidViewModel(app) {
    private val scrablozaur = app as ScrablozaurApplication
    private var bench: Benchmark? = null
    private var replay: Job? = null
    private val _ui = MutableStateFlow(BenchmarkUi())
    val ui: StateFlow<BenchmarkUi> = _ui

    /** Whether a run was ever started here -- after process death it was not. */
    val started get() = _ui.value.total > 0

    fun start(request: BenchmarkRequest) {
        cancel()
        _ui.value = BenchmarkUi(total = request.games, language = scrablozaur.engine.language(request.language))
        viewModelScope.launch {
            val pack = scrablozaur.engine.pack(request.language)
            val points = withContext(Dispatchers.Default) { pack.letterPoints() }
            val b = Benchmark.start(pack, request.players, request.games.toUInt(), 0uL)
            bench = b
            _ui.update { it.copy(points = points) }
            while (isActive) {
                val done = b.gamesDone().toInt()
                if (b.isFinished()) {
                    _ui.update { it.copy(done = done, report = b.result(), step = 0) }
                    break
                }
                _ui.update { it.copy(done = done) }
                delay(300)
            }
        }
    }

    fun cancel() {
        replay?.cancel()
        bench?.cancel()
        bench = null
    }

    fun stepTo(step: Int) {
        val moves = _ui.value.report?.bestGame?.moves?.size ?: return
        _ui.update { it.copy(step = step.coerceIn(0, moves), definition = null) }
    }

    fun togglePlay() {
        if (replay != null) {
            replay?.cancel()
            return
        }
        val moves = _ui.value.report?.bestGame?.moves?.size ?: return
        if (_ui.value.step >= moves) stepTo(0)
        _ui.update { it.copy(playing = true) }
        replay = viewModelScope.launch {
            try {
                while (_ui.value.step < moves) {
                    delay(scrablozaur.settings.current.playback.replayMs)
                    stepTo(_ui.value.step + 1)
                }
            } finally {
                replay = null
                _ui.update { it.copy(playing = false) }
            }
        }
    }

    fun lookupDefinition(word: String) {
        if (!scrablozaur.settings.current.onlineDefinitions) return
        val providers = _ui.value.language?.definitions ?: return
        _ui.update { it.copy(definition = DefinitionUi(word, null)) }
        viewModelScope.launch {
            val r = scrablozaur.definitions.lookup(word, providers)
            _ui.update { if (it.definition?.word == word) it.copy(definition = DefinitionUi(word, r)) else it }
        }
    }

    fun closeDefinition() = _ui.update { it.copy(definition = null) }

    override fun onCleared() {
        cancel()
        super.onCleared()
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkScreen(vm: BenchmarkViewModel, premiums: List<Premium>, onBack: () -> Unit, onRerun: () -> Unit) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.benchmark_results), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = { vm.cancel(); onBack() }) {
                        Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back))
                    }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            val report = ui.report
            if (report == null) {
                Text(stringResource(R.string.benchmark_running, ui.done, ui.total))
                LinearProgressIndicator(
                    progress = { if (ui.total == 0) 0f else ui.done.toFloat() / ui.total },
                    modifier = Modifier.fillMaxWidth(),
                )
                return@Column
            }
            report.error?.let { Text(it, color = Palette.danger) }
            Summary(report)
            report.seats.forEachIndexed { i, s -> SeatCard(i, s) }
            Replay(ui, vm, report, premiums)
            Spacer(Modifier.height(4.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(onClick = onRerun) { Text(stringResource(R.string.benchmark_rerun)) }
                OutlinedButton(colors = outlinedButtonColors(), onClick = onBack) { Text(stringResource(R.string.benchmark_back)) }
            }
        }
    }
}

@Composable
private fun Summary(report: BenchmarkReport) {
    val games = report.gamesPlayed.toInt()
    val lines = buildList {
        add(
            pluralStringResource(R.plurals.benchmark_games, games, games) + " · " +
                stringResource(R.string.benchmark_seconds, "%.1f".format(report.durationMs.toDouble() / 1000)),
        )
        add(stringResource(R.string.benchmark_avg_length, "%.1f".format(report.avgGameLength)))
        report.longestWord?.let { add(stringResource(R.string.benchmark_longest, it.uppercase(), (report.longestWordScore ?: 0u).toInt())) }
        report.highestSingleMoveScore?.let { add(stringResource(R.string.benchmark_highest, it.toInt())) }
    }
    Column {
        lines.forEachIndexed { i, line ->
            Text(line, fontWeight = if (i == 0) FontWeight.SemiBold else FontWeight.Normal, color = if (i == 0) Palette.text else Palette.muted)
        }
    }
}

@Composable
private fun SeatCard(index: Int, s: SeatStats) {
    val names = stringArrayResource(R.array.level_names)
    val level = s.difficulty.toInt().coerceIn(1, names.size)
    val winPct = if (s.gamesPlayed == 0u) 0 else Math.round(s.wins.toDouble() / s.gamesPlayed.toDouble() * 100).toInt()
    Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column(Modifier.fillMaxWidth().padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(10.dp).background(Palette.owners[index % 4].second, CircleShape))
                Spacer(Modifier.width(6.dp))
                Text(s.name, fontWeight = FontWeight.Bold)
            }
            Text(
                "${s.strength}% · " + stringResource(R.string.level_label, names[level - 1], level),
                color = Palette.muted,
                fontSize = 13.sp,
            )
            Spacer(Modifier.height(6.dp))
            val stats = listOf(
                R.string.stat_games to s.gamesPlayed.toString(),
                R.string.stat_wins to stringResource(R.string.stat_wins_value, s.wins.toInt(), winPct),
                R.string.stat_ties to s.ties.toString(),
                R.string.stat_avg_score to "%.1f".format(s.avgScore),
                R.string.stat_high to s.highScore.toString(),
                R.string.stat_low to s.lowScore.toString(),
                R.string.stat_avg_word to "%.1f".format(s.avgWordScore),
            )
            stats.chunked(2).forEach { pair ->
                Row(Modifier.fillMaxWidth()) {
                    pair.forEach { (label, value) ->
                        Row(Modifier.weight(1f).padding(vertical = 1.dp)) {
                            Text(stringResource(label), color = Palette.muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
                            Text(value, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.padding(end = 12.dp))
                        }
                    }
                    if (pair.size == 1) Spacer(Modifier.weight(1f))
                }
            }
        }
    }
}

/** The best game, move by move, as the web's replay steps through it. */
@Composable
private fun Replay(ui: BenchmarkUi, vm: BenchmarkViewModel, report: BenchmarkReport, premiums: List<Premium>) {
    val best = report.bestGame?.takeIf { it.moves.isNotEmpty() } ?: return
    val winner = when (val o = best.outcome) {
        is Outcome.Winner -> best.finalPlayers[o.player.toInt()].name
        else -> stringResource(R.string.tie)
    }
    Text(stringResource(R.string.benchmark_best_game, winner, best.winnerScore), fontWeight = FontWeight.SemiBold)
    val move = best.moves.getOrNull(ui.step - 1)
    val cells = move?.cells ?: List(225) { "" }
    val marks = if (move != null && !move.passed) {
        (0 until move.word.length).associate {
            val cell = if (move.horizontal) Cell(move.row.toInt(), move.col.toInt() + it) else Cell(move.row.toInt() + it, move.col.toInt())
            cell to Mark.LastMove
        }
    } else {
        emptyMap()
    }
    ui.definition?.let { DefinitionCard(it, vm::closeDefinition) }
    BoardView(
        cells = cells,
        blanks = move?.blanks ?: List(225) { false },
        owners = move?.owners ?: List(225) { (-1).toByte() },
        premiums = premiums,
        points = ui.points,
        overlay = BoardOverlay(marks = marks),
        modifier = Modifier.fillMaxWidth(),
        onTap = { cell ->
            val grid = Grid { r, c -> cells.getOrNull(r * 15 + c)?.firstOrNull() }
            wordThrough(grid, cell)?.let(vm::lookupDefinition)
        },
    )
    // Every seat's rack is fair to show here: there is no "your" rack to hide.
    best.finalPlayers.forEachIndexed { i, p ->
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(9.dp).background(Palette.owners[i % 4].second, CircleShape))
            Spacer(Modifier.width(6.dp))
            Text(p.name, Modifier.weight(1f), fontSize = 13.sp)
            Text(
                move?.lettersAfter?.getOrNull(i).orEmpty().map { if (it == '?') '★' else it.uppercaseChar() }.joinToString(" "),
                fontSize = 13.sp,
                color = Palette.muted,
            )
            Spacer(Modifier.width(12.dp))
            Text((move?.scoresAfter?.getOrNull(i) ?: 0).toString(), fontWeight = FontWeight.Bold, fontSize = 14.sp)
        }
    }
    Text(
        when {
            move == null -> stringResource(R.string.replay_start)
            move.passed -> stringResource(R.string.replay_passed, best.finalPlayers[move.player.toInt()].name)
            else -> stringResource(
                R.string.replay_played,
                best.finalPlayers[move.player.toInt()].name,
                move.word.uppercase(),
                move.score.toInt(),
            )
        },
        color = Palette.muted,
    )
    Row(verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = { vm.stepTo(ui.step - 1) }, enabled = ui.step > 0) { Icon(Icons.Filled.ChevronLeft, stringResource(R.string.replay_prev)) }
        OutlinedButton(colors = outlinedButtonColors(), onClick = vm::togglePlay) {
            Icon(if (ui.playing) Icons.Filled.Pause else Icons.Filled.PlayArrow, null)
            Spacer(Modifier.width(4.dp))
            Text(stringResource(R.string.replay_play))
        }
        IconButton(onClick = { vm.stepTo(ui.step + 1) }, enabled = ui.step < best.moves.size) { Icon(Icons.Filled.ChevronRight, stringResource(R.string.replay_next)) }
        Text(stringResource(R.string.replay_move, ui.step, best.moves.size), color = Palette.muted)
    }
}
