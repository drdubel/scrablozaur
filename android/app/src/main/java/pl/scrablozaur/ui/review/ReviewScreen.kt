package pl.scrablozaur.ui.review

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import pl.scrablozaur.R
import pl.scrablozaur.core.Definitions
import pl.scrablozaur.core.GameRecord
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.Position
import pl.scrablozaur.core.RecordedPlay
import pl.scrablozaur.core.RecordedTurn
import pl.scrablozaur.core.Result
import pl.scrablozaur.core.Seat
import pl.scrablozaur.core.accuracy
import pl.scrablozaur.core.bestPlay
import pl.scrablozaur.core.bingos
import pl.scrablozaur.core.endAdjustments
import pl.scrablozaur.core.endgameLost
import pl.scrablozaur.core.equityLost
import pl.scrablozaur.core.missedPoints
import pl.scrablozaur.core.opponent
import pl.scrablozaur.core.opponentKey
import pl.scrablozaur.core.plays
import pl.scrablozaur.core.positions
import pl.scrablozaur.core.resigned
import pl.scrablozaur.core.result
import pl.scrablozaur.core.scoreProgress
import pl.scrablozaur.core.seats
import pl.scrablozaur.core.wasBest
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.engine.TurnKind
import pl.scrablozaur.ui.components.BarChart
import pl.scrablozaur.ui.components.BoardOverlay
import pl.scrablozaur.ui.components.BoardView
import pl.scrablozaur.ui.components.DataTable
import pl.scrablozaur.ui.components.EngineBusy
import pl.scrablozaur.ui.components.LineChart
import pl.scrablozaur.ui.components.Mark
import pl.scrablozaur.ui.components.SectionTitle
import pl.scrablozaur.ui.components.Series
import pl.scrablozaur.ui.components.StatTile
import pl.scrablozaur.ui.components.TableCell
import pl.scrablozaur.ui.components.TableColumn
import pl.scrablozaur.ui.components.TileGrid
import pl.scrablozaur.ui.components.formatAmount
import pl.scrablozaur.ui.components.formatValue
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.game.DefinitionCard
import pl.scrablozaur.ui.game.DefinitionUi
import pl.scrablozaur.ui.game.Grid
import pl.scrablozaur.ui.game.ratingColor
import pl.scrablozaur.ui.game.wordThrough
import pl.scrablozaur.ui.stats.formatDateTime
import pl.scrablozaur.ui.stats.opponentLabel
import pl.scrablozaur.ui.stats.percent
import pl.scrablozaur.ui.stats.resultLabel
import pl.scrablozaur.ui.theme.Palette
import kotlin.math.roundToInt

/** Which better play the board shows over the position: none, or the one a verdict names. */
private enum class Shown { Turn, TopScore, Equity, Endgame }

/**
 * "Ocena gry": a finished game gone through again. How well a player played
 * -- each move's rating against the highest-scoring play, its equity against
 * what Strateg would have played, the endgame against a full search -- the
 * score as it went, and the board at any turn. A game several people played
 * is reviewed from each one's seat in turn.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReviewScreen(
    record: GameRecord?,
    loaded: Boolean,
    language: LanguageMeta?,
    premiums: List<Premium>,
    replayMs: Long,
    /** Where a tapped word is looked up; null when definitions are switched off. */
    definitions: Definitions?,
    onDelete: () -> Unit,
    onBack: () -> Unit,
) {
    var confirmDelete by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.review_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
                actions = {
                    if (record != null) {
                        IconButton(onClick = { confirmDelete = true }) { Icon(Icons.Filled.Delete, stringResource(R.string.review_delete)) }
                    }
                },
            )
        },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when {
                record != null -> Review(record, language, premiums, replayMs, definitions)
                !loaded -> EngineBusy(Modifier.align(Alignment.Center).size(32.dp))
                else -> Text(stringResource(R.string.review_not_found), Modifier.align(Alignment.Center).padding(24.dp), color = Palette.muted)
            }
        }
    }
    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            text = { Text(stringResource(R.string.review_delete_confirm)) },
            confirmButton = {
                TextButton(onClick = { confirmDelete = false; onDelete() }) { Text(stringResource(R.string.delete), color = Palette.danger) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text(stringResource(R.string.cancel)) } },
        )
    }
}

@Composable
private fun Review(record: GameRecord, language: LanguageMeta?, premiums: List<Premium>, replayMs: Long, definitions: Definitions?) {
    val positions = remember(record) { record.positions() }
    val progress = remember(record) { record.scoreProgress() }
    val turns = record.turns
    // Whose game is reviewed: the phone's player first, any person in it on request.
    var seatIndex by rememberSaveable(record.id) { mutableIntStateOf(record.human) }
    val seat = Seat(record, seatIndex)
    // Turns applied to the board shown: 0 is the empty board, the last the final one.
    var step by rememberSaveable(record.id) { mutableIntStateOf(turns.size) }
    var shownName by rememberSaveable(record.id) { mutableStateOf(Shown.Turn.name) }
    val shown = Shown.valueOf(shownName)
    var playing by remember { mutableStateOf(false) }
    var definition by remember { mutableStateOf<DefinitionUi?>(null) }
    val scope = rememberCoroutineScope()

    fun select(s: Int) {
        step = s.coerceIn(0, turns.size)
        shownName = Shown.Turn.name
        definition = null
    }

    LaunchedEffect(playing) {
        if (!playing) return@LaunchedEffect
        if (step >= turns.size) select(0)
        while (step < turns.size) {
            delay(replayMs)
            select(step + 1)
        }
        playing = false
    }

    Column(
        Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        val people = record.seats
        if (people.size > 1) {
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                people.forEach { p ->
                    FilterChip(selected = p.index == seatIndex, onClick = { seatIndex = p.index }, label = { Text(p.name) })
                }
            }
        }
        Header(seat, language)
        Metrics(seat)

        // A log missing the first turns would start every score from zero.
        if (positions != null) {
            SectionTitle(stringResource(R.string.chart_score_progress))
            LineChart(
                series = progress.mapIndexed { i, scores ->
                    val color = when {
                        record.players.size > 2 -> Palette.owners[i % 4].second
                        i == seatIndex -> Palette.chartYou
                        else -> Palette.chartOpponent
                    }
                    Series(record.players[i].name, color, scores.map { it.toFloat() })
                },
                xLabels = (0..turns.size).map { it.toString() },
                selected = step,
                onSelect = ::select,
            )
        }

        val rated = turns.withIndex().filter { (_, t) -> t.player == seatIndex && t.rating != null }
        if (rated.isNotEmpty()) {
            SectionTitle(stringResource(R.string.chart_move_ratings))
            BarChart(
                values = rated.map { it.value.rating!!.toFloat() },
                colors = rated.map { ratingColor(it.value.rating!!) },
                xLabels = rated.map { (it.index + 1).toString() },
                selected = rated.indexOfFirst { it.index == step - 1 }.takeIf { it >= 0 },
                onSelect = { if (positions != null) select(rated[it].index + 1) },
            )
        }

        SectionTitle(stringResource(R.string.board_description))
        if (positions == null) {
            Text(stringResource(R.string.review_partial), color = Palette.muted, fontSize = 13.sp)
        }
        definition?.let { DefinitionCard(it, onClose = { definition = null }) }
        val turn = turns.getOrNull(step - 1)
        val better = turn?.let(::betterPlays) ?: emptyMap()
        val overBefore = better[shown]?.takeIf { positions != null }
        val position = when {
            positions == null -> Position(record.cells, record.owners)
            overBefore != null -> positions[step - 1]
            else -> positions[step]
        }
        Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
            BoardView(
                cells = position.cells,
                blanks = position.cells.indices.map { record.blanks[it] && position.cells[it].isNotEmpty() },
                owners = position.owners.map { it.toByte() },
                premiums = premiums,
                points = language?.points.orEmpty(),
                overlay = when {
                    positions == null -> BoardOverlay()
                    overBefore != null -> playOverlay(overBefore, position)
                    turn?.kind == TurnKind.PLAY -> BoardOverlay(marks = cellsOf(turn).associateWith { Mark.LastMove })
                    else -> BoardOverlay()
                },
                modifier = Modifier.widthIn(max = 480.dp).fillMaxWidth(),
                onTap = { cell ->
                    val source = definitions ?: return@BoardView
                    val providers = language?.definitions ?: return@BoardView
                    val word = wordThrough(Grid { r, c -> position.cells[r * 15 + c].firstOrNull() }, cell) ?: return@BoardView
                    definition = DefinitionUi(word, null)
                    scope.launch {
                        val found = source.lookup(word, providers)
                        if (definition?.word == word) definition = DefinitionUi(word, found)
                    }
                },
            )
        }
        if (positions != null) {
            TurnCaption(record, turn, better, shown) { which -> shownName = (if (shown == which) Shown.Turn else which).name }
            Row(verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { playing = false; select(step - 1) }, enabled = step > 0) {
                    Icon(Icons.Filled.ChevronLeft, stringResource(R.string.replay_prev))
                }
                IconButton(onClick = { playing = !playing }) {
                    Icon(if (playing) Icons.Filled.Pause else Icons.Filled.PlayArrow, stringResource(R.string.replay_play))
                }
                IconButton(onClick = { playing = false; select(step + 1) }, enabled = step < turns.size) {
                    Icon(Icons.Filled.ChevronRight, stringResource(R.string.replay_next))
                }
                Text(stringResource(R.string.replay_move, step, turns.size), color = Palette.muted)
            }
        }

        SectionTitle(stringResource(R.string.review_moves))
        MoveTable(record, selected = (step - 1).takeIf { positions != null && it >= 0 }) { i ->
            if (positions != null) {
                playing = false
                select(i + 1)
            }
        }
        if (positions != null) EndOfGame(record)
    }
}

/** The plays each verdict on a turn names as better than the one made. */
private fun betterPlays(t: RecordedTurn): Map<Shown, RecordedPlay> = buildMap {
    t.best?.takeIf { !t.wasBest && t.rating != null }?.let { put(Shown.TopScore, it) }
    t.equity?.play?.let { put(Shown.Equity, it) }
    t.endgame?.takeIf { it.loss > 0 }?.play?.let { put(Shown.Endgame, it) }
}

@Composable
private fun Header(seat: Seat, language: LanguageMeta?) {
    val record = seat.game
    val result = seat.result
    val tint = when (result) {
        Result.Win -> Palette.highlightNew
        Result.Loss -> Palette.invalid
        Result.Tie -> Palette.typingNew
    }
    Card(colors = CardDefaults.cardColors(containerColor = tint)) {
        Column(Modifier.fillMaxWidth().padding(14.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                resultLabel(result) + if (seat.resigned) " · " + stringResource(R.string.review_resigned) else "",
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(4.dp))
            if (record.players.size == 2) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(seat.name, Modifier.weight(1f), fontWeight = FontWeight.SemiBold, textAlign = TextAlign.End, maxLines = 1)
                    Text(
                        stringResource(R.string.score_vs, seat.score, seat.opponent.score),
                        Modifier.padding(horizontal = 12.dp),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                    )
                    Text(seat.opponent.name, Modifier.weight(1f), fontWeight = FontWeight.SemiBold, maxLines = 1)
                }
                if (seat.opponent.isComputer) Text(opponentLabel(seat.opponentKey), color = Palette.muted, fontSize = 13.sp)
            } else {
                // Three or four at the table: everyone, highest first.
                record.players.withIndex().sortedByDescending { it.value.score }.forEachIndexed { place, (i, p) ->
                    Row(Modifier.fillMaxWidth(0.8f)) {
                        Text(
                            "${place + 1}. ${p.name}",
                            Modifier.weight(1f),
                            fontWeight = if (i == seat.index) FontWeight.Bold else FontWeight.Normal,
                            maxLines = 1,
                        )
                        Text(stringResource(R.string.points_short, p.score), fontWeight = if (i == seat.index) FontWeight.Bold else FontWeight.Normal)
                    }
                }
            }
            val minutes = ((record.finishedAt - record.startedAt) / 60_000).toInt()
            val parts = listOfNotNull(
                formatDateTime(record.finishedAt),
                language?.let { "${it.flag} ${it.name}".trim() },
                stringResource(R.string.duration_minutes, minutes).takeIf { minutes > 0 },
            )
            Text(parts.joinToString(" · "), color = Palette.muted, fontSize = 12.sp)
        }
    }
}

@Composable
private fun Metrics(seat: Seat) {
    val record = seat.game
    val rated = seat.plays.filter { it.rating != null }
    val accuracy = seat.accuracy
    val best = seat.bestPlay
    val equity = seat.equityLost
    val valued = record.turns.count { it.player == seat.index && it.equity != null }
    val endgame = seat.endgameLost
    val tiles = buildList<@Composable (Modifier) -> Unit> {
        add { m ->
            StatTile(
                stringResource(R.string.metric_accuracy),
                accuracy?.let { percent(it) } ?: "—",
                m,
                color = accuracy?.let { ratingColor(it.roundToInt()) } ?: Palette.text,
            )
        }
        add { m ->
            StatTile(
                stringResource(R.string.metric_found_best),
                stringResource(R.string.found_best_value, rated.count { it.wasBest }, rated.size),
                m,
            )
        }
        add { m -> StatTile(stringResource(R.string.metric_missed), seat.missedPoints.toString(), m) }
        if (equity != null) {
            add { m ->
                StatTile(stringResource(R.string.metric_equity_lost), formatAmount(equity), m, detail = pluralStringResource(R.plurals.metric_equity_detail, valued, valued))
            }
        }
        if (endgame != null) {
            add { m ->
                StatTile(
                    stringResource(R.string.metric_endgame),
                    if (endgame == 0) "✓" else "−$endgame",
                    m,
                    color = if (endgame == 0) Palette.validText else Palette.invalidText,
                    detail = stringResource(R.string.metric_endgame_detail),
                )
            }
        }
        add { m ->
            StatTile(
                stringResource(R.string.metric_best_move),
                best?.word?.uppercase() ?: "—",
                m,
                detail = best?.let { stringResource(R.string.points_short, it.score) },
            )
        }
        add { m -> StatTile(stringResource(R.string.metric_bingos), seat.bingos.toString(), m) }
        add { m ->
            StatTile(
                stringResource(R.string.metric_hints),
                record.hintTurns.toString(),
                m,
                detail = stringResource(R.string.metric_undos_detail, record.undos),
            )
        }
    }
    TileGrid(tiles)
}

/** What the turn on the board was, and what each verdict on it says: rating, equity, endgame. */
@Composable
private fun TurnCaption(record: GameRecord, turn: RecordedTurn?, better: Map<Shown, RecordedPlay>, shown: Shown, onShow: (Shown) -> Unit) {
    if (turn == null) {
        Text(stringResource(R.string.replay_start), color = Palette.muted)
        return
    }
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(
            stringResource(R.string.turn_caption, record.players[turn.player].name, turnText(turn)) +
                if (turn.kind == TurnKind.PLAY) " · " + stringResource(R.string.points_short, turn.score) else "",
            fontWeight = FontWeight.SemiBold,
        )
        if (turn.rack.isNotEmpty()) {
            Text(stringResource(R.string.review_rack, turn.rack.uppercase().replace("?", "★")), color = Palette.muted, fontSize = 13.sp)
        }
        // A play two verdicts agree on gets one chip, under the first.
        val offered = mutableSetOf<RecordedPlay>()
        @Composable
        fun Chip(which: Shown, label: Int) {
            val play = better[which] ?: return
            if (!offered.add(play)) return
            FilterChip(
                selected = shown == which,
                onClick = { onShow(which) },
                label = { Text(stringResource(label, play.word.uppercase(), play.score)) },
            )
        }
        val rating = turn.rating
        if (rating != null) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Text(stringResource(R.string.rating_value, rating), color = ratingColor(rating), fontWeight = FontWeight.Bold)
                if (Shown.TopScore in better) Chip(Shown.TopScore, R.string.review_show_best)
                else Text(stringResource(R.string.review_was_best), color = Palette.validText, fontSize = 13.sp)
            }
        }
        turn.equity?.let { e ->
            Text(
                if (e.loss > 0) stringResource(R.string.review_equity, formatValue(e.played), formatValue(e.best))
                else stringResource(R.string.review_equity_best, formatValue(e.played)),
                color = if (e.loss > 0) Palette.text else Palette.validText,
                fontSize = 13.sp,
            )
            Chip(Shown.Equity, R.string.review_show_equity_best)
            e.exchange?.let { Text(stringResource(R.string.review_better_exchange, it.uppercase().replace("?", "★")), fontSize = 13.sp) }
        }
        turn.endgame?.let { e ->
            val exact = if (e.exact) "" else " " + stringResource(R.string.review_endgame_estimate)
            Text(
                if (e.loss > 0) stringResource(R.string.review_endgame, signedInt(e.played), signedInt(e.best)) + exact
                else stringResource(R.string.review_endgame_best, signedInt(e.played)) + exact,
                color = if (e.loss > 0) Palette.text else Palette.validText,
                fontSize = 13.sp,
            )
            if (e.loss > 0 && e.play == null) Text(stringResource(R.string.review_endgame_pass), fontSize = 13.sp)
            Chip(Shown.Endgame, R.string.review_show_endgame_best)
        }
    }
}

private fun signedInt(n: Int) = when {
    n > 0 -> "+$n"
    n < 0 -> "−${-n}"
    else -> "0"
}

@Composable
private fun turnText(t: RecordedTurn): String = when (t.kind) {
    TurnKind.PLAY -> t.word.uppercase()
    TurnKind.EXCHANGE -> stringResource(R.string.turn_exchange, t.tiles)
    TurnKind.PASS -> stringResource(R.string.turn_pass)
    TurnKind.RESIGN -> stringResource(R.string.turn_resign)
}

@Composable
private fun MoveTable(record: GameRecord, selected: Int?, onSelect: (Int) -> Unit) {
    val columns = listOf(
        TableColumn(stringResource(R.string.col_number), 0.5f),
        TableColumn(stringResource(R.string.col_turn), 2.0f),
        TableColumn(stringResource(R.string.col_points), 0.7f, end = true),
        TableColumn(stringResource(R.string.col_rating), 0.8f, end = true),
        TableColumn(stringResource(R.string.col_loss), 0.8f, end = true),
        TableColumn(stringResource(R.string.col_best), 2.0f, end = true),
    )
    val rows = record.turns.mapIndexed { i, t ->
        val play = t.kind == TurnKind.PLAY
        // The verdict that weighs the most: equity or the endgame over points alone.
        val better = t.equity?.let { e -> e.play?.let { "${it.word.uppercase()} ${it.score}" } ?: e.exchange?.let { "⇄ ${it.uppercase()}" } }
            ?: t.endgame?.takeIf { it.loss > 0 }?.let { e -> e.play?.let { "${it.word.uppercase()} ${it.score}" } ?: stringResource(R.string.turn_pass) }
            ?: t.best?.takeIf { t.rating != null && !t.wasBest }?.let { "${it.word.uppercase()} ${it.score}" }
        val loss = t.equity?.loss?.takeIf { it > 0 }?.let { formatAmount(it) } ?: t.endgame?.loss?.takeIf { it > 0 }?.toString()
        val reviewed = t.rating != null || t.equity != null || t.endgame != null
        listOf(
            TableCell("${i + 1}", Palette.muted),
            TableCell(turnText(t), if (play) Palette.text else Palette.muted, bold = play, dot = Palette.owners[t.player % 4].second),
            TableCell(if (play) t.score.toString() else ""),
            t.rating?.let { TableCell(it.toString(), ratingColor(it), bold = true) } ?: TableCell(""),
            loss?.let { TableCell(it, Palette.invalidText) } ?: TableCell(""),
            when {
                better != null -> TableCell(better, Palette.muted)
                reviewed -> TableCell("✓", Palette.validText, bold = true)
                else -> TableCell("")
            },
        )
    }
    DataTable(columns, rows, selected = selected, onRowClick = onSelect)
}

/** The racks left at the end, when they moved the score. */
@Composable
private fun EndOfGame(record: GameRecord) {
    val adjustments = record.endAdjustments()
    if (adjustments.all { it == 0 }) return
    val text = record.players.indices.joinToString(", ") { p ->
        val a = adjustments[p]
        "${record.players[p].name} ${if (a > 0) "+$a" else if (a < 0) "−${-a}" else "0"}"
    }
    Text(stringResource(R.string.review_end, text), color = Palette.muted, fontSize = 13.sp)
}

private fun cellsOf(t: RecordedTurn): List<Cell> = t.word.indices.map {
    if (t.horizontal) Cell(t.row, t.col + it) else Cell(t.row + it, t.col)
}

/** A play laid over the position it was there to be played on. */
private fun playOverlay(play: RecordedPlay, before: Position): BoardOverlay {
    val cells = play.word.indices.map { if (play.horizontal) Cell(play.row, play.col + it) else Cell(play.row + it, play.col) }
    val typed = cells.withIndex()
        .filter { (_, c) -> before.cells.getOrNull(c.index).isNullOrEmpty() }
        .associate { (i, c) -> c to play.word[i] }
    // `used` has one tile per newly covered square, in board order.
    val blanks = typed.keys.filterIndexed { i, _ -> play.used.getOrNull(i) == '?' }.toSet()
    return BoardOverlay(typed = typed, typedBlanks = blanks, typedThrough = cells.toSet() - typed.keys, typedValid = true)
}
