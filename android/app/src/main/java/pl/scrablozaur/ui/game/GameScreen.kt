package pl.scrablozaur.ui.game

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Backspace
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.CameraAlt
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.Pause
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SkipNext
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import pl.scrablozaur.R
import pl.scrablozaur.core.DefinitionResult
import pl.scrablozaur.engine.Hint
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.Outcome
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.engine.PreviewResult
import pl.scrablozaur.ui.components.BoardOverlay
import pl.scrablozaur.ui.components.BoardView
import pl.scrablozaur.ui.components.EngineBusy
import pl.scrablozaur.ui.components.HintList
import pl.scrablozaur.ui.components.OrderChips
import pl.scrablozaur.ui.components.SmallButton
import pl.scrablozaur.ui.components.cells
import pl.scrablozaur.ui.components.LetterKeyboard
import pl.scrablozaur.ui.components.Mark
import pl.scrablozaur.ui.components.RackView
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors

/** What the board shows on top of the position: a word being typed, a hint
 *  being previewed, or else where the computer just played. */
private fun overlayFor(ui: GameUi): BoardOverlay {
    val typing = ui.typing
    if (typing != null) {
        // Only a competitive game has a rack to draw blanks from; the sandbox
        // scores every typed tile at face value.
        val view = ui.view
        val blank = ui.language?.blank ?: '?'
        val rack = if (view?.mode == Mode.COMPETITIVE) view.players.firstOrNull { !it.isComputer }?.letters.orEmpty() else ""
        val allocation = rackAllocation(rack, typing.typedLetters, blank)
        val placed = typing.entries.filter { it.letter != null }.map { it.cell }
        return BoardOverlay(
            typed = typing.placed,
            typedBlanks = placed.filterIndexed { i, _ -> allocation.getOrNull(i)?.let { rack[it] == blank } == true }.toSet(),
            typedThrough = typing.passedThrough,
            cursor = typing.cursor,
            typedValid = ui.preview?.let { it is PreviewResult.Score },
        )
    }
    val hint = ui.selectedHint
    if (hint != null) {
        val grid = ui.grid
        val cells = hint.cells()
        val typed = cells.withIndex()
            .filter { (_, c) -> grid.at(c.row, c.col) == null }
            .associate { (i, c) -> c to hint.word[i] }
        // `used` has one tile per newly covered square, in board order.
        val blanks = typed.keys.filterIndexed { i, _ -> hint.used.getOrNull(i) == '?' }.toSet()
        return BoardOverlay(typed = typed, typedBlanks = blanks, typedThrough = cells.toSet() - typed.keys, typedValid = true)
    }
    val last = ui.view?.lastComputerMove
    if (last != null && !last.passed) {
        val cells = (0 until last.word.length).map {
            if (last.horizontal) Cell(last.row.toInt(), last.col.toInt() + it) else Cell(last.row.toInt() + it, last.col.toInt())
        }
        return BoardOverlay(marks = cells.associateWith { Mark.LastMove })
    }
    return BoardOverlay()
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun GameScreen(
    vm: GameViewModel,
    premiums: List<Premium>,
    onNewGame: () -> Unit,
    onScan: () -> Unit,
) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    var confirmResign by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.app_name), fontWeight = FontWeight.Bold) },
                actions = {
                    // Undo also stops autoplay, so it stays available while that runs.
                    IconButton(onClick = vm::undo, enabled = ui.view?.canUndo == true && (ui.idle || ui.autoplay)) {
                        Icon(Icons.AutoMirrored.Filled.Undo, stringResource(R.string.undo))
                    }
                    IconButton(onClick = onScan) { Icon(Icons.Filled.CameraAlt, stringResource(R.string.scan)) }
                    IconButton(onClick = onNewGame) { Icon(Icons.Filled.Add, stringResource(R.string.new_game)) }
                },
            )
        },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize()) {
            when {
                ui.loading -> Column(Modifier.align(Alignment.Center), horizontalAlignment = Alignment.CenterHorizontally) {
                    EngineBusy(Modifier.size(40.dp))
                    Spacer(Modifier.height(12.dp))
                    Text(stringResource(R.string.loading_engine))
                }
                ui.loadError != null -> Column(Modifier.align(Alignment.Center).padding(24.dp)) {
                    Text(stringResource(R.string.loading_failed, ui.loadError!!), color = Palette.danger)
                    Spacer(Modifier.height(12.dp))
                    Button(onClick = onNewGame) { Text(stringResource(R.string.new_game)) }
                }
                ui.view != null -> GameLayout(ui, vm, premiums, onNewGame) { confirmResign = true }
            }
        }
    }

    if (confirmResign) {
        AlertDialog(
            onDismissRequest = { confirmResign = false },
            text = { Text(stringResource(R.string.resign_confirm)) },
            confirmButton = {
                TextButton(onClick = { confirmResign = false; vm.resign() }) { Text(stringResource(R.string.resign)) }
            },
            dismissButton = { TextButton(onClick = { confirmResign = false }) { Text(stringResource(R.string.cancel)) } },
        )
    }
    if (ui.askBlank) BlankDialog(ui.language?.alphabet.orEmpty(), vm::onBlankChosen)
}

/** Board and controls stacked on a portrait phone, side by side in landscape. */
@Composable
private fun GameLayout(ui: GameUi, vm: GameViewModel, premiums: List<Premium>, onNewGame: () -> Unit, onResign: () -> Unit) {
    val view = ui.view!!
    val overlay = overlayFor(ui)
    @Composable
    fun Board(modifier: Modifier) = BoardView(
        cells = view.cells,
        blanks = view.blanks,
        owners = view.owners,
        premiums = premiums,
        points = ui.points,
        overlay = overlay,
        modifier = modifier,
        onTap = vm::onCellTap,
    )
    BoxWithConstraints(Modifier.fillMaxSize()) {
        val width = maxWidth
        val height = maxHeight
        if (width > height) {
            Row(Modifier.fillMaxSize()) {
                Board(Modifier.size(minOf(height, width * 0.6f)).padding(4.dp))
                Column(Modifier.weight(1f).fillMaxHeight()) {
                    Scoreboard(ui)
                    Panel(ui, vm, onNewGame, onResign, Modifier.weight(1f))
                }
            }
        } else {
            // The board gives way before the controls do, so a short window
            // (split screen) still leaves room to play.
            val board = minOf(width, (height - 220.dp).coerceAtLeast(160.dp))
            Column(Modifier.fillMaxSize()) {
                Scoreboard(ui)
                Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
                    Board(Modifier.size(board).padding(horizontal = 4.dp))
                }
                Panel(ui, vm, onNewGame, onResign, Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun Scoreboard(ui: GameUi) {
    val view = ui.view!!
    // The web highlights whose turn it is only where seats take turns at one
    // device; in a competitive game the panel already says it.
    val showActive = view.mode != Mode.COMPETITIVE
    Row(
        Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp),
        horizontalArrangement = Arrangement.spacedBy(10.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(stringResource(R.string.move_n, view.moveNumber.toInt() + 1), color = Palette.muted, fontSize = 13.sp)
        if (view.mode != Mode.SANDBOX) {
            Text("${stringResource(R.string.bag)} ${view.tilesRemaining}", color = Palette.muted, fontSize = 13.sp)
        }
        Spacer(Modifier.weight(1f))
        view.players.forEachIndexed { i, p ->
            val active = showActive && i == view.currentPlayer.toInt() && !view.gameOver
            Row(
                Modifier
                    .background(if (active) Palette.highlightNew else Palette.surface, RoundedCornerShape(12.dp))
                    .padding(horizontal = 8.dp, vertical = 3.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Box(Modifier.size(9.dp).background(Palette.owners[i % 4].second, CircleShape))
                Spacer(Modifier.width(4.dp))
                Text(p.name, fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis, modifier = Modifier.weight(1f, fill = false))
                Spacer(Modifier.width(4.dp))
                Text(p.score.toString(), fontWeight = FontWeight.Bold, fontSize = 14.sp)
            }
        }
    }
}

private enum class PanelKind { Over, Competitive, Hints, Auto, SandboxHuman, SandboxComputer }

@Composable
private fun Panel(ui: GameUi, vm: GameViewModel, onNewGame: () -> Unit, onResign: () -> Unit, modifier: Modifier) {
    val view = ui.view!!
    val kind = when {
        view.gameOver -> PanelKind.Over
        view.mode == Mode.COMPETITIVE -> if (ui.hints != null) PanelKind.Hints else PanelKind.Competitive
        view.mode == Mode.SANDBOX_AUTO -> PanelKind.Auto
        ui.current?.isComputer == true -> PanelKind.SandboxComputer
        else -> PanelKind.SandboxHuman
    }
    val scroll = rememberScrollState()
    LaunchedEffect(kind, ui.suggestions == null) { scroll.scrollTo(0) }
    Column(
        modifier.fillMaxWidth().verticalScroll(scroll).padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        ui.definition?.let { DefinitionCard(it, vm::closeDefinition) }
        when (kind) {
            PanelKind.Over -> GameOverCard(ui, onNewGame)
            PanelKind.Competitive -> CompetitivePanel(ui, vm, onResign)
            PanelKind.Hints -> HintsPanel(ui, vm)
            PanelKind.Auto -> AutoPanel(ui, vm)
            PanelKind.SandboxHuman -> SandboxHumanPanel(ui, vm, onResign)
            PanelKind.SandboxComputer -> SandboxComputerPanel(ui, vm)
        }
    }
}

// ── Shared pieces ────────────────────────────────────────────────────────────

@Composable
private fun ConfirmButton(ui: GameUi, onClick: () -> Unit) {
    Button(onClick = onClick, enabled = ui.idle) {
        if (ui.busy == Busy.Move) EngineBusy(Modifier.size(18.dp))
        else Text(stringResource(R.string.confirm))
    }
}

@Composable
private fun ErrorLine(ui: GameUi, vm: GameViewModel) {
    ui.error?.let {
        Text(it, color = Palette.danger, fontSize = 14.sp, modifier = Modifier.fillMaxWidth().clickable(onClick = vm::dismissError))
    }
}

@Composable
private fun Labeled(label: String, value: String, modifier: Modifier = Modifier, color: Color = Palette.text) {
    Column(modifier) {
        Text(label, fontSize = 11.sp, color = Palette.muted)
        Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis)
    }
}

/** Direction, and the word being typed with its score -- or how to start one. */
@Composable
private fun WordStatus(ui: GameUi, vm: GameViewModel) {
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        FilterChip(
            selected = ui.typing != null,
            onClick = vm::toggleDirection,
            label = {
                Text((if (ui.horizontal) "→ " else "↓ ") + stringResource(if (ui.horizontal) R.string.horizontal else R.string.vertical))
            },
        )
        if (ui.typing == null) {
            Text(stringResource(R.string.tap_start_square), color = Palette.muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
        } else {
            Labeled(stringResource(R.string.word), ui.typedWord?.word?.uppercase() ?: "—", Modifier.weight(1f))
            val preview = ui.preview
            Labeled(
                stringResource(R.string.points),
                if (preview is PreviewResult.Score) stringResource(R.string.points_short, preview.points.toInt()) else "—",
                color = when (preview) {
                    is PreviewResult.Score -> Palette.validText
                    null -> Palette.text
                    else -> Palette.invalidText
                },
            )
        }
    }
}

// ── Competitive ──────────────────────────────────────────────────────────────

@Composable
private fun CompetitivePanel(ui: GameUi, vm: GameViewModel, onResign: () -> Unit) {
    val view = ui.view!!
    val letters = view.players.firstOrNull { !it.isComputer }?.letters.orEmpty()
    val blank = ui.language?.blank ?: '?'
    val last = view.lastComputerMove
    if (ui.thinking) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            EngineBusy(Modifier.size(16.dp))
            Spacer(Modifier.width(8.dp))
            Text(stringResource(R.string.computer_thinking), color = Palette.muted)
        }
    } else if (last != null) {
        Text(
            if (last.passed) stringResource(R.string.computer_passes)
            else "${last.word.uppercase()} · ${stringResource(R.string.points_short, last.score.toInt())}",
            color = Palette.muted,
        )
    }
    RackView(
        letters = letters,
        blank = blank,
        points = ui.points,
        used = ui.typing?.let { rackTilesUsed(letters, it.typedLetters, blank) } ?: emptySet(),
        selected = ui.exchangeSelection,
        placing = ui.typing != null,
        onTap = vm::onRackTap,
    )
    WordStatus(ui, vm)
    ErrorLine(ui, vm)
    if (ui.typing != null) {
        // The rack is the keyboard now; these are the ways out of typing.
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            IconButton(onClick = vm::backspace) { Icon(Icons.AutoMirrored.Filled.Backspace, stringResource(R.string.backspace)) }
            SmallButton(R.string.cancel, vm::cancelTyping)
            Spacer(Modifier.weight(1f))
            ConfirmButton(ui, vm::submitTyped)
        }
    } else {
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
            SmallButton(R.string.hints, vm::toggleHints, enabled = ui.idle, busy = ui.busy == Busy.Hints)
            SmallButton(R.string.exchange, vm::exchange, enabled = ui.idle, busy = ui.busy == Busy.Exchange)
            SmallButton(R.string.skip, vm::skip, enabled = ui.idle)
            SmallButton(R.string.resign, onResign, enabled = ui.idle, danger = true)
        }
        ExchangeHelp()
    }
    if (ui.ratings.isNotEmpty()) RatingCard(ui)
}

@Composable
private fun ExchangeHelp() {
    var open by rememberSaveable { mutableStateOf(false) }
    Column {
        Text(
            (if (open) "▾ " else "▸ ") + stringResource(R.string.how_to_exchange),
            fontSize = 13.sp,
            color = Palette.muted,
            modifier = Modifier.clickable { open = !open }.padding(vertical = 4.dp),
        )
        if (open) Text(stringResource(R.string.exchange_help), fontSize = 13.sp, color = Palette.muted)
    }
}

@Composable
private fun HintsPanel(ui: GameUi, vm: GameViewModel) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(stringResource(R.string.hints), fontWeight = FontWeight.SemiBold, fontSize = 16.sp, modifier = Modifier.weight(1f))
        TextButton(onClick = vm::toggleHints) { Text(stringResource(R.string.hide)) }
    }
    OrderChips(ui.hintOrder, ui.language?.hasLeaveNet == true, enabled = ui.idle, onChange = vm::setHintOrder)
    ErrorLine(ui, vm)
    if (ui.busy == Busy.Hints) EngineBusy(Modifier.size(20.dp))
    HintList(ui.hints.orEmpty(), ui.hintOrder, ui.selectedHint, enabled = ui.idle, onSelect = vm::selectHint, onPlace = vm::placeHint)
}

private fun ratingColor(r: Int) = when {
    r >= 70 -> Palette.ratingGood
    r >= 40 -> Palette.ratingOk
    else -> Palette.ratingBad
}

@Composable
private fun Gauge(rating: Int, color: Color, modifier: Modifier) {
    Canvas(modifier) {
        val stroke = size.height * 0.26f
        val d = minOf(size.width, size.height * 2) - stroke
        val topLeft = Offset((size.width - d) / 2, stroke / 2)
        val arc = Size(d, d)
        drawArc(Palette.border, 180f, 180f, false, topLeft, arc, style = Stroke(stroke, cap = StrokeCap.Round))
        drawArc(color, 180f, 180f * rating / 100f, false, topLeft, arc, style = Stroke(stroke, cap = StrokeCap.Round))
    }
}

/** How the last move compared with the best the rack allowed; tap for the history. */
@Composable
private fun RatingCard(ui: GameUi) {
    val last = ui.ratings.first()
    var expanded by rememberSaveable { mutableStateOf(false) }
    val color = ratingColor(last.rating)
    Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column(
            Modifier
                .fillMaxWidth()
                .clickable(enabled = ui.ratings.size > 1) { expanded = !expanded }
                .padding(horizontal = 12.dp, vertical = 8.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Gauge(last.rating, color, Modifier.size(56.dp, 30.dp))
                Spacer(Modifier.width(12.dp))
                Column(Modifier.weight(1f)) {
                    Text(
                        "${stringResource(R.string.last_move)} · ${last.word.uppercase()}",
                        fontSize = 12.sp,
                        color = Palette.muted,
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                    )
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(stringResource(R.string.rating_value, last.rating), color = color, fontWeight = FontWeight.Bold, fontSize = 18.sp)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            stringResource(
                                when {
                                    last.rating >= 85 -> R.string.rating_great
                                    last.rating >= 60 -> R.string.rating_good
                                    last.rating >= 35 -> R.string.rating_ok
                                    else -> R.string.rating_missed
                                },
                            ),
                            fontSize = 14.sp,
                        )
                    }
                }
                if (ui.ratings.size > 1) {
                    Icon(if (expanded) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore, stringResource(R.string.history))
                }
            }
            if (expanded) {
                Spacer(Modifier.height(8.dp))
                ui.ratings.take(20).forEach { m ->
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(m.word.uppercase(), Modifier.width(96.dp), fontWeight = FontWeight.SemiBold, maxLines = 1, overflow = TextOverflow.Ellipsis)
                        Text(stringResource(R.string.points_short, m.score), Modifier.width(52.dp), fontSize = 12.sp)
                        Box(Modifier.weight(1f).height(6.dp).background(Palette.border, RoundedCornerShape(3.dp))) {
                            Box(Modifier.fillMaxWidth(m.rating / 100f).height(6.dp).background(ratingColor(m.rating), RoundedCornerShape(3.dp)))
                        }
                        Text(m.rating.toString(), color = ratingColor(m.rating), fontSize = 12.sp)
                    }
                }
            }
        }
    }
}

// ── Sandbox ──────────────────────────────────────────────────────────────────

@Composable
private fun SandboxHumanPanel(ui: GameUi, vm: GameViewModel, onResign: () -> Unit) {
    Text(stringResource(R.string.player_turn, ui.current?.name.orEmpty()), fontWeight = FontWeight.SemiBold)
    WordStatus(ui, vm)
    ErrorLine(ui, vm)
    if (ui.typing != null) {
        LetterKeyboard(ui.language?.alphabet.orEmpty(), onLetter = vm::typeLetter, onBackspace = vm::backspace)
        Row(verticalAlignment = Alignment.CenterVertically) {
            SmallButton(R.string.cancel, vm::cancelTyping)
            Spacer(Modifier.weight(1f))
            ConfirmButton(ui, vm::submitTyped)
        }
    } else {
        Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            SmallButton(R.string.skip, vm::skip, enabled = ui.idle)
            SmallButton(R.string.resign, onResign, enabled = ui.idle, danger = true)
        }
    }
}

@Composable
private fun SandboxComputerPanel(ui: GameUi, vm: GameViewModel) {
    val suggestions = ui.suggestions
    Text(stringResource(R.string.computer_turn, ui.current?.name.orEmpty()), fontWeight = FontWeight.SemiBold)
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Column(Modifier.weight(1f).clickable(enabled = suggestions != null, onClick = vm::editComputerLetters)) {
            Text(stringResource(R.string.computer_letters), fontSize = 11.sp, color = Palette.muted)
            Text(
                ui.computerLetters.uppercase().replace("?", "★").ifEmpty { stringResource(R.string.computer_letters_hint) },
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                color = if (ui.computerLetters.isEmpty()) Palette.muted else Palette.text,
            )
        }
        Button(onClick = vm::searchSuggestions, enabled = ui.idle && ui.computerLetters.isNotEmpty()) {
            if (ui.busy == Busy.Suggest) EngineBusy(Modifier.size(18.dp))
            else Text(stringResource(R.string.search))
        }
        SmallButton(R.string.skip, vm::skip, enabled = ui.idle)
    }
    ErrorLine(ui, vm)
    if (suggestions == null) {
        LetterKeyboard(
            ui.language?.alphabet.orEmpty(),
            extraKeys = ui.language?.blank?.toString().orEmpty(),
            onLetter = { vm.setComputerLetters(ui.computerLetters + it) },
            onBackspace = { vm.setComputerLetters(ui.computerLetters.dropLast(1)) },
        )
    } else {
        Row(verticalAlignment = Alignment.CenterVertically) {
            OrderChips(ui.suggestionOrder, ui.language?.hasLeaveNet == true, ui.idle, vm::setSuggestionOrder, Modifier.weight(1f))
            TextButton(onClick = vm::editComputerLetters) { Text(stringResource(R.string.change_letters)) }
        }
        if (ui.busy == Busy.Suggest) EngineBusy(Modifier.size(20.dp))
        HintList(suggestions, ui.suggestionOrder, ui.selectedHint, enabled = ui.idle, onSelect = vm::selectHint, onPlace = vm::placeSuggestion)
    }
}

@Composable
private fun AutoPanel(ui: GameUi, vm: GameViewModel) {
    Text(stringResource(R.string.auto_game), fontWeight = FontWeight.SemiBold)
    ui.current?.let { Text(stringResource(R.string.auto_on_move, it.name, it.strength.toInt()), color = Palette.muted, fontSize = 13.sp) }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Button(onClick = vm::nextAutoMove, enabled = !ui.autoplay && ui.idle) {
            if (ui.busy == Busy.Auto && !ui.autoplay) {
                EngineBusy(Modifier.size(18.dp))
            } else {
                Icon(Icons.Filled.SkipNext, null)
                Spacer(Modifier.width(4.dp))
                Text(stringResource(R.string.auto_next))
            }
        }
        OutlinedButton(onClick = vm::toggleAutoplay, enabled = ui.autoplay || ui.idle, colors = outlinedButtonColors()) {
            Icon(if (ui.autoplay) Icons.Filled.Pause else Icons.Filled.PlayArrow, null)
            Spacer(Modifier.width(4.dp))
            Text(stringResource(if (ui.autoplay) R.string.auto_stop else R.string.auto_play))
        }
    }
    ErrorLine(ui, vm)
    ui.autoLog.take(60).forEach { e ->
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(e.player, Modifier.weight(1f), color = Palette.muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
            if (e.passed) {
                Text(stringResource(R.string.passed), color = Palette.muted)
            } else {
                Text(e.word.uppercase(), fontWeight = FontWeight.SemiBold)
                Text(stringResource(R.string.points_short, e.score))
            }
        }
    }
}

// ── Endings and dialogs ──────────────────────────────────────────────────────

@Composable
private fun GameOverCard(ui: GameUi, onNewGame: () -> Unit) {
    val view = ui.view!!
    val outcome = view.outcome
    Card(colors = CardDefaults.cardColors(containerColor = Palette.highlightNew)) {
        Column(Modifier.fillMaxWidth().padding(16.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            Text(
                when (outcome) {
                    is Outcome.Winner -> stringResource(R.string.winner, view.players[outcome.player.toInt()].name)
                    else -> stringResource(R.string.tie)
                },
                style = MaterialTheme.typography.headlineSmall,
                fontWeight = FontWeight.Bold,
            )
            Spacer(Modifier.height(8.dp))
            view.players.sortedByDescending { it.score }.forEach { p ->
                Row(Modifier.fillMaxWidth(0.7f)) {
                    Text(p.name, Modifier.weight(1f))
                    Text(stringResource(R.string.points_short, p.score))
                }
            }
            Spacer(Modifier.height(12.dp))
            Button(onClick = onNewGame) { Text(stringResource(R.string.new_game)) }
        }
    }
}

@Composable
private fun BlankDialog(alphabet: String, onChosen: (Char?) -> Unit) {
    AlertDialog(
        onDismissRequest = { onChosen(null) },
        title = { Text(stringResource(R.string.blank_letter_title)) },
        text = { LetterKeyboard(alphabet, perRow = 8, onLetter = { onChosen(it) }, onBackspace = { onChosen(null) }) },
        confirmButton = {},
        dismissButton = { TextButton(onClick = { onChosen(null) }) { Text(stringResource(R.string.cancel)) } },
    )
}

/** A word's definition, inline above the controls like the web's side panel:
 *  never modal, so the board stays usable while it is open. */
@Composable
fun DefinitionCard(definition: DefinitionUi, onClose: () -> Unit, modifier: Modifier = Modifier) {
    Card(modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column(Modifier.padding(start = 12.dp, end = 4.dp, bottom = 10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    definition.word.uppercase(),
                    style = MaterialTheme.typography.titleMedium,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.weight(1f),
                )
                IconButton(onClick = onClose) { Icon(Icons.Filled.Close, stringResource(R.string.close)) }
            }
            Column(Modifier.padding(end = 8.dp)) {
                when (val r = definition.result) {
                    null -> Text(stringResource(R.string.definition_loading), color = Palette.muted)
                    is DefinitionResult.Found -> r.entries.forEach { entry ->
                        entry.split("<br>").forEach { line -> Text(line, fontSize = 14.sp, modifier = Modifier.padding(vertical = 1.dp)) }
                        Spacer(Modifier.height(4.dp))
                    }
                    DefinitionResult.NotFound -> Text(stringResource(R.string.definition_none), color = Palette.muted)
                    DefinitionResult.Offline -> Text(stringResource(R.string.definition_offline), color = Palette.muted)
                }
            }
        }
    }
}
