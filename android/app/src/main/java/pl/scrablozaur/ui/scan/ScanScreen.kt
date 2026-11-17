package pl.scrablozaur.ui.scan

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.view.HapticFeedbackConstants
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.systemBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.FlashlightOff
import androidx.compose.material.icons.filled.FlashlightOn
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import pl.scrablozaur.R
import pl.scrablozaur.engine.Hint
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.engine.ReadingVerdict
import pl.scrablozaur.engine.Sighting
import pl.scrablozaur.ui.components.BoardOverlay
import pl.scrablozaur.ui.components.BoardView
import pl.scrablozaur.ui.components.EngineBusy
import pl.scrablozaur.ui.components.HintList
import pl.scrablozaur.ui.components.LetterKeyboard
import pl.scrablozaur.ui.components.Mark
import pl.scrablozaur.ui.components.OrderChips
import pl.scrablozaur.ui.components.SmallButton
import pl.scrablozaur.ui.components.cells
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.theme.Palette

/** No seat owns a scanned tile: they are drawn plain. */
private val NO_OWNERS: List<Byte> = List(225) { -1 }

/** The grid on the preview: amber while the board moves, green once it reads as words, red when it does not. */
private val GRID_MOVING = Color(0xFFF59E0B)
private val GRID_GOOD = Color(0xFF22C55E)
private val GRID_BAD = Color(0xFFEF4444)

@Composable
fun ScanScreen(vm: ScanViewModel, premiums: List<Premium>, onBack: () -> Unit) {
    val ui by vm.ui.collectAsStateWithLifecycle()
    val view = LocalView.current
    LaunchedEffect(ui.accepted) {
        if (ui.accepted > 0) {
            view.performHapticFeedback(
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) HapticFeedbackConstants.CONFIRM else HapticFeedbackConstants.VIRTUAL_KEY,
            )
        }
    }
    // Back steps back through the loop before it leaves the screen.
    BackHandler(enabled = ui.phase == ScanPhase.Review) { vm.cancelReview() }
    BackHandler(enabled = ui.phase == ScanPhase.Camera && ui.board != null) { vm.backToAssistant() }

    when {
        ui.loading -> Message(onBack) {
            EngineBusy(Modifier.size(40.dp))
            Text(stringResource(R.string.scan_loading))
        }
        ui.loadError != null -> Message(onBack) {
            Text(stringResource(R.string.loading_failed, ui.loadError!!), color = Palette.danger, textAlign = TextAlign.Center)
        }
        ui.phase == ScanPhase.Camera && !ui.scannable -> Message(onBack) {
            Text(stringResource(R.string.scan_unavailable), textAlign = TextAlign.Center)
        }
        ui.phase == ScanPhase.Camera -> CameraPhase(ui, vm, onBack)
        ui.phase == ScanPhase.Review -> ReviewPhase(ui, vm, premiums)
        else -> AssistantPhase(ui, vm, premiums, onBack)
    }
}

@Composable
private fun Message(onBack: () -> Unit, content: @Composable () -> Unit) {
    Column(
        Modifier.fillMaxSize().systemBarsPadding().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp, Alignment.CenterVertically),
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        content()
        Button(onClick = onBack) { Text(stringResource(R.string.back)) }
    }
}

// ── Camera ───────────────────────────────────────────────────────────────────

@Composable
private fun CameraPhase(ui: ScanUi, vm: ScanViewModel, onBack: () -> Unit) {
    val context = LocalContext.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) ask.launch(Manifest.permission.CAMERA) }
    var torch by rememberSaveable { mutableStateOf(false) }
    val camera = ui.camera
    val reading = camera.reading
    val verdict = reading?.verdict

    Box(Modifier.fillMaxSize().background(Color.Black)) {
        if (granted) {
            BoardCamera(
                executor = vm.analysis,
                quad = camera.quad,
                color = when (verdict) {
                    ReadingVerdict.CHECKING, ReadingVerdict.ACCEPTED -> GRID_GOOD
                    ReadingVerdict.ILLEGAL -> GRID_BAD
                    else -> GRID_MOVING
                },
                torch = torch,
                onFrame = vm::onFrame,
                modifier = Modifier.fillMaxSize(),
            )
        } else {
            Column(
                Modifier.align(Alignment.Center).padding(24.dp),
                verticalArrangement = Arrangement.spacedBy(12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
            ) {
                Text(stringResource(R.string.scan_camera_needed), color = Color.White, textAlign = TextAlign.Center)
                Button(onClick = { ask.launch(Manifest.permission.CAMERA) }) { Text(stringResource(R.string.scan_camera_allow)) }
            }
        }
        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = onBack) {
                Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back), tint = Color.White)
            }
            Text(stringResource(R.string.scan_title), color = Color.White, fontWeight = FontWeight.Bold, fontSize = 18.sp, modifier = Modifier.weight(1f))
            if (granted) {
                IconButton(onClick = { torch = !torch }) {
                    Icon(if (torch) Icons.Filled.FlashlightOff else Icons.Filled.FlashlightOn, stringResource(R.string.scan_torch), tint = Color.White)
                }
            }
        }
        Column(
            Modifier
                .align(Alignment.BottomCenter)
                .navigationBarsPadding()
                .padding(12.dp)
                .fillMaxWidth()
                .background(Palette.surface, RoundedCornerShape(12.dp))
                .padding(14.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            val status = when {
                camera.sighting == Sighting.NO_BOARD -> stringResource(R.string.scan_aim)
                verdict == null -> stringResource(R.string.scan_hold_still)
                verdict == ReadingVerdict.UNSEEN -> stringResource(R.string.scan_unseen)
                verdict == ReadingVerdict.ILLEGAL ->
                    reading!!.board.flagged.count { it }.let { pluralStringResource(R.plurals.scan_not_words, it, it) }
                else -> stringResource(R.string.scan_checking)
            }
            Text(status, fontSize = 15.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                if (ui.board != null) {
                    SmallButton(R.string.scan_back_to_board, vm::backToAssistant)
                } else {
                    SmallButton(R.string.scan_empty_board, vm::startEmpty)
                }
                Spacer(Modifier.weight(1f))
                if (reading != null && reading.verdict != ReadingVerdict.UNSEEN) SmallButton(R.string.scan_fix_by_hand, vm::fixByHand)
            }
        }
    }
}

// ── Review ───────────────────────────────────────────────────────────────────

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun ReviewPhase(ui: ScanUi, vm: ScanViewModel, premiums: List<Premium>) {
    val review = ui.review ?: return
    val language = ui.language ?: return
    val prior = ui.board
    val marks = buildMap {
        for (i in 0 until 225) {
            when {
                review.flagged[i] -> put(Cell.of(i), Mark.Flagged)
                review.corrected[i] -> put(Cell.of(i), Mark.Corrected)
            }
        }
    }
    // Only a later scan has new tiles worth pointing out: on the first, all are.
    val fresh = if (prior == null) emptySet() else {
        (0 until 225).filter { review.letters[it].isNotEmpty() && prior.letters[it].isEmpty() }.map(Cell::of).toSet()
    }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.scan_review_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = vm::cancelReview) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            if (review.flaggedCount > 0) {
                Text(
                    pluralStringResource(R.plurals.scan_flagged, review.flaggedCount, review.flaggedCount),
                    color = Palette.danger,
                    fontSize = 14.sp,
                    modifier = Modifier.padding(horizontal = 4.dp),
                )
            }
            BoardView(
                cells = review.letters,
                blanks = review.blanks,
                owners = NO_OWNERS,
                premiums = premiums,
                points = ui.points,
                overlay = BoardOverlay(marks = marks, fresh = fresh, cursor = review.selected),
                modifier = Modifier.fillMaxWidth(),
                onTap = vm::selectCell,
            )
            val selected = review.selected
            if (selected != null) {
                val i = selected.index
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(stringResource(R.string.scan_cell, selected.row + 1, selected.col + 1), fontSize = 13.sp, color = Palette.muted)
                    // The reader's other guesses for this square, one tap each.
                    FlowRow(Modifier.weight(1f), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        review.alternatives[i].filter { it != review.letters[i] }.forEach { alt ->
                            Box(
                                Modifier
                                    .background(Palette.highlightExisting, RoundedCornerShape(6.dp))
                                    .clickable { vm.typeLetter(alt.first()) }
                                    .padding(horizontal = 10.dp, vertical = 4.dp),
                            ) { Text(alt.uppercase(), fontWeight = FontWeight.Bold) }
                        }
                    }
                    TextButton(onClick = vm::clearCell) { Text(stringResource(R.string.scan_clear_cell)) }
                }
                LetterKeyboard(
                    language.alphabet,
                    extraKeys = language.blank.toString(),
                    onLetter = vm::typeLetter,
                    onBackspace = vm::backspace,
                )
            } else {
                Text(stringResource(R.string.scan_review_help), fontSize = 13.sp, color = Palette.muted, modifier = Modifier.padding(horizontal = 4.dp))
            }
            if (review.fromCamera) {
                Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.clickable { vm.setSaveExample(!review.saveExample) }) {
                    Checkbox(checked = review.saveExample, onCheckedChange = vm::setSaveExample)
                    Text(stringResource(R.string.scan_save_example), fontSize = 14.sp)
                }
            }
            Row(Modifier.fillMaxWidth().padding(bottom = 12.dp), verticalAlignment = Alignment.CenterVertically) {
                SmallButton(if (review.fromCamera) R.string.scan_rescan else R.string.cancel, vm::cancelReview)
                Spacer(Modifier.weight(1f))
                Button(onClick = vm::confirm) { Text(stringResource(R.string.confirm)) }
            }
        }
    }
}

// ── Assistant ────────────────────────────────────────────────────────────────

/** A suggestion on the board: its new tiles typed in, the tiles it runs through tinted. */
private fun hintOverlay(hint: Hint?, letters: List<String>): BoardOverlay {
    if (hint == null) return BoardOverlay()
    val cells = hint.cells()
    val typed = cells.withIndex()
        .filter { (_, c) -> letters[c.index].isEmpty() }
        .associate { (i, c) -> c to hint.word[i] }
    // `used` has one tile per newly covered square, in board order.
    val blanks = typed.keys.filterIndexed { i, _ -> hint.used.getOrNull(i) == '?' }.toSet()
    return BoardOverlay(typed = typed, typedBlanks = blanks, typedThrough = cells.toSet() - typed.keys, typedValid = true)
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun AssistantPhase(ui: ScanUi, vm: ScanViewModel, premiums: List<Premium>, onBack: () -> Unit) {
    val board = ui.board ?: return
    val language = ui.language ?: return
    val assistant = ui.assistant
    var confirmNew by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.scan_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(horizontal = 8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            BoardView(
                cells = board.letters,
                blanks = board.blanks,
                owners = NO_OWNERS,
                premiums = premiums,
                points = ui.points,
                overlay = hintOverlay(assistant.selected, board.letters),
                modifier = Modifier.fillMaxWidth(),
            )
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Column(Modifier.weight(1f).clickable(enabled = assistant.suggestions != null, onClick = vm::editRack)) {
                    Text(stringResource(R.string.your_letters), fontSize = 11.sp, color = Palette.muted)
                    Text(
                        assistant.rack.uppercase().replace("?", "★").ifEmpty { stringResource(R.string.computer_letters_hint) },
                        fontSize = 20.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (assistant.rack.isEmpty()) Palette.muted else Palette.text,
                    )
                }
                Button(onClick = vm::search, enabled = !assistant.busy && assistant.rack.isNotEmpty()) {
                    if (assistant.busy) EngineBusy(Modifier.size(18.dp)) else Text(stringResource(R.string.search))
                }
            }
            val suggestions = assistant.suggestions
            if (suggestions == null) {
                LetterKeyboard(
                    language.alphabet,
                    extraKeys = language.blank.toString(),
                    onLetter = { vm.setRack(assistant.rack + it) },
                    onBackspace = { vm.setRack(assistant.rack.dropLast(1)) },
                )
            } else {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    OrderChips(assistant.order, language.hasLeaveNet, !assistant.busy, vm::setOrder, Modifier.weight(1f))
                    TextButton(onClick = vm::editRack) { Text(stringResource(R.string.change_letters)) }
                }
                HintList(
                    suggestions,
                    assistant.order,
                    assistant.selected,
                    enabled = !assistant.busy,
                    onSelect = vm::selectHint,
                    empty = R.string.no_moves_for_letters,
                )
            }
            ui.message?.let {
                Text(it, fontSize = 14.sp, color = Palette.muted, modifier = Modifier.fillMaxWidth().clickable(onClick = vm::dismissMessage))
            }
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth().padding(bottom = 12.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                Button(onClick = vm::scanAgain, enabled = ui.scannable) { Text(stringResource(R.string.scan_next)) }
                SmallButton(R.string.scan_correct_board, vm::correctBoard)
                Spacer(Modifier.weight(1f))
                SmallButton(R.string.scan_new_session, { confirmNew = true }, danger = true)
            }
        }
    }
    if (confirmNew) {
        AlertDialog(
            onDismissRequest = { confirmNew = false },
            text = { Text(stringResource(R.string.scan_new_session_confirm)) },
            confirmButton = {
                TextButton(onClick = { confirmNew = false; vm.newSession() }) { Text(stringResource(R.string.scan_new_session)) }
            },
            dismissButton = { TextButton(onClick = { confirmNew = false }) { Text(stringResource(R.string.cancel)) } },
        )
    }
}
