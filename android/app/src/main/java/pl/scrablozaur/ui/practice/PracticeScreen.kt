package pl.scrablozaur.ui.practice

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Backspace
import androidx.compose.material.icons.filled.ChevronLeft
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material.icons.filled.Shuffle
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.scrablozaur.R
import pl.scrablozaur.core.Definitions
import pl.scrablozaur.core.GameRecord
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.Puzzle
import pl.scrablozaur.core.RecordedPlay
import pl.scrablozaur.core.allSeats
import pl.scrablozaur.core.playerKey
import pl.scrablozaur.core.positions
import pl.scrablozaur.core.puzzles
import pl.scrablozaur.engine.LanguagePackHandle
import pl.scrablozaur.engine.PlayCheck
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.ui.components.BoardOverlay
import pl.scrablozaur.ui.components.BoardView
import pl.scrablozaur.ui.components.EngineBusy
import pl.scrablozaur.ui.components.RackView
import pl.scrablozaur.ui.components.SectionTitle
import pl.scrablozaur.ui.components.formatValue
import pl.scrablozaur.ui.game.BlankDialog
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.game.DefinitionCard
import pl.scrablozaur.ui.game.DefinitionUi
import pl.scrablozaur.ui.game.Grid
import pl.scrablozaur.ui.game.Typing
import pl.scrablozaur.ui.game.rackAllocation
import pl.scrablozaur.ui.game.rackTilesUsed
import pl.scrablozaur.ui.theme.Palette
import kotlin.random.Random

enum class PracticeTab { Puzzles, Anagrams, Words }

/** A rack an anagram drill deals: seven tiles, no blank. */
private const val DRILL_TILES = 7

/** The letters a search may draw on, and how many of them blanks: enough to be useful, few enough to stay quick. */
private const val SEARCH_POOL = 10
private const val SEARCH_BLANKS = 2

/** Words a search shows; the rest are counted. */
private const val SEARCH_SHOWN = 300

/**
 * "Trening": the player's own worst moves set up again to be found properly,
 * anagram drills on seven drawn tiles, and the dictionary to check a word in
 * or search by pattern.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PracticeScreen(
    languages: List<LanguageMeta>,
    defaultLanguage: String,
    records: List<GameRecord>?,
    playerName: String,
    premiums: List<Premium>,
    loadPack: suspend (String) -> LanguagePackHandle,
    /** Where a word is looked up; null when definitions are switched off. */
    definitions: Definitions?,
    onBack: () -> Unit,
) {
    var tab by rememberSaveable { mutableStateOf(PracticeTab.Puzzles) }
    var languageCode by rememberSaveable { mutableStateOf(defaultLanguage) }
    val language = languages.firstOrNull { it.code == languageCode } ?: languages.first()
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.practice_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = tab.ordinal, containerColor = Palette.background) {
                PracticeTab.entries.forEach { t ->
                    Tab(
                        selected = tab == t,
                        onClick = { tab = t },
                        text = {
                            Text(
                                stringResource(
                                    when (t) {
                                        PracticeTab.Puzzles -> R.string.practice_puzzles
                                        PracticeTab.Anagrams -> R.string.practice_anagrams
                                        PracticeTab.Words -> R.string.practice_words
                                    },
                                ),
                            )
                        },
                    )
                }
            }
            // A puzzle comes in its game's language; the other tabs ask.
            if (tab != PracticeTab.Puzzles && languages.size > 1) {
                FlowRow(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    languages.forEach { l ->
                        FilterChip(selected = l.code == language.code, onClick = { languageCode = l.code }, label = { Text("${l.flag} ${l.name}".trim()) })
                    }
                }
            }
            when (tab) {
                PracticeTab.Puzzles -> Puzzles(records, playerName, languages, premiums, loadPack)
                PracticeTab.Anagrams -> WithPack(language, loadPack) { Anagrams(language, it) }
                PracticeTab.Words -> WithPack(language, loadPack) { Words(language, it, definitions) }
            }
        }
    }
}

/** `content` once the language's dictionary is loaded. */
@Composable
private fun WithPack(language: LanguageMeta, loadPack: suspend (String) -> LanguagePackHandle, content: @Composable (LanguagePackHandle) -> Unit) {
    val pack by produceState<LanguagePackHandle?>(null, language.code) { value = null; value = loadPack(language.code) }
    val loaded = pack
    if (loaded == null) {
        Column(Modifier.fillMaxWidth().padding(24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
            EngineBusy(Modifier.size(32.dp))
            Text(stringResource(R.string.loading_engine), color = Palette.muted)
        }
    } else {
        content(loaded)
    }
}

// ── Zadania: the player's own mistakes ───────────────────────────────────────

@Composable
private fun Puzzles(
    records: List<GameRecord>?,
    playerName: String,
    languages: List<LanguageMeta>,
    premiums: List<Premium>,
    loadPack: suspend (String) -> LanguagePackHandle,
) {
    if (records == null) {
        EngineBusy(Modifier.padding(24.dp).size(32.dp))
        return
    }
    // The phone's player's games; with no name set, everyone's.
    val list = remember(records, playerName) {
        val seats = records.sortedBy { it.finishedAt }.allSeats()
        val mine = seats.filter { playerKey(it.name) == playerKey(playerName) }
        puzzles(mine.ifEmpty { seats })
    }
    if (list.isEmpty()) {
        Text(stringResource(R.string.puzzles_empty), Modifier.padding(24.dp), color = Palette.muted)
        return
    }
    var index by rememberSaveable { mutableIntStateOf(0) }
    val puzzle = list[index.coerceIn(0, list.lastIndex)]
    val language = languages.firstOrNull { it.code == puzzle.seat.game.language } ?: return
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            IconButton(onClick = { index-- }, enabled = index > 0) { Icon(Icons.Filled.ChevronLeft, stringResource(R.string.replay_prev)) }
            Text(stringResource(R.string.puzzle_n, index + 1, list.size), Modifier.weight(1f), fontWeight = FontWeight.SemiBold)
            IconButton(onClick = { index++ }, enabled = index < list.lastIndex) { Icon(Icons.Filled.ChevronRight, stringResource(R.string.replay_next)) }
        }
        Text(stringResource(R.string.puzzle_help), color = Palette.muted, fontSize = 13.sp)
        WithPack(language, loadPack) { pack -> PuzzleBoard(puzzle, language, pack, premiums) }
    }
}

/** One puzzle: the position before the turn, the rack, and a word to find on it. */
@Composable
private fun PuzzleBoard(puzzle: Puzzle, language: LanguageMeta, pack: LanguagePackHandle, premiums: List<Premium>) {
    val game = puzzle.seat.game
    val turn = puzzle.played
    val key = "${game.id}:${puzzle.turn}"
    val position = remember(key) { game.positions()!![puzzle.turn] }
    val blanks = remember(key) { position.cells.indices.map { game.blanks[it] && position.cells[it].isNotEmpty() } }
    val grid = remember(key) { Grid { r, c -> position.cells[r * 15 + c].firstOrNull() } }
    val rack = turn.rack
    var typing by remember(key) { mutableStateOf<Typing?>(null) }
    var horizontal by remember(key) { mutableStateOf(true) }
    var askBlank by remember(key) { mutableStateOf(false) }
    var checking by remember(key) { mutableStateOf(false) }
    var check by remember(key) { mutableStateOf<Pair<String, PlayCheck>?>(null) }
    var revealed by remember(key) { mutableStateOf<RecordedPlay?>(null) }
    var showAnswers by remember(key) { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun type(letter: Char) {
        val t = typing ?: return
        typing = t.type(letter, grid)
        check = null
    }

    val overlay = run {
        val shown = revealed
        val t = typing
        when {
            shown != null -> playOverlay(shown, position.cells)
            t != null -> {
                val allocation = rackAllocation(rack, t.typedLetters, language.blank)
                val placed = t.entries.filter { it.letter != null }.map { it.cell }
                BoardOverlay(
                    typed = t.placed,
                    typedBlanks = placed.filterIndexed { i, _ -> allocation.getOrNull(i)?.let { rack[it] == language.blank } == true }.toSet(),
                    typedThrough = t.passedThrough,
                    cursor = t.cursor,
                    typedValid = check?.second?.let { it.score != null },
                )
            }
            else -> BoardOverlay()
        }
    }
    Box(Modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        BoardView(
            cells = position.cells,
            blanks = blanks,
            owners = position.owners.map { it.toByte() },
            premiums = premiums,
            points = language.points,
            overlay = overlay,
            modifier = Modifier.widthIn(max = 480.dp).fillMaxWidth(),
            onTap = { cell ->
                revealed = null
                // Tapping the start square again turns the word round, as in a game.
                if (typing?.start == cell) horizontal = !horizontal
                typing = Typing(cell, horizontal)
                check = null
            },
        )
    }
    RackView(
        letters = rack,
        blank = language.blank,
        points = language.points,
        used = typing?.let { rackTilesUsed(rack, it.typedLetters, language.blank) } ?: emptySet(),
        selected = emptySet(),
        placing = typing != null,
        onTap = { i ->
            if (typing == null) return@RackView
            val ch = rack.getOrNull(i) ?: return@RackView
            if (ch == language.blank) askBlank = true else type(ch)
        },
    )
    if (typing == null) Text(stringResource(R.string.tap_start_square), color = Palette.muted, fontSize = 13.sp)
    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        IconButton(onClick = { typing = typing?.backspace(); check = null }, enabled = typing != null) {
            Icon(Icons.AutoMirrored.Filled.Backspace, stringResource(R.string.backspace))
        }
        val word = typing?.wordData(grid)
        Text(word?.word?.uppercase() ?: "—", Modifier.weight(1f), fontWeight = FontWeight.Bold, fontSize = 18.sp)
        Button(
            onClick = {
                val w = word ?: return@Button
                checking = true
                scope.launch {
                    val result = withContext(Dispatchers.Default) {
                        runCatching {
                            pack.checkPlay(position.cells, blanks, rack, w.word, w.row.toUInt(), w.col.toUInt(), w.horizontal)
                        }.getOrNull()
                    }
                    check = result?.let { w.word to it }
                    checking = false
                }
            },
            enabled = word != null && word.word.length >= 2 && !checking,
        ) {
            if (checking) EngineBusy(Modifier.size(18.dp)) else Text(stringResource(R.string.puzzle_check))
        }
    }
    check?.let { (word, c) -> CheckResult(word, c) }
    OutlinedButton(onClick = { showAnswers = !showAnswers }) {
        Text(stringResource(if (showAnswers) R.string.puzzle_hide_answer else R.string.puzzle_show_answer))
    }
    if (showAnswers) {
        Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
            Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(stringResource(R.string.puzzle_you_played, turn.word.uppercase(), turn.score), fontSize = 14.sp)
                val answers = listOfNotNull(
                    turn.best?.let { R.string.puzzle_top_score to it },
                    turn.equity?.play?.let { R.string.puzzle_strateg to it },
                ).distinctBy { it.second }
                answers.forEach { (label, play) ->
                    FilterChip(
                        selected = revealed == play,
                        onClick = { revealed = if (revealed == play) null else play; typing = null },
                        label = { Text(stringResource(label, play.word.uppercase(), play.score)) },
                    )
                }
                turn.equity?.exchange?.let { Text(stringResource(R.string.review_better_exchange, it.uppercase().replace("?", "★")), fontSize = 14.sp) }
            }
        }
    }
    if (askBlank) {
        BlankDialog(language.alphabet) { ch ->
            askBlank = false
            if (ch != null) type(ch)
        }
    }
}

@Composable
private fun CheckResult(word: String, c: PlayCheck) {
    val score = c.score
    Card(colors = CardDefaults.cardColors(containerColor = if (score == null) Palette.invalid else Palette.surface)) {
        Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            if (score == null) {
                Text(stringResource(R.string.puzzle_not_a_play, word.uppercase()), color = Palette.invalidText, fontWeight = FontWeight.SemiBold)
                return@Column
            }
            val top = score.toInt() >= c.bestScore.toInt()
            Text(
                stringResource(R.string.puzzle_scores, word.uppercase(), score.toInt()),
                fontWeight = FontWeight.Bold,
                color = if (top) Palette.validText else Palette.text,
            )
            Text(
                if (top) stringResource(R.string.puzzle_top)
                else stringResource(R.string.puzzle_better, c.bestScore.toInt(), c.better.toInt(), c.plays.toInt()),
                fontSize = 14.sp,
            )
            val played = c.equity
            val best = c.bestEquity
            if (played != null && best != null) {
                Text(
                    if (best - played < 0.05) stringResource(R.string.puzzle_equity_best, formatValue(played))
                    else stringResource(R.string.puzzle_equity, formatValue(played), formatValue(best)),
                    color = if (best - played < 0.05) Palette.validText else Palette.muted,
                    fontSize = 14.sp,
                )
            }
        }
    }
}

/** A play laid over the position it was there to be played on. */
private fun playOverlay(play: RecordedPlay, before: List<String>): BoardOverlay {
    val cells = play.word.indices.map { if (play.horizontal) Cell(play.row, play.col + it) else Cell(play.row + it, play.col) }
    val typed = cells.withIndex().filter { (_, c) -> before.getOrNull(c.index).isNullOrEmpty() }.associate { (i, c) -> c to play.word[i] }
    // `used` has one tile per newly covered square, in board order.
    val blanks = typed.keys.filterIndexed { i, _ -> play.used.getOrNull(i) == '?' }.toSet()
    return BoardOverlay(typed = typed, typedBlanks = blanks, typedThrough = cells.toSet() - typed.keys, typedValid = true)
}

// ── Anagramy ─────────────────────────────────────────────────────────────────

/** A drill: the rack, every word it makes, and those found so far. */
private data class Drill(val rack: String, val words: List<String>, val found: Set<String> = emptySet()) {
    val bingos get() = words.filter { it.length == DRILL_TILES }
}

/**
 * Seven tiles drawn from a full bag, blanks left out, until they make a word
 * of all seven -- a drill with no bingo in it has no point. Gives up after a
 * few hundred tries with whatever it drew last.
 */
private fun drawDrill(pack: LanguagePackHandle, random: Random = Random.Default): Drill {
    val blank = pack.blank()
    val bag = pack.tileCounts().filterKeys { it != blank }.flatMap { (letter, n) -> List(n.toInt()) { letter } }
    var rack = ""
    for (attempt in 0 until 400) {
        rack = bag.shuffled(random).take(DRILL_TILES).joinToString("")
        if (pack.search("-".repeat(DRILL_TILES), rack).isNotEmpty()) break
    }
    val words = pack.search("*", rack).filter { it.length >= 2 }.sortedWith(compareByDescending<String> { it.length }.thenBy { it })
    return Drill(rack, words)
}

@Composable
private fun Anagrams(language: LanguageMeta, pack: LanguagePackHandle) {
    var drill by remember(language.code) { mutableStateOf<Drill?>(null) }
    var order by remember { mutableStateOf("") }
    var guess by rememberSaveable { mutableStateOf("") }
    var message by remember { mutableStateOf<Pair<String, Boolean>?>(null) }
    var showAll by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun deal() {
        drill = null
        message = null
        showAll = false
        scope.launch {
            val d = withContext(Dispatchers.Default) { drawDrill(pack) }
            drill = d
            order = d.rack
        }
    }
    LaunchedEffect(language.code) { deal() }

    val d = drill
    if (d == null) {
        EngineBusy(Modifier.padding(24.dp).size(32.dp))
        return
    }
    val notWord = stringResource(R.string.drill_not_word)
    val notFromRack = stringResource(R.string.drill_not_from_rack)
    val again = stringResource(R.string.drill_already)
    val newWord = stringResource(R.string.drill_found)
    fun submit() {
        val w = guess.trim().lowercase()
        if (w.isEmpty()) return
        guess = ""
        message = when {
            !fromRack(w, d.rack) -> notFromRack.format(w.uppercase()) to false
            w in d.found -> again.format(w.uppercase()) to false
            w !in d.words -> notWord.format(w.uppercase()) to false
            else -> {
                drill = d.copy(found = d.found + w)
                newWord.format(w.uppercase(), w.length) to true
            }
        }
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text(stringResource(R.string.drill_help), color = Palette.muted, fontSize = 13.sp)
        Row(verticalAlignment = Alignment.CenterVertically) {
            RackView(
                letters = order,
                blank = language.blank,
                points = language.points,
                used = emptySet(),
                selected = emptySet(),
                placing = false,
                modifier = Modifier.weight(1f),
                onTap = {},
            )
            // Seeing the tiles in another order is half of finding a word in them.
            IconButton(onClick = { order = order.toList().shuffled().joinToString("") }) {
                Icon(Icons.Filled.Shuffle, stringResource(R.string.drill_shuffle))
            }
        }
        val bingos = d.bingos
        Text(
            stringResource(R.string.drill_progress, bingos.count { it in d.found }, bingos.size, d.found.size, d.words.size),
            fontWeight = FontWeight.SemiBold,
        )
        OutlinedTextField(
            value = guess,
            onValueChange = { guess = it.take(DRILL_TILES) },
            label = { Text(stringResource(R.string.drill_your_word)) },
            singleLine = true,
            keyboardOptions = KeyboardOptions(capitalization = KeyboardCapitalization.None, imeAction = ImeAction.Done),
            keyboardActions = KeyboardActions(onDone = { submit() }),
            modifier = Modifier.fillMaxWidth(),
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(onClick = ::submit, enabled = guess.isNotBlank()) { Text(stringResource(R.string.puzzle_check)) }
            OutlinedButton(onClick = { showAll = !showAll }) {
                Text(stringResource(if (showAll) R.string.drill_hide_all else R.string.drill_show_all))
            }
            OutlinedButton(onClick = ::deal) { Text(stringResource(R.string.drill_new)) }
        }
        message?.let { (text, ok) -> Text(text, color = if (ok) Palette.validText else Palette.danger) }
        if (d.found.isNotEmpty() || showAll) {
            SectionTitle(stringResource(if (showAll) R.string.drill_all_words else R.string.drill_found_words))
            WordsByLength(if (showAll) d.words else d.words.filter { it in d.found }, highlight = d.found)
        }
    }
}

/** Whether `word` can be laid from `rack`'s tiles, each used once. */
private fun fromRack(word: String, rack: String): Boolean {
    val left = rack.toMutableList()
    return word.all { left.remove(it) }
}

/** Words grouped under their length, longest first; `highlight`ed ones bold. */
@Composable
private fun WordsByLength(words: List<String>, highlight: Set<String> = emptySet()) {
    words.groupBy { it.length }.toSortedMap(compareByDescending { it }).forEach { (length, group) ->
        Column {
            Text(stringResource(R.string.words_of_length, length, group.size), color = Palette.muted, fontSize = 12.sp)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                group.forEach { w ->
                    Text(
                        w.uppercase(),
                        fontWeight = if (w in highlight) FontWeight.Bold else FontWeight.Normal,
                        color = if (highlight.isEmpty() || w in highlight) Palette.text else Palette.muted,
                    )
                }
            }
        }
    }
}

// ── Słownik ──────────────────────────────────────────────────────────────────

@Composable
private fun Words(language: LanguageMeta, pack: LanguagePackHandle, definitions: Definitions?) {
    var word by rememberSaveable { mutableStateOf("") }
    var verdict by remember { mutableStateOf<Pair<String, Boolean>?>(null) }
    var definition by remember { mutableStateOf<DefinitionUi?>(null) }
    var pattern by rememberSaveable { mutableStateOf("") }
    var letters by rememberSaveable { mutableStateOf("") }
    var results by remember { mutableStateOf<List<String>?>(null) }
    var searching by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun checkWord() {
        val w = word.trim().lowercase()
        if (w.isEmpty()) return
        val ok = pack.isWord(w)
        verdict = w to ok
        definition = null
        if (ok && definitions != null && language.definitions.isNotEmpty()) {
            definition = DefinitionUi(w, null)
            scope.launch {
                val found = definitions.lookup(w, language.definitions)
                if (definition?.word == w) definition = DefinitionUi(w, found)
            }
        }
    }

    val pool = letters.lowercase().filter { it == '?' || it in language.alphabet }
    val tooManyBlanks = pool.count { it == '?' } > SEARCH_BLANKS
    fun search() {
        val p = pattern.trim().lowercase().ifEmpty { "*" }
        searching = true
        scope.launch {
            results = withContext(Dispatchers.Default) {
                pack.search(p, pool).sortedWith(compareByDescending<String> { it.length }.thenBy { it })
            }
            searching = false
        }
    }

    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        SectionTitle(stringResource(R.string.words_check_title))
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedTextField(
                value = word,
                onValueChange = { word = it.take(15); verdict = null },
                label = { Text(stringResource(R.string.word)) },
                singleLine = true,
                keyboardOptions = KeyboardOptions(imeAction = ImeAction.Done),
                keyboardActions = KeyboardActions(onDone = { checkWord() }),
                modifier = Modifier.weight(1f),
            )
            Button(onClick = ::checkWord, enabled = word.isNotBlank()) { Text(stringResource(R.string.puzzle_check)) }
        }
        verdict?.let { (w, ok) ->
            Text(
                stringResource(if (ok) R.string.words_valid else R.string.words_invalid, w.uppercase()),
                color = if (ok) Palette.validText else Palette.invalidText,
                fontWeight = FontWeight.SemiBold,
            )
        }
        definition?.let { DefinitionCard(it, onClose = { definition = null }) }

        SectionTitle(stringResource(R.string.words_search_title))
        Text(stringResource(R.string.words_search_help), color = Palette.muted, fontSize = 13.sp)
        OutlinedTextField(
            value = letters,
            onValueChange = { letters = it.take(SEARCH_POOL); results = null },
            label = { Text(stringResource(R.string.words_letters)) },
            placeholder = { Text("aeinrst?") },
            singleLine = true,
            isError = tooManyBlanks,
            supportingText = if (tooManyBlanks) ({ Text(stringResource(R.string.words_too_many_blanks, SEARCH_BLANKS)) }) else null,
            modifier = Modifier.fillMaxWidth(),
        )
        OutlinedTextField(
            value = pattern,
            onValueChange = { pattern = it.take(15); results = null },
            label = { Text(stringResource(R.string.words_pattern)) },
            placeholder = { Text("*") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(imeAction = ImeAction.Search),
            keyboardActions = KeyboardActions(onSearch = { if (pool.isNotEmpty() && !tooManyBlanks) search() }),
            modifier = Modifier.fillMaxWidth(),
        )
        Button(onClick = ::search, enabled = pool.isNotEmpty() && !tooManyBlanks && !searching) {
            if (searching) EngineBusy(Modifier.size(18.dp)) else Text(stringResource(R.string.search))
        }
        results?.let { found ->
            Text(
                if (found.size > SEARCH_SHOWN) stringResource(R.string.words_found_many, found.size, SEARCH_SHOWN)
                else stringResource(R.string.words_found, found.size),
                fontWeight = FontWeight.SemiBold,
            )
            WordsByLength(found.take(SEARCH_SHOWN))
        }
    }
}
