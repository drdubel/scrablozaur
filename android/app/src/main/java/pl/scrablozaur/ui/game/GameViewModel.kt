package pl.scrablozaur.ui.game

import android.app.Application
import android.os.SystemClock
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.job
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.scrablozaur.R
import pl.scrablozaur.ScrablozaurApplication
import pl.scrablozaur.core.AutoLogEntry
import pl.scrablozaur.core.DefinitionResult
import pl.scrablozaur.core.LOG_TAG
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.RatedMove
import pl.scrablozaur.core.SavedGame
import pl.scrablozaur.core.describe
import pl.scrablozaur.engine.Game
import pl.scrablozaur.engine.GameView
import pl.scrablozaur.engine.Hint
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.MoveException
import pl.scrablozaur.engine.PlayerSetup
import pl.scrablozaur.engine.PreviewResult

/** A new game as the setup screen describes it. */
data class NewGame(val language: String, val mode: Mode, val players: List<PlayerSetup>)

data class DefinitionUi(val word: String, val result: DefinitionResult?)

/** Which control is waiting on the engine, so only it shows progress. */
enum class Busy { Move, Hints, Suggest, Exchange, Undo, Auto }

data class GameUi(
    val loading: Boolean = true,
    val loadError: String? = null,
    val language: LanguageMeta? = null,
    val view: GameView? = null,
    val points: Map<String, UInt> = emptyMap(),
    val typing: Typing? = null,
    val horizontal: Boolean = true,
    val typedWord: WordData? = null,
    val preview: PreviewResult? = null,
    val exchangeSelection: Set<Int> = emptySet(),
    /** Competitive hints; null while the list is closed. */
    val hints: List<Hint>? = null,
    val hintOrder: HintOrder = HintOrder.SCORE,
    val selectedHint: Hint? = null,
    /** Sandbox suggestions for the computer seat; null while its letters are being entered. */
    val suggestions: List<Hint>? = null,
    val suggestionOrder: HintOrder = HintOrder.SCORE,
    val computerLetters: String = "",
    val askBlank: Boolean = false,
    val error: String? = null,
    val busy: Busy? = null,
    val thinking: Boolean = false,
    val ratings: List<RatedMove> = emptyList(),
    val autoLog: List<AutoLogEntry> = emptyList(),
    val autoplay: Boolean = false,
    val definition: DefinitionUi? = null,
) {
    val current get() = view?.players?.getOrNull(view.currentPlayer.toInt())

    /** Whether a human can type a word now. */
    val humanTurn get() = view != null && !view.gameOver && !view.computerToMove &&
        view.mode != Mode.SANDBOX_AUTO && current?.isComputer == false

    /** Whether an action may start: nothing is waiting on the engine. */
    val idle get() = busy == null && !thinking

    val grid: Grid get() = Grid { r, c -> view?.cells?.getOrNull(r * 15 + c)?.firstOrNull() }
}

@OptIn(ExperimentalCoroutinesApi::class)
class GameViewModel(app: Application) : AndroidViewModel(app) {
    private val scrablozaur = app as ScrablozaurApplication
    private val repo = scrablozaur.engine
    private val saves = scrablozaur.saves
    /** One game is never touched by two threads at once. */
    private val engine = Dispatchers.Default.limitedParallelism(1)

    private var game: Game? = null
    /**
     * Everything tied to the current game runs here, and is cancelled when
     * another game replaces it -- so a reply still being computed for the old
     * game cannot land on the new one.
     */
    private var session = newSession()
    private var previewJob: Job? = null
    private var autoplayJob: Job? = null

    private val _ui = MutableStateFlow(GameUi())
    val ui: StateFlow<GameUi> = _ui

    private fun newSession() = CoroutineScope(
        viewModelScope.coroutineContext + SupervisorJob(viewModelScope.coroutineContext.job),
    )

    private fun str(id: Int, vararg args: Any) = getApplication<Application>().getString(id, *args)

    // ── Lifecycle ────────────────────────────────────────────────────────────

    fun start(config: NewGame) = launchLoading {
        val pack = repo.pack(config.language)
        val g = withContext(engine) { Game(pack, config.mode, config.players, 0uL) }
        install(g, config.language, emptyList(), emptyList())
    }

    /** Load the saved game, unless this one is already showing it. */
    fun resumeIfNeeded() {
        if (game != null) return
        launchLoading {
            val saved = withContext(Dispatchers.IO) { saves.load() } ?: error("no saved game")
            val pack = repo.pack(saved.language)
            val g = withContext(engine) { Game.restore(pack, saved.session) }
            install(g, saved.language, saved.ratings, saved.autoLog)
        }
    }

    val hasGame get() = game != null

    private fun launchLoading(block: suspend () -> Unit) {
        session.cancel()
        session = newSession()
        previewJob = null
        autoplayJob = null
        _ui.value = GameUi(loading = true)
        session.launch {
            try {
                block()
            } catch (e: MoveException) {
                _ui.update { it.copy(loading = false, loadError = e.describe(getApplication())) }
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                _ui.update { it.copy(loading = false, loadError = e.message ?: e.toString()) }
            }
        }
    }

    private suspend fun install(g: Game, language: String, ratings: List<RatedMove>, log: List<AutoLogEntry>) {
        game?.close()
        game = g
        val pack = repo.pack(language)
        val points = withContext(engine) { pack.letterPoints() }
        val view = withContext(engine) { g.state() }
        _ui.value = GameUi(
            loading = false,
            language = repo.language(language),
            view = view,
            points = points,
            ratings = ratings,
            autoLog = log,
        ).withComputerLetters()
        persist(g)
        playComputerIfDue(g)
    }

    override fun onCleared() {
        session.cancel()
        game?.close()
        super.onCleared()
    }

    private suspend fun persist(g: Game) {
        val state = _ui.value
        val language = state.language?.code ?: return
        val session = withContext(engine) { g.save() }
        if (game !== g) return
        val over = state.view?.gameOver == true
        withContext(Dispatchers.IO) { saves.save(SavedGame(language, session, over, state.ratings, state.autoLog)) }
    }

    /** The web pre-fills the sandbox computer seat with the letters it kept. */
    private fun GameUi.withComputerLetters(): GameUi {
        val seat = current ?: return this
        return if (view?.mode == Mode.SANDBOX && seat.isComputer && seat.letters.isNotEmpty()) {
            copy(computerLetters = seat.letters)
        } else {
            this
        }
    }

    /** Re-read the position after an action; drops anything tied to the old one. */
    private suspend fun refresh(g: Game, keep: (GameUi) -> GameUi = { it }) {
        val view = withContext(engine) { g.state() }
        // Replaced by another game while this one was computing: stale.
        if (game !== g) return
        previewJob?.cancel()
        _ui.update {
            keep(
                it.copy(
                    view = view, typing = null, typedWord = null, preview = null,
                    exchangeSelection = emptySet(), hints = null, selectedHint = null,
                    suggestions = null, askBlank = false, busy = null,
                ),
            ).withComputerLetters()
        }
        persist(g)
    }

    /** Run an engine action; a refusal becomes the panel's error line. */
    private fun act(busy: Busy?, block: suspend (Game) -> Unit) {
        val g = game ?: return
        if (!_ui.value.idle) return
        _ui.update { it.copy(busy = busy, error = null) }
        session.launch {
            try {
                block(g)
            } catch (e: MoveException) {
                _ui.update { it.copy(busy = null, error = e.describe(getApplication())) }
            }
        }
    }

    /** The web plays a competitive computer's reply inside the human's request;
     *  here it is its own step, so the human's move is on screen first. */
    private suspend fun playComputerIfDue(g: Game) {
        while (game === g && _ui.value.view?.computerToMove == true) {
            _ui.update { it.copy(thinking = true) }
            try {
                val start = SystemClock.elapsedRealtime()
                val move = withContext(engine) { g.computerTurn() }
                Log.i(LOG_TAG, "computer played '${move.word}' after ${SystemClock.elapsedRealtime() - start} ms")
            } catch (e: MoveException) {
                _ui.update { it.copy(thinking = false, error = e.describe(getApplication())) }
                return
            }
            refresh(g) { it.copy(thinking = false) }
        }
    }

    // ── Typing a word ────────────────────────────────────────────────────────

    fun onCellTap(cell: Cell) {
        val state = _ui.value
        val occupied = state.grid.at(cell.row, cell.col) != null
        if (occupied) lookupDefinition(cell)
        if (!state.humanTurn || state.busy != null) return
        // Re-tapping the start square flips the direction, as on the web.
        val horizontal = if (state.typing?.start == cell) !state.horizontal else state.horizontal
        _ui.update {
            it.copy(
                hints = null,
                selectedHint = null,
                definition = if (occupied) it.definition else null,
            )
        }
        setTyping(Typing(cell, horizontal), horizontal, clearSelection = true)
    }

    /** Flip the direction; a word in progress restarts from its square. */
    fun toggleDirection() {
        val horizontal = !_ui.value.horizontal
        val t = _ui.value.typing
        if (t == null) {
            _ui.update { it.copy(horizontal = horizontal) }
        } else {
            setTyping(Typing(t.start, horizontal), horizontal)
        }
    }

    fun typeLetter(letter: Char) {
        val state = _ui.value
        val t = state.typing ?: return
        setTyping(t.type(letter, state.grid), state.horizontal)
    }

    fun backspace() {
        val t = _ui.value.typing ?: return
        setTyping(t.backspace(), _ui.value.horizontal)
    }

    fun cancelTyping() {
        previewJob?.cancel()
        _ui.update { it.copy(typing = null, typedWord = null, preview = null, askBlank = false, error = null) }
    }

    private fun setTyping(typing: Typing, horizontal: Boolean, clearSelection: Boolean = false) {
        val state = _ui.value
        val word = typing.wordData(state.grid)
        _ui.update {
            it.copy(
                typing = typing,
                horizontal = horizontal,
                typedWord = word,
                preview = null,
                error = null,
                exchangeSelection = if (clearSelection) emptySet() else it.exchangeSelection,
            )
        }
        previewJob?.cancel()
        val g = game ?: return
        if (word == null || word.word.length < 2) return
        previewJob = session.launch {
            delay(100)
            val result = withContext(engine) {
                g.preview(word.word, word.row.toUInt(), word.col.toUInt(), word.horizontal)
            }
            _ui.update { if (it.typedWord == word) it.copy(preview = result) else it }
        }
    }

    fun submitTyped() {
        val word = _ui.value.typedWord
        if (word == null) {
            _ui.update { it.copy(error = str(R.string.tap_start_square)) }
            return
        }
        playWord(word.word, word.row, word.col, word.horizontal)
    }

    fun placeHint(hint: Hint) = playWord(hint.word, hint.row.toInt(), hint.col.toInt(), hint.horizontal)

    private fun playWord(word: String, row: Int, col: Int, horizontal: Boolean) = act(Busy.Move) { g ->
        val score = withContext(engine) { g.humanMove(word, row.toUInt(), col.toUInt(), horizontal) }
        val rating = withContext(engine) { g.state() }.lastMoveRating
        refresh(g) {
            if (rating == null) it else it.copy(ratings = listOf(RatedMove(word, score.toInt(), rating.toInt())) + it.ratings)
        }
        playComputerIfDue(g)
    }

    // ── The rack ─────────────────────────────────────────────────────────────

    fun onRackTap(index: Int) {
        val state = _ui.value
        val letters = state.current?.letters ?: return
        if (!state.humanTurn) return
        if (state.typing != null) {
            // With a square selected the rack is a keyboard.
            val ch = letters.getOrNull(index) ?: return
            if (ch == state.language?.blank) _ui.update { it.copy(askBlank = true) } else typeLetter(ch)
            return
        }
        _ui.update {
            val sel = it.exchangeSelection
            it.copy(exchangeSelection = if (index in sel) sel - index else sel + index, error = null)
        }
    }

    fun onBlankChosen(letter: Char?) {
        _ui.update { it.copy(askBlank = false) }
        if (letter != null) typeLetter(letter)
    }

    fun exchange() {
        val state = _ui.value
        val letters = state.current?.letters ?: return
        if (state.exchangeSelection.isEmpty()) {
            // Picking tiles needs the rack back from typing duty.
            cancelTyping()
            _ui.update { it.copy(error = str(R.string.select_exchange_tiles)) }
            return
        }
        val picked = state.exchangeSelection.sorted().map { letters[it] }.joinToString("")
        act(Busy.Exchange) { g ->
            withContext(engine) { g.exchange(picked) }
            refresh(g)
            playComputerIfDue(g)
        }
    }

    fun skip() = act(Busy.Move) { g ->
        withContext(engine) { g.skip() }
        refresh(g)
        playComputerIfDue(g)
    }

    fun resign() = act(Busy.Move) { g ->
        withContext(engine) { g.resign() }
        refresh(g)
    }

    fun undo() {
        val running = autoplayJob
        stopAutoplay()
        session.launch {
            // An automatic move still in flight lands first, as on the web, so
            // undo takes back the move the board shows rather than one it never did.
            running?.join()
            act(Busy.Undo) { g ->
                withContext(engine) { g.undo() }
                refresh(g) { s -> if (s.view?.mode == Mode.SANDBOX_AUTO) s.copy(autoLog = s.autoLog.drop(1)) else s }
            }
        }
    }

    // ── Hints (competitive) ──────────────────────────────────────────────────

    fun toggleHints() {
        if (_ui.value.hints != null) {
            _ui.update { it.copy(hints = null, selectedHint = null) }
        } else {
            cancelTyping()
            loadHints()
        }
    }

    fun setHintOrder(order: HintOrder) {
        _ui.update { it.copy(hintOrder = order) }
        if (_ui.value.hints != null) loadHints()
    }

    private fun loadHints() = act(Busy.Hints) { g ->
        val list = withContext(engine) { g.hints(_ui.value.hintOrder) }
        _ui.update { it.copy(hints = list, selectedHint = null, busy = null) }
    }

    /** Preview a hint or suggestion on the board; tapping it again clears it. */
    fun selectHint(hint: Hint) {
        _ui.update {
            it.copy(selectedHint = if (it.selectedHint == hint) null else hint, typing = null, typedWord = null, preview = null)
        }
    }

    // ── The computer seat (sandbox) ──────────────────────────────────────────

    fun setComputerLetters(letters: String) {
        _ui.update { it.copy(computerLetters = letters.take(7), error = null) }
    }

    /** Back from the suggestion list to entering the letters. */
    fun editComputerLetters() {
        _ui.update { it.copy(suggestions = null, selectedHint = null) }
    }

    fun setSuggestionOrder(order: HintOrder) {
        _ui.update { it.copy(suggestionOrder = order) }
        if (_ui.value.suggestions != null) searchSuggestions()
    }

    fun searchSuggestions() {
        val letters = _ui.value.computerLetters.trim().lowercase()
        if (letters.isEmpty()) return
        act(Busy.Suggest) { g ->
            val list = withContext(engine) {
                g.setLetters(letters)
                g.sandboxSuggestions(_ui.value.suggestionOrder)
            }
            // The seat's rack is part of the game now, as on the web.
            persist(g)
            _ui.update {
                it.copy(
                    suggestions = list.takeIf { l -> l.isNotEmpty() },
                    selectedHint = null,
                    busy = null,
                    error = if (list.isEmpty()) str(R.string.no_moves_for_letters) else null,
                )
            }
        }
    }

    fun placeSuggestion(hint: Hint) = act(Busy.Move) { g ->
        withContext(engine) { g.placeComputerWord(hint.word, hint.row, hint.col, hint.horizontal) }
        refresh(g) { it.copy(computerLetters = "") }
    }

    // ── All-computer games ───────────────────────────────────────────────────

    fun nextAutoMove() = act(Busy.Auto) { g -> stepAuto(g) }

    /** One automatic move, shown once made: stopping autoplay mid-move lets it land. */
    private suspend fun stepAuto(g: Game) = withContext(NonCancellable) {
        val before = _ui.value.view ?: return@withContext
        val mover = before.players[before.currentPlayer.toInt()]
        val move = withContext(engine) { g.nextAutoMove() }
        val entry = AutoLogEntry(mover.name, move.word, move.score.toInt(), move.passed)
        refresh(g) { it.copy(autoLog = listOf(entry) + it.autoLog) }
    }

    fun toggleAutoplay() {
        if (autoplayJob != null) {
            stopAutoplay()
            return
        }
        val g = game ?: return
        if (!_ui.value.idle) return
        _ui.update { it.copy(autoplay = true, error = null) }
        autoplayJob = session.launch {
            try {
                while (_ui.value.view?.gameOver == false) {
                    _ui.update { it.copy(busy = Busy.Auto) }
                    stepAuto(g)
                    delay(550)
                }
            } catch (e: MoveException) {
                _ui.update { it.copy(error = e.describe(getApplication())) }
            } finally {
                autoplayJob = null
                _ui.update { it.copy(autoplay = false, busy = null) }
            }
        }
    }

    private fun stopAutoplay() {
        autoplayJob?.cancel()
        autoplayJob = null
        _ui.update { it.copy(autoplay = false) }
    }

    // ── Definitions ──────────────────────────────────────────────────────────

    fun lookupDefinition(cell: Cell) {
        val word = wordThrough(_ui.value.grid, cell) ?: return
        val providers = _ui.value.language?.definitions ?: return
        _ui.update { it.copy(definition = DefinitionUi(word, null)) }
        viewModelScope.launch {
            val result = scrablozaur.definitions.lookup(word, providers)
            _ui.update { if (it.definition?.word == word) it.copy(definition = DefinitionUi(word, result)) else it }
        }
    }

    fun closeDefinition() {
        _ui.update { it.copy(definition = null) }
    }

    fun dismissError() {
        _ui.update { it.copy(error = null) }
    }

    /** Drop the saved game: the setup screen is about to start another. */
    fun forgetGame() {
        stopAutoplay()
        saves.clear()
    }
}

/**
 * The word a tile belongs to, for a definition: the longer of the across and
 * down words through it, as the web picks it. Null for a lone tile.
 */
fun wordThrough(grid: Grid, cell: Cell): String? {
    if (grid.at(cell.row, cell.col) == null) return null
    fun scan(dr: Int, dc: Int): String? {
        var r = cell.row
        var c = cell.col
        while (r - dr >= 0 && c - dc >= 0 && grid.at(r - dr, c - dc) != null) {
            r -= dr
            c -= dc
        }
        val sb = StringBuilder()
        while (r < Cell.SIZE && c < Cell.SIZE) {
            sb.append(grid.at(r, c) ?: break)
            r += dr
            c += dc
        }
        return sb.toString().takeIf { it.length > 1 }
    }
    val h = scan(0, 1)
    val v = scan(1, 0)
    return if (h != null && v != null) (if (h.length >= v.length) h else v) else h ?: v
}
