package pl.scrablozaur.ui.scan

import android.app.Application
import android.util.Log
import androidx.compose.ui.geometry.Offset
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.scrablozaur.R
import pl.scrablozaur.ScrablozaurApplication
import pl.scrablozaur.core.KeptFrame
import pl.scrablozaur.core.LOG_TAG
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.ScannedBoard
import pl.scrablozaur.core.describe
import pl.scrablozaur.engine.BoardScanner
import pl.scrablozaur.engine.CameraFrame
import pl.scrablozaur.engine.FrameReading
import pl.scrablozaur.engine.Hint
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.engine.LanguagePackHandle
import pl.scrablozaur.engine.MoveException
import pl.scrablozaur.engine.ReadingVerdict
import pl.scrablozaur.engine.ScanException
import pl.scrablozaur.engine.Sighting
import pl.scrablozaur.ui.game.Cell
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicInteger

/**
 * The scan assistant's loop, the web's: the camera reads the board, the player
 * checks and confirms it, enters their letters, gets the best plays, places one on
 * the real board, and scans again. The confirmed board is kept across that loop
 * and across leaving the screen.
 */
enum class ScanPhase { Camera, Review, Assistant }

/** What the camera sees. */
data class CameraUi(
    val sighting: Sighting = Sighting.NO_BOARD,
    /** The grid's corners in the preview's pixels; empty when none shows. */
    val quad: List<Offset> = emptyList(),
    /** The latest reading of the board as it now lies: the status line, and what "fix by hand" starts from. */
    val reading: FrameReading? = null,
)

/** A board being checked: a reading, or the confirmed board, as the player corrects it. */
data class ReviewUi(
    val letters: List<String>,
    val blanks: List<Boolean>,
    val flagged: List<Boolean>,
    val corrected: List<Boolean>,
    /** The last confirmed board's tiles: already vouched for, so never flagged. */
    val locked: List<Boolean>,
    /** Per square, the reader's likeliest letters, best first. */
    val alternatives: List<List<String>>,
    /** Read off the camera, rather than the confirmed board opened for correcting. */
    val fromCamera: Boolean,
    val selected: Cell? = null,
    val saveExample: Boolean = false,
) {
    val flaggedCount get() = flagged.count { it }
}

data class AssistantUi(
    val rack: String = "",
    val order: HintOrder = HintOrder.SCORE,
    /** Null while the letters are being entered. */
    val suggestions: List<Hint>? = null,
    val selected: Hint? = null,
    val busy: Boolean = false,
)

data class ScanUi(
    val loading: Boolean = true,
    val loadError: String? = null,
    val language: LanguageMeta? = null,
    val points: Map<String, UInt> = emptyMap(),
    val phase: ScanPhase = ScanPhase.Camera,
    /** The last confirmed board; null before the first. */
    val board: ScannedBoard? = null,
    val camera: CameraUi = CameraUi(),
    val review: ReviewUi? = null,
    val assistant: AssistantUi = AssistantUi(),
    /** A line under the controls: a refusal, or where a training example went. */
    val message: String? = null,
    /** Counts the scans the camera finished by itself, for a tap of feedback. */
    val accepted: Int = 0,
) {
    val scannable get() = language?.hasScanner == true
}

class ScanViewModel(app: Application) : AndroidViewModel(app) {
    private val scrablozaur = app as ScrablozaurApplication
    private val repo = scrablozaur.engine
    private val sessions = scrablozaur.scans
    private var pack: LanguagePackHandle? = null
    private var scanner: BoardScanner? = null

    /** Camera frames are looked at one at a time, on a thread of their own. */
    val analysis: ExecutorService = Executors.newSingleThreadExecutor()

    /** Bumped whenever the camera starts watching afresh: a look begun before is stale. */
    private val watch = AtomicInteger()
    /** The frame of the latest reading, for a training example. */
    @Volatile private var kept: KeptFrame? = null
    /** Debug builds: recorded frames standing in for the camera, and which one is in view. */
    @Volatile private var replay: List<Pair<CameraFrame, Float>>? = null
    @Volatile private var replayed = 0
    /** Frames looked at since the last line in the log, how long they took, how many steady. */
    private var looked = 0
    private var lookMs = 0L
    private var steadyFrames = 0
    private var loggedAt = 0L

    private val _ui = MutableStateFlow(ScanUi())
    val ui: StateFlow<ScanUi> = _ui

    private fun str(id: Int, vararg args: Any) = getApplication<Application>().getString(id, *args)

    init {
        viewModelScope.launch {
            try {
                val saved = withContext(Dispatchers.IO) { sessions.load() }
                val code = saved?.language ?: repo.languages.firstOrNull { it.hasScanner }?.code ?: repo.defaultLanguage
                val meta = repo.language(code)
                val p = repo.pack(meta.code)
                val points = withContext(Dispatchers.Default) { p.letterPoints() }
                val s = if (meta.hasScanner) repo.scanner(meta.code) else null
                withContext(Dispatchers.Default) { s?.restart(saved?.letters, saved?.blanks) }
                pack = p
                scanner = s
                _ui.value = ScanUi(
                    loading = false,
                    language = meta,
                    points = points,
                    board = saved,
                    phase = if (saved != null) ScanPhase.Assistant else ScanPhase.Camera,
                )
            } catch (e: Exception) {
                if (e is kotlinx.coroutines.CancellationException) throw e
                Log.e(LOG_TAG, "scanner failed to load", e)
                _ui.value = ScanUi(loading = false, loadError = e.message ?: e.toString())
            }
        }
    }

    override fun onCleared() {
        analysis.shutdown()
        super.onCleared()
    }

    // ── Camera ───────────────────────────────────────────────────────────────

    /**
     * One camera frame, taken at `timeMs` through a lens of `focalPx` pixels'
     * focal length. `toView` maps frame pixels to the preview's. Called on
     * `analysis`; blocks while the engine reads the board.
     */
    fun onFrame(frame: CameraFrame, timeMs: Double, focalPx: Float, toView: (FloatArray) -> List<Offset>) {
        if (replay == null) look(frame, timeMs, focalPx, toView)
    }

    /**
     * Debug builds: show the camera the frames of a SCRBFRM1 file (`scrablozaur
     * frame-bench`'s) instead, one held still per scan -- the next one each time
     * the camera starts watching afresh. Tests everything past the lens on a
     * phone with no board in front of it.
     */
    fun replay(path: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val frames = try {
                readFrames(File(path))
            } catch (e: Exception) {
                Log.e(LOG_TAG, "no frames to replay in $path", e)
                return@launch
            }
            Log.i(LOG_TAG, "replaying ${frames.size} frames from $path")
            replay = frames
            withContext(analysis.asCoroutineDispatcher()) {
                while (true) {
                    if (_ui.value.phase == ScanPhase.Camera) {
                        val (frame, focal) = frames[replayed % frames.size]
                        look(frame, android.os.SystemClock.elapsedRealtime().toDouble(), focal) { emptyList() }
                    }
                    delay(100)
                }
            }
        }
    }

    private fun look(frame: CameraFrame, timeMs: Double, focalPx: Float, toView: (FloatArray) -> List<Offset>) {
        val s = scanner ?: return
        if (_ui.value.phase != ScanPhase.Camera) return
        val generation = watch.get()
        val start = android.os.SystemClock.elapsedRealtime()
        val look = try {
            s.look(frame, timeMs, focalPx)
        } catch (e: ScanException) {
            Log.w(LOG_TAG, "scanning a frame failed: $e")
            return
        }
        val now = android.os.SystemClock.elapsedRealtime()
        looked++
        lookMs += now - start
        if (look.sighting == Sighting.STEADY) steadyFrames++
        if (now - loggedAt >= 5000) {
            if (loggedAt != 0L) {
                Log.i(
                    LOG_TAG,
                    "camera: %.1f frames/s, %d ms a frame, steady in %d of %d".format(
                        looked * 1000f / (now - loggedAt), lookMs / looked, steadyFrames, looked,
                    ),
                )
            }
            looked = 0
            lookMs = 0
            steadyFrames = 0
            loggedAt = now
        }
        if (look.reading != null) {
            val board = look.reading!!.board
            Log.i(
                LOG_TAG,
                "read a ${frame.width}x${frame.height} frame in ${android.os.SystemClock.elapsedRealtime() - start} ms: " +
                    "${board.letters.count { it.isNotEmpty() }} tiles, ${look.reading!!.verdict}",
            )
        }
        if (watch.get() != generation) return
        val quad = if (look.corners.isEmpty()) emptyList() else toView(look.corners.toFloatArray())
        val reading = look.reading
        if (reading != null) {
            kept = KeptFrame(frame.rgba.copyOf(), frame.width.toInt(), frame.height.toInt(), frame.rowStride.toInt(), look.corners.toFloatArray(), focalPx)
        }
        _ui.update { ui ->
            if (ui.phase != ScanPhase.Camera) return@update ui
            val camera = CameraUi(
                sighting = look.sighting,
                quad = quad,
                // An earlier reading still describes a board that has not moved.
                reading = reading ?: ui.camera.reading.takeIf { look.sighting == Sighting.SETTLING },
            )
            if (reading?.verdict == ReadingVerdict.ACCEPTED) {
                ui.copy(phase = ScanPhase.Review, camera = camera, review = reviewOf(reading, ui.board), message = null, accepted = ui.accepted + 1)
            } else {
                ui.copy(camera = camera)
            }
        }
    }

    /** "Popraw ręcznie": the latest reading into review, whatever the dictionary says of it. */
    fun fixByHand() {
        _ui.update { ui ->
            val reading = ui.camera.reading?.takeIf { it.verdict != ReadingVerdict.UNSEEN } ?: return@update ui
            ui.copy(phase = ScanPhase.Review, review = reviewOf(reading, ui.board), message = null)
        }
    }

    /**
     * "Pusta plansza": a game's first move has no board to scan -- the camera
     * never takes an empty one, a wall reads as empty too -- so the assistant
     * starts from an empty board, as the web does from a photo of one.
     */
    fun startEmpty() {
        val language = _ui.value.language ?: return
        val board = ScannedBoard(language.code, List(225) { "" }, List(225) { false })
        watch.incrementAndGet()
        _ui.update {
            it.copy(
                phase = ScanPhase.Assistant, board = board, review = null, camera = CameraUi(),
                assistant = it.assistant.copy(suggestions = null, selected = null), message = null,
            )
        }
        viewModelScope.launch(Dispatchers.IO) { sessions.save(board) }
    }

    private fun reviewOf(reading: FrameReading, prior: ScannedBoard?): ReviewUi {
        val b = reading.board
        return ReviewUi(
            letters = b.letters,
            blanks = b.blanks,
            flagged = b.flagged,
            corrected = b.corrected,
            locked = List(225) { prior?.letters?.get(it)?.isNotEmpty() == true },
            alternatives = reading.alternatives,
            fromCamera = true,
        )
    }

    /** Back to the camera, watching afresh for a board holding the confirmed one's tiles. */
    fun scanAgain() {
        watch.incrementAndGet()
        replayed++
        _ui.update { it.copy(phase = ScanPhase.Camera, camera = CameraUi(), review = null, message = null) }
        val board = _ui.value.board
        viewModelScope.launch(Dispatchers.Default) {
            try {
                scanner?.restart(board?.letters, board?.blanks)
            } catch (e: ScanException) {
                Log.w(LOG_TAG, "restarting the scanner failed: $e")
            }
            watch.incrementAndGet()
        }
    }

    /** From the camera back to the confirmed board, if there is one. */
    fun backToAssistant() {
        if (_ui.value.board == null) return
        watch.incrementAndGet()
        _ui.update { it.copy(phase = ScanPhase.Assistant, camera = CameraUi(), review = null) }
    }

    // ── Review ───────────────────────────────────────────────────────────────

    /** The confirmed board opened for correcting, without scanning again. */
    fun correctBoard() {
        _ui.update { ui ->
            val b = ui.board ?: return@update ui
            ui.copy(
                phase = ScanPhase.Review,
                review = ReviewUi(
                    letters = b.letters, blanks = b.blanks, flagged = List(225) { false }, corrected = List(225) { false },
                    locked = List(225) { false }, alternatives = List(225) { emptyList() }, fromCamera = false,
                ),
                message = null,
            )
        }
    }

    fun cancelReview() {
        if (_ui.value.review?.fromCamera == true) scanAgain() else _ui.update { it.copy(phase = ScanPhase.Assistant, review = null) }
    }

    fun selectCell(cell: Cell) {
        _ui.update { ui -> ui.review?.let { ui.copy(review = it.copy(selected = if (it.selected == cell) null else cell)) } ?: ui }
    }

    /** A letter on the selected square; the blank key makes its tile a blank, or a letter again. */
    fun typeLetter(letter: Char) {
        val review = _ui.value.review ?: return
        val cell = review.selected ?: return
        val i = cell.row * 15 + cell.col
        val blank = _ui.value.language?.blank
        if (letter == blank) {
            if (review.letters[i].isEmpty()) return
            edit(review.copy(blanks = review.blanks.with(i, !review.blanks[i]), locked = review.locked.with(i, false)))
            return
        }
        // As on the web, typing moves on to the next square.
        val next = if (cell.col < 14) Cell(cell.row, cell.col + 1) else cell
        edit(
            review.copy(
                letters = review.letters.with(i, letter.lowercase()),
                flagged = review.flagged.with(i, false),
                locked = review.locked.with(i, false),
                selected = next,
            ),
        )
    }

    /** Empty the selected square, or step back when it is empty already. */
    fun backspace() {
        val review = _ui.value.review ?: return
        val cell = review.selected ?: return
        val i = cell.row * 15 + cell.col
        if (review.letters[i].isEmpty()) {
            if (cell.col > 0) _ui.update { it.copy(review = review.copy(selected = Cell(cell.row, cell.col - 1))) }
            return
        }
        clear(review, i)
    }

    fun clearCell() {
        val review = _ui.value.review ?: return
        val cell = review.selected ?: return
        clear(review, cell.row * 15 + cell.col)
    }

    private fun clear(review: ReviewUi, i: Int) = edit(
        review.copy(
            letters = review.letters.with(i, ""),
            blanks = review.blanks.with(i, false),
            flagged = review.flagged.with(i, false),
            locked = review.locked.with(i, false),
        ),
    )

    /** Apply an edit, then re-check every word it touched -- the web's /scan/recheck. */
    private fun edit(review: ReviewUi) {
        _ui.update { it.copy(review = review, message = null) }
        val p = pack ?: return
        viewModelScope.launch(Dispatchers.Default) {
            val flagged = try {
                p.flagBoard(review.letters, review.locked)
            } catch (e: ScanException) {
                Log.w(LOG_TAG, "rechecking the board failed: $e")
                return@launch
            }
            _ui.update { ui ->
                val now = ui.review
                if (now?.letters == review.letters) ui.copy(review = now.copy(flagged = flagged)) else ui
            }
        }
    }

    fun setSaveExample(on: Boolean) {
        _ui.update { ui -> ui.review?.let { ui.copy(review = it.copy(saveExample = on)) } ?: ui }
    }

    fun confirm() {
        val ui = _ui.value
        val review = ui.review ?: return
        val language = ui.language ?: return
        val board = ScannedBoard(
            language = language.code,
            letters = review.letters,
            blanks = review.letters.indices.map { review.blanks[it] && review.letters[it].isNotEmpty() },
        )
        val frame = kept.takeIf { review.saveExample && review.fromCamera }
        _ui.update {
            it.copy(
                phase = ScanPhase.Assistant, board = board, review = null, camera = CameraUi(),
                assistant = it.assistant.copy(suggestions = null, selected = null), message = null,
            )
        }
        viewModelScope.launch(Dispatchers.IO) {
            sessions.save(board)
            if (frame != null) {
                val message = try {
                    str(R.string.scan_example_saved, sessions.saveExample(frame, board))
                } catch (e: Exception) {
                    str(R.string.scan_example_failed, e.message ?: e.toString())
                }
                _ui.update { it.copy(message = message) }
            }
        }
    }

    /** "Nowa sesja": forget the confirmed board -- a new game on the real board. */
    fun newSession() {
        viewModelScope.launch(Dispatchers.IO) { sessions.clear() }
        _ui.update { it.copy(board = null, assistant = AssistantUi(order = it.assistant.order), message = null) }
        scanAgain()
    }

    // ── Assistant ────────────────────────────────────────────────────────────

    fun setRack(letters: String) {
        _ui.update { it.copy(assistant = it.assistant.copy(rack = letters.take(7)), message = null) }
    }

    fun editRack() {
        _ui.update { it.copy(assistant = it.assistant.copy(suggestions = null, selected = null)) }
    }

    fun setOrder(order: HintOrder) {
        _ui.update { it.copy(assistant = it.assistant.copy(order = order)) }
        if (_ui.value.assistant.suggestions != null) search()
    }

    fun search() {
        val ui = _ui.value
        val board = ui.board ?: return
        val p = pack ?: return
        val assistant = ui.assistant
        val rack = assistant.rack.trim().lowercase()
        if (rack.isEmpty() || assistant.busy) return
        _ui.update { it.copy(assistant = it.assistant.copy(busy = true, selected = null), message = null) }
        viewModelScope.launch {
            try {
                val list = withContext(Dispatchers.Default) { p.suggestOnBoard(board.letters, board.blanks, rack, assistant.order) }
                _ui.update { it.copy(assistant = it.assistant.copy(busy = false, suggestions = list)) }
            } catch (e: MoveException) {
                // The board came from the player's own hands: an edit that put more of
                // a letter on it than one game holds is the one way it can be refused.
                val message = if (e is MoveException.Engine) str(R.string.scan_bad_board) else e.describe(getApplication())
                _ui.update { it.copy(assistant = it.assistant.copy(busy = false), message = message) }
            }
        }
    }

    fun selectHint(hint: Hint) {
        _ui.update { it.copy(assistant = it.assistant.copy(selected = if (it.assistant.selected == hint) null else hint)) }
    }

    fun dismissMessage() {
        _ui.update { it.copy(message = null) }
    }
}

private fun <T> List<T>.with(i: Int, value: T): List<T> = toMutableList().also { it[i] = value }

/** The frames of a SCRBFRM1 file, as RGBA camera frames with their focal lengths. */
private fun readFrames(file: File): List<Pair<CameraFrame, Float>> {
    val data = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    val magic = ByteArray(8).also { data.get(it) }
    check(String(magic, Charsets.US_ASCII) == "SCRBFRM1") { "not a SCRBFRM1 file" }
    return List(data.getInt()) {
        val (w, h, focal) = Triple(data.getInt(), data.getInt(), data.getFloat())
        val rgba = ByteArray(w * h * 4)
        for (px in 0 until w * h) {
            data.get(rgba, px * 4, 3)
            rgba[px * 4 + 3] = -1
        }
        CameraFrame(rgba, w.toUInt(), h.toUInt(), (w * 4).toUInt()) to focal
    }
}
