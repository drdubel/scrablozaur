package pl.scrablozaur.core

import android.util.Log
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.first
import org.json.JSONArray
import org.json.JSONObject
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.TurnKind
import java.io.File

/** A seat as the game ended. */
data class RecordedPlayer(
    val name: String,
    val isComputer: Boolean,
    val difficulty: Int,
    val strength: Int,
    val score: Int,
)

/** A play a rack had: the engine's `BestPlay`. */
data class RecordedPlay(
    val word: String,
    val score: Int,
    val row: Int,
    val col: Int,
    val horizontal: Boolean,
    /** One tile per new square, in board order; `?` for a blank. */
    val used: String,
)

/** A turn by equity -- its score plus the leave net's value of the tiles kept -- against the best there was (`EquityReview`). */
data class RecordedEquity(
    val played: Double,
    /** Never below `played`. */
    val best: Double,
    /** The better turn, when there was one: a play, or the tiles to exchange. */
    val play: RecordedPlay? = null,
    val exchange: String? = null,
) {
    /** What the turn gave away, to the tenth the engine rounds to. */
    val loss: Double get() = (best - played).coerceAtLeast(0.0)
}

/** A turn with the bag empty, by the endgame search (`EndgameReview`): final margins, racks included. */
data class RecordedEndgame(
    val played: Int,
    val best: Int,
    /** Both searches finished: the numbers are proved, not estimated. */
    val exact: Boolean,
    /** The play leading to `best`; null when passing did. */
    val play: RecordedPlay? = null,
) {
    val loss: Int get() = (best - played).coerceAtLeast(0)
}

/** One turn, as the engine logged it (`Turn`). */
data class RecordedTurn(
    val player: Int,
    val kind: TurnKind,
    val word: String = "",
    val score: Int = 0,
    val row: Int = 0,
    val col: Int = 0,
    val horizontal: Boolean = true,
    val tiles: Int = 0,
    val rack: String = "",
    val rating: Int? = null,
    val best: RecordedPlay? = null,
    val equity: RecordedEquity? = null,
    val endgame: RecordedEndgame? = null,
)

/**
 * A finished game: what the statistics, the ranking and the game's review are
 * made from. Competitive and multiplayer games -- the modes played for real.
 */
data class GameRecord(
    val id: String,
    val mode: Mode,
    val language: String,
    val startedAt: Long,
    val finishedAt: Long,
    val players: List<RecordedPlayer>,
    /** The seat the phone's player had: in a multiplayer game, the one under their name, else the first person's. */
    val human: Int,
    /** The engine's outcome: the seat with the most points, -1 for a tie. */
    val winner: Int,
    val turns: List<RecordedTurn>,
    /** The final position, 225 squares row-major: a letter or "". */
    val cells: List<String>,
    val blanks: List<Boolean>,
    /** The seat that placed each tile, -1 for none. */
    val owners: List<Int>,
    /** Turns on which the player opened the hints. */
    val hintTurns: Int,
    val undos: Int,
)

private const val RECORD_VERSION = 1

private fun RecordedPlay.toJson(): JSONObject =
    JSONObject().put("word", word).put("score", score).put("row", row).put("col", col).put("h", horizontal).put("used", used)

private fun recordedPlayOf(b: JSONObject) =
    RecordedPlay(b.getString("word"), b.getInt("score"), b.getInt("row"), b.getInt("col"), b.getBoolean("h"), b.getString("used"))

fun GameRecord.toJson(): JSONObject = JSONObject()
    .put("v", RECORD_VERSION)
    .put("id", id)
    .put("mode", mode.name)
    .put("language", language)
    .put("started", startedAt)
    .put("finished", finishedAt)
    .put("players", JSONArray(players.map {
        JSONObject().put("name", it.name).put("computer", it.isComputer).put("level", it.difficulty)
            .put("strength", it.strength).put("score", it.score)
    }))
    .put("human", human)
    .put("winner", winner)
    .put("turns", JSONArray(turns.map { t ->
        JSONObject().put("p", t.player).put("kind", t.kind.name).put("word", t.word).put("score", t.score)
            .put("row", t.row).put("col", t.col).put("h", t.horizontal).put("tiles", t.tiles).put("rack", t.rack)
            .also { o -> t.rating?.let { o.put("rating", it) } }
            .also { o -> t.best?.let { o.put("best", it.toJson()) } }
            // The web writes these two the same way (web/leaderboard.py).
            .also { o ->
                t.equity?.let { e ->
                    o.put("eq", JSONObject().put("played", e.played).put("best", e.best)
                        .also { j -> e.play?.let { j.put("play", it.toJson()) } }
                        .also { j -> e.exchange?.let { j.put("exchange", it) } })
                }
            }
            .also { o ->
                t.endgame?.let { e ->
                    o.put("eg", JSONObject().put("played", e.played).put("best", e.best).put("exact", e.exact)
                        .also { j -> e.play?.let { j.put("play", it.toJson()) } })
                }
            }
    }))
    // One character a square: letters are single characters in every language.
    .put("cells", cells.joinToString("") { it.ifEmpty { "." } })
    .put("blanks", JSONArray(blanks.indices.filter { blanks[it] }))
    .put("owners", owners.joinToString("") { if (it < 0) "." else it.toString() })
    .put("hints", hintTurns)
    .put("undos", undos)

fun gameRecordOf(o: JSONObject): GameRecord {
    require(o.getInt("v") == RECORD_VERSION) { "record version ${o.getInt("v")}" }
    val players = o.getJSONArray("players")
    val turns = o.getJSONArray("turns")
    val cells = o.getString("cells")
    val owners = o.getString("owners")
    val blankList = o.getJSONArray("blanks")
    val blanks = (0 until blankList.length()).map { blankList.getInt(it) }.toSet()
    require(cells.length == 225 && owners.length == 225) { "a board has 225 squares" }
    return GameRecord(
        id = o.getString("id"),
        mode = Mode.valueOf(o.getString("mode")),
        language = o.getString("language"),
        startedAt = o.getLong("started"),
        finishedAt = o.getLong("finished"),
        players = (0 until players.length()).map {
            val p = players.getJSONObject(it)
            RecordedPlayer(p.getString("name"), p.getBoolean("computer"), p.getInt("level"), p.getInt("strength"), p.getInt("score"))
        },
        human = o.getInt("human"),
        winner = o.getInt("winner"),
        turns = (0 until turns.length()).map {
            val t = turns.getJSONObject(it)
            RecordedTurn(
                player = t.getInt("p"),
                kind = TurnKind.valueOf(t.getString("kind")),
                word = t.getString("word"),
                score = t.getInt("score"),
                row = t.getInt("row"),
                col = t.getInt("col"),
                horizontal = t.getBoolean("h"),
                tiles = t.getInt("tiles"),
                rack = t.getString("rack"),
                rating = if (t.has("rating")) t.getInt("rating") else null,
                best = t.optJSONObject("best")?.let(::recordedPlayOf),
                equity = t.optJSONObject("eq")?.let { e ->
                    RecordedEquity(
                        e.getDouble("played"),
                        e.getDouble("best"),
                        e.optJSONObject("play")?.let(::recordedPlayOf),
                        if (e.has("exchange")) e.getString("exchange") else null,
                    )
                },
                endgame = t.optJSONObject("eg")?.let { e ->
                    RecordedEndgame(e.getInt("played"), e.getInt("best"), e.getBoolean("exact"), e.optJSONObject("play")?.let(::recordedPlayOf))
                },
            )
        },
        cells = cells.map { if (it == '.') "" else it.toString() },
        blanks = List(225) { it in blanks },
        owners = owners.map { if (it == '.') -1 else it.digitToInt() },
        hintTurns = o.getInt("hints"),
        undos = o.getInt("undos"),
    )
}

/** What reading a history file into the history did. */
data class ImportSummary(val added: Int, val updated: Int, val unchanged: Int, val unreadable: Int)

/**
 * `imported` merged into `existing`: a game not there yet is added, one there
 * under the same id is replaced only by a copy that finished later (the game
 * reopened by an undo and ended again), and the rest stay as they are. The
 * history stays in the order games first ended.
 */
fun mergeHistory(existing: List<GameRecord>, imported: List<GameRecord>): Pair<List<GameRecord>, ImportSummary> {
    val byId = existing.associateBy { it.id }.toMutableMap()
    val added = mutableListOf<GameRecord>()
    var updated = 0
    var unchanged = 0
    for (r in imported.distinctBy { it.id }) {
        val known = byId[r.id]
        when {
            known == null -> added += r.also { byId[it.id] = it }
            r.finishedAt > known.finishedAt && r != known -> { byId[r.id] = r; updated++ }
            else -> unchanged++
        }
    }
    val merged = existing.map { byId.getValue(it.id) } + added.sortedBy { it.finishedAt }
    return merged to ImportSummary(added.size, updated, unchanged, 0)
}

/** The records in a history file's lines, and how many lines were not one. */
fun parseHistory(lines: List<String>): Pair<List<GameRecord>, Int> {
    val text = lines.filter { it.isNotBlank() }
    val records = text.mapNotNull { runCatching { gameRecordOf(JSONObject(it)) }.getOrNull() }
    return records to text.size - records.size
}

/**
 * Every finished game, in the order they first ended, one JSON object a line
 * in `history.jsonl`. New games are appended; replacing or deleting one
 * rewrites the file through a temporary copy, so a kill mid-write leaves the
 * old file. Reading skips a line a kill cut short and rewrites the file
 * without it, so the next append starts on a line of its own.
 *
 * File work runs in order on one background thread, the first job being the
 * read. Once read, a change shows in `records` at once -- a screen opened
 * right after a save already sees it -- and reaches the file after.
 */
@OptIn(ExperimentalCoroutinesApi::class)
class GameHistory(dir: File) {
    private val file = File(dir, "history.jsonl")
    // A failed write must not take the app down: the next change retries it.
    private val io = CoroutineScope(
        SupervisorJob() + Dispatchers.IO.limitedParallelism(1) +
            CoroutineExceptionHandler { _, e -> Log.e(LOG_TAG, "could not write the game history", e) },
    )
    /** What the file holds. Only touched on `io`. */
    private var onDisk: List<GameRecord> = emptyList()

    private val _records = MutableStateFlow<List<GameRecord>?>(null)
    /** Null until the file has been read. */
    val records: StateFlow<List<GameRecord>?> = _records

    init {
        io.launch {
            // Unreadable: start empty, and only ever append -- rewriting would lose the file.
            val lines = runCatching { if (file.exists()) file.readLines().filter { it.isNotBlank() } else emptyList() }
                .onFailure { Log.e(LOG_TAG, "could not read the game history", it) }
                .getOrDefault(emptyList())
            val read = lines.mapNotNull { line ->
                runCatching { gameRecordOf(JSONObject(line)) }
                    .onFailure { Log.w(LOG_TAG, "skipping a history line: $it") }
                    .getOrNull()
            }
            onDisk = read
            _records.value = read
            if (read.size != lines.size) rewrite(read)
        }
    }

    /**
     * Keep a finished game. A game finished again after an undo replaces its
     * earlier ending; one saved again unchanged (a finished game reopened)
     * keeps its time and writes nothing.
     */
    fun save(record: GameRecord) = change { list ->
        val i = list.indexOfFirst { it.id == record.id }
        when {
            i < 0 -> list + record
            list[i] == record.copy(finishedAt = list[i].finishedAt) -> list
            else -> list.toMutableList().also { it[i] = record }
        }
    }

    fun delete(id: String) = change { list -> list.filterNot { it.id == id } }

    /** The whole history as a file to keep somewhere safe: what `history.jsonl` holds, read back by `importLines`. */
    suspend fun exportText(): String = records.first { it != null }!!.joinToString("") { it.toJson().toString() + "\n" }

    /** Read a history file into the history: see `mergeHistory`. */
    suspend fun importLines(lines: List<String>): ImportSummary {
        val (imported, unreadable) = parseHistory(lines)
        // Read first, so the merge below runs on the whole history at once.
        records.first { it != null }
        var summary = ImportSummary(0, 0, 0, unreadable)
        change { list ->
            val (merged, s) = mergeHistory(list, imported)
            summary = s.copy(unreadable = unreadable)
            merged
        }
        return summary
    }

    fun clear() = change { emptyList() }

    private fun change(edit: (List<GameRecord>) -> List<GameRecord>) {
        // Not read yet: the edit waits its turn behind the read.
        val loaded = _records.value != null
        if (loaded) _records.update { it?.let(edit) }
        io.launch {
            if (!loaded) _records.update { it?.let(edit) }
            sync()
        }
    }

    /** Bring the file up to `records`: append when only new games were added. */
    private fun sync() {
        val now = _records.value ?: return
        if (now === onDisk) return
        val appendOnly = now.size > onDisk.size && onDisk.indices.all { now[it] === onDisk[it] }
        when {
            appendOnly -> file.appendText(now.subList(onDisk.size, now.size).joinToString("") { it.toJson().toString() + "\n" })
            now.isEmpty() -> file.delete()
            else -> rewrite(now)
        }
        onDisk = now
    }

    private fun rewrite(all: List<GameRecord>) {
        val tmp = File(file.path + ".tmp")
        tmp.bufferedWriter().use { w -> all.forEach { w.write(it.toJson().toString()); w.write("\n") } }
        check(tmp.renameTo(file)) { "could not replace ${file.name}" }
    }
}
