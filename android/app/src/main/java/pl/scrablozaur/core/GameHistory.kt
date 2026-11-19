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
)

/**
 * A finished game: what the statistics, the ranking and the game's review are
 * made from. Competitive games only -- the one mode with a player to follow.
 */
data class GameRecord(
    val id: String,
    val mode: Mode,
    val language: String,
    val startedAt: Long,
    val finishedAt: Long,
    val players: List<RecordedPlayer>,
    /** The seat the phone's player had. */
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
            .also { o ->
                t.best?.let {
                    o.put("best", JSONObject().put("word", it.word).put("score", it.score).put("row", it.row)
                        .put("col", it.col).put("h", it.horizontal).put("used", it.used))
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
                best = t.optJSONObject("best")?.let { b ->
                    RecordedPlay(b.getString("word"), b.getInt("score"), b.getInt("row"), b.getInt("col"), b.getBoolean("h"), b.getString("used"))
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
