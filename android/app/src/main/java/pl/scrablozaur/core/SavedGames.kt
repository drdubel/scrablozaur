package pl.scrablozaur.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/** What the history needs to know of a game beyond the engine's own save. */
data class GameMeta(
    /** The game's record in the history: finishing it again replaces that record. */
    val id: String = UUID.randomUUID().toString(),
    val startedAt: Long = System.currentTimeMillis(),
    /** Move numbers on which the hints were opened. */
    val hintTurns: Set<Int> = emptySet(),
    val undos: Int = 0,
)

/** The game in progress, as written to disk after every move. */
data class SavedGame(
    val language: String,
    /** `Game.save()`: the whole session, undo history and turn log included. */
    val session: String,
    /** Finished: kept so the last board can still be looked at, never resumed into. */
    val over: Boolean,
    val meta: GameMeta,
)

/**
 * The one game in progress. Android kills background apps without warning, so
 * the session is saved after every action and restored on launch. Written to a
 * temporary file and renamed, so a kill mid-write leaves the previous save.
 */
class SavedGames(dir: File) {
    private val file = File(dir, "current_game.json")

    /** Whether there is a game to go back to: saved, and not yet over. */
    fun hasUnfinishedGame(): Boolean = load()?.over == false

    fun load(): SavedGame? = runCatching {
        val json = JSONObject(file.readText())
        // A save from before the history has no meta: the game gets a new id.
        val hints = json.optJSONArray("hint_turns") ?: JSONArray()
        SavedGame(
            language = json.getString("language"),
            session = json.getString("session"),
            over = json.optBoolean("over", false),
            meta = GameMeta(
                id = json.optString("id").ifEmpty { UUID.randomUUID().toString() },
                startedAt = json.optLong("started_at").takeIf { it > 0 } ?: file.lastModified(),
                hintTurns = (0 until hints.length()).map { hints.getInt(it) }.toSet(),
                undos = json.optInt("undos", 0),
            ),
        )
    }.getOrNull()

    fun save(game: SavedGame) {
        val json = JSONObject()
            .put("language", game.language)
            .put("session", game.session)
            .put("over", game.over)
            .put("id", game.meta.id)
            .put("started_at", game.meta.startedAt)
            .put("hint_turns", JSONArray(game.meta.hintTurns.sorted()))
            .put("undos", game.meta.undos)
        val tmp = File(file.path + ".tmp")
        tmp.writeText(json.toString())
        tmp.renameTo(file)
    }

    fun clear() {
        file.delete()
    }
}
