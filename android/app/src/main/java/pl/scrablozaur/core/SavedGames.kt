package pl.scrablozaur.core

import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/** A rated human move, for the rating history (the web keeps it client-side too). */
data class RatedMove(val word: String, val score: Int, val rating: Int)

/** One line of an automatic game's move log. */
data class AutoLogEntry(val player: String, val word: String, val score: Int, val passed: Boolean)

/** The game in progress, as written to disk after every move. */
data class SavedGame(
    val language: String,
    /** `Game.save()`: the whole session, undo history included. */
    val session: String,
    /** Finished: kept so the last board can still be looked at, never resumed into. */
    val over: Boolean,
    val ratings: List<RatedMove>,
    val autoLog: List<AutoLogEntry>,
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
        val ratings = json.optJSONArray("ratings") ?: JSONArray()
        val log = json.optJSONArray("auto_log") ?: JSONArray()
        SavedGame(
            language = json.getString("language"),
            session = json.getString("session"),
            over = json.optBoolean("over", false),
            ratings = (0 until ratings.length()).map {
                val r = ratings.getJSONObject(it)
                RatedMove(r.getString("word"), r.getInt("score"), r.getInt("rating"))
            },
            autoLog = (0 until log.length()).map {
                val e = log.getJSONObject(it)
                AutoLogEntry(e.getString("player"), e.getString("word"), e.getInt("score"), e.getBoolean("passed"))
            },
        )
    }.getOrNull()

    fun save(game: SavedGame) {
        val json = JSONObject()
            .put("language", game.language)
            .put("session", game.session)
            .put("over", game.over)
            .put("ratings", JSONArray(game.ratings.map {
                JSONObject().put("word", it.word).put("score", it.score).put("rating", it.rating)
            }))
            .put("auto_log", JSONArray(game.autoLog.map {
                JSONObject().put("player", it.player).put("word", it.word).put("score", it.score)
                    .put("passed", it.passed)
            }))
        val tmp = File(file.path + ".tmp")
        tmp.writeText(json.toString())
        tmp.renameTo(file)
    }

    fun clear() {
        file.delete()
    }
}
