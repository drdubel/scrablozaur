package pl.scrablozaur.core

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/** One player's line in the shared ranking (`standings` in web/leaderboard.py). */
data class OnlinePlayer(
    val name: String,
    val games: Int,
    val wins: Int,
    val losses: Int,
    val ties: Int,
    val winRate: Double,
    val avgScore: Double,
    val bestScore: Int,
    val accuracy: Double?,
)

/** One of the shared ranking's best scores. */
data class OnlineScore(
    val name: String,
    val score: Int,
    val opponent: String,
    val opponentComputer: Boolean,
    val opponentLevel: Int?,
    val opponentStrength: Int?,
    val result: Result,
    val language: String,
    val finished: Long,
)

data class OnlineStandings(
    /** False where the server keeps no ranking. */
    val enabled: Boolean,
    val games: Int,
    /** Every language played there, whatever the filter. */
    val languages: List<String>,
    val players: List<OnlinePlayer>,
    val topScores: List<OnlineScore>,
)

/** What an upload came to: the games sent, and the ones the server could read and kept. */
data class UploadResult(val sent: Int, val kept: Int)

/** Why the ranking server could not be used; the screen words it. */
sealed class LeaderboardException(message: String) : Exception(message) {
    /** No such address, no network, or no answer. */
    class Unreachable(cause: Throwable) : LeaderboardException(cause.message ?: cause.toString())

    /** No ranking there: switched off, or not a Scrablozaur server at all. */
    class Disabled : LeaderboardException("no ranking on this server")

    /** The server wants a key, and not this one. */
    class BadToken : LeaderboardException("the server refused the key")

    class Server(code: Int, detail: String) : LeaderboardException("HTTP $code: $detail")
}

/**
 * The shared ranking on a Scrablozaur web server (web/leaderboard.py). The app
 * sends its finished games there when the player asks it to, and reads
 * everyone's standings back; it never sends anything by itself.
 */
object Leaderboard {
    /** Games per request; the server takes up to 2000. */
    private const val BATCH = 500

    /** Send `games`, replacing whatever this phone (`source`) sent of them before. */
    suspend fun upload(server: String, token: String, source: String, games: List<GameRecord>): UploadResult =
        withContext(Dispatchers.IO) {
            var kept = 0
            for (batch in games.chunked(BATCH)) {
                val body = JSONObject().put("source", source).put("games", JSONArray(batch.map { it.toJson() }))
                kept += request("POST", endpoint(server, "/api/leaderboard/games"), token, body.toString()).getInt("kept")
            }
            UploadResult(games.size, kept)
        }

    /** Everyone's standings, in `language` or in all of them. */
    suspend fun standings(server: String, language: String?): OnlineStandings = withContext(Dispatchers.IO) {
        val query = language?.let { "?language=" + URLEncoder.encode(it, "UTF-8") }.orEmpty()
        parseStandings(request("GET", endpoint(server, "/api/leaderboard$query"), null, null))
    }

    /** `path` on the server as the player typed its address: with or without a scheme (HTTPS by default) or a trailing slash. */
    fun endpoint(server: String, path: String): String {
        val base = server.trim().trimEnd('/')
        return (if ("://" in base) base else "https://$base") + path
    }

    /** Whether the key and the games travel encrypted. */
    fun isEncrypted(server: String) = endpoint(server, "").startsWith("https://", ignoreCase = true)

    private fun request(method: String, url: String, token: String?, body: String?): JSONObject {
        val conn = try {
            URL(url).openConnection() as HttpURLConnection
        } catch (e: Exception) {
            throw LeaderboardException.Unreachable(e)
        }
        try {
            conn.requestMethod = method
            conn.connectTimeout = 10_000
            conn.readTimeout = 30_000
            conn.setRequestProperty("Accept", "application/json")
            if (!token.isNullOrBlank()) conn.setRequestProperty("X-Leaderboard-Token", token.trim())
            if (body != null) {
                conn.doOutput = true
                conn.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                conn.outputStream.use { it.write(body.toByteArray()) }
            }
            val code = conn.responseCode
            val text = (if (code in 200..299) conn.inputStream else conn.errorStream)
                ?.bufferedReader()?.use { it.readText() }.orEmpty()
            return when (code) {
                in 200..299 -> runCatching { JSONObject(text) }.getOrElse { throw LeaderboardException.Disabled() }
                403 -> throw LeaderboardException.BadToken()
                404 -> throw LeaderboardException.Disabled()
                else -> throw LeaderboardException.Server(code, runCatching { JSONObject(text).optString("detail") }.getOrNull().orEmpty())
            }
        } catch (e: IOException) {
            throw LeaderboardException.Unreachable(e)
        } finally {
            conn.disconnect()
        }
    }
}

private fun JSONObject.intOrNull(key: String): Int? = if (has(key) && !isNull(key)) getInt(key) else null

private fun JSONObject.doubleOrNull(key: String): Double? = if (has(key) && !isNull(key)) getDouble(key) else null

/** `GET /api/leaderboard`'s answer. */
fun parseStandings(o: JSONObject): OnlineStandings {
    val players = o.optJSONArray("players") ?: JSONArray()
    val top = o.optJSONArray("top_scores") ?: JSONArray()
    val languages = o.optJSONArray("languages") ?: JSONArray()
    return OnlineStandings(
        enabled = o.optBoolean("enabled", false),
        games = o.optInt("games", 0),
        languages = (0 until languages.length()).map { languages.getString(it) },
        players = (0 until players.length()).map {
            val p = players.getJSONObject(it)
            OnlinePlayer(
                name = p.getString("name"),
                games = p.getInt("games"),
                wins = p.getInt("wins"),
                losses = p.getInt("losses"),
                ties = p.getInt("ties"),
                winRate = p.getDouble("win_rate"),
                avgScore = p.getDouble("avg_score"),
                bestScore = p.getInt("best_score"),
                accuracy = p.doubleOrNull("accuracy"),
            )
        },
        topScores = (0 until top.length()).map {
            val s = top.getJSONObject(it)
            OnlineScore(
                name = s.getString("name"),
                score = s.getInt("score"),
                opponent = s.getString("opponent"),
                opponentComputer = s.getBoolean("opponent_computer"),
                opponentLevel = s.intOrNull("opponent_level"),
                opponentStrength = s.intOrNull("opponent_strength"),
                result = when (s.getString("result")) {
                    "win" -> Result.Win
                    "tie" -> Result.Tie
                    else -> Result.Loss
                },
                language = s.getString("language"),
                finished = s.getLong("finished"),
            )
        },
    )
}
