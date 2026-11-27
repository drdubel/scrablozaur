package pl.scrablozaur.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.TurnKind

/** The history's file format, merging a history file into it, and the shared ranking's answers. */
class HistoryTest {
    private fun record(id: String, finishedAt: Long, score: Int = 100, turns: List<RecordedTurn> = emptyList()) = GameRecord(
        id = id,
        mode = Mode.MULTIPLAYER,
        language = "pl",
        startedAt = finishedAt - 60_000,
        finishedAt = finishedAt,
        players = listOf(
            RecordedPlayer("Ala", false, 5, 100, score),
            RecordedPlayer("Ola", false, 5, 100, 90),
            RecordedPlayer("Komputer", true, 10, 75, 80),
        ),
        human = 1,
        winner = 0,
        turns = turns,
        cells = List(225) { if (it == 112) "k" else "" },
        blanks = List(225) { it == 112 },
        owners = List(225) { if (it == 112) 0 else -1 },
        hintTurns = 2,
        undos = 1,
    )

    @Test
    fun aRecordReadsBackAsItWasWritten() {
        val play = RecordedPlay("kotka", 32, 7, 3, true, "ko?ka")
        val turns = listOf(
            RecordedTurn(0, TurnKind.PLAY, "kot", 10, 7, 7, true, 3, "kotabcd", 40, play, RecordedEquity(12.5, 30.25, play)),
            RecordedTurn(1, TurnKind.EXCHANGE, tiles = 2, rack = "aabbccd", equity = RecordedEquity(-3.0, 4.5, exchange = "bc")),
            RecordedTurn(0, TurnKind.PASS, rack = "ab", endgame = RecordedEndgame(-12, 3, exact = false, play = play)),
            RecordedTurn(1, TurnKind.PLAY, "ab", 4, 8, 8, false, 1, "b", endgame = RecordedEndgame(5, 5, exact = true)),
            RecordedTurn(2, TurnKind.RESIGN),
        )
        val original = record("g1", 1_700_000_000_000, turns = turns)
        val json = original.toJson()
        val back = gameRecordOf(JSONObject(json.toString()))
        assertEquals(original, back)
        // The keys the web and the server read (web/leaderboard.py).
        val first = json.getJSONArray("turns").getJSONObject(0)
        assertEquals(setOf("played", "best", "play"), first.getJSONObject("eq").keys().asSequence().toSet())
        assertEquals("MULTIPLAYER", json.getString("mode"))
        assertEquals(setOf("played", "best", "exact", "play"), json.getJSONArray("turns").getJSONObject(2).getJSONObject("eg").keys().asSequence().toSet())
    }

    @Test
    fun aRecordFromBeforeTheReviewsStillReads() {
        val old = record("old", 1).toJson()
        // Written by a build that knew neither key.
        val t = JSONObject().put("p", 0).put("kind", "PLAY").put("word", "kot").put("score", 10).put("row", 7).put("col", 7)
            .put("h", true).put("tiles", 3).put("rack", "kotabcd").put("rating", 70)
        old.put("turns", org.json.JSONArray().put(t))
        val turn = gameRecordOf(old).turns.single()
        assertEquals(70, turn.rating)
        assertNull(turn.equity)
        assertNull(turn.endgame)
    }

    @Test
    fun importingAddsNewGamesAndKeepsTheRest() {
        val a = record("a", 10)
        val b = record("b", 20)
        val bLater = record("b", 30, score = 150)
        val bEarlier = record("b", 15, score = 50)
        val c = record("c", 5)
        val (merged, summary) = mergeHistory(listOf(a, b), listOf(c, bLater, a))
        // In the order games first ended: what was there, then what is new.
        assertEquals(listOf("a", "b", "c"), merged.map { it.id })
        assertEquals(150, merged[1].players[0].score)
        assertEquals(ImportSummary(added = 1, updated = 1, unchanged = 1, unreadable = 0), summary)
        // An older copy of a game does not replace a newer ending.
        val (kept, s2) = mergeHistory(listOf(b), listOf(bEarlier))
        assertEquals(b, kept.single())
        assertEquals(1, s2.unchanged)
    }

    @Test
    fun aHistoryFileSkipsWhatItCannotRead() {
        val lines = listOf(record("a", 1).toJson().toString(), "", "{not json", JSONObject().put("v", 1).toString())
        val (records, unreadable) = parseHistory(lines)
        assertEquals(listOf("a"), records.map { it.id })
        assertEquals(2, unreadable)
    }

    @Test
    fun theRankingServersAnswerReads() {
        val answer = JSONObject(
            """
            {"enabled": true, "games": 3, "languages": ["en", "pl"],
             "players": [{"name": "Ala", "games": 3, "wins": 2, "losses": 1, "ties": 0, "win_rate": 0.667,
                          "avg_score": 301.5, "best_score": 412, "accuracy": null}],
             "top_scores": [{"name": "Ala", "score": 412, "opponent": "Komputer", "opponent_computer": true,
                             "opponent_level": 10, "opponent_strength": 100, "result": "win", "language": "pl",
                             "finished": 1700000000000},
                            {"name": "Ala", "score": 200, "opponent": "Ola", "opponent_computer": false,
                             "opponent_level": null, "opponent_strength": null, "result": "tie", "language": "en",
                             "finished": 1700000000001}]}
            """,
        )
        val s = parseStandings(answer)
        assertTrue(s.enabled)
        assertEquals(listOf("en", "pl"), s.languages)
        assertNull(s.players.single().accuracy)
        assertEquals(412, s.players.single().bestScore)
        assertEquals(listOf(Result.Win, Result.Tie), s.topScores.map { it.result })
        assertEquals(10, s.topScores[0].opponentLevel)
        assertNull(s.topScores[1].opponentStrength)
        // A server with the ranking switched off answers with no players.
        assertFalse(parseStandings(JSONObject("""{"enabled": false, "games": 0, "languages": [], "players": [], "top_scores": []}""")).enabled)
    }

    @Test
    fun aServerAddressWorksAsTyped() {
        assertEquals("https://example.org/api/leaderboard", Leaderboard.endpoint("example.org/", "/api/leaderboard"))
        assertEquals("http://192.168.1.5:8000/api/leaderboard", Leaderboard.endpoint(" http://192.168.1.5:8000 ", "/api/leaderboard"))
        assertTrue(Leaderboard.isEncrypted("example.org"))
        assertFalse(Leaderboard.isEncrypted("http://192.168.1.5:8000"))
    }
}
