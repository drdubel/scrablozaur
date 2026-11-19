package pl.scrablozaur.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Test
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.TurnKind

/** What the statistics, the ranking and a game's review read off the history. */
class StatsTest {
    private fun play(
        player: Int,
        word: String,
        score: Int,
        row: Int = 7,
        col: Int = 7,
        horizontal: Boolean = true,
        tiles: Int = word.length,
        rating: Int? = null,
        best: RecordedPlay? = null,
    ) = RecordedTurn(player, TurnKind.PLAY, word, score, row, col, horizontal, tiles, "", rating, best)

    private fun turn(player: Int, kind: TurnKind, tiles: Int = 0) = RecordedTurn(player, kind, tiles = tiles)

    private fun best(word: String, score: Int) = RecordedPlay(word, score, 7, 7, true, word)

    private fun game(
        id: String = "g",
        name: String = "Ania",
        winner: Int = 0,
        scores: Pair<Int, Int> = 100 to 80,
        turns: List<RecordedTurn> = emptyList(),
        at: Long = 0,
        opponent: Opponent = Opponent(10, 100),
        cells: List<String> = List(225) { "" },
    ) = GameRecord(
        id = id,
        mode = Mode.COMPETITIVE,
        language = "pl",
        startedAt = at - 60_000,
        finishedAt = at,
        players = listOf(
            RecordedPlayer(name, false, 5, 100, scores.first),
            RecordedPlayer("Komputer", true, opponent.difficulty, opponent.strength, scores.second),
        ),
        human = 0,
        winner = winner,
        turns = turns,
        cells = cells,
        blanks = List(225) { false },
        owners = List(225) { -1 },
        hintTurns = 0,
        undos = 0,
    )

    @Test
    fun theResultIsTheEnginesOutcome() {
        assertEquals(Result.Win, game(winner = 0).result)
        assertEquals(Result.Loss, game(winner = 1).result)
        assertEquals(Result.Tie, game(winner = -1).result)
    }

    @Test
    fun aResignationIsALossWhateverTheScore() {
        val g = game(winner = 0, scores = 120 to 40, turns = listOf(play(0, "kot", 120), turn(0, TurnKind.RESIGN)))
        assertEquals(Result.Loss, g.result)
        // The computer has no resignation to make; only the player's counts.
        assertEquals(Result.Win, game(winner = 0, turns = listOf(turn(1, TurnKind.PASS))).result)
    }

    @Test
    fun accuracyAveragesTheRatedPlaysOnly() {
        val g = game(
            turns = listOf(
                play(0, "kot", 10, rating = 100, best = best("kot", 10)),
                play(1, "pies", 30),
                turn(0, TurnKind.EXCHANGE, tiles = 3),
                play(0, "dom", 5, rating = 50, best = best("domek", 25)),
            ),
        )
        assertEquals(75.0, g.accuracy!!, 1e-9)
        assertEquals(2, g.plays.size)
        assertNull(game(turns = listOf(play(1, "pies", 30))).accuracy)
    }

    @Test
    fun bingosMissedPointsAndTheBestPlay() {
        val g = game(
            turns = listOf(
                play(0, "kotkami", 80, tiles = 7, rating = 100, best = best("kotkami", 80)),
                play(0, "dom", 5, rating = 20, best = best("domek", 25)),
                play(0, "las", 12, rating = 90, best = best("lasy", 14)),
            ),
        )
        assertEquals(1, g.bingos)
        assertEquals(20 + 2, g.missedPoints)
        assertEquals("kotkami", g.bestPlay!!.word)
        assertEquals(listOf(true, false, false), g.plays.map { it.wasBest })
    }

    @Test
    fun standingsGroupANameWhateverItsCase() {
        val games = listOf(
            game(id = "1", name = "ania", winner = 0, scores = 300 to 200, at = 1),
            game(id = "2", name = "Ania ", winner = 1, scores = 200 to 250, at = 2),
            game(id = "3", name = "Tomek", winner = -1, scores = 222 to 222, at = 3),
        )
        val table = standings(games).associateBy { playerKey(it.name) }
        val ania = table.getValue("ania")
        assertEquals("Ania ", ania.name)
        assertEquals(2, ania.games)
        assertEquals(0.5, ania.winRate, 1e-9)
        assertEquals(250.0, ania.avgScore, 1e-9)
        assertEquals(300, ania.bestScore)
        assertEquals(25.0, ania.avgSpread, 1e-9)
        assertEquals(1, table.getValue("tomek").ties)
    }

    @Test
    fun theRankingOrdersByTheChosenFigureThenByGamesPlayed() {
        val games = listOf(
            game(id = "1", name = "Ania", winner = 0, scores = 200 to 100, at = 1),
            game(id = "2", name = "Ania", winner = 1, scores = 150 to 200, at = 2),
            game(id = "3", name = "Tomek", winner = 0, scores = 160 to 100, at = 3),
            game(id = "4", name = "Ola", winner = 0, scores = 400 to 100, at = 4),
            game(id = "5", name = "Ola", winner = 0, scores = 100 to 50, at = 5),
        )
        val s = standings(games)
        assertEquals(listOf("Ola", "Tomek", "Ania"), s.ranked(RankBy.WinRate).map { it.name })
        assertEquals(listOf("Ola", "Ania", "Tomek"), s.ranked(RankBy.Best).map { it.name })
    }

    @Test
    fun aRecordStaysWithWhoeverSetItFirst() {
        val games = listOf(
            game(id = "late", name = "Tomek", scores = 300 to 100, at = 2),
            game(id = "early", name = "Ania", scores = 300 to 100, at = 1),
        )
        assertEquals("early", records(games).highScore!!.game.id)
        assertEquals(listOf("early", "late"), topScores(games).map { it.id })
    }

    @Test
    fun theBestAccuracyNeedsEnoughRatedPlays() {
        val few = game(id = "few", at = 1, turns = List(MIN_RATED_PLAYS - 1) { play(0, "kot", 10, rating = 100) })
        val enough = game(id = "enough", at = 2, turns = List(MIN_RATED_PLAYS) { play(0, "kot", 10, rating = 80) })
        assertEquals("enough", records(listOf(few, enough)).bestAccuracy!!.game.id)
        assertNull(records(listOf(few)).bestAccuracy)
    }

    @Test
    fun theBiggestWinIsAWin() {
        val games = listOf(
            game(id = "loss", winner = 1, scores = 100 to 400, at = 1),
            game(id = "win", winner = 0, scores = 200 to 150, at = 2),
        )
        val r = records(games)
        assertEquals("win", r.biggestWin!!.game.id)
        assertEquals(50, r.biggestWin.value)
    }

    @Test
    fun resultsAgainstEachComputerStrongestFirst() {
        val games = listOf(
            game(id = "1", opponent = Opponent(8, 50), winner = 0),
            game(id = "2", opponent = Opponent(10, 100), winner = 1),
            game(id = "3", opponent = Opponent(10, 100), winner = 0),
        )
        val rows = byOpponent(games)
        assertEquals(listOf(Opponent(10, 100), Opponent(8, 50)), rows.map { it.first })
        assertEquals(2, rows[0].second.games)
        assertEquals(1, rows[0].second.wins)
    }

    /** "kot" across the centre, then "koc" down from its "k". */
    private fun crossing(): Pair<List<RecordedTurn>, List<String>> {
        val turns = listOf(
            play(0, "kot", 10, row = 7, col = 7),
            play(1, "koc", 8, row = 7, col = 7, horizontal = false, tiles = 2),
        )
        val cells = MutableList(225) { "" }
        cells[7 * 15 + 7] = "k"
        cells[7 * 15 + 8] = "o"
        cells[7 * 15 + 9] = "t"
        cells[8 * 15 + 7] = "o"
        cells[9 * 15 + 7] = "c"
        return turns to cells
    }

    @Test
    fun positionsRebuildTheBoardTurnByTurn() {
        val (turns, cells) = crossing()
        val positions = game(turns = turns, cells = cells).positions()
        assertNotNull(positions)
        positions!!
        assertEquals(3, positions.size)
        assertEquals(List(225) { "" }, positions[0].cells)
        assertEquals("t", positions[1].cells[7 * 15 + 9])
        assertEquals("", positions[1].cells[8 * 15 + 7])
        assertEquals(cells, positions[2].cells)
        // The shared "k" stays the first player's.
        assertEquals(0, positions[2].owners[7 * 15 + 7])
        assertEquals(1, positions[2].owners[9 * 15 + 7])
    }

    @Test
    fun aLogMissingItsFirstTurnsReplaysNothing() {
        val (turns, cells) = crossing()
        assertNull(game(turns = turns.drop(1), cells = cells).positions())
    }

    @Test
    fun theScoreAsItWentAndWhatTheEndAdded() {
        val (turns, cells) = crossing()
        val g = game(turns = turns, cells = cells, scores = 14 to 4)
        assertEquals(listOf(listOf(0, 10, 10), listOf(0, 0, 8)), g.scoreProgress())
        assertEquals(listOf(4, -4), g.endAdjustments())
    }

    @Test
    fun aMovingAverageLeavesGapsAsGaps() {
        val avg = movingAverage(listOf(10.0, null, 20.0, 30.0, 40.0), window = 3)
        assertEquals(listOf(10.0, null, 15.0, 25.0, 30.0), avg)
    }
}
