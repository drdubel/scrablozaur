package pl.scrablozaur.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.TurnKind
import kotlin.math.abs

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
        rack: String = "",
        equity: RecordedEquity? = null,
    ) = RecordedTurn(player, TurnKind.PLAY, word, score, row, col, horizontal, tiles, rack, rating, best, equity)

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
        language: String = "pl",
    ) = GameRecord(
        id = id,
        mode = Mode.COMPETITIVE,
        language = language,
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

    /** A game passed round the table: `seats` as (name, is a computer, score). */
    private fun table(
        id: String,
        seats: List<Triple<String, Boolean, Int>>,
        winner: Int,
        turns: List<RecordedTurn> = emptyList(),
        at: Long = 0,
    ) = GameRecord(
        id = id,
        mode = Mode.MULTIPLAYER,
        language = "pl",
        startedAt = at - 60_000,
        finishedAt = at,
        players = seats.map { (name, computer, score) -> RecordedPlayer(name, computer, 8, 100, score) },
        human = 0,
        winner = winner,
        turns = turns,
        cells = List(225) { "" },
        blanks = List(225) { false },
        owners = List(225) { -1 },
        hintTurns = 0,
        undos = 0,
    )

    @Test
    fun theResultIsTheEnginesOutcome() {
        assertEquals(Result.Win, game(winner = 0).mySeat.result)
        assertEquals(Result.Loss, game(winner = 1).mySeat.result)
        assertEquals(Result.Tie, game(winner = -1).mySeat.result)
    }

    @Test
    fun aResignationIsALossWhateverTheScore() {
        val g = game(winner = 0, scores = 120 to 40, turns = listOf(play(0, "kot", 120), turn(0, TurnKind.RESIGN)))
        assertEquals(Result.Loss, g.mySeat.result)
        // The computer has no resignation to make; only the player's counts.
        assertEquals(Result.Win, game(winner = 0, turns = listOf(turn(1, TurnKind.PASS))).mySeat.result)
    }

    @Test
    fun everyPersonAtATableHasTheirOwnResult() {
        // Ala resigned while ahead; Ola and Ela finished level; the computer trailed.
        val g = table(
            "t",
            listOf(Triple("Ala", false, 300), Triple("Ola", false, 250), Triple("Ela", false, 250), Triple("Bot", true, 200)),
            winner = -1,
            turns = listOf(turn(0, TurnKind.RESIGN)),
        )
        assertEquals(listOf("Ala", "Ola", "Ela"), g.seats.map { it.name })
        assertEquals(listOf(Result.Loss, Result.Tie, Result.Tie), g.seats.map { it.result })
        // A seat behind the winner lost, the winner won.
        val won = table("w", listOf(Triple("Ala", false, 300), Triple("Ola", false, 250)), winner = 0)
        assertEquals(listOf(Result.Win, Result.Loss), won.seats.map { it.result })
        // The opponent is the best other seat, a person here.
        assertEquals(Opponent(0, 0, "ala"), won.seats[1].opponentKey)
        assertEquals(-50, won.seats[1].spread)
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
        assertEquals(75.0, g.mySeat.accuracy!!, 1e-9)
        assertEquals(2, g.mySeat.plays.size)
        assertNull(game(turns = listOf(play(1, "pies", 30))).mySeat.accuracy)
    }

    @Test
    fun bingosMissedPointsEquityAndTheBestPlay() {
        val g = game(
            turns = listOf(
                play(0, "kotkami", 80, tiles = 7, rating = 100, best = best("kotkami", 80), equity = RecordedEquity(90.0, 90.0)),
                play(0, "dom", 5, rating = 20, best = best("domek", 25), equity = RecordedEquity(7.5, 30.0, best("domek", 25))),
                play(0, "las", 12, rating = 90, best = best("lasy", 14)),
            ),
        )
        val me = g.mySeat
        assertEquals(1, me.bingos)
        assertEquals(20 + 2, me.missedPoints)
        assertEquals(22.5, me.equityLost!!, 1e-9)
        assertNull(me.endgameLost)
        assertEquals("kotkami", me.bestPlay!!.word)
        assertEquals(listOf(true, false, false), me.plays.map { it.wasBest })
    }

    @Test
    fun standingsGroupANameWhateverItsCase() {
        val games = listOf(
            game(id = "1", name = "ania", winner = 0, scores = 300 to 200, at = 1),
            game(id = "2", name = "Ania ", winner = 1, scores = 200 to 250, at = 2),
            game(id = "3", name = "Tomek", winner = -1, scores = 222 to 222, at = 3),
        )
        val table = standings(games.allSeats()).associateBy { playerKey(it.name) }
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
    fun aTableGameCountsForEveryoneAtIt() {
        val games = listOf(
            game(id = "1", name = "Ala", winner = 0, at = 1),
            table("2", listOf(Triple("Ala", false, 200), Triple("Ola", false, 300)), winner = 1, at = 2),
        )
        val s = standings(games.allSeats()).associateBy { it.name }
        assertEquals(2, s.getValue("Ala").games)
        assertEquals(1, s.getValue("Ala").wins)
        assertEquals(1, s.getValue("Ola").wins)
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
        val s = standings(games.allSeats())
        assertEquals(listOf("Ola", "Tomek", "Ania"), s.ranked(RankBy.WinRate).map { it.name })
        assertEquals(listOf("Ola", "Ania", "Tomek"), s.ranked(RankBy.Best).map { it.name })
    }

    @Test
    fun aRecordStaysWithWhoeverSetItFirst() {
        val games = listOf(
            game(id = "late", name = "Tomek", scores = 300 to 100, at = 2),
            game(id = "early", name = "Ania", scores = 300 to 100, at = 1),
        )
        assertEquals("early", records(games.allSeats()).highScore!!.game.id)
        assertEquals(listOf("early", "late"), topScores(games.allSeats()).map { it.game.id })
    }

    @Test
    fun theBestAccuracyNeedsEnoughRatedPlays() {
        val few = game(id = "few", at = 1, turns = List(MIN_RATED_PLAYS - 1) { play(0, "kot", 10, rating = 100) })
        val enough = game(id = "enough", at = 2, turns = List(MIN_RATED_PLAYS) { play(0, "kot", 10, rating = 80) })
        assertEquals("enough", records(listOf(few, enough).allSeats()).bestAccuracy!!.game.id)
        assertNull(records(listOf(few).allSeats()).bestAccuracy)
    }

    @Test
    fun theBiggestWinIsAWin() {
        val games = listOf(
            game(id = "loss", winner = 1, scores = 100 to 400, at = 1),
            game(id = "win", winner = 0, scores = 200 to 150, at = 2),
        )
        val r = records(games.allSeats())
        assertEquals("win", r.biggestWin!!.game.id)
        assertEquals(50, r.biggestWin.value)
    }

    @Test
    fun resultsAgainstEachOpponentComputersStrongestFirstThenPeople() {
        val games = listOf(
            game(id = "1", opponent = Opponent(8, 50), winner = 0),
            game(id = "2", opponent = Opponent(10, 100), winner = 1),
            game(id = "3", opponent = Opponent(10, 100), winner = 0),
            table("4", listOf(Triple("Ania", false, 200), Triple("ola", false, 100)), winner = 0, at = 1),
            table("5", listOf(Triple("Ania", false, 200), Triple("Ola", false, 250)), winner = 1, at = 2),
        )
        val rows = byOpponent(games.allSeats().filter { it.name == "Ania" })
        assertEquals(listOf(Opponent(10, 100), Opponent(8, 50), Opponent(0, 0, "Ola")), rows.map { it.first })
        assertEquals(2, rows[0].second.games)
        assertEquals(1, rows[0].second.wins)
        // One person under either spelling, shown as they last played.
        assertEquals(2, rows[2].second.games)
    }

    @Test
    fun headToHeadCountsWhoFinishedAhead() {
        val games = listOf(
            table("1", listOf(Triple("Ala", false, 300), Triple("Ola", false, 200), Triple("Bot", true, 250)), winner = 0, at = 1),
            table("2", listOf(Triple("ola", false, 310), Triple("Ala", false, 300)), winner = 0, at = 2),
            // Ala resigned while ahead: Ola finished above her.
            table("3", listOf(Triple("Ala", false, 300), Triple("Ola", false, 100)), winner = 1, turns = listOf(turn(0, TurnKind.RESIGN)), at = 3),
            table("4", listOf(Triple("Ala", false, 222), Triple("Ola", false, 222)), winner = -1, at = 4),
        )
        val duels = headToHead(games)
        assertEquals(1, duels.size)
        val d = duels.single()
        assertEquals("Ala" to "Ola", d.first to d.second)
        assertEquals(Triple(1, 2, 1), Triple(d.firstWins, d.secondWins, d.ties))
        // Games against the computer alone set nobody against anybody.
        assertTrue(headToHead(listOf(game())).isEmpty())
    }

    @Test
    fun aRatingFollowsTheResultsAndStaysFiniteOnASweep() {
        assertEquals(0.5, expectedScore(0.0), 1e-12)
        assertTrue(expectedScore(-200.0) > 0.75)
        // No games: the prior alone, the anchor.
        assertEquals(ANCHOR_RATING, performanceRating(emptyList()), 0.5)
        // Even results against an opponent rated 1300 put the player at 1300, about.
        val even = List(20) { RatedResult(1300.0, if (it % 2 == 0) 1.0 else 0.0) }
        assertTrue(abs(performanceRating(even) - 1300.0) < 40.0)
        // Winning every game is better than winning most, and still a number.
        val sweep = performanceRating(List(10) { RatedResult(1500.0, 1.0) })
        val most = performanceRating(List(10) { RatedResult(1500.0, if (it < 8) 1.0 else 0.0) })
        assertTrue(sweep > most && most > 1500.0 && sweep < 2500.0)
        // The history ends where a fit of everything ends.
        val history = ratingHistory(even)
        assertEquals(even.size, history.size)
        assertEquals(performanceRating(even), history.last(), 1e-6)
    }

    @Test
    fun ratedResultsAreOnePersonAgainstOneRatedComputer() {
        val games = listOf(
            game(id = "1", winner = 0, opponent = Opponent(10, 100)),
            game(id = "2", winner = -1, opponent = Opponent(8, 50)),
            game(id = "3", winner = 0, language = "xx"),
            table("4", listOf(Triple("Ania", false, 1), Triple("Bot", true, 2), Triple("Ola", false, 0)), winner = 1),
        )
        val rated = ratedResults(games.allSeats().filter { it.name == "Ania" }) { language, level, strength ->
            if (language == "pl") level * 100.0 + strength else null
        }
        assertEquals(listOf("1", "2"), rated.map { it.first.game.id })
        assertEquals(listOf(RatedResult(1100.0, 1.0), RatedResult(850.0, 0.5)), rated.map { it.second })
    }

    @Test
    fun aRatingIsDescribedByTheStrongestEngineThatReachesIt() {
        // Each engine spans [base, base + 99] over its strengths.
        val ratingOf = { level: Int, strength: Int -> (level - 8) * 50.0 + 1000.0 + strength - 1 }
        val levels = listOf(8, 9, 10)
        // 1120 is within 9's [1050, 1149] and 10's [1100, 1199]: the stronger engine names it.
        assertEquals(Likeness(10, 21), likeness(1120.0, levels, ratingOf))
        assertEquals(Likeness(8, 31), likeness(1030.0, levels, ratingOf))
        assertEquals(Likeness(10, 100, beyond = 1), likeness(1300.0, levels, ratingOf))
        assertEquals(Likeness(8, 1, beyond = -1), likeness(900.0, levels, ratingOf))
        assertNull(likeness(1000.0, levels) { _, _ -> null })
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
    fun puzzlesAreTheWeakPlaysThatCanBeSetUpAgain() {
        val (turns, cells) = crossing()
        val weak = turns[0].copy(rack = "kotabcd", rating = 30, best = best("kotka", 30))
        val noRack = turns[0].copy(rating = 10, best = best("kotka", 30))
        val good = turns[0].copy(rack = "kotabcd", rating = 95, best = best("kotka", 30))
        val p = puzzles(listOf(game(id = "a", turns = listOf(weak, turns[1]), cells = cells)).allSeats())
        assertEquals(listOf("a" to 0), p.map { it.seat.game.id to it.turn })
        // No rack on record, a play near the best, or a board that does not rebuild: no puzzle.
        assertTrue(puzzles(listOf(game(turns = listOf(noRack, turns[1]), cells = cells)).allSeats()).isEmpty())
        assertTrue(puzzles(listOf(game(turns = listOf(good, turns[1]), cells = cells)).allSeats()).isEmpty())
        assertTrue(puzzles(listOf(game(turns = listOf(turns[1]), cells = cells)).allSeats()).isEmpty())
        // Equity alone makes one too, when it lost enough.
        val costly = turns[0].copy(rack = "kotabcd", equity = RecordedEquity(5.0, 20.0, best("kotka", 30)))
        assertEquals(1, puzzles(listOf(game(turns = listOf(costly, turns[1]), cells = cells)).allSeats()).size)
    }

    @Test
    fun aMovingAverageLeavesGapsAsGaps() {
        val avg = movingAverage(listOf(10.0, null, 20.0, 30.0, 40.0), window = 3)
        assertEquals(listOf(10.0, null, 15.0, 25.0, 30.0), avg)
    }
}
