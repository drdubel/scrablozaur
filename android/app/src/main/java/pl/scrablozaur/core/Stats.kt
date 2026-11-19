package pl.scrablozaur.core

import pl.scrablozaur.engine.TurnKind
import kotlin.math.max

// What the statistics, the ranking and a game's review read off the history.
// Plain functions of the records, so they are tested without a phone.

/** A full rack: a play that uses it all earns the bingo bonus. */
const val RACK_SIZE = 7

/** Plays a game needs before its accuracy can hold a record. */
const val MIN_RATED_PLAYS = 5

enum class Result { Win, Loss, Tie }

val GameRecord.me: RecordedPlayer get() = players[human]

/** The computer the player faced (the strongest other seat, were there more). */
val GameRecord.opponent: RecordedPlayer
    get() = players.filterIndexed { i, _ -> i != human }.maxBy { it.score }

/** The player gave the game up: always the last turn, since it ends the game. */
val GameRecord.resigned: Boolean
    get() = turns.lastOrNull()?.let { it.kind == TurnKind.RESIGN && it.player == human } == true

/**
 * The game as the player's result. A resignation is a loss whatever the score
 * said -- otherwise resigning while ahead would bank a win -- and the rest is
 * the engine's outcome.
 */
val GameRecord.result: Result
    get() = when {
        resigned -> Result.Loss
        winner == human -> Result.Win
        winner < 0 -> Result.Tie
        else -> Result.Loss
    }

val GameRecord.spread: Int get() = me.score - opponent.score

/** The player's own plays, in order. */
val GameRecord.plays: List<RecordedTurn>
    get() = turns.filter { it.player == human && it.kind == TurnKind.PLAY }

/** The mean move rating of the player's plays ("celność"); null when none was rated. */
val GameRecord.accuracy: Double?
    get() = plays.mapNotNull { it.rating }.takeIf { it.isNotEmpty() }?.average()

val GameRecord.bingos: Int get() = plays.count { it.tiles == RACK_SIZE }

/** Points left on the table: what each best play would have scored over the play made. */
val GameRecord.missedPoints: Int
    get() = plays.sumOf { t -> t.best?.let { (it.score - t.score).coerceAtLeast(0) } ?: 0 }

val GameRecord.bestPlay: RecordedTurn? get() = plays.maxByOrNull { it.score }

/** The longest word played, the higher-scoring of equally long ones. */
val GameRecord.longestPlay: RecordedTurn?
    get() = plays.maxWithOrNull(compareBy<RecordedTurn> { it.word.length }.thenBy { it.score })

/** A rated play that scored as much as the best the rack had. */
val RecordedTurn.wasBest: Boolean get() = best != null && score >= best.score

/** One player's -- or, unnamed, everyone's -- results over some games. */
data class Standing(
    val name: String,
    val games: Int,
    val wins: Int,
    val losses: Int,
    val ties: Int,
    val avgScore: Double,
    val bestScore: Int,
    val avgSpread: Double,
    /** The mean of the games' accuracies; null when no game had a rated play. */
    val accuracy: Double?,
    val bingos: Int,
) {
    val winRate: Double get() = if (games == 0) 0.0 else wins.toDouble() / games
}

fun standing(name: String, games: List<GameRecord>): Standing = Standing(
    name = name,
    games = games.size,
    wins = games.count { it.result == Result.Win },
    losses = games.count { it.result == Result.Loss },
    ties = games.count { it.result == Result.Tie },
    avgScore = games.map { it.me.score }.average().orZero(),
    bestScore = games.maxOfOrNull { it.me.score } ?: 0,
    avgSpread = games.map { it.spread }.average().orZero(),
    accuracy = games.mapNotNull { it.accuracy }.takeIf { it.isNotEmpty() }?.average(),
    bingos = games.sumOf { it.bingos },
)

private fun Double.orZero() = if (isNaN()) 0.0 else this

/** How a player's name groups: "Ania" and "ania " are one player. */
fun playerKey(name: String) = name.trim().lowercase()

/** Every player who played here, under the spelling of their latest game. */
fun standings(games: List<GameRecord>): List<Standing> =
    games.groupBy { playerKey(it.me.name) }.values.map { mine ->
        standing(mine.maxBy { it.finishedAt }.me.name, mine)
    }

/** What the ranking orders players by. Ties go to whoever played more. */
enum class RankBy(val key: (Standing) -> Double) {
    WinRate({ it.winRate }),
    AvgScore({ it.avgScore }),
    Accuracy({ it.accuracy ?: -1.0 }),
    Best({ it.bestScore.toDouble() }),
}

fun List<Standing>.ranked(by: RankBy): List<Standing> =
    sortedWith(compareByDescending<Standing> { by.key(it) }.thenByDescending { it.games }.thenBy { it.name })

/** A computer as the history tells them apart: its level and strength. */
data class Opponent(val difficulty: Int, val strength: Int)

val GameRecord.opponentKey: Opponent get() = opponent.let { Opponent(it.difficulty, it.strength) }

/** The player's results against each computer they faced, strongest first. */
fun byOpponent(games: List<GameRecord>): List<Pair<Opponent, Standing>> =
    games.groupBy { it.opponentKey }
        .map { (o, mine) -> o to standing("", mine) }
        .sortedWith(compareByDescending<Pair<Opponent, Standing>> { it.first.difficulty }.thenByDescending { it.first.strength })

/** A record and the game it was set in. */
data class Feat<T>(val game: GameRecord, val value: T)

data class Records(
    val highScore: Feat<Int>?,
    val bestMove: Feat<RecordedTurn>?,
    val longestWord: Feat<RecordedTurn>?,
    val bestAccuracy: Feat<Double>?,
    val biggestWin: Feat<Int>?,
    val mostBingos: Feat<Int>?,
)

/** The bests across `games`. A record tied later stays with whoever set it first. */
fun records(games: List<GameRecord>): Records {
    val chronological = games.sortedBy { it.finishedAt }
    val plays = chronological.flatMap { g -> g.plays.map { Feat(g, it) } }
    return Records(
        highScore = chronological.maxByOrNull { it.me.score }?.let { Feat(it, it.me.score) },
        bestMove = plays.maxByOrNull { it.value.score },
        longestWord = plays.maxWithOrNull(compareBy<Feat<RecordedTurn>> { it.value.word.length }.thenBy { it.value.score }),
        bestAccuracy = chronological
            .filter { g -> g.plays.count { it.rating != null } >= MIN_RATED_PLAYS }
            .mapNotNull { g -> g.accuracy?.let { Feat(g, it) } }
            .maxByOrNull { it.value },
        biggestWin = chronological.filter { it.result == Result.Win }.maxByOrNull { it.spread }?.let { Feat(it, it.spread) },
        mostBingos = chronological.filter { it.bingos > 0 }.maxByOrNull { it.bingos }?.let { Feat(it, it.bingos) },
    )
}

/** The best game scores, highest first; the earlier of two equal ones first. */
fun topScores(games: List<GameRecord>, n: Int = 10): List<GameRecord> =
    games.sortedWith(compareByDescending<GameRecord> { it.me.score }.thenBy { it.finishedAt }).take(n)

/** Each value averaged with up to `window - 1` values before it; a gap stays a gap. */
fun movingAverage(values: List<Double?>, window: Int): List<Double?> = values.indices.map { i ->
    if (values[i] == null) null else values.subList(max(0, i - window + 1), i + 1).filterNotNull().average()
}

/** A position in a game's replay, 225 squares row-major. */
data class Position(val cells: List<String>, val owners: List<Int>)

/**
 * The board after each turn, rebuilt from the plays: `[0]` is the empty
 * board, `[i]` the one after turn `i`. Null when the plays do not add up to
 * the final board -- a game resumed from a save made before turns were
 * logged, whose first turns are missing.
 */
fun GameRecord.positions(): List<Position>? {
    val cells = MutableList(225) { "" }
    val owners = MutableList(225) { -1 }
    val out = mutableListOf(Position(cells.toList(), owners.toList()))
    for (t in turns) {
        if (t.kind == TurnKind.PLAY) {
            t.word.forEachIndexed { i, ch ->
                val r = if (t.horizontal) t.row else t.row + i
                val c = if (t.horizontal) t.col + i else t.col
                if (r !in 0..14 || c !in 0..14) return null
                val square = r * 15 + c
                when {
                    cells[square].isEmpty() -> {
                        cells[square] = ch.toString()
                        owners[square] = t.player
                    }
                    cells[square] != ch.toString() -> return null
                }
            }
        }
        out += Position(cells.toList(), owners.toList())
    }
    return out.takeIf { cells == this.cells }
}

/** Each seat's score after each turn, `[seat][0]` being the start: the turns' points alone. */
fun GameRecord.scoreProgress(): List<List<Int>> {
    val running = IntArray(players.size)
    val out = List(players.size) { mutableListOf(0) }
    for (t in turns) {
        running[t.player] += t.score
        players.indices.forEach { out[it] += running[it] }
    }
    return out
}

/** What the end of the game added to each seat: the racks left over, and the bonus for going out. */
fun GameRecord.endAdjustments(): List<Int> =
    players.indices.map { p -> players[p].score - turns.filter { it.player == p }.sumOf { it.score } }
