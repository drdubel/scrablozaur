package pl.scrablozaur.core

import pl.scrablozaur.engine.TurnKind
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.pow

// What the statistics, the ranking and a game's review read off the history.
// Plain functions of the records, so they are tested without a phone.

/** A full rack: a play that uses it all earns the bingo bonus. */
const val RACK_SIZE = 7

/** Plays a game needs before its accuracy can hold a record. */
const val MIN_RATED_PLAYS = 5

enum class Result { Win, Loss, Tie }

/**
 * One person's part in a game: what the statistics count. A competitive game
 * has one; a game passed around the table has one for every person in it.
 */
data class Seat(val game: GameRecord, val index: Int) {
    val player: RecordedPlayer get() = game.players[index]
    val name: String get() = player.name
    val score: Int get() = player.score
}

/** Everyone who played the game in person. */
val GameRecord.seats: List<Seat> get() = players.indices.filter { !players[it].isComputer }.map { Seat(this, it) }

/** The seat the phone's player had: where a game's review starts. */
val GameRecord.mySeat: Seat get() = Seat(this, human)

/** Every person's seat in every game, the order of the games kept. */
fun List<GameRecord>.allSeats(): List<Seat> = flatMap { it.seats }

/** The seat gave the game up. A competitive resignation ends the game; a multiplayer one leaves it. */
val Seat.resigned: Boolean
    get() = game.turns.any { it.kind == TurnKind.RESIGN && it.player == index }

/** The strongest other seat: the one that finished highest. */
val Seat.opponent: RecordedPlayer
    get() = game.players.filterIndexed { i, _ -> i != index }.maxBy { it.score }

/**
 * The game as this seat's result. A resignation is a loss whatever the score
 * said -- otherwise resigning while ahead would bank a win; otherwise the
 * engine's outcome, a tie shared by the seats level at the top.
 */
val Seat.result: Result
    get() = when {
        resigned -> Result.Loss
        game.winner == index -> Result.Win
        game.winner < 0 -> {
            val standing = game.players.indices.filter { !Seat(game, it).resigned }.map { game.players[it].score }
            if (standing.isNotEmpty() && score == standing.max()) Result.Tie else Result.Loss
        }
        else -> Result.Loss
    }

val Seat.spread: Int get() = score - opponent.score

/** The seat's own plays, in order. */
val Seat.plays: List<RecordedTurn>
    get() = game.turns.filter { it.player == index && it.kind == TurnKind.PLAY }

/** The seat's turns the engine reviewed: plays and exchanges by equity, the endgame by search. */
val Seat.reviewedTurns: List<RecordedTurn>
    get() = game.turns.filter { it.player == index && (it.equity != null || it.endgame != null) }

/** The mean move rating of the seat's plays ("celność"); null when none was rated. */
val Seat.accuracy: Double?
    get() = plays.mapNotNull { it.rating }.takeIf { it.isNotEmpty() }?.average()

val Seat.bingos: Int get() = plays.count { it.tiles == RACK_SIZE }

/** Points left on the table: what each best play would have scored over the play made. */
val Seat.missedPoints: Int
    get() = plays.sumOf { t -> t.best?.let { (it.score - t.score).coerceAtLeast(0) } ?: 0 }

/** Equity given away over the turns valued by it: what Strateg (level 9) would have kept. Null when none was. */
val Seat.equityLost: Double?
    get() = game.turns.filter { it.player == index }.mapNotNull { it.equity?.loss }.takeIf { it.isNotEmpty() }?.sum()

/** Points the endgame cost against the best play there was. Null when no endgame turn was searched. */
val Seat.endgameLost: Int?
    get() = game.turns.filter { it.player == index }.mapNotNull { it.endgame?.loss }.takeIf { it.isNotEmpty() }?.sum()

val Seat.bestPlay: RecordedTurn? get() = plays.maxByOrNull { it.score }

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

fun standing(name: String, seats: List<Seat>): Standing = Standing(
    name = name,
    games = seats.size,
    wins = seats.count { it.result == Result.Win },
    losses = seats.count { it.result == Result.Loss },
    ties = seats.count { it.result == Result.Tie },
    avgScore = seats.map { it.score }.average().orZero(),
    bestScore = seats.maxOfOrNull { it.score } ?: 0,
    avgSpread = seats.map { it.spread }.average().orZero(),
    accuracy = seats.mapNotNull { it.accuracy }.takeIf { it.isNotEmpty() }?.average(),
    bingos = seats.sumOf { it.bingos },
)

private fun Double.orZero() = if (isNaN()) 0.0 else this

/** How a player's name groups: "Ania" and "ania " are one player. */
fun playerKey(name: String) = name.trim().lowercase()

/** Every player who played here, under the spelling of their latest game. */
fun standings(seats: List<Seat>): List<Standing> =
    seats.groupBy { playerKey(it.name) }.values.map { mine ->
        standing(mine.maxBy { it.game.finishedAt }.name, mine)
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

/** Who a seat played: a computer by its level and strength, a person by name. */
data class Opponent(val difficulty: Int, val strength: Int, val person: String? = null)

/** The strongest other seat as `Opponent`, a person under `playerKey` so spellings group. */
val Seat.opponentKey: Opponent
    get() = opponent.let { if (it.isComputer) Opponent(it.difficulty, it.strength) else Opponent(0, 0, playerKey(it.name)) }

/** The player's results against each opponent they faced: computers strongest first, then people. */
fun byOpponent(seats: List<Seat>): List<Pair<Opponent, Standing>> =
    seats.groupBy { it.opponentKey }
        .map { (o, mine) ->
            // A person shows under the spelling of their latest game.
            val shown = o.person?.let { mine.maxBy { s -> s.game.finishedAt }.opponent.name }
            o.copy(person = shown) to standing("", mine)
        }
        .sortedWith(
            compareBy<Pair<Opponent, Standing>> { it.first.person != null }
                .thenByDescending { it.first.difficulty }
                .thenByDescending { it.first.strength }
                .thenByDescending { it.second.games },
        )

/** Two people's games against each other: how often each finished ahead. */
data class HeadToHead(val first: String, val second: String, val firstWins: Int, val secondWins: Int, val ties: Int) {
    val games: Int get() = firstWins + secondWins + ties
}

/**
 * Every pair of people who sat at the same game. One finished ahead of the
 * other on points, unless one of them resigned (then the other did); a game
 * both resigned decides nothing between them.
 */
fun headToHead(games: List<GameRecord>): List<HeadToHead> {
    data class Tally(var a: String, var b: String, var aWins: Int = 0, var bWins: Int = 0, var ties: Int = 0)
    val tallies = LinkedHashMap<Pair<String, String>, Tally>()
    for (g in games.sortedBy { it.finishedAt }) {
        val people = g.seats
        for (i in people.indices) for (j in i + 1 until people.size) {
            val (x, y) = listOf(people[i], people[j]).sortedBy { playerKey(it.name) }
            if (playerKey(x.name) == playerKey(y.name)) continue
            val t = tallies.getOrPut(playerKey(x.name) to playerKey(y.name)) { Tally(x.name, y.name) }
            // The latest spelling of each name.
            t.a = x.name
            t.b = y.name
            when {
                x.resigned && y.resigned -> continue
                x.resigned -> t.bWins++
                y.resigned -> t.aWins++
                x.score > y.score -> t.aWins++
                y.score > x.score -> t.bWins++
                else -> t.ties++
            }
        }
    }
    return tallies.values
        .map { HeadToHead(it.a, it.b, it.aWins, it.bWins, it.ties) }
        .filter { it.games > 0 }
        .sortedByDescending { it.games }
}

// ── Elo ──────────────────────────────────────────────────────────────────────

/** What the bots' ratings are anchored to: level 8 at full strength (`game::ladder`). */
const val ANCHOR_RATING = 1500.0

/** A game against a computer, for the rating: the computer's Elo and how it went (1, ½ or 0). */
data class RatedResult(val opponent: Double, val score: Double)

/** Elo's expected score against an opponent `diff` points stronger. */
fun expectedScore(diff: Double): Double = 1.0 / (1.0 + 10.0.pow(diff / 400.0))

/**
 * The rating that best explains `results` -- Elo's model fitted by maximum
 * likelihood -- with `priorGames` drawn games against an anchor-rated
 * opponent added, so that a few games, or nothing but wins, still give a
 * finite and modest number.
 */
fun performanceRating(results: List<RatedResult>, priorGames: Int = 2): Double {
    val all = results + List(priorGames) { RatedResult(ANCHOR_RATING, 0.5) }
    // The likelihood's slope, sum(score - expected), falls as the rating rises: bisect for its zero.
    fun slope(r: Double) = all.sumOf { it.score - expectedScore(it.opponent - r) }
    var lo = (all.minOf { it.opponent }) - 1200.0
    var hi = (all.maxOf { it.opponent }) + 1200.0
    repeat(60) {
        val mid = (lo + hi) / 2
        if (slope(mid) > 0) lo = mid else hi = mid
    }
    return (lo + hi) / 2
}

/** The rating after each game in turn: everything up to it, refitted. */
fun ratingHistory(results: List<RatedResult>): List<Double> = results.indices.map { performanceRating(results.subList(0, it + 1)) }

/** The rated games among `seats`: one person against one computer whose rating `ratingOf` knows. */
fun ratedResults(seats: List<Seat>, ratingOf: (language: String, level: Int, strength: Int) -> Double?): List<Pair<Seat, RatedResult>> =
    seats.mapNotNull { s ->
        val bot = s.opponent.takeIf { s.game.players.size == 2 && it.isComputer } ?: return@mapNotNull null
        val rating = ratingOf(s.game.language, bot.difficulty, bot.strength) ?: return@mapNotNull null
        val score = when (s.result) {
            Result.Win -> 1.0
            Result.Tie -> 0.5
            Result.Loss -> 0.0
        }
        s to RatedResult(rating, score)
    }

/**
 * The computer a rating plays like: the strongest engine whose range of
 * strengths takes it in, at the strength nearest it. `beyond` is 1 above
 * every engine at full strength, -1 below every one at the weakest, else 0.
 */
data class Likeness(val level: Int, val strength: Int, val beyond: Int = 0)

fun likeness(rating: Double, levels: List<Int>, ratingOf: (level: Int, strength: Int) -> Double?): Likeness? {
    val ranges = levels.mapNotNull { level ->
        val table = (1..100).mapNotNull { s -> ratingOf(level, s)?.let { s to it } }
        table.takeIf { it.isNotEmpty() }?.let { level to it }
    }
    if (ranges.isEmpty()) return null
    val strongest = ranges.maxBy { (_, t) -> t.maxOf { it.second } }
    if (rating > strongest.second.maxOf { it.second }) return Likeness(strongest.first, 100, beyond = 1)
    val weakest = ranges.minBy { (_, t) -> t.minOf { it.second } }
    if (rating < weakest.second.minOf { it.second }) return Likeness(weakest.first, 1, beyond = -1)
    // Strongest engine first: a player is likelier to know it from their own games.
    val covering = ranges.sortedByDescending { it.first }
        .firstOrNull { (_, t) -> rating in t.minOf { it.second }..t.maxOf { it.second } } ?: ranges.first()
    val (level, table) = covering
    return Likeness(level, table.minBy { abs(it.second - rating) }.first)
}

// ── Records ──────────────────────────────────────────────────────────────────

/** A record and the seat that set it. */
data class Feat<T>(val seat: Seat, val value: T) {
    val game: GameRecord get() = seat.game
}

data class Records(
    val highScore: Feat<Int>?,
    val bestMove: Feat<RecordedTurn>?,
    val longestWord: Feat<RecordedTurn>?,
    val bestAccuracy: Feat<Double>?,
    val biggestWin: Feat<Int>?,
    val mostBingos: Feat<Int>?,
)

/** The bests across `seats`. A record tied later stays with whoever set it first. */
fun records(seats: List<Seat>): Records {
    val chronological = seats.sortedBy { it.game.finishedAt }
    val plays = chronological.flatMap { s -> s.plays.map { Feat(s, it) } }
    return Records(
        highScore = chronological.maxByOrNull { it.score }?.let { Feat(it, it.score) },
        bestMove = plays.maxByOrNull { it.value.score },
        longestWord = plays.maxWithOrNull(compareBy<Feat<RecordedTurn>> { it.value.word.length }.thenBy { it.value.score }),
        bestAccuracy = chronological
            .filter { s -> s.plays.count { it.rating != null } >= MIN_RATED_PLAYS }
            .mapNotNull { s -> s.accuracy?.let { Feat(s, it) } }
            .maxByOrNull { it.value },
        biggestWin = chronological.filter { it.result == Result.Win }.maxByOrNull { it.spread }?.let { Feat(it, it.spread) },
        mostBingos = chronological.filter { it.bingos > 0 }.maxByOrNull { it.bingos }?.let { Feat(it, it.bingos) },
    )
}

/** The best game scores, highest first; the earlier of two equal ones first. */
fun topScores(seats: List<Seat>, n: Int = 10): List<Seat> =
    seats.sortedWith(compareByDescending<Seat> { it.score }.thenBy { it.game.finishedAt }).take(n)

/** Each value averaged with up to `window - 1` values before it; a gap stays a gap. */
fun movingAverage(values: List<Double?>, window: Int): List<Double?> = values.indices.map { i ->
    if (values[i] == null) null else values.subList(max(0, i - window + 1), i + 1).filterNotNull().average()
}

// ── Replay ───────────────────────────────────────────────────────────────────

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

// ── Practice ─────────────────────────────────────────────────────────────────

/** A turn of the player's that is worth trying again: the position before it, the rack, what was played. */
data class Puzzle(val seat: Seat, val turn: Int) {
    val played: RecordedTurn get() = seat.game.turns[turn]
}

/**
 * The plays a person got most wrong, worst first: rated plays below
 * `maxRating`, or valued by equity and `minEquityLoss` short of the best --
 * with the rack on record and the position rebuildable, so the turn can be
 * set up again.
 */
fun puzzles(seats: List<Seat>, maxRating: Int = 60, minEquityLoss: Double = 8.0): List<Puzzle> =
    seats.filter { it.game.positions() != null }.flatMap { s ->
        s.game.turns.withIndex()
            .filter { (_, t) -> t.player == s.index && t.kind == TurnKind.PLAY && t.rack.isNotEmpty() }
            .filter { (_, t) -> (t.rating != null && t.rating < maxRating && !t.wasBest) || (t.equity?.loss ?: 0.0) >= minEquityLoss }
            .map { (i, _) -> Puzzle(s, i) }
    }.sortedByDescending { p -> p.played.equity?.loss ?: ((100 - (p.played.rating ?: 100)) / 10.0) }
