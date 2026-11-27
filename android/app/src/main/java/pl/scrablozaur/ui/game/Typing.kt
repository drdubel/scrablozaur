package pl.scrablozaur.ui.game

/** A square on the 15x15 board. */
data class Cell(val row: Int, val col: Int) {
    val index get() = row * SIZE + col

    companion object {
        const val SIZE = 15
        fun of(index: Int) = Cell(index / SIZE, index % SIZE)
    }
}

/** What the board holds: a letter, or `null` for an empty square. */
fun interface Grid {
    fun at(row: Int, col: Int): Char?
}

/** The word the typed letters spell on the board, as the engine wants it. */
data class WordData(val word: String, val row: Int, val col: Int, val horizontal: Boolean)

/**
 * One letter of a word being typed onto the board. `letter == null` marks a
 * pass-through: the player typed the letter already on that square.
 * `skipped` lists tiles already on the board that the cursor jumped over to
 * reach this square.
 */
data class TypedEntry(val cell: Cell, val letter: Char?, val skipped: List<Cell>)

/**
 * A word being typed onto the board, the web's typing mode (`board.js`): pick
 * a start square and a direction, then each letter lands on the next free
 * square, skipping tiles already on the board -- or passing through one when the
 * player types the letter it already holds.
 */
data class Typing(
    val start: Cell,
    val horizontal: Boolean,
    val entries: List<TypedEntry> = emptyList(),
    val cursor: Cell? = start,
) {
    private fun next(c: Cell): Cell? {
        val n = if (horizontal) Cell(c.row, c.col + 1) else Cell(c.row + 1, c.col)
        return if (n.row < Cell.SIZE && n.col < Cell.SIZE) n else null
    }

    fun type(letter: Char, grid: Grid): Typing {
        var at = cursor ?: return this
        val existing = grid.at(at.row, at.col)
        if (existing != null && existing == letter) {
            return copy(entries = entries + TypedEntry(at, null, emptyList()), cursor = next(at))
        }
        val skipped = mutableListOf<Cell>()
        while (grid.at(at.row, at.col) != null) {
            skipped += at
            at = next(at) ?: return this
        }
        return copy(entries = entries + TypedEntry(at, letter, skipped), cursor = next(at))
    }

    fun backspace(): Typing {
        val last = entries.lastOrNull() ?: return this
        val back = if (last.letter == null) last.cell else last.skipped.firstOrNull() ?: last.cell
        return copy(entries = entries.dropLast(1), cursor = back)
    }

    /** Letters placed so far, in order -- the tiles leaving the rack. */
    val typedLetters: List<Char> get() = entries.mapNotNull { it.letter }

    /** Squares that receive a new tile, with their letters. */
    val placed: Map<Cell, Char> get() = entries.filter { it.letter != null }.associate { it.cell to it.letter!! }

    /** Existing tiles the word runs through. */
    val passedThrough: Set<Cell> get() = entries.flatMap { e -> e.skipped + listOfNotNull(e.cell.takeIf { e.letter == null }) }.toSet()

    /**
     * The whole word along the line: extended through any tiles touching
     * either end, exactly as it will lie on the board (`getWordData`).
     */
    fun wordData(grid: Grid): WordData? {
        if (entries.isEmpty()) return null
        val dr = if (horizontal) 0 else 1
        val dc = if (horizontal) 1 else 0
        var (sr, sc) = entries.first().cell.let { it.row to it.col }
        while (sr - dr >= 0 && sc - dc >= 0 && grid.at(sr - dr, sc - dc) != null) {
            sr -= dr
            sc -= dc
        }
        var (er, ec) = entries.last().cell.let { it.row to it.col }
        while (er + dr < Cell.SIZE && ec + dc < Cell.SIZE && grid.at(er + dr, ec + dc) != null) {
            er += dr
            ec += dc
        }
        val placed = placed
        val word = StringBuilder()
        var r = sr
        var c = sc
        while (r <= er && c <= ec) {
            word.append(grid.at(r, c) ?: placed[Cell(r, c)] ?: '?')
            r += dr
            c += dc
        }
        return WordData(word.toString(), sr, sc, horizontal)
    }
}

/**
 * The rack tile each typed letter takes, in order: a matching real tile first,
 * else a blank -- the same greedy allocation the engine makes, so what the
 * rack and board show is exactly what the move will take. `null` where the
 * rack has nothing left for a letter.
 */
fun rackAllocation(rack: String, typed: List<Char>, blank: Char): List<Int?> {
    val used = mutableSetOf<Int>()
    return typed.map { letter ->
        val real = rack.indices.firstOrNull { it !in used && rack[it] != blank && rack[it] == letter }
        val idx = real ?: rack.indices.firstOrNull { it !in used && rack[it] == blank }
        idx?.also { used += it }
    }
}

/** Which rack tiles the typed letters use, to grey them out. */
fun rackTilesUsed(rack: String, typed: List<Char>, blank: Char): Set<Int> =
    rackAllocation(rack, typed, blank).filterNotNull().toSet()
