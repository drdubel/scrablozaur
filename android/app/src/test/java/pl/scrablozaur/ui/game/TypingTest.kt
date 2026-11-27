package pl.scrablozaur.ui.game

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The typing model must assemble exactly the word the web's board.js does. */
class TypingTest {
    private fun grid(vararg tiles: Pair<Cell, Char>): Grid {
        val map = tiles.toMap()
        return Grid { r, c -> map[Cell(r, c)] }
    }

    @Test
    fun typesAcrossEmptySquares() {
        val g = grid()
        val t = Typing(Cell(7, 6), horizontal = true).type('k', g).type('o', g).type('t', g)
        assertEquals(WordData("kot", 7, 6, true), t.wordData(g))
        assertEquals(listOf('k', 'o', 't'), t.typedLetters)
    }

    @Test
    fun skipsATileThatDoesNotMatchAndKeepsItInTheWord() {
        // "a" already at (7,7); typing k-t from (7,6) jumps over it.
        val g = grid(Cell(7, 7) to 'a')
        val t = Typing(Cell(7, 6), horizontal = true).type('k', g).type('t', g)
        assertEquals("kat", t.wordData(g)!!.word)
        assertEquals(setOf(Cell(7, 7)), t.passedThrough)
    }

    @Test
    fun passesThroughATileWhenItsLetterIsTyped() {
        val g = grid(Cell(7, 7) to 'o')
        val t = Typing(Cell(7, 6), horizontal = true).type('k', g).type('o', g).type('t', g)
        assertEquals("kot", t.wordData(g)!!.word)
        assertEquals(listOf('k', 't'), t.typedLetters)
    }

    @Test
    fun extendsThroughTilesTouchingEitherEnd() {
        // "ko" on the board at (3,4..5); typing "t" at (3,6) spells "kot".
        val g = grid(Cell(3, 4) to 'k', Cell(3, 5) to 'o')
        val t = Typing(Cell(3, 6), horizontal = true).type('t', g)
        assertEquals(WordData("kot", 3, 4, true), t.wordData(g))
    }

    @Test
    fun backspaceReturnsTheCursorToBeforeTheSkippedTiles() {
        val g = grid(Cell(7, 7) to 'a')
        val t = Typing(Cell(7, 6), horizontal = true).type('k', g).type('t', g).backspace()
        assertEquals(Cell(7, 7), t.cursor)
        assertEquals(listOf('k'), t.typedLetters)
    }

    @Test
    fun stopsAtTheEdge() {
        val g = grid()
        val t = Typing(Cell(14, 13), horizontal = true).type('a', g).type('b', g).type('c', g)
        assertEquals(listOf('a', 'b'), t.typedLetters)
        assertNull(t.cursor)
    }

    @Test
    fun rackUsesRealTilesBeforeBlanks() {
        assertEquals(setOf(0, 2), rackTilesUsed("k?ot", listOf('k', 'o'), '?'))
        assertEquals(setOf(0, 1), rackTilesUsed("k?ot", listOf('k', 'k'), '?'))
    }

    @Test
    fun allocationSaysWhichLettersABlankFills() {
        // The second k has no real tile left, so the blank (index 1) covers it.
        assertEquals(listOf(0, 1, 2), rackAllocation("k?ot", listOf('k', 'k', 'o'), '?'))
        // A letter the rack cannot supply at all takes nothing.
        assertEquals(listOf(0, null), rackAllocation("kot", listOf('k', 'x'), '?'))
    }
}
