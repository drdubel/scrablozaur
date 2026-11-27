package pl.scrablozaur.ui.components

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ChartsTest {
    @Test
    fun anAxisHasRoundEndsAndSteps() {
        assertEquals(Axis(0f, 500f, listOf(0f, 100f, 200f, 300f, 400f, 500f)), niceAxis(0f, 412f))
        assertEquals(Axis(0f, 100f, listOf(0f, 20f, 40f, 60f, 80f, 100f)), niceAxis(0f, 100f))
    }

    @Test
    fun anAxisCoversItsValues() {
        for ((lo, hi) in listOf(0f to 1f, 0f to 7f, 13f to 87f, 250f to 260f, -12f to 30f)) {
            val axis = niceAxis(lo, hi)
            assertTrue("$lo..$hi in $axis", axis.min <= lo && axis.max >= hi)
            assertTrue("$axis has room for a line", axis.ticks.size in 3..11)
        }
    }

    @Test
    fun aFlatRunStillGetsAnAxis() {
        val axis = niceAxis(5f, 5f)
        assertTrue(axis.max > axis.min)
    }
}
