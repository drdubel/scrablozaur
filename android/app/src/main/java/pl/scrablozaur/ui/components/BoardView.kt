package pl.scrablozaur.ui.components

import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import pl.scrablozaur.R
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.theme.Palette

/** A highlight laid over a square, beneath any tile on it. */
enum class Mark { HintNew, HintExisting, LastMove }

/** Everything drawn on the board beyond the position itself. */
data class BoardOverlay(
    /** Squares being typed onto, with their letters. */
    val typed: Map<Cell, Char> = emptyMap(),
    /** Typed squares a blank will fill: they score 0, whatever the letter. */
    val typedBlanks: Set<Cell> = emptySet(),
    /** Tiles on the board the typed word runs through. */
    val typedThrough: Set<Cell> = emptySet(),
    val cursor: Cell? = null,
    /** Whether the typed word is playable: null while unknown. */
    val typedValid: Boolean? = null,
    val marks: Map<Cell, Mark> = emptyMap(),
)

private val premiumLabels = mapOf(
    Premium.TRIPLE_WORD to "3S",
    Premium.DOUBLE_WORD to "2S",
    Premium.TRIPLE_LETTER to "3L",
    Premium.DOUBLE_LETTER to "2L",
    Premium.CENTRE to "★",
)

private fun premiumColors(p: Premium): Pair<Color, Color> = when (p) {
    Premium.TRIPLE_WORD -> Palette.tw to Palette.twText
    Premium.DOUBLE_WORD, Premium.CENTRE -> Palette.dw to Palette.dwText
    Premium.TRIPLE_LETTER -> Palette.tl to Palette.tlText
    Premium.DOUBLE_LETTER -> Palette.dl to Palette.dlText
    Premium.NONE -> Palette.empty to Palette.empty
}

/** Text layouts cached by string and style, so a redraw measures nothing new. */
private class TextCache(private val measurer: TextMeasurer) {
    private val cache = HashMap<Pair<String, TextStyle>, TextLayoutResult>()
    fun layout(text: String, style: TextStyle) = cache.getOrPut(text to style) { measurer.measure(text, style) }
}

/** Where each square sits on a board `width` pixels across. */
private class Geometry(width: Float) {
    val step = width / 15f
    val gap = step * 0.06f
    val tile = step - gap
    val radius = CornerRadius(step * 0.12f)
    val square = Size(tile, tile)
    fun topLeft(cell: Cell) = Offset(cell.col * step + gap / 2, cell.row * step + gap / 2)
}

/**
 * The 15x15 board: premium squares, tiles tinted by the seat that played them,
 * blanks marked, and the typing/hint overlays. A tap reports the square.
 */
@Composable
fun BoardView(
    cells: List<String>,
    blanks: List<Boolean>,
    owners: List<Byte>,
    premiums: List<Premium>,
    points: Map<String, UInt>,
    overlay: BoardOverlay,
    modifier: Modifier = Modifier,
    onTap: (Cell) -> Unit = {},
) {
    val measurer = rememberTextMeasurer()
    val texts = remember(measurer) { TextCache(measurer) }
    val tap by rememberUpdatedState(onTap)
    val description = stringResource(R.string.board_description)

    Box(
        modifier
            .aspectRatio(1f)
            .semantics { contentDescription = description }
            .pointerInput(Unit) {
                detectTapGestures { offset ->
                    val step = size.width / 15f
                    tap(Cell((offset.y / step).toInt().coerceIn(0, 14), (offset.x / step).toInt().coerceIn(0, 14)))
                }
            },
    ) {
        Canvas(Modifier.fillMaxSize()) {
            drawBoard(Geometry(size.width), texts, cells, blanks, owners, premiums, points, overlay)
        }
        // The cursor blinks on a layer of its own that exists only while a word
        // is being typed. Read by the board's draw block, the animation would
        // redraw all 225 squares every frame for as long as a game is open.
        overlay.cursor?.let { CursorLayer(it, Modifier.fillMaxSize()) }
    }
}

@Composable
private fun CursorLayer(cursor: Cell, modifier: Modifier) {
    val blink by rememberInfiniteTransition(label = "cursor").animateFloat(
        initialValue = 1f,
        targetValue = 0.15f,
        animationSpec = infiniteRepeatable(tween(550), RepeatMode.Reverse),
        label = "cursorAlpha",
    )
    Canvas(modifier) {
        val g = Geometry(size.width)
        drawRoundRect(
            Palette.cursor.copy(alpha = blink), g.topLeft(cursor), g.square, g.radius,
            style = Stroke(width = g.gap * 3f),
        )
    }
}

private fun DrawScope.drawBoard(
    g: Geometry,
    texts: TextCache,
    cells: List<String>,
    blanks: List<Boolean>,
    owners: List<Byte>,
    premiums: List<Premium>,
    points: Map<String, UInt>,
    overlay: BoardOverlay,
) {
    val tile = g.tile
    val letterStyle = TextStyle(fontSize = (tile * 0.50f).toSp(), fontWeight = FontWeight.Bold, color = Palette.text)
    val valueStyle = TextStyle(fontSize = (tile * 0.28f).toSp(), color = Palette.muted)
    val labelStyle = TextStyle(fontSize = (tile * 0.30f).toSp(), fontWeight = FontWeight.SemiBold)

    for (index in 0 until 225) {
        val cell = Cell.of(index)
        val topLeft = g.topLeft(cell)
        val letter = cells.getOrNull(index).orEmpty()
        val typed = overlay.typed[cell]
        val mark = overlay.marks[cell]

        when {
            typed != null -> {
                val (bg, fg) = when (overlay.typedValid) {
                    true -> Palette.valid to Palette.validText
                    false -> Palette.invalid to Palette.invalidText
                    null -> Palette.typingNew to Palette.typingNewText
                }
                drawRoundRect(bg, topLeft, g.square, g.radius)
                val value = if (cell in overlay.typedBlanks) 0u else points[typed.toString()]
                drawTile(texts, typed.toString(), value, topLeft, tile, letterStyle.copy(color = fg), valueStyle.copy(color = fg))
            }
            letter.isNotEmpty() -> {
                val owner = owners.getOrNull(index)?.toInt() ?: -1
                val (bg, border) = Palette.owners.getOrNull(owner) ?: (Palette.placed to Palette.placedBorder)
                val fill = if (mark == Mark.HintExisting || cell in overlay.typedThrough) Palette.highlightExisting else bg
                drawRoundRect(fill, topLeft, g.square, g.radius)
                val blank = blanks.getOrNull(index) == true
                if (blank) {
                    drawRoundRect(
                        Palette.blankAccent, topLeft, g.square, g.radius,
                        style = Stroke(width = g.gap * 1.6f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(tile / 8, tile / 10))),
                    )
                } else {
                    drawRoundRect(border, topLeft, g.square, g.radius, style = Stroke(width = g.gap * 0.8f))
                }
                drawTile(texts, letter, if (blank) 0u else points[letter], topLeft, tile, letterStyle, valueStyle)
            }
            else -> {
                val premium = premiums.getOrNull(index) ?: Premium.NONE
                val (bg, fg) = if (mark == Mark.HintNew) Palette.highlightNew to Palette.highlightNewText else premiumColors(premium)
                drawRoundRect(bg, topLeft, g.square, g.radius)
                premiumLabels[premium]?.takeIf { mark != Mark.HintNew }?.let { label ->
                    val layout = texts.layout(label, labelStyle.copy(color = fg))
                    drawText(layout, topLeft = topLeft + Offset((tile - layout.size.width) / 2, (tile - layout.size.height) / 2))
                }
            }
        }
        if (mark == Mark.LastMove) {
            drawRoundRect(Palette.primary, topLeft, g.square, g.radius, style = Stroke(width = g.gap * 1.8f))
        }
    }
}

private fun DrawScope.drawTile(
    texts: TextCache,
    letter: String,
    value: UInt?,
    topLeft: Offset,
    tile: Float,
    letterStyle: TextStyle,
    valueStyle: TextStyle,
) {
    val l = texts.layout(letter.uppercase(), letterStyle)
    drawText(l, topLeft = topLeft + Offset((tile - l.size.width) / 2 - tile * 0.04f, (tile - l.size.height) / 2 - tile * 0.02f))
    if (value != null) {
        val v = texts.layout(value.toString(), valueStyle)
        drawText(v, topLeft = topLeft + Offset(tile - v.size.width - tile * 0.06f, tile - v.size.height))
    }
}
