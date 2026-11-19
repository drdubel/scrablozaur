package pl.scrablozaur.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.text.TextLayoutResult
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.ui.theme.Palette
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.roundToInt

/** One line of a `LineChart`. A null value leaves a gap in it. */
data class Series(val label: String, val color: Color, val values: List<Float?>, val dashed: Boolean = false)

/** A chart's value axis: round ends and the round values between them. */
data class Axis(val min: Float, val max: Float, val ticks: List<Float>)

/**
 * An axis spanning `min..max` with about `count` ticks on steps of 1, 2 or 5
 * times a power of ten (Heckbert's "nice numbers").
 */
fun niceAxis(min: Float, max: Float, count: Int = 5): Axis {
    val hi = if (max - min < 1e-6f) min + 1f else max
    fun nice(x: Double, round: Boolean): Double {
        val exp = floor(log10(x))
        val f = x / 10.0.pow(exp)
        val n = if (round) {
            when {
                f < 1.5 -> 1.0
                f < 3 -> 2.0
                f < 7 -> 5.0
                else -> 10.0
            }
        } else {
            when {
                f <= 1 -> 1.0
                f <= 2 -> 2.0
                f <= 5 -> 5.0
                else -> 10.0
            }
        }
        return n * 10.0.pow(exp)
    }
    val step = nice(nice(hi.toDouble() - min, false) / (count - 1), true)
    val lo = floor(min / step) * step
    val top = ceil(hi / step) * step
    val ticks = generateSequence(lo) { it + step }.takeWhile { it <= top + step / 2 }.map { it.toFloat() }.toList()
    return Axis(lo.toFloat(), top.toFloat(), ticks)
}

private fun tickText(v: Float) = if (v == v.roundToInt().toFloat()) v.roundToInt().toString() else "%.1f".format(v)

/** Which of `n` evenly spread points (or bars) a tap at `x` is nearest. */
private fun nearest(x: Float, left: Float, right: Float, n: Int, bars: Boolean): Int {
    if (n <= 1) return 0
    val i = if (bars) ((x - left) / ((right - left) / n)).toInt() else ((x - left) / ((right - left) / (n - 1))).roundToInt()
    return i.coerceIn(0, n - 1)
}

/** Where the plot sits inside the chart, the axis labels around it. */
private class Plot(val left: Float, val top: Float, val right: Float, val bottom: Float, val axis: Axis) {
    fun y(v: Float) = bottom - (bottom - top) * (v - axis.min) / (axis.max - axis.min)
}

private class ChartText(measurer: TextMeasurer, axis: Axis, xLabels: List<String>) {
    private val style = TextStyle(fontSize = 10.sp, color = Palette.muted)
    val ticks: List<TextLayoutResult> = axis.ticks.map { measurer.measure(tickText(it), style) }
    val xs: List<TextLayoutResult> = xLabels.map { measurer.measure(it, style) }
    val gutter = ticks.maxOfOrNull { it.size.width } ?: 0
    val labelHeight = (xs + ticks).maxOfOrNull { it.size.height } ?: 0
}

private fun DrawScope.plotFor(text: ChartText, axis: Axis) = Plot(
    left = text.gutter + 6.dp.toPx(),
    top = text.labelHeight / 2f + 2.dp.toPx(),
    right = size.width - 4.dp.toPx(),
    bottom = size.height - text.labelHeight - 4.dp.toPx(),
    axis = axis,
)

/** The grid lines with their values, and every k-th label along the bottom. */
private fun DrawScope.drawFrame(plot: Plot, text: ChartText, xAt: (Int) -> Float) {
    plot.axis.ticks.forEachIndexed { k, t ->
        val y = plot.y(t)
        drawLine(Palette.border, Offset(plot.left, y), Offset(plot.right, y), strokeWidth = 1.dp.toPx())
        val label = text.ticks[k]
        drawText(label, topLeft = Offset(plot.left - 4.dp.toPx() - label.size.width, y - label.size.height / 2f))
    }
    val n = text.xs.size
    if (n == 0) return
    // Labels never crowd: every k-th, and always the last.
    val widest = text.xs.maxOf { it.size.width } + 6.dp.toPx()
    val room = (plot.right - plot.left) / max(1, n)
    val every = max(1, ceil(widest / room).toInt())
    for (i in 0 until n) {
        if (i % every != 0 && i != n - 1) continue
        if (i != n - 1 && n - 1 - i < every) continue
        val label = text.xs[i]
        val x = (xAt(i) - label.size.width / 2f).coerceIn(0f, size.width - label.size.width)
        drawText(label, topLeft = Offset(x, plot.bottom + 3.dp.toPx()))
    }
}

/**
 * Lines over a shared run of points: the scores move by move, accuracy game
 * by game. Tapping picks the nearest point.
 */
@Composable
fun LineChart(
    series: List<Series>,
    modifier: Modifier = Modifier,
    xLabels: List<String> = emptyList(),
    yMin: Float? = null,
    yMax: Float? = null,
    selected: Int? = null,
    onSelect: ((Int) -> Unit)? = null,
    height: Dp = 180.dp,
) {
    val n = series.maxOfOrNull { it.values.size } ?: 0
    val values = series.flatMap { s -> s.values.filterNotNull() }
    val axis = remember(values, yMin, yMax) {
        niceAxis(yMin ?: minOf(0f, values.minOrNull() ?: 0f), yMax ?: (values.maxOrNull() ?: 1f))
    }
    val measurer = rememberTextMeasurer()
    val text = remember(axis, xLabels) { ChartText(measurer, axis, xLabels) }
    val gutter = with(LocalDensity.current) { (text.gutter + 6.dp.toPx()) to 4.dp.toPx() }
    val select by rememberUpdatedState(onSelect)
    val count by rememberUpdatedState(n)
    Column(modifier) {
        Canvas(
            Modifier.fillMaxWidth().height(height).pointerInput(Unit) {
                detectTapGestures { p ->
                    select?.invoke(nearest(p.x, gutter.first, size.width - gutter.second, count, bars = false))
                }
            },
        ) {
            val plot = plotFor(text, axis)
            fun x(i: Int) = if (n <= 1) (plot.left + plot.right) / 2 else plot.left + (plot.right - plot.left) * i / (n - 1)
            drawFrame(plot, text, ::x)
            selected?.takeIf { it in 0 until n }?.let {
                drawLine(Palette.muted.copy(alpha = 0.6f), Offset(x(it), plot.top), Offset(x(it), plot.bottom), strokeWidth = 1.dp.toPx())
            }
            for (s in series) {
                val path = Path()
                var pen = false
                s.values.forEachIndexed { i, v ->
                    if (v == null) {
                        pen = false
                    } else {
                        if (pen) path.lineTo(x(i), plot.y(v)) else path.moveTo(x(i), plot.y(v))
                        pen = true
                    }
                }
                val dash = if (s.dashed) PathEffect.dashPathEffect(floatArrayOf(6.dp.toPx(), 4.dp.toPx())) else null
                drawPath(path, s.color, style = Stroke(2.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round, pathEffect = dash))
                if (s.dashed) continue
                s.values.forEachIndexed { i, v ->
                    if (v == null) return@forEachIndexed
                    val big = i == selected
                    if (big || n <= 40 || s.values.count { it != null } == 1) {
                        drawCircle(s.color, radius = (if (big) 4.5f else 2.5f) * density, center = Offset(x(i), plot.y(v)))
                    }
                }
            }
        }
        val legend = series.filter { it.label.isNotEmpty() }
        if (legend.isNotEmpty()) ChartLegend(legend.map { it.label to it.color })
    }
}

/** One bar per value, each in its own colour: move ratings, say. Tapping picks a bar. */
@Composable
fun BarChart(
    values: List<Float>,
    colors: List<Color>,
    modifier: Modifier = Modifier,
    xLabels: List<String> = emptyList(),
    yMax: Float = 100f,
    selected: Int? = null,
    onSelect: ((Int) -> Unit)? = null,
    height: Dp = 140.dp,
) {
    val n = values.size
    val axis = remember(yMax) { niceAxis(0f, yMax) }
    val measurer = rememberTextMeasurer()
    val text = remember(axis, xLabels) { ChartText(measurer, axis, xLabels) }
    val gutter = with(LocalDensity.current) { (text.gutter + 6.dp.toPx()) to 4.dp.toPx() }
    val select by rememberUpdatedState(onSelect)
    val count by rememberUpdatedState(n)
    Canvas(
        modifier.fillMaxWidth().height(height).pointerInput(Unit) {
            detectTapGestures { p ->
                if (count > 0) select?.invoke(nearest(p.x, gutter.first, size.width - gutter.second, count, bars = true))
            }
        },
    ) {
        val plot = plotFor(text, axis)
        val slot = (plot.right - plot.left) / max(1, n)
        fun center(i: Int) = plot.left + slot * (i + 0.5f)
        drawFrame(plot, text, ::center)
        val width = (slot * 0.7f).coerceAtMost(28.dp.toPx())
        values.forEachIndexed { i, v ->
            val top = plot.y(v.coerceIn(axis.min, axis.max))
            val corner = CornerRadius(minOf(width / 3, 4.dp.toPx()))
            // A zero still shows as a sliver, so it reads as a bad move, not a missing one.
            val h = (plot.bottom - top).coerceAtLeast(2.dp.toPx())
            val topLeft = Offset(center(i) - width / 2, plot.bottom - h)
            drawRoundRect(colors.getOrElse(i) { Palette.primary }, topLeft, Size(width, h), corner)
            if (i == selected) {
                drawRoundRect(Palette.text, topLeft - Offset(2.dp.toPx(), 2.dp.toPx()), Size(width + 4.dp.toPx(), h + 2.dp.toPx()), corner, style = Stroke(1.5f.dp.toPx()))
            }
        }
    }
}

@Composable
fun ChartLegend(items: List<Pair<String, Color>>, modifier: Modifier = Modifier) {
    FlowRow(modifier.padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        items.forEach { (label, color) ->
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(9.dp).background(color, CircleShape))
                Spacer(Modifier.width(4.dp))
                Text(label, fontSize = 12.sp, color = Palette.muted)
            }
        }
    }
}

/** Wins, ties and losses as one bar split in proportion. */
@Composable
fun ResultBar(wins: Int, ties: Int, losses: Int, modifier: Modifier = Modifier) {
    val total = wins + ties + losses
    if (total == 0) return
    Row(modifier.fillMaxWidth().height(10.dp)) {
        listOf(wins to Palette.ratingGood, ties to Palette.ratingOk, losses to Palette.ratingBad).forEach { (count, color) ->
            if (count > 0) Box(Modifier.weight(count.toFloat()).height(10.dp).background(color))
        }
    }
}
