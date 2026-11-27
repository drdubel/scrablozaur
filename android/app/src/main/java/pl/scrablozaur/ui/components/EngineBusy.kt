package pl.scrablozaur.ui.components

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.delay
import pl.scrablozaur.ui.theme.Palette

/**
 * A spinner for while the engine works, stepping ten times a second instead
 * of animating every frame. A smooth spinner redraws the whole screen at the
 * display's refresh rate -- on a 120 Hz Galaxy A54 that is most of a fast
 * core, taken from the very computation it is waiting on.
 */
@Composable
fun EngineBusy(modifier: Modifier = Modifier.size(18.dp), color: Color = Palette.primary) {
    var step by remember { mutableIntStateOf(0) }
    LaunchedEffect(Unit) {
        while (true) {
            delay(100)
            step = (step + 1) % 12
        }
    }
    // Read only while drawing: a step redraws this spinner and recomposes nothing.
    Canvas(modifier) {
        val stroke = size.minDimension * 0.14f
        drawArc(
            color = color,
            startAngle = step * 30f,
            sweepAngle = 270f,
            useCenter = false,
            topLeft = Offset(stroke / 2, stroke / 2),
            size = Size(size.width - stroke, size.height - stroke),
            style = Stroke(stroke, cap = StrokeCap.Round),
        )
    }
}
