package pl.scrablozaur.ui.components

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.ui.theme.Palette

/**
 * A player's tiles. While a word is being typed, a tap plays that tile onto the
 * board (the rack acts as a keyboard); otherwise a tap picks it for exchange.
 */
@Composable
fun RackView(
    letters: String,
    blank: Char,
    points: Map<String, UInt>,
    used: Set<Int>,
    selected: Set<Int>,
    placing: Boolean,
    modifier: Modifier = Modifier,
    onTap: (Int) -> Unit,
) {
    Row(
        modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(6.dp, Alignment.CenterHorizontally),
    ) {
        letters.forEachIndexed { i, ch ->
            val isBlank = ch == blank
            val isUsed = i in used
            val shape = RoundedCornerShape(8.dp)
            val border = when {
                isUsed -> BorderStroke(1.5.dp, Palette.border)
                i in selected -> BorderStroke(3.dp, Palette.primary)
                placing -> BorderStroke(2.dp, Palette.primary)
                isBlank -> BorderStroke(2.dp, Palette.muted)
                else -> BorderStroke(2.dp, Palette.placedBorder)
            }
            Box(
                Modifier
                    // Share the row on a narrow screen, but stop growing at a
                    // tile's natural size on a wide one: `fill = false` lets the cap win.
                    .weight(1f, fill = false)
                    .widthIn(max = 56.dp)
                    .aspectRatio(1f)
                    .background(
                        when {
                            isUsed -> Palette.empty
                            i in selected -> Palette.highlightNew
                            isBlank -> Palette.background
                            else -> Palette.placed
                        },
                        shape,
                    )
                    .border(border, shape)
                    .clickable(enabled = !isUsed) { onTap(i) },
            ) {
                if (!isUsed) {
                    Text(
                        if (isBlank) "★" else ch.uppercase(),
                        Modifier.align(Alignment.Center),
                        fontSize = 22.sp,
                        fontWeight = FontWeight.Bold,
                        color = if (isBlank) Palette.muted else Palette.text,
                    )
                    if (!isBlank) {
                        Text(
                            (points[ch.toString()] ?: 0u).toString(),
                            Modifier.align(Alignment.BottomEnd).padding(end = 4.dp, bottom = 1.dp),
                            fontSize = 11.sp,
                            color = Palette.muted,
                        )
                    }
                }
            }
        }
    }
}
