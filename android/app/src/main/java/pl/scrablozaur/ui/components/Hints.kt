package pl.scrablozaur.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.engine.Hint
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.ui.game.Cell
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors
import kotlin.math.floor

// Pieces the game screen and the scan assistant share.

fun Hint.cells(): List<Cell> = (0 until word.length).map {
    if (horizontal) Cell(row.toInt(), col.toInt() + it) else Cell(row.toInt() + it, col.toInt())
}

/** A ranking value as the web prints it: signed, and no decimals when whole. */
fun formatValue(v: Double): String {
    val text = if (v == floor(v)) v.toLong().toString() else "%.1f".format(v)
    return if (v > 0) "+$text" else text
}

@Composable
fun SmallButton(
    label: Int,
    onClick: () -> Unit,
    enabled: Boolean = true,
    busy: Boolean = false,
    danger: Boolean = false,
) {
    OutlinedButton(
        onClick = onClick,
        enabled = enabled,
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 6.dp),
        colors = outlinedButtonColors(danger),
    ) {
        if (busy) {
            EngineBusy(Modifier.size(14.dp))
            Spacer(Modifier.width(6.dp))
        }
        Text(stringResource(label), fontSize = 14.sp)
    }
}

@Composable
fun OrderChips(order: HintOrder, hasLeaveNet: Boolean, enabled: Boolean, onChange: (HintOrder) -> Unit, modifier: Modifier = Modifier) {
    // Strateg and Analityk consult the leave net, so a language without one
    // only offers plain score order -- as the web hides those buttons.
    val options = buildList {
        add(HintOrder.SCORE to R.string.sort_score)
        if (hasLeaveNet) {
            add(HintOrder.SMART to R.string.sort_smart)
            add(HintOrder.SIM to R.string.sort_sim)
        }
    }
    Row(modifier, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        options.forEach { (o, label) ->
            FilterChip(selected = order == o, onClick = { onChange(o) }, enabled = enabled, label = { Text(stringResource(label)) })
        }
    }
}

/** Tap a play to see it on the board, "Połóż" (where given) to play it. */
@Composable
fun HintList(
    hints: List<Hint>,
    order: HintOrder,
    selected: Hint?,
    enabled: Boolean,
    onSelect: (Hint) -> Unit,
    onPlace: ((Hint) -> Unit)? = null,
    empty: Int = R.string.no_moves,
) {
    if (hints.isEmpty()) {
        Text(stringResource(empty), color = Palette.muted)
        return
    }
    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
        hints.forEachIndexed { i, h ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(if (h == selected) Palette.highlightNew else Palette.surface, RoundedCornerShape(8.dp))
                    .clickable { onSelect(h) }
                    .padding(start = 10.dp, end = 2.dp),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                Text("${i + 1}.", color = Palette.muted, fontSize = 13.sp)
                Column(Modifier.weight(1f)) {
                    Text(h.word.uppercase(), fontWeight = FontWeight.Bold, fontSize = 16.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                    val position = stringResource(R.string.hint_position, h.row.toInt(), h.col.toInt(), if (h.horizontal) "→" else "↓")
                    // The ordering value only says something when it is not the score.
                    val value = if (order == HintOrder.SCORE) "" else " · ${formatValue(h.value)}"
                    Text(position + value, fontSize = 12.sp, color = Palette.muted)
                }
                Text(stringResource(R.string.points_short, h.score.toInt()), fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                if (onPlace != null) {
                    TextButton(onClick = { onPlace(h) }, enabled = enabled) { Text(stringResource(R.string.place)) }
                } else {
                    Spacer(Modifier.width(8.dp))
                }
            }
        }
    }
}
