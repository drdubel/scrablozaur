package pl.scrablozaur.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.ui.theme.Palette

/** A column of a `DataTable`: its title, its share of the width, and whether it hugs the right edge. */
data class TableColumn(val title: String, val weight: Float, val end: Boolean = false)

/** One cell's text, with an optional seat dot before it. */
data class TableCell(
    val text: String,
    val color: Color = Palette.text,
    val bold: Boolean = false,
    val dot: Color? = null,
)

/**
 * A small table: the history, the ranking, a game's moves. Not lazy -- every
 * table shown is bounded (ten best games, one game's turns), and it sits
 * inside screens that scroll as a whole.
 */
@Composable
fun DataTable(
    columns: List<TableColumn>,
    rows: List<List<TableCell>>,
    modifier: Modifier = Modifier,
    selected: Int? = null,
    onRowClick: ((Int) -> Unit)? = null,
) {
    // The same gap between header cells as between row cells keeps the columns aligned.
    val gap = Arrangement.spacedBy(8.dp)
    Column(modifier.fillMaxWidth()) {
        Row(Modifier.fillMaxWidth().padding(horizontal = 6.dp, vertical = 4.dp), horizontalArrangement = gap) {
            columns.forEach { c ->
                Text(
                    c.title,
                    Modifier.weight(c.weight),
                    fontSize = 11.sp,
                    color = Palette.muted,
                    textAlign = if (c.end) TextAlign.End else TextAlign.Start,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
            }
        }
        Box(Modifier.fillMaxWidth().height(1.dp).background(Palette.border))
        rows.forEachIndexed { i, row ->
            Row(
                Modifier
                    .fillMaxWidth()
                    .background(if (i == selected) Palette.highlightNew else Color.Transparent, RoundedCornerShape(6.dp))
                    .then(if (onRowClick != null) Modifier.clickable { onRowClick(i) } else Modifier)
                    .padding(horizontal = 6.dp, vertical = 7.dp),
                horizontalArrangement = gap,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                columns.forEachIndexed { k, c ->
                    val cell = row.getOrNull(k) ?: TableCell("")
                    Row(Modifier.weight(c.weight), verticalAlignment = Alignment.CenterVertically) {
                        if (c.end) Spacer(Modifier.weight(1f))
                        cell.dot?.let {
                            Box(Modifier.size(8.dp).background(it, CircleShape))
                            Spacer(Modifier.width(5.dp))
                        }
                        Text(
                            cell.text,
                            fontSize = 13.sp,
                            color = cell.color,
                            fontWeight = if (cell.bold) FontWeight.SemiBold else FontWeight.Normal,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis,
                            modifier = if (c.end) Modifier else Modifier.weight(1f, fill = false),
                        )
                    }
                }
            }
        }
    }
}

/** A labelled figure: one cell of a summary grid. */
@Composable
fun StatTile(label: String, value: String, modifier: Modifier = Modifier, color: Color = Palette.text, detail: String? = null) {
    Card(modifier, colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column(Modifier.fillMaxWidth().padding(horizontal = 10.dp, vertical = 8.dp)) {
            Text(label, fontSize = 11.sp, color = Palette.muted, maxLines = 1, overflow = TextOverflow.Ellipsis)
            Text(value, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis)
            detail?.let { Text(it, fontSize = 11.sp, color = Palette.muted, maxLines = 1, overflow = TextOverflow.Ellipsis) }
        }
    }
}

/** Tiles `perRow` to a row, the last row padded so every tile keeps its width. */
@Composable
fun TileGrid(tiles: List<@Composable (Modifier) -> Unit>, perRow: Int = 3) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        tiles.chunked(perRow).forEach { row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                row.forEach { tile -> tile(Modifier.weight(1f)) }
                repeat(perRow - row.size) { Spacer(Modifier.weight(1f)) }
            }
        }
    }
}

/** A section heading inside a scrolling screen. */
@Composable
fun SectionTitle(text: String, modifier: Modifier = Modifier) {
    Text(text, modifier.padding(top = 6.dp), fontWeight = FontWeight.SemiBold, fontSize = 16.sp)
}
