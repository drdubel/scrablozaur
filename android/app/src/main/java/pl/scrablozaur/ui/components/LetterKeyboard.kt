package pl.scrablozaur.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.Backspace
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.ui.theme.Palette

/**
 * A keyboard of exactly this language's tiles, in its collation order -- so a
 * Polish ź is one tap away whatever keyboard the phone has installed, and no
 * letter outside the alphabet can be typed at all.
 */
@Composable
fun LetterKeyboard(
    alphabet: String,
    modifier: Modifier = Modifier,
    extraKeys: String = "",
    perRow: Int = 11,
    onLetter: (Char) -> Unit,
    onBackspace: () -> Unit,
) {
    val keys = (alphabet + extraKeys).toList()
    val rows = keys.chunked(perRow)
    Column(modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(4.dp)) {
        rows.forEachIndexed { r, row ->
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                row.forEach { ch ->
                    Key(Modifier.weight(1f), onClick = { onLetter(ch) }) {
                        Text(if (ch == '?') "★" else ch.uppercase(), fontSize = 17.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
                if (r == rows.lastIndex) {
                    Key(Modifier.weight(1.6f), onClick = onBackspace) {
                        Icon(Icons.AutoMirrored.Filled.Backspace, contentDescription = stringResource(R.string.backspace))
                    }
                    repeat(perRow - row.size - 1) { Box(Modifier.weight(1f)) }
                }
            }
        }
    }
}

@Composable
private fun Key(modifier: Modifier, onClick: () -> Unit, content: @Composable () -> Unit) {
    Box(
        modifier
            .height(42.dp)
            .background(Palette.surface, RoundedCornerShape(6.dp))
            .clickable(onClick = onClick),
        contentAlignment = Alignment.Center,
    ) { content() }
}
