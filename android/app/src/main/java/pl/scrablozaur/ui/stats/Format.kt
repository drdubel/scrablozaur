package pl.scrablozaur.ui.stats

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringArrayResource
import androidx.compose.ui.res.stringResource
import pl.scrablozaur.R
import pl.scrablozaur.core.Opponent
import pl.scrablozaur.core.Result
import pl.scrablozaur.ui.theme.Palette
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import kotlin.math.roundToInt

// Wording shared by the statistics and a game's review.

private val dateTime = DateTimeFormatter.ofPattern("d.MM.yyyy, HH:mm")
private val dateOnly = DateTimeFormatter.ofPattern("d.MM.yyyy")

fun formatDateTime(ms: Long): String = dateTime.format(Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()))

fun formatDate(ms: Long): String = dateOnly.format(Instant.ofEpochMilli(ms).atZone(ZoneId.systemDefault()))

/** A computer as the level dial names it, e.g. "Analityk · 100%"; a person by `name`, or the name the key holds. */
@Composable
fun opponentLabel(o: Opponent, name: String? = null): String {
    o.person?.let { return name ?: it }
    val names = stringArrayResource(R.array.level_names)
    return stringResource(R.string.opponent_label, names[(o.difficulty - 1).coerceIn(0, names.lastIndex)], o.strength)
}

@Composable
fun resultLabel(r: Result): String = stringResource(
    when (r) {
        Result.Win -> R.string.result_win
        Result.Loss -> R.string.result_loss
        Result.Tie -> R.string.result_tie
    },
)

@Composable
fun resultBadge(r: Result): String = stringResource(
    when (r) {
        Result.Win -> R.string.result_short_win
        Result.Loss -> R.string.result_short_loss
        Result.Tie -> R.string.result_short_tie
    },
)

fun resultColor(r: Result): Color = when (r) {
    Result.Win -> Palette.ratingGood
    Result.Loss -> Palette.ratingBad
    Result.Tie -> Palette.ratingOk
}

@Composable
fun percent(x: Double): String = stringResource(R.string.percent, x.roundToInt())

/** A signed whole number: "+12", "−3", "0". */
fun signed(x: Double): String {
    val n = x.roundToInt()
    return when {
        n > 0 -> "+$n"
        n < 0 -> "−${-n}"
        else -> "0"
    }
}
