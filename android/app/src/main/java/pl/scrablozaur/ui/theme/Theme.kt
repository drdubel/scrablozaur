package pl.scrablozaur.ui.theme

import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

/** The web's palette (`web/static/style.css`), so both apps look like one product. */
object Palette {
    val background = Color(0xFFFAFBFA)
    val surface = Color(0xFFFFFFFF)
    val border = Color(0xFFE1E5E2)
    val text = Color(0xFF202923)
    val muted = Color(0xFF68716B)
    val primary = Color(0xFF315C45)
    val primaryDark = Color(0xFF254735)
    val danger = Color(0xFFA54444)

    val tw = Color(0xFFE9C9C9)
    val twText = Color(0xFF773A3A)
    val dw = Color(0xFFF2E1D2)
    val dwText = Color(0xFF815637)
    val tl = Color(0xFFC5DBE5)
    val tlText = Color(0xFF345C70)
    val dl = Color(0xFFE0EAF0)
    val dlText = Color(0xFF466678)
    val empty = Color(0xFFF0F3F0)
    val placed = Color(0xFFFFFFFF)
    val placedBorder = Color(0xFFBBC6BD)
    val blankAccent = Color(0xFF6B7FD7)

    /** Per-seat tile tint and border, by seat index. */
    val owners = listOf(
        Color(0xFFFFFDF4) to Color(0xFFD8CFA9),
        Color(0xFFF0F7FB) to Color(0xFFB6CCDA),
        Color(0xFFEFF8F1) to Color(0xFFB8D3C0),
        Color(0xFFF7F2FA) to Color(0xFFCEC0DB),
    )

    val typingNew = Color(0xFFFEF3C7)
    val typingNewText = Color(0xFF92400E)
    val cursor = Color(0xFFF59E0B)
    val valid = Color(0xFFD1FAE5)
    val validText = Color(0xFF065F46)
    val invalid = Color(0xFFFEE2E2)
    val invalidText = Color(0xFF991B1B)
    val highlightNew = Color(0xFFD4EADC)
    val highlightNewText = Color(0xFF1A5C33)
    val highlightExisting = Color(0xFFDCEAF3)

    val ratingGood = Color(0xFF22C55E)
    val ratingOk = Color(0xFFF59E0B)
    val ratingBad = Color(0xFFEF4444)
}

@Composable
fun ScrablozaurTheme(content: @Composable () -> Unit) {
    // Every role is set, not just the obvious ones: Material's defaults are
    // derived from its baseline purple, which leaks into chips, sliders and
    // containers through the roles left unset.
    MaterialTheme(
        colorScheme = lightColorScheme(
            primary = Palette.primary,
            onPrimary = Color.White,
            primaryContainer = Palette.highlightNew,
            onPrimaryContainer = Palette.highlightNewText,
            secondary = Palette.primaryDark,
            onSecondary = Color.White,
            secondaryContainer = Palette.highlightNew,
            onSecondaryContainer = Palette.highlightNewText,
            tertiary = Palette.tlText,
            onTertiary = Color.White,
            tertiaryContainer = Palette.tl,
            onTertiaryContainer = Palette.tlText,
            background = Palette.background,
            onBackground = Palette.text,
            surface = Palette.surface,
            onSurface = Palette.text,
            surfaceVariant = Palette.empty,
            onSurfaceVariant = Palette.muted,
            surfaceTint = Palette.primary,
            surfaceContainerLowest = Color.White,
            surfaceContainerLow = Color(0xFFF7F9F7),
            surfaceContainer = Color(0xFFF2F5F2),
            surfaceContainerHigh = Color(0xFFECF0EC),
            surfaceContainerHighest = Color(0xFFE6EBE7),
            outline = Palette.placedBorder,
            outlineVariant = Palette.border,
            error = Palette.danger,
            onError = Color.White,
        ),
        content = content,
    )
}

/** Outlined buttons in the text colour. The web's ghost buttons are muted grey,
 *  which reads as disabled on a touchscreen, where no hover state answers a
 *  finger; a disabled button stays visibly lighter than this. */
@Composable
fun outlinedButtonColors(danger: Boolean = false) =
    ButtonDefaults.outlinedButtonColors(contentColor = if (danger) Palette.danger else Palette.text)
