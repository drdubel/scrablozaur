package pl.scrablozaur.ui.setup

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.engine.Mode
import pl.scrablozaur.engine.PlayerSetup
import pl.scrablozaur.ui.game.NewGame
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors

/** The bot settings the web's "Zaawansowane" section holds. */
data class BotOptions(val strength: Int = 100, val neural: Boolean = true, val simulation: Boolean = true) {
    /** `_readComputerOptions`: the switches pick the engine, capped by the language. */
    fun level(lang: LanguageMeta): Int {
        val neural = lang.hasLeaveNet && neural
        val level = if (neural) (if (simulation) 10 else 9) else 8
        return minOf(level, if (lang.hasLeaveNet) 10 else 8)
    }
}

enum class SetupMode { Competitive, Sandbox }

data class SeatConfig(val name: String, val isComputer: Boolean, val advanced: BotOptions? = null)

/** A benchmark as the setup screen describes it. */
data class BenchmarkRequest(val language: String, val players: List<PlayerSetup>, val games: Int)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SetupScreen(
    languages: List<LanguageMeta>,
    defaultLanguage: String,
    canContinue: Boolean,
    onContinue: () -> Unit,
    onStart: (NewGame) -> Unit,
    onBenchmark: (BenchmarkRequest) -> Unit,
    onScan: () -> Unit,
) {
    var mode by remember { mutableStateOf(SetupMode.Competitive) }
    var languageCode by remember { mutableStateOf(defaultLanguage) }
    var humanName by remember { mutableStateOf("") }
    var auto by remember { mutableStateOf(false) }
    var count by remember { mutableStateOf(2) }
    var manualSeats by remember { mutableStateOf(List(4) { SeatConfig("", isComputer = it == 1) }) }
    var autoSeats by remember { mutableStateOf(List(4) { SeatConfig("", isComputer = true) }) }
    var options by remember { mutableStateOf(BotOptions()) }
    var showAdvanced by remember { mutableStateOf(false) }
    var benchmarkGames by remember { mutableStateOf("20") }
    var error by remember { mutableStateOf<String?>(null) }

    val lang = languages.firstOrNull { it.code == languageCode } ?: languages.first()
    val defaultName = stringResource(R.string.default_human_name)
    val computerName = stringResource(R.string.computer)
    val seatNames = (1..4).map { stringResource(R.string.player_n, it) }
    val oneComputer = stringResource(R.string.exactly_one_computer)

    fun setup(name: String, isComputer: Boolean, o: BotOptions) =
        PlayerSetup(name, isComputer, o.level(lang).toUByte(), o.strength.toUByte())

    fun autoPlayers() = autoSeats.take(count).mapIndexed { i, s ->
        setup(s.name.trim().ifEmpty { seatNames[i] }, true, s.advanced ?: options)
    }

    fun start() {
        error = null
        when {
            mode == SetupMode.Competitive -> onStart(
                NewGame(
                    lang.code, Mode.COMPETITIVE,
                    listOf(
                        PlayerSetup(humanName.trim().ifEmpty { defaultName }, false, 5u, 100u),
                        setup(computerName, true, options),
                    ),
                ),
            )
            auto -> onStart(NewGame(lang.code, Mode.SANDBOX_AUTO, autoPlayers()))
            else -> {
                val seats = manualSeats.take(count)
                if (seats.count { it.isComputer } != 1) {
                    error = oneComputer
                    return
                }
                onStart(
                    NewGame(
                        lang.code, Mode.SANDBOX,
                        seats.mapIndexed { i, s -> setup(s.name.trim().ifEmpty { seatNames[i] }, s.isComputer, options) },
                    ),
                )
            }
        }
    }

    Scaffold(topBar = { TopAppBar(title = { Text(stringResource(R.string.new_game), fontWeight = FontWeight.Bold) }) }) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            if (canContinue) {
                Button(onClick = onContinue, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.continue_game)) }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                ModeCard(stringResource(R.string.mode_competitive), mode == SetupMode.Competitive, Modifier.weight(1f)) { mode = SetupMode.Competitive }
                ModeCard(stringResource(R.string.mode_sandbox), mode == SetupMode.Sandbox, Modifier.weight(1f)) { mode = SetupMode.Sandbox }
                ModeCard(stringResource(R.string.mode_scan), false, Modifier.weight(1f), onScan)
            }
            if (languages.size > 1) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.language), color = Palette.muted)
                    languages.forEach { l ->
                        FilterChip(selected = l.code == lang.code, onClick = { languageCode = l.code }, label = { Text("${l.flag} ${l.name}") })
                    }
                }
            }

            if (mode == SetupMode.Competitive) {
                OutlinedTextField(
                    value = humanName,
                    onValueChange = { humanName = it.take(20) },
                    label = { Text(stringResource(R.string.your_name)) },
                    placeholder = { Text(defaultName) },
                    singleLine = true,
                    modifier = Modifier.fillMaxWidth(),
                )
            } else {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    FilterChip(selected = !auto, onClick = { auto = false }, label = { Text(stringResource(R.string.sandbox_manual)) })
                    FilterChip(selected = auto, onClick = { auto = true }, label = { Text(stringResource(R.string.sandbox_auto)) })
                }
                Text(stringResource(if (auto) R.string.sandbox_auto_desc else R.string.sandbox_manual_desc), color = Palette.muted, fontSize = 13.sp)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Text(stringResource(R.string.player_count), color = Palette.muted)
                    (2..4).forEach { n -> FilterChip(selected = count == n, onClick = { count = n }, label = { Text("$n") }) }
                }
                for (i in 0 until count) {
                    if (auto) {
                        AutoSeatRow(i, autoSeats[i], options, lang, seatNames[i]) { seat ->
                            autoSeats = autoSeats.toMutableList().also { it[i] = seat }
                        }
                    } else {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("${i + 1}.", Modifier.width(24.dp))
                            OutlinedTextField(
                                value = manualSeats[i].name,
                                onValueChange = { v -> manualSeats = manualSeats.toMutableList().also { it[i] = it[i].copy(name = v.take(20)) } },
                                placeholder = { Text(seatNames[i]) },
                                singleLine = true,
                                modifier = Modifier.weight(1f),
                            )
                            RadioButton(
                                selected = manualSeats[i].isComputer,
                                onClick = { manualSeats = manualSeats.mapIndexed { j, s -> s.copy(isComputer = j == i) } },
                            )
                            Text(computerName)
                        }
                    }
                }
                if (auto) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                        OutlinedTextField(
                            value = benchmarkGames,
                            onValueChange = { v -> benchmarkGames = v.filter(Char::isDigit).take(4) },
                            label = { Text(stringResource(R.string.games_count)) },
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                            modifier = Modifier.width(140.dp),
                        )
                        OutlinedButton(colors = outlinedButtonColors(), onClick = {
                            val games = benchmarkGames.toIntOrNull()?.coerceAtLeast(1) ?: 20
                            onBenchmark(BenchmarkRequest(lang.code, autoPlayers(), games))
                        }) { Text(stringResource(R.string.run_benchmark)) }
                    }
                }
            }

            TextButton(onClick = { showAdvanced = !showAdvanced }) {
                Text((if (showAdvanced) "▾ " else "▸ ") + stringResource(R.string.advanced))
            }
            if (showAdvanced) BotOptionsEditor(options, lang) { options = it }

            error?.let { Text(it, color = Palette.danger) }
            Button(onClick = ::start, modifier = Modifier.fillMaxWidth()) { Text(stringResource(R.string.play), fontSize = 18.sp) }
        }
    }
}

@Composable
private fun ModeCard(label: String, active: Boolean, modifier: Modifier, onClick: () -> Unit) {
    Card(
        onClick = onClick,
        modifier = modifier,
        colors = CardDefaults.cardColors(containerColor = if (active) Palette.primary else Palette.surface),
    ) {
        Text(
            label,
            Modifier.fillMaxWidth().padding(vertical = 18.dp),
            color = if (active) Palette.surface else Palette.text,
            fontWeight = FontWeight.SemiBold,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
    }
}

@Composable
fun BotOptionsEditor(options: BotOptions, lang: LanguageMeta, onChange: (BotOptions) -> Unit) {
    Column {
        Text("${stringResource(R.string.computer_strength)} ${options.strength}%")
        Slider(
            value = options.strength.toFloat(),
            onValueChange = { onChange(options.copy(strength = it.toInt().coerceIn(1, 100))) },
            valueRange = 1f..100f,
        )
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(checked = options.neural, onCheckedChange = { onChange(options.copy(neural = it)) }, enabled = lang.hasLeaveNet)
            Text(stringResource(R.string.neural_net))
        }
        Row(verticalAlignment = Alignment.CenterVertically) {
            Checkbox(
                checked = options.simulation,
                onCheckedChange = { onChange(options.copy(simulation = it)) },
                enabled = lang.hasLeaveNet && options.neural,
            )
            Text(stringResource(R.string.simulation))
        }
        Text(
            stringResource(
                when {
                    !lang.hasLeaveNet -> R.string.note_no_model
                    options.level(lang) == 10 -> R.string.note_sim
                    options.level(lang) == 9 -> R.string.note_smart
                    else -> R.string.note_ranked
                },
            ),
            color = Palette.muted,
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

@Composable
private fun AutoSeatRow(i: Int, seat: SeatConfig, shared: BotOptions, lang: LanguageMeta, placeholder: String, onChange: (SeatConfig) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Column {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("${i + 1}.", Modifier.width(24.dp))
            OutlinedTextField(
                value = seat.name,
                onValueChange = { onChange(seat.copy(name = it.take(20))) },
                placeholder = { Text(placeholder) },
                singleLine = true,
                modifier = Modifier.weight(1f),
            )
            TextButton(onClick = { open = !open }) { Text(stringResource(R.string.advanced)) }
        }
        if (open) {
            Column(Modifier.padding(start = 24.dp)) {
                BotOptionsEditor(seat.advanced ?: shared, lang) { onChange(seat.copy(advanced = it)) }
                if (seat.advanced != null) {
                    TextButton(onClick = { onChange(seat.copy(advanced = null)) }) { Text(stringResource(R.string.defaults)) }
                }
            }
        }
    }
}
