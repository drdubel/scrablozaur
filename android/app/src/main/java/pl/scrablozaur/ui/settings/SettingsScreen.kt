package pl.scrablozaur.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
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
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.core.AppSettings
import pl.scrablozaur.core.Playback
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settings: AppSettings,
    /** Games in the history; null while it is being read. */
    historySize: Int?,
    version: String,
    onChange: ((AppSettings) -> AppSettings) -> Unit,
    onClearHistory: () -> Unit,
    onBack: () -> Unit,
) {
    var confirmClear by remember { mutableStateOf(false) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.settings_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier.padding(padding).fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            Section(R.string.settings_section_player)
            OutlinedTextField(
                value = settings.playerName,
                onValueChange = { v -> onChange { it.copy(playerName = v.take(20)) } },
                label = { Text(stringResource(R.string.your_name)) },
                placeholder = { Text(stringResource(R.string.default_human_name)) },
                singleLine = true,
                modifier = Modifier.fillMaxWidth(),
            )
            Help(R.string.settings_name_help)

            Section(R.string.settings_section_game)
            SwitchRow(R.string.settings_show_rating, R.string.settings_show_rating_help, settings.showMoveRating) { on ->
                onChange { it.copy(showMoveRating = on) }
            }
            Text(stringResource(R.string.settings_hint_order))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(HintOrder.SCORE to R.string.sort_score, HintOrder.SMART to R.string.sort_smart, HintOrder.SIM to R.string.sort_sim)
                    .forEach { (order, label) ->
                        FilterChip(
                            selected = settings.hintOrder == order,
                            onClick = { onChange { it.copy(hintOrder = order) } },
                            label = { Text(stringResource(label)) },
                        )
                    }
            }
            Help(R.string.settings_hint_order_help)
            Text(stringResource(R.string.settings_playback))
            FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                listOf(Playback.Slow to R.string.speed_slow, Playback.Normal to R.string.speed_normal, Playback.Fast to R.string.speed_fast)
                    .forEach { (speed, label) ->
                        FilterChip(
                            selected = settings.playback == speed,
                            onClick = { onChange { it.copy(playback = speed) } },
                            label = { Text(stringResource(label)) },
                        )
                    }
            }
            SwitchRow(R.string.settings_keep_screen_on, R.string.settings_keep_screen_on_help, settings.keepScreenOn) { on ->
                onChange { it.copy(keepScreenOn = on) }
            }

            Section(R.string.settings_section_dictionary)
            SwitchRow(R.string.settings_definitions, R.string.settings_definitions_help, settings.onlineDefinitions) { on ->
                onChange { it.copy(onlineDefinitions = on) }
            }

            Section(R.string.settings_section_data)
            if (historySize != null) {
                Text(pluralStringResource(R.plurals.settings_history_size, historySize, historySize), color = Palette.muted)
            }
            OutlinedButton(
                onClick = { confirmClear = true },
                enabled = (historySize ?: 0) > 0,
                colors = outlinedButtonColors(danger = true),
            ) { Text(stringResource(R.string.settings_clear_history)) }

            Section(R.string.settings_section_about)
            Text(stringResource(R.string.app_name), fontWeight = FontWeight.SemiBold)
            Text(stringResource(R.string.settings_version, version), color = Palette.muted)
        }
    }
    if (confirmClear) {
        AlertDialog(
            onDismissRequest = { confirmClear = false },
            text = { Text(stringResource(R.string.settings_clear_confirm)) },
            confirmButton = {
                TextButton(onClick = { confirmClear = false; onClearHistory() }) {
                    Text(stringResource(R.string.delete), color = Palette.danger)
                }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text(stringResource(R.string.cancel)) } },
        )
    }
}

@Composable
private fun Section(title: Int) {
    Column(Modifier.padding(top = 8.dp)) {
        Text(stringResource(title), fontWeight = FontWeight.Bold, fontSize = 15.sp, color = Palette.primary)
        HorizontalDivider(Modifier.padding(top = 4.dp), color = Palette.border)
    }
}

@Composable
private fun Help(text: Int) {
    Text(stringResource(text), color = Palette.muted, fontSize = 13.sp)
}

@Composable
private fun SwitchRow(title: Int, help: Int, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().clickable { onChange(!checked) }.padding(vertical = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(stringResource(title))
            Text(stringResource(help), color = Palette.muted, fontSize = 13.sp)
        }
        Spacer(Modifier.width(12.dp))
        Switch(checked = checked, onCheckedChange = onChange)
    }
}
