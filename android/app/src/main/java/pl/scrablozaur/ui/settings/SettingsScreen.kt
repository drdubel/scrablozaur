package pl.scrablozaur.ui.settings

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
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
import androidx.compose.foundation.text.KeyboardOptions
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
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import pl.scrablozaur.R
import pl.scrablozaur.core.AppSettings
import pl.scrablozaur.core.ImportSummary
import pl.scrablozaur.core.Leaderboard
import pl.scrablozaur.core.LeaderboardException
import pl.scrablozaur.core.Playback
import pl.scrablozaur.core.UploadResult
import pl.scrablozaur.engine.HintOrder
import pl.scrablozaur.ui.stats.leaderboardError
import pl.scrablozaur.ui.theme.Palette
import pl.scrablozaur.ui.theme.outlinedButtonColors
import java.time.LocalDate

/** What the settings do with the history and the ranking server, beyond changing a setting. */
class SettingsActions(
    val onChange: ((AppSettings) -> AppSettings) -> Unit,
    val onClearHistory: () -> Unit,
    /** The history as a file's text. */
    val onExport: suspend () -> String,
    /** A history file's lines, merged into the history. */
    val onImport: suspend (List<String>) -> ImportSummary,
    /** Every game in the history, sent to the ranking server. */
    val onUpload: suspend () -> UploadResult,
)

/** A line under a section saying how its last action went. */
private sealed interface Outcome {
    data class Exported(val games: Int) : Outcome
    data class Imported(val summary: ImportSummary) : Outcome
    data class Uploaded(val result: UploadResult) : Outcome
    data class Failed(val message: String) : Outcome
    data class ServerFailed(val error: LeaderboardException) : Outcome
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    settings: AppSettings,
    /** Games in the history; null while it is being read. */
    historySize: Int?,
    version: String,
    actions: SettingsActions,
    onBack: () -> Unit,
) {
    var confirmClear by remember { mutableStateOf(false) }
    var backupOutcome by remember { mutableStateOf<Outcome?>(null) }
    var onlineOutcome by remember { mutableStateOf<Outcome?>(null) }
    var uploading by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()
    val context = LocalContext.current
    val onChange = actions.onChange

    // The system's file picker: the player chooses where the copy goes, and
    // can put it on a drive or a card that outlives the phone.
    val exporter = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("application/x-ndjson")) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            backupOutcome = runCatching {
                val text = actions.onExport()
                withContext(Dispatchers.IO) {
                    checkNotNull(context.contentResolver.openOutputStream(uri, "wt")) { "no file" }.use { it.write(text.toByteArray()) }
                }
                Outcome.Exported(text.lineSequence().count { it.isNotBlank() })
            }.getOrElse { Outcome.Failed(it.message ?: it.toString()) }
        }
    }
    val importer = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri == null) return@rememberLauncherForActivityResult
        scope.launch {
            backupOutcome = runCatching {
                val lines = withContext(Dispatchers.IO) {
                    checkNotNull(context.contentResolver.openInputStream(uri)) { "no file" }.bufferedReader().use { it.readLines() }
                }
                Outcome.Imported(actions.onImport(lines))
            }.getOrElse { Outcome.Failed(it.message ?: it.toString()) }
        }
    }

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
            Help(R.string.settings_backup_help)
            FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedButton(
                    onClick = { exporter.launch("scrablozaur-${LocalDate.now()}.jsonl") },
                    enabled = (historySize ?: 0) > 0,
                    colors = outlinedButtonColors(),
                ) { Text(stringResource(R.string.settings_export)) }
                OutlinedButton(onClick = { importer.launch(arrayOf("*/*")) }, enabled = historySize != null, colors = outlinedButtonColors()) {
                    Text(stringResource(R.string.settings_import))
                }
            }
            backupOutcome?.let { OutcomeLine(it) }
            OutlinedButton(
                onClick = { confirmClear = true },
                enabled = (historySize ?: 0) > 0,
                colors = outlinedButtonColors(danger = true),
            ) { Text(stringResource(R.string.settings_clear_history)) }

            Section(R.string.settings_section_online)
            Help(R.string.settings_online_help)
            OutlinedTextField(
                value = settings.leaderboardUrl,
                onValueChange = { v -> onChange { it.copy(leaderboardUrl = v.trim().take(200)) } },
                label = { Text(stringResource(R.string.settings_server)) },
                placeholder = { Text("https://…") },
                singleLine = true,
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri),
                modifier = Modifier.fillMaxWidth(),
            )
            if (settings.leaderboardUrl.isNotBlank() && !Leaderboard.isEncrypted(settings.leaderboardUrl)) {
                Text(stringResource(R.string.settings_server_unencrypted), color = Palette.danger, fontSize = 13.sp)
            }
            OutlinedTextField(
                value = settings.leaderboardToken,
                onValueChange = { v -> onChange { it.copy(leaderboardToken = v.trim().take(200)) } },
                label = { Text(stringResource(R.string.settings_server_key)) },
                singleLine = true,
                visualTransformation = PasswordVisualTransformation(),
                modifier = Modifier.fillMaxWidth(),
            )
            OutlinedButton(
                onClick = {
                    uploading = true
                    scope.launch {
                        onlineOutcome = try {
                            Outcome.Uploaded(actions.onUpload())
                        } catch (e: LeaderboardException) {
                            Outcome.ServerFailed(e)
                        }
                        uploading = false
                    }
                },
                enabled = !uploading && settings.leaderboardUrl.isNotBlank() && (historySize ?: 0) > 0,
                colors = outlinedButtonColors(),
            ) { Text(stringResource(if (uploading) R.string.settings_uploading else R.string.settings_upload)) }
            onlineOutcome?.let { OutcomeLine(it) }

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
                TextButton(onClick = { confirmClear = false; actions.onClearHistory() }) {
                    Text(stringResource(R.string.delete), color = Palette.danger)
                }
            },
            dismissButton = { TextButton(onClick = { confirmClear = false }) { Text(stringResource(R.string.cancel)) } },
        )
    }
}

@Composable
private fun OutcomeLine(outcome: Outcome) {
    val (text, ok) = when (outcome) {
        is Outcome.Exported -> pluralStringResource(R.plurals.settings_exported, outcome.games, outcome.games) to true
        is Outcome.Imported -> with(outcome.summary) {
            stringResource(R.string.settings_imported, added, updated, unchanged) +
                if (unreadable > 0) " " + pluralStringResource(R.plurals.settings_import_unreadable, unreadable, unreadable) else ""
        } to (outcome.summary.unreadable == 0)
        is Outcome.Uploaded -> stringResource(R.string.settings_uploaded, outcome.result.sent, outcome.result.kept) to
            (outcome.result.kept == outcome.result.sent)
        is Outcome.Failed -> stringResource(R.string.settings_file_failed, outcome.message) to false
        is Outcome.ServerFailed -> leaderboardError(outcome.error) to false
    }
    Text(text, color = if (ok) Palette.validText else Palette.danger, fontSize = 13.sp)
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
