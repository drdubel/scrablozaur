package pl.scrablozaur.ui.stats

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.ChevronRight
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.PrimaryTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.core.GameRecord
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.RankBy
import pl.scrablozaur.core.Result
import pl.scrablozaur.core.accuracy
import pl.scrablozaur.core.byOpponent
import pl.scrablozaur.core.me
import pl.scrablozaur.core.movingAverage
import pl.scrablozaur.core.opponent
import pl.scrablozaur.core.opponentKey
import pl.scrablozaur.core.playerKey
import pl.scrablozaur.core.ranked
import pl.scrablozaur.core.records
import pl.scrablozaur.core.result
import pl.scrablozaur.core.standing
import pl.scrablozaur.core.standings
import pl.scrablozaur.core.topScores
import pl.scrablozaur.ui.components.ChartLegend
import pl.scrablozaur.ui.components.DataTable
import pl.scrablozaur.ui.components.EngineBusy
import pl.scrablozaur.ui.components.LineChart
import pl.scrablozaur.ui.components.ResultBar
import pl.scrablozaur.ui.components.SectionTitle
import pl.scrablozaur.ui.components.Series
import pl.scrablozaur.ui.components.StatTile
import pl.scrablozaur.ui.components.TableCell
import pl.scrablozaur.ui.components.TableColumn
import pl.scrablozaur.ui.components.TileGrid
import pl.scrablozaur.ui.game.ratingColor
import pl.scrablozaur.ui.theme.Palette
import kotlin.math.roundToInt

enum class StatsTab { Overview, History, Ranking }

/** Games a chart shows: the latest ones, so a long history stays readable. */
private const val CHART_GAMES = 30

/**
 * "Statystyki": the player's results over time, every game played, and the
 * ranking of everyone who played on this phone with the records they hold.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StatsScreen(
    records: List<GameRecord>?,
    languages: List<LanguageMeta>,
    playerName: String,
    initialTab: StatsTab,
    onOpenGame: (String) -> Unit,
    onBack: () -> Unit,
) {
    var tab by rememberSaveable { mutableStateOf(initialTab) }
    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text(stringResource(R.string.stats_title), fontWeight = FontWeight.Bold) },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, stringResource(R.string.back)) }
                },
            )
        },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            PrimaryTabRow(selectedTabIndex = tab.ordinal, containerColor = Palette.background) {
                StatsTab.entries.forEach { t ->
                    Tab(
                        selected = tab == t,
                        onClick = { tab = t },
                        text = {
                            Text(
                                stringResource(
                                    when (t) {
                                        StatsTab.Overview -> R.string.tab_overview
                                        StatsTab.History -> R.string.tab_history
                                        StatsTab.Ranking -> R.string.tab_ranking
                                    },
                                ),
                            )
                        },
                    )
                }
            }
            when {
                records == null -> Box(Modifier.fillMaxSize()) { EngineBusy(Modifier.align(Alignment.Center).size(32.dp)) }
                records.isEmpty() -> Text(stringResource(R.string.stats_empty), Modifier.padding(24.dp), color = Palette.muted)
                else -> Filtered(records, languages, playerName, tab, onOpenGame)
            }
        }
    }
}

/** The filters -- whose games, in which language -- and the tab's content under them. */
@Composable
private fun Filtered(all: List<GameRecord>, languages: List<LanguageMeta>, playerName: String, tab: StatsTab, onOpenGame: (String) -> Unit) {
    val chronological = remember(all) { all.sortedBy { it.finishedAt } }
    val playedIn = remember(chronological) { chronological.map { it.language }.distinct() }
    val players = remember(chronological) { standings(chronological).sortedByDescending { it.games } }
    // The phone's player by default -- or the only one there is.
    val defaultPlayer = playerKey(playerName).takeIf { me -> players.any { playerKey(it.name) == me } }
        ?: players.singleOrNull()?.let { playerKey(it.name) }
    var player by rememberSaveable { mutableStateOf(defaultPlayer) }
    var language by rememberSaveable { mutableStateOf<String?>(null) }

    val inLanguage = chronological.filter { language == null || it.language == language }
    val mine = inLanguage.filter { player == null || playerKey(it.me.name) == player }

    @Composable
    fun Filters() {
        val showPlayers = tab != StatsTab.Ranking && players.size > 1
        if (!showPlayers && playedIn.size < 2) return
        Column(Modifier.padding(horizontal = 12.dp, vertical = 6.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
            if (showPlayers) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FilterChip(selected = player == null, onClick = { player = null }, label = { Text(stringResource(R.string.filter_everyone)) })
                    players.forEach { s ->
                        val key = playerKey(s.name)
                        FilterChip(selected = player == key, onClick = { player = key }, label = { Text(s.name) })
                    }
                }
            }
            if (playedIn.size > 1) {
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    FilterChip(selected = language == null, onClick = { language = null }, label = { Text(stringResource(R.string.filter_all_languages)) })
                    playedIn.forEach { code ->
                        val meta = languages.firstOrNull { it.code == code }
                        FilterChip(
                            selected = language == code,
                            onClick = { language = code },
                            label = { Text(meta?.let { "${it.flag} ${it.name}".trim() } ?: code) },
                        )
                    }
                }
            }
        }
    }

    when (tab) {
        StatsTab.Overview -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Filters()
            Overview(mine, everyone = player == null, onOpenGame)
        }
        StatsTab.History -> HistoryList(mine, showNames = player == null, header = { Filters() }, onOpenGame)
        StatsTab.Ranking -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Filters()
            Ranking(inLanguage, playerName, onOpenGame)
        }
    }
}

// ── Przegląd ─────────────────────────────────────────────────────────────────

@Composable
private fun Overview(games: List<GameRecord>, everyone: Boolean, onOpenGame: (String) -> Unit) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (games.isEmpty()) {
            Text(stringResource(R.string.stats_empty), color = Palette.muted)
            return@Column
        }
        val s = standing("", games)
        TileGrid(
            listOf<@Composable (Modifier) -> Unit>(
                { m -> StatTile(stringResource(R.string.stat_games), s.games.toString(), m, detail = stringResource(R.string.stat_wlt_value, s.wins, s.losses, s.ties)) },
                { m -> StatTile(stringResource(R.string.stat_win_rate), percent(s.winRate * 100), m) },
                { m -> StatTile(stringResource(R.string.stat_avg_score), s.avgScore.roundToInt().toString(), m) },
                { m -> StatTile(stringResource(R.string.stat_best_score), s.bestScore.toString(), m) },
                { m ->
                    StatTile(
                        stringResource(R.string.stat_accuracy),
                        s.accuracy?.let { percent(it) } ?: "—",
                        m,
                        color = s.accuracy?.let { ratingColor(it.roundToInt()) } ?: Palette.text,
                    )
                },
                { m -> StatTile(stringResource(R.string.stat_bingos), s.bingos.toString(), m) },
            ),
        )
        ResultBar(s.wins, s.ties, s.losses)

        val recent = games.takeLast(CHART_GAMES)
        val first = games.size - recent.size
        val labels = recent.indices.map { (first + it + 1).toString() }
        SectionTitle(stringResource(R.string.chart_accuracy))
        val accuracy = recent.map { it.accuracy }
        LineChart(
            series = listOf(
                Series(stringResource(R.string.legend_accuracy), Palette.chartYou, accuracy.map { it?.toFloat() }),
                Series(stringResource(R.string.legend_moving_average), Palette.ratingOk, movingAverage(accuracy, 5).map { it?.toFloat() }, dashed = true),
            ),
            xLabels = labels,
            yMin = 0f,
            yMax = 100f,
            onSelect = { onOpenGame(recent[it].id) },
        )

        SectionTitle(stringResource(R.string.chart_scores))
        LineChart(
            series = listOf(
                Series(
                    if (everyone) stringResource(R.string.legend_players) else games.last().me.name,
                    Palette.chartYou,
                    recent.map { it.me.score.toFloat() },
                ),
                Series(stringResource(R.string.legend_opponent), Palette.chartOpponent, recent.map { it.opponent.score.toFloat() }),
            ),
            xLabels = labels,
            onSelect = { onOpenGame(recent[it].id) },
        )

        SectionTitle(stringResource(R.string.by_opponent))
        DataTable(
            columns = listOf(
                TableColumn(stringResource(R.string.col_opponent), 2.4f),
                TableColumn(stringResource(R.string.col_games), 0.7f, end = true),
                TableColumn(stringResource(R.string.col_wlt), 1.3f, end = true),
                TableColumn(stringResource(R.string.col_spread), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_accuracy), 0.8f, end = true),
            ),
            rows = byOpponent(games).map { (o, r) ->
                listOf(
                    TableCell(opponentLabel(o)),
                    TableCell(r.games.toString()),
                    TableCell("${r.wins}–${r.losses}–${r.ties}"),
                    TableCell(signed(r.avgSpread), if (r.avgSpread >= 0) Palette.validText else Palette.invalidText),
                    r.accuracy?.let { TableCell(percent(it), ratingColor(it.roundToInt())) } ?: TableCell("—", Palette.muted),
                )
            },
        )
    }
}

// ── Historia ─────────────────────────────────────────────────────────────────

@Composable
private fun HistoryList(games: List<GameRecord>, showNames: Boolean, header: @Composable () -> Unit, onOpenGame: (String) -> Unit) {
    val newest = remember(games) { games.asReversed() }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(bottom = 12.dp)) {
        item { header() }
        items(newest, key = { it.id }) { g ->
            Card(
                Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp).clickable { onOpenGame(g.id) },
                colors = CardDefaults.cardColors(containerColor = Palette.surface),
            ) {
                Row(Modifier.padding(horizontal = 12.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    ResultDot(g.result)
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(stringResource(R.string.score_vs, g.me.score, g.opponent.score), fontWeight = FontWeight.Bold, fontSize = 16.sp)
                            if (showNames) {
                                Spacer(Modifier.width(8.dp))
                                Text(g.me.name, color = Palette.muted, fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            }
                        }
                        Text(opponentLabel(g.opponentKey), color = Palette.muted, fontSize = 13.sp)
                    }
                    Column(horizontalAlignment = Alignment.End) {
                        Text(formatDateTime(g.finishedAt), color = Palette.muted, fontSize = 12.sp)
                        g.accuracy?.let {
                            Text(stringResource(R.string.history_accuracy, it.roundToInt()), color = ratingColor(it.roundToInt()), fontSize = 13.sp)
                        }
                    }
                    Icon(Icons.Filled.ChevronRight, null, tint = Palette.muted)
                }
            }
        }
    }
}

@Composable
private fun ResultDot(r: Result) {
    Box(Modifier.size(30.dp).background(resultColor(r), CircleShape), contentAlignment = Alignment.Center) {
        Text(resultBadge(r), color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp)
    }
}

// ── Ranking ──────────────────────────────────────────────────────────────────

@Composable
private fun Ranking(games: List<GameRecord>, playerName: String, onOpenGame: (String) -> Unit) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (games.isEmpty()) {
            Text(stringResource(R.string.stats_empty), color = Palette.muted)
            return@Column
        }
        var order by rememberSaveable { mutableStateOf(RankBy.WinRate) }
        SectionTitle(stringResource(R.string.ranking_players))
        FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            listOf(
                RankBy.WinRate to R.string.rank_by_wins,
                RankBy.AvgScore to R.string.rank_by_score,
                RankBy.Accuracy to R.string.rank_by_accuracy,
                RankBy.Best to R.string.rank_by_best,
            ).forEach { (key, label) ->
                FilterChip(selected = order == key, onClick = { order = key }, label = { Text(stringResource(label)) })
            }
        }
        val table = standings(games).ranked(order)
        val me = playerKey(playerName)
        DataTable(
            columns = listOf(
                TableColumn(stringResource(R.string.col_number), 0.5f),
                TableColumn(stringResource(R.string.col_player), 2.2f),
                TableColumn(stringResource(R.string.col_games), 0.7f, end = true),
                TableColumn(stringResource(R.string.col_win_rate), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_average), 0.8f, end = true),
                TableColumn(stringResource(R.string.col_accuracy), 0.8f, end = true),
                TableColumn(stringResource(R.string.col_record), 0.8f, end = true),
            ),
            rows = table.mapIndexed { i, s ->
                listOf(
                    TableCell("${i + 1}", Palette.muted),
                    TableCell(s.name, bold = playerKey(s.name) == me),
                    TableCell(s.games.toString()),
                    TableCell(percent(s.winRate * 100), bold = order == RankBy.WinRate),
                    TableCell(s.avgScore.roundToInt().toString(), bold = order == RankBy.AvgScore),
                    s.accuracy?.let { TableCell(percent(it), ratingColor(it.roundToInt()), bold = order == RankBy.Accuracy) } ?: TableCell("—", Palette.muted),
                    TableCell(s.bestScore.toString(), bold = order == RankBy.Best),
                )
            },
            selected = table.indexOfFirst { playerKey(it.name) == me }.takeIf { it >= 0 },
        )
        if (table.size < 2) Text(stringResource(R.string.ranking_players_help), color = Palette.muted, fontSize = 13.sp)

        SectionTitle(stringResource(R.string.ranking_top_scores))
        val top = topScores(games)
        DataTable(
            columns = listOf(
                TableColumn(stringResource(R.string.col_number), 0.5f),
                TableColumn(stringResource(R.string.col_player), 1.8f),
                TableColumn(stringResource(R.string.col_score), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_opponent), 2.0f, end = true),
                TableColumn(stringResource(R.string.col_date), 1.4f, end = true),
            ),
            rows = top.mapIndexed { i, g ->
                listOf(
                    TableCell("${i + 1}", Palette.muted),
                    TableCell(g.me.name, dot = resultColor(g.result)),
                    TableCell(g.me.score.toString(), bold = true),
                    TableCell(opponentLabel(g.opponentKey), Palette.muted),
                    TableCell(formatDate(g.finishedAt), Palette.muted),
                )
            },
            onRowClick = { onOpenGame(top[it].id) },
        )
        ChartLegend(
            listOf(
                stringResource(R.string.result_win) to resultColor(Result.Win),
                stringResource(R.string.result_tie) to resultColor(Result.Tie),
                stringResource(R.string.result_loss) to resultColor(Result.Loss),
            ),
        )

        SectionTitle(stringResource(R.string.ranking_records))
        Records(games, onOpenGame)
    }
}

@Composable
private fun Records(games: List<GameRecord>, onOpenGame: (String) -> Unit) {
    val r = remember(games) { records(games) }
    val rows = buildList {
        r.highScore?.let { add(Triple(R.string.rec_high_score, it.value.toString(), it.game)) }
        r.bestMove?.let { add(Triple(R.string.rec_best_move, wordValue(it.value.word, it.value.score), it.game)) }
        r.longestWord?.let { add(Triple(R.string.rec_longest_word, wordValue(it.value.word, it.value.score), it.game)) }
        r.bestAccuracy?.let { add(Triple(R.string.rec_best_accuracy, percent(it.value), it.game)) }
        r.biggestWin?.let { add(Triple(R.string.rec_biggest_win, stringResource(R.string.rec_margin_value, it.value), it.game)) }
        r.mostBingos?.let { add(Triple(R.string.rec_most_bingos, it.value.toString(), it.game)) }
    }
    Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column {
            rows.forEach { (label, value, game) ->
                Row(
                    Modifier.fillMaxWidth().clickable { onOpenGame(game.id) }.padding(horizontal = 12.dp, vertical = 9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(label), fontSize = 12.sp, color = Palette.muted)
                        Text(value, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    }
                    Text(
                        stringResource(R.string.rec_holder, game.me.name, formatDate(game.finishedAt)),
                        color = Palette.muted,
                        fontSize = 12.sp,
                    )
                    Icon(Icons.Filled.ChevronRight, null, tint = Palette.muted)
                }
            }
        }
    }
}

@Composable
private fun wordValue(word: String, score: Int) = stringResource(R.string.rec_word_value, word.uppercase(), score)
