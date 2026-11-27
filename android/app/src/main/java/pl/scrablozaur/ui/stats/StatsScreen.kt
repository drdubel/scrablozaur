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
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.PrimaryScrollableTabRow
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Tab
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.pluralStringResource
import androidx.compose.ui.res.stringArrayResource
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import pl.scrablozaur.R
import pl.scrablozaur.core.GameRecord
import pl.scrablozaur.core.LanguageMeta
import pl.scrablozaur.core.Leaderboard
import pl.scrablozaur.core.LeaderboardException
import pl.scrablozaur.core.Likeness
import pl.scrablozaur.core.OnlineStandings
import pl.scrablozaur.core.RankBy
import pl.scrablozaur.core.Result
import pl.scrablozaur.core.Seat
import pl.scrablozaur.core.accuracy
import pl.scrablozaur.core.allSeats
import pl.scrablozaur.core.byOpponent
import pl.scrablozaur.core.headToHead
import pl.scrablozaur.core.likeness
import pl.scrablozaur.core.movingAverage
import pl.scrablozaur.core.opponent
import pl.scrablozaur.core.opponentKey
import pl.scrablozaur.core.performanceRating
import pl.scrablozaur.core.playerKey
import pl.scrablozaur.core.ranked
import pl.scrablozaur.core.ratedResults
import pl.scrablozaur.core.ratingHistory
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

enum class StatsTab { Overview, History, Ranking, Online }

/** Games a chart shows: the latest ones, so a long history stays readable. */
private const val CHART_GAMES = 30

/** The engines a rating is described by: what the setup screen offers. */
private val LIKENESS_LEVELS = listOf(8, 9, 10)

/** A computer's Elo in a language, from the bots' calibration; null where none was measured. */
typealias BotRating = (language: String, level: Int, strength: Int) -> Double?

/**
 * "Statystyki": the player's results over time, every game played, the
 * ranking of everyone who played on this phone with the records they hold,
 * and -- once a server is set -- the shared ranking online.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun StatsScreen(
    records: List<GameRecord>?,
    languages: List<LanguageMeta>,
    playerName: String,
    initialTab: StatsTab,
    botRating: BotRating,
    /** The ranking server's address; empty for none, and then no Online tab. */
    server: String,
    onOpenGame: (String) -> Unit,
    onBack: () -> Unit,
) {
    val tabs = StatsTab.entries.filter { it != StatsTab.Online || server.isNotBlank() }
    var tab by rememberSaveable { mutableStateOf(initialTab.takeIf { it in tabs } ?: StatsTab.Overview) }
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
            PrimaryScrollableTabRow(selectedTabIndex = tabs.indexOf(tab), containerColor = Palette.background, edgePadding = 8.dp) {
                tabs.forEach { t ->
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
                                        StatsTab.Online -> R.string.tab_online
                                    },
                                ),
                            )
                        },
                    )
                }
            }
            when {
                tab == StatsTab.Online -> OnlineRanking(server, languages)
                records == null -> Box(Modifier.fillMaxSize()) { EngineBusy(Modifier.align(Alignment.Center).size(32.dp)) }
                records.isEmpty() -> Text(stringResource(R.string.stats_empty), Modifier.padding(24.dp), color = Palette.muted)
                else -> Filtered(records, languages, playerName, tab, botRating, onOpenGame)
            }
        }
    }
}

/** The filters -- whose games, in which language -- and the tab's content under them. */
@Composable
private fun Filtered(
    all: List<GameRecord>,
    languages: List<LanguageMeta>,
    playerName: String,
    tab: StatsTab,
    botRating: BotRating,
    onOpenGame: (String) -> Unit,
) {
    val chronological = remember(all) { all.sortedBy { it.finishedAt } }
    val playedIn = remember(chronological) { chronological.map { it.language }.distinct() }
    val everySeat = remember(chronological) { chronological.allSeats() }
    val players = remember(everySeat) { standings(everySeat).sortedByDescending { it.games } }
    // The phone's player by default -- or the only one there is.
    val defaultPlayer = playerKey(playerName).takeIf { me -> players.any { playerKey(it.name) == me } }
        ?: players.singleOrNull()?.let { playerKey(it.name) }
    var player by rememberSaveable { mutableStateOf(defaultPlayer) }
    var language by rememberSaveable { mutableStateOf<String?>(null) }

    val gamesInLanguage = chronological.filter { language == null || it.language == language }
    val inLanguage = everySeat.filter { language == null || it.game.language == language }
    val mine = inLanguage.filter { player == null || playerKey(it.name) == player }

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
                        FilterChip(selected = language == code, onClick = { language = code }, label = { Text(languageLabel(languages, code)) })
                    }
                }
            }
        }
    }

    when (tab) {
        StatsTab.Overview -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Filters()
            Overview(mine, everyone = player == null, languages, botRating, onOpenGame)
        }
        StatsTab.History -> HistoryList(mine, showNames = player == null, header = { Filters() }, onOpenGame)
        StatsTab.Ranking -> Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState())) {
            Filters()
            // Elo is anchored per language, so one column only when one language is in view.
            val eloLanguage = language ?: playedIn.singleOrNull()
            Ranking(inLanguage, gamesInLanguage, playerName, eloLanguage, botRating, onOpenGame)
        }
        StatsTab.Online -> {}
    }
}

private fun languageLabel(languages: List<LanguageMeta>, code: String) =
    languages.firstOrNull { it.code == code }?.let { "${it.flag} ${it.name}".trim() } ?: code

// ── Przegląd ─────────────────────────────────────────────────────────────────

@Composable
private fun Overview(seats: List<Seat>, everyone: Boolean, languages: List<LanguageMeta>, botRating: BotRating, onOpenGame: (String) -> Unit) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (seats.isEmpty()) {
            Text(stringResource(R.string.stats_empty), color = Palette.muted)
            return@Column
        }
        val s = standing("", seats)
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

        // One person's rating: everyone's games pooled would say nothing.
        if (!everyone) EloSection(seats, languages, botRating, onOpenGame)

        val recent = seats.takeLast(CHART_GAMES)
        val first = seats.size - recent.size
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
            onSelect = { onOpenGame(recent[it].game.id) },
        )

        SectionTitle(stringResource(R.string.chart_scores))
        LineChart(
            series = listOf(
                Series(
                    if (everyone) stringResource(R.string.legend_players) else seats.last().name,
                    Palette.chartYou,
                    recent.map { it.score.toFloat() },
                ),
                Series(stringResource(R.string.legend_opponent), Palette.chartOpponent, recent.map { it.opponent.score.toFloat() }),
            ),
            xLabels = labels,
            onSelect = { onOpenGame(recent[it].game.id) },
        )

        SectionTitle(stringResource(R.string.by_opponent))
        DataTable(
            columns = listOf(
                TableColumn(stringResource(R.string.col_opponent), 2.4f),
                TableColumn(stringResource(R.string.col_games), 0.95f, end = true),
                TableColumn(stringResource(R.string.col_wlt), 1.3f, end = true),
                TableColumn(stringResource(R.string.col_spread), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_accuracy), 0.8f, end = true),
            ),
            rows = byOpponent(seats).map { (o, r) ->
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

/**
 * The player's Elo in each language they played the computer in -- fitted to
 * their results against bots whose ratings the calibration measured -- with
 * the computer that plays like it, and the rating game by game.
 */
@Composable
private fun EloSection(seats: List<Seat>, languages: List<LanguageMeta>, botRating: BotRating, onOpenGame: (String) -> Unit) {
    val byLanguage = remember(seats) {
        seats.groupBy { it.game.language }.mapValues { (_, mine) -> ratedResults(mine, botRating) }.filterValues { it.isNotEmpty() }
    }
    if (byLanguage.isEmpty()) return
    SectionTitle(stringResource(R.string.elo_title))
    byLanguage.forEach { (code, rated) ->
        val rating = remember(rated) { performanceRating(rated.map { it.second }) }
        val like = remember(rating, code) { likeness(rating, LIKENESS_LEVELS) { level, strength -> botRating(code, level, strength) } }
        Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
            Column(Modifier.fillMaxWidth().padding(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text(rating.roundToInt().toString(), fontWeight = FontWeight.Bold, fontSize = 26.sp)
                    Spacer(Modifier.width(10.dp))
                    Column(Modifier.weight(1f)) {
                        like?.let { Text(likenessLabel(it), fontWeight = FontWeight.SemiBold) }
                        Text(
                            pluralStringResource(R.plurals.elo_basis, rated.size, rated.size, languageLabel(languages, code)),
                            color = Palette.muted,
                            fontSize = 12.sp,
                        )
                    }
                }
            }
        }
        if (rated.size >= 2) {
            val history = remember(rated) { ratingHistory(rated.map { it.second }) }
            val shown = history.takeLast(CHART_GAMES)
            val first = history.size - shown.size
            LineChart(
                series = listOf(Series(stringResource(R.string.legend_elo), Palette.chartYou, shown.map { it.toFloat() })),
                xLabels = shown.indices.map { (first + it + 1).toString() },
                onSelect = { onOpenGame(rated[first + it].first.game.id) },
            )
        }
    }
    Text(stringResource(R.string.elo_help), color = Palette.muted, fontSize = 12.sp)
}

/** "gra jak Strateg · 70%", or beyond either end of what the computer can be set to. */
@Composable
fun likenessLabel(l: Likeness): String {
    val names = stringArrayResource(R.array.level_names)
    val engine = names[(l.level - 1).coerceIn(0, names.lastIndex)]
    return when (l.beyond) {
        1 -> stringResource(R.string.elo_beyond_strongest, engine)
        -1 -> stringResource(R.string.elo_below_weakest, engine)
        else -> stringResource(R.string.elo_plays_like, engine, l.strength)
    }
}

// ── Historia ─────────────────────────────────────────────────────────────────

@Composable
private fun HistoryList(seats: List<Seat>, showNames: Boolean, header: @Composable () -> Unit, onOpenGame: (String) -> Unit) {
    val newest = remember(seats) { seats.asReversed() }
    LazyColumn(Modifier.fillMaxSize(), contentPadding = PaddingValues(bottom = 12.dp)) {
        item { header() }
        items(newest, key = { "${it.game.id}:${it.index}" }) { s ->
            Card(
                Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp).clickable { onOpenGame(s.game.id) },
                colors = CardDefaults.cardColors(containerColor = Palette.surface),
            ) {
                Row(Modifier.padding(horizontal = 12.dp, vertical = 10.dp), verticalAlignment = Alignment.CenterVertically) {
                    ResultDot(s.result)
                    Spacer(Modifier.width(12.dp))
                    Column(Modifier.weight(1f)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(stringResource(R.string.score_vs, s.score, s.opponent.score), fontWeight = FontWeight.Bold, fontSize = 16.sp)
                            if (showNames) {
                                Spacer(Modifier.width(8.dp))
                                Text(s.name, color = Palette.muted, fontSize = 13.sp, maxLines = 1, overflow = TextOverflow.Ellipsis)
                            }
                        }
                        Text(opponentLabel(s.opponentKey, s.opponent.name), color = Palette.muted, fontSize = 13.sp)
                    }
                    Column(horizontalAlignment = Alignment.End) {
                        Text(formatDateTime(s.game.finishedAt), color = Palette.muted, fontSize = 12.sp)
                        s.accuracy?.let {
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
private fun Ranking(
    seats: List<Seat>,
    games: List<GameRecord>,
    playerName: String,
    /** The one language in view, whose bots rate the players; null for several. */
    eloLanguage: String?,
    botRating: BotRating,
    onOpenGame: (String) -> Unit,
) {
    Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        if (seats.isEmpty()) {
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
        val elo = remember(seats, eloLanguage) {
            if (eloLanguage == null) {
                emptyMap()
            } else {
                seats.groupBy { playerKey(it.name) }
                    .mapValues { (_, mine) -> ratedResults(mine, botRating).takeIf { it.isNotEmpty() }?.let { r -> performanceRating(r.map { it.second }) } }
            }
        }
        val table = standings(seats).ranked(order)
        val me = playerKey(playerName)
        val showElo = elo.values.any { it != null }
        DataTable(
            columns = listOfNotNull(
                TableColumn(stringResource(R.string.col_number), 0.5f),
                TableColumn(stringResource(R.string.col_player), 2.0f),
                TableColumn(stringResource(R.string.col_games), 0.95f, end = true),
                TableColumn(stringResource(R.string.col_win_rate), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_average), 0.8f, end = true),
                TableColumn(stringResource(R.string.col_accuracy), 0.8f, end = true),
                TableColumn(stringResource(R.string.col_record), 0.8f, end = true),
                TableColumn(stringResource(R.string.col_elo), 0.9f, end = true).takeIf { showElo },
            ),
            rows = table.mapIndexed { i, s ->
                listOfNotNull(
                    TableCell("${i + 1}", Palette.muted),
                    TableCell(s.name, bold = playerKey(s.name) == me),
                    TableCell(s.games.toString()),
                    TableCell(percent(s.winRate * 100), bold = order == RankBy.WinRate),
                    TableCell(s.avgScore.roundToInt().toString(), bold = order == RankBy.AvgScore),
                    s.accuracy?.let { TableCell(percent(it), ratingColor(it.roundToInt()), bold = order == RankBy.Accuracy) } ?: TableCell("—", Palette.muted),
                    TableCell(s.bestScore.toString(), bold = order == RankBy.Best),
                    (elo[playerKey(s.name)]?.let { TableCell(it.roundToInt().toString()) } ?: TableCell("—", Palette.muted)).takeIf { showElo },
                )
            },
            selected = table.indexOfFirst { playerKey(it.name) == me }.takeIf { it >= 0 },
        )
        if (table.size < 2) Text(stringResource(R.string.ranking_players_help), color = Palette.muted, fontSize = 13.sp)

        val duels = remember(games) { headToHead(games) }
        if (duels.isNotEmpty()) {
            SectionTitle(stringResource(R.string.ranking_head_to_head))
            DataTable(
                columns = listOf(
                    TableColumn(stringResource(R.string.col_player), 1.8f),
                    TableColumn(stringResource(R.string.col_duel), 1.2f, end = false),
                    TableColumn(stringResource(R.string.col_player), 1.8f, end = true),
                ),
                rows = duels.map { d ->
                    listOf(
                        TableCell(d.first, bold = d.firstWins > d.secondWins),
                        TableCell(if (d.ties > 0) "${d.firstWins} : ${d.secondWins} (${d.ties})" else "${d.firstWins} : ${d.secondWins}"),
                        TableCell(d.second, bold = d.secondWins > d.firstWins),
                    )
                },
            )
            Text(stringResource(R.string.ranking_head_to_head_help), color = Palette.muted, fontSize = 13.sp)
        }

        SectionTitle(stringResource(R.string.ranking_top_scores))
        val top = topScores(seats)
        DataTable(
            columns = listOf(
                TableColumn(stringResource(R.string.col_number), 0.5f),
                TableColumn(stringResource(R.string.col_player), 1.8f),
                TableColumn(stringResource(R.string.col_score), 0.9f, end = true),
                TableColumn(stringResource(R.string.col_opponent), 2.0f, end = true),
                TableColumn(stringResource(R.string.col_date), 1.4f, end = true),
            ),
            rows = top.mapIndexed { i, s ->
                listOf(
                    TableCell("${i + 1}", Palette.muted),
                    TableCell(s.name, dot = resultColor(s.result)),
                    TableCell(s.score.toString(), bold = true),
                    TableCell(opponentLabel(s.opponentKey, s.opponent.name), Palette.muted),
                    TableCell(formatDate(s.game.finishedAt), Palette.muted),
                )
            },
            onRowClick = { onOpenGame(top[it].game.id) },
        )
        ResultLegend()

        SectionTitle(stringResource(R.string.ranking_records))
        Records(seats, onOpenGame)
    }
}

@Composable
private fun ResultLegend() = ChartLegend(
    listOf(
        stringResource(R.string.result_win) to resultColor(Result.Win),
        stringResource(R.string.result_tie) to resultColor(Result.Tie),
        stringResource(R.string.result_loss) to resultColor(Result.Loss),
    ),
)

@Composable
private fun Records(seats: List<Seat>, onOpenGame: (String) -> Unit) {
    val r = remember(seats) { records(seats) }
    val rows = buildList {
        r.highScore?.let { add(Triple(R.string.rec_high_score, it.value.toString(), it.seat)) }
        r.bestMove?.let { add(Triple(R.string.rec_best_move, wordValue(it.value.word, it.value.score), it.seat)) }
        r.longestWord?.let { add(Triple(R.string.rec_longest_word, wordValue(it.value.word, it.value.score), it.seat)) }
        r.bestAccuracy?.let { add(Triple(R.string.rec_best_accuracy, percent(it.value), it.seat)) }
        r.biggestWin?.let { add(Triple(R.string.rec_biggest_win, stringResource(R.string.rec_margin_value, it.value), it.seat)) }
        r.mostBingos?.let { add(Triple(R.string.rec_most_bingos, it.value.toString(), it.seat)) }
    }
    Card(colors = CardDefaults.cardColors(containerColor = Palette.surface)) {
        Column {
            rows.forEach { (label, value, seat) ->
                Row(
                    Modifier.fillMaxWidth().clickable { onOpenGame(seat.game.id) }.padding(horizontal = 12.dp, vertical = 9.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text(stringResource(label), fontSize = 12.sp, color = Palette.muted)
                        Text(value, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                    }
                    Text(
                        stringResource(R.string.rec_holder, seat.name, formatDate(seat.game.finishedAt)),
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

// ── Online ───────────────────────────────────────────────────────────────────

private sealed interface OnlineState {
    data object Loading : OnlineState
    data class Loaded(val standings: OnlineStandings) : OnlineState
    data class Failed(val error: LeaderboardException) : OnlineState
}

/** The shared ranking on the server set in the settings: everyone who sent games there. */
@Composable
private fun OnlineRanking(server: String, languages: List<LanguageMeta>) {
    var language by rememberSaveable { mutableStateOf<String?>(null) }
    var reload by remember { mutableIntStateOf(0) }
    var state by remember { mutableStateOf<OnlineState>(OnlineState.Loading) }
    LaunchedEffect(server, language, reload) {
        state = OnlineState.Loading
        state = try {
            OnlineState.Loaded(Leaderboard.standings(server, language))
        } catch (e: LeaderboardException) {
            OnlineState.Failed(e)
        }
    }
    Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(12.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(Leaderboard.endpoint(server, ""), color = Palette.muted, fontSize = 13.sp, modifier = Modifier.weight(1f))
            IconButton(onClick = { reload++ }) { Icon(Icons.Filled.Refresh, stringResource(R.string.online_refresh)) }
        }
        when (val s = state) {
            OnlineState.Loading -> EngineBusy(Modifier.size(28.dp))
            is OnlineState.Failed -> Text(leaderboardError(s.error), color = Palette.danger)
            is OnlineState.Loaded -> {
                val board = s.standings
                if (!board.enabled) {
                    Text(stringResource(R.string.online_disabled), color = Palette.muted)
                    return@Column
                }
                if (board.languages.size > 1) {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        FilterChip(selected = language == null, onClick = { language = null }, label = { Text(stringResource(R.string.filter_all_languages)) })
                        board.languages.forEach { code ->
                            FilterChip(selected = language == code, onClick = { language = code }, label = { Text(languageLabel(languages, code)) })
                        }
                    }
                }
                if (board.players.isEmpty()) {
                    Text(stringResource(R.string.online_empty), color = Palette.muted)
                    return@Column
                }
                SectionTitle(stringResource(R.string.ranking_players))
                DataTable(
                    columns = listOf(
                        TableColumn(stringResource(R.string.col_number), 0.5f),
                        TableColumn(stringResource(R.string.col_player), 2.0f),
                        TableColumn(stringResource(R.string.col_games), 0.95f, end = true),
                        TableColumn(stringResource(R.string.col_wlt), 1.3f, end = true),
                        TableColumn(stringResource(R.string.col_win_rate), 0.9f, end = true),
                        TableColumn(stringResource(R.string.col_average), 0.8f, end = true),
                        TableColumn(stringResource(R.string.col_accuracy), 0.8f, end = true),
                    ),
                    rows = board.players.mapIndexed { i, p ->
                        listOf(
                            TableCell("${i + 1}", Palette.muted),
                            TableCell(p.name),
                            TableCell(p.games.toString()),
                            TableCell("${p.wins}–${p.losses}–${p.ties}"),
                            TableCell(percent(p.winRate * 100)),
                            TableCell(p.avgScore.roundToInt().toString()),
                            p.accuracy?.let { TableCell(percent(it), ratingColor(it.roundToInt())) } ?: TableCell("—", Palette.muted),
                        )
                    },
                )
                SectionTitle(stringResource(R.string.ranking_top_scores))
                val names = stringArrayResource(R.array.level_names)
                DataTable(
                    columns = listOf(
                        TableColumn(stringResource(R.string.col_number), 0.5f),
                        TableColumn(stringResource(R.string.col_player), 1.8f),
                        TableColumn(stringResource(R.string.col_score), 0.9f, end = true),
                        TableColumn(stringResource(R.string.col_opponent), 2.0f, end = true),
                        TableColumn(stringResource(R.string.col_date), 1.4f, end = true),
                    ),
                    rows = board.topScores.mapIndexed { i, t ->
                        val opponent = if (t.opponentComputer && t.opponentLevel != null) {
                            stringResource(R.string.opponent_label, names[(t.opponentLevel - 1).coerceIn(0, names.lastIndex)], t.opponentStrength ?: 100)
                        } else {
                            t.opponent
                        }
                        listOf(
                            TableCell("${i + 1}", Palette.muted),
                            TableCell(t.name, dot = resultColor(t.result)),
                            TableCell(t.score.toString(), bold = true),
                            TableCell(opponent, Palette.muted),
                            TableCell(formatDate(t.finished), Palette.muted),
                        )
                    },
                )
                ResultLegend()
                Text(stringResource(R.string.online_games, board.games), color = Palette.muted, fontSize = 12.sp)
            }
        }
    }
}

/** Why the ranking server could not be used, in words. */
@Composable
fun leaderboardError(e: LeaderboardException): String = when (e) {
    is LeaderboardException.Unreachable -> stringResource(R.string.online_unreachable, e.message.orEmpty())
    is LeaderboardException.Disabled -> stringResource(R.string.online_disabled)
    is LeaderboardException.BadToken -> stringResource(R.string.online_bad_token)
    is LeaderboardException.Server -> stringResource(R.string.online_server_error, e.message.orEmpty())
}
