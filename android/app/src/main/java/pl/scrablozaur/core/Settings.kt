package pl.scrablozaur.core

import android.content.Context
import androidx.core.content.edit
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.updateAndGet
import org.json.JSONArray
import pl.scrablozaur.engine.HintOrder

/** How fast an automatic game steps, and a replay with it. */
enum class Playback(val autoMoveMs: Long, val replayMs: Long) {
    Slow(1100, 1400),
    Normal(550, 700),
    Fast(150, 250),
}

/**
 * The new-game screen as it was last used: the next game starts from it, so
 * nothing has to be picked again. Names and the bot's options only -- a seat's
 * own "Zaawansowane" override is for one benchmark, not worth remembering.
 */
data class SetupChoices(
    /** A `SetupMode` name. */
    val mode: String = "Competitive",
    val auto: Boolean = false,
    val count: Int = 2,
    val language: String? = null,
    val strength: Int = 100,
    val neural: Boolean = true,
    val simulation: Boolean = true,
    val seatNames: List<String> = List(4) { "" },
    val computerSeat: Int = 1,
    val autoSeatNames: List<String> = List(4) { "" },
    /** Rywalizacja: how many people play at this phone (1 is a game against the computer)... */
    val people: Int = 1,
    /** ...whether a computer joins them, and the names of the second to fourth. */
    val withComputer: Boolean = true,
    val tableNames: List<String> = List(4) { "" },
)

data class AppSettings(
    /** The name a competitive game is played under: the player the statistics follow. */
    val playerName: String = "",
    val showMoveRating: Boolean = true,
    /** Where hints start; Strateg and Analityk fall back to Punkty without a leave net. */
    val hintOrder: HintOrder = HintOrder.SCORE,
    val playback: Playback = Playback.Normal,
    val keepScreenOn: Boolean = false,
    /** Definitions are the one thing the app fetches from the internet. */
    val onlineDefinitions: Boolean = true,
    val setup: SetupChoices = SetupChoices(),
    /** A Scrablozaur web server keeping a shared ranking (`web/leaderboard.py`); empty for none. */
    val leaderboardUrl: String = "",
    /** The key that server asks for, if it asks for one. */
    val leaderboardToken: String = "",
    /** This phone's id on that server, made on the first upload: sending again replaces rather than duplicates. */
    val deviceId: String = "",
)

/** The settings, kept in shared preferences and handed out as a flow. */
class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val _state = MutableStateFlow(load())
    val state: StateFlow<AppSettings> = _state

    val current: AppSettings get() = _state.value

    fun update(change: (AppSettings) -> AppSettings) {
        val before = _state.value
        val after = _state.updateAndGet(change)
        if (after != before) store(after)
    }

    private fun load(): AppSettings {
        val d = AppSettings()
        val s = d.setup
        fun names(key: String) = prefs.getString(key, null)?.let { text ->
            runCatching { JSONArray(text).let { a -> List(4) { a.optString(it) } } }.getOrNull()
        } ?: List(4) { "" }
        return AppSettings(
            playerName = prefs.getString("player_name", d.playerName)!!,
            showMoveRating = prefs.getBoolean("show_move_rating", d.showMoveRating),
            hintOrder = enumOr(prefs.getString("hint_order", null), d.hintOrder),
            playback = enumOr(prefs.getString("playback", null), d.playback),
            keepScreenOn = prefs.getBoolean("keep_screen_on", d.keepScreenOn),
            onlineDefinitions = prefs.getBoolean("online_definitions", d.onlineDefinitions),
            leaderboardUrl = prefs.getString("leaderboard_url", d.leaderboardUrl)!!,
            leaderboardToken = prefs.getString("leaderboard_token", d.leaderboardToken)!!,
            deviceId = prefs.getString("device_id", d.deviceId)!!,
            setup = SetupChoices(
                mode = prefs.getString("setup_mode", s.mode)!!,
                auto = prefs.getBoolean("setup_auto", s.auto),
                count = prefs.getInt("setup_count", s.count).coerceIn(2, 4),
                language = prefs.getString("setup_language", s.language),
                strength = prefs.getInt("setup_strength", s.strength).coerceIn(1, 100),
                neural = prefs.getBoolean("setup_neural", s.neural),
                simulation = prefs.getBoolean("setup_simulation", s.simulation),
                seatNames = names("setup_seat_names"),
                computerSeat = prefs.getInt("setup_computer_seat", s.computerSeat).coerceIn(0, 3),
                autoSeatNames = names("setup_auto_seat_names"),
                people = prefs.getInt("setup_people", s.people).coerceIn(1, 4),
                withComputer = prefs.getBoolean("setup_with_computer", s.withComputer),
                tableNames = names("setup_table_names"),
            ),
        )
    }

    private fun store(a: AppSettings) {
        val s = a.setup
        prefs.edit {
            putString("player_name", a.playerName)
            putBoolean("show_move_rating", a.showMoveRating)
            putString("hint_order", a.hintOrder.name)
            putString("playback", a.playback.name)
            putBoolean("keep_screen_on", a.keepScreenOn)
            putBoolean("online_definitions", a.onlineDefinitions)
            putString("leaderboard_url", a.leaderboardUrl)
            putString("leaderboard_token", a.leaderboardToken)
            putString("device_id", a.deviceId)
            putString("setup_mode", s.mode)
            putBoolean("setup_auto", s.auto)
            putInt("setup_count", s.count)
            putString("setup_language", s.language)
            putInt("setup_strength", s.strength)
            putBoolean("setup_neural", s.neural)
            putBoolean("setup_simulation", s.simulation)
            putString("setup_seat_names", JSONArray(s.seatNames).toString())
            putInt("setup_computer_seat", s.computerSeat)
            putString("setup_auto_seat_names", JSONArray(s.autoSeatNames).toString())
            putInt("setup_people", s.people)
            putBoolean("setup_with_computer", s.withComputer)
            putString("setup_table_names", JSONArray(s.tableNames).toString())
        }
    }

    private inline fun <reified E : Enum<E>> enumOr(name: String?, default: E): E =
        name?.let { n -> enumValues<E>().firstOrNull { it.name == n } } ?: default
}
