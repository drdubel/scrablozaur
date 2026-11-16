package pl.scrablozaur.core

import android.content.Context
import android.os.SystemClock
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.json.JSONObject
import pl.scrablozaur.engine.LanguagePackHandle
import pl.scrablozaur.engine.Premium
import pl.scrablozaur.engine.premiumLayout
import java.io.File
import java.security.MessageDigest

/** The app's logcat tag. */
internal const val LOG_TAG = "Scrablozaur"

/** What the app needs to know about a language before (or without) loading it. */
data class LanguageMeta(
    val code: String,
    val name: String,
    val flag: String,
    val alphabet: String,
    val blank: Char,
    /** Word-definition providers, in the order to try them. */
    val definitions: List<String>,
    val hasLeaveNet: Boolean,
    internal val dawg: String,
    internal val gaddag: String,
    internal val leaveNet: String?,
)

/**
 * The engine's data and the languages built from it.
 *
 * The APK carries every `languages/<code>.json`, the lexicons and the leave nets under
 * `assets/engine/`, laid out as in the repository so the paths inside each
 * definition resolve. They are copied out once (the engine opens files), checked
 * against the build's manifest of SHA-256s, and re-copied whenever a new build
 * ships different data.
 */
class EngineRepository(private val context: Context) {
    private val root = File(context.filesDir, "engine")
    private val lock = Mutex()
    private var ready = false
    private val packs = HashMap<String, LanguagePackHandle>()

    val premiums: List<Premium> by lazy { premiumLayout() }

    /** Every installed language, read from the asset definitions. */
    val languages: List<LanguageMeta> by lazy {
        val dir = "engine/languages"
        context.assets.list(dir).orEmpty().filter { it.endsWith(".json") }.sorted().map { name ->
            parse(context.assets.open("$dir/$name").bufferedReader().use { it.readText() })
        }
    }

    fun language(code: String): LanguageMeta =
        languages.firstOrNull { it.code == code } ?: languages.first()

    /** The default the web uses: Polish where installed. */
    val defaultLanguage: String get() = languages.firstOrNull { it.code == "pl" }?.code ?: languages.first().code

    /** The loaded pack for `code`, extracting the data first if needed. */
    suspend fun pack(code: String): LanguagePackHandle = withContext(Dispatchers.IO) {
        lock.withLock {
            packs[code]?.let { return@withLock it }
            ensureExtracted()
            val meta = language(code)
            val definition = File(root, "languages/${meta.code}.json").readText()
            val start = SystemClock.elapsedRealtime()
            val pack = LanguagePackHandle.load(
                definition,
                File(root, meta.dawg).path,
                File(root, meta.gaddag).path,
                meta.leaveNet?.let { File(root, it).path },
            )
            Log.i(LOG_TAG, "loaded '${meta.code}' in ${SystemClock.elapsedRealtime() - start} ms")
            packs[meta.code] = pack
            pack
        }
    }

    private fun ensureExtracted() {
        if (ready) return
        val manifestText = context.assets.open("engine/manifest.json").bufferedReader().use { it.readText() }
        val stamp = File(root, "manifest.json")
        if (!stamp.exists() || stamp.readText() != manifestText) {
            val start = SystemClock.elapsedRealtime()
            // Copy into a scratch directory and swap it in, so a copy cut short
            // by the system killing the app never looks complete.
            val staging = File(context.filesDir, "engine.partial")
            staging.deleteRecursively()
            val files = JSONObject(manifestText).getJSONArray("files")
            for (i in 0 until files.length()) {
                val entry = files.getJSONObject(i)
                val path = entry.getString("path")
                val target = File(staging, path)
                target.parentFile?.mkdirs()
                val digest = MessageDigest.getInstance("SHA-256")
                context.assets.open("engine/$path").use { input ->
                    target.outputStream().buffered(1 shl 16).use { output ->
                        val buf = ByteArray(1 shl 16)
                        while (true) {
                            val n = input.read(buf)
                            if (n < 0) break
                            digest.update(buf, 0, n)
                            output.write(buf, 0, n)
                        }
                    }
                }
                val sha = digest.digest().joinToString("") { "%02x".format(it) }
                check(sha == entry.getString("sha256")) { "$path is corrupt in this build" }
            }
            File(staging, "manifest.json").writeText(manifestText)
            root.deleteRecursively()
            check(staging.renameTo(root)) { "could not install the engine data" }
            Log.i(LOG_TAG, "extracted the engine data in ${SystemClock.elapsedRealtime() - start} ms")
        }
        ready = true
    }

    private fun parse(text: String): LanguageMeta {
        val json = JSONObject(text)
        val leave = json.optJSONObject("leave_net")
        val defs = json.optJSONArray("definitions")
        return LanguageMeta(
            code = json.getString("code"),
            name = json.getString("name"),
            flag = json.optString("flag"),
            alphabet = json.getString("alphabet"),
            blank = json.getString("blank").first(),
            definitions = (0 until (defs?.length() ?: 0)).map { defs!!.getString(it) },
            hasLeaveNet = leave != null,
            dawg = json.getString("dawg"),
            gaddag = json.getString("gaddag"),
            leaveNet = leave?.getString("weights"),
        )
    }
}
