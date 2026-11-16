package pl.scrablozaur.core

import android.util.LruCache
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

/** What a lookup found. */
sealed interface DefinitionResult {
    /** Each entry is one sense group; `<br>` separates numbered senses. */
    data class Found(val entries: List<String>) : DefinitionResult
    data object NotFound : DefinitionResult
    /** Every provider was unreachable -- usually no network. */
    data object Offline : DefinitionResult
}

/**
 * Word definitions -- the one thing the app fetches from the internet. A port of
 * `web/definitions.py`: the same providers (named per language in
 * `languages/<code>.json`), tried in the same order, scraped the same way, and
 * formatted the same way (a lemma prefix only when it differs from the word).
 */
class Definitions(@Suppress("unused") private val cacheDir: File) {
    private val cache = LruCache<String, List<String>>(256)

    private class Unreachable : IOException()

    private fun interface Provider {
        /** `(lemma, definition)` pairs, empty if the provider has nothing. */
        fun fetch(word: String): List<Pair<String, String>>
    }

    private val providers: Map<String, Provider> = mapOf(
        "sjp" to Provider(::fetchSjp),
        "pwn" to Provider(::fetchPwn),
        "wiktionary" to Provider(::fetchWiktionary),
        "dictionaryapi" to Provider(::fetchDictionaryApi),
    )

    suspend fun lookup(word: String, providerNames: List<String>): DefinitionResult = withContext(Dispatchers.IO) {
        val w = word.lowercase()
        val key = "${providerNames.joinToString(",")}:$w"
        cache.get(key)?.let { return@withContext DefinitionResult.Found(it) }
        var reached = false
        for (name in providerNames) {
            val provider = providers[name] ?: continue
            val entries = try {
                provider.fetch(w).also { reached = true }
            } catch (_: Unreachable) {
                emptyList()
            }
            if (entries.isNotEmpty()) {
                // A lemma only earns a prefix when it differs from what was asked:
                // "zwie -> zwać" is worth showing, "kot -> kot" is noise.
                val formatted = entries.map { (lemma, defn) ->
                    if (lemma.lowercase() == w) breakSenses(defn) else "$lemma — ${breakSenses(defn)}"
                }
                cache.put(key, formatted)
                return@withContext DefinitionResult.Found(formatted)
            }
        }
        if (reached) DefinitionResult.NotFound else DefinitionResult.Offline
    }

    // ── Shared helpers (web/definitions.py) ───────────────────────────────────

    private val skipP = Regex(
        "^(SŁOWNIK SJP|KOMENTARZE|PROSIMY|POWIĄZANE|dopuszczal|niedopuszczal|function |-$|\\(brak\\)|dodaj$|OK$|nazwisko$|imię$)",
        RegexOption.IGNORE_CASE,
    )
    private val htmlTag = Regex("<[^>]+>")
    private val entities = mapOf("&quot;" to "\"", "&amp;" to "&", "&nbsp;" to " ", "&lt;" to "<", "&gt;" to ">")
    private val senseBreak = Regex("\\s*;?\\s*(?=\\d{1,2}\\.\\s)")

    /** Tags become spaces, not nothing: senses are separated by bare `<br />`. */
    private fun cleanHtml(s: String): String {
        var out = htmlTag.replace(s, " ")
        for ((ent, ch) in entities) out = out.replace(ent, ch)
        return out.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
    }

    private fun breakSenses(text: String): String =
        text.split(senseBreak).filter { it.isNotEmpty() }.joinToString("<br>")

    private fun quote(word: String) = URLEncoder.encode(word, "UTF-8").replace("+", "%20")

    private fun get(url: String): String? {
        val conn = try {
            URL(url).openConnection() as HttpURLConnection
        } catch (e: IOException) {
            throw Unreachable()
        }
        return try {
            conn.connectTimeout = 5000
            conn.readTimeout = 5000
            conn.setRequestProperty("User-Agent", "scrablozaur/1.0")
            conn.instanceFollowRedirects = true
            val code = try {
                conn.responseCode
            } catch (e: IOException) {
                throw Unreachable()
            }
            // A 4xx is an answer ("not here"); only a failure to reach the
            // server counts as being offline.
            if (code !in 200..299) null else conn.inputStream.bufferedReader().use { it.readText() }
        } finally {
            conn.disconnect()
        }
    }

    // ── Polish ────────────────────────────────────────────────────────────────

    private val paragraph = Regex("<p[^>]*>(.*?)</p>", RegexOption.DOT_MATCHES_ALL)
    private val digitDot = Regex("\\d\\.")

    /** sjp.pl maps inflected forms to their base form ("zwie" -> "zwać"). */
    private fun fetchSjp(word: String): List<Pair<String, String>> {
        val html = get("https://sjp.pl/${quote(word)}") ?: return emptyList()
        val paragraphs = mutableListOf<String>()
        for (m in paragraph.findAll(html)) {
            val p = cleanHtml(m.groupValues[1])
            // Everything after the comments section is user noise.
            if (p.startsWith("KOMENTARZE", ignoreCase = true) || p.startsWith("POWIĄZANE", ignoreCase = true)) break
            if (p.isNotEmpty() && !skipP.containsMatchIn(p)) paragraphs += p
        }
        val entries = mutableListOf<Pair<String, String>>()
        var i = 0
        while (i < paragraphs.size) {
            val t = paragraphs[i]
            val isLemma = !t.startsWith("znaczenie") && t.length < 60 && !digitDot.containsMatchIn(t)
            if (isLemma) {
                // Skip proper nouns (capitalised lemmas not matching the word).
                if (t[0].isUpperCase() && t.lowercase() != word) {
                    i++
                    continue
                }
                var j = i + 1
                if (j < paragraphs.size && paragraphs[j].startsWith("znaczenie")) j++
                if (j < paragraphs.size) {
                    val defn = paragraphs[j]
                    if (!skipP.containsMatchIn(defn)) {
                        entries += t to defn
                        i = j + 1
                        continue
                    }
                }
            }
            i++
        }
        return entries.take(3)
    }

    private val pwnHeadword = Regex(
        "<span class=\"tytul\"><a[^>]*title=\"([^\"]*)\"[^>]*>.*?</a></span>.*?<li[^>]*>(.*?)</li>",
        RegexOption.DOT_MATCHES_ALL,
    )
    private val pwnSense = Regex("<div class=\"znacz\">(.*?)</div>", RegexOption.DOT_MATCHES_ALL)
    private val pwnBodyEnd = Regex("<br|<div class=\"s-przykl\"")

    /** sjp.pwn.pl, for words sjp.pl has no entry for. */
    private fun fetchPwn(word: String): List<Pair<String, String>> {
        val html = get("https://sjp.pwn.pl/slowniki/${quote(word)}.html") ?: return emptyList()
        val entries = mutableListOf<Pair<String, String>>()
        for (m in pwnHeadword.findAll(html)) {
            val lemma = cleanHtml(m.groupValues[1])
            val li = m.groupValues[2]
            val senses = pwnSense.findAll(li).map { it.groupValues[1] }.toList()
            val defn = if (senses.isNotEmpty()) {
                senses.joinToString(" ") { cleanHtml(it) }
            } else {
                cleanHtml(li.split(pwnBodyEnd).first())
            }
            if (defn.isNotEmpty()) entries += lemma to defn
        }
        return entries.take(3)
    }

    // ── English ───────────────────────────────────────────────────────────────

    /** Wikimedia's REST endpoint; content is CC-BY-SA. */
    private fun fetchWiktionary(word: String): List<Pair<String, String>> {
        val body = get("https://en.wiktionary.org/api/rest_v1/page/definition/${quote(word)}") ?: return emptyList()
        val sections = runCatching { JSONObject(body).optJSONArray("en") }.getOrNull() ?: return emptyList()
        val entries = mutableListOf<Pair<String, String>>()
        for (s in 0 until sections.length()) {
            val section = sections.getJSONObject(s)
            val part = cleanHtml(section.optString("partOfSpeech"))
            val defs = section.optJSONArray("definitions") ?: continue
            for (d in 0 until defs.length()) {
                val defn = cleanHtml(defs.getJSONObject(d).optString("definition"))
                if (defn.isEmpty()) continue
                entries += (if (part.isNotEmpty()) part.lowercase() else word) to defn
                if (entries.size >= 3) return entries
            }
        }
        return entries
    }

    /** dictionaryapi.dev: free and clean, thin on rare words, so second. */
    private fun fetchDictionaryApi(word: String): List<Pair<String, String>> {
        val body = get("https://api.dictionaryapi.dev/api/v2/entries/en/${quote(word)}") ?: return emptyList()
        val data = runCatching { JSONArray(body) }.getOrNull() ?: return emptyList()
        val entries = mutableListOf<Pair<String, String>>()
        for (e in 0 until data.length()) {
            val meanings = data.getJSONObject(e).optJSONArray("meanings") ?: continue
            for (m in 0 until meanings.length()) {
                val meaning = meanings.getJSONObject(m)
                val part = meaning.optString("partOfSpeech")
                val defs = meaning.optJSONArray("definitions") ?: continue
                for (d in 0 until defs.length()) {
                    val defn = defs.getJSONObject(d).optString("definition").trim()
                    if (defn.isEmpty()) continue
                    entries += (part.ifEmpty { word }) to defn
                    if (entries.size >= 3) return entries
                }
            }
        }
        return entries
    }
}
