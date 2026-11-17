package pl.scrablozaur.core

import android.graphics.Bitmap
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer

/** The board last confirmed in the scan assistant, 225 squares row-major. */
data class ScannedBoard(
    val language: String,
    /** A lowercase letter, or "" for an empty square. */
    val letters: List<String>,
    val blanks: List<Boolean>,
)

/** A camera frame kept for saving as a training example: RGBA, `stride` bytes a row. */
class KeptFrame(
    val rgba: ByteArray,
    val width: Int,
    val height: Int,
    val stride: Int,
    /** The grid's corners as x, y pairs, clockwise from the board's own top left. */
    val corners: FloatArray,
    val focalPx: Float,
)

/**
 * The scan assistant's state between visits: the web's ScanSession, which keeps
 * the confirmed board server-side across photos and across leaving the view.
 * Written to a temporary file and renamed, as the game save is.
 */
class ScanSessions(private val dir: File, private val examples: File?) {
    private val file = File(dir, "scan_session.json")

    fun load(): ScannedBoard? = runCatching {
        val json = JSONObject(file.readText())
        val letters = json.getJSONArray("letters")
        val blanks = json.getJSONArray("blanks")
        ScannedBoard(
            language = json.getString("language"),
            letters = (0 until letters.length()).map { letters.getString(it) },
            blanks = (0 until blanks.length()).map { blanks.getBoolean(it) },
        ).takeIf { it.letters.size == 225 && it.blanks.size == 225 }
    }.getOrNull()

    fun save(board: ScannedBoard) {
        val json = JSONObject()
            .put("language", board.language)
            .put("letters", JSONArray(board.letters))
            .put("blanks", JSONArray(board.blanks))
        val tmp = File(file.path + ".tmp")
        tmp.writeText(json.toString())
        tmp.renameTo(file)
    }

    fun clear() {
        file.delete()
    }

    /**
     * The web's "Zapisz do treningu OCR": the frame as a JPEG and what the player
     * confirmed it shows, for `board_reader/`'s training set. They land in the
     * app's external files (`adb pull /sdcard/Android/data/pl.scrablozaur/files/scans`).
     * Returns the example's name.
     */
    fun saveExample(frame: KeptFrame, board: ScannedBoard): String {
        val out = checkNotNull(examples) { "no external storage" }
        out.mkdirs()
        val name = "scan_" + System.currentTimeMillis()
        val bitmap = Bitmap.createBitmap(frame.stride / 4, frame.height, Bitmap.Config.ARGB_8888)
        bitmap.copyPixelsFromBuffer(ByteBuffer.wrap(frame.rgba, 0, frame.stride * frame.height))
        val cropped = Bitmap.createBitmap(bitmap, 0, 0, frame.width, frame.height)
        File(out, "$name.jpg").outputStream().use { cropped.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        val json = JSONObject()
            .put("language", board.language)
            .put("width", frame.width)
            .put("height", frame.height)
            .put("focal_px", frame.focalPx.toDouble())
            .put("corners", JSONArray(frame.corners.map { it.toDouble() }))
            .put("letters", JSONArray(board.letters))
            .put("blanks", JSONArray(board.blanks))
        File(out, "$name.json").writeText(json.toString(1))
        return name
    }
}
