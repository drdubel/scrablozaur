package pl.scrablozaur

import android.app.Application
import pl.scrablozaur.core.Definitions
import pl.scrablozaur.core.EngineRepository
import pl.scrablozaur.core.GameHistory
import pl.scrablozaur.core.SavedGames
import pl.scrablozaur.core.ScanSessions
import pl.scrablozaur.core.Settings

class ScrablozaurApplication : Application() {
    lateinit var engine: EngineRepository
        private set
    lateinit var saves: SavedGames
        private set
    lateinit var scans: ScanSessions
        private set
    lateinit var settings: Settings
        private set
    lateinit var history: GameHistory
        private set
    val definitions by lazy { Definitions(cacheDir) }

    override fun onCreate() {
        super.onCreate()
        engine = EngineRepository(this)
        saves = SavedGames(filesDir)
        scans = ScanSessions(filesDir, getExternalFilesDir("scans"))
        settings = Settings(this)
        history = GameHistory(filesDir)
    }
}
