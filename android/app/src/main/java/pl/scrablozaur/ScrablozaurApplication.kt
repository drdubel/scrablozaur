package pl.scrablozaur

import android.app.Application
import pl.scrablozaur.core.Definitions
import pl.scrablozaur.core.EngineRepository
import pl.scrablozaur.core.SavedGames

class ScrablozaurApplication : Application() {
    lateinit var engine: EngineRepository
        private set
    lateinit var saves: SavedGames
        private set
    val definitions by lazy { Definitions(cacheDir) }

    override fun onCreate() {
        super.onCreate()
        engine = EngineRepository(this)
        saves = SavedGames(filesDir)
    }
}
