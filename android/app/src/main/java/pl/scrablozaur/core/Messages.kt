package pl.scrablozaur.core

import android.content.Context
import pl.scrablozaur.R
import pl.scrablozaur.engine.MoveException

/** The player-facing wording of an engine refusal. */
fun MoveException.describe(context: Context): String = when (this) {
    is MoveException.NotInDictionary -> context.getString(R.string.err_not_in_dictionary, word.uppercase())
    is MoveException.FirstMoveMustCoverCentre -> context.getString(R.string.err_first_move_centre)
    is MoveException.NotConnected -> context.getString(R.string.err_not_connected)
    is MoveException.TooShort -> context.getString(R.string.err_too_short)
    is MoveException.OutOfBounds -> context.getString(R.string.err_out_of_bounds)
    is MoveException.LetterMismatch -> context.getString(
        R.string.err_letter_mismatch, row.toInt(), col.toInt(), onBoard.uppercase(), inWord.uppercase(),
    )
    is MoveException.CrossWordInvalid -> context.getString(R.string.err_cross_word, word.uppercase())
    is MoveException.NoNewTiles -> context.getString(R.string.err_no_new_tiles)
    is MoveException.WordNotWhole -> context.getString(R.string.err_word_not_whole)
    is MoveException.MissingTiles -> context.getString(R.string.err_missing_tiles)
    is MoveException.TilesNotOnRack -> context.getString(R.string.err_tiles_not_on_rack)
    is MoveException.BagTooSmall -> context.getString(R.string.err_bag_too_small)
    is MoveException.GameOver -> context.getString(R.string.err_game_over)
    is MoveException.NothingToUndo -> context.getString(R.string.err_nothing_to_undo)
    is MoveException.WrongMode -> context.getString(R.string.err_wrong_mode)
    is MoveException.SortUnavailable -> context.getString(R.string.err_sort_unavailable)
    is MoveException.InvalidPlayers -> context.getString(R.string.err_invalid_players)
    is MoveException.RackTooLong -> context.getString(R.string.err_rack_too_long)
    is MoveException.InvalidSave -> context.getString(R.string.err_invalid_save)
    is MoveException.Engine -> context.getString(R.string.err_engine, detail)
}
