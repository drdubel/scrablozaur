//! The scan assistant: plays for a rack on a board scanned off a real game,
//! `POST /scan/suggest`. No session -- the board is whatever the player
//! last confirmed, and the bag is what that board leaves.

use super::bot::{rank_suggestions, SortMode, Suggestion, SUGGEST_SIM_ITERATIONS};
use super::session::{GameError, LanguagePack};
use crate::scan::Grid;
use crate::{Board, RACK_SIZE};

/// The ten best plays for `rack` on a board of `letters` (`'-'` for an empty
/// square), the tiles marked in `blanks` being blanks, ordered by `sort`.
pub fn scan_suggestions(
    pack: &LanguagePack,
    letters: &Grid<char>,
    blanks: &Grid<bool>,
    rack: &str,
    sort: SortMode,
) -> Result<Vec<Suggestion>, GameError> {
    if sort.needs_leave_net() && pack.leave_net.is_none() {
        return Err(GameError::SortUnavailable(sort));
    }
    let rack = rack.to_lowercase();
    if rack.chars().count() > RACK_SIZE {
        return Err(GameError::RackTooLong);
    }
    let grid: Vec<Vec<String>> = letters
        .iter()
        .map(|row| row.iter().map(char::to_string).collect())
        .collect();
    let mask: Vec<Vec<bool>> = blanks.iter().map(|row| row.to_vec()).collect();
    let board = Board::from_grid(pack.lang, &grid, Some(&mask))?;
    Ok(rank_suggestions(
        &board,
        &pack.lexicon,
        pack.net(),
        &rack,
        10,
        sort,
        SUGGEST_SIM_ITERATIONS,
        // From the clock, as the web's assistant does.
        0,
    )?)
}
