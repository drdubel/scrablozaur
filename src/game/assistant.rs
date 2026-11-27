//! The scan assistant: plays for a rack on a board scanned off a real game,
//! `POST /scan/suggest`. No session -- the board is whatever the player
//! last confirmed, and the bag is what that board leaves.

use super::bot::{
    rank_suggestions, review_equity, EquityCheck, SortMode, Suggestion, TakenTurn,
    SUGGEST_SIM_ITERATIONS,
};
use super::session::{GameError, LanguagePack};
use crate::scan::Grid;
use crate::{Board, RACK_SIZE};

/// The engine's board for `letters` (`'-'` for an empty square) with the tiles
/// marked in `blanks` being blanks.
fn board_of(
    pack: &LanguagePack,
    letters: &Grid<char>,
    blanks: &Grid<bool>,
) -> Result<Board, GameError> {
    let grid: Vec<Vec<String>> = letters
        .iter()
        .map(|row| row.iter().map(char::to_string).collect())
        .collect();
    let mask: Vec<Vec<bool>> = blanks.iter().map(|row| row.to_vec()).collect();
    Ok(Board::from_grid(pack.lang, &grid, Some(&mask))?)
}

fn rack_of(rack: &str) -> Result<String, GameError> {
    let rack = rack.to_lowercase();
    if rack.chars().count() > RACK_SIZE {
        return Err(GameError::RackTooLong);
    }
    Ok(rack)
}

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
    let rack = rack_of(rack)?;
    let board = board_of(pack, letters, blanks)?;
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

/// A play tried on a board of the caller's -- a practice position: whether it
/// is legal there for the rack, what it scores, and what it is up against.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayCheck {
    /// `None` when the play is not one of the rack's legal plays there.
    pub score: Option<u32>,
    /// The best any legal play scores.
    pub best_score: u32,
    /// How many legal plays score more than this one.
    pub better: usize,
    pub plays: usize,
    /// By equity, where the language has a leave net: this play against the
    /// best the static player weighs (no exchanges -- a play was asked for).
    pub equity: Option<EquityCheck>,
}

/// Check `word` at `row`, `col` for `rack` on a board of `letters` and
/// `blanks` (as `scan_suggestions` takes them). Legal means one of the rack's
/// legal plays there, so every rule a game applies, applies.
#[allow(clippy::too_many_arguments)]
pub fn check_play(
    pack: &LanguagePack,
    letters: &Grid<char>,
    blanks: &Grid<bool>,
    rack: &str,
    word: &str,
    row: usize,
    col: usize,
    horizontal: bool,
) -> Result<PlayCheck, GameError> {
    let rack = rack_of(rack)?;
    let board = board_of(pack, letters, blanks)?;
    let word = word.to_lowercase();
    let moves = board.all_moves(&pack.lexicon, &rack, true)?;
    let best_score = moves.iter().map(|m| m.1).max().unwrap_or(0);
    let Some(found) = moves
        .iter()
        .find(|m| m.0 == word && m.2 == (row, col, horizontal))
    else {
        return Ok(PlayCheck {
            score: None,
            best_score,
            better: 0,
            plays: moves.len(),
            equity: None,
        });
    };
    let equity = pack.net().map(|net| {
        let taken = TakenTurn::Play {
            score: found.1,
            used: &found.3,
        };
        review_equity(&board, &pack.lexicon, net, &rack, taken, false)
    });
    Ok(PlayCheck {
        score: Some(found.1),
        best_score,
        better: moves.iter().filter(|m| m.1 > found.1).count(),
        plays: moves.len(),
        equity,
    })
}
