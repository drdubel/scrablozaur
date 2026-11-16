//! Turning letter readings from a photographed board into a board the player
//! can confirm: merging them with the last confirmed state, correcting
//! single-letter misreads against the dictionary, and flagging what is still
//! wrong. The port of the dictionary half of `web/scan.py`; the vision that
//! produces the readings lives elsewhere.
//!
//! A scan session tracks a real game being played on a real board, so tiles
//! are only ever added: anything the last confirmed state had should still be
//! there, which is a far stronger signal than the dictionary. Such cells are
//! "locked" -- never corrected, never flagged.

use std::collections::HashMap;

use crate::{Lexicon, BOARD_SIZE};

pub mod decode;

pub type Grid<T> = [[T; BOARD_SIZE]; BOARD_SIZE];

/// Rounds of single-letter correction: fixing one word can change a crossing
/// word's letter too, so a couple of extra rounds let those knock-on fixes
/// settle instead of only fixing whichever word happens to come first.
const CORRECTION_ROUNDS: usize = 3;
/// Alternatives shown per cell in the review step.
pub const MAX_ALTERNATIVES_SHOWN: usize = 4;
/// A classifier's ranked alternatives below this are noise, not second
/// guesses: without the floor, correction happily "fixes" a correct letter
/// into one that completes some other short word.
pub const MIN_ALTERNATIVE_PROB: f32 = 0.15;

/// Every contiguous horizontal, then vertical, run of tiles of length >= 2.
pub fn runs(grid: &Grid<char>) -> Vec<Vec<(usize, usize)>> {
    let mut out = Vec::new();
    for horizontal in [true, false] {
        for line in 0..BOARD_SIZE {
            let at = |i: usize| if horizontal { (line, i) } else { (i, line) };
            let mut i = 0;
            while i < BOARD_SIZE {
                let (r, c) = at(i);
                if grid[r][c] == '-' {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < BOARD_SIZE && {
                    let (r, c) = at(i);
                    grid[r][c] != '-'
                } {
                    i += 1;
                }
                if i - start >= 2 {
                    out.push((start..i).map(at).collect());
                }
            }
        }
    }
    out
}

/// Tiles with no neighbour in either direction -- never valid on a real board
/// (the minimum word is two letters), so almost always a false detection
/// rather than a misread letter.
pub fn isolated_tiles(grid: &Grid<char>) -> Vec<(usize, usize)> {
    let mut in_run = [[false; BOARD_SIZE]; BOARD_SIZE];
    for run in runs(grid) {
        for (r, c) in run {
            in_run[r][c] = true;
        }
    }
    let mut out = Vec::new();
    for r in 0..BOARD_SIZE {
        for c in 0..BOARD_SIZE {
            if grid[r][c] != '-' && !in_run[r][c] {
                out.push((r, c));
            }
        }
    }
    out
}

fn run_word(grid: &Grid<char>, run: &[(usize, usize)]) -> String {
    run.iter().map(|&(r, c)| grid[r][c]).collect()
}

/// Cells still part of an invalid word, or isolated, that are not locked. A run
/// mixing locked and new cells is still checked -- a misread new tile can
/// combine with old ones into a non-word -- but only its new cells are flagged.
pub fn flag_invalid(lexicon: &Lexicon, grid: &Grid<char>, locked: &Grid<bool>) -> Grid<bool> {
    let mut flagged = [[false; BOARD_SIZE]; BOARD_SIZE];
    for run in runs(grid) {
        if run.iter().all(|&(r, c)| locked[r][c]) {
            continue;
        }
        if !lexicon.contains(&run_word(grid, &run)) {
            for (r, c) in run {
                if !locked[r][c] {
                    flagged[r][c] = true;
                }
            }
        }
    }
    for (r, c) in isolated_tiles(grid) {
        if !locked[r][c] {
            flagged[r][c] = true;
        }
    }
    flagged
}

/// Substitute a cell's letter with one of its ranked alternatives wherever
/// that turns an invalid word into a valid one, then flag what is still wrong.
///
/// One substitution per word per round: multi-letter misreads inside one word
/// are rare enough at the classifier's accuracy that a combinatorial search is
/// not worth it, and are left for the review step instead.
pub fn verify_and_correct(
    lexicon: &Lexicon,
    grid: &mut Grid<char>,
    alternatives: &Grid<Vec<char>>,
    locked: &Grid<bool>,
) -> Grid<bool> {
    for _ in 0..CORRECTION_ROUNDS {
        let mut changed = false;
        // Corrections swap letters for letters, never letters for empties, so
        // the runs found at the start of a round stay the runs all round.
        for run in runs(grid) {
            if run.iter().all(|&(r, c)| locked[r][c]) {
                continue;
            }
            if lexicon.contains(&run_word(grid, &run)) {
                continue;
            }
            for &(r, c) in &run {
                if locked[r][c] {
                    continue;
                }
                let original = grid[r][c];
                let mut fixed_here = false;
                for &cand in &alternatives[r][c] {
                    if cand == original {
                        continue;
                    }
                    grid[r][c] = cand;
                    if lexicon.contains(&run_word(grid, &run)) {
                        fixed_here = true;
                        break;
                    }
                    grid[r][c] = original;
                }
                if fixed_here {
                    changed = true;
                    break;
                }
            }
        }
        if !changed {
            break;
        }
    }
    flag_invalid(lexicon, grid, locked)
}

/// What the classifier made of one tile.
#[derive(Clone, Debug, PartialEq)]
pub struct TileReading {
    /// The best guess, already lowercased; `None` if it could not be read.
    pub letter: Option<char>,
    pub confidence: f32,
    /// Ranked guesses with their probabilities, best first.
    pub alternatives: Vec<(char, f32)>,
}

/// One cell of the board offered for review.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanCell {
    /// `'-'` empty, `'?'` a tile nobody could read.
    pub letter: char,
    pub confidence: f32,
    pub alternatives: Vec<char>,
    pub flagged: bool,
    /// The previous confirmed board had a different letter here, and won.
    pub carried_over: bool,
}

/// A photo's readings turned into a board for review (`scan_board_image`'s
/// second half): unreadable or foreign letters become `'?'`, the previous
/// confirmed board overrides this photo wherever it had a tile, and the
/// dictionary pass corrects and flags the rest.
pub fn assemble(
    lexicon: &Lexicon,
    readings: &HashMap<(usize, usize), TileReading>,
    prior: Option<&Grid<char>>,
) -> Grid<ScanCell> {
    let lang = lexicon.lang();
    let is_letter = |ch: char| lang.letter_index(ch).is_some();
    let mut grid = [['-'; BOARD_SIZE]; BOARD_SIZE];
    let mut confidence = [[0f32; BOARD_SIZE]; BOARD_SIZE];
    let mut alternatives: Grid<Vec<char>> =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    for (&(r, c), reading) in readings {
        if r >= BOARD_SIZE || c >= BOARD_SIZE {
            continue;
        }
        grid[r][c] = match reading.letter {
            Some(ch) if is_letter(ch) => ch,
            _ => '?',
        };
        confidence[r][c] = reading.confidence;
        alternatives[r][c] = reading
            .alternatives
            .iter()
            .filter(|&&(ch, p)| p >= MIN_ALTERNATIVE_PROB && is_letter(ch))
            .map(|&(ch, _)| ch)
            .collect();
    }

    let mut locked = [[false; BOARD_SIZE]; BOARD_SIZE];
    let mut carried_over = [[false; BOARD_SIZE]; BOARD_SIZE];
    if let Some(prior) = prior {
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let had = prior[r][c];
                if had == '-' {
                    continue;
                }
                locked[r][c] = true;
                if grid[r][c] != had {
                    // Trust history -- tiles do not change once played -- but
                    // keep this photo's own reading as the first alternative,
                    // in case the prior was the mistake.
                    let fresh = grid[r][c];
                    if fresh != '-' && fresh != '?' && !alternatives[r][c].contains(&fresh) {
                        alternatives[r][c].insert(0, fresh);
                    }
                    grid[r][c] = had;
                    carried_over[r][c] = true;
                }
            }
        }
    }

    let flagged = verify_and_correct(lexicon, &mut grid, &alternatives, &locked);
    std::array::from_fn(|r| {
        std::array::from_fn(|c| ScanCell {
            letter: grid[r][c],
            confidence: (confidence[r][c] * 1000.0).round() / 1000.0,
            alternatives: alternatives[r][c]
                .iter()
                .copied()
                .filter(|&a| a != grid[r][c])
                .take(MAX_ALTERNATIVES_SHOWN)
                .collect(),
            flagged: flagged[r][c],
            carried_over: carried_over[r][c],
        })
    })
}
