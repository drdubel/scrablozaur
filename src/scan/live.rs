//! Watching a camera feed for a board to read: when a frame is worth
//! reading, and when a reading is worth keeping. The app's camera screen
//! feeds it every frame's grid and every reading it makes.
//!
//! A frame is read once the grid has stayed put for `STEADY_SECONDS`, no
//! corner straying more than `STEADY_SQUARES` squares from where it was: a
//! board still moving blurs, and a scan the player did not mean to make is
//! worse than one made half a second late. A reading is accepted once two
//! in a row, the board held still between them, give the same legal board
//! -- every word in the dictionary, every tile joined up -- so one lucky
//! frame cannot slip a misread through.
//!
//! Legal is not enough on its own: the decoder keeps the last confirmed
//! board's tiles whatever the frame shows, so a grid found in something that
//! is not the board -- a tiled floor, a wall -- decodes to that board,
//! legal. A reading only counts if the reader itself saw nearly all of its
//! tiles.

use super::decode::{CellReading, Decoded, ScanBoard};
use super::grid::Quad;
use super::Grid;
use crate::BOARD_SIZE;

/// How long the grid must stay put before a frame is read.
pub const STEADY_SECONDS: f64 = 0.4;
/// How far a corner may wander, in squares, with the board still counted
/// as held still.
pub const STEADY_SQUARES: f64 = 0.2;
/// The share of a reading's tiles the reader must have seen as tiles, not
/// left to the decoder to keep or add.
pub const SEEN_SHARE: f32 = 0.9;
/// After a reading that cannot be accepted, how long until the next of a
/// board that has not moved.
pub const RETRY_SECONDS: f64 = 1.0;

/// What the latest frame shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Watch {
    /// No grid.
    NoBoard,
    /// A grid somewhere other than where it last rested: whatever was read
    /// before no longer holds.
    Moving,
    /// A grid at rest, not yet for long enough to read.
    Settling,
    /// A grid that has stayed put: this frame is worth reading.
    Steady,
}

/// What a reading of a steady frame is worth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Not a board the reader saw: under two tiles, or too few of them read
    /// as tiles at all.
    Unseen,
    /// A line that is not a word, or a tile on its own.
    Illegal,
    /// Legal: waiting for the next reading to agree.
    Checking,
    /// The second reading in a row to give this same legal board.
    Accepted,
}

/// The state between frames.
#[derive(Default)]
pub struct LiveScan {
    /// Where the grid was when it last came to rest, and when.
    rest: Option<(f64, Quad)>,
    /// When the latest frame was taken.
    now: f64,
    /// No reading before this, unless the board moves.
    retry_at: f64,
    /// The last reading's board, if legal and made since the board came to
    /// rest.
    last: Option<ScanBoard>,
}

impl LiveScan {
    /// A frame taken at `seconds`, and the grid found in it.
    pub fn watch(&mut self, seconds: f64, corners: Option<&Quad>) -> Watch {
        self.now = seconds;
        let Some(corners) = corners else {
            *self = LiveScan::default();
            return Watch::NoBoard;
        };
        match self.rest {
            Some((since, rest)) if near(&rest, corners) => {
                if seconds - since >= STEADY_SECONDS && seconds >= self.retry_at {
                    Watch::Steady
                } else {
                    Watch::Settling
                }
            }
            _ => {
                self.rest = Some((seconds, *corners));
                self.last = None;
                self.retry_at = 0.0;
                Watch::Moving
            }
        }
    }

    /// A steady frame's reading: `decoded` from `readings`. One that cannot
    /// be accepted puts off the next reading of the board where it lies by
    /// `RETRY_SECONDS`: a grid found in a wall, or a board with a word the
    /// dictionary lacks, would otherwise be read back to back for as long as
    /// the camera looks, every core busy.
    pub fn judge(&mut self, decoded: &Decoded, readings: &Grid<CellReading>) -> Verdict {
        let tiles: Vec<(usize, usize)> = (0..BOARD_SIZE * BOARD_SIZE)
            .map(|i| (i / BOARD_SIZE, i % BOARD_SIZE))
            .filter(|&(r, c)| decoded.board.letters[r][c] != '-')
            .collect();
        let seen = tiles
            .iter()
            .filter(|&&(r, c)| readings[r][c].p_tile >= 0.5)
            .count();
        let verdict = if tiles.len() < 2 || (seen as f32) < SEEN_SHARE * tiles.len() as f32 {
            Verdict::Unseen
        } else if !decoded.legal {
            Verdict::Illegal
        } else if self.last.as_ref() == Some(&decoded.board) {
            return Verdict::Accepted;
        } else {
            self.last = Some(decoded.board.clone());
            return Verdict::Checking;
        };
        self.last = None;
        self.retry_at = self.now + RETRY_SECONDS;
        verdict
    }
}

/// Every corner of `b` within `STEADY_SQUARES` squares of `a`'s.
fn near(a: &Quad, b: &Quad) -> bool {
    let dist = |p: [f64; 2], q: [f64; 2]| ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2)).sqrt();
    let perimeter: f64 = (0..4).map(|i| dist(a[i], a[(i + 1) % 4])).sum();
    let square = perimeter / (4 * BOARD_SIZE) as f64;
    (0..4).all(|i| dist(a[i], b[i]) <= STEADY_SQUARES * square)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 600-pixel grid, 40 pixels a square, moved `dx` pixels right.
    fn grid(dx: f64) -> Quad {
        [
            [100.0 + dx, 100.0],
            [700.0 + dx, 100.0],
            [700.0 + dx, 700.0],
            [100.0 + dx, 700.0],
        ]
    }

    /// A legal reading of `letter` and 'a' across the centre.
    fn legal(letter: char) -> Decoded {
        let mut board = ScanBoard::empty();
        board.letters[7][7] = letter;
        board.letters[7][8] = 'a';
        Decoded {
            board,
            corrected: [[false; BOARD_SIZE]; BOARD_SIZE],
            flagged: [[false; BOARD_SIZE]; BOARD_SIZE],
            legal: true,
            complete: true,
            cost: 0.0,
            work: 0,
        }
    }

    #[test]
    fn a_board_held_still_is_read_after_a_while() {
        let mut live = LiveScan::default();
        assert_eq!(live.watch(0.0, Some(&grid(0.0))), Watch::Moving);
        assert_eq!(live.watch(0.2, Some(&grid(4.0))), Watch::Settling);
        // 8 pixels is a fifth of a square: still at rest.
        assert_eq!(live.watch(0.4, Some(&grid(8.0))), Watch::Steady);
        // Drifting further counts as moving, and the wait starts over.
        assert_eq!(live.watch(0.5, Some(&grid(10.0))), Watch::Moving);
        assert_eq!(live.watch(0.8, Some(&grid(10.0))), Watch::Settling);
        assert_eq!(live.watch(0.95, Some(&grid(10.0))), Watch::Steady);
        assert_eq!(live.watch(1.0, None), Watch::NoBoard);
        assert_eq!(live.watch(1.1, Some(&grid(10.0))), Watch::Moving);
    }

    /// Readings that saw tiles on `squares` and nothing elsewhere.
    fn seeing(squares: &[(usize, usize)]) -> Grid<CellReading> {
        let mut grid: Grid<CellReading> =
            std::array::from_fn(|_| std::array::from_fn(|_| CellReading::empty()));
        for &(r, c) in squares {
            grid[r][c].p_tile = 0.99;
        }
        grid
    }

    #[test]
    fn two_legal_readings_in_a_row_must_agree() {
        let seen = seeing(&[(7, 7), (7, 8)]);
        let mut live = LiveScan::default();
        live.watch(0.0, Some(&grid(0.0)));
        assert_eq!(live.judge(&legal('n'), &seen), Verdict::Checking);
        assert_eq!(live.judge(&legal('t'), &seen), Verdict::Checking);
        assert_eq!(live.judge(&legal('t'), &seen), Verdict::Accepted);
        // An illegal reading in between breaks the run.
        let mut bad = legal('t');
        bad.legal = false;
        assert_eq!(live.judge(&bad, &seen), Verdict::Illegal);
        assert_eq!(live.judge(&legal('t'), &seen), Verdict::Checking);
        assert_eq!(live.judge(&legal('t'), &seen), Verdict::Accepted);
        // So does the board moving.
        live.watch(1.0, Some(&grid(100.0)));
        assert_eq!(live.judge(&legal('t'), &seen), Verdict::Checking);
    }

    #[test]
    fn a_board_the_reader_did_not_see_is_not_accepted() {
        let mut live = LiveScan::default();
        live.watch(0.0, Some(&grid(0.0)));
        // The decoder kept two tiles the reader saw nothing of: a grid found in
        // something that is not the board.
        let blind = seeing(&[]);
        assert_eq!(live.watch(0.4, Some(&grid(0.0))), Watch::Steady);
        assert_eq!(live.judge(&legal('t'), &blind), Verdict::Unseen);
        // The next reading waits a while, unless the board moves.
        assert_eq!(live.watch(0.8, Some(&grid(0.0))), Watch::Settling);
        assert_eq!(live.watch(1.4, Some(&grid(0.0))), Watch::Steady);
        assert_eq!(live.judge(&legal('t'), &blind), Verdict::Unseen);
        assert_eq!(live.watch(1.5, Some(&grid(40.0))), Watch::Moving);
        assert_eq!(live.watch(2.0, Some(&grid(40.0))), Watch::Steady);
        // An empty board is never a scan.
        let mut empty = legal('t');
        empty.board = ScanBoard::empty();
        assert_eq!(live.judge(&empty, &blind), Verdict::Unseen);
    }
}
