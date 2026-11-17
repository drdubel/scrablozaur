//! The cell reader: a small CNN that reads every square of a straightened
//! board -- empty, a blank, or which letter -- run by the engine itself, so
//! the app and the web read boards identically and no inference library
//! comes along. `board_reader/scripts/train_cell_reader.py` trains it and
//! writes the file loaded here; the `.pt` stays the source of truth.
//!
//! Its input is the board's tiles straightened: `(15 + 2) * cell` pixels
//! square, RGB, one square of margin around the grid, every tile's top face
//! in its square. Each square is read from a crop one and a half squares
//! wide around it.
//!
//! A board can be seen from any side, and which way up it is shows only in
//! its letters, so every letter is `turns` classes: one per quarter turn
//! clockwise that would stand the tile upright, class `2 + turns * letter +
//! turn` after empty (0) and blank (1). One read of the board as it lies
//! says what each tile is and, summed over the tiles, how far to turn the
//! board.

use rayon::prelude::*;

use super::decode::{CellReading, ScanBoard};
use super::net::{Bytes, Map, Net};
use super::Grid;
use crate::{value_error, EngineError, EngineResult, Language, BOARD_SIZE};

const MAGIC: &[u8; 8] = b"SCRBCNN2";

/// The trained reader, loaded from a SCRBCNN2 file.
pub struct CellReader {
    lang: &'static Language,
    /// Pixels per square, and the crop's side.
    cell: usize,
    crop: usize,
    /// Input normalisation: `x * scale + offset` on 0-255 values.
    scale: f32,
    offset: f32,
    /// Softmax temperature that makes held-out probabilities honest.
    temperature: f32,
    /// Classes per letter: 4 if the reader tells turned tiles apart, else 1.
    turns: usize,
    net: Net,
}

/// A board read as it lies: every square's probabilities over the classes.
pub struct BoardReading {
    lang: &'static Language,
    turns: usize,
    /// Per square, row-major as the image lies.
    probs: Vec<Vec<f32>>,
}

impl CellReader {
    pub fn open(lang: &'static Language, path: &str) -> EngineResult<Self> {
        let data = std::fs::read(path).map_err(|e| EngineError::Io(format!("{path}: {e}")))?;
        CellReader::from_bytes(lang, &data).map_err(|e| match e {
            EngineError::Io(m) => EngineError::Io(format!("{path}: {m}")),
            other => other,
        })
    }

    /// Load a reader for `lang` from the bytes of its file. Refuses one
    /// trained on another alphabet: it would not fail, it would read every
    /// tile as confident nonsense.
    pub fn from_bytes(lang: &'static Language, data: &[u8]) -> EngineResult<Self> {
        let bad = |m: &str| EngineError::Io(m.to_string());
        if data.len() < 8 || &data[..8] != MAGIC {
            return Err(bad(
                "not a SCRBCNN2 file (re-run train_cell_reader.py --final)",
            ));
        }
        let mut r = Bytes { data, at: 8 };
        let n_letters = r.u32()?;
        let mut alphabet = String::new();
        for _ in 0..n_letters {
            let cp = r.u32()? as u32;
            alphabet.push(char::from_u32(cp).ok_or_else(|| bad("bad letter in the alphabet"))?);
        }
        if alphabet != lang.alphabet() {
            return Err(bad(&format!(
                "this reader was trained on the alphabet '{alphabet}', not '{}''s -- \
                 is it for a different language?",
                lang.code()
            )));
        }
        let cell = r.u32()?;
        let crop = r.u32()?;
        let scale = r.f32()?;
        let offset = r.f32()?;
        let temperature = r.f32()?;
        let turns = r.u32()?;
        if cell == 0 || crop == 0 || temperature.is_nan() || temperature <= 0.0 {
            return Err(bad("bad reader geometry"));
        }
        if turns != 1 && turns != 4 {
            return Err(bad("a reader's letters span 1 or 4 turns"));
        }
        let net = Net::parse(&mut r, 3)?;
        if !net.flat || net.out_channels != 2 + turns * lang.letters.len() {
            return Err(bad("the reader must end in one score per class"));
        }
        Ok(CellReader {
            lang,
            cell,
            crop,
            scale,
            offset,
            temperature,
            turns,
            net,
        })
    }

    pub fn lang(&self) -> &'static Language {
        self.lang
    }

    /// Pixels per square the reader expects; the board image is
    /// `(15 + 2) * cell_size()` pixels square.
    pub fn cell_size(&self) -> usize {
        self.cell
    }

    /// The scores for one crop, `crop * crop * 3` values 0-255, row-major
    /// RGB: the logits, before temperature. Exists so a test can check the
    /// engine's arithmetic against PyTorch's on identical input.
    pub fn raw(&self, crop: &[f32]) -> EngineResult<Vec<f32>> {
        if crop.len() != self.crop * self.crop * 3 {
            return Err(value_error(format!(
                "expected {} values, got {}",
                self.crop * self.crop * 3,
                crop.len()
            )));
        }
        Ok(self.forward(crop))
    }

    /// Read every square of a straightened board as it lies: `rgb` holds
    /// `side * side` pixels, row-major, three bytes each, with `side == 17 *
    /// cell_size()`.
    pub fn read(&self, rgb: &[u8], side: usize) -> EngineResult<BoardReading> {
        let want = (BOARD_SIZE + 2) * self.cell;
        if side != want || rgb.len() != side * side * 3 {
            return Err(value_error(format!(
                "expected a {want}x{want} RGB board, got {} bytes for {side}x{side}",
                rgb.len()
            )));
        }
        // The crop around square (r, c) starts half a crop before its centre,
        // which sits (r + 1.5) squares in: whole pixels, so no resampling.
        let start = |i: usize| (2 * i + 3) * self.cell / 2 - self.crop / 2;
        let probs = (0..BOARD_SIZE * BOARD_SIZE)
            .into_par_iter()
            .map(|square| {
                let (y0, x0) = (start(square / BOARD_SIZE), start(square % BOARD_SIZE));
                let mut crop = Vec::with_capacity(self.crop * self.crop * 3);
                for y in y0..y0 + self.crop {
                    let row = &rgb[(y * side + x0) * 3..(y * side + x0 + self.crop) * 3];
                    crop.extend(row.iter().map(|&v| v as f32));
                }
                let logits = self.forward(&crop);
                let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let e: Vec<f32> = logits
                    .iter()
                    .map(|&z| ((z - top) / self.temperature).exp())
                    .collect();
                let sum: f32 = e.iter().sum();
                e.into_iter().map(|v| v / sum).collect()
            })
            .collect();
        Ok(BoardReading {
            lang: self.lang,
            turns: self.turns,
            probs,
        })
    }

    /// Every square of a board known to be upright, as the decoder takes it.
    pub fn read_board(&self, rgb: &[u8], side: usize) -> EngineResult<Grid<CellReading>> {
        Ok(self.read(rgb, side)?.cells(0))
    }

    /// The logits for one crop of 0-255 values.
    fn forward(&self, crop: &[f32]) -> Vec<f32> {
        let x = Map {
            h: self.crop,
            w: self.crop,
            c: 3,
            data: crop.iter().map(|&v| v * self.scale + self.offset).collect(),
        };
        self.net.forward(x).data
    }
}

/// The square of the board as it lies that is square `(r, c)` of the board
/// turned `turns` quarter turns clockwise: each turn took (r, c) to
/// (c, n - 1 - r).
fn lying(r: usize, c: usize, turns: usize) -> usize {
    let (mut r, mut c) = (r, c);
    for _ in 0..turns % 4 {
        (r, c) = (BOARD_SIZE - 1 - c, r);
    }
    r * BOARD_SIZE + c
}

impl BoardReading {
    /// The class of `letter` (an alphabet index) on a tile that `turn`
    /// quarter turns clockwise stand upright.
    fn class(&self, letter: usize, turn: usize) -> usize {
        2 + letter * self.turns + if self.turns > 1 { turn % 4 } else { 0 }
    }

    /// How many quarter turns clockwise stand the board upright: the turn
    /// whose letter classes hold the most probability on the squares it
    /// takes for lettered tiles. A grid's corners cannot say -- its premium
    /// squares look the same from all four sides. None if no tile shows, or
    /// the reader cannot tell turned tiles apart.
    pub fn upright(&self) -> Option<usize> {
        if self.turns == 1 {
            return None;
        }
        let n = self.lang.letters.len();
        let tiles: Vec<&Vec<f32>> = self
            .probs
            .iter()
            .filter(|p| p[0] < 0.5 && p[1] < 0.5)
            .collect();
        if tiles.is_empty() {
            return None;
        }
        let score = |turn: usize| -> f32 {
            tiles
                .iter()
                .map(|p| {
                    (0..n)
                        .map(|l| p[self.class(l, turn)])
                        .sum::<f32>()
                        .max(1e-9)
                        .ln()
                })
                .sum()
        };
        let scores = [score(0), score(1), score(2), score(3)];
        (0..4).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
    }

    /// The quarter turns clockwise under which the tiles of `prior`, the
    /// last confirmed board, read as themselves: on a later scan that
    /// settles which way up the board is far more surely than the letters
    /// alone. None if `prior` has no tiles.
    pub fn agreeing_turn(&self, prior: &ScanBoard) -> Option<usize> {
        let mut known = Vec::new(); // (row, col, class at no turn, blank)
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let ch = prior.letters[r][c];
                if ch == '-' {
                    continue;
                }
                if prior.blanks[r][c] {
                    known.push((r, c, None));
                } else if let Some(l) = self.lang.letter_index(ch) {
                    known.push((r, c, Some(l)));
                }
            }
        }
        if known.is_empty() {
            return None;
        }
        let score = |turn: usize| -> f32 {
            known
                .iter()
                .map(|&(r, c, letter)| {
                    let p = &self.probs[lying(r, c, turn)];
                    let class = letter.map_or(1, |l| self.class(l, turn));
                    p[class].max(1e-9).ln()
                })
                .sum()
        };
        let scores = [score(0), score(1), score(2), score(3)];
        (0..4).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
    }

    /// Every square of the board turned `turns` quarter turns clockwise, as
    /// the decoder takes it: P(tile) = 1 - P(empty), the blank's share of
    /// that, and each letter's share of the letters upright at that turn.
    pub fn cells(&self, turns: usize) -> Grid<CellReading> {
        let n = self.lang.letters.len();
        std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                let p = &self.probs[lying(r, c, turns)];
                let p_tile = (1.0 - p[0]).max(1e-6);
                let lettered: f32 = (0..n)
                    .map(|l| p[self.class(l, turns)])
                    .sum::<f32>()
                    .max(1e-9);
                CellReading {
                    p_tile,
                    letters: (0..n)
                        .map(|l| (self.lang.letters[l], p[self.class(l, turns)] / lettered))
                        .collect(),
                    p_blank: (p[1] / p_tile).min(1.0),
                }
            })
        })
    }
}
