//! The cell reader: a small CNN that reads every square of a straightened
//! board -- empty, a blank, or which letter -- run by the engine itself, so
//! the app and the web read boards identically and no inference library
//! comes along. `board_reader/scripts/train_cell_reader.py` trains it and
//! writes the file loaded here; the `.pt` stays the source of truth.
//!
//! Its input is the board's tiles straightened: `(15 + 2) * cell` pixels
//! square, RGB, one square of margin around the grid, every tile's top face
//! in its square. Each square is read from a crop one and a half squares
//! wide around it, and its softmax over (empty, blank, every letter) becomes
//! the `CellReading` the decoder takes.

use rayon::prelude::*;

use super::decode::{CellReading, ScanBoard};
use super::net::{Bytes, Map, Net};
use super::Grid;
use crate::{value_error, EngineError, EngineResult, Language, BOARD_SIZE};

const MAGIC: &[u8; 8] = b"SCRBCNN1";
/// Tiles `upright` judges a board's orientation by.
const UPRIGHT_TILES: usize = 24;

/// The trained reader, loaded from a SCRBCNN1 file.
pub struct CellReader {
    lang: &'static Language,
    /// Pixels per square, and the crop's side.
    cell: usize,
    crop: usize,
    /// Input normalisation: `x * scale + offset` on 0-255 values.
    scale: f32,
    offset: f32,
    /// Softmax temperature that makes held-out probabilities honest.
    /// Classes are empty, blank, then the language's letters in order.
    temperature: f32,
    net: Net,
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
                "not a SCRBCNN1 file (re-run train_cell_reader.py --final)",
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
        if cell == 0 || crop == 0 || temperature.is_nan() || temperature <= 0.0 {
            return Err(bad("bad reader geometry"));
        }
        let net = Net::parse(&mut r, 3)?;
        if !net.flat || net.out_channels != 2 + lang.letters.len() {
            return Err(bad("the reader must end in one score per class"));
        }
        Ok(CellReader {
            lang,
            cell,
            crop,
            scale,
            offset,
            temperature,
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

    /// Read every square of a straightened board: `rgb` holds `side * side`
    /// pixels, row-major, three bytes each, with `side == 17 * cell_size()`.
    pub fn read_board(&self, rgb: &[u8], side: usize) -> EngineResult<Grid<CellReading>> {
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
        let readings: Vec<CellReading> = (0..BOARD_SIZE * BOARD_SIZE)
            .into_par_iter()
            .map(|square| {
                let (y0, x0) = (start(square / BOARD_SIZE), start(square % BOARD_SIZE));
                let mut crop = Vec::with_capacity(self.crop * self.crop * 3);
                for y in y0..y0 + self.crop {
                    let row = &rgb[(y * side + x0) * 3..(y * side + x0 + self.crop) * 3];
                    crop.extend(row.iter().map(|&v| v as f32));
                }
                self.reading(&self.forward(&crop))
            })
            .collect();
        let mut it = readings.into_iter();
        Ok(std::array::from_fn(|_| {
            std::array::from_fn(|_| it.next().unwrap())
        }))
    }

    /// How many quarter turns clockwise the board image needs for its
    /// letters to stand upright: the turn under which the reader is surest
    /// of the letters on the squares it most takes for tiles. A grid's
    /// corners say nothing about which way up the board is -- its premium
    /// squares look the same from all four sides -- but the letters do. None
    /// if it sees no tiles to tell by.
    pub fn upright(&self, rgb: &[u8], side: usize) -> EngineResult<Option<usize>> {
        let readings = self.read_board(rgb, side)?;
        Ok(self.upright_given(rgb, side, &readings))
    }

    /// `upright`, for a board already read as it lies.
    pub fn upright_given(
        &self,
        rgb: &[u8],
        side: usize,
        readings: &Grid<CellReading>,
    ) -> Option<usize> {
        let mut tiles: Vec<(f32, usize)> = readings
            .iter()
            .flatten()
            .enumerate()
            .filter(|(_, c)| c.p_tile > 0.5 && c.p_blank < 0.5)
            .map(|(i, c)| (c.p_tile, i))
            .collect();
        tiles.sort_by(|a, b| b.0.total_cmp(&a.0));
        tiles.truncate(UPRIGHT_TILES);
        if tiles.is_empty() {
            return None;
        }
        self.best_turn(|k| {
            tiles
                .iter()
                .map(|&(_, square)| {
                    // The best letter's log-probability among letters.
                    let logits = self.forward(&self.turned_crop(rgb, side, square, k));
                    let letters = &logits[2..];
                    let top = letters.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    -letters
                        .iter()
                        .map(|&z| ((z - top) / self.temperature).exp())
                        .sum::<f32>()
                        .ln()
                })
                .sum()
        })
    }

    /// The quarter turns clockwise under which the board image shows the
    /// tiles of `prior`, the last confirmed board, as they were: on a later
    /// scan, that settles which way up the board is far more surely than
    /// letters read cold. None if `prior` has no tiles.
    pub fn agreeing_turn(&self, rgb: &[u8], side: usize, prior: &ScanBoard) -> Option<usize> {
        let mut known: Vec<(usize, usize)> = Vec::new(); // (board square, class)
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                let ch = prior.letters[r][c];
                if ch == '-' {
                    continue;
                }
                let class = if prior.blanks[r][c] {
                    1
                } else {
                    match self.lang.letter_index(ch) {
                        Some(l) => 2 + l,
                        None => continue,
                    }
                };
                known.push((r * BOARD_SIZE + c, class));
            }
        }
        if known.is_empty() {
            return None;
        }
        // Spread the sample over the whole board, not its first rows.
        let step = known.len().div_ceil(UPRIGHT_TILES);
        let known: Vec<(usize, usize)> = known.into_iter().step_by(step).collect();
        self.best_turn(|k| {
            known
                .iter()
                .map(|&(square, class)| {
                    // Where board square (r, c) lies in the image before the
                    // turns: each clockwise turn took (r, c) to (c, n - 1 - r).
                    let (mut r, mut c) = (square / BOARD_SIZE, square % BOARD_SIZE);
                    for _ in 0..k {
                        (r, c) = (BOARD_SIZE - 1 - c, r);
                    }
                    let logits = self.forward(&self.turned_crop(rgb, side, r * BOARD_SIZE + c, k));
                    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let sum: f32 = logits
                        .iter()
                        .map(|&z| ((z - top) / self.temperature).exp())
                        .sum();
                    (logits[class] - top) / self.temperature - sum.ln()
                })
                .sum()
        })
    }

    /// The turn, of the four, scoring highest.
    fn best_turn(&self, score: impl Fn(usize) -> f32 + Sync + Send) -> Option<usize> {
        let scores: Vec<f32> = (0..4).into_par_iter().map(&score).collect();
        (0..4).max_by(|&a, &b| scores[a].total_cmp(&scores[b]))
    }

    /// The crop around image square `square`, turned `k` quarter turns
    /// clockwise: what that square shows once the board is turned so.
    fn turned_crop(&self, rgb: &[u8], side: usize, square: usize, k: usize) -> Vec<f32> {
        let start = |i: usize| (2 * i + 3) * self.cell / 2 - self.crop / 2;
        let (y0, x0) = (start(square / BOARD_SIZE), start(square % BOARD_SIZE));
        let n = self.crop;
        let mut crop = Vec::with_capacity(n * n * 3);
        for y in 0..n {
            for x in 0..n {
                // Output (y, x) comes from the source pixel the turn moves there.
                let (sy, sx) = match k % 4 {
                    0 => (y, x),
                    1 => (n - 1 - x, y),
                    2 => (n - 1 - y, n - 1 - x),
                    _ => (x, n - 1 - y),
                };
                let at = ((y0 + sy) * side + x0 + sx) * 3;
                crop.extend(rgb[at..at + 3].iter().map(|&v| v as f32));
            }
        }
        crop
    }

    /// Softmax with the calibration temperature, turned into what the
    /// decoder takes: P(tile) = 1 - P(empty), the blank's share of that, and
    /// each letter's share of the rest.
    fn reading(&self, logits: &[f32]) -> CellReading {
        let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = logits
            .iter()
            .map(|&z| ((z - top) / self.temperature).exp())
            .collect();
        let sum: f32 = e.iter().sum();
        let p: Vec<f32> = e.iter().map(|&v| v / sum).collect();
        let p_tile = (1.0 - p[0]).max(1e-6);
        let lettered = (p_tile - p[1]).max(1e-6);
        CellReading {
            p_tile,
            letters: self
                .lang
                .letters
                .iter()
                .enumerate()
                .map(|(i, &l)| (l, p[2 + i] / lettered))
                .collect(),
            p_blank: p[1] / p_tile,
        }
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
