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

use super::decode::CellReading;
use super::net::{Bytes, Map, Net};
use super::Grid;
use crate::{value_error, EngineError, EngineResult, Language, BOARD_SIZE};

const MAGIC: &[u8; 8] = b"SCRBCNN1";

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
