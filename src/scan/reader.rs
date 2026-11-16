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
use super::Grid;
use crate::{value_error, EngineError, EngineResult, Language, BOARD_SIZE};

const MAGIC: &[u8; 8] = b"SCRBCNN1";

enum Layer {
    /// 3x3, padding 1, batch norm folded in, then ReLU. Weights are
    /// `[ky][kx][cin][cout]`, the order the patch gather produces.
    Conv {
        cin: usize,
        cout: usize,
        w: Vec<f32>,
        b: Vec<f32>,
    },
    /// 2x2, stride 2.
    MaxPool,
    /// Every pixel's channels averaged into one vector.
    Mean,
    /// Weights `[out][in]`.
    Linear {
        cin: usize,
        cout: usize,
        w: Vec<f32>,
        b: Vec<f32>,
    },
}

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
    layers: Vec<Layer>,
}

/// Little-endian reads over a byte slice, failing instead of panicking.
struct Bytes<'a> {
    data: &'a [u8],
    at: usize,
}

impl Bytes<'_> {
    fn take(&mut self, n: usize) -> EngineResult<&[u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.data.len());
        let end = end.ok_or_else(|| EngineError::Io("truncated cell reader".into()))?;
        let out = &self.data[self.at..end];
        self.at = end;
        Ok(out)
    }

    fn u32(&mut self) -> EngineResult<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }

    fn f32(&mut self) -> EngineResult<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32s(&mut self, n: usize) -> EngineResult<Vec<f32>> {
        Ok(self
            .take(
                n.checked_mul(4)
                    .ok_or_else(|| EngineError::Io("bad layer size".into()))?,
            )?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    }
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
        let n_layers = r.u32()?;
        let mut layers = Vec::with_capacity(n_layers);
        let (mut channels, mut side, mut flat) = (3usize, crop, false);
        for _ in 0..n_layers {
            let layer = match r.u32()? {
                0 => {
                    let (cin, cout) = (r.u32()?, r.u32()?);
                    if flat || cin != channels {
                        return Err(bad("convolution does not fit the layer before"));
                    }
                    if cout > MAX_CHANNELS {
                        return Err(bad("convolution wider than the engine runs"));
                    }
                    channels = cout;
                    Layer::Conv {
                        cin,
                        cout,
                        w: r.f32s(9 * cin * cout)?,
                        b: r.f32s(cout)?,
                    }
                }
                1 => {
                    if flat || side % 2 != 0 {
                        return Err(bad("pooling an odd-sized map"));
                    }
                    side /= 2;
                    Layer::MaxPool
                }
                2 => {
                    flat = true;
                    Layer::Mean
                }
                3 => {
                    let (cin, cout) = (r.u32()?, r.u32()?);
                    if !flat || cin != channels {
                        return Err(bad("linear layer does not fit the layer before"));
                    }
                    channels = cout;
                    Layer::Linear {
                        cin,
                        cout,
                        w: r.f32s(cin * cout)?,
                        b: r.f32s(cout)?,
                    }
                }
                k => return Err(bad(&format!("unknown layer kind {k}"))),
            };
            layers.push(layer);
        }
        if !flat || channels != 2 + lang.letters.len() {
            return Err(bad("the reader must end in one score per class"));
        }
        Ok(CellReader {
            lang,
            cell,
            crop,
            scale,
            offset,
            temperature,
            layers,
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
        let mut side = self.crop;
        let mut x: Vec<f32> = crop.iter().map(|&v| v * self.scale + self.offset).collect();
        let mut patches = Vec::new();
        for layer in &self.layers {
            x = match layer {
                Layer::Conv { cin, cout, w, b } => {
                    conv3x3(&x, side, *cin, *cout, w, b, &mut patches)
                }
                Layer::MaxPool => max_pool(&x, side, x.len() / (side * side)),
                Layer::Mean => {
                    let c = x.len() / (side * side);
                    let mut out = vec![0f32; c];
                    for px in x.chunks_exact(c) {
                        for (o, &v) in out.iter_mut().zip(px) {
                            *o += v;
                        }
                    }
                    let n = (side * side) as f32;
                    out.iter_mut().for_each(|o| *o /= n);
                    out
                }
                Layer::Linear { cin, cout, w, b } => (0..*cout)
                    .map(|o| {
                        b[o] + w[o * cin..(o + 1) * cin]
                            .iter()
                            .zip(&x)
                            .map(|(a, v)| a * v)
                            .sum::<f32>()
                    })
                    .collect(),
            };
            if let Layer::MaxPool = layer {
                side /= 2;
            }
        }
        x
    }
}

/// How many output pixels the convolution computes together: each weight
/// loaded serves this many of them.
const ROWS: usize = 4;
/// The widest convolution: its accumulators live on the stack.
const MAX_CHANNELS: usize = 256;

/// A 3x3 convolution with padding 1 and a ReLU, on an HWC map `side` pixels
/// square. Each output pixel's 3x3xcin patch is gathered into a row of
/// `patches`, and rows are multiplied by the weights ROWS at a time, the
/// output channels innermost so the compiler can vectorise them.
fn conv3x3(
    x: &[f32],
    side: usize,
    cin: usize,
    cout: usize,
    w: &[f32],
    b: &[f32],
    patches: &mut Vec<f32>,
) -> Vec<f32> {
    let k = 9 * cin;
    let pixels = side * side;
    patches.clear();
    patches.resize(pixels * k, 0.0);
    for y in 0..side {
        for xx in 0..side {
            let row = &mut patches[(y * side + xx) * k..(y * side + xx + 1) * k];
            for ky in 0..3 {
                let sy = y as isize + ky as isize - 1;
                if sy < 0 || sy >= side as isize {
                    continue; // zeros, as gathered
                }
                for kx in 0..3 {
                    let sx = xx as isize + kx as isize - 1;
                    if sx < 0 || sx >= side as isize {
                        continue;
                    }
                    let src = (sy as usize * side + sx as usize) * cin;
                    let dst = (ky * 3 + kx) * cin;
                    row[dst..dst + cin].copy_from_slice(&x[src..src + cin]);
                }
            }
        }
    }
    let mut out = vec![0f32; pixels * cout];
    for p0 in (0..pixels).step_by(ROWS) {
        let n = ROWS.min(pixels - p0);
        let mut acc = [[0f32; MAX_CHANNELS]; ROWS];
        for a in acc.iter_mut().take(n) {
            a[..cout].copy_from_slice(b);
        }
        for kk in 0..k {
            let wrow = &w[kk * cout..(kk + 1) * cout];
            for (r, a) in acc.iter_mut().enumerate().take(n) {
                let v = patches[(p0 + r) * k + kk];
                if v != 0.0 {
                    for (o, &wv) in a[..cout].iter_mut().zip(wrow) {
                        *o += v * wv;
                    }
                }
            }
        }
        for (r, a) in acc.iter().enumerate().take(n) {
            for (o, &v) in out[(p0 + r) * cout..(p0 + r + 1) * cout]
                .iter_mut()
                .zip(&a[..cout])
            {
                *o = v.max(0.0);
            }
        }
    }
    out
}

fn max_pool(x: &[f32], side: usize, c: usize) -> Vec<f32> {
    let half = side / 2;
    let mut out = vec![f32::NEG_INFINITY; half * half * c];
    for y in 0..half * 2 {
        for xx in 0..half * 2 {
            let src = &x[(y * side + xx) * c..(y * side + xx + 1) * c];
            let dst = &mut out[((y / 2) * half + xx / 2) * c..((y / 2) * half + xx / 2 + 1) * c];
            for (o, &v) in dst.iter_mut().zip(src) {
                *o = o.max(v);
            }
        }
    }
    out
}
