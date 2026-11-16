//! The engine's own small-CNN runtime, what the cell reader and the grid
//! finder are made of: 3x3 convolutions (batch norm folded in by the
//! exporter, dilated or not, ReLU or not), 2x2 max pooling, a global mean,
//! and a linear layer. Enough for both nets, small enough to check against
//! PyTorch to the last few bits, and no inference library in the app.
//!
//! Feature maps are HWC: channels innermost, so a convolution's output
//! channels are the contiguous inner loop the compiler vectorises.

use rayon::prelude::*;

use crate::{EngineError, EngineResult};

/// How many output pixels a convolution computes together: each weight
/// loaded serves this many of them.
const ROWS: usize = 4;
/// The widest convolution: its accumulators live on the stack.
const MAX_CHANNELS: usize = 256;
/// Maps with at least this many pixels are convolved on several threads.
const PARALLEL_PIXELS: usize = 4096;

pub(crate) enum Layer {
    /// 3x3, padding = dilation, batch norm folded in. Weights are
    /// `[ky][kx][cin][cout]`, the order the patch gather produces.
    Conv {
        cin: usize,
        cout: usize,
        dilation: usize,
        relu: bool,
        w: Vec<f32>,
        b: Vec<f32>,
    },
    /// 2x2, stride 2, dropping an odd last row or column as PyTorch does.
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

/// A feature map, HWC.
pub(crate) struct Map {
    pub h: usize,
    pub w: usize,
    pub c: usize,
    pub data: Vec<f32>,
}

/// Little-endian reads over a byte slice, failing instead of panicking.
pub(crate) struct Bytes<'a> {
    pub data: &'a [u8],
    pub at: usize,
}

impl Bytes<'_> {
    pub fn take(&mut self, n: usize) -> EngineResult<&[u8]> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.data.len());
        let end = end.ok_or_else(|| EngineError::Io("truncated net".into()))?;
        let out = &self.data[self.at..end];
        self.at = end;
        Ok(out)
    }

    pub fn u32(&mut self) -> EngineResult<usize> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }

    pub fn f32(&mut self) -> EngineResult<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn f32s(&mut self, n: usize) -> EngineResult<Vec<f32>> {
        let len = n
            .checked_mul(4)
            .ok_or_else(|| EngineError::Io("bad layer size".into()))?;
        Ok(self
            .take(len)?
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect())
    }
}

pub(crate) struct Net {
    layers: Vec<Layer>,
    /// Channels of the output, and whether it is a vector (after `Mean`).
    pub out_channels: usize,
    pub flat: bool,
}

impl Net {
    /// A layer count and the layers, for an input of `channels` channels.
    /// Kinds: 0 a 3x3 convolution with ReLU, 1 max pooling, 2 the mean,
    /// 3 linear, 4 a 3x3 convolution with its dilation and whether a ReLU
    /// follows.
    pub fn parse(r: &mut Bytes, mut channels: usize) -> EngineResult<Net> {
        let bad = |m: &str| EngineError::Io(m.to_string());
        let n_layers = r.u32()?;
        let mut layers = Vec::with_capacity(n_layers.min(64));
        let mut flat = false;
        for _ in 0..n_layers {
            let kind = r.u32()?;
            let layer = match kind {
                0 | 4 => {
                    let (cin, cout) = (r.u32()?, r.u32()?);
                    let (dilation, relu) = if kind == 4 {
                        (r.u32()?, r.u32()? != 0)
                    } else {
                        (1, true)
                    };
                    if flat || cin != channels {
                        return Err(bad("convolution does not fit the layer before"));
                    }
                    if cout == 0 || cout > MAX_CHANNELS || dilation == 0 {
                        return Err(bad("convolution the engine cannot run"));
                    }
                    channels = cout;
                    Layer::Conv {
                        cin,
                        cout,
                        dilation,
                        relu,
                        w: r.f32s(9 * cin * cout)?,
                        b: r.f32s(cout)?,
                    }
                }
                1 if !flat => Layer::MaxPool,
                2 if !flat => {
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
                k => return Err(bad(&format!("layer kind {k} cannot come here"))),
            };
            layers.push(layer);
        }
        Ok(Net {
            layers,
            out_channels: channels,
            flat,
        })
    }

    pub fn forward(&self, mut x: Map) -> Map {
        let mut patches = Vec::new();
        for layer in &self.layers {
            x = match layer {
                Layer::Conv {
                    cin,
                    cout,
                    dilation,
                    relu,
                    w,
                    b,
                } => conv3x3(&x, *cin, *cout, *dilation, *relu, w, b, &mut patches),
                Layer::MaxPool => max_pool(&x),
                Layer::Mean => {
                    let mut out = vec![0f32; x.c];
                    for px in x.data.chunks_exact(x.c) {
                        for (o, &v) in out.iter_mut().zip(px) {
                            *o += v;
                        }
                    }
                    let n = (x.h * x.w) as f32;
                    out.iter_mut().for_each(|o| *o /= n);
                    Map {
                        h: 1,
                        w: 1,
                        c: x.c,
                        data: out,
                    }
                }
                Layer::Linear { cin, cout, w, b } => Map {
                    h: 1,
                    w: 1,
                    c: *cout,
                    data: (0..*cout)
                        .map(|o| {
                            b[o] + w[o * cin..(o + 1) * cin]
                                .iter()
                                .zip(&x.data)
                                .map(|(a, v)| a * v)
                                .sum::<f32>()
                        })
                        .collect(),
                },
            };
        }
        x
    }
}

/// A 3x3 convolution, padding and dilation `d`. Each output pixel's 3x3xcin
/// patch is gathered into a row of `patches`, and rows are multiplied by the
/// weights ROWS at a time, output channels innermost. A big map (the grid
/// finder's whole frame) is split across threads; a small one (a cell
/// reader's crop, read alongside 224 others) is not worth it.
#[allow(clippy::too_many_arguments)]
fn conv3x3(
    x: &Map,
    cin: usize,
    cout: usize,
    d: usize,
    relu: bool,
    w: &[f32],
    b: &[f32],
    patches: &mut Vec<f32>,
) -> Map {
    let k = 9 * cin;
    let pixels = x.h * x.w;
    patches.clear();
    patches.resize(pixels * k, 0.0);
    let mut out = vec![0f32; pixels * cout];
    let layer = Conv {
        k,
        cout,
        relu,
        w,
        b,
    };
    if pixels >= PARALLEL_PIXELS {
        patches
            .par_chunks_mut(x.w * k)
            .enumerate()
            .for_each(|(y, rows)| gather(x, cin, d, y, rows));
        let patches = patches.as_slice();
        out.par_chunks_mut(ROWS * cout)
            .enumerate()
            .for_each(|(i, chunk)| layer.multiply(patches, i * ROWS, chunk));
    } else {
        for (y, rows) in patches.chunks_mut(x.w * k).enumerate() {
            gather(x, cin, d, y, rows);
        }
        for (i, chunk) in out.chunks_mut(ROWS * cout).enumerate() {
            layer.multiply(patches, i * ROWS, chunk);
        }
    }
    Map {
        h: x.h,
        w: x.w,
        c: cout,
        data: out,
    }
}

/// Row `y`'s 3x3xcin patches, one after another into `rows`; zeros where a
/// patch hangs off the map, as `rows` came.
fn gather(x: &Map, cin: usize, d: usize, y: usize, rows: &mut [f32]) {
    let k = 9 * cin;
    for xx in 0..x.w {
        let row = &mut rows[xx * k..(xx + 1) * k];
        for ky in 0..3 {
            let sy = y as isize + (ky as isize - 1) * d as isize;
            if sy < 0 || sy >= x.h as isize {
                continue;
            }
            for kx in 0..3 {
                let sx = xx as isize + (kx as isize - 1) * d as isize;
                if sx < 0 || sx >= x.w as isize {
                    continue;
                }
                let src = (sy as usize * x.w + sx as usize) * cin;
                let dst = (ky * 3 + kx) * cin;
                row[dst..dst + cin].copy_from_slice(&x.data[src..src + cin]);
            }
        }
    }
}

struct Conv<'a> {
    k: usize,
    cout: usize,
    relu: bool,
    w: &'a [f32],
    b: &'a [f32],
}

impl Conv<'_> {
    /// Output pixels `p0..` into `out`, up to ROWS of them.
    fn multiply(&self, patches: &[f32], p0: usize, out: &mut [f32]) {
        let (k, cout) = (self.k, self.cout);
        let n = out.len() / cout;
        let mut acc = [[0f32; MAX_CHANNELS]; ROWS];
        for a in acc.iter_mut().take(n) {
            a[..cout].copy_from_slice(self.b);
        }
        for kk in 0..k {
            let wrow = &self.w[kk * cout..(kk + 1) * cout];
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
            for (o, &v) in out[r * cout..(r + 1) * cout].iter_mut().zip(&a[..cout]) {
                *o = if self.relu { v.max(0.0) } else { v };
            }
        }
    }
}

fn max_pool(x: &Map) -> Map {
    let (h, w, c) = (x.h / 2, x.w / 2, x.c);
    let mut out = vec![f32::NEG_INFINITY; h * w * c];
    for y in 0..h * 2 {
        for xx in 0..w * 2 {
            let src = &x.data[(y * x.w + xx) * c..(y * x.w + xx + 1) * c];
            let at = ((y / 2) * w + xx / 2) * c;
            for (o, &v) in out[at..at + c].iter_mut().zip(src) {
                *o = o.max(v);
            }
        }
    }
    Map { h, w, c, data: out }
}
