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

/// Output pixels a tile computes at once: each weight loaded serves this
/// many of them.
const ROWS: usize = 4;
/// The widest convolution the engine runs.
const MAX_CHANNELS: usize = 256;
/// Output channels a tile computes at once: 16 floats, four NEON registers.
const LANES: usize = 16;
/// Maps with at least this many pixels are convolved on several threads.
const PARALLEL_PIXELS: usize = 4096;

pub(crate) enum Layer {
    /// 3x3, padding = dilation, batch norm folded in. The file's weights
    /// are `[ky][kx][cin][cout]`, the order the patch gather produces; they
    /// are kept repacked into panels of LANES output channels, `[block][k]
    /// [LANES]`, zero beyond `cout`, so a tile streams one contiguous panel.
    Conv {
        cin: usize,
        cout: usize,
        dilation: usize,
        relu: bool,
        panels: Vec<f32>,
        bias: Vec<f32>,
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
                    let (w, b) = (r.f32s(9 * cin * cout)?, r.f32s(cout)?);
                    let (panels, bias) = repack(&w, &b, 9 * cin, cout);
                    Layer::Conv {
                        cin,
                        cout,
                        dilation,
                        relu,
                        panels,
                        bias,
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
        for layer in &self.layers {
            x = match layer {
                Layer::Conv {
                    cin,
                    cout,
                    dilation,
                    relu,
                    panels,
                    bias,
                } => conv3x3(&x, *cin, *cout, *dilation, *relu, panels, bias),
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

/// A 3x3 convolution, padding and dilation `d`, computed ROWS output
/// pixels at a time: their 3x3xcin patches gathered into a small buffer,
/// then multiplied by the weight panels. Gathering every pixel's patch up
/// front made a buffer of over a megabyte per crop, past a phone core's L2
/// cache -- the arithmetic waited on memory there, not on a laptop. A big
/// map (the grid finder's whole frame) is split across threads; a small one
/// (a cell reader's crop, read alongside 224 others) is not worth it.
#[allow(clippy::too_many_arguments)]
fn conv3x3(
    x: &Map,
    cin: usize,
    cout: usize,
    d: usize,
    relu: bool,
    panels: &[f32],
    bias: &[f32],
) -> Map {
    let k = 9 * cin;
    let pixels = x.h * x.w;
    let mut out = vec![0f32; pixels * cout];
    let layer = Conv {
        k,
        cout,
        relu,
        panels,
        bias,
    };
    let block = |i: usize, chunk: &mut [f32], patches: &mut Vec<f32>| {
        let n = chunk.len() / cout;
        patches.clear();
        patches.resize(n * k, 0.0);
        for r in 0..n {
            gather(x, cin, d, i * ROWS + r, &mut patches[r * k..(r + 1) * k]);
        }
        layer.multiply(patches, 0, chunk);
    };
    if pixels >= PARALLEL_PIXELS {
        out.par_chunks_mut(ROWS * cout)
            .enumerate()
            .for_each_init(Vec::new, |patches, (i, chunk)| block(i, chunk, patches));
    } else {
        let mut patches = Vec::with_capacity(ROWS * k);
        for (i, chunk) in out.chunks_mut(ROWS * cout).enumerate() {
            block(i, chunk, &mut patches);
        }
    }
    Map {
        h: x.h,
        w: x.w,
        c: cout,
        data: out,
    }
}

/// Pixel `p`'s 3x3xcin patch into `row`, which comes zeroed: zeros stay
/// where the patch hangs off the map.
fn gather(x: &Map, cin: usize, d: usize, p: usize, row: &mut [f32]) {
    let (y, xx) = (p / x.w, p % x.w);
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

struct Conv<'a> {
    k: usize,
    cout: usize,
    relu: bool,
    panels: &'a [f32],
    bias: &'a [f32],
}

impl Conv<'_> {
    /// Output pixels `p0..` into `out`, up to ROWS of them, one panel of
    /// LANES channels at a time.
    fn multiply(&self, patches: &[f32], p0: usize, out: &mut [f32]) {
        let (k, cout) = (self.k, self.cout);
        let n = out.len() / cout;
        for block in 0..cout.div_ceil(LANES) {
            let panel = &self.panels[block * k * LANES..(block + 1) * k * LANES];
            let bias: &[f32; LANES] = self.bias[block * LANES..(block + 1) * LANES]
                .try_into()
                .unwrap();
            let mut acc = [[0f32; LANES]; ROWS];
            match n {
                ROWS => acc = tile::<ROWS>(patches, k, p0, panel, bias),
                3 => acc[..3].copy_from_slice(&tile::<3>(patches, k, p0, panel, bias)),
                2 => acc[..2].copy_from_slice(&tile::<2>(patches, k, p0, panel, bias)),
                _ => acc[..1].copy_from_slice(&tile::<1>(patches, k, p0, panel, bias)),
            }
            let width = LANES.min(cout - block * LANES);
            for (r, a) in acc.iter().enumerate().take(n) {
                let dst = &mut out[r * cout + block * LANES..r * cout + block * LANES + width];
                for (o, &v) in dst.iter_mut().zip(a) {
                    *o = if self.relu { v.max(0.0) } else { v };
                }
            }
        }
    }
}

/// `R` output pixels by LANES channels, the sums kept in registers: per
/// step, R patch values and one row of the panel make R x LANES fused
/// multiply-adds.
#[inline(always)]
fn tile<const R: usize>(
    patches: &[f32],
    k: usize,
    p0: usize,
    panel: &[f32],
    bias: &[f32; LANES],
) -> [[f32; LANES]; R] {
    let mut acc = [*bias; R];
    let rows: [&[f32]; R] = std::array::from_fn(|r| &patches[(p0 + r) * k..(p0 + r + 1) * k]);
    for (kk, w) in panel.as_chunks::<LANES>().0.iter().enumerate() {
        for r in 0..R {
            let a = rows[r][kk];
            for j in 0..LANES {
                acc[r][j] = fma(a, w[j], acc[r][j]);
            }
        }
    }
    acc
}

/// `a * b + c`, fused where that is one instruction (it is on every phone).
#[inline(always)]
fn fma(a: f32, b: f32, c: f32) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        a.mul_add(b, c)
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        a * b + c
    }
}

/// The file's `[k][cout]` weights as panels of LANES channels, `[block][k]
/// [LANES]`, and the bias padded to whole blocks, zeros beyond `cout`.
fn repack(w: &[f32], b: &[f32], k: usize, cout: usize) -> (Vec<f32>, Vec<f32>) {
    let blocks = cout.div_ceil(LANES);
    let mut panels = vec![0f32; blocks * k * LANES];
    let mut bias = vec![0f32; blocks * LANES];
    for block in 0..blocks {
        for kk in 0..k {
            for j in 0..LANES.min(cout - block * LANES) {
                panels[(block * k + kk) * LANES + j] = w[kk * cout + block * LANES + j];
            }
        }
    }
    bias[..cout].copy_from_slice(b);
    (panels, bias)
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
