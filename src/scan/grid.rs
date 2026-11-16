//! The grid finder: where the playing grid's four outer corners are in a
//! camera frame. A small fully-convolutional net draws a heatmap peaking at
//! each corner; the four peaks that make the likeliest grid are kept, then
//! found again on the region they outline, scaled up, for precision.
//! `board_reader/scripts/train_grid_finder.py` trains it and writes the file
//! loaded here, and its `find` is the reference this mirrors step by step.
//!
//! The corners come clockwise from the one nearest the frame's top left.
//! Which corner is the board's own top left is the cell reader's to say
//! (`CellReader::upright`): the grid looks the same from all four sides.

use super::net::{Bytes, Map, Net};
use crate::{value_error, EngineError, EngineResult};

const MAGIC: &[u8; 8] = b"SCRBGRD1";
/// The strongest peaks considered as the grid's corners.
const PEAKS: usize = 8;
/// A peak must reach this, as a probability, to be considered.
const MIN_PEAK: f32 = 0.01;
/// Margin around the first pass's grid, as a share of its size, cropped
/// for the second.
const ZOOM_MARGIN: f64 = 0.12;

pub type Quad = [[f64; 2]; 4];

pub struct GridFinder {
    /// The net's input is `side` pixels square; its heatmap is `stride`
    /// times smaller.
    side: usize,
    stride: usize,
    scale: f32,
    offset: f32,
    net: Net,
}

impl GridFinder {
    pub fn open(path: &str) -> EngineResult<Self> {
        let data = std::fs::read(path).map_err(|e| EngineError::Io(format!("{path}: {e}")))?;
        GridFinder::from_bytes(&data).map_err(|e| match e {
            EngineError::Io(m) => EngineError::Io(format!("{path}: {m}")),
            other => other,
        })
    }

    pub fn from_bytes(data: &[u8]) -> EngineResult<Self> {
        let bad = |m: &str| EngineError::Io(m.to_string());
        if data.len() < 8 || &data[..8] != MAGIC {
            return Err(bad(
                "not a SCRBGRD1 file (re-run train_grid_finder.py --final)",
            ));
        }
        let mut r = Bytes { data, at: 8 };
        let (side, stride) = (r.u32()?, r.u32()?);
        let (scale, offset) = (r.f32()?, r.f32()?);
        if side == 0 || stride == 0 || side % stride != 0 {
            return Err(bad("bad grid finder geometry"));
        }
        let net = Net::parse(&mut r, 3)?;
        if net.flat || net.out_channels != 1 {
            return Err(bad("the grid finder must end in one heatmap"));
        }
        Ok(GridFinder {
            side,
            stride,
            scale,
            offset,
            net,
        })
    }

    /// The heatmap's logits for a `side x side` RGB input of 0-255 values,
    /// row-major: for checking against PyTorch.
    pub fn raw(&self, input: &[f32]) -> EngineResult<Vec<f32>> {
        if input.len() != self.side * self.side * 3 {
            return Err(value_error(format!(
                "expected {} values, got {}",
                self.side * self.side * 3,
                input.len()
            )));
        }
        Ok(self.heatmap(input.to_vec()).data)
    }

    /// The grid's corners in an RGB frame (`width * height` pixels, three
    /// bytes each), clockwise from the one nearest its top left; None if no
    /// grid-like quadrilateral shows.
    pub fn find(&self, rgb: &[u8], width: usize, height: usize) -> EngineResult<Option<Quad>> {
        if rgb.len() != width * height * 3 || width == 0 || height == 0 {
            return Err(value_error(format!(
                "expected a {width}x{height} RGB frame, got {} bytes",
                rgb.len()
            )));
        }
        let frame = Frame { rgb, width };
        let Some(first) = self.once(&frame, 0, 0, width, height) else {
            return Ok(None);
        };
        let (lo, hi) = bounds(&first);
        let pad = [(hi[0] - lo[0]) * ZOOM_MARGIN, (hi[1] - lo[1]) * ZOOM_MARGIN];
        let x0 = (lo[0] - pad[0]).max(0.0) as usize;
        let y0 = (lo[1] - pad[1]).max(0.0) as usize;
        let x1 = (hi[0] + pad[0]).min(width as f64) as usize;
        let y1 = (hi[1] + pad[1]).min(height as f64) as usize;
        if x1 <= x0 || y1 <= y0 {
            return Ok(Some(first));
        }
        Ok(Some(self.once(&frame, x0, y0, x1, y1).unwrap_or(first)))
    }

    /// One pass over the region `[x0, x1) x [y0, y1)`, in frame pixels.
    fn once(&self, frame: &Frame, x0: usize, y0: usize, x1: usize, y1: usize) -> Option<Quad> {
        let (rw, rh) = (x1 - x0, y1 - y0);
        let scale = self.side as f64 / rw.max(rh) as f64;
        let (ow, oh) = (
            (rw as f64 * scale).round() as usize,
            (rh as f64 * scale).round() as usize,
        );
        let (ow, oh) = (ow.clamp(1, self.side), oh.clamp(1, self.side));
        // The region area-averaged to oh x ow, PyTorch's adaptive average
        // pooling: output row i covers input rows [floor(i h / oh),
        // ceil((i + 1) h / oh)). Then placed top left on a black canvas.
        let mut canvas = vec![0f32; self.side * self.side * 3];
        let span = |i: usize, n: usize, out: usize| (i * n / out, ((i + 1) * n).div_ceil(out));
        for oy in 0..oh {
            let (ya, yb) = span(oy, rh, oh);
            for ox in 0..ow {
                let (xa, xb) = span(ox, rw, ow);
                let mut acc = [0f32; 3];
                for y in ya..yb {
                    let row = ((y0 + y) * frame.width + x0 + xa) * 3;
                    for px in frame.rgb[row..row + (xb - xa) * 3].as_chunks::<3>().0 {
                        for (a, &v) in acc.iter_mut().zip(px) {
                            *a += v as f32;
                        }
                    }
                }
                let n = ((yb - ya) * (xb - xa)) as f32;
                canvas[(oy * self.side + ox) * 3..(oy * self.side + ox) * 3 + 3]
                    .copy_from_slice(&acc.map(|a| a / n));
            }
        }
        let heat = self.heatmap(canvas);
        let peaks = peaks(&heat, self.stride);
        let quad = best_quad(&peaks)?;
        Some(quad.map(|[x, y]| [x / scale + x0 as f64, y / scale + y0 as f64]))
    }

    fn heatmap(&self, mut input: Vec<f32>) -> Map {
        input
            .iter_mut()
            .for_each(|v| *v = *v * self.scale + self.offset);
        self.net.forward(Map {
            h: self.side,
            w: self.side,
            c: 3,
            data: input,
        })
    }
}

struct Frame<'a> {
    rgb: &'a [u8],
    width: usize,
}

fn bounds(q: &Quad) -> ([f64; 2], [f64; 2]) {
    let lo = [0, 1].map(|i| q.iter().map(|p| p[i]).fold(f64::INFINITY, f64::min));
    let hi = [0, 1].map(|i| q.iter().map(|p| p[i]).fold(f64::NEG_INFINITY, f64::max));
    (lo, hi)
}

/// The strongest local maxima of a heatmap's logits, as probabilities, each
/// refined to the centroid of its 3x3 neighbourhood: (x, y in input pixels,
/// probability), strongest first.
fn peaks(heat: &Map, stride: usize) -> Vec<([f64; 2], f32)> {
    let (h, w) = (heat.h, heat.w);
    let p: Vec<f32> = heat
        .data
        .iter()
        .map(|&z| 1.0 / (1.0 + (-z).exp()))
        .collect();
    let at = |y: usize, x: usize| p[y * w + x];
    let mut found: Vec<(f32, usize, usize)> = Vec::new();
    for y in 0..h {
        for x in 0..w {
            let v = at(y, x);
            if v <= MIN_PEAK {
                continue;
            }
            let local = (y.saturating_sub(1)..(y + 2).min(h))
                .flat_map(|yy| (x.saturating_sub(1)..(x + 2).min(w)).map(move |xx| (yy, xx)))
                .map(|(yy, xx)| at(yy, xx))
                .fold(f32::NEG_INFINITY, f32::max);
            if v == local {
                found.push((v, y, x));
            }
        }
    }
    found.sort_by(|a, b| b.0.total_cmp(&a.0));
    found.truncate(PEAKS);
    found
        .into_iter()
        .map(|(v, y, x)| {
            let (mut sum, mut cy, mut cx) = (0f64, 0f64, 0f64);
            for yy in y.saturating_sub(1)..(y + 2).min(h) {
                for xx in x.saturating_sub(1)..(x + 2).min(w) {
                    let wgt = at(yy, xx) as f64;
                    sum += wgt;
                    cy += wgt * yy as f64;
                    cx += wgt * xx as f64;
                }
            }
            let s = stride as f64;
            ([(cx / sum + 0.5) * s, (cy / sum + 0.5) * s], v)
        })
        .collect()
}

/// Clockwise, starting from the corner nearest the top left.
fn order(mut q: Quad) -> Quad {
    let c = [
        q.iter().map(|p| p[0]).sum::<f64>() / 4.0,
        q.iter().map(|p| p[1]).sum::<f64>() / 4.0,
    ];
    q.sort_by(|a, b| {
        (a[1] - c[1])
            .atan2(a[0] - c[0])
            .total_cmp(&(b[1] - c[1]).atan2(b[0] - c[0]))
    });
    let start = (0..4)
        .min_by(|&i, &j| (q[i][0] + q[i][1]).total_cmp(&(q[j][0] + q[j][1])))
        .unwrap();
    q.rotate_left(start);
    q
}

fn convex(q: &Quad) -> bool {
    let turn = |i: usize| {
        let (a, b, c) = (q[i], q[(i + 1) % 4], q[(i + 2) % 4]);
        (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
    };
    let turns = [turn(0), turn(1), turn(2), turn(3)];
    turns.iter().all(|&t| t > 0.0) || turns.iter().all(|&t| t < 0.0)
}

/// Of the peaks, the four that make the likeliest grid: convex, no side
/// under a third of the longest, the highest joint probability. The four
/// strongest alone sometimes include a table edge's corner.
fn best_quad(peaks: &[([f64; 2], f32)]) -> Option<Quad> {
    let n = peaks.len();
    let mut best: Option<(f64, Quad)> = None;
    for a in 0..n {
        for b in a + 1..n {
            for c in b + 1..n {
                for d in c + 1..n {
                    let q = order([peaks[a].0, peaks[b].0, peaks[c].0, peaks[d].0]);
                    let side = |i: usize| {
                        let (p, r) = (q[i], q[(i + 1) % 4]);
                        ((p[0] - r[0]).powi(2) + (p[1] - r[1]).powi(2)).sqrt()
                    };
                    let sides = [side(0), side(1), side(2), side(3)];
                    let longest = sides.iter().copied().fold(0.0, f64::max);
                    if !convex(&q) || sides.iter().any(|&s| s < longest / 3.0) {
                        continue;
                    }
                    let score: f64 = [a, b, c, d].iter().map(|&i| (peaks[i].1 as f64).ln()).sum();
                    if best.is_none_or(|(s, _)| score > s) {
                        best = Some((score, q));
                    }
                }
            }
        }
    }
    best.map(|(_, q)| q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corners_come_clockwise_from_the_top_left() {
        let q = order([[90.0, 10.0], [10.0, 90.0], [12.0, 8.0], [95.0, 92.0]]);
        assert_eq!(q, [[12.0, 8.0], [90.0, 10.0], [95.0, 92.0], [10.0, 90.0]]);
        assert!(convex(&q));
        assert!(!convex(&[
            [0.0, 0.0],
            [10.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0]
        ]));
    }

    #[test]
    fn the_likeliest_grid_beats_the_strongest_four() {
        // A strong peak far off the grid's line, and the grid's fourth corner
        // a little weaker: three plus the stray make no grid-like shape.
        let peaks = vec![
            ([10.0, 10.0], 0.9),
            ([100.0, 10.0], 0.9),
            ([100.0, 100.0], 0.9),
            ([200.0, 200.0], 0.8),
            ([10.0, 100.0], 0.7),
        ];
        let q = best_quad(&peaks).unwrap();
        assert_eq!(
            q,
            [[10.0, 10.0], [100.0, 10.0], [100.0, 100.0], [10.0, 100.0]]
        );
    }
}
