//! From a camera frame and the grid's four corners in it to the image the
//! cell reader reads: the tiles' top faces straightened, one square of
//! margin around the grid.
//!
//! Tiles stand about a quarter of a square proud of the board, so in a frame
//! taken at an angle their tops -- where the letters are -- are displaced
//! from the printed grid the corners outline. The camera's pose follows from
//! the grid's homography and the lens's focal length; with it, the plane of
//! the tiles' tops is one more homography (fitted to the test photos, the
//! model predicts the displacement to 0.05 squares).

use crate::{value_error, EngineResult, BOARD_SIZE};

/// A tile's top face above the board, in squares (fitted: 0.253).
pub const TILE_HEIGHT: f64 = 0.25;

type Mat3 = [[f64; 3]; 3];

/// The homography taking four points `src` to four points `dst`.
pub fn homography(src: &[[f64; 2]; 4], dst: &[[f64; 2]; 4]) -> Option<Mat3> {
    let mut a = [[0f64; 9]; 8];
    for i in 0..4 {
        let ([x, y], [u, v]) = (src[i], dst[i]);
        a[2 * i] = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u];
        a[2 * i + 1] = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v];
    }
    // Gaussian elimination with partial pivoting on the 8x9 augmented system.
    for col in 0..8 {
        let pivot = (col..8).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[pivot][col].abs() < 1e-12 {
            return None;
        }
        a.swap(col, pivot);
        let pivot_row = a[col];
        for (row, r) in a.iter_mut().enumerate() {
            if row != col {
                let f = r[col] / pivot_row[col];
                for (x, p) in r.iter_mut().zip(pivot_row).skip(col) {
                    *x -= f * p;
                }
            }
        }
    }
    let h: Vec<f64> = (0..8).map(|i| a[i][8] / a[i][i]).collect();
    Some([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])
}

fn apply(m: &Mat3, x: f64, y: f64) -> [f64; 2] {
    let w = m[2][0] * x + m[2][1] * y + m[2][2];
    [
        (m[0][0] * x + m[0][1] * y + m[0][2]) / w,
        (m[1][0] * x + m[1][1] * y + m[1][2]) / w,
    ]
}

/// The grid's outer corners in grid units: TL, TR, BR, BL.
fn grid_corners() -> [[f64; 2]; 4] {
    let n = BOARD_SIZE as f64;
    [[0.0, 0.0], [n, 0.0], [n, n], [0.0, n]]
}

/// Where the grid's corners would be at the height of the tiles' tops,
/// given its corners on the board (`corners`, TL TR BR BL, frame pixels),
/// the frame's size and the lens's focal length in pixels. The camera's
/// pose comes from the grid's homography: `H ~ K [r1 r2 t]`, and the plane
/// `TILE_HEIGHT` above it is `K [r1 r2 (t + h r3)]`. None for a degenerate
/// quadrilateral.
pub fn tile_tops(
    corners: &[[f64; 2]; 4],
    width: usize,
    height: usize,
    focal_px: f64,
) -> Option<[[f64; 2]; 4]> {
    let h = homography(&grid_corners(), corners)?;
    let (cx, cy) = (width as f64 / 2.0, height as f64 / 2.0);
    // K^-1 H, column by column.
    let col = |j: usize| {
        [
            (h[0][j] - cx * h[2][j]) / focal_px,
            (h[1][j] - cy * h[2][j]) / focal_px,
            h[2][j],
        ]
    };
    let (b1, b2, b3) = (col(0), col(1), col(2));
    let norm = |v: [f64; 3]| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    let mut s = 2.0 / (norm(b1) + norm(b2));
    if s * b3[2] < 0.0 {
        s = -s; // the board in front of the camera
    }
    let (r1, r2, t) = (b1.map(|v| v * s), b2.map(|v| v * s), b3.map(|v| v * s));
    let mut r3 = [
        r1[1] * r2[2] - r1[2] * r2[1],
        r1[2] * r2[0] - r1[0] * r2[2],
        r1[0] * r2[1] - r1[1] * r2[0],
    ];
    // Tiles rise toward the camera, whose centre is -R^T t.
    let camera_z = -(r3[0] * t[0] + r3[1] * t[1] + r3[2] * t[2]);
    if camera_z < 0.0 {
        r3 = r3.map(|v| -v);
    }
    let lift = |p: [f64; 2]| {
        let v: [f64; 3] =
            std::array::from_fn(|i| r1[i] * p[0] + r2[i] * p[1] + t[i] + TILE_HEIGHT * r3[i]);
        [focal_px * v[0] / v[2] + cx, focal_px * v[1] / v[2] + cy]
    };
    let tops = grid_corners().map(lift);
    tops.iter().flatten().all(|v| v.is_finite()).then_some(tops)
}

/// The reader's image of a frame: `(15 + 2) * cell` pixels square, RGB, the
/// grid whose corners are `corners` (TL TR BR BL, frame pixels) filling all
/// but a square's margin. `rgb` is the frame, `width * height` pixels, three
/// bytes each.
///
/// The frame's squares are usually several times the reader's, so it is
/// first shrunk by a whole factor, averaging blocks -- or bilinear sampling
/// would alias -- then sampled bilinearly through the homography.
pub fn straighten(
    rgb: &[u8],
    width: usize,
    height: usize,
    corners: &[[f64; 2]; 4],
    cell: usize,
) -> EngineResult<Vec<u8>> {
    if rgb.len() != width * height * 3 || width == 0 || height == 0 || cell == 0 {
        return Err(value_error(format!(
            "expected a {width}x{height} RGB frame, got {} bytes",
            rgb.len()
        )));
    }
    let side_px = (0..4)
        .map(|i| {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
        })
        .fold(0.0, f64::max);
    // Keep about two frame pixels per output pixel after shrinking.
    let factor = ((side_px / BOARD_SIZE as f64) / (2.0 * cell as f64))
        .floor()
        .max(1.0) as usize;
    let (fw, fh) = (width / factor, height / factor);
    let small: Vec<f32> = if factor == 1 {
        rgb.iter().map(|&v| v as f32).collect()
    } else {
        let mut out = vec![0f32; fw * fh * 3];
        let norm = 1.0 / (factor * factor) as f32;
        for y in 0..fh {
            for x in 0..fw {
                let mut acc = [0f32; 3];
                for dy in 0..factor {
                    let row = ((y * factor + dy) * width + x * factor) * 3;
                    for px in rgb[row..row + factor * 3].as_chunks::<3>().0 {
                        for (a, &v) in acc.iter_mut().zip(px) {
                            *a += v as f32;
                        }
                    }
                }
                out[(y * fw + x) * 3..(y * fw + x) * 3 + 3].copy_from_slice(&acc.map(|a| a * norm));
            }
        }
        out
    };
    let scaled = corners.map(|[x, y]| [x / factor as f64, y / factor as f64]);
    // Output pixel -> grid units -> frame pixel.
    let to_frame = homography(&grid_corners(), &scaled)
        .ok_or_else(|| value_error("the grid's corners make no quadrilateral"))?;
    let side = (BOARD_SIZE + 2) * cell;
    let mut out = vec![0u8; side * side * 3];
    for oy in 0..side {
        for ox in 0..side {
            let gx = (ox as f64 + 0.5) / cell as f64 - 1.0;
            let gy = (oy as f64 + 0.5) / cell as f64 - 1.0;
            let [fx, fy] = apply(&to_frame, gx, gy);
            // Pixel centres sit at +0.5; clamp to the frame's edge.
            let (sx, sy) = (
                (fx - 0.5).clamp(0.0, (fw - 1) as f64),
                (fy - 0.5).clamp(0.0, (fh - 1) as f64),
            );
            let (x0, y0) = (sx.floor() as usize, sy.floor() as usize);
            let (x1, y1) = ((x0 + 1).min(fw - 1), (y0 + 1).min(fh - 1));
            let (u, v) = ((sx - x0 as f64) as f32, (sy - y0 as f64) as f32);
            for ch in 0..3 {
                let at = |x: usize, y: usize| small[(y * fw + x) * 3 + ch];
                let value = (1.0 - u) * (1.0 - v) * at(x0, y0)
                    + u * (1.0 - v) * at(x1, y0)
                    + (1.0 - u) * v * at(x0, y1)
                    + u * v * at(x1, y1);
                out[(oy * side + ox) * 3 + ch] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    Ok(out)
}

/// A square RGB image turned `k` quarter turns clockwise.
pub fn rotate(rgb: &[u8], side: usize, k: usize) -> Vec<u8> {
    let mut out = vec![0u8; rgb.len()];
    for y in 0..side {
        for x in 0..side {
            let (ty, tx) = match k % 4 {
                0 => (y, x),
                1 => (x, side - 1 - y),
                2 => (side - 1 - y, side - 1 - x),
                _ => (side - 1 - x, y),
            };
            out[(ty * side + tx) * 3..(ty * side + tx) * 3 + 3]
                .copy_from_slice(&rgb[(y * side + x) * 3..(y * side + x) * 3 + 3]);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_homography_maps_its_four_points() {
        let src = [[0.0, 0.0], [15.0, 0.0], [15.0, 15.0], [0.0, 15.0]];
        let dst = [[120.0, 80.0], [910.0, 140.0], [870.0, 900.0], [60.0, 860.0]];
        let h = homography(&src, &dst).unwrap();
        for (s, d) in src.iter().zip(&dst) {
            let p = apply(&h, s[0], s[1]);
            assert!((p[0] - d[0]).abs() < 1e-6 && (p[1] - d[1]).abs() < 1e-6);
        }
        assert!(homography(&src, &[[0.0, 0.0]; 4]).is_none());
    }

    #[test]
    fn looking_straight_down_the_tops_stay_put_at_the_centre() {
        // A camera squarely above the board's centre sees no parallax there,
        // and the tops of tiles near the edges lean outward.
        let f = 1000.0;
        let corners = [
            [200.0, 200.0],
            [800.0, 200.0],
            [800.0, 800.0],
            [200.0, 800.0],
        ];
        let tops = tile_tops(&corners, 1000, 1000, f).unwrap();
        for (c, t) in corners.iter().zip(&tops) {
            let (dx, dy) = (t[0] - c[0], t[1] - c[1]);
            assert!(
                dx * (c[0] - 500.0) > 0.0 && dy * (c[1] - 500.0) > 0.0,
                "{c:?} -> {t:?}"
            );
        }
        let centre = |q: &[[f64; 2]; 4]| {
            q.iter()
                .fold([0.0, 0.0], |a, p| [a[0] + p[0] / 4.0, a[1] + p[1] / 4.0])
        };
        let (a, b) = (centre(&corners), centre(&tops));
        assert!((a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6);
    }

    #[test]
    fn straightening_a_frame_of_its_own_board_gives_it_back() {
        // A frame that is already the reader's image, corners where its grid is.
        let cell = 8;
        let side = (BOARD_SIZE + 2) * cell;
        let rgb: Vec<u8> = (0..side * side * 3)
            .map(|i| ((i / 3) % 251) as u8)
            .collect();
        let (lo, hi) = (cell as f64, (side - cell) as f64);
        let out = straighten(
            &rgb,
            side,
            side,
            &[[lo, lo], [hi, lo], [hi, hi], [lo, hi]],
            cell,
        )
        .unwrap();
        assert_eq!(out, rgb);
        assert_eq!(rotate(&rotate(&rgb, side, 1), side, 3), rgb);
    }
}
