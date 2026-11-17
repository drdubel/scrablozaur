//! A camera frame to a decoded board: the live scanner's whole pipeline,
//! in the engine so the app and the web's tools run the same one.
//!
//! 1. The grid finder finds the grid's corners.
//! 2. The tiles' tops are straightened: the camera's pose, from the corners
//!    and the lens's focal length, lifts the grid a tile's height.
//! 3. The cell reader reads every square as the board lies, and with it
//!    which way up the board is: as the last confirmed board's tiles say,
//!    or, on a first scan, as the letters read best.
//! 4. The decoder finds the most probable legal board.

use super::decode::{decode, CellReading, DecodeOptions, Decoded, ScanBoard};
use super::geometry;
use super::grid::{GridFinder, Quad};
use super::reader::CellReader;
use super::Grid;
use crate::{EngineResult, Lexicon, BOARD_SIZE};

pub struct FrameScan {
    /// The grid's corners in the frame, starting from the board's own top
    /// left, clockwise.
    pub corners: Quad,
    /// The board straightened and turned upright, as it was read: RGB,
    /// `side` pixels square, one square of margin around the grid.
    pub image: Vec<u8>,
    pub side: usize,
    pub readings: Grid<CellReading>,
    pub decoded: Decoded,
}

/// Scan one frame: `rgb` is `width * height` pixels, three bytes each,
/// taken with a focal length of `focal_px` pixels. `prior` is the last
/// confirmed board, whose tiles stay as they were. None if no grid shows.
#[allow(clippy::too_many_arguments)]
pub fn scan_frame(
    finder: &GridFinder,
    reader: &CellReader,
    lexicon: &Lexicon,
    rgb: &[u8],
    width: usize,
    height: usize,
    focal_px: f64,
    prior: Option<&ScanBoard>,
    options: &DecodeOptions,
) -> EngineResult<Option<FrameScan>> {
    let Some(corners) = finder.find(rgb, width, height)? else {
        return Ok(None);
    };
    read_frame(
        reader, lexicon, rgb, width, height, corners, focal_px, prior, options,
    )
    .map(Some)
}

/// Steps 2-4 for a frame whose grid was already found at `corners`, as the
/// grid finder gives them: the app tracks the grid frame by frame to see the
/// board held still, then reads the frame it last found it in.
#[allow(clippy::too_many_arguments)]
pub fn read_frame(
    reader: &CellReader,
    lexicon: &Lexicon,
    rgb: &[u8],
    width: usize,
    height: usize,
    corners: Quad,
    focal_px: f64,
    prior: Option<&ScanBoard>,
    options: &DecodeOptions,
) -> EngineResult<FrameScan> {
    // A pose that cannot be solved for leaves the tiles' tops on the grid.
    let tops = geometry::tile_tops(&corners, width, height, focal_px).unwrap_or(corners);
    let cell = reader.cell_size();
    let side = (BOARD_SIZE + 2) * cell;
    let lying = geometry::straighten(rgb, width, height, &tops, cell)?;
    // One read of the board as it lies says both what each tile is and
    // which way up the board is.
    let reading = reader.read(&lying, side)?;
    let turns = prior
        .and_then(|p| reading.agreeing_turn(p))
        .or_else(|| reading.upright())
        .unwrap_or(0);
    let readings = reading.cells(turns);
    let image = geometry::rotate(&lying, side, turns);
    // Turning the image clockwise k times puts the frame's corner k back
    // from its top left at the board's top left.
    let mut corners = corners;
    corners.rotate_right(turns);
    let decoded = decode(lexicon, &readings, prior, options)?;
    Ok(FrameScan {
        corners,
        image,
        side,
        readings,
        decoded,
    })
}
