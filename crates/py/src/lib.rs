//! The `scrablozaur` Python module: a thin PyO3 layer over `scrablozaur_core`.
//!
//! Every method here converts arguments, calls the engine, and maps
//! `EngineError` onto the exception Python callers have always caught --
//! `ValueError`, `OSError` or `RuntimeError`. Nothing in this file decides
//! anything; behaviour belongs in the engine, where the Android bindings share
//! it. `scrablozaur.pyi` documents this surface and must change with it.

use std::collections::HashMap;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use scrablozaur_core::scan::decode::{self as scan_decode, CellReading, DecodeOptions, ScanBoard};
use scrablozaur_core::scan::frame;
use scrablozaur_core::scan::geometry;
use scrablozaur_core::scan::grid::GridFinder;
use scrablozaur_core::scan::reader::CellReader;
use scrablozaur_core::{
    self as engine, BestWord, Board, EngineError, Language, LeaveNet, Lexicon, SimParams,
};

/// Board side; grids cross into Python as nested lists of this size.
const SIZE: usize = 15;

fn to_py(err: EngineError) -> PyErr {
    match err {
        EngineError::Value(m) => PyValueError::new_err(m),
        EngineError::Io(m) => PyIOError::new_err(m),
        EngineError::State(m) => PyRuntimeError::new_err(m),
    }
}

/// A language definition, built from the values in `languages/<code>.json`.
/// Every `Board` and `Dawg` is bound to one, and pairing a board with a
/// dictionary in another language is rejected rather than silently mis-scored.
// `skip_from_py_object` for the same reason `Board` uses it: `Clone` is here so
// getters can hand one back by value, not so pyo3 extracts one by value at the
// FFI boundary. Every parameter takes `&LanguagePy`.
#[pyclass(name = "Language", frozen, skip_from_py_object)]
#[derive(Clone)]
struct LanguagePy {
    inner: &'static Language,
}

#[pymethods]
impl LanguagePy {
    /// `counts` and `points` must each have an entry for every letter of
    /// `alphabet` plus the blank. The bag is built in `alphabet` order with the
    /// blanks last, so `alphabet` fixes both the cross-check bit indexes and
    /// the shuffle order a seeded board replays.
    #[new]
    #[pyo3(signature = (code, alphabet, counts, points, blank='?'))]
    fn new(
        code: &str,
        alphabet: &str,
        counts: HashMap<char, u32>,
        points: HashMap<char, u32>,
        blank: char,
    ) -> PyResult<Self> {
        let inner = Language::define(code, alphabet, blank, &counts, &points).map_err(to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn code(&self) -> &str {
        self.inner.code()
    }

    #[getter]
    fn alphabet(&self) -> String {
        self.inner.alphabet()
    }

    #[getter]
    fn blank(&self) -> char {
        self.inner.blank()
    }

    /// The full tile distribution, in the order a fresh bag holds it.
    fn fresh_tile_bag(&self) -> Vec<char> {
        self.inner.bag().to_vec()
    }

    /// Tile symbols in the leave-value net's feature order.
    fn eval_alphabet(&self) -> Vec<char> {
        self.inner.eval_letters().to_vec()
    }

    fn letter_points(&self, letter: char) -> u32 {
        self.inner.letter_points(letter)
    }

    fn __repr__(&self) -> String {
        format!(
            "Language('{}', {} letters, {} tiles)",
            self.inner.code(),
            self.inner.alphabet().chars().count(),
            self.inner.bag().len()
        )
    }
}

/// The DAWG for word lookups plus, when present, the GADDAG that drives move
/// generation (`Lexicon` in the engine).
#[pyclass(name = "Dawg")]
struct DawgPy {
    inner: Lexicon,
}

#[pymethods]
impl DawgPy {
    /// Load a `lang` DAWG from `path`. If `gaddag_path` is omitted, a sibling
    /// `gaddag.bin` (same directory, `dawg`→`gaddag` in the filename) is loaded
    /// when present; move generation uses it automatically. Passing a path that
    /// does not exist is an error; omitting it and finding no sibling simply
    /// leaves the GADDAG absent (legacy generation).
    #[new]
    #[pyo3(signature = (lang, path, gaddag_path=None))]
    fn new(lang: &LanguagePy, path: &str, gaddag_path: Option<&str>) -> PyResult<Self> {
        let inner = Lexicon::open(lang.inner, path, gaddag_path).map_err(to_py)?;
        Ok(DawgPy { inner })
    }

    /// The language this lexicon was loaded under.
    #[getter]
    fn lang(&self) -> LanguagePy {
        LanguagePy {
            inner: self.inner.lang(),
        }
    }

    fn contains(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    fn __contains__(&self, word: &str) -> bool {
        self.inner.contains(word)
    }

    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Whether a GADDAG is loaded (so `get_best_words` uses the fast path).
    fn has_gaddag(&self) -> bool {
        self.inner.has_gaddag()
    }

    fn search(&self, pattern: &str, letters: &str) -> Vec<String> {
        self.inner.search(pattern, letters)
    }
}

// `skip_from_py_object`: deriving Clone would otherwise opt `Board` into
// by-value extraction from Python. Nothing here wants that -- every Rust-side
// use borrows -- and a silent clone at an FFI boundary is exactly the kind of
// thing that makes a search layer mysteriously slow.
#[pyclass(name = "Board", skip_from_py_object)]
#[derive(Clone)]
struct BoardPy {
    inner: Board,
}

type EndgameTuple = (
    String,
    u32,
    (usize, usize, bool),
    Vec<char>,
    i32,
    usize,
    bool,
);
type SimTuple = (
    String,
    u32,
    (usize, usize, bool),
    Vec<char>,
    f64,
    f64,
    usize,
);

#[pymethods]
impl BoardPy {
    #[new]
    fn new(lang: &LanguagePy) -> Self {
        BoardPy {
            inner: Board::new(lang.inner),
        }
    }

    /// A fresh board whose bag is shuffled deterministically from `seed`. Two
    /// boards built with the same seed deal the same tiles in the same order,
    /// which is what lets the arena play one bag twice with the seats swapped.
    #[staticmethod]
    fn seeded(lang: &LanguagePy, seed: u64) -> Self {
        BoardPy {
            inner: Board::seeded(lang.inner, seed),
        }
    }

    /// The language this position is played under.
    #[getter]
    fn lang(&self) -> LanguagePy {
        LanguagePy {
            inner: self.inner.lang(),
        }
    }

    /// Deep copy, including the bag and the RNG stream position. The only way
    /// to branch a position without the lossy `str()` -> `from_grid()` round
    /// trip, and the basis of make/unmake search.
    fn copy(&self) -> BoardPy {
        self.clone()
    }

    /// Construct a board pre-filled from a 15x15 grid of single-character
    /// cells (e.g. loaded from a saved game or a scanned photo), each cell
    /// either a letter or `'-'` for empty.
    ///
    /// Starts from `lang`'s full tile bag and **removes every tile already on
    /// the grid**, so the remaining bag is what could still be drawn. A grid
    /// holding more copies of a letter than the distribution allows is an
    /// error rather than a silently over-drawn bag -- which is why a played
    /// blank must be declared in `blanks`: it consumes a blank from the bag,
    /// not a copy of the letter it stands in for.
    #[staticmethod]
    #[pyo3(signature = (lang, board, blanks=None))]
    fn from_grid(
        lang: &LanguagePy,
        board: Vec<Vec<String>>,
        blanks: Option<Vec<Vec<bool>>>,
    ) -> PyResult<Self> {
        let inner = Board::from_grid(lang.inner, &board, blanks.as_deref()).map_err(to_py)?;
        Ok(BoardPy { inner })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn give_letters(&mut self, letters: &str) -> String {
        self.inner.give_letters(letters)
    }

    /// How many tiles are left in the bag. Public information in Scrabble --
    /// unlike `bag_tiles`, this is safe to expose to a player.
    fn bag_remaining(&self) -> usize {
        self.inner.bag_remaining()
    }

    /// The bag's actual contents, in draw order — the *last* element is the
    /// next tile out. Hidden information: for the arena, tests and endgame
    /// setup, never for a player's own decision-making.
    fn bag_tiles(&self) -> Vec<char> {
        self.inner.bag_tiles()
    }

    /// Replace the bag wholesale, in draw order (last element drawn first).
    fn set_bag(&mut self, tiles: Vec<char>) {
        self.inner.set_bag(tiles)
    }

    /// Which occupied squares hold a blank, as a 15x15 mask. Pairs with
    /// `from_grid(grid, blanks)` to round-trip a position losslessly.
    fn blank_mask(&self) -> Vec<Vec<bool>> {
        self.inner.blank_mask()
    }

    fn exchange_letters(&mut self, letters: &str, letters_to_exchange: &str) -> String {
        self.inner.exchange_letters(letters, letters_to_exchange)
    }

    /// Standard Scrabble rule: exchanging is only allowed while at least a
    /// full rack's worth of tiles remain in the bag.
    fn can_exchange(&self) -> bool {
        self.inner.can_exchange()
    }

    /// This language's full tile distribution, as `Board()` and
    /// `Board.from_grid()` each start with.
    fn fresh_tile_bag(&self) -> Vec<char> {
        self.inner.fresh_tile_bag()
    }

    /// Sum of face point values of a rack (blank tiles score 0).
    fn rack_value(&self, letters: &str) -> u32 {
        self.inner.rack_value(letters)
    }

    /// Face point value of a single letter. Blanks score 0.
    fn letter_points(&self, letter: char) -> u32 {
        self.inner.letter_points(letter)
    }

    /// Index into `draws` of who goes first: closest to the start of the
    /// alphabet wins, a blank beats every letter, first index wins ties.
    fn first_draw_winner(&self, draws: Vec<char>) -> usize {
        self.inner.first_draw_winner(&draws)
    }

    fn calculate_word_points(
        &self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        letters: &str,
    ) -> PyResult<u32> {
        self.inner
            .calculate_word_points(word, row, col, horizontal, letters)
            .map_err(to_py)
    }

    fn check_word_placement(
        &self,
        dawg: &DawgPy,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> PyResult<()> {
        self.inner
            .check_word_placement(&dawg.inner, word, row, col, horizontal)
            .map_err(to_py)
    }

    /// Write `word` onto the board and return the cells it actually filled, in
    /// placement order — feed that straight back to `unplace_word` to undo.
    ///
    /// `used` is the move's tile list as returned by `get_best_words`: one
    /// entry per newly covered square, `'?'` where a blank stands in. Passing
    /// it is what lets the board remember that a square holds a blank, so the
    /// square keeps scoring 0 for every later cross-word. Omitting it treats
    /// every placed tile as a real one.
    #[pyo3(signature = (word, row, col, horizontal, used=None))]
    fn place_word(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        used: Option<Vec<char>>,
    ) -> PyResult<Vec<(usize, usize)>> {
        self.inner
            .place_word(word, row, col, horizontal, used.as_deref())
            .map_err(to_py)
    }

    /// Undo a `place_word`, given the cells it returned. `was_first` restores
    /// the opening-move flag (a search that unwinds the first play must put it
    /// back). Cheaper than cloning the board at every node of a search.
    #[pyo3(signature = (cells, was_first=false))]
    fn unplace_word(&mut self, cells: Vec<(usize, usize)>, was_first: bool) -> PyResult<()> {
        self.inner.unplace_word(&cells, was_first).map_err(to_py)
    }

    fn get_row_patterns(&self, row_idx: usize) -> Vec<(usize, usize)> {
        self.inner.get_row_patterns(row_idx)
    }

    fn get_col_patterns(&self, col_idx: usize) -> Vec<(usize, usize)> {
        self.inner.get_col_patterns(col_idx)
    }

    fn get_all_patterns(&self) -> Vec<(usize, usize, usize, bool)> {
        self.inner.get_all_patterns()
    }

    fn best_word_from_pattern(
        &self,
        dawg: &DawgPy,
        row: usize,
        start: usize,
        end: usize,
        horizontal: bool,
        letters: &str,
    ) -> String {
        self.inner
            .best_word_from_pattern(&dawg.inner, row, start, end, horizontal, letters)
    }

    #[pyo3(signature = (dawg, letters, n, parallel=true))]
    fn get_best_words(
        &self,
        dawg: &DawgPy,
        letters: &str,
        n: usize,
        parallel: bool,
    ) -> Vec<BestWord> {
        self.inner.get_best_words(&dawg.inner, letters, n, parallel)
    }

    /// Best-scoring first moves: the centre square must be covered, so
    /// every word/offset combination that does so is a candidate.
    fn best_opening_words(&self, dawg: &DawgPy, letters: &str, n: usize) -> Vec<BestWord> {
        self.inner.best_opening_words(&dawg.inner, letters, n)
    }

    /// Best-scoring moves across every valid placement pattern already on
    /// the board (i.e. every move but the opening one).
    fn best_words_from_patterns(
        &self,
        dawg: &DawgPy,
        letters: &str,
        n: usize,
        parallel: bool,
    ) -> Vec<BestWord> {
        self.inner
            .best_words_from_patterns(&dawg.inner, letters, n, parallel)
    }

    #[pyo3(signature = (dawg, letters, parallel=true))]
    fn get_best_word(&self, dawg: &DawgPy, letters: &str, parallel: bool) -> BestWord {
        self.inner.get_best_word(&dawg.inner, letters, parallel)
    }

    /// Every legal play, unsorted and untruncated. Requires a GADDAG, and
    /// raises rather than returning an empty list, which a caller would read
    /// as "no legal plays".
    #[pyo3(signature = (dawg, letters, parallel=true))]
    fn all_moves(&self, dawg: &DawgPy, letters: &str, parallel: bool) -> PyResult<Vec<BestWord>> {
        self.inner
            .all_moves(&dawg.inner, letters, parallel)
            .map_err(to_py)
    }

    /// `(tw_open, dw_open, tl_open, dl_open, board_fill)` for this board --
    /// the same five scalars `smart_player/board_features.encode_board`
    /// computes, straight from the engine's own bonus layout instead of a
    /// transcription of it.
    fn board_feature_scalars(&self) -> [f32; 5] {
        self.inner.board_feature_scalars()
    }

    /// Best play with the bag empty, searched rather than sampled.
    ///
    /// Returns `(word, score, (row, col, horizontal), used, differential,
    /// nodes, exact)`. An empty word means passing beats every play.
    #[pyo3(signature = (dawg, my_rack, opp_rack, max_nodes=300000))]
    fn solve_endgame(
        &self,
        py: Python<'_>,
        dawg: &DawgPy,
        my_rack: &str,
        opp_rack: &str,
        max_nodes: usize,
    ) -> PyResult<EndgameTuple> {
        // Nothing in a search touches Python, and it is long enough that
        // holding the GIL through it would stall every other thread.
        let out = py
            .detach(|| {
                self.inner
                    .solve_endgame(&dawg.inner, my_rack, opp_rack, max_nodes)
            })
            .map_err(to_py)?;
        let (word, score, pos, used) = out
            .best
            .unwrap_or_else(|| (String::new(), 0, (0, 0, true), Vec::new()));
        Ok((
            word,
            score,
            pos,
            used,
            out.differential,
            out.nodes,
            out.exact,
        ))
    }

    /// Counts of the tiles neither on the board nor in `rack` -- the bag plus
    /// every opponent's rack -- indexed by `LeaveNet.alphabet()`.
    fn unseen_tile_counts(&self, rack: &str) -> Vec<u8> {
        self.inner.unseen_tile_counts(rack)
    }

    /// Rank this rack's plays by Monte-Carlo simulation.
    ///
    /// Returns `(word, score, (row, col, horizontal), used, equity, stderr,
    /// iterations)` per surviving candidate, best equity first.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (dawg, net, letters, candidates=20, iterations=200, plies=1, batch=25, root_eval_cap=150, leave_weight=1.0, seed=0))]
    fn simulate(
        &self,
        py: Python<'_>,
        dawg: &DawgPy,
        net: &LeaveNetPy,
        letters: &str,
        candidates: usize,
        iterations: usize,
        plies: usize,
        batch: usize,
        root_eval_cap: usize,
        leave_weight: f64,
        seed: u64,
    ) -> PyResult<Vec<SimTuple>> {
        let params = SimParams {
            candidates,
            iterations,
            plies,
            batch,
            root_eval_cap,
            leave_weight,
            seed,
        };
        // See `solve_endgame`: a simulation must not hold the GIL.
        let out = py
            .detach(|| {
                self.inner
                    .simulate(&dawg.inner, &net.inner, letters, params)
            })
            .map_err(to_py)?;
        Ok(out
            .into_iter()
            .map(|c| {
                let (word, score, pos, used) = c.play;
                (word, score, pos, used, c.equity, c.stderr, c.iterations)
            })
            .collect())
    }
}

/// The leave-value net, loaded from `export_weights.py`'s output.
#[pyclass(name = "LeaveNet")]
struct LeaveNetPy {
    inner: LeaveNet,
}

#[pymethods]
impl LeaveNetPy {
    #[new]
    fn new(lang: &LanguagePy, path: &str) -> PyResult<Self> {
        let inner = LeaveNet::open(lang.inner, path).map_err(to_py)?;
        Ok(LeaveNetPy { inner })
    }

    /// Forward pass on an already-encoded feature vector. Exists so a test can
    /// check the engine's arithmetic against PyTorch's on identical input.
    fn raw(&self, x: Vec<f32>) -> PyResult<f32> {
        self.inner.raw(&x).map_err(to_py)
    }

    /// Predicted value of holding `leave`, in `board`'s context, for a player
    /// whose full rack is `rack` (which fixes the unseen-tile count).
    fn value(&self, board: &BoardPy, leave: &str, rack: &str) -> f32 {
        self.inner.value(&board.inner, leave, rack)
    }

    /// The tile-symbol order the feature vector's leading slots use, so a test
    /// can assert it matches `model.ALPHABET`.
    fn alphabet(&self) -> Vec<char> {
        self.inner.alphabet()
    }
}

/// The cell reader: a small CNN, run by the engine, that reads every square
/// of a straightened board.
#[pyclass(name = "CellReader")]
struct CellReaderPy {
    inner: CellReader,
}

#[pymethods]
impl CellReaderPy {
    /// Load `train_cell_reader.py --final`'s `.bin` for `lang`.
    #[new]
    fn new(lang: &LanguagePy, path: &str) -> PyResult<Self> {
        let inner = CellReader::open(lang.inner, path).map_err(to_py)?;
        Ok(CellReaderPy { inner })
    }

    #[getter]
    fn cell_size(&self) -> usize {
        self.inner.cell_size()
    }

    /// The logits for one crop, before temperature.
    fn raw(&self, crop: Vec<f32>) -> PyResult<Vec<f32>> {
        self.inner.raw(&crop).map_err(to_py)
    }

    /// Quarter turns clockwise that stand the board's letters upright, or
    /// None if it sees no tiles to tell by.
    fn upright(&self, py: Python<'_>, rgb: &[u8], side: usize) -> PyResult<Option<usize>> {
        py.detach(|| self.inner.read(rgb, side).map(|reading| reading.upright()))
            .map_err(to_py)
    }

    /// Every square's reading, as `decode_scan` takes them.
    fn read_board(
        &self,
        py: Python<'_>,
        rgb: &[u8],
        side: usize,
    ) -> PyResult<Vec<Vec<ReadingTuple>>> {
        let grid = py
            .detach(|| self.inner.read_board(rgb, side))
            .map_err(to_py)?;
        Ok(grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| (c.p_tile, c.letters.clone(), c.p_blank))
                    .collect()
            })
            .collect())
    }
}

type Corners = [(f64, f64); 4];

fn corner_array(corners: Corners) -> [[f64; 2]; 4] {
    corners.map(|(x, y)| [x, y])
}

/// Where the grid's corners (TL, TR, BR, BL, frame pixels) would be at the
/// height of the tiles' tops, for a `width` x `height` frame taken with a
/// focal length of `focal_px` pixels. None for a degenerate quadrilateral.
#[pyfunction]
fn scan_tile_tops(corners: Corners, width: usize, height: usize, focal_px: f64) -> Option<Corners> {
    geometry::tile_tops(&corner_array(corners), width, height, focal_px)
        .map(|t| t.map(|[x, y]| (x, y)))
}

/// The cell reader's image of an RGB frame: the grid with corners `corners`
/// straightened to `17 * cell` pixels square, one square of margin around it.
#[pyfunction]
fn scan_straighten(
    py: Python<'_>,
    rgb: &[u8],
    width: usize,
    height: usize,
    corners: Corners,
    cell: usize,
) -> PyResult<Vec<u8>> {
    py.detach(|| geometry::straighten(rgb, width, height, &corner_array(corners), cell))
        .map_err(to_py)
}

/// The grid finder: where the playing grid's four corners are in a frame.
#[pyclass(name = "GridFinder")]
struct GridFinderPy {
    inner: GridFinder,
}

#[pymethods]
impl GridFinderPy {
    /// Load `train_grid_finder.py --final`'s `.bin`.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = GridFinder::open(path).map_err(to_py)?;
        Ok(GridFinderPy { inner })
    }

    /// The heatmap's logits for a square input of 0-255 values.
    fn raw(&self, input: Vec<f32>) -> PyResult<Vec<f32>> {
        self.inner.raw(&input).map_err(to_py)
    }

    /// The grid's corners in an RGB frame, clockwise from the one nearest
    /// its top left, or None.
    fn find(
        &self,
        py: Python<'_>,
        rgb: &[u8],
        width: usize,
        height: usize,
    ) -> PyResult<Option<Corners>> {
        let quad = py
            .detach(|| self.inner.find(rgb, width, height))
            .map_err(to_py)?;
        Ok(quad.map(|q| q.map(|[x, y]| (x, y))))
    }
}

/// What `scan_frame` made of a camera frame.
#[pyclass(name = "FrameScan", frozen)]
struct FrameScanPy {
    /// The grid's corners in the frame, from the board's own top left, clockwise.
    #[pyo3(get)]
    corners: Corners,
    /// The board straightened and turned upright, as read: RGB, `side` square.
    #[pyo3(get)]
    image: Vec<u8>,
    #[pyo3(get)]
    side: usize,
    #[pyo3(get)]
    readings: Vec<Vec<ReadingTuple>>,
    #[pyo3(get)]
    decoded: Py<DecodedScanPy>,
}

/// A camera frame to a decoded board: find the grid, straighten the tiles'
/// tops, turn the board upright, read and decode. None if no grid shows.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (finder, reader, dawg, rgb, width, height, focal_px, prior=None, prior_blanks=None))]
fn scan_frame(
    py: Python<'_>,
    finder: &GridFinderPy,
    reader: &CellReaderPy,
    dawg: &DawgPy,
    rgb: &[u8],
    width: usize,
    height: usize,
    focal_px: f64,
    prior: Option<Vec<Vec<char>>>,
    prior_blanks: Option<Vec<Vec<bool>>>,
) -> PyResult<Option<FrameScanPy>> {
    let prior = prior_board(prior, prior_blanks)?;
    let out = py
        .detach(|| {
            frame::scan_frame(
                &finder.inner,
                &reader.inner,
                &dawg.inner,
                rgb,
                width,
                height,
                focal_px,
                prior.as_ref(),
                &DecodeOptions::default(),
            )
        })
        .map_err(to_py)?;
    let Some(scan) = out else { return Ok(None) };
    Ok(Some(FrameScanPy {
        corners: scan.corners.map(|[x, y]| (x, y)),
        image: scan.image,
        side: scan.side,
        readings: scan
            .readings
            .iter()
            .map(|row| {
                row.iter()
                    .map(|c| (c.p_tile, c.letters.clone(), c.p_blank))
                    .collect()
            })
            .collect(),
        decoded: Py::new(py, decoded_py(&scan.decoded))?,
    }))
}

fn prior_board(
    prior: Option<Vec<Vec<char>>>,
    blanks: Option<Vec<Vec<bool>>>,
) -> PyResult<Option<ScanBoard>> {
    let Some(letters) = prior else {
        return Ok(None);
    };
    Ok(Some(ScanBoard {
        letters: grid_of(&letters, "prior", |&ch| ch)?,
        blanks: match &blanks {
            Some(b) => grid_of(b, "prior_blanks", |&x| x)?,
            None => [[false; SIZE]; SIZE],
        },
    }))
}

fn decoded_py(out: &scan_decode::Decoded) -> DecodedScanPy {
    DecodedScanPy {
        letters: rows_of(&out.board.letters),
        blanks: rows_of(&out.board.blanks),
        corrected: rows_of(&out.corrected),
        flagged: rows_of(&out.flagged),
        legal: out.legal,
        complete: out.complete,
        cost: out.cost,
        work: out.work,
    }
}

/// The board `decode_scan` settled on.
#[pyclass(name = "DecodedScan", frozen)]
struct DecodedScanPy {
    /// 15x15, `'-'` for an empty square.
    #[pyo3(get)]
    letters: Vec<Vec<char>>,
    #[pyo3(get)]
    blanks: Vec<Vec<bool>>,
    #[pyo3(get)]
    corrected: Vec<Vec<bool>>,
    #[pyo3(get)]
    flagged: Vec<Vec<bool>>,
    #[pyo3(get)]
    legal: bool,
    #[pyo3(get)]
    complete: bool,
    #[pyo3(get)]
    cost: f32,
    #[pyo3(get)]
    work: usize,
}

/// One square's reading as Python passes it: `(p_tile, [(letter, p)], p_blank)`.
type ReadingTuple = (f32, Vec<(char, f32)>, f32);

fn grid_of<T: Clone, U>(
    rows: &[Vec<T>],
    what: &str,
    f: impl Fn(&T) -> U,
) -> PyResult<[[U; SIZE]; SIZE]> {
    if rows.len() != SIZE || rows.iter().any(|row| row.len() != SIZE) {
        return Err(PyValueError::new_err(format!("{what} must be 15x15")));
    }
    Ok(std::array::from_fn(|r| {
        std::array::from_fn(|c| f(&rows[r][c]))
    }))
}

fn rows_of<T: Copy>(grid: &[[T; SIZE]; SIZE]) -> Vec<Vec<T>> {
    grid.iter().map(|row| row.to_vec()).collect()
}

/// The most probable legal board given per-square readings: see
/// `scan::decode` in the engine. Options left as `None` take the engine's
/// defaults.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (dawg, readings, prior=None, prior_blanks=None, *, min_p_tile=None, max_p_tile=None, letter_floor=None, blank_floor=None, max_flips=None, budget=None))]
fn decode_scan(
    py: Python<'_>,
    dawg: &DawgPy,
    readings: Vec<Vec<ReadingTuple>>,
    prior: Option<Vec<Vec<char>>>,
    prior_blanks: Option<Vec<Vec<bool>>>,
    min_p_tile: Option<f32>,
    max_p_tile: Option<f32>,
    letter_floor: Option<f32>,
    blank_floor: Option<f32>,
    max_flips: Option<usize>,
    budget: Option<usize>,
) -> PyResult<DecodedScanPy> {
    let readings = grid_of(&readings, "readings", |(p_tile, letters, p_blank)| {
        CellReading {
            p_tile: *p_tile,
            letters: letters.clone(),
            p_blank: *p_blank,
        }
    })?;
    let prior = prior_board(prior, prior_blanks)?;
    let d = DecodeOptions::default();
    let options = DecodeOptions {
        min_p_tile: min_p_tile.unwrap_or(d.min_p_tile),
        max_p_tile: max_p_tile.unwrap_or(d.max_p_tile),
        letter_floor: letter_floor.unwrap_or(d.letter_floor),
        blank_floor: blank_floor.unwrap_or(d.blank_floor),
        max_flips: max_flips.unwrap_or(d.max_flips),
        budget: budget.unwrap_or(d.budget),
    };
    let out = py
        .detach(|| scan_decode::decode(&dawg.inner, &readings, prior.as_ref(), &options))
        .map_err(to_py)?;
    Ok(decoded_py(&out))
}

/// Set how many threads parallel move generation (`get_best_words(..., parallel=True)`)
/// uses. Must be called before the first generation; raises `RuntimeError` if the
/// pool is already built. Overrides the `RAYON_NUM_THREADS` env var and the
/// `min(8, cores)` default.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    engine::set_num_threads(n).map_err(to_py)
}

/// Number of threads parallel move generation will use (builds the pool on the
/// first call if it does not exist yet).
#[pyfunction]
fn num_threads() -> usize {
    engine::num_threads()
}

#[pymodule]
fn scrablozaur(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<LanguagePy>()?;
    m.add_class::<DawgPy>()?;
    m.add_class::<BoardPy>()?;
    m.add_class::<LeaveNetPy>()?;
    m.add_class::<DecodedScanPy>()?;
    m.add_class::<CellReaderPy>()?;
    m.add_class::<GridFinderPy>()?;
    m.add_function(pyo3::wrap_pyfunction!(decode_scan, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(scan_tile_tops, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(scan_straighten, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(scan_frame, m)?)?;
    m.add_class::<FrameScanPy>()?;
    m.add_function(pyo3::wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}
