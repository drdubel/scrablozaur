use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

// Per-quadrant bonus lookup. Index by (min(r, 14-r), min(c, 14-c)).
// Entry: (letter_multiplier, word_multiplier).
static BONUS_TABLE: [[(u8, u8); 8]; 8] = [
    //       0       1       2       3       4       5       6       7
    [
        (1, 3),
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 3),
    ],
    [
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (3, 1),
        (1, 1),
        (1, 1),
    ],
    [
        (1, 1),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 1),
    ],
    [
        (2, 1),
        (1, 1),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 1),
    ],
    [
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 2),
        (1, 1),
        (1, 1),
        (1, 1),
    ],
    [
        (1, 1),
        (3, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (3, 1),
        (1, 1),
        (1, 1),
    ],
    [
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 1),
    ],
    [
        (1, 3),
        (1, 1),
        (1, 1),
        (2, 1),
        (1, 1),
        (1, 1),
        (1, 1),
        (1, 2),
    ],
];

const BOARD_SIZE: usize = 15;
const CENTER: usize = BOARD_SIZE / 2;
// Standard Scrabble rack capacity; also the tile count that earns the
// end-of-move bonus and the bag-size exchange threshold.
const RACK_SIZE: usize = 7;

/// Board coordinates of the `i`-th letter of a word placed at (row, col)
/// running horizontally or vertically from there.
fn word_cell(row: usize, col: usize, horizontal: bool, i: usize) -> (usize, usize) {
    if horizontal {
        (row, col + i)
    } else {
        (row + i, col)
    }
}

fn in_bounds(row: usize, col: usize) -> bool {
    row < BOARD_SIZE && col < BOARD_SIZE
}

/// A candidate move: the word, its score, its (row, col, horizontal)
/// placement, and the letters drawn from the player's hand to play it.
type BestWord = (String, u32, (usize, usize, bool), Vec<char>);

/// Count of each non-blank letter in a rack. Used to allocate a word's
/// letters to real tiles before falling back to blanks, so a letter that
/// appears in the word more times than the rack has real copies of it
/// correctly runs out and defers the extra occurrences to a blank.
fn real_letter_counts(letters: &str) -> HashMap<char, u32> {
    let mut freq = HashMap::new();
    for c in letters.chars().filter(|&c| c != '?') {
        *freq.entry(c).or_insert(0) += 1;
    }
    freq
}

/// Letter- and word-multiplier for the board square at (row, col), read via
/// the BONUS_TABLE's single quadrant (the table is symmetric, so every
/// square maps to one of its 8x8 entries by mirroring across the centre).
fn quadrant_bonus(row: usize, col: usize) -> (u8, u8) {
    let last = (BOARD_SIZE - 1) as u8;
    let r = (row as u8).min(last - row as u8);
    let c = (col as u8).min(last - col as u8);
    BONUS_TABLE[r as usize][c as usize]
}

// Polish letter point values, shared by calculate_word_points and rack_value
// so the two can never drift apart.
#[pyfunction]
fn letter_points(c: char) -> u32 {
    match c.to_uppercase().next().unwrap_or(c) {
        'A' | 'E' | 'I' | 'O' | 'Z' | 'W' | 'N' | 'S' | 'R' => 1,
        'D' | 'Y' | 'C' | 'K' | 'L' | 'M' | 'P' | 'T' => 2,
        'B' | 'G' | 'H' | 'J' | 'Ł' | 'U' => 3,
        'Ą' | 'Ę' | 'F' | 'Ó' | 'Ś' | 'Ż' => 5,
        'Ć' => 6,
        'Ń' => 7,
        'Ź' => 9,
        _ => 0,
    }
}

// Covers all Polish letters (max 'ż' = U+017C = 380) and the blank tile '?' (U+003F = 63).
const FREQ_SIZE: usize = 400;
type LetterFreq = [u8; FREQ_SIZE];

fn build_freq(letters: &str) -> (LetterFreq, usize) {
    let mut freq = [0u8; FREQ_SIZE];
    let mut count = 0usize;
    for c in letters.chars() {
        freq[c as usize] += 1;
        count += 1;
    }
    (freq, count)
}

// ---------------------------------------------------------------------------
// GADDAG support: 32-letter Polish alphabet + cross-check bitsets
// ---------------------------------------------------------------------------

/// The 32 letters of the Polish alphabet, in collation order. A cross-check
/// set is a `u32` with one bit per letter (see `letter_bit`), so a single
/// word forms a legal perpendicular cross-word at a square iff its bit is set.
const POLISH_ALPHABET: &str = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż";
const ALPHABET_SIZE: usize = 32;
/// All 32 alphabet bits set — the cross-check for a square with no
/// perpendicular neighbours (any letter is allowed, no cross-word forms).
const CROSS_ANY: u32 = if ALPHABET_SIZE == 32 { u32::MAX } else { (1u32 << ALPHABET_SIZE) - 1 };

/// Delimiter separating a GADDAG entry's reversed prefix from its forward
/// suffix. `'\0'` sorts before every letter, so the builder's sorted-input
/// requirement and the runtime binary search both treat it consistently.
const SEP: char = '\0';

// Codepoint -> alphabet index (0..31), or -1 for non-alphabet chars (`SEP`,
// blank `'?'`, punctuation). Filled once from POLISH_ALPHABET; the range
// matches FREQ_SIZE so a lookup is a single bounds-checked array index.
static LETTER_INDEX: OnceLock<[i8; FREQ_SIZE]> = OnceLock::new();
fn letter_index_table() -> &'static [i8; FREQ_SIZE] {
    LETTER_INDEX.get_or_init(|| {
        let mut table = [-1i8; FREQ_SIZE];
        for (i, c) in POLISH_ALPHABET.chars().enumerate() {
            table[c as usize] = i as i8;
        }
        table
    })
}

/// Bit index (0..32) of a Polish letter, or `None` for anything else.
fn letter_index(c: char) -> Option<usize> {
    let i = c as usize;
    if i < FREQ_SIZE {
        let idx = letter_index_table()[i];
        if idx >= 0 {
            return Some(idx as usize);
        }
    }
    None
}

/// Single-bit cross-check mask for `c` (0 if `c` is not an alphabet letter).
#[inline]
fn letter_bit(c: char) -> u32 {
    letter_index(c).map_or(0, |i| 1u32 << i)
}

/// Symbol index for the flat DAWG's per-node child bitmap: `0..31` for the
/// alphabet letters (matching `letter_index`), `32` for the GADDAG `SEP` edge.
/// `None` for anything else — there should be no such edges in a well-formed
/// lexicon, and the runtime falls back to a linear scan for them.
#[inline]
fn symbol_index(c: char) -> Option<u32> {
    if c == SEP {
        Some(ALPHABET_SIZE as u32)
    } else {
        letter_index(c).map(|i| i as u32)
    }
}

// ---------------------------------------------------------------------------
// Move-generation thread pool
// ---------------------------------------------------------------------------

// Parallel move generation fans out across board anchors. A single move is
// cheap, so a pool wider than ~8 threads spends more on coordination than it
// saves (it can even regress past that). The engine therefore runs its own
// pool capped at 8 threads by default, instead of rayon's all-cores global
// pool. Resolution order for the thread count: an explicit `set_num_threads(n)`
// (must precede the first generation) > the `RAYON_NUM_THREADS` env var (which
// also propagates to spawned worker processes) > `min(8, cores)`.
const DEFAULT_MAX_THREADS: usize = 8;
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0); // 0 = unset
static GEN_POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();

fn desired_threads() -> usize {
    let requested = REQUESTED_THREADS.load(Ordering::Relaxed);
    if requested != 0 {
        return requested;
    }
    if let Some(n) = std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return n;
    }
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    cores.min(DEFAULT_MAX_THREADS)
}

/// The move-generation pool, built once on first use from `desired_threads()`.
fn gen_pool() -> &'static rayon::ThreadPool {
    GEN_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(desired_threads())
            .thread_name(|i| format!("scrablozaur-gen-{i}"))
            .build()
            .expect("failed to build move-generation thread pool")
    })
}

// ---------------------------------------------------------------------------
// Build-time DAWG node
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Node {
    is_terminal: bool,
    children: HashMap<char, u32>,
}

struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn alloc(&mut self) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(Node::default());
        id
    }

    fn node(&self, id: u32) -> &Node {
        &self.nodes[id as usize]
    }

    fn node_mut(&mut self, id: u32) -> &mut Node {
        &mut self.nodes[id as usize]
    }
}

// ---------------------------------------------------------------------------
// Flat, read-only DAWG loaded from a binary file
// ---------------------------------------------------------------------------

/// One decoded outgoing edge of a flat DAWG node: the edge letter, its
/// precomputed cross-check bit (`letter_bit`, `0` for `SEP`/non-alphabet), and
/// the child node id. Decoding the codepoint and bit once at load keeps them
/// off the traversal hot path.
struct Child {
    ch: char,
    bit: u32,
    id: u32,
}

/// Flat, read-only DAWG/GADDAG decoded from its binary file into a
/// direct-indexed form. Per node: a `terminal` flag and a 33-bit child-presence
/// `bitmap` (bit `symbol_index(letter)`); `child_start` delimits each node's
/// slice of the contiguous `children`, packed in symbol-index order. `find_child`
/// is O(1) via the bitmap + a popcount; iteration reads the decoded `Child`
/// slice directly, with no per-edge codepoint decoding.
struct Dawg {
    root: u32,
    terminal: Vec<bool>,
    child_start: Vec<u32>,
    bitmap: Vec<u64>,
    children: Vec<Child>,
}

impl Dawg {
    fn load(path: &str) -> io::Result<Self> {
        Self::from_bytes(fs::read(path)?)
    }

    fn from_bytes(data: Vec<u8>) -> io::Result<Self> {
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }
        let root = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let node_count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let mut terminal = Vec::with_capacity(node_count);
        let mut child_start = Vec::with_capacity(node_count + 1);
        let mut bitmap = Vec::with_capacity(node_count);
        let mut children: Vec<Child> = Vec::new();

        let mut pos = 8usize;
        for _ in 0..node_count {
            child_start.push(children.len() as u32);
            terminal.push(data[pos] != 0);
            pos += 1;
            let n_children = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let start = children.len();
            for _ in 0..n_children {
                let cp = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                let cid = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
                pos += 8;
                let ch = char::from_u32(cp).unwrap();
                children.push(Child { ch, bit: letter_bit(ch), id: cid });
            }
            // Pack this node's children in symbol-index order (any non-alphabet,
            // non-SEP edge sorts after all known symbols) so the child bitmap and
            // the popcount indexing in `find_child` stay exact.
            children[start..].sort_by_key(|c| symbol_index(c.ch).unwrap_or(u32::MAX));
            let mut bm = 0u64;
            for c in &children[start..] {
                if let Some(sym) = symbol_index(c.ch) {
                    bm |= 1u64 << sym;
                }
            }
            bitmap.push(bm);
        }
        child_start.push(children.len() as u32);

        Ok(Self {
            root,
            terminal,
            child_start,
            bitmap,
            children,
        })
    }

    #[inline]
    fn node_is_terminal(&self, id: u32) -> bool {
        self.terminal[id as usize]
    }

    #[inline]
    fn node_children_count(&self, id: u32) -> usize {
        let i = id as usize;
        (self.child_start[i + 1] - self.child_start[i]) as usize
    }

    /// The node's decoded outgoing edges, packed in symbol-index order.
    #[inline]
    fn node_children(&self, id: u32) -> &[Child] {
        let i = id as usize;
        &self.children[self.child_start[i] as usize..self.child_start[i + 1] as usize]
    }

    #[inline]
    fn node_child(&self, id: u32, i: usize) -> (char, u32) {
        let ch = &self.children[self.child_start[id as usize] as usize + i];
        (ch.ch, ch.id)
    }

    /// O(1) edge lookup: the child bitmap tells whether an edge for `c` exists,
    /// and a popcount of the bits below it gives its index in the packed slice.
    #[inline]
    fn find_child(&self, id: u32, c: char) -> Option<u32> {
        let i = id as usize;
        match symbol_index(c) {
            Some(sym) => {
                let bit = 1u64 << sym;
                let bm = self.bitmap[i];
                if bm & bit == 0 {
                    return None;
                }
                let off = (bm & (bit - 1)).count_ones() as usize;
                Some(self.children[self.child_start[i] as usize + off].id)
            }
            // Non-alphabet, non-SEP edges are absent from the bitmap; scan for them.
            None => self
                .node_children(id)
                .iter()
                .find(|ch| ch.ch == c)
                .map(|ch| ch.id),
        }
    }

    fn contains(&self, word: &str) -> bool {
        let mut curr = self.root;
        for c in word.chars() {
            match self.find_child(curr, c) {
                Some(next) => curr = next,
                None => return false,
            }
        }
        self.node_is_terminal(curr)
    }

    fn search_inner(&self, pattern: &str, letters: &str) -> Vec<String> {
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let mandatory_slots = pattern_chars.iter().filter(|&&c| c == '-').count();
        let (mut freq, bag_count) = build_freq(letters);
        let mut results = Vec::new();
        let mut current = String::with_capacity(pattern.len());
        self.match_pattern(
            &pattern_chars,
            0,
            self.root,
            &mut freq,
            bag_count,
            mandatory_slots,
            &mut results,
            &mut current,
        );
        results.sort_unstable();
        results.dedup();
        results
    }

    fn node_count(&self) -> usize {
        self.terminal.len()
    }

    /// Traverse the DAWG matching `pattern` against the given letter bag.
    ///
    /// Pattern tokens:
    ///   - fixed char — must match exactly at this position
    ///   - `'-'`      — consume exactly one letter from the bag
    ///   - `'*'`      — consume zero or more letters from the bag
    ///
    /// `freq` / `bag_count` represent the current bag state. `mandatory_slots` counts
    /// how many `-` tokens remain so `*` expansions cannot starve them.
    #[allow(clippy::too_many_arguments)]
    fn match_pattern(
        &self,
        pattern: &[char],
        pat_pos: usize,
        node_id: u32,
        freq: &mut LetterFreq,
        bag_count: usize,
        mandatory_slots: usize,
        results: &mut Vec<String>,
        current: &mut String,
    ) {
        if pat_pos == pattern.len() {
            if self.node_is_terminal(node_id) {
                results.push(current.clone());
            }
            return;
        }

        match pattern[pat_pos] {
            '-' => {
                let n = self.node_children_count(node_id);
                for i in 0..n {
                    let (c, child_id) = self.node_child(node_id, i);
                    self.try_consume_letter(
                        pattern,
                        pat_pos + 1,
                        child_id,
                        c,
                        freq,
                        bag_count - 1,
                        mandatory_slots - 1,
                        results,
                        current,
                    );
                }
            }
            '*' => {
                // consume zero letters for this `*`
                self.match_pattern(
                    pattern,
                    pat_pos + 1,
                    node_id,
                    freq,
                    bag_count,
                    mandatory_slots,
                    results,
                    current,
                );
                // consume one more letter and stay at the same `*` position
                if bag_count > mandatory_slots {
                    let n = self.node_children_count(node_id);
                    for i in 0..n {
                        let (c, child_id) = self.node_child(node_id, i);
                        self.try_consume_letter(
                            pattern,
                            pat_pos,
                            child_id,
                            c,
                            freq,
                            bag_count - 1,
                            mandatory_slots,
                            results,
                            current,
                        );
                    }
                }
            }
            fixed_char => {
                if let Some(child_id) = self.find_child(node_id, fixed_char) {
                    current.push(fixed_char);
                    self.match_pattern(
                        pattern,
                        pat_pos + 1,
                        child_id,
                        freq,
                        bag_count,
                        mandatory_slots,
                        results,
                        current,
                    );
                    current.pop();
                }
            }
        }
    }

    /// Recurse into `child_id` for letter `c`, once as the exact drawn tile
    /// and once as a blank standing in for it, backtracking `freq` after
    /// each. Shared by the `'-'` and `'*'` branches of `match_pattern`,
    /// which differ only in which pattern position and bag counts to
    /// recurse with.
    #[allow(clippy::too_many_arguments)]
    fn try_consume_letter(
        &self,
        pattern: &[char],
        next_pat_pos: usize,
        child_id: u32,
        c: char,
        freq: &mut LetterFreq,
        next_bag_count: usize,
        next_mandatory_slots: usize,
        results: &mut Vec<String>,
        current: &mut String,
    ) {
        let ci = c as usize;
        if ci < FREQ_SIZE && freq[ci] > 0 {
            freq[ci] -= 1;
            current.push(c);
            self.match_pattern(
                pattern,
                next_pat_pos,
                child_id,
                freq,
                next_bag_count,
                next_mandatory_slots,
                results,
                current,
            );
            current.pop();
            freq[ci] += 1;
        }

        let qi = '?' as usize;
        if freq[qi] > 0 {
            freq[qi] -= 1;
            current.push(c);
            self.match_pattern(
                pattern,
                next_pat_pos,
                child_id,
                freq,
                next_bag_count,
                next_mandatory_slots,
                results,
                current,
            );
            current.pop();
            freq[qi] += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Python-exposed types
// ---------------------------------------------------------------------------

#[pyclass(name = "Dawg")]
struct DawgPy {
    inner: Dawg,
    /// The GADDAG for the same lexicon, driving `get_best_words`. Loaded from
    /// `gaddag.bin` alongside the DAWG (auto-located next to it, or given
    /// explicitly). `None` if no GADDAG file is present, in which case move
    /// generation falls back to the legacy DAWG pattern search.
    gaddag: Option<Dawg>,
}

/// If `path` is `.../dawg.bin`, the sibling `.../gaddag.bin`; otherwise `None`.
fn sibling_gaddag_path(path: &str) -> Option<String> {
    let p = std::path::Path::new(path);
    let name = p.file_name()?.to_str()?;
    let sibling = name.replacen("dawg", "gaddag", 1);
    if sibling == name {
        return None;
    }
    Some(p.with_file_name(sibling).to_str()?.to_string())
}

#[pymethods]
impl DawgPy {
    /// Load a DAWG from `path`. If `gaddag_path` is omitted, a sibling
    /// `gaddag.bin` (same directory, `dawg`→`gaddag` in the filename) is loaded
    /// when present; move generation uses it automatically. Passing a path that
    /// does not exist is an error; omitting it and finding no sibling simply
    /// leaves the GADDAG absent (legacy generation).
    #[new]
    #[pyo3(signature = (path, gaddag_path=None))]
    fn new(path: &str, gaddag_path: Option<&str>) -> PyResult<Self> {
        let inner = Dawg::load(path).map_err(|e| PyIOError::new_err(e.to_string()))?;
        let gaddag = match gaddag_path {
            Some(gp) => Some(Dawg::load(gp).map_err(|e| PyIOError::new_err(e.to_string()))?),
            None => match sibling_gaddag_path(path) {
                Some(gp) if std::path::Path::new(&gp).exists() => {
                    Some(Dawg::load(&gp).map_err(|e| PyIOError::new_err(e.to_string()))?)
                }
                _ => None,
            },
        };
        Ok(DawgPy { inner, gaddag })
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
        self.gaddag.is_some()
    }

    fn search(&self, pattern: &str, letters: &str) -> Vec<String> {
        self.inner.search_inner(pattern, letters)
    }
}

fn row_to_string(row: &[char; BOARD_SIZE]) -> String {
    row.iter()
        .map(|c| c.to_string())
        .collect::<Vec<_>>()
        .join(" ")
}

/// One step of a xorshift64 PRNG, used to draw tiles without pulling in a
/// full RNG crate for something this simple.
fn xorshift(seed: &mut u64) {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
}

/// Default RNG seed for an unseeded `Board`, from the current time mixed with
/// a per-process counter so two boards built in the same nanosecond still
/// diverge. Seeded boards (`Board::seeded`) bypass this entirely.
fn time_seed() -> u64 {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed) as u64;
    // Mix so consecutive counter values don't produce correlated streams.
    (nanos ^ n.wrapping_mul(0x9E3779B97F4A7C15)) | 1
}

/// Alphabetical rank of `c` for `first_draw_winner`'s "closest to 'A'"
/// tiebreak: a blank outranks every letter, unknown characters sort last.
fn alphabet_rank(c: char) -> i32 {
    const ALPHABET: &str = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż";
    if c == '?' {
        -1
    } else {
        ALPHABET
            .chars()
            .position(|a| a == c)
            .map_or(i32::MAX, |p| p as i32)
    }
}

/// Shuffle in place (Fisher-Yates) using `rng`. The modulo bias here is on the
/// order of 100/2^64 and not worth a rejection loop.
fn shuffle(tiles: &mut [char], rng: &mut u64) {
    for i in (1..tiles.len()).rev() {
        xorshift(rng);
        let j = (*rng as usize) % (i + 1);
        tiles.swap(i, j);
    }
}

/// Standard Polish Scrabble tile distribution (100 tiles).
fn fresh_tile_bag() -> Vec<char> {
    vec![
        'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'ą', 'b', 'b', 'c', 'c', 'c', 'ć', 'd', 'd',
        'd', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'ę', 'f', 'g', 'g', 'h', 'h', 'i', 'i', 'i', 'i',
        'i', 'i', 'i', 'i', 'j', 'j', 'k', 'k', 'k', 'l', 'l', 'l', 'ł', 'ł', 'm', 'm', 'm', 'n',
        'n', 'n', 'n', 'n', 'ń', 'o', 'o', 'o', 'o', 'o', 'o', 'ó', 'p', 'p', 'p', 'r', 'r', 'r',
        'r', 's', 's', 's', 's', 'ś', 't', 't', 't', 'u', 'u', 'w', 'w', 'w', 'w', 'y', 'y', 'y',
        'y', 'z', 'z', 'z', 'z', 'z', 'ź', 'ż', '?', '?',
    ]
}

// `skip_from_py_object`: deriving Clone would otherwise opt `Board` into
// by-value extraction from Python. Nothing here wants that -- every Rust-side
// use borrows -- and a silent clone at an FFI boundary is exactly the kind of
// thing that makes a search layer mysteriously slow.
#[pyclass(name = "Board", skip_from_py_object)]
#[derive(Clone)]
struct Board {
    board: [[char; BOARD_SIZE]; BOARD_SIZE],
    /// Which occupied squares hold a blank tile playing *as* the letter in
    /// `board`. A blank keeps its own zero value forever, so every later
    /// cross-word or extension through that square must score it as 0 — hence
    /// the mask travels with the grid rather than being reconstructed.
    blanks: [[bool; BOARD_SIZE]; BOARD_SIZE],
    tile_bag: Vec<char>,
    first: bool,
    /// Draw RNG state. Advanced by `give_letters`, so a `Board::seeded(s)`
    /// replays an identical bag order — the basis of the paired-seed arena.
    rng: u64,
}

#[pymethods]
impl Board {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Self::with_seed(time_seed()))
    }

    /// A fresh board whose bag is shuffled deterministically from `seed`. Two
    /// boards built with the same seed deal the same tiles in the same order,
    /// which is what lets the arena play one bag twice with the seats swapped.
    #[staticmethod]
    fn seeded(seed: u64) -> PyResult<Self> {
        // 0 is a fixed point of xorshift64; nudge it to keep every seed usable.
        Ok(Self::with_seed(if seed == 0 {
            0x9E3779B97F4A7C15
        } else {
            seed
        }))
    }

    /// Deep copy, including the bag and the RNG stream position. The only way
    /// to branch a position without the lossy `str()` -> `from_grid()` round
    /// trip, and the basis of make/unmake search.
    fn copy(&self) -> Board {
        self.clone()
    }

    /// Construct a board pre-filled from a 15x15 grid of single-character
    /// cells (e.g. loaded from a saved game or a scanned photo), each cell
    /// either a letter or `'-'` for empty. Starts with a full standard
    /// tile bag, same as `Board()` -- letters already on the grid are not
    /// subtracted from it, since callers that load a grid this way manage
    /// their own separate tile-bag bookkeeping rather than relying on this
    /// board's.
    #[staticmethod]
    #[pyo3(signature = (board, blanks=None))]
    fn from_grid(board: Vec<Vec<String>>, blanks: Option<Vec<Vec<bool>>>) -> PyResult<Self> {
        if board.len() != BOARD_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "board must have exactly 15 rows",
            ));
        }
        if let Some(mask) = &blanks {
            if mask.len() != BOARD_SIZE || mask.iter().any(|row| row.len() != BOARD_SIZE) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "blanks mask must be 15x15",
                ));
            }
        }
        let mut result = [['-'; BOARD_SIZE]; BOARD_SIZE];
        let mut blank_mask = [[false; BOARD_SIZE]; BOARD_SIZE];
        let mut first = true;
        let mut tile_bag = fresh_tile_bag();
        for (r, row) in board.iter().enumerate() {
            if row.len() != BOARD_SIZE {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each row must have exactly 15 columns",
                ));
            }
            for (c, cell) in row.iter().enumerate() {
                let mut chars = cell.chars();
                let ch = chars
                    .next()
                    .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("empty cell"))?;
                if chars.next().is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cell must contain exactly one character",
                    ));
                }
                result[r][c] = ch;
                if ch != '-' {
                    first = false;
                    let is_blank = blanks.as_ref().is_some_and(|m| m[r][c]);
                    blank_mask[r][c] = is_blank;
                    // A blank on the board consumes a '?' from the bag, not a
                    // copy of the letter it happens to be standing in for.
                    let consumed = if is_blank { '?' } else { ch };
                    if let Some(pos) = tile_bag.iter().position(|&x| x == consumed) {
                        tile_bag.remove(pos);
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "letter '{}' not available in tile bag",
                            consumed
                        )));
                    }
                }
            }
        }
        // Whatever the grid left in the bag is still in distribution order;
        // shuffle it so the first draw isn't alphabetical.
        let mut rng = time_seed();
        shuffle(&mut tile_bag, &mut rng);
        Ok(Board {
            board: result,
            blanks: blank_mask,
            tile_bag,
            first,
            rng,
        })
    }

    fn __str__(&self) -> String {
        self.board
            .iter()
            .map(row_to_string)
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn give_letters(&mut self, letters: &str) -> String {
        let mut drawn = String::new();
        let held = letters.chars().count();
        // A rack over capacity (shouldn't happen, but don't underflow on it).
        let draw_count = RACK_SIZE.saturating_sub(held).min(self.tile_bag.len());
        for _ in 0..draw_count {
            // The bag is shuffled once at construction and drawn from the end,
            // so the tile *sequence* is fixed up front. Picking a random index
            // per draw instead would make the sequence depend on how many
            // tiles each draw happened to take, which destroys the
            // correlation a paired benchmark relies on.
            if let Some(tile) = self.tile_bag.pop() {
                drawn.push(tile);
            }
        }
        drawn
    }

    /// How many tiles are left in the bag. Public information in Scrabble --
    /// unlike `bag_tiles`, this is safe to expose to a player.
    fn bag_remaining(&self) -> usize {
        self.tile_bag.len()
    }

    /// The bag's actual contents, in draw order — the *last* element is the
    /// next tile out. Hidden information: for the arena, tests and endgame
    /// setup, never for a player's own decision-making.
    fn bag_tiles(&self) -> Vec<char> {
        self.tile_bag.clone()
    }

    /// Replace the bag wholesale, in draw order (last element drawn first).
    /// Used to construct exact positions (endgame tests, pre-endgame
    /// enumeration) without replaying a whole game. Not shuffled — the caller
    /// decides the order, which is the point.
    fn set_bag(&mut self, tiles: Vec<char>) {
        self.tile_bag = tiles;
    }

    /// Which occupied squares hold a blank, as a 15x15 mask. Pairs with
    /// `from_grid(grid, blanks)` to round-trip a position losslessly.
    fn blank_mask(&self) -> Vec<Vec<bool>> {
        self.blanks.iter().map(|row| row.to_vec()).collect()
    }

    fn exchange_letters(&mut self, letters: &str, letters_to_exchange: &str) -> String {
        let mut remaining: Vec<char> = letters.chars().collect();
        for ch in letters_to_exchange.chars() {
            if let Some(pos) = remaining.iter().position(|&c| c == ch) {
                remaining.remove(pos);
            }
        }
        let remaining: String = remaining.into_iter().collect();
        // Draw the replacements *before* returning the discards, so a player
        // can't draw back the tiles they just threw away.
        let drawn = self.give_letters(&remaining);
        for ch in letters_to_exchange.chars() {
            // Slot each returned tile at a uniformly random depth. Pushing to
            // the end would put it straight back on top of the draw pile.
            xorshift(&mut self.rng);
            let idx = (self.rng as usize) % (self.tile_bag.len() + 1);
            self.tile_bag.insert(idx, ch);
        }
        remaining + &drawn
    }

    /// Standard Scrabble rule: exchanging tiles for new ones from the bag is
    /// only allowed while at least a full rack's worth of tiles remain in
    /// the bag, regardless of how many tiles the player wants to exchange.
    fn can_exchange(&self) -> bool {
        let bag_remaining = self.tile_bag.len();
        bag_remaining >= RACK_SIZE
    }

    /// The standard Polish Scrabble tile distribution (100 tiles) that
    /// `Board()` and `Board.from_grid()` each start with.
    #[staticmethod]
    fn fresh_tile_bag() -> Vec<char> {
        fresh_tile_bag()
    }

    /// Sum of face point values of a rack (blank tiles score 0, matching
    /// their in-play scoring) -- used for the standard end-of-game scoring
    /// adjustment: the player who goes out gains this value from each
    /// opponent's rack, everyone else loses it from their own.
    #[staticmethod]
    fn rack_value(letters: &str) -> u32 {
        letters.chars().map(letter_points).sum()
    }

    /// Face point value of a single letter. Blanks (`'?'`) score 0 here,
    /// same as their fixed in-play scoring in `calculate_word_points`.
    #[staticmethod]
    fn letter_points(letter: char) -> u32 {
        letter_points(letter)
    }

    /// Standard rule for who goes first: each player draws one tile, the
    /// one closest to 'A' in alphabet order goes first, and a blank beats
    /// every letter. Returns the *index* into `draws` of the winner (first
    /// index wins ties). Drawn tiles are not consumed here -- the caller is
    /// responsible for returning them to the bag before dealing real racks.
    #[staticmethod]
    fn first_draw_winner(draws: Vec<char>) -> usize {
        draws
            .iter()
            .enumerate()
            .min_by_key(|&(_, &c)| alphabet_rank(c))
            .map_or(0, |(i, _)| i)
    }

    fn calculate_word_points(
        &self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        letters: &str,
    ) -> PyResult<u32> {
        // The main word and every cross-word it forms are each scored
        // independently (own letter multipliers + own word multiplier, from
        // only the one square where the shared new tile lands), then summed
        // — never merge their tile totals before multiplying, or a word
        // multiplier from elsewhere in the main word would incorrectly leak
        // into an unrelated cross-word's score.
        let mut main_total = 0u32;
        let mut main_word_mul = 1u32;
        let mut cross_words_total = 0u32;
        let mut tiles_from_hand = 0usize;
        // Depleted as real letters are claimed by earlier tiles in the
        // word, so a letter repeated more times than the rack has real
        // copies of it correctly falls back to a blank (0 points) for the
        // extra occurrences instead of being scored as real every time.
        let mut hand_freq = real_letter_counts(letters);

        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if !in_bounds(r, c) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word out of bounds",
                ));
            }
            let bonus = quadrant_bonus(r, c);

            if self.board[r][c] == '-' {
                let this_letter_value = match hand_freq.get_mut(&ch) {
                    Some(count) if *count > 0 => {
                        *count -= 1;
                        letter_points(ch)
                    }
                    // No real copy left in hand — a blank stands in for
                    // this letter and always scores 0.
                    _ => 0,
                };
                tiles_from_hand += 1;

                main_total += this_letter_value * bonus.0 as u32;
                main_word_mul *= bonus.1 as u32;

                // cross-word formed by this newly placed letter, if any,
                // scored on its own (this tile's value + existing
                // perpendicular neighbours), multiplied only by this tile's
                // own word bonus.
                if let Some(neighbor_points) = self.cross_neighbor_points(r, c, horizontal) {
                    let cross_total = this_letter_value * bonus.0 as u32 + neighbor_points;
                    cross_words_total += cross_total * bonus.1 as u32;
                }
            } else {
                if ch != self.board[r][c] {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "letter on board '{}' does not match word letter '{}'",
                        self.board[r][c], ch,
                    )));
                }
                main_total += self.tile_points(r, c);
            }
        }

        // Bonus for using the whole rack in one move
        Ok(main_total * main_word_mul
            + cross_words_total
            + if tiles_from_hand == RACK_SIZE { 50 } else { 0 })
    }

    fn check_word_placement(
        &self,
        dawg: &DawgPy,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> PyResult<()> {
        // Standard rule: a play must be at least 2 letters -- enforced
        // explicitly rather than relying on the dictionary happening to
        // have no 1-letter entries.
        if word.chars().count() < 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "word must be at least 2 letters",
            ));
        }
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if !in_bounds(r, c) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word out of bounds",
                ));
            }
            if self.board[r][c] != '-' {
                continue;
            }

            let adjacent = self.cross_word(r, c, horizontal, ch);
            if adjacent.len() > 1 && !dawg.contains(&adjacent) {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "cross-word '{adjacent}' formed by '{ch}' is not in the dictionary",
                )));
            }
        }
        Ok(())
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
        let mut filled = Vec::new();
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if !in_bounds(r, c) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word out of bounds",
                ));
            }
            if self.board[r][c] == '-' {
                filled.push((r, c));
            }
            self.board[r][c] = ch;
        }
        if let Some(used) = used {
            if used.len() != filled.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "used has {} tiles but the play covers {} empty squares",
                    used.len(),
                    filled.len(),
                )));
            }
            for (&(r, c), &tile) in filled.iter().zip(used.iter()) {
                self.blanks[r][c] = tile == '?';
            }
        }
        self.first = false;
        Ok(filled)
    }

    /// Undo a `place_word`, given the cells it returned. `was_first` restores
    /// the opening-move flag (a search that unwinds the first play must put it
    /// back). Cheaper than cloning the board at every node of a search.
    #[pyo3(signature = (cells, was_first=false))]
    fn unplace_word(&mut self, cells: Vec<(usize, usize)>, was_first: bool) -> PyResult<()> {
        for &(r, c) in &cells {
            if !in_bounds(r, c) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "cell out of bounds",
                ));
            }
            self.board[r][c] = '-';
            self.blanks[r][c] = false;
        }
        self.first = was_first;
        Ok(())
    }

    fn get_row_patterns(&self, row_idx: usize) -> Vec<(usize, usize)> {
        let row = &self.board.as_slice()[row_idx];
        let n = row.len();
        let empty = |i: usize| row.get(i).copied() == Some('-');
        let valid_start = |i: usize| i == 0 || empty(i - 1);
        let valid_end = |i: usize| i == n - 1 || empty(i + 1);

        let mut patterns = Vec::new();

        // "Crossing" patterns: spans within this row that mix empty and
        // filled cells -- a new word overlaps an existing tile in this row.
        for start in 0..n {
            for end in (start + 1)..n {
                if !valid_start(start) || !valid_end(end) {
                    continue;
                }
                let slice = &row[start..=end];
                if slice.contains(&'-') && slice.iter().any(|&c| c != '-') {
                    patterns.push((start, end));
                }
            }
        }

        // "Parallel" patterns: fully-empty spans in this row that connect
        // to the board only via a filled neighbour directly above/below --
        // e.g. a new word running alongside an existing one, one row over,
        // touching it only through the cross-words it forms. `get_all_patterns`
        // fed only the "crossing" patterns above would never find these.
        let has_adjacent_tile = |i: usize| {
            (row_idx > 0 && self.board[row_idx - 1][i] != '-')
                || (row_idx < n - 1 && self.board[row_idx + 1][i] != '-')
        };
        let mut run_start = 0;
        while run_start < n {
            if !empty(run_start) {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start;
            while run_end + 1 < n && empty(run_end + 1) {
                run_end += 1;
            }
            for start in run_start..=run_end {
                for end in start..=run_end {
                    // Same boundary rule as "crossing" patterns: a sub-span
                    // touching a same-row tile right at its start/end would
                    // silently glue that tile onto the new word (e.g. "nitowa"
                    // ending right before an existing 'c' becomes "nitowac"
                    // on the board) -- that case belongs to a "crossing"
                    // pattern that includes the tile, not this one.
                    if !valid_start(start) || !valid_end(end) {
                        continue;
                    }
                    if (start..=end).any(has_adjacent_tile) {
                        patterns.push((start, end));
                    }
                }
            }
            run_start = run_end + 1;
        }

        patterns
    }

    fn get_col_patterns(&self, col_idx: usize) -> Vec<(usize, usize)> {
        let n = BOARD_SIZE;
        let empty = |i: usize| self.board[i][col_idx] == '-';
        let valid_start = |i: usize| i == 0 || empty(i - 1);
        let valid_end = |i: usize| i == n - 1 || empty(i + 1);

        let mut patterns = Vec::new();

        // "Crossing" patterns: spans within this column that mix empty and
        // filled cells -- a new word overlaps an existing tile in this column.
        for start in 0..n {
            for end in (start + 1)..n {
                if !valid_start(start) || !valid_end(end) {
                    continue;
                }
                let mut has_empty = false;
                let mut has_tile = false;
                for i in start..=end {
                    if self.board[i][col_idx] == '-' {
                        has_empty = true;
                    } else {
                        has_tile = true;
                    }
                    if has_empty && has_tile {
                        break;
                    }
                }
                if has_empty && has_tile {
                    patterns.push((start, end));
                }
            }
        }

        // "Parallel" patterns: fully-empty spans in this column that connect
        // to the board only via a filled neighbour directly left/right --
        // see get_row_patterns for the full rationale.
        let has_adjacent_tile = |i: usize| {
            (col_idx > 0 && self.board[i][col_idx - 1] != '-')
                || (col_idx < n - 1 && self.board[i][col_idx + 1] != '-')
        };
        let mut run_start = 0;
        while run_start < n {
            if !empty(run_start) {
                run_start += 1;
                continue;
            }
            let mut run_end = run_start;
            while run_end + 1 < n && empty(run_end + 1) {
                run_end += 1;
            }
            for start in run_start..=run_end {
                for end in start..=run_end {
                    // Same boundary rule as "crossing" patterns -- see the
                    // matching comment in get_row_patterns.
                    if !valid_start(start) || !valid_end(end) {
                        continue;
                    }
                    if (start..=end).any(has_adjacent_tile) {
                        patterns.push((start, end));
                    }
                }
            }
            run_start = run_end + 1;
        }

        patterns
    }

    fn get_all_patterns(&self) -> Vec<(usize, usize, usize, bool)> {
        let mut patterns = Vec::new();
        for i in 0..BOARD_SIZE {
            for (start, end) in self.get_row_patterns(i) {
                patterns.push((i, start, end, true));
            }
            for (start, end) in self.get_col_patterns(i) {
                patterns.push((i, start, end, false));
            }
        }
        patterns
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
        self.best_word_from_pattern_inner(&dawg.inner, row, start, end, horizontal, letters)
            .0
    }

    #[pyo3(signature = (dawg, letters, n, parallel=true))]
    fn get_best_words(
        &self,
        dawg: &DawgPy,
        letters: &str,
        n: usize,
        parallel: bool,
    ) -> Vec<BestWord> {
        if self.first {
            self.best_opening_words(dawg, letters, n)
        } else if let Some(gaddag) = &dawg.gaddag {
            // Fast path: GADDAG anchor generation (equivalent to the legacy
            // pattern search, verified by the `gen-verify` CLI command).
            self.gaddag_best_words(&dawg.inner, gaddag, letters, n, parallel)
        } else {
            self.best_words_from_patterns(dawg, letters, n, parallel)
        }
    }

    /// Best-scoring first moves: the centre square must be covered, so
    /// every word/offset combination that does so is a candidate.
    fn best_opening_words(&self, dawg: &DawgPy, letters: &str, n: usize) -> Vec<BestWord> {
        let mut candidates: Vec<BestWord> = Vec::new();
        for word in dawg.search("*", letters) {
            for offset in 0..CENTER {
                if offset >= word.len() {
                    break;
                }
                let col = CENTER - offset;
                let score = self
                    .calculate_word_points(&word, CENTER, col, true, letters)
                    .unwrap_or(0);
                if score == 0 {
                    continue;
                }
                let used = self.hand_tiles_for_word(&word, CENTER, col, true, letters);
                candidates.push((word.clone(), score, (CENTER, col, true), used));
            }
        }
        candidates.sort_by_key(|&(_, score, ..)| std::cmp::Reverse(score));
        candidates.truncate(n);
        candidates
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
        let patterns = self.get_all_patterns();
        let compute = |(row, start, end, horizontal): (usize, usize, usize, bool)| {
            let (ar, ac) = if horizontal {
                (row, start)
            } else {
                (start, row)
            };
            let (word, score) =
                self.best_word_from_pattern_inner(&dawg.inner, ar, ac, end, horizontal, letters);
            if word.is_empty() {
                None
            } else {
                Some((word, score, ar, ac, horizontal))
            }
        };
        let mut best: Vec<(String, u32, usize, usize, bool)> = if parallel {
            patterns.into_par_iter().filter_map(compute).collect()
        } else {
            patterns.into_iter().filter_map(compute).collect()
        };
        best.sort_by_key(|&(_, score, ..)| std::cmp::Reverse(score));
        best.truncate(n);

        best.into_iter()
            .map(|(word, score, ar, ac, horiz)| {
                let used = self.hand_tiles_for_word(&word, ar, ac, horiz, letters);
                (word, score, (ar, ac, horiz), used)
            })
            .collect()
    }

    #[pyo3(signature = (dawg, letters, parallel=true))]
    fn get_best_word(&self, dawg: &DawgPy, letters: &str, parallel: bool) -> BestWord {
        self.get_best_words(dawg, letters, 1, parallel)
            .into_iter()
            .next()
            .unwrap_or_else(|| (String::new(), 0, (0, 0, true), Vec::new()))
    }

    /// Every legal play, unsorted and untruncated.
    ///
    /// `get_best_words` keeps the top `n` *by raw score*, which is the wrong
    /// filter for anything that ranks by equity: a tile-dumping play, a
    /// blocking play, or the play that goes out first in an endgame can all be
    /// worth more than their score suggests and fall outside the cut. Callers
    /// that rank candidates themselves want this instead, and skip the sort.
    /// Requires a GADDAG: the legacy pattern search only ever yields the best
    /// word per board span, so it cannot enumerate. Raises rather than
    /// returning an empty list, which a caller would read as "no legal plays".
    #[pyo3(signature = (dawg, letters, parallel=true))]
    fn all_moves(&self, dawg: &DawgPy, letters: &str, parallel: bool) -> PyResult<Vec<BestWord>> {
        if self.first {
            return Ok(self.best_opening_words(dawg, letters, usize::MAX));
        }
        let Some(gaddag) = &dawg.gaddag else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "all_moves needs a GADDAG; this Dawg was loaded without one",
            ));
        };
        Ok(self
            .gaddag_generate(&dawg.inner, gaddag, letters, parallel, true)
            .into_iter()
            .map(|(r, c, horizontal, word, score)| {
                let used = self.hand_tiles_for_word(&word, r, c, horizontal, letters);
                (word, score, (r, c, horizontal), used)
            })
            .collect())
    }
}

// Pure Rust methods — no PyO3 overhead, safe to call from rayon threads.
impl Board {
    /// Empty board with a full bag, shuffled once from `seed`.
    fn with_seed(seed: u64) -> Board {
        let mut rng = seed;
        let mut tile_bag = fresh_tile_bag();
        shuffle(&mut tile_bag, &mut rng);
        Board {
            board: [['-'; BOARD_SIZE]; BOARD_SIZE],
            blanks: [[false; BOARD_SIZE]; BOARD_SIZE],
            tile_bag,
            first: true,
            rng,
        }
    }

    /// Face value of the tile at (r, c) as it actually scores: 0 for a blank,
    /// whatever letter it is standing in for. Every read of an *already
    /// placed* tile's value must go through here — reading `letter_points` off
    /// the grid character would score a played blank at face value for the
    /// rest of the game.
    #[inline]
    fn tile_points(&self, r: usize, c: usize) -> u32 {
        if self.blanks[r][c] {
            0
        } else {
            letter_points(self.board[r][c])
        }
    }

    /// Summed value of the contiguous run of tiles immediately before (r, c)
    /// along a row (`horizontal`) or column. `None` when no run is there.
    fn run_points_before(&self, r: usize, c: usize, horizontal: bool) -> Option<u32> {
        let mut total = 0u32;
        let mut any = false;
        let mut k = 1usize;
        loop {
            let cell = if horizontal {
                c.checked_sub(k).map(|cc| (r, cc))
            } else {
                r.checked_sub(k).map(|rr| (rr, c))
            };
            let (rr, cc) = match cell {
                Some(rc) => rc,
                None => break,
            };
            if self.board[rr][cc] == '-' {
                break;
            }
            total += self.tile_points(rr, cc);
            any = true;
            k += 1;
        }
        if any {
            Some(total)
        } else {
            None
        }
    }

    /// Summed value of the contiguous run of tiles immediately after (r, c).
    fn run_points_after(&self, r: usize, c: usize, horizontal: bool) -> Option<u32> {
        let mut total = 0u32;
        let mut any = false;
        let mut k = 1usize;
        loop {
            let (rr, cc) = if horizontal { (r, c + k) } else { (r + k, c) };
            if rr >= BOARD_SIZE || cc >= BOARD_SIZE || self.board[rr][cc] == '-' {
                break;
            }
            total += self.tile_points(rr, cc);
            any = true;
            k += 1;
        }
        if any {
            Some(total)
        } else {
            None
        }
    }

    /// Return the cross-word formed at `(row, col)` when placing `ch` in the
    /// direction perpendicular to `horizontal`. Returns an empty string if no
    /// neighbour tiles exist.
    fn cross_word(&self, row: usize, col: usize, horizontal: bool, ch: char) -> String {
        let mut word = String::new();
        if !horizontal {
            // placing vertically → check horizontal neighbours
            let mut x = 1;
            while x <= col && self.board[row][col - x] != '-' {
                x += 1;
            }
            let start = col - x + 1;
            let mut x = 1;
            while col + x < BOARD_SIZE && self.board[row][col + x] != '-' {
                x += 1;
            }
            let end = col + x - 1;
            if end > start || (end == start && (start < col || col < end)) {
                for ci in start..=end {
                    word.push(if ci == col { ch } else { self.board[row][ci] });
                }
            }
        } else {
            // placing horizontally → check vertical neighbours
            let mut y = 1;
            while y <= row && self.board[row - y][col] != '-' {
                y += 1;
            }
            let start = row - y + 1;
            let mut y = 1;
            while row + y < BOARD_SIZE && self.board[row + y][col] != '-' {
                y += 1;
            }
            let end = row + y - 1;
            if end > start || (end == start && (start < row || row < end)) {
                for ri in start..=end {
                    word.push(if ri == row { ch } else { self.board[ri][col] });
                }
            }
        }
        word
    }

    /// Value of the existing perpendicular neighbours at `(row, col)` — the
    /// part of a cross-word's score that does not depend on which tile is
    /// about to be placed there. `None` if no cross-word forms (no adjacent
    /// tiles). Blanks among the neighbours contribute 0, which is why this
    /// walks the board rather than summing `letter_points` over `cross_word`'s
    /// string: that string has already lost the blank/real distinction.
    fn cross_neighbor_points(&self, row: usize, col: usize, horizontal: bool) -> Option<u32> {
        // The main word runs along `horizontal`, so its cross-words run along
        // the other axis.
        let before = self.run_points_before(row, col, !horizontal);
        let after = self.run_points_after(row, col, !horizontal);
        match (before, after) {
            (None, None) => None,
            (b, a) => Some(b.unwrap_or(0) + a.unwrap_or(0)),
        }
    }

    fn check_word_placement_inner(
        &self,
        dawg: &Dawg,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> bool {
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if !in_bounds(r, c) {
                return false;
            }
            if self.board[r][c] != '-' {
                continue;
            }

            let adjacent = self.cross_word(r, c, horizontal, ch);
            if adjacent.len() > 1 && !dawg.contains(&adjacent) {
                return false;
            }
        }
        true
    }

    /// Letters actually drawn from `letters` to place `word` at (row, col):
    /// one entry per still-empty board cell, in placement order. Real
    /// tiles are used first; once the hand's supply of a letter is
    /// exhausted, a blank (`'?'`) stands in for it instead — matching how
    /// `calculate_word_points` scores the same placement, so the two never
    /// disagree about which tiles a move actually consumes.
    fn hand_tiles_for_word(
        &self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        letters: &str,
    ) -> Vec<char> {
        let mut hand_freq = real_letter_counts(letters);
        word.chars()
            .enumerate()
            .filter_map(|(i, ch)| {
                let (r, c) = word_cell(row, col, horizontal, i);
                if self.board[r][c] != '-' {
                    return None;
                }
                match hand_freq.get_mut(&ch) {
                    Some(count) if *count > 0 => {
                        *count -= 1;
                        Some(ch)
                    }
                    _ => Some('?'),
                }
            })
            .collect()
    }

    fn best_word_from_pattern_inner(
        &self,
        dawg: &Dawg,
        row: usize,
        start: usize,
        end: usize,
        horizontal: bool,
        letters: &str,
    ) -> (String, u32) {
        let mut pattern = String::new();
        if horizontal {
            for i in start..=end {
                pattern.push(self.board[row][i]);
            }
        } else {
            for i in row..=end {
                pattern.push(self.board[i][start]);
            }
        }

        // Skip the DAWG search if the hand can't fill all empty slots.
        let mandatory = pattern.chars().filter(|&c| c == '-').count();
        let hand_size = letters.chars().count();
        if hand_size < mandatory {
            return (String::new(), 0);
        }

        let mut best_word = String::new();
        let mut best_score = 0u32;
        for word in dawg.search_inner(&pattern, letters) {
            if self.check_word_placement_inner(dawg, &word, row, start, horizontal) {
                let score = self
                    .calculate_word_points(&word, row, start, horizontal, letters)
                    .unwrap_or(0);
                if score > best_score {
                    best_score = score;
                    best_word = word;
                }
            }
        }
        (best_word, best_score)
    }
}

// ---------------------------------------------------------------------------
// GADDAG move generation (Gordon 1994): anchor + precomputed cross-checks +
// bidirectional extension. The GADDAG is stored in the same flat format as the
// DAWG (a `Dawg` loaded from gaddag.bin), so the same node accessors apply; the
// only difference is that its edges spell reversed-prefix `SEP` forward-suffix.
// ---------------------------------------------------------------------------

/// Cross-check bitset for placing a tile on an empty square whose perpendicular
/// neighbours spell `prefix` (above/left, in reading order) and `suffix`
/// (below/right): bit `letter_index(L)` is set iff `prefix + L + suffix` is a
/// word. Uses the DAWG (`dawg`), since cross-words are ordinary dictionary
/// words. A square with no perpendicular neighbours never calls this — it keeps
/// `CROSS_ANY`.
fn cross_bits(dawg: &Dawg, prefix: &[char], suffix: &[char]) -> u32 {
    let mut node = dawg.root;
    for &ch in prefix {
        match dawg.find_child(node, ch) {
            Some(n) => node = n,
            None => return 0,
        }
    }
    let mut bits = 0u32;
    let cnt = dawg.node_children_count(node);
    for i in 0..cnt {
        let (l, child) = dawg.node_child(node, i);
        let bit = letter_bit(l);
        if bit == 0 {
            continue;
        }
        let mut n = child;
        let mut ok = true;
        for &ch in suffix {
            match dawg.find_child(n, ch) {
                Some(x) => n = x,
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if ok && dawg.node_is_terminal(n) {
            bits |= bit;
        }
    }
    bits
}

/// Immutable per-(anchor, orientation) context for one GADDAG traversal. The
/// mutable search state (rack `freq`, the accumulating word halves, and the
/// output list) is threaded through the recursion as `&mut` arguments so a
/// `GenCtx` can be shared read-only and each rayon thread keeps its own state.
struct GenCtx<'a> {
    board: &'a Board,
    gaddag: &'a Dawg,
    cross: &'a [[u32; BOARD_SIZE]; BOARD_SIZE],
    /// Sum of face values of the perpendicular neighbours at each empty square
    /// (0 where none) — the cross-word's existing contribution, tile-independent.
    cross_score: &'a [[u32; BOARD_SIZE]; BOARD_SIZE],
    /// Whether a cross-word forms at each empty square (has a perpendicular tile).
    has_cross: &'a [[bool; BOARD_SIZE]; BOARD_SIZE],
    /// Count of each real (non-blank) rack letter, indexed by `letter_index`.
    /// Precomputed once per move so `score_word`'s word-order real-vs-blank
    /// allocation is a cheap array copy, not a per-word map allocation.
    rack_counts: [u8; ALPHABET_SIZE],
    ar: usize,
    ac: usize,
    horizontal: bool,
}

/// A generated, scored placement: (row, col, horizontal, word, score) of the
/// leftmost/topmost cell. Scored during generation via `GenCtx::score_word`,
/// which reproduces `calculate_word_points` exactly using precomputed
/// cross-scores instead of re-walking the board.
type GenMove = (usize, usize, bool, String, u32);

impl<'a> GenCtx<'a> {
    /// Board cell `off` squares along the play direction from the anchor, or
    /// `None` if it falls off the board.
    #[inline]
    fn cell(&self, off: isize) -> Option<(usize, usize)> {
        let (r, c) = if self.horizontal {
            (self.ar as isize, self.ac as isize + off)
        } else {
            (self.ar as isize + off, self.ac as isize)
        };
        if r >= 0 && r < BOARD_SIZE as isize && c >= 0 && c < BOARD_SIZE as isize {
            Some((r as usize, c as usize))
        } else {
            None
        }
    }

    /// True if the cell at `off` is empty or off the board — i.e. a clean word
    /// boundary, with no existing tile that the word would have to include.
    #[inline]
    fn is_free(&self, off: isize) -> bool {
        match self.cell(off) {
            Some((r, c)) => self.board.board[r][c] == '-',
            None => true,
        }
    }

    /// Fill the square at `off`: follow an existing tile's arc, or try every
    /// rack letter the current GADDAG node and the square's cross-check allow.
    fn gen(
        &self,
        node: u32,
        off: isize,
        budget: usize,
        freq: &mut LetterFreq,
        left: &mut Vec<char>,
        right: &mut Vec<char>,
        out: &mut Vec<GenMove>,
    ) {
        let (r, c) = match self.cell(off) {
            Some(rc) => rc,
            None => return,
        };
        let bch = self.board.board[r][c];
        if bch != '-' {
            // Existing tile: forced letter, no rack cost, no cross-check, and it
            // does not consume the left budget (only new tiles do).
            if let Some(next) = self.gaddag.find_child(node, bch) {
                self.go_on(off, bch, next, budget, freq, left, right, out);
            }
            return;
        }
        // A new tile strictly left of the anchor spends one unit of left budget.
        // When it is exhausted, words extending further left belong to a
        // different anchor (Appel–Jacobson dedup), so stop.
        if off < 0 && budget == 0 {
            return;
        }
        let next_budget = if off < 0 { budget - 1 } else { budget };
        // Empty square: try each arc letter permitted by the cross-check. Use a
        // real tile if one is left, else a blank (greedy real-first is optimal
        // for feasibility — reals only fit their own letter, blanks fit any).
        let allowed = self.cross[r][c];
        let qi = '?' as usize;
        // Iterate the node's decoded children directly: `bit` is the precomputed
        // cross-check mask (0 for SEP), so the hot loop avoids re-decoding the
        // edge codepoint and recomputing `letter_bit` on every arc.
        for entry in self.gaddag.node_children(node) {
            let l = entry.ch;
            if l == SEP {
                continue; // direction switch is handled in go_on, not here
            }
            if allowed & entry.bit == 0 {
                continue;
            }
            let li = l as usize;
            let used_real = li < FREQ_SIZE && freq[li] > 0;
            if used_real {
                freq[li] -= 1;
            } else if freq[qi] > 0 {
                freq[qi] -= 1;
            } else {
                continue; // no tile can play this letter
            }
            self.go_on(off, l, entry.id, next_budget, freq, left, right, out);
            if used_real {
                freq[li] += 1;
            } else {
                freq[qi] += 1;
            }
        }
    }

    /// Having committed letter `l` at `off` (reaching `new_node`), record any
    /// completed word and recurse: keep extending left, switch direction across
    /// `SEP`, or keep extending right.
    fn go_on(
        &self,
        off: isize,
        l: char,
        new_node: u32,
        budget: usize,
        freq: &mut LetterFreq,
        left: &mut Vec<char>,
        right: &mut Vec<char>,
        out: &mut Vec<GenMove>,
    ) {
        if off <= 0 {
            // Left phase: letters accrue anchor-first, i.e. reversed board order.
            left.push(l);
            // Switch to the right half only if the word's left end is clean.
            if self.is_free(off - 1) {
                if let Some(sep_node) = self.gaddag.find_child(new_node, SEP) {
                    if self.gaddag.node_is_terminal(sep_node) && self.is_free(1) {
                        self.record(left, right, out);
                    }
                    // Right phase is unbounded; budget only limits the left half.
                    self.gen(sep_node, 1, budget, freq, left, right, out);
                }
            }
            // Keep extending further left (onto an empty or existing-tile square).
            self.gen(new_node, off - 1, budget, freq, left, right, out);
            left.pop();
        } else {
            // Right phase: letters accrue in board order.
            right.push(l);
            if self.gaddag.node_is_terminal(new_node) && self.is_free(off + 1) {
                self.record(left, right, out);
            }
            self.gen(new_node, off + 1, budget, freq, left, right, out);
            right.pop();
        }
    }

    /// Emit the word currently spelled by `left` (reversed) followed by `right`,
    /// anchored at its leftmost/topmost cell, scored in place.
    fn record(&self, left: &[char], right: &[char], out: &mut Vec<GenMove>) {
        let len = left.len() + right.len();
        if len < 2 {
            return; // a legal play spells a word of at least two letters
        }
        let mut word = String::with_capacity(len * 2);
        for &ch in left.iter().rev() {
            word.push(ch);
        }
        for &ch in right {
            word.push(ch);
        }
        let leftmost_off = -(left.len() as isize - 1);
        // The leftmost cell was placed, so it is always in bounds.
        if let Some((r, c)) = self.cell(leftmost_off) {
            let score = self.score_word(r, c, &word);
            out.push((r, c, self.horizontal, word, score));
        }
    }

    /// Score the word placed at (r0, c0) along the play direction. Reproduces
    /// `Board::calculate_word_points` exactly — same word-order real-vs-blank
    /// allocation, same per-tile letter/word multipliers, same independent
    /// cross-word scoring and +50 bingo — but reads the perpendicular
    /// contribution from `cross_score`/`has_cross` instead of re-walking the
    /// board. `calculate_word_points` stays the authority; `gen-verify` asserts
    /// they agree.
    fn score_word(&self, r0: usize, c0: usize, word: &str) -> u32 {
        let mut main_total = 0u32;
        let mut main_word_mul = 1u32;
        let mut cross_total = 0u32;
        let mut tiles_from_hand = 0usize;
        // A fresh copy of the real-letter counts to spend as we walk the word --
        // a 32-byte stack array, not a per-word HashMap allocation.
        let mut hand = self.rack_counts;
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(r0, c0, self.horizontal, i);
            let (lm, wm) = quadrant_bonus(r, c);
            let (lm, wm) = (lm as u32, wm as u32);
            if self.board.board[r][c] == '-' {
                // New tile: a real copy if the hand still has one, else a blank (0).
                let v = match letter_index(ch) {
                    Some(idx) if hand[idx] > 0 => {
                        hand[idx] -= 1;
                        letter_points(ch)
                    }
                    _ => 0,
                };
                tiles_from_hand += 1;
                main_total += v * lm;
                main_word_mul *= wm;
                if self.has_cross[r][c] {
                    cross_total += (v * lm + self.cross_score[r][c]) * wm;
                }
            } else {
                main_total += self.board.tile_points(r, c);
            }
        }
        main_total * main_word_mul + cross_total + if tiles_from_hand == RACK_SIZE { 50 } else { 0 }
    }
}

impl Board {
    /// Anchors — empty squares orthogonally adjacent to at least one tile, which
    /// every non-opening play must cover. Returns both the boolean grid (used by
    /// the left-limit) and the list to iterate.
    fn anchor_grid(&self) -> ([[bool; BOARD_SIZE]; BOARD_SIZE], Vec<(usize, usize)>) {
        let mut grid = [[false; BOARD_SIZE]; BOARD_SIZE];
        let mut anchors = Vec::new();
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.board[r][c] != '-' {
                    continue;
                }
                let adjacent = (r > 0 && self.board[r - 1][c] != '-')
                    || (r + 1 < BOARD_SIZE && self.board[r + 1][c] != '-')
                    || (c > 0 && self.board[r][c - 1] != '-')
                    || (c + 1 < BOARD_SIZE && self.board[r][c + 1] != '-');
                if adjacent {
                    grid[r][c] = true;
                    anchors.push((r, c));
                }
            }
        }
        (grid, anchors)
    }

    /// Appel–Jacobson left-limit: how many new tiles a play from this anchor may
    /// place before the anchor square (leftward for horizontal, upward for
    /// vertical). Counts empty squares until the first tile or the first other
    /// anchor — words extending past those belong to a different anchor, so this
    /// makes each play reachable from exactly one anchor (no duplicates).
    fn left_budget(
        &self,
        is_anchor: &[[bool; BOARD_SIZE]; BOARD_SIZE],
        r: usize,
        c: usize,
        horizontal: bool,
    ) -> usize {
        let mut budget = 0usize;
        let mut k = 1usize;
        loop {
            let cell = if horizontal {
                c.checked_sub(k).map(|cc| (r, cc))
            } else {
                r.checked_sub(k).map(|rr| (rr, c))
            };
            let (rr, cc) = match cell {
                Some(rc) => rc,
                None => break,
            };
            if self.board[rr][cc] != '-' || is_anchor[rr][cc] {
                break;
            }
            budget += 1;
            k += 1;
        }
        budget
    }

    /// Precompute, for every empty square and plays in the given direction
    /// (`horizontal` → vertical cross-words): the cross-check bitset, the sum of
    /// the perpendicular neighbours' face values (`cross_score`), and whether a
    /// cross-word forms (`has_cross`). Squares with no perpendicular neighbour
    /// keep `CROSS_ANY` / 0 / false. One neighbour walk feeds all three.
    #[allow(clippy::type_complexity)]
    fn compute_cross_data(
        &self,
        dawg: &Dawg,
        horizontal: bool,
    ) -> (
        [[u32; BOARD_SIZE]; BOARD_SIZE],
        [[u32; BOARD_SIZE]; BOARD_SIZE],
        [[bool; BOARD_SIZE]; BOARD_SIZE],
    ) {
        let mut checks = [[CROSS_ANY; BOARD_SIZE]; BOARD_SIZE];
        let mut scores = [[0u32; BOARD_SIZE]; BOARD_SIZE];
        let mut has = [[false; BOARD_SIZE]; BOARD_SIZE];
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.board[r][c] != '-' {
                    continue;
                }
                // Perpendicular to the play direction.
                let (prefix, suffix) = (
                    self.tiles_before(r, c, !horizontal),
                    self.tiles_after(r, c, !horizontal),
                );
                if prefix.is_empty() && suffix.is_empty() {
                    continue;
                }
                has[r][c] = true;
                // Value walked off the board (blank-aware), not summed over
                // `prefix`/`suffix` — those are letters, and a blank standing
                // in for one still scores 0.
                scores[r][c] = self.run_points_before(r, c, !horizontal).unwrap_or(0)
                    + self.run_points_after(r, c, !horizontal).unwrap_or(0);
                checks[r][c] = cross_bits(dawg, &prefix, &suffix);
            }
        }
        (checks, scores, has)
    }

    /// Contiguous run of tiles immediately before (r, c) along a row
    /// (`horizontal`) or column, in reading order (left→right / top→down).
    fn tiles_before(&self, r: usize, c: usize, horizontal: bool) -> Vec<char> {
        let mut run = Vec::new();
        let mut k = 1usize;
        loop {
            let cell = if horizontal {
                c.checked_sub(k).map(|cc| (r, cc))
            } else {
                r.checked_sub(k).map(|rr| (rr, c))
            };
            let (rr, cc) = match cell {
                Some(rc) => rc,
                None => break,
            };
            if self.board[rr][cc] == '-' {
                break;
            }
            run.push(self.board[rr][cc]);
            k += 1;
        }
        run.reverse(); // collected nearest-first; reading order wants farthest-first
        run
    }

    /// Contiguous run of tiles immediately after (r, c), in reading order.
    fn tiles_after(&self, r: usize, c: usize, horizontal: bool) -> Vec<char> {
        let mut run = Vec::new();
        let mut k = 1usize;
        loop {
            let (rr, cc) = if horizontal { (r, c + k) } else { (r + k, c) };
            if rr >= BOARD_SIZE || cc >= BOARD_SIZE || self.board[rr][cc] == '-' {
                break;
            }
            run.push(self.board[rr][cc]);
            k += 1;
        }
        run
    }

    /// Generate every legal play via anchor + cross-check + bidirectional
    /// extension, each scored in place. With `use_limit`, the Appel–Jacobson
    /// left-limit makes every play come from exactly one anchor (no duplicates);
    /// without it, left extension is unbounded and duplicates must be deduped by
    /// the caller (used only by `gen-verify` to prove the limit drops only dups).
    fn gaddag_generate(
        &self,
        dawg: &Dawg,
        gaddag: &Dawg,
        letters: &str,
        parallel: bool,
        use_limit: bool,
    ) -> Vec<GenMove> {
        let (checks_h, scores_h, has_h) = self.compute_cross_data(dawg, true);
        let (checks_v, scores_v, has_v) = self.compute_cross_data(dawg, false);
        let (is_anchor, anchors) = self.anchor_grid();

        // Real (non-blank) rack letter counts, computed once and shared read-only
        // by every anchor's GenCtx so `score_word` needn't rebuild a map per word.
        let mut rack_counts = [0u8; ALPHABET_SIZE];
        for ch in letters.chars() {
            if let Some(i) = letter_index(ch) {
                rack_counts[i] += 1;
            }
        }

        let gen_one = |(ar, ac): (usize, usize)| -> Vec<GenMove> {
            let mut out = Vec::new();
            let (mut freq, _) = build_freq(letters);
            let mut left = Vec::new();
            let mut right = Vec::new();
            let dirs = [
                (true, &checks_h, &scores_h, &has_h),
                (false, &checks_v, &scores_v, &has_v),
            ];
            for (horizontal, cross, cross_score, has_cross) in dirs {
                let budget = if use_limit {
                    self.left_budget(&is_anchor, ar, ac, horizontal)
                } else {
                    BOARD_SIZE
                };
                let ctx = GenCtx {
                    board: self,
                    gaddag,
                    cross,
                    cross_score,
                    has_cross,
                    rack_counts,
                    ar,
                    ac,
                    horizontal,
                };
                ctx.gen(gaddag.root, 0, budget, &mut freq, &mut left, &mut right, &mut out);
                left.clear();
                right.clear();
            }
            out
        };

        if parallel {
            // Run on the engine's own (default 8-thread) pool, not rayon's
            // all-cores global pool — see `gen_pool`.
            gen_pool().install(|| anchors.par_iter().flat_map_iter(|&a| gen_one(a)).collect())
        } else {
            anchors.iter().flat_map(|&a| gen_one(a)).collect()
        }
    }

    /// GADDAG-based replacement for `best_words_from_patterns`: generate every
    /// legal play (already scored, no duplicates), keep the top `n`. `dawg`
    /// supplies cross-word checks; `gaddag` drives the traversal.
    fn gaddag_best_words(
        &self,
        dawg: &Dawg,
        gaddag: &Dawg,
        letters: &str,
        n: usize,
        parallel: bool,
    ) -> Vec<BestWord> {
        let mut best = self.gaddag_generate(dawg, gaddag, letters, parallel, true);
        best.sort_by_key(|m| std::cmp::Reverse(m.4));
        best.truncate(n);
        best.into_iter()
            .map(|(r, c, horizontal, word, score)| {
                let used = self.hand_tiles_for_word(&word, r, c, horizontal, letters);
                (word, score, (r, c, horizontal), used)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// DAWG construction
// ---------------------------------------------------------------------------

fn node_key(arena: &Arena, id: u32) -> String {
    let node = arena.node(id);
    let mut out = String::new();
    out.push(if node.is_terminal { '1' } else { '0' });
    let mut pairs: Vec<(char, u32)> = node.children.iter().map(|(&c, &id)| (c, id)).collect();
    pairs.sort_unstable_by_key(|&(c, _)| c);
    for (c, child_id) in pairs {
        out.push('_');
        out.push(c);
        out.push('_');
        out.push_str(&child_id.to_string());
    }
    out
}

fn prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

fn minimize(
    arena: &mut Arena,
    pref_len: usize,
    minimized: &mut HashMap<String, u32>,
    stack: &mut Vec<(u32, char, u32)>,
) {
    let pop_count = stack.len() - pref_len;
    for _ in 0..pop_count {
        let (parent_id, letter, child_id) = stack.pop().unwrap();
        let key = node_key(arena, child_id);
        let canonical = *minimized.entry(key).or_insert(child_id);
        arena.node_mut(parent_id).children.insert(letter, canonical);
    }
}

fn build_dawg(words: &[&str]) -> (Arena, u32, usize) {
    let mut arena = Arena::new();
    let root = arena.alloc();
    let mut minimized: HashMap<String, u32> = HashMap::new();
    minimized.insert(node_key(&arena, root), root);
    let mut stack: Vec<(u32, char, u32)> = Vec::new();
    let mut curr = root;
    let mut prev_word = "";
    let total = words.len();
    let report_every = (total / 40).max(1);

    for (i, &word) in words.iter().enumerate() {
        if i % report_every == 0 {
            eprint!("\r  building: {}/{}", i, total);
            let _ = io::stderr().flush();
        }
        let pref = prefix_len(prev_word, word);
        if !stack.is_empty() {
            minimize(&mut arena, pref, &mut minimized, &mut stack);
            curr = stack.last().map(|&(_, _, c)| c).unwrap_or(root);
        }
        for c in word.chars().skip(pref) {
            let child = arena.alloc();
            arena.node_mut(curr).children.insert(c, child);
            stack.push((curr, c, child));
            curr = child;
        }
        arena.node_mut(curr).is_terminal = true;
        prev_word = word;
    }
    minimize(&mut arena, 0, &mut minimized, &mut stack);
    eprintln!("\r  building: {}/{}", total, total);

    (arena, root, minimized.len())
}

/// Rebuild the arena keeping only nodes reachable from `root`, renumbered into a
/// dense 0..N range. Minimization rewires parents to canonical nodes but leaves the
/// merged-away duplicates sitting in `arena.nodes`; without this pass `serialize` would
/// dump them all (they were ~97% of the file). Node IDs double as file offsets, so the
/// reachable set must be renumbered, not merely filtered. Children are visited in
/// sorted-char order for deterministic, reproducible output.
fn compact(arena: &Arena, root: u32) -> (Arena, u32) {
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut order: Vec<u32> = Vec::new();
    let mut stack: Vec<u32> = vec![root];
    remap.insert(root, 0);
    order.push(root);

    while let Some(old) = stack.pop() {
        let mut children: Vec<(char, u32)> =
            arena.node(old).children.iter().map(|(&c, &id)| (c, id)).collect();
        children.sort_unstable_by_key(|&(c, _)| c);
        for (_, cid) in children {
            if let std::collections::hash_map::Entry::Vacant(e) = remap.entry(cid) {
                e.insert(order.len() as u32);
                order.push(cid);
                stack.push(cid);
            }
        }
    }

    let mut new_arena = Arena::new();
    for &old in &order {
        let nid = new_arena.alloc();
        let src = arena.node(old);
        let is_terminal = src.is_terminal;
        let children: Vec<(char, u32)> =
            src.children.iter().map(|(&c, &cid)| (c, remap[&cid])).collect();
        let node = new_arena.node_mut(nid);
        node.is_terminal = is_terminal;
        for (c, cid) in children {
            node.children.insert(c, cid);
        }
    }

    (new_arena, 0)
}

fn serialize(arena: &Arena, root: u32) -> Vec<u8> {
    let n = arena.nodes.len();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(&root.to_le_bytes());
    buf.extend_from_slice(&(n as u32).to_le_bytes());
    for id in 0..n as u32 {
        let node = arena.node(id);
        buf.push(node.is_terminal as u8);
        let mut children: Vec<(char, u32)> =
            node.children.iter().map(|(&c, &cid)| (c, cid)).collect();
        children.sort_unstable_by_key(|&(c, _)| c);
        buf.extend_from_slice(&(children.len() as u32).to_le_bytes());
        for (c, cid) in children {
            buf.extend_from_slice(&(c as u32).to_le_bytes());
            buf.extend_from_slice(&cid.to_le_bytes());
        }
    }
    buf
}

// ---------------------------------------------------------------------------
// CLI entry points
// ---------------------------------------------------------------------------

fn usage(prog: &str) {
    eprintln!(
        "Usage:\n  {prog} build         <words.txt>  <dawg.bin>\n  \
                    {prog} build-gaddag  <words.txt>  <gaddag.bin>\n  \
                    {prog} lookup        <dawg.bin>   <word>\n  \
                    {prog} bench         <dawg.bin>   <words.txt>\n  \
                    {prog} gen-verify    <dawg.bin>   <gaddag.bin>  [games]\n  \
                    {prog} gen-bench     <dawg.bin>   <gaddag.bin>  [games]"
    );
}

fn cmd_build(words_path: &str, dawg_path: &str) -> io::Result<()> {
    eprintln!("Reading '{words_path}'…");
    let text = fs::read_to_string(words_path)?;
    let mut words: Vec<&str> = text.split_whitespace().collect();
    words.sort_unstable();
    words.dedup();
    eprintln!("  {} unique words", words.len());

    let t0 = Instant::now();
    let (arena, root, node_count) = build_dawg(&words);
    let (arena, root) = compact(&arena, root);
    eprintln!(
        "  done in {:.2?}  │  {} canonical nodes ({} nodes after minimization + compaction)",
        t0.elapsed(),
        node_count,
        arena.nodes.len()
    );

    let data = serialize(&arena, root);
    {
        let file = fs::File::create(dawg_path)?;
        BufWriter::new(file).write_all(&data)?;
    }
    eprintln!(
        "  {:.3} MiB → '{dawg_path}'",
        data.len() as f64 / (1 << 20) as f64
    );
    Ok(())
}

/// Append every GADDAG entry for `word`: for each split point `a`, the
/// reversed prefix `c_a c_{a-1} … c_0`, then `SEP`, then the forward suffix
/// `c_{a+1} … c_{n-1}`. During search a word is placed starting at any of its
/// letters (the "anchor"): walk left through the reversed prefix, cross `SEP`,
/// then walk right through the suffix. The node after the last suffix letter
/// (or after `SEP` for an anchor on the final letter) is terminal.
fn gaddag_entries(word: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = word.chars().collect();
    let n = chars.len();
    for a in 0..n {
        let mut s = String::with_capacity(n + 1);
        for i in (0..=a).rev() {
            s.push(chars[i]);
        }
        s.push(SEP);
        for &c in &chars[a + 1..] {
            s.push(c);
        }
        out.push(s);
    }
}

fn cmd_build_gaddag(words_path: &str, gaddag_path: &str) -> io::Result<()> {
    eprintln!("Reading '{words_path}'…");
    let text = fs::read_to_string(words_path)?;
    let mut words: Vec<&str> = text.split_whitespace().collect();
    words.sort_unstable();
    words.dedup();
    eprintln!("  {} unique words", words.len());

    eprintln!("Expanding GADDAG entries…");
    let mut entries: Vec<String> = Vec::new();
    for w in &words {
        gaddag_entries(w, &mut entries);
    }
    eprintln!("  {} entries; sorting…", entries.len());
    entries.sort_unstable();
    entries.dedup();
    eprintln!("  {} unique entries", entries.len());

    let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
    let t0 = Instant::now();
    let (arena, root, node_count) = build_dawg(&refs);
    let (arena, root) = compact(&arena, root);
    eprintln!(
        "  done in {:.2?}  │  {} canonical nodes ({} after minimization + compaction)",
        t0.elapsed(),
        node_count,
        arena.nodes.len()
    );

    let data = serialize(&arena, root);
    {
        let file = fs::File::create(gaddag_path)?;
        BufWriter::new(file).write_all(&data)?;
    }
    eprintln!(
        "  {:.3} MiB → '{gaddag_path}'",
        data.len() as f64 / (1 << 20) as f64
    );
    Ok(())
}

fn cmd_lookup(dawg_path: &str, word: &str) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let t0 = Instant::now();
    let found = dawg.contains(word);
    let elapsed = t0.elapsed();
    if found {
        println!("✓  \"{word}\" found  ({elapsed:.2?})");
    } else {
        println!("✗  \"{word}\" not found  ({elapsed:.2?})");
    }
    Ok(())
}

fn cmd_bench(dawg_path: &str, words_path: &str) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let text = fs::read_to_string(words_path)?;
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();

    for w in &words[..(n / 10).max(1000).min(n)] {
        std::hint::black_box(dawg.contains(w));
    }

    const PASSES: u32 = 5;
    let mut found = 0usize;
    let t0 = Instant::now();
    for _ in 0..PASSES {
        for w in &words {
            if dawg.contains(w) {
                found += 1;
            }
        }
    }
    let elapsed = t0.elapsed();
    let total = n * PASSES as usize;
    let secs = elapsed.as_secs_f64();
    println!("\nResults ({PASSES} × {n} = {total} lookups):");
    println!("  total time  : {elapsed:.3?}");
    println!("  throughput  : {:.0} lookups/s", total as f64 / secs);
    println!("  per lookup  : {:.1} ns", secs * 1e9 / total as f64);
    println!(
        "  hits        : {found}/{total} ({:.1}%)",
        100.0 * found as f64 / total as f64
    );
    Ok(())
}

/// Advance a self-play position by one move using the current best play,
/// returning false when no move is possible (game over for our purposes).
/// Draws the rack up to a full hand, plays the engine's best word, and removes
/// the used tiles from `rack`.
fn selfplay_step(board: &mut Board, dpy: &DawgPy, rack: &mut String) -> bool {
    let drawn = board.give_letters(rack);
    rack.push_str(&drawn);
    if rack.is_empty() {
        return false;
    }
    let (word, _score, (r, c, horizontal), used) = board.get_best_word(dpy, rack, false);
    if word.is_empty() {
        return false;
    }
    let _ = board.place_word(&word, r, c, horizontal, Some(used.clone()));
    for ch in used {
        if let Some(pos) = rack.char_indices().find(|&(_, x)| x == ch).map(|(i, _)| i) {
            rack.remove(pos);
        }
    }
    true
}

/// Differential test: for many self-play positions, the GADDAG generator's best
/// score must equal the legacy pattern search's. Exits non-zero on any mismatch.
fn cmd_gen_verify(dawg_path: &str, gaddag_path: &str, games: usize) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let gaddag = Dawg::load(gaddag_path)?;
    let dpy = DawgPy {
        inner: dawg,
        gaddag: Some(gaddag),
    };
    let gad = dpy.gaddag.as_ref().unwrap();

    let mut positions = 0usize;
    let mut mismatches = 0usize; // gaddag best score != legacy best score
    let mut score_mismatches = 0usize; // fast in-place score != calculate_word_points
    let mut limit_mismatches = 0usize; // limited moves != unlimited deduped
    let mut dup_found = 0usize; // duplicate placement survived the left-limit
    for g in 0..games {
        let mut board = Board::new().expect("new board");
        let mut rack = String::new();
        for _ in 0..40 {
            // Draw first so the comparison sees the same rack the move uses.
            let drawn = board.give_letters(&rack);
            rack.push_str(&drawn);
            if rack.is_empty() {
                break;
            }
            if !board.first {
                positions += 1;
                // Production path: generate with the left-limit, already scored.
                let limited = board.gaddag_generate(&dpy.inner, gad, &rack, false, true);
                let gs = limited.iter().map(|m| m.4).max().unwrap_or(0);

                // Best-move parity vs the legacy pattern search.
                let legacy = board.best_words_from_patterns(&dpy, &rack, 1, false);
                let ls = legacy.first().map(|m| m.1).unwrap_or(0);
                if ls != gs {
                    mismatches += 1;
                    if mismatches <= 15 {
                        eprintln!(
                            "\nMISMATCH game {g}: legacy={ls} {:?}  gaddag={gs}\nrack: {rack}\n{}",
                            legacy.first().map(|m| (&m.0, m.1)),
                            board.__str__(),
                        );
                    }
                }

                // (a) The in-place fast score must equal calculate_word_points.
                for (r, c, h, word, score) in &limited {
                    let auth = board.calculate_word_points(word, *r, *c, *h, &rack).unwrap_or(0);
                    if *score != auth {
                        score_mismatches += 1;
                        if score_mismatches <= 10 {
                            eprintln!("SCORE MISMATCH {word} @({r},{c},{h}): fast={score} auth={auth}");
                        }
                    }
                }

                // (b) The left-limit must drop only duplicates: the limited move
                // set (which should already be duplicate-free) must equal the
                // unlimited set after deduping.
                let mut placements = std::collections::HashSet::new();
                for (r, c, h, w, _) in &limited {
                    if !placements.insert((w.clone(), *r, *c, *h)) {
                        dup_found += 1;
                    }
                }
                let unlimited = board.gaddag_generate(&dpy.inner, gad, &rack, false, false);
                let mut seen = std::collections::HashSet::new();
                let mut unl_scores: Vec<u32> = Vec::new();
                for (r, c, h, w, s) in &unlimited {
                    if seen.insert((w.clone(), *r, *c, *h)) {
                        unl_scores.push(*s);
                    }
                }
                let mut lim_scores: Vec<u32> = limited.iter().map(|m| m.4).collect();
                lim_scores.sort_unstable();
                unl_scores.sort_unstable();
                if lim_scores != unl_scores {
                    limit_mismatches += 1;
                    if limit_mismatches <= 10 {
                        eprintln!(
                            "LIMIT MISMATCH: limited={} unlimited-deduped={}\nrack: {rack}",
                            lim_scores.len(),
                            unl_scores.len()
                        );
                    }
                }
            }
            // Advance the position with the (gaddag) best move.
            let (word, _s, (r, c, h), used) = board.get_best_word(&dpy, &rack, false);
            if word.is_empty() {
                break;
            }
            let _ = board.place_word(&word, r, c, h, Some(used.clone()));
            for ch in used {
                if let Some(p) = rack.char_indices().find(|&(_, x)| x == ch).map(|(i, _)| i) {
                    rack.remove(p);
                }
            }
        }
    }
    println!(
        "\ngen-verify: {positions} positions | best-score mismatches={mismatches} | \
         fast-score mismatches={score_mismatches} | limit mismatches={limit_mismatches} | \
         duplicates past limit={dup_found}"
    );
    if mismatches + score_mismatches + limit_mismatches + dup_found > 0 {
        std::process::exit(2);
    }
    Ok(())
}

/// Benchmark: single-threaded move-generation time, legacy vs GADDAG, over
/// self-play positions. Reports average per position and the speedup.
fn cmd_gen_bench(dawg_path: &str, gaddag_path: &str, games: usize) -> io::Result<()> {
    let dawg = Dawg::load(dawg_path)?;
    let gaddag = Dawg::load(gaddag_path)?;
    let dpy = DawgPy {
        inner: dawg,
        gaddag: Some(gaddag),
    };
    let gad = dpy.gaddag.as_ref().unwrap();

    let mut positions = 0usize;
    let mut leg_secs = 0.0f64;
    let mut gad_secs = 0.0f64;
    for _ in 0..games {
        let mut board = Board::new().expect("new board");
        let mut rack = String::new();
        for _ in 0..40 {
            let drawn = board.give_letters(&rack);
            rack.push_str(&drawn);
            if rack.is_empty() {
                break;
            }
            if !board.first {
                let t = Instant::now();
                let leg = std::hint::black_box(board.best_words_from_patterns(&dpy, &rack, 1, false));
                leg_secs += t.elapsed().as_secs_f64();
                let t = Instant::now();
                let g = std::hint::black_box(board.gaddag_best_words(&dpy.inner, gad, &rack, 1, false));
                gad_secs += t.elapsed().as_secs_f64();
                let _ = (leg, g);
                positions += 1;
            }
            if !selfplay_step(&mut board, &dpy, &mut rack) {
                break;
            }
        }
    }

    let leg_ms = 1e3 * leg_secs / positions.max(1) as f64;
    let gad_ms = 1e3 * gad_secs / positions.max(1) as f64;
    println!("\ngen-bench ({positions} positions, single-threaded):");
    println!("  legacy pattern search : {leg_ms:.3} ms/position");
    println!("  gaddag generation     : {gad_ms:.3} ms/position");
    println!("  speedup               : {:.2}x", leg_secs / gad_secs.max(1e-12));
    Ok(())
}

pub fn main_cli() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 4 => cmd_build(&args[2], &args[3]),
        Some("build-gaddag") if args.len() == 4 => cmd_build_gaddag(&args[2], &args[3]),
        Some("lookup") if args.len() == 4 => cmd_lookup(&args[2], &args[3]),
        Some("bench") if args.len() == 4 => cmd_bench(&args[2], &args[3]),
        Some("gen-verify") if args.len() == 4 || args.len() == 5 => {
            let games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
            cmd_gen_verify(&args[2], &args[3], games)
        }
        Some("gen-bench") if args.len() == 4 || args.len() == 5 => {
            let games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200);
            cmd_gen_bench(&args[2], &args[3], games)
        }
        _ => {
            usage(&args[0]);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

/// Set how many threads parallel move generation (`get_best_words(..., parallel=True)`)
/// uses. Must be called before the first generation; raises `RuntimeError` if the
/// pool is already built. Overrides the `RAYON_NUM_THREADS` env var and the
/// `min(8, cores)` default.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    if n == 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "num_threads must be >= 1",
        ));
    }
    if GEN_POOL.get().is_some() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "thread pool already initialised; call set_num_threads before the first move generation",
        ));
    }
    REQUESTED_THREADS.store(n, Ordering::Relaxed);
    Ok(())
}

/// Number of threads parallel move generation will use (builds the pool on the
/// first call if it does not exist yet).
#[pyfunction]
fn num_threads() -> usize {
    gen_pool().current_num_threads()
}

#[pymodule]
fn scrablozaur(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DawgPy>()?;
    m.add_class::<Board>()?;
    m.add_function(pyo3::wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(num_threads, m)?)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod gaddag_tests {
    use super::*;

    /// Build a flat DAWG/GADDAG in memory from the given entries (any strings,
    /// including GADDAG entries containing `SEP`), via the real build pipeline.
    fn compile(entries: &[&str]) -> Dawg {
        let mut ws: Vec<&str> = entries.to_vec();
        ws.sort_unstable();
        ws.dedup();
        let (arena, root, _) = build_dawg(&ws);
        let (arena, root) = compact(&arena, root);
        Dawg::from_bytes(serialize(&arena, root)).unwrap()
    }

    fn compile_gaddag(words: &[&str]) -> Dawg {
        let mut entries = Vec::new();
        for w in words {
            gaddag_entries(w, &mut entries);
        }
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        compile(&refs)
    }

    /// Every word in the lexicon must be reconstructible from every one of its
    /// letters (anchors), following the reversed-prefix / SEP / suffix path.
    #[test]
    fn gaddag_reconstructs_every_word_from_every_anchor() {
        let words = ["kot", "kota", "koty", "as", "ma", "mama", "tom"];
        let g = compile_gaddag(&words);
        for w in words {
            let chars: Vec<char> = w.chars().collect();
            for a in 0..chars.len() {
                // Walk reversed prefix, SEP, then suffix; the end must be terminal.
                let mut node = g.root;
                let mut ok = true;
                for i in (0..=a).rev() {
                    match g.find_child(node, chars[i]) {
                        Some(n) => node = n,
                        None => {
                            ok = false;
                            break;
                        }
                    }
                }
                assert!(ok, "prefix walk failed for {w} anchor {a}");
                node = g.find_child(node, SEP).expect("SEP edge");
                for &c in &chars[a + 1..] {
                    node = g.find_child(node, c).expect("suffix edge");
                }
                assert!(g.node_is_terminal(node), "{w} not terminal at anchor {a}");
            }
        }
        // A non-word must not appear as any anchor's path.
        let n = g.find_child(g.root, 'x');
        assert!(n.is_none() || !g.node_is_terminal(g.find_child(n.unwrap(), SEP).unwrap_or(g.root)));
    }

    /// cross_bits sets exactly the letters that complete a valid cross-word.
    #[test]
    fn cross_bits_matches_dictionary() {
        // Dictionary where only some completions of "k_t" / "_t" are words.
        let dawg = compile(&["kot", "kit", "at", "ot", "ma"]);
        // prefix "k", suffix "t": allowed middles are o (kot) and i (kit).
        let bits = cross_bits(&dawg, &['k'], &['t']);
        assert_ne!(bits & letter_bit('o'), 0);
        assert_ne!(bits & letter_bit('i'), 0);
        assert_eq!(bits & letter_bit('a'), 0, "kat is not in the lexicon");
        // prefix empty, suffix "t": allowed leaders are a (at) and o (ot).
        let bits = cross_bits(&dawg, &[], &['t']);
        assert_ne!(bits & letter_bit('a'), 0);
        assert_ne!(bits & letter_bit('o'), 0);
        assert_eq!(bits & letter_bit('k'), 0);
    }

    /// The GADDAG generator finds a simple hook, and blanks stand in for a
    /// missing letter.
    #[test]
    fn generator_finds_hook_and_uses_blank() {
        let dawg = compile(&["kot", "koty", "ty", "oto"]);
        let gaddag = compile_gaddag(&["kot", "koty", "ty", "oto"]);
        let mut board = Board::new().unwrap();
        board.place_word("kot", 7, 7, true, None).unwrap();
        board.first = false;

        // With a 'y' in hand, "koty" (hooking 'y' after "kot") must be found.
        let moves = board.gaddag_best_words(&dawg, &gaddag, "y", 5, false);
        assert!(
            moves.iter().any(|(w, ..)| w == "koty"),
            "expected 'koty' hook, got {:?}",
            moves.iter().map(|m| &m.0).collect::<Vec<_>>()
        );

        // With only a blank, the same hook must still be found (blank = y).
        let moves_blank = board.gaddag_best_words(&dawg, &gaddag, "?", 5, false);
        assert!(
            moves_blank.iter().any(|(w, ..)| w == "koty"),
            "expected 'koty' via blank, got {:?}",
            moves_blank.iter().map(|m| &m.0).collect::<Vec<_>>()
        );
    }

    /// A blank keeps its zero value for the rest of the game: every later word
    /// that runs through or extends its square must score it as 0, not as the
    /// letter it stands in for.
    #[test]
    fn played_blank_keeps_scoring_zero() {
        let dawg = compile(&["kot", "koty", "ty", "oto"]);
        let gaddag = compile_gaddag(&["kot", "koty", "ty", "oto"]);

        // "kot" at (7,7) horizontally, with the 't' played as a blank. (7,7) is
        // the centre double-word square; k=2, o=1, t=2 at face value.
        let mut board = Board::new().unwrap();
        board
            .place_word("kot", 7, 7, true, Some(vec!['k', 'o', '?']))
            .unwrap();
        assert!(board.blanks[7][9], "the 't' square should be flagged blank");

        // Extending to "koty" re-scores the whole main word. k(2) + o(1) +
        // blank t(0) + y(2) = 5. Without blank tracking this comes out 7.
        assert_eq!(board.calculate_word_points("koty", 7, 7, true, "y").unwrap(), 5);

        // The same must hold through the fast generator path used in search.
        let moves = board.gaddag_best_words(&dawg, &gaddag, "y", 5, false);
        let koty = moves.iter().find(|(w, ..)| w == "koty").expect("koty");
        assert_eq!(koty.1, 5, "generator disagreed with calculate_word_points");

        // And through a cross-word: "ty" played vertically off the blank 't'
        // scores blank t(0) + y(2) = 2.
        assert_eq!(board.calculate_word_points("ty", 7, 9, false, "y").unwrap(), 2);
    }

    /// The same seed deals the same tiles in the same order, and different
    /// seeds don't. The arena's whole design rests on this.
    #[test]
    fn seeded_boards_deal_identically() {
        let deal = |seed: u64| {
            let mut b = Board::seeded(seed).unwrap();
            (0..6).map(|_| b.give_letters("")).collect::<Vec<_>>()
        };
        assert_eq!(deal(12345), deal(12345));
        assert_ne!(deal(12345), deal(12346));
        // Seed 0 is xorshift64's fixed point and must not deal a constant bag.
        let zero = deal(0);
        assert!(zero[0] != zero[1], "seed 0 degenerated into a constant stream");
    }

    /// Exchanged tiles go back into the bag, but not in time to be drawn as
    /// their own replacements.
    #[test]
    fn exchange_cannot_redraw_the_discarded_tiles() {
        let mut board = Board::new().unwrap();
        // Draw order is from the end, so this deals h, g, f, e, d, c, b.
        board.set_bag(vec!['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']);
        let rack = board.exchange_letters("zzz", "zzz");

        assert_eq!(rack.chars().count(), RACK_SIZE);
        assert!(
            !rack.contains('z'),
            "drew back a just-discarded tile: {rack}"
        );
        assert_eq!(rack, "hgfedcb");
        // The three z's are back in the bag, along with the undrawn 'a'.
        let mut left = board.bag_tiles();
        left.sort_unstable();
        assert_eq!(left, vec!['a', 'z', 'z', 'z']);
    }

    /// `copy()` is a real deep copy, and `unplace_word` restores the position
    /// exactly -- both are load-bearing for make/unmake search.
    #[test]
    fn copy_and_unplace_restore_the_position() {
        let mut board = Board::new().unwrap();
        let before = board.__str__();
        let cells = board
            .place_word("kot", 7, 7, true, Some(vec!['k', 'o', '?']))
            .unwrap();

        let branch = board.copy();
        board.unplace_word(cells, true).unwrap();
        assert_eq!(board.__str__(), before, "unplace did not restore the grid");
        assert!(board.first, "unplace did not restore the opening flag");
        assert!(!board.blanks[7][9], "unplace left a stale blank flag");

        // The copy is unaffected by the mutation of its source.
        assert!(branch.blanks[7][9]);
        assert_eq!(branch.board[7][7], 'k');
    }
}
