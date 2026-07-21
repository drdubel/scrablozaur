use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
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

struct Dawg {
    data: Vec<u8>,
    root: u32,
    offset_table: Vec<usize>,
}

impl Dawg {
    fn load(path: &str) -> io::Result<Self> {
        let data = fs::read(path)?;
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too short"));
        }
        let root = u32::from_le_bytes(data[0..4].try_into().unwrap());
        let node_count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;

        let mut offset_table = Vec::with_capacity(node_count);
        let mut pos = 8usize;
        for _ in 0..node_count {
            offset_table.push(pos);
            pos += 1;
            let n_children = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + n_children * 8;
        }

        Ok(Self {
            data,
            root,
            offset_table,
        })
    }

    #[inline]
    fn node_is_terminal(&self, id: u32) -> bool {
        self.data[self.offset_table[id as usize]] != 0
    }

    #[inline]
    fn node_children_count(&self, id: u32) -> usize {
        let base = self.offset_table[id as usize] + 1;
        u32::from_le_bytes(self.data[base..base + 4].try_into().unwrap()) as usize
    }

    #[inline]
    fn node_child(&self, id: u32, i: usize) -> (char, u32) {
        let base = self.offset_table[id as usize] + 1 + 4 + i * 8;
        let cp = u32::from_le_bytes(self.data[base..base + 4].try_into().unwrap());
        let cid = u32::from_le_bytes(self.data[base + 4..base + 8].try_into().unwrap());
        (char::from_u32(cp).unwrap(), cid)
    }

    #[inline]
    fn find_child(&self, id: u32, c: char) -> Option<u32> {
        let (mut lo, mut hi) = (0usize, self.node_children_count(id));
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let (mc, mid_id) = self.node_child(id, mid);
            match mc.cmp(&c) {
                std::cmp::Ordering::Equal => return Some(mid_id),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
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
        self.offset_table.len()
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
}

#[pymethods]
impl DawgPy {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        Dawg::load(path)
            .map(|inner| DawgPy { inner })
            .map_err(|e| PyIOError::new_err(e.to_string()))
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

/// Seed for `give_letters`' draw, from the current time mixed with the bag
/// size so repeated draws (even within the same nanosecond) don't collide.
fn draw_seed(bag_len: usize) -> u64 {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    nanos ^ (bag_len as u64)
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

#[pyclass(name = "Board")]
struct Board {
    board: [[char; BOARD_SIZE]; BOARD_SIZE],
    tile_bag: Vec<char>,
    first: bool,
}

#[pymethods]
impl Board {
    #[new]
    fn new() -> PyResult<Self> {
        Ok(Board {
            board: [['-'; BOARD_SIZE]; BOARD_SIZE],
            tile_bag: fresh_tile_bag(),
            first: true,
        })
    }

    /// Construct a board pre-filled from a 15x15 grid of single-character
    /// cells (e.g. loaded from a saved game or a scanned photo), each cell
    /// either a letter or `'-'` for empty. Starts with a full standard
    /// tile bag, same as `Board()` -- letters already on the grid are not
    /// subtracted from it, since callers that load a grid this way manage
    /// their own separate tile-bag bookkeeping rather than relying on this
    /// board's.
    #[staticmethod]
    fn from_grid(board: Vec<Vec<String>>) -> PyResult<Self> {
        if board.len() != BOARD_SIZE {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "board must have exactly 15 rows",
            ));
        }
        let mut result = [['-'; BOARD_SIZE]; BOARD_SIZE];
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
                    if let Some(pos) = tile_bag.iter().position(|&x| x == ch) {
                        tile_bag.remove(pos);
                    } else {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "letter '{}' not available in tile bag",
                            ch
                        )));
                    }
                }
            }
        }
        Ok(Board {
            board: result,
            tile_bag: tile_bag,
            first: first,
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
        let mut seed = draw_seed(self.tile_bag.len());
        let mut drawn = String::new();
        let draw_count = (RACK_SIZE - letters.chars().count()).min(self.tile_bag.len());
        for _ in 0..draw_count {
            xorshift(&mut seed);
            let idx = (seed as usize) % self.tile_bag.len();
            drawn.push(self.tile_bag.swap_remove(idx));
        }
        drawn
    }

    fn exchange_letters(&mut self, letters: &str, letters_to_exchange: &str) -> String {
        for ch in letters_to_exchange.chars() {
            self.tile_bag.push(ch);
        }
        for ch in letters_to_exchange.chars() {
            if let Some(pos) = letters.find(ch) {
                let mut letters_vec: Vec<char> = letters.chars().collect();
                letters_vec.remove(pos);
                let new_letters: String = letters_vec.into_iter().collect();
                return self.exchange_letters(&new_letters, &letters_to_exchange.replace(ch, ""));
            }
        }
        self.give_letters(letters)
    }

    /// Standard Scrabble rule: exchanging tiles for new ones from the bag is
    /// only allowed while at least a full rack's worth of tiles remain in
    /// the bag, regardless of how many tiles the player wants to exchange.
    #[staticmethod]
    fn can_exchange(bag_remaining: usize) -> bool {
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
                if let Some(neighbor_points) = self.cross_neighbor_points(r, c, horizontal, ch) {
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
                main_total += letter_points(ch);
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

    fn place_word(&mut self, word: &str, row: usize, col: usize, horizontal: bool) -> PyResult<()> {
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if !in_bounds(r, c) {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "word out of bounds",
                ));
            }
            self.board[r][c] = ch;
        }
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
        &mut self,
        dawg: &DawgPy,
        letters: &str,
        n: usize,
        parallel: bool,
    ) -> Vec<BestWord> {
        if self.first {
            self.first = false;
            self.best_opening_words(dawg, letters, n)
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
    fn get_best_word(&mut self, dawg: &DawgPy, letters: &str, parallel: bool) -> BestWord {
        self.get_best_words(dawg, letters, 1, parallel)
            .into_iter()
            .next()
            .unwrap_or_else(|| (String::new(), 0, (0, 0, true), Vec::new()))
    }
}

// Pure Rust methods — no PyO3 overhead, safe to call from rayon threads.
impl Board {
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

    /// Sum of face point values of the cross-word formed at `(row, col)`
    /// when placing `ch`, excluding `ch`'s own value — i.e. just the
    /// existing perpendicular neighbours. `None` if no cross-word forms
    /// here (no adjacent tiles). Reuses `cross_word` rather than
    /// re-walking the neighbours, since `ch` appears in that string
    /// exactly once.
    fn cross_neighbor_points(
        &self,
        row: usize,
        col: usize,
        horizontal: bool,
        ch: char,
    ) -> Option<u32> {
        let cross = self.cross_word(row, col, horizontal, ch);
        if cross.is_empty() {
            return None;
        }
        let total: u32 = cross.chars().map(letter_points).sum();
        Some(total - letter_points(ch))
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
        "Usage:\n  {prog} build  <words.txt>  <dawg.bin>\n  \
                    {prog} lookup <dawg.bin>   <word>\n  \
                    {prog} bench  <dawg.bin>   <words.txt>"
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
    eprintln!(
        "  done in {:.2?}  │  {} nodes after minimization",
        t0.elapsed(),
        node_count
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

pub fn main_cli() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 4 => cmd_build(&args[2], &args[3]),
        Some("lookup") if args.len() == 4 => cmd_lookup(&args[2], &args[3]),
        Some("bench") if args.len() == 4 => cmd_bench(&args[2], &args[3]),
        _ => {
            usage(&args[0]);
            std::process::exit(1);
        }
    }
}

// ---------------------------------------------------------------------------
// Python module
// ---------------------------------------------------------------------------

#[pymodule]
fn scrablozaur(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DawgPy>()?;
    m.add_class::<Board>()?;
    Ok(())
}
