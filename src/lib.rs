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
    [(1,3), (1,1), (1,1), (2,1), (1,1), (1,1), (1,1), (1,3)],
    [(1,1), (1,2), (1,1), (1,1), (1,1), (3,1), (1,1), (1,1)],
    [(1,1), (1,1), (1,2), (1,1), (1,1), (1,1), (2,1), (1,1)],
    [(2,1), (1,1), (1,1), (1,2), (1,1), (1,1), (1,1), (2,1)],
    [(1,1), (1,1), (1,1), (1,1), (1,2), (1,1), (1,1), (1,1)],
    [(1,1), (3,1), (1,1), (1,1), (1,1), (3,1), (1,1), (1,1)],
    [(1,1), (1,1), (2,1), (1,1), (1,1), (1,1), (2,1), (1,1)],
    [(1,3), (1,1), (1,1), (2,1), (1,1), (1,1), (1,1), (1,2)],
];

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
            let n_children =
                u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4 + n_children * 8;
        }

        Ok(Self { data, root, offset_table })
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
            &pattern_chars, 0, self.root,
            &mut freq, bag_count, mandatory_slots,
            &mut results, &mut current,
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
                    let ci = c as usize;
                    // try the exact letter
                    if ci < FREQ_SIZE && freq[ci] > 0 {
                        freq[ci] -= 1;
                        current.push(c);
                        self.match_pattern(
                            pattern, pat_pos + 1, child_id,
                            freq, bag_count - 1, mandatory_slots - 1,
                            results, current,
                        );
                        current.pop();
                        freq[ci] += 1;
                    }
                    // try blank tile ('?') as this letter
                    let qi = '?' as usize;
                    if freq[qi] > 0 {
                        freq[qi] -= 1;
                        current.push(c);
                        self.match_pattern(
                            pattern, pat_pos + 1, child_id,
                            freq, bag_count - 1, mandatory_slots - 1,
                            results, current,
                        );
                        current.pop();
                        freq[qi] += 1;
                    }
                }
            }
            '*' => {
                // consume zero letters for this `*`
                self.match_pattern(
                    pattern, pat_pos + 1, node_id,
                    freq, bag_count, mandatory_slots,
                    results, current,
                );
                // consume one more letter and stay at the same `*` position
                if bag_count > mandatory_slots {
                    let n = self.node_children_count(node_id);
                    for i in 0..n {
                        let (c, child_id) = self.node_child(node_id, i);
                        let ci = c as usize;
                        if ci < FREQ_SIZE && freq[ci] > 0 {
                            freq[ci] -= 1;
                            current.push(c);
                            self.match_pattern(
                                pattern, pat_pos, child_id,
                                freq, bag_count - 1, mandatory_slots,
                                results, current,
                            );
                            current.pop();
                            freq[ci] += 1;
                        }
                        let qi = '?' as usize;
                        if freq[qi] > 0 {
                            freq[qi] -= 1;
                            current.push(c);
                            self.match_pattern(
                                pattern, pat_pos, child_id,
                                freq, bag_count - 1, mandatory_slots,
                                results, current,
                            );
                            current.pop();
                            freq[qi] += 1;
                        }
                    }
                }
            }
            fixed_char => {
                if let Some(child_id) = self.find_child(node_id, fixed_char) {
                    current.push(fixed_char);
                    self.match_pattern(
                        pattern, pat_pos + 1, child_id,
                        freq, bag_count, mandatory_slots,
                        results, current,
                    );
                    current.pop();
                }
            }
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

#[pyclass(name = "Board")]
struct Board {
    board: [[char; 15]; 15],
    tile_bag: Vec<char>,
}

#[pymethods]
impl Board {
    #[new]
    fn new(board: Vec<Vec<String>>) -> PyResult<Self> {
        if board.len() != 15 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "board must have exactly 15 rows",
            ));
        }
        let mut result = [['-'; 15]; 15];
        for (r, row) in board.iter().enumerate() {
            if row.len() != 15 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "each row must have exactly 15 columns",
                ));
            }
            for (c, cell) in row.iter().enumerate() {
                let mut chars = cell.chars();
                let ch = chars.next().ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("empty cell")
                })?;
                if chars.next().is_some() {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cell must contain exactly one character",
                    ));
                }
                result[r][c] = ch;
            }
        }
        let tile_bag = vec![
            'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'ą', 'b', 'b', 'c', 'c', 'c', 'ć',
            'd', 'd', 'd', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'ę', 'f', 'g', 'g', 'h', 'h',
            'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'j', 'j', 'k', 'k', 'k', 'l', 'l', 'l',
            'ł', 'ł', 'm', 'm', 'm', 'n', 'n', 'n', 'n', 'n', 'ń', 'o', 'o', 'o', 'o', 'o',
            'o', 'ó', 'p', 'p', 'p', 'r', 'r', 'r', 'r', 's', 's', 's', 's', 'ś', 't', 't',
            't', 'u', 'u', 'w', 'w', 'w', 'w', 'y', 'y', 'y', 'y', 'z', 'z', 'z', 'z', 'z',
            'ź', 'ż',
        ];
        Ok(Board { board: result, tile_bag })
    }

    fn __str__(&self) -> String {
        self.board
            .iter()
            .map(|row| row.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(" "))
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn give_letters(&mut self, letters: &str) -> String {
        // XOR-shift seeded by time + bag size to avoid repeated draws.
        let mut seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0)
            ^ (self.tile_bag.len() as u64);

        let mut drawn = String::new();
        let draw_count = (7 - letters.len()).min(self.tile_bag.len());
        for _ in 0..draw_count {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let idx = (seed as usize) % self.tile_bag.len();
            drawn.push(self.tile_bag.swap_remove(idx));
        }
        drawn
    }

    fn calculate_word_points(
        &self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        letters: &str,
    ) -> PyResult<u32> {
        let letter_points = |c: char| match c.to_uppercase().next().unwrap_or(c) {
            'A' | 'E' | 'I' | 'O' | 'Z' | 'W' | 'N' | 'S' | 'R' => 1,
            'D' | 'Y' | 'C' | 'K' | 'L' | 'M' | 'P' | 'T' => 2,
            'B' | 'G' | 'H' | 'J' | 'Ł' | 'U' => 3,
            'Ą' | 'Ę' | 'F' | 'Ó' | 'Ś' | 'Ż' => 4,
            'Ć' => 6,
            'Ń' => 7,
            'Ź' => 9,
            _ => 0,
        };

        let mut total = 0u32;
        let mut word_mul = 1u32;
        let mut tiles_from_hand = 0usize;

        for (i, ch) in word.chars().enumerate() {
            let r = if horizontal { row } else { row + i };
            let c = if horizontal { col + i } else { col };
            if r >= 15 || c >= 15 {
                return Err(pyo3::exceptions::PyValueError::new_err("word out of bounds"));
            }

            let (r2, c2) = ((r as u8).min(14 - r as u8), (c as u8).min(14 - c as u8));
            let bonus = BONUS_TABLE[r2 as usize][c2 as usize];

            if self.board[r][c] == '-' {
                // new tile placed from hand
                if letters.contains(ch) {
                    total += letter_points(ch) * bonus.0 as u32;
                    tiles_from_hand += 1;
                } else {
                    // blank tile — counts as the letter but scores 0
                    tiles_from_hand += 1;
                }
                word_mul *= bonus.1 as u32;

                // add perpendicular cross-word tiles
                if !horizontal {
                    let mut x = 1;
                    while x <= c && self.board[r][c - x] != '-' {
                        total += letter_points(self.board[r][c - x]);
                        x += 1;
                    }
                    let mut x = 1;
                    while c + x < 15 && self.board[r][c + x] != '-' {
                        total += letter_points(self.board[r][c + x]);
                        x += 1;
                    }
                } else {
                    let mut y = 1;
                    while y <= r && self.board[r - y][c] != '-' {
                        total += letter_points(self.board[r - y][c]);
                        y += 1;
                    }
                    let mut y = 1;
                    while r + y < 15 && self.board[r + y][c] != '-' {
                        total += letter_points(self.board[r + y][c]);
                        y += 1;
                    }
                }
            } else {
                if ch != self.board[r][c] {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "letter on board '{}' does not match word letter '{}'",
                        self.board[r][c], ch,
                    )));
                }
                total += letter_points(ch);
            }
        }

        // 50-point bonus for using all 7 tiles in one move
        Ok(total * word_mul + if tiles_from_hand == 7 { 50 } else { 0 })
    }

    fn check_word_placement(
        &self,
        dawg: &DawgPy,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> PyResult<()> {
        for (i, ch) in word.chars().enumerate() {
            let r = if horizontal { row } else { row + i };
            let c = if horizontal { col + i } else { col };
            if r >= 15 || c >= 15 {
                return Err(pyo3::exceptions::PyValueError::new_err("word out of bounds"));
            }
            if self.board[r][c] != '-' {
                continue;
            }

            let adjacent = self.cross_word(r, c, horizontal, ch);
            if adjacent.len() > 1 {
                dawg.contains(&adjacent).then_some(()).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "cross-word '{adjacent}' formed by '{}' is not in the dictionary", ch,
                    ))
                })?;
            }
        }
        Ok(())
    }

    fn place_word(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> PyResult<()> {
        for (i, ch) in word.chars().enumerate() {
            let r = if horizontal { row } else { row + i };
            let c = if horizontal { col + i } else { col };
            if r >= 15 || c >= 15 {
                return Err(pyo3::exceptions::PyValueError::new_err("word out of bounds"));
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
        for start in 0..n {
            for end in (start + 1)..n {
                if !valid_start(start) || !valid_end(end) {
                    continue;
                }
                let slice = &row[start..=end];
                if slice.iter().any(|&c| c == '-') && slice.iter().any(|&c| c != '-') {
                    patterns.push((start, end));
                }
            }
        }
        patterns
    }

    fn get_col_patterns(&self, col_idx: usize) -> Vec<(usize, usize)> {
        let n = 15usize;
        let empty = |i: usize| self.board[i][col_idx] == '-';
        let valid_start = |i: usize| i == 0 || empty(i - 1);
        let valid_end = |i: usize| i == n - 1 || empty(i + 1);

        let mut patterns = Vec::new();
        for start in 0..n {
            for end in (start + 1)..n {
                if !valid_start(start) || !valid_end(end) {
                    continue;
                }
                let mut has_empty = false;
                let mut has_tile = false;
                for i in start..=end {
                    if self.board[i][col_idx] == '-' { has_empty = true; } else { has_tile = true; }
                    if has_empty && has_tile { break; }
                }
                if has_empty && has_tile {
                    patterns.push((start, end));
                }
            }
        }
        patterns
    }

    fn get_all_patterns(&self) -> Vec<(usize, usize, usize, bool)> {
        let mut patterns = Vec::new();
        for i in 0..15 {
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
        self.best_word_from_pattern_inner(&dawg.inner, row, start, end, horizontal, letters).0
    }

    fn get_best_word(
        &self,
        dawg: &DawgPy,
        letters: &str,
        first: bool,
    ) -> (String, u32, (usize, usize, bool), Vec<char>) {
        let mut best_word = String::new();
        let mut best_pos = (0usize, 0usize, true);
        let mut best_score = 0u32;
        let mut used: Vec<char> = Vec::new();

        if first {
            // First move must cover the centre square (7, 7); try all offsets.
            for word in dawg.search("*", letters) {
                for offset in 1..7usize {
                    if offset >= word.len() { break; }
                    let score = self
                        .calculate_word_points(&word, 7, 7 - offset, true, letters)
                        .unwrap_or(0);
                    if score > best_score {
                        best_score = score;
                        best_pos = (7, 7 - offset, true);
                        best_word = word.clone();
                        used = best_word.chars().collect();
                    }
                }
            }
        } else {
            let patterns = self.get_all_patterns();
            if let Some((word, score, ar, ac, horiz)) = patterns
                .into_par_iter()
                .filter_map(|(row, start, end, horizontal)| {
                    let (ar, ac) = if horizontal { (row, start) } else { (start, row) };
                    let (word, score) =
                        self.best_word_from_pattern_inner(&dawg.inner, ar, ac, end, horizontal, letters);
                    if word.is_empty() { None } else { Some((word, score, ar, ac, horizontal)) }
                })
                .max_by_key(|&(_, score, ..)| score)
            {
                best_score = score;
                best_pos = (ar, ac, horiz);
                used = word.chars().enumerate()
                    .filter_map(|(i, ch)| {
                        let r = if horiz { ar } else { ar + i };
                        let c = if horiz { ac + i } else { ac };
                        if self.board[r][c] == '-' { Some(ch) } else { None }
                    })
                    .collect();
                best_word = word;
            }
        }

        (best_word, best_score, best_pos, used)
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
            while x <= col && self.board[row][col - x] != '-' { x += 1; }
            let start = col - x + 1;
            let mut x = 1;
            while col + x < 15 && self.board[row][col + x] != '-' { x += 1; }
            let end = col + x - 1;
            if end > start || (end == start && (start < col || col < end)) {
                for ci in start..=end {
                    word.push(if ci == col { ch } else { self.board[row][ci] });
                }
            }
        } else {
            // placing horizontally → check vertical neighbours
            let mut y = 1;
            while y <= row && self.board[row - y][col] != '-' { y += 1; }
            let start = row - y + 1;
            let mut y = 1;
            while row + y < 15 && self.board[row + y][col] != '-' { y += 1; }
            let end = row + y - 1;
            if end > start || (end == start && (start < row || row < end)) {
                for ri in start..=end {
                    word.push(if ri == row { ch } else { self.board[ri][col] });
                }
            }
        }
        word
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
            let r = if horizontal { row } else { row + i };
            let c = if horizontal { col + i } else { col };
            if r >= 15 || c >= 15 { return false; }
            if self.board[r][c] != '-' { continue; }

            let adjacent = self.cross_word(r, c, horizontal, ch);
            if adjacent.len() > 1 && !dawg.contains(&adjacent) {
                return false;
            }
        }
        true
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
            for i in start..=end { pattern.push(self.board[row][i]); }
        } else {
            for i in row..=end { pattern.push(self.board[i][start]); }
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
    eprintln!("  done in {:.2?}  │  {} nodes after minimization", t0.elapsed(), node_count);

    let data = serialize(&arena, root);
    {
        let file = fs::File::create(dawg_path)?;
        BufWriter::new(file).write_all(&data)?;
    }
    eprintln!("  {:.3} MiB → '{dawg_path}'", data.len() as f64 / (1 << 20) as f64);
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
            if dawg.contains(w) { found += 1; }
        }
    }
    let elapsed = t0.elapsed();
    let total = n * PASSES as usize;
    let secs = elapsed.as_secs_f64();
    println!("\nResults ({PASSES} × {n} = {total} lookups):");
    println!("  total time  : {elapsed:.3?}");
    println!("  throughput  : {:.0} lookups/s", total as f64 / secs);
    println!("  per lookup  : {:.1} ns", secs * 1e9 / total as f64);
    println!("  hits        : {found}/{total} ({:.1}%)", 100.0 * found as f64 / total as f64);
    Ok(())
}

pub fn main_cli() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("build")  if args.len() == 4 => cmd_build(&args[2], &args[3]),
        Some("lookup") if args.len() == 4 => cmd_lookup(&args[2], &args[3]),
        Some("bench")  if args.len() == 4 => cmd_bench(&args[2], &args[3]),
        _ => { usage(&args[0]); std::process::exit(1); }
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
