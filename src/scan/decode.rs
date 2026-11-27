//! The most probable legal board, given what the camera made of each square.
//!
//! Read one square at a time, a board picks up mistakes no real board has: a
//! word that is not a word, a tile on its own, two of a letter the bag holds
//! one of. The decoder searches for the board that breaks none of those rules
//! and that the readings make most likely -- so the dictionary is weighed
//! together with the pixels, instead of patching their verdict afterwards the
//! way `verify_and_correct` does.
//!
//! A board costs `-ln P(readings | board)`, the squares taken as independent:
//!
//! - an empty square: `-ln(1 - p_tile)`;
//! - a tile showing letter `L`: `-ln p_tile - ln((1 - p_blank) P(L))`;
//! - a blank playing as `L`: `-ln p_tile - ln p_blank`;
//!
//! with every probability floored, so that no single reading can veto a board,
//! and a square with no letter reading at all charging no letter anything: the
//! words decide it. The search mostly deals in *excess*: what a square's value
//! costs above the cheapest value that square could take, so the readings' own
//! best guess costs nothing and every correction is paid for.
//!
//! A decoded board obeys: every line of two or more tiles is a word; no tile
//! stands alone; no letter is on the board more often than the bag holds it,
//! blanks covering the excess; the tiles form one group through the centre
//! square; the previous confirmed board's tiles stay as they were.
//!
//! The search has two layers. With the tiles where the readings put them, a
//! branch and bound over the board's words finds the cheapest letters: each
//! word's candidates come cheapest first from a best-first walk of the DAWG,
//! and the words crossing an assigned one must keep at least one candidate.
//! Around that, squares near whatever had to be corrected or broke a rule are
//! toggled between tile and empty one at a time, keeping the toggle that
//! allows the cheapest board. Both run in stages allowed to stray further and
//! further from the readings (`STAGES`), so that a board one restored tile
//! away from legal is found before the budget goes on proving some many-word
//! rewrite optimal.
//!
//! `board_reader/scripts/eval_decoder.py` measures all this on legal boards
//! read the way the classical scanner reads real photos; tune against it.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::hash::{BuildHasherDefault, Hasher};

use super::{verify_and_correct, Grid, MIN_ALTERNATIVE_PROB};
use crate::{value_error, Dawg, EngineResult, Lexicon, ALPHABET_MAX, BOARD_SIZE};

const CELLS: usize = BOARD_SIZE * BOARD_SIZE;
/// No letter decided: an unassigned square, or a free position in a pattern.
const FREE: u8 = u8::MAX;
/// The readings' best guess for a square is a blank.
const GUESS_BLANK: u8 = u8::MAX - 1;
const NO_SLOT: u16 = u16::MAX;
/// Toggles tried per round, cheapest first.
const MAX_CANDIDATES: usize = 48;

/// What the vision stage made of one square.
#[derive(Clone, Debug, PartialEq)]
pub struct CellReading {
    /// Probability that the square holds a tile.
    pub p_tile: f32,
    /// Probabilities of the letter a tile here shows, in any order. Letters
    /// left out share whatever the listed ones leave; an empty list says
    /// nothing about the letter at all.
    pub letters: Vec<(char, f32)>,
    /// Probability that a tile here is a blank, which shows no letter.
    pub p_blank: f32,
}

impl CellReading {
    /// A square the vision stage is sure is empty.
    pub fn empty() -> Self {
        CellReading {
            p_tile: 0.0,
            letters: Vec::new(),
            p_blank: 0.0,
        }
    }
}

/// Tiles on a board: their letters (`'-'` for an empty square) and which of
/// them are blanks playing as that letter.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanBoard {
    pub letters: Grid<char>,
    pub blanks: Grid<bool>,
}

impl ScanBoard {
    pub fn empty() -> Self {
        ScanBoard {
            letters: [['-'; BOARD_SIZE]; BOARD_SIZE],
            blanks: [[false; BOARD_SIZE]; BOARD_SIZE],
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecodeOptions {
    /// `p_tile` is clamped into `[min_p_tile, max_p_tile]`: how far one
    /// square's tile-or-empty verdict is trusted against the rules. A tile
    /// read with conviction must cost more to delete than a letter costs to
    /// change (`letter_floor`): words ending in 'ś' mostly stay words without
    /// it ("coś", "ktoś"), and a reader unsure of an 'ś' had the decoder
    /// dropping the tile instead of mending the letter.
    pub min_p_tile: f32,
    pub max_p_tile: f32,
    /// The least probability any letter gets, however sure the reader is.
    pub letter_floor: f32,
    /// The least probability that a tile is a blank. Below `letter_floor`:
    /// a blank shows no glyph, so a legible letter the words reject is far
    /// likelier misread than a blank.
    pub blank_floor: f32,
    /// Squares toggled between tile and empty, at most, over the decode.
    pub max_flips: usize,
    /// Work allowed over the whole decode: DAWG nodes expanded plus search
    /// nodes visited. Past it, the best board found so far is returned.
    pub budget: usize,
}

impl Default for DecodeOptions {
    fn default() -> Self {
        DecodeOptions {
            min_p_tile: 0.005,
            max_p_tile: 0.9999,
            letter_floor: 1e-3,
            blank_floor: 1e-4,
            max_flips: 6,
            budget: 300_000,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    pub board: ScanBoard,
    /// Squares where the board differs from the readings' own best guess: a
    /// letter changed, a tile added or removed, a blank seen where a letter
    /// was read or the reverse. Never the previous board's tiles, which are
    /// kept whatever was read.
    pub corrected: Grid<bool>,
    /// Squares still in a non-word or standing alone. Only ever set when the
    /// board is not `legal`.
    pub flagged: Grid<bool>,
    /// Every rule holds. When no legal board was found, `board` is instead
    /// the readings' best guess after `verify_and_correct`, flagged as
    /// `assemble` flags it.
    pub legal: bool,
    /// The search finished within its budget.
    pub complete: bool,
    /// The board's cost minus that of the readings' best guess, in nats:
    /// how much the rules forced the decoder away from the readings.
    /// Infinite when the board is not legal.
    pub cost: f32,
    /// Work done, in the units of `DecodeOptions::budget`.
    pub work: usize,
}

/// Above this cost a legal board is the decoder's guess more than a reading:
/// it overrode letters the reader was sure of. Overriding one costs about 6.9
/// nats, the letter floor's -- as making a word the dictionary lacks into one
/// it has does; a letter the reader was unsure of costs far less. On legal
/// boards read with held-out readings, later scans cost 0 nine times in ten;
/// with one word made a non-word, the decoder's legal rewrite costs 6.9 or
/// more nine times in ten.
pub const TRUSTED_COST: f32 = 4.0;

impl Decoded {
    /// Legal only by overriding letters the reader was sure of.
    pub fn doubtful(&self) -> bool {
        self.legal && self.cost > TRUSTED_COST
    }

    /// The squares a player should look at: those in a non-word or alone or,
    /// on a doubtful board, those the decoder changed (never the previous
    /// board's tiles, which it only keeps).
    pub fn to_check(&self) -> Grid<bool> {
        if self.doubtful() {
            self.corrected
        } else {
            self.flagged
        }
    }
}

/// Per DAWG node, bit `k` set when some word ends exactly `k` letters below
/// it. Longer words are dropped: no line on the board is longer.
pub(crate) fn word_lengths(dawg: &Dawg) -> Vec<u16> {
    let n = dawg.node_count();
    let mut mask = vec![0u16; n];
    let mut seen = vec![false; n];
    // Post-order, iteratively: in a DAG every child is finished before its
    // parent, so one pass suffices. Node ids are not in topological order.
    let mut stack: Vec<(u32, usize)> = vec![(dawg.root, 0)];
    seen[dawg.root as usize] = true;
    while let Some(top) = stack.last_mut() {
        let (node, next) = *top;
        let children = dawg.node_children(node);
        if next < children.len() {
            top.1 += 1;
            let child = children[next].id;
            if !seen[child as usize] {
                seen[child as usize] = true;
                stack.push((child, 0));
            }
        } else {
            let mut m = dawg.node_is_terminal(node) as u16;
            for ch in children {
                m |= mask[ch.id as usize] << 1;
            }
            mask[node as usize] = m;
            stack.pop();
        }
    }
    mask
}

/// The node below `node` along letter index `l`: alphabet letters' symbol
/// indexes are their letter indexes, so the child bitmap answers directly.
#[inline]
fn child_of(dawg: &Dawg, node: u32, l: u8) -> Option<u32> {
    let i = node as usize;
    let bit = 1u64 << l;
    let bm = dawg.bitmap[i];
    if bm & bit == 0 {
        return None;
    }
    let off = (bm & (bit - 1)).count_ones() as usize;
    Some(dawg.children[dawg.child_start[i] as usize + off].id)
}

fn finite_or(x: f32, default: f32) -> f32 {
    if x.is_finite() {
        x
    } else {
        default
    }
}

/// The readings turned into costs, and what the previous board fixes.
struct Model {
    letters: Vec<char>,
    /// A tile here costs this much more than an empty square (may be < 0).
    tile: [f32; CELLS],
    /// The cheapest value a tile here can take.
    lb: [f32; CELLS],
    /// Per letter, the cost of a real tile showing it.
    real: Vec<[f32; ALPHABET_MAX]>,
    /// The cost of a blank, whatever letter it plays.
    blank: [f32; CELLS],
    /// Per letter, `min(real, blank) - lb`.
    ex: Vec<[f32; ALPHABET_MAX]>,
    /// `p_tile >= 0.5`: the readings say there is a tile here.
    seen: [bool; CELLS],
    /// The readings' best value for a tile here: a letter, or `GUESS_BLANK`.
    guess: [u8; CELLS],
    /// The reading named at least one letter.
    informed: [bool; CELLS],
    /// The previous board's letter here, or `FREE`.
    locked: [u8; CELLS],
    locked_blank: [bool; CELLS],
    /// Real tiles of each letter, and blanks, the previous board leaves.
    avail: [u8; ALPHABET_MAX],
    blanks_avail: u8,
}

impl Model {
    fn new(
        lexicon: &Lexicon,
        readings: &Grid<CellReading>,
        prior: Option<&ScanBoard>,
        o: &DecodeOptions,
    ) -> EngineResult<Self> {
        let lang = lexicon.lang();
        let letters = lang.letters.clone();
        let n = letters.len();
        let lo = finite_or(o.min_p_tile, 0.005).clamp(1e-6, 0.5);
        let hi = finite_or(o.max_p_tile, 0.9999).clamp(0.5, 1.0 - 1e-6);
        let letter_floor = finite_or(o.letter_floor, 1e-3).clamp(1e-12, 1.0);
        let blank_floor = finite_or(o.blank_floor, 1e-4).clamp(1e-12, 1.0);

        let mut m = Model {
            letters,
            tile: [0.0; CELLS],
            lb: [0.0; CELLS],
            real: vec![[f32::INFINITY; ALPHABET_MAX]; CELLS],
            blank: [0.0; CELLS],
            ex: vec![[f32::INFINITY; ALPHABET_MAX]; CELLS],
            seen: [false; CELLS],
            guess: [0; CELLS],
            informed: [false; CELLS],
            locked: [FREE; CELLS],
            locked_blank: [false; CELLS],
            avail: [0; ALPHABET_MAX],
            blanks_avail: 0,
        };
        for &ch in &lang.bag {
            match lang.letter_index(ch) {
                Some(l) => m.avail[l] = m.avail[l].saturating_add(1),
                None if ch == lang.blank => m.blanks_avail = m.blanks_avail.saturating_add(1),
                None => {}
            }
        }

        for c in 0..CELLS {
            let reading = &readings[c / BOARD_SIZE][c % BOARD_SIZE];
            let p_raw = finite_or(reading.p_tile, 0.0);
            m.seen[c] = p_raw >= 0.5;
            let p = p_raw.clamp(lo, hi);
            m.tile[c] = (1.0 - p).ln() - p.ln();

            let p_blank = finite_or(reading.p_blank, 0.0).clamp(0.0, 1.0);
            let mut prob = [0f32; ALPHABET_MAX];
            let mut listed = [false; ALPHABET_MAX];
            let (mut sum, mut count) = (0f32, 0usize);
            for &(ch, q) in &reading.letters {
                let Some(l) = lang.letter_index(ch) else {
                    continue;
                };
                let q = finite_or(q, 0.0).max(0.0);
                if !listed[l] {
                    listed[l] = true;
                    count += 1;
                }
                prob[l] += q;
                sum += q;
            }
            m.informed[c] = count > 0;
            // Nothing read at all is equally likely whatever the letter, so
            // it costs no letter anything: the words decide. (Pricing it at
            // 1/n made restoring a tile the detector missed cost more than
            // deleting a correctly read neighbour, the commoner mistake by
            // far.)
            let share = match count {
                0 => 1.0,
                k if k < n => (1.0 - sum).max(0.0) / (n - k) as f32,
                _ => 0.0,
            };
            let (mut best, mut best_l) = (f32::INFINITY, 0);
            for l in 0..n {
                let q = if listed[l] { prob[l] } else { share };
                let cost = -((1.0 - p_blank) * q).max(letter_floor).ln();
                m.real[c][l] = cost;
                if cost < best {
                    best = cost;
                    best_l = l;
                }
            }
            m.blank[c] = -p_blank.max(blank_floor).ln();
            m.lb[c] = best.min(m.blank[c]);
            m.guess[c] = if m.blank[c] < best {
                GUESS_BLANK
            } else {
                best_l as u8
            };
            for l in 0..n {
                m.ex[c][l] = m.real[c][l].min(m.blank[c]) - m.lb[c];
            }
        }

        if let Some(prior) = prior {
            for c in 0..CELLS {
                let (r, col) = (c / BOARD_SIZE, c % BOARD_SIZE);
                let ch = prior.letters[r][col];
                if ch == '-' {
                    continue;
                }
                let l = lang.letter_index(ch).ok_or_else(|| {
                    value_error(format!(
                        "the previous board holds '{ch}', which is not a letter of '{}'",
                        lang.code
                    ))
                })?;
                m.locked[c] = l as u8;
                m.locked_blank[c] = prior.blanks[r][col];
                if prior.blanks[r][col] {
                    m.blanks_avail = m.blanks_avail.saturating_sub(1);
                } else {
                    m.avail[l] = m.avail[l].saturating_sub(1);
                }
            }
        }
        Ok(m)
    }

    /// What tiles on these squares cost before any excess: each one's price
    /// over an empty square plus its cheapest value. The previous board's
    /// tiles are not the readings' to price.
    fn base(&self, occ: &[bool; CELLS]) -> f32 {
        (0..CELLS)
            .filter(|&c| occ[c] && self.locked[c] == FREE)
            .map(|c| self.tile[c] + self.lb[c])
            .sum()
    }

    /// Which squares the readings themselves put tiles on.
    fn seen_occupancy(&self) -> [bool; CELLS] {
        std::array::from_fn(|c| self.seen[c] || self.locked[c] != FREE)
    }
}

/// A line of two or more tiles: a word on the board.
#[derive(Clone, Copy)]
struct Slot {
    cells: [u8; BOARD_SIZE],
    len: u8,
    /// 0 across, 1 down.
    dir: u8,
}

impl Slot {
    fn cells(&self) -> &[u8] {
        &self.cells[..self.len as usize]
    }
}

struct Layout {
    slots: Vec<Slot>,
    /// Per square, its slot across and its slot down, or `NO_SLOT`.
    of: [[u16; 2]; CELLS],
}

fn layout(occ: &[bool; CELLS]) -> Layout {
    let mut slots = Vec::new();
    let mut of = [[NO_SLOT; 2]; CELLS];
    for dir in [0, 1] {
        for line in 0..BOARD_SIZE {
            let at = |i: usize| {
                if dir == 0 {
                    line * BOARD_SIZE + i
                } else {
                    i * BOARD_SIZE + line
                }
            };
            let mut i = 0;
            while i < BOARD_SIZE {
                if !occ[at(i)] {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < BOARD_SIZE && occ[at(i)] {
                    i += 1;
                }
                if i - start >= 2 {
                    let mut cells = [0u8; BOARD_SIZE];
                    for k in start..i {
                        cells[k - start] = at(k) as u8;
                        of[at(k)][dir] = slots.len() as u16;
                    }
                    slots.push(Slot {
                        cells,
                        len: (i - start) as u8,
                        dir: dir as u8,
                    });
                }
            }
        }
    }
    Layout { slots, of }
}

/// A word being spelled out along a DAWG path.
#[derive(Clone, Copy)]
struct Partial {
    cost: f32,
    node: u32,
    depth: u8,
    word: [u8; BOARD_SIZE],
}

impl Ord for Partial {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap pops the greatest: make that the cheapest, then the
        // furthest along, so ties finish a word instead of widening.
        other
            .cost
            .total_cmp(&self.cost)
            .then(self.depth.cmp(&other.depth))
    }
}

impl PartialOrd for Partial {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Partial {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Partial {}

/// The words that fit one line of squares under one pattern of fixed
/// letters, produced cheapest first and kept.
struct Words {
    cells: [u8; BOARD_SIZE],
    len: u8,
    pattern: [u8; BOARD_SIZE],
    found: Vec<([u8; BOARD_SIZE], f32)>,
    frontier: BinaryHeap<Partial>,
}

/// Word-list keys are already well mixed bits; std's SipHash, built to
/// resist adversarial keys, was a tenth of a decode. Fold the two halves and
/// multiply, Fibonacci-style.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut word = [0u8; 8];
            word[..chunk.len()].copy_from_slice(chunk);
            self.write_u64(u64::from_le_bytes(word));
        }
    }

    fn write_u64(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }

    fn write_u128(&mut self, v: u128) {
        self.write_u64(v as u64);
        self.write_u64((v >> 64) as u64);
    }
}

/// Every word list of one decode. A list depends only on its squares and the
/// letters fixed on them, so it is shared by every branch of a search that
/// meets the same pattern -- and by every trial toggle, which leaves most of
/// the board's words as they were.
#[derive(Default)]
struct Lists {
    words: Vec<Words>,
    index: HashMap<u128, u32, BuildHasherDefault<KeyHasher>>,
}

/// Where the search may return to, see `Solver::undo`.
struct Mark {
    assigned: usize,
    undo: usize,
    relaxed: f32,
    adjust: f32,
    prefer: usize,
    prefer_gain: f32,
}

/// The branch and bound for one choice of which squares hold tiles.
struct Solver<'a> {
    m: &'a Model,
    dawg: &'a Dawg,
    lengths: &'a [u16],
    layout: Layout,
    order: Vec<u16>,
    /// The letter on each square so far, `FREE` if none yet.
    letter: [u8; CELLS],
    /// Slots whose word is chosen, or that need none (wholly the prior's).
    done: Vec<bool>,
    /// Per open slot, its cheapest word's excess under its current pattern.
    minex: Vec<f32>,
    /// Squares the search assigned, per letter.
    counts: [u8; ALPHABET_MAX],
    /// Assigned squares where a blank is cheaper than the letter's real
    /// tile, in assignment order, and what a blank saves on them in all.
    prefer: Vec<u8>,
    prefer_gain: f32,
    /// Assigned squares, in order, for undoing; and the same per letter.
    assigned: Vec<u8>,
    by_letter: [Vec<u8>; ALPHABET_MAX],
    /// Excess of the assigned squares, each taking whichever of real tile and
    /// blank is cheaper for it.
    relaxed: f32,
    /// What the bag's limits on blanks and letters add to `relaxed`.
    adjust: f32,
    /// `relaxed + adjust`: the assigned squares' true excess.
    excess: f32,
    minex_undo: Vec<(u16, f32)>,
    /// Reused by `blanks`.
    scratch: Vec<(f32, u8, u8)>,
    lists: &'a mut Lists,
    best: f32,
    best_letter: Option<[u8; CELLS]>,
    best_blank: [bool; CELLS],
    work: usize,
    budget: usize,
    aborted: bool,
}

enum Outcome {
    Solved {
        letter: Box<[u8; CELLS]>,
        blank: Box<[bool; CELLS]>,
        excess: f32,
    },
    /// No legal letters for these squares. `violations` counts the rules
    /// they break whatever the letters -- tiles no word can hold, words with
    /// no candidate at all, tiles cut off from the rest -- and is 0 when the
    /// search merely found nothing under its cap. `problems` are the squares
    /// involved; `strain` is the squares' cost plus each word's cheapest
    /// correction taken alone, a rough measure of how far from legal they are.
    Failed {
        violations: u32,
        problems: Vec<u8>,
        strain: f32,
    },
}

impl<'a> Solver<'a> {
    fn new(m: &'a Model, lexicon: &'a Lexicon, lists: &'a mut Lists, layout: Layout) -> Self {
        let n = layout.slots.len();
        Solver {
            m,
            dawg: &lexicon.dawg,
            lengths: lexicon.word_lengths(),
            layout,
            order: Vec::new(),
            letter: m.locked,
            done: vec![false; n],
            minex: vec![0.0; n],
            counts: [0; ALPHABET_MAX],
            prefer: Vec::new(),
            prefer_gain: 0.0,
            assigned: Vec::new(),
            by_letter: std::array::from_fn(|_| Vec::new()),
            relaxed: 0.0,
            adjust: 0.0,
            excess: 0.0,
            minex_undo: Vec::new(),
            scratch: Vec::new(),
            lists,
            best: f32::INFINITY,
            best_letter: None,
            best_blank: [false; CELLS],
            work: 0,
            budget: 0,
            aborted: false,
        }
    }

    /// The word list for slot `s` under the letters placed so far.
    fn words_for(&mut self, s: usize) -> u32 {
        let slot = self.layout.slots[s];
        let mut pattern = [FREE; BOARD_SIZE];
        // 6 bits per position fill the low 90 bits; above them, the line's
        // length, direction and first square, which together name it.
        let bits = 6 * BOARD_SIZE as u32;
        let mut key = (slot.len as u128) << bits
            | (slot.dir as u128) << (bits + 4)
            | (slot.cells[0] as u128) << (bits + 5);
        for (j, &c) in slot.cells().iter().enumerate() {
            let l = self.letter[c as usize];
            pattern[j] = l;
            key |= (if l == FREE { 63 } else { l as u128 }) << (6 * j);
        }
        if let Some(&w) = self.lists.index.get(&key) {
            return w;
        }
        let mut frontier = BinaryHeap::new();
        if self.lengths[self.dawg.root as usize] >> slot.len & 1 != 0 {
            frontier.push(Partial {
                cost: 0.0,
                node: self.dawg.root,
                depth: 0,
                word: [FREE; BOARD_SIZE],
            });
        }
        let w = self.lists.words.len() as u32;
        self.lists.words.push(Words {
            cells: slot.cells,
            len: slot.len,
            pattern,
            found: Vec::new(),
            frontier,
        });
        self.lists.index.insert(key, w);
        w
    }

    /// The `i`-th cheapest word of list `w`, spelling more of it if needed.
    fn word(&mut self, w: u32, i: usize) -> Option<([u8; BOARD_SIZE], f32)> {
        loop {
            let words = &self.lists.words[w as usize];
            if let Some(&found) = words.found.get(i) {
                return Some(found);
            }
            if words.frontier.is_empty() || self.aborted {
                return None;
            }
            self.advance(w as usize);
        }
    }

    /// Walk list `w`'s frontier until one more word is complete.
    fn advance(&mut self, w: usize) {
        let Solver {
            lists,
            m,
            dawg,
            lengths,
            work,
            budget,
            aborted,
            ..
        } = self;
        let words = &mut lists.words[w];
        let len = words.len as usize;
        while let Some(p) = words.frontier.pop() {
            *work += 1;
            if *work > *budget {
                *aborted = true;
                return;
            }
            let d = p.depth as usize;
            if d == len {
                words.found.push((p.word, p.cost));
                return;
            }
            // Only follow edges from which a word ends exactly at the slot's end.
            let rest = len - d - 1;
            let fixed = words.pattern[d];
            if fixed != FREE {
                if let Some(child) = child_of(dawg, p.node, fixed) {
                    if lengths[child as usize] >> rest & 1 != 0 {
                        let mut word = p.word;
                        word[d] = fixed;
                        words.frontier.push(Partial {
                            cost: p.cost,
                            node: child,
                            depth: p.depth + 1,
                            word,
                        });
                    }
                }
                continue;
            }
            let cell = words.cells[d] as usize;
            for ch in dawg.node_children(p.node) {
                if ch.bit == 0 || lengths[ch.id as usize] >> rest & 1 == 0 {
                    continue;
                }
                let l = ch.bit.trailing_zeros() as usize;
                let mut word = p.word;
                word[d] = l as u8;
                words.frontier.push(Partial {
                    cost: p.cost + m.ex[cell][l],
                    node: ch.id,
                    depth: p.depth + 1,
                    word,
                });
            }
        }
    }

    /// The order slots are filled in: next, always the open slot with the
    /// most squares already fixed, then the most open crossings, then the
    /// longest -- the one least likely to have a choice.
    fn plan(&mut self) {
        let slots = &self.layout.slots;
        let mut fixed: [bool; CELLS] = std::array::from_fn(|c| self.m.locked[c] != FREE);
        let mut planned = self.done.clone();
        loop {
            let mut pick: Option<(usize, (usize, usize, u8))> = None;
            for (s, slot) in slots.iter().enumerate() {
                if planned[s] {
                    continue;
                }
                let cross = 1 - slot.dir as usize;
                let key = (
                    slot.cells().iter().filter(|&&c| fixed[c as usize]).count(),
                    slot.cells()
                        .iter()
                        .filter(|&&c| {
                            let t = self.layout.of[c as usize][cross];
                            t != NO_SLOT && !planned[t as usize]
                        })
                        .count(),
                    slot.len,
                );
                if pick.is_none_or(|(_, k)| key > k) {
                    pick = Some((s, key));
                }
            }
            let Some((s, _)) = pick else { break };
            planned[s] = true;
            self.order.push(s as u16);
            for &c in slots[s].cells() {
                fixed[c as usize] = true;
            }
        }
    }

    /// The cheapest choice of which assigned squares are blanks, as what it
    /// adds to `relaxed` -- where every square took whichever of real tile and
    /// blank was cheaper for it -- or `None` if the bag cannot cover the
    /// letters. With `mask`, also marks the chosen squares.
    ///
    /// Each letter used more often than the bag has it needs that many
    /// blanks: cheapest, the squares where a blank costs least over the real
    /// tile. Blanks left over go to squares that look more like a blank than
    /// their letter, most so first.
    fn blanks(&mut self, mask: Option<&mut [bool; CELLS]>) -> Option<f32> {
        let n = self.m.letters.len();
        let over = (0..n).any(|l| self.counts[l] > self.m.avail[l]);
        if !over && self.prefer.is_empty() {
            return Some(0.0);
        }
        let mut need = [0u8; ALPHABET_MAX];
        for (l, short) in need.iter_mut().enumerate().take(n) {
            *short = self.counts[l].saturating_sub(self.m.avail[l]);
        }
        let needed: u32 = need.iter().map(|&k| k as u32).sum();
        let blanks = self.m.blanks_avail as u32;
        if needed > blanks {
            return None;
        }
        // Only squares that would rather be a blank, or that hold a letter
        // the bag is short of, can end up one: (cost of a blank over the real
        // tile, square, letter) for those, cheapest first. This runs on every
        // assignment once any square prefers a blank, so it reads the short
        // list kept for them and scans every square only when a letter runs
        // out, which is rare.
        let mut cand = std::mem::take(&mut self.scratch);
        cand.clear();
        let entry = |c: u8| {
            let l = self.letter[c as usize];
            (
                self.m.blank[c as usize] - self.m.real[c as usize][l as usize],
                c,
                l,
            )
        };
        for &c in &self.prefer {
            let e = entry(c);
            if need[e.2 as usize] == 0 {
                cand.push(e); // the short letters' squares all come next
            }
        }
        for (l, &short) in need.iter().enumerate() {
            if short > 0 {
                cand.extend(self.by_letter[l].iter().map(|&c| entry(c)));
            }
        }
        let mut adjust = self.prefer_gain;
        cand.sort_unstable_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut used = 0u32;
        for entry in cand.iter_mut() {
            let l = entry.2 as usize;
            if need[l] > 0 {
                need[l] -= 1;
                used += 1;
                adjust += entry.0;
                entry.2 = FREE; // chosen
            }
        }
        for entry in cand.iter_mut() {
            if used >= blanks || entry.0 >= 0.0 {
                break;
            }
            if entry.2 != FREE {
                used += 1;
                adjust += entry.0;
                entry.2 = FREE;
            }
        }
        if let Some(mask) = mask {
            for &(_, c, chosen) in &cand {
                mask[c as usize] = chosen == FREE;
            }
        }
        self.scratch = cand;
        Some(adjust)
    }

    fn mark(&self) -> Mark {
        Mark {
            assigned: self.assigned.len(),
            undo: self.minex_undo.len(),
            relaxed: self.relaxed,
            adjust: self.adjust,
            prefer: self.prefer.len(),
            prefer_gain: self.prefer_gain,
        }
    }

    /// Put `word` in slot `s`. False if that leaves the bag short or a
    /// crossing slot without any word; either way, `undo` puts things back.
    fn assign(&mut self, s: usize, word: &[u8; BOARD_SIZE]) -> bool {
        let slot = self.layout.slots[s];
        self.done[s] = true;
        let start = self.assigned.len();
        let mut touches_blanks = false;
        for (j, &c) in slot.cells().iter().enumerate() {
            let c = c as usize;
            if self.letter[c] != FREE {
                continue; // fixed by a crossing word, or the previous board
            }
            let l = word[j] as usize;
            self.letter[c] = l as u8;
            self.assigned.push(c as u8);
            self.by_letter[l].push(c as u8);
            self.relaxed += self.m.ex[c][l];
            self.counts[l] += 1;
            if self.m.blank[c] < self.m.real[c][l] {
                self.prefer.push(c as u8);
                self.prefer_gain += self.m.real[c][l] - self.m.blank[c];
                touches_blanks = true;
            }
            touches_blanks |= self.counts[l] > self.m.avail[l];
        }
        // A square that neither prefers a blank nor holds a letter the bag
        // is short of cannot change which squares are best made blanks.
        if touches_blanks {
            match self.blanks(None) {
                Some(adjust) => self.adjust = adjust,
                None => return false,
            }
        }
        self.excess = self.relaxed + self.adjust;
        let cross = 1 - slot.dir as usize;
        for k in start..self.assigned.len() {
            let t = self.layout.of[self.assigned[k] as usize][cross];
            if t == NO_SLOT || self.done[t as usize] {
                continue;
            }
            let w = self.words_for(t as usize);
            self.minex_undo.push((t, self.minex[t as usize]));
            match self.word(w, 0) {
                Some((_, e)) => self.minex[t as usize] = e,
                None => return false,
            }
        }
        true
    }

    fn undo(&mut self, s: usize, mark: Mark) {
        while self.assigned.len() > mark.assigned {
            let c = self.assigned.pop().unwrap() as usize;
            let l = self.letter[c] as usize;
            self.counts[l] -= 1;
            self.by_letter[l].pop();
            self.letter[c] = FREE;
        }
        while self.minex_undo.len() > mark.undo {
            let (t, e) = self.minex_undo.pop().unwrap();
            self.minex[t as usize] = e;
        }
        self.relaxed = mark.relaxed;
        self.adjust = mark.adjust;
        self.excess = mark.relaxed + mark.adjust;
        self.prefer.truncate(mark.prefer);
        self.prefer_gain = mark.prefer_gain;
        self.done[s] = false;
    }

    /// A lower bound on the excess the open slots still add. Slots running
    /// the same way share no square, so each direction's cheapest words add
    /// up to a bound; the larger of the two is the better one.
    fn bound(&self) -> [f32; 2] {
        let mut sum = [0f32; 2];
        for (s, slot) in self.layout.slots.iter().enumerate() {
            if !self.done[s] {
                sum[slot.dir as usize] += self.minex[s];
            }
        }
        sum
    }

    fn search(&mut self, mut depth: usize) {
        while depth < self.order.len() && self.done[self.order[depth] as usize] {
            depth += 1;
        }
        if depth == self.order.len() {
            if self.excess < self.best {
                let mut mask = [false; CELLS];
                self.blanks(Some(&mut mask));
                self.best = self.excess;
                self.best_letter = Some(self.letter);
                self.best_blank = mask;
            }
            return;
        }
        self.work += 1;
        if self.work > self.budget {
            self.aborted = true;
            return;
        }
        let s = self.order[depth] as usize;
        let dir = self.layout.slots[s].dir as usize;
        let w = self.words_for(s);
        let base = self.excess;
        // Without this slot; the others running its way share no square with it.
        let mut open = self.bound();
        open[dir] -= self.minex[s];
        let mut i = 0;
        while let Some((word, cost)) = self.word(w, i) {
            i += 1;
            if base + cost + open[dir] >= self.best {
                break; // words come cheapest first
            }
            // Each word tried is work: on a board of real words read
            // confidently but laid out as non-words, nodes try hundreds of
            // cached words each, and counting only nodes let a decode run
            // for seconds within its budget.
            self.work += 1;
            if self.work > self.budget {
                self.aborted = true;
                return;
            }
            let mark = self.mark();
            if self.assign(s, &word) {
                // Only the slots crossing this one changed.
                let mut after = open;
                for &(t, old) in &self.minex_undo[mark.undo..] {
                    after[1 - dir] += self.minex[t as usize] - old;
                }
                if self.excess + after[0].max(after[1]) < self.best {
                    self.search(depth + 1);
                }
            }
            self.undo(s, mark);
            if self.aborted {
                return;
            }
        }
    }
}

const CENTRE: usize = CELLS / 2;

/// How far `occ` is from one group of touching tiles through the centre
/// square, which every real board is: a violation per group beyond the
/// largest, and one for an empty centre under a non-empty board. The squares
/// to toggle are the stray groups' tiles and the empty squares around them --
/// a missed tile there would join them up -- and the centre.
///
/// Without this a false tile far from the board, read as a clear 'o', gets a
/// made-up neighbour for a two-letter word instead of being dropped: a letter
/// nobody read costs nothing, so inventing one was the cheaper way out.
fn scattered(occ: &[bool; CELLS]) -> (u32, Vec<u8>) {
    let mut group = [u8::MAX; CELLS];
    let mut sizes: Vec<usize> = Vec::new();
    let mut stack = Vec::new();
    for start in 0..CELLS {
        if !occ[start] || group[start] != u8::MAX {
            continue;
        }
        let g = sizes.len() as u8;
        sizes.push(0);
        group[start] = g;
        stack.push(start);
        while let Some(c) = stack.pop() {
            sizes[g as usize] += 1;
            for n in neighbours(c) {
                if occ[n] && group[n] == u8::MAX {
                    group[n] = g;
                    stack.push(n);
                }
            }
        }
    }
    let mut violations = 0;
    let mut problems = Vec::new();
    if sizes.is_empty() {
        return (violations, problems);
    }
    if !occ[CENTRE] {
        violations += 1;
        problems.push(CENTRE as u8);
    }
    if sizes.len() > 1 {
        violations += sizes.len() as u32 - 1;
        // The main group: the centre's, or else the largest.
        let main = if occ[CENTRE] {
            group[CENTRE]
        } else {
            (0..sizes.len()).max_by_key(|&g| sizes[g]).unwrap() as u8
        };
        for c in 0..CELLS {
            if occ[c] && group[c] != main {
                problems.push(c as u8);
                problems.extend(neighbours(c).filter(|&n| !occ[n]).map(|n| n as u8));
            }
        }
    }
    (violations, problems)
}

/// The squares beside `c`, up to four.
fn neighbours(c: usize) -> impl Iterator<Item = usize> {
    let (r, col) = (c / BOARD_SIZE, c % BOARD_SIZE);
    [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)]
        .into_iter()
        .map(move |(dr, dc)| (r as i32 + dr, col as i32 + dc))
        .filter(|&(r, c)| {
            (0..BOARD_SIZE as i32).contains(&r) && (0..BOARD_SIZE as i32).contains(&c)
        })
        .map(|(r, c)| r as usize * BOARD_SIZE + c as usize)
}

/// The cheapest legal letters with tiles exactly on `occ`, if any cost less
/// than `cap`. `work` carries the budget across calls.
fn solve(
    m: &Model,
    lexicon: &Lexicon,
    lists: &mut Lists,
    occ: &[bool; CELLS],
    cap: f32,
    work: &mut usize,
    budget: usize,
) -> Outcome {
    let layout = layout(occ);
    let (mut violations, mut problems) = scattered(occ);
    for (c, &tile) in occ.iter().enumerate() {
        if tile && m.locked[c] == FREE && layout.of[c] == [NO_SLOT; 2] {
            violations += 1;
            problems.push(c as u8);
        }
    }
    let mut solver = Solver::new(m, lexicon, lists, layout);
    solver.work = *work;
    solver.budget = budget;
    solver.best = cap;
    let mut soft = Vec::new();
    let mut strain = m.base(occ);
    for s in 0..solver.layout.slots.len() {
        let slot = solver.layout.slots[s];
        if slot.cells().iter().all(|&c| m.locked[c as usize] != FREE) {
            solver.done[s] = true; // the previous board's, already confirmed
            continue;
        }
        let w = solver.words_for(s);
        match solver.word(w, 0) {
            Some((_, e)) => {
                solver.minex[s] = e;
                strain += e;
                if e > 0.0 {
                    soft.extend_from_slice(slot.cells());
                }
            }
            None if solver.aborted => break,
            None => {
                violations += 1;
                problems.extend_from_slice(slot.cells());
            }
        }
    }
    if violations == 0 && !solver.aborted {
        solver.plan();
        solver.search(0);
    }
    *work = solver.work;
    match solver.best_letter {
        Some(letter) if violations == 0 => Outcome::Solved {
            letter: Box::new(letter),
            blank: Box::new(solver.best_blank),
            excess: solver.best,
        },
        _ if violations > 0 => Outcome::Failed {
            violations,
            problems,
            strain,
        },
        // Every word has candidates, but no combination of them fits (or
        // none beats `cap`): suspect whatever needed correcting on its own.
        _ => Outcome::Failed {
            violations: 0,
            problems: soft,
            strain,
        },
    }
}

/// One choice of tile squares and what the search made of it.
struct Trial {
    occ: [bool; CELLS],
    outcome: Outcome,
}

impl Trial {
    /// Ranks trials, best first: legal boards, by cost; then squares where
    /// nothing was found under the cap; then fewer broken rules, by strain.
    fn score(&self, m: &Model) -> (u32, f32) {
        match &self.outcome {
            Outcome::Solved { excess, .. } => (0, m.base(&self.occ) + excess),
            Outcome::Failed {
                violations, strain, ..
            } => (1 + violations, *strain),
        }
    }

    /// The squares worth toggling around: whatever the board had to correct,
    /// or whatever broke a rule.
    fn problems(&self, m: &Model) -> Vec<u8> {
        match &self.outcome {
            Outcome::Solved { letter, blank, .. } => (0..CELLS)
                .filter(|&c| self.occ[c] && m.locked[c] == FREE)
                .filter(|&c| {
                    let value = if blank[c] { GUESS_BLANK } else { letter[c] };
                    !m.seen[c] || value != m.guess[c]
                })
                .map(|c| c as u8)
                .collect(),
            Outcome::Failed { problems, .. } => problems.clone(),
        }
    }
}

/// Strain a restored tile must save, between two trials that each found
/// nothing under the cap, to count as progress.
const RESTORE_GAIN: f32 = 3.0;

/// Whether trial score `a`, reached by adding a tile if `added`, is progress
/// over `b`. Between two trials that each found nothing under the cap, strain
/// is a poor guide: dropping a tile that needed correcting lowers it, and that
/// is how a real tile got thrown away. Restoring a tile that mends the words
/// around it lowers it a lot, though -- and a board missing several tiles has
/// nothing better to go on until all but the last are back.
fn beats(a: (u32, f32), b: (u32, f32), added: bool) -> bool {
    if a.0 == 1 && b.0 == 1 {
        return added && a.1 < b.1 - RESTORE_GAIN;
    }
    a.0 < b.0 || (a.0 == b.0 && a.1 < b.1)
}

/// Squares to try toggling: every square of a word through a problem square
/// (a false tile there would break it), the squares just past that word's
/// ends (a missed tile would), and the empty neighbours of a lone tile.
/// Cheapest toggle first.
fn candidates(
    m: &Model,
    occ: &[bool; CELLS],
    problems: &[u8],
    toggled: &[bool; CELLS],
) -> Vec<usize> {
    let layout = layout(occ);
    let mut want = [false; CELLS];
    let mut take = |c: usize| {
        if m.locked[c] == FREE && !toggled[c] {
            want[c] = true;
        }
    };
    for &p in problems {
        let p = p as usize;
        take(p);
        if occ[p] && layout.of[p] == [NO_SLOT; 2] {
            for n in neighbours(p) {
                take(n);
            }
        }
        for &s in &layout.of[p] {
            if s == NO_SLOT {
                continue;
            }
            let slot = layout.slots[s as usize];
            for &q in slot.cells() {
                take(q as usize);
            }
            let step = if slot.dir == 0 { 1 } else { BOARD_SIZE };
            let (first, last) = (
                slot.cells[0] as usize,
                slot.cells[slot.len as usize - 1] as usize,
            );
            let line_start = if slot.dir == 0 {
                first % BOARD_SIZE
            } else {
                first / BOARD_SIZE
            };
            let line_end = if slot.dir == 0 {
                last % BOARD_SIZE
            } else {
                last / BOARD_SIZE
            };
            if line_start > 0 {
                take(first - step);
            }
            if line_end + 1 < BOARD_SIZE {
                take(last + step);
            }
        }
    }
    let toggle_cost = |c: usize| {
        let cost = m.tile[c] + m.lb[c];
        if occ[c] {
            -cost
        } else {
            cost
        }
    };
    let mut out: Vec<usize> = (0..CELLS).filter(|&c| want[c]).collect();
    out.sort_by(|&a, &b| toggle_cost(a).total_cmp(&toggle_cost(b)));
    out.truncate(MAX_CANDIDATES);
    out
}

/// How far above the readings' own best guess, in nats, each stage of the
/// search may look: about two corrections, then five, twelve, and anything.
/// Looking close first keeps a dense board cheap -- a tight cap prunes hard --
/// and lets restoring one missed tile win before the branch and bound spends
/// the budget proving some many-word rewrite optimal.
const STAGES: [f32; 4] = [14.0, 35.0, 85.0, f32::INFINITY];

/// The most probable legal board under `readings`, keeping every tile of
/// `prior` (the last confirmed board) as it was. See the module docs.
pub fn decode(
    lexicon: &Lexicon,
    readings: &Grid<CellReading>,
    prior: Option<&ScanBoard>,
    options: &DecodeOptions,
) -> EngineResult<Decoded> {
    let m = Model::new(lexicon, readings, prior, options)?;
    let budget = options.budget;
    let mut work = 0usize;
    let mut lists = Lists::default();
    let floor = m.base(&m.seen_occupancy());

    let mut occ = m.seen_occupancy();
    let mut toggled = [false; CELLS];
    let mut flips = 0;
    let mut current = None;
    for stage in STAGES {
        let limit = floor + stage;
        let outcome = solve(
            &m,
            lexicon,
            &mut lists,
            &occ,
            limit - m.base(&occ),
            &mut work,
            budget,
        );
        let mut trial = Trial { occ, outcome };
        while flips < options.max_flips && work <= budget {
            let problems = trial.problems(&m);
            if problems.is_empty() {
                break;
            }
            let mut next: Option<(usize, Trial)> = None;
            for c in candidates(&m, &trial.occ, &problems, &toggled) {
                let mut occ = trial.occ;
                occ[c] = !occ[c];
                let incumbent = next.as_ref().map_or(trial.score(&m), |(_, t)| t.score(&m));
                let cap = if incumbent.0 == 0 {
                    incumbent.1.min(limit)
                } else {
                    limit
                } - m.base(&occ);
                if cap <= 0.0 {
                    continue; // the toggle alone costs more than allowed
                }
                let outcome = solve(&m, lexicon, &mut lists, &occ, cap, &mut work, budget);
                let candidate = Trial { occ, outcome };
                if beats(candidate.score(&m), incumbent, occ[c]) {
                    next = Some((c, candidate));
                }
                if work > budget {
                    break;
                }
            }
            match next {
                Some((c, better)) => {
                    toggled[c] = true;
                    flips += 1;
                    trial = better;
                }
                None => break,
            }
        }
        occ = trial.occ;
        let solved = matches!(trial.outcome, Outcome::Solved { .. });
        current = Some(trial);
        if solved || work > budget {
            break;
        }
    }
    let current = current.expect("at least one stage");

    let complete = work <= budget;
    let Outcome::Solved {
        letter,
        blank,
        excess,
    } = current.outcome
    else {
        return Ok(fallback(lexicon, &m, readings, complete, work));
    };
    let mut board = ScanBoard::empty();
    let mut corrected = [[false; BOARD_SIZE]; BOARD_SIZE];
    for c in 0..CELLS {
        let (r, col) = (c / BOARD_SIZE, c % BOARD_SIZE);
        if !current.occ[c] {
            corrected[r][col] = m.seen[c];
            continue;
        }
        board.letters[r][col] = m.letters[letter[c] as usize];
        if m.locked[c] != FREE {
            board.blanks[r][col] = m.locked_blank[c];
            continue;
        }
        board.blanks[r][col] = blank[c];
        let value = if blank[c] { GUESS_BLANK } else { letter[c] };
        corrected[r][col] = !m.seen[c] || value != m.guess[c];
    }
    Ok(Decoded {
        board,
        corrected,
        flagged: [[false; BOARD_SIZE]; BOARD_SIZE],
        legal: true,
        complete,
        cost: m.base(&current.occ) + excess - floor,
        work,
    })
}

/// No legal board found: the readings' best guess, given the single-letter
/// corrections and flags `assemble` would give it.
fn fallback(
    lexicon: &Lexicon,
    m: &Model,
    readings: &Grid<CellReading>,
    complete: bool,
    work: usize,
) -> Decoded {
    let lang = lexicon.lang();
    let mut board = ScanBoard::empty();
    let mut alternatives: Grid<Vec<char>> =
        std::array::from_fn(|_| std::array::from_fn(|_| Vec::new()));
    let mut locked = [[false; BOARD_SIZE]; BOARD_SIZE];
    for c in 0..CELLS {
        let (r, col) = (c / BOARD_SIZE, c % BOARD_SIZE);
        if m.locked[c] != FREE {
            board.letters[r][col] = m.letters[m.locked[c] as usize];
            board.blanks[r][col] = m.locked_blank[c];
            locked[r][col] = true;
        } else if m.seen[c] {
            let g = m.guess[c];
            board.letters[r][col] = if g == GUESS_BLANK || !m.informed[c] {
                '?'
            } else {
                m.letters[g as usize]
            };
            let mut alts: Vec<(char, f32)> = readings[r][col]
                .letters
                .iter()
                .filter(|&&(ch, p)| p >= MIN_ALTERNATIVE_PROB && lang.letter_index(ch).is_some())
                .copied()
                .collect();
            alts.sort_by(|a, b| b.1.total_cmp(&a.1));
            alternatives[r][col] = alts.into_iter().map(|(ch, _)| ch).collect();
        }
    }
    let before = board.letters;
    let flagged = verify_and_correct(lexicon, &mut board.letters, &alternatives, &locked);
    let corrected =
        std::array::from_fn(|r| std::array::from_fn(|c| board.letters[r][c] != before[r][c]));
    Decoded {
        board,
        corrected,
        flagged,
        legal: false,
        complete,
        cost: f32::INFINITY,
        work,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::OnceLock;

    use super::*;
    use crate::Language;

    fn polish() -> &'static Lexicon {
        static LEXICON: OnceLock<Lexicon> = OnceLock::new();
        LEXICON.get_or_init(|| {
            let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let text = std::fs::read_to_string(root.join("languages/pl.json")).unwrap();
            let lang = Language::from_json(&text).unwrap();
            let dawg = root.join("words/pl/dawg.bin");
            Lexicon::open(lang, dawg.to_str().unwrap(), None).unwrap()
        })
    }

    /// Words placed as (row, col, word, across).
    fn board(words: &[(usize, usize, &str, bool)]) -> Grid<char> {
        let mut g = [['-'; BOARD_SIZE]; BOARD_SIZE];
        for &(r, c, word, across) in words {
            for (i, ch) in word.chars().enumerate() {
                let (rr, cc) = if across { (r, c + i) } else { (r + i, c) };
                g[rr][cc] = ch;
            }
        }
        g
    }

    fn tile(letters: &[(char, f32)]) -> CellReading {
        CellReading {
            p_tile: 0.99,
            letters: letters.to_vec(),
            p_blank: 0.001,
        }
    }

    /// A reader sure of every square of `g`.
    fn sure(g: &Grid<char>) -> Grid<CellReading> {
        std::array::from_fn(|r| {
            std::array::from_fn(|c| match g[r][c] {
                '-' => CellReading {
                    p_tile: 0.01,
                    ..CellReading::empty()
                },
                ch => tile(&[(ch, 0.95)]),
            })
        })
    }

    fn run(readings: &Grid<CellReading>, prior: Option<&ScanBoard>) -> Decoded {
        decode(polish(), readings, prior, &DecodeOptions::default()).unwrap()
    }

    fn count(g: &Grid<bool>) -> usize {
        g.iter().flatten().filter(|&&x| x).count()
    }

    /// "nos" across, "oko" down from its 'o': no other letter fits both.
    fn nos_oko() -> Grid<char> {
        let g = board(&[(7, 6, "nos", true), (7, 7, "oko", false)]);
        let lex = polish();
        let fits: Vec<char> = lex
            .lang()
            .letters
            .iter()
            .copied()
            .filter(|&x| lex.contains(&format!("n{x}s")) && lex.contains(&format!("{x}ko")))
            .collect();
        assert_eq!(fits, ['o'], "the tests below rely on this");
        g
    }

    #[test]
    fn word_lengths_mark_where_words_end() {
        let lex = polish();
        let lengths = lex.word_lengths();
        let node = |w: &str| {
            w.chars()
                .try_fold(lex.dawg.root, |n, ch| lex.dawg.find_child(n, ch))
                .unwrap() as usize
        };
        assert_eq!(
            lengths[node("kot")] & 0b11,
            0b11,
            "'kot' and 'kota' are words"
        );
        assert_eq!(
            lengths[lex.dawg.root as usize] & 0b111,
            0b100,
            "no words under 2 letters"
        );
        assert_eq!(lengths[node("kote")] & 1, 0);
        assert_ne!(lengths[node("kote")] & 0b10, 0, "'kotek'");
    }

    #[test]
    fn a_legal_reading_comes_back_unchanged() {
        let g = nos_oko();
        let d = run(&sure(&g), None);
        assert!(d.legal && d.complete);
        assert_eq!(d.board.letters, g);
        assert_eq!(d.cost, 0.0);
        assert_eq!(count(&d.corrected), 0);
        assert_eq!(count(&d.board.blanks), 0);
    }

    #[test]
    fn an_empty_board_stays_empty() {
        let d = run(
            &std::array::from_fn(|_| std::array::from_fn(|_| CellReading::empty())),
            None,
        );
        assert!(d.legal);
        assert_eq!(d.board, ScanBoard::empty());
    }

    #[test]
    fn a_misread_is_fixed_by_the_words_through_it() {
        let g = nos_oko();
        let mut readings = sure(&g);
        // The shared 'o' read as a confident 'ę', with no second guess.
        readings[7][7] = tile(&[('ę', 0.95)]);
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert!(d.corrected[7][7]);
        assert_eq!(count(&d.corrected), 1);
        assert!(d.cost > 0.0);
    }

    #[test]
    fn a_blank_plays_the_letter_its_words_need() {
        let g = nos_oko();
        let mut readings = sure(&g);
        readings[7][7] = CellReading {
            p_tile: 0.99,
            letters: Vec::new(),
            p_blank: 0.95,
        };
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert!(d.board.blanks[7][7]);
        assert_eq!(count(&d.board.blanks), 1);
        assert!(!d.corrected[7][7], "read as a blank, decoded as one");
    }

    #[test]
    fn the_bag_decides_between_look_alike_letters() {
        // Polish has one 'ź'. "źle" holds it, so the 'z' of "jedz" -- read
        // as 'ź', making the equally valid "jedź" -- must be a 'z' after all.
        let g = board(&[(7, 6, "źle", true), (6, 8, "jedz", false)]);
        let mut readings = sure(&g);
        readings[9][8] = tile(&[('ź', 0.6), ('z', 0.35)]);
        assert!(polish().contains("jedź"));
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert_eq!(count(&d.board.blanks), 0);
        assert!(d.corrected[9][8]);
    }

    #[test]
    fn a_doubted_tile_comes_back_when_the_words_need_it() {
        // "kotek" with the detector unsure of the 'e': without it, the last
        // 'k' stands alone.
        let g = board(&[(7, 4, "kotek", true)]);
        let mut readings = sure(&g);
        readings[7][7] = CellReading {
            p_tile: 0.3,
            letters: vec![('e', 0.9)],
            p_blank: 0.001,
        };
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert!(d.corrected[7][7]);
        assert_eq!(count(&d.corrected), 1);
    }

    #[test]
    fn a_stray_tile_is_dropped() {
        let g = nos_oko();
        let mut readings = sure(&g);
        readings[2][2] = CellReading {
            p_tile: 0.8,
            letters: vec![('a', 0.9)],
            p_blank: 0.001,
        };
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert!(d.corrected[2][2]);
    }

    #[test]
    fn a_clear_false_tile_far_away_gets_no_invented_neighbour() {
        // A two-letter word with any free letter beside it would be legal on
        // its own, but cut off from the board.
        let g = nos_oko();
        let mut readings = sure(&g);
        readings[1][12] = tile(&[('o', 0.95)]);
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert_eq!(count(&d.corrected), 1);
    }

    #[test]
    fn a_missed_tile_that_cut_the_board_in_two_comes_back() {
        // "kot", "tok" down from its 't', "ko" across from that one's 'k'.
        // Without the 'o' of "tok", every word left is still a word -- only
        // "ko" hanging apart from the rest gives the miss away.
        let g = board(&[
            (7, 6, "kot", true),
            (7, 8, "tok", false),
            (9, 8, "ko", true),
        ]);
        assert!(polish().contains("tok"));
        let mut readings = sure(&g);
        readings[8][8] = CellReading {
            p_tile: 0.01,
            letters: vec![('o', 0.6)],
            p_blank: 0.001,
        };
        let d = run(&readings, None);
        assert!(d.legal);
        assert_eq!(d.board.letters, g);
        assert_eq!(count(&d.corrected), 1);
    }

    #[test]
    fn the_previous_board_outranks_the_photo() {
        let mut prior = ScanBoard::empty();
        prior.letters = board(&[(7, 6, "kot", true)]);
        prior.blanks[7][8] = true;
        // "kot" became "kota", and the photo now reads its old 'o' as an 'a'
        // -- "kata" is a word too, but tiles do not change once played.
        let mut readings = sure(&board(&[(7, 6, "kota", true)]));
        readings[7][7] = tile(&[('a', 0.95)]);
        let d = run(&readings, Some(&prior));
        assert!(d.legal);
        assert_eq!(d.board.letters, board(&[(7, 6, "kota", true)]));
        assert!(d.board.blanks[7][8], "the previous board's blank stays one");
        assert_eq!(count(&d.corrected), 0);
    }

    #[test]
    fn a_previous_board_in_another_alphabet_is_refused() {
        let mut prior = ScanBoard::empty();
        prior.letters[7][7] = 'q';
        let readings = sure(&ScanBoard::empty().letters);
        assert!(decode(polish(), &readings, Some(&prior), &DecodeOptions::default()).is_err());
    }

    #[test]
    fn out_of_budget_it_flags_instead() {
        let g = nos_oko();
        let mut readings = sure(&g);
        readings[7][7] = tile(&[('ę', 0.95)]);
        let options = DecodeOptions {
            budget: 1,
            ..DecodeOptions::default()
        };
        let d = decode(polish(), &readings, None, &options).unwrap();
        assert!(!d.legal && !d.complete);
        assert_eq!(d.board.letters[7][7], 'ę');
        assert!(d.flagged[7][7] && d.flagged[7][6]);
        assert!(d.cost.is_infinite());
    }
}
