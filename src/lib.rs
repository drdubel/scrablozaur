use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::{self, BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
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

// Letters are looked up by direct codepoint indexing, so every letter of every
// supported language must fall below this. 400 covers Latin script and its
// diacritics (the highest in use is 'ż' = U+017C = 380) plus the blank '?'
// (U+003F = 63); Cyrillic and Greek start at U+0370 and would need a different
// scheme entirely.
const FREQ_SIZE: usize = 400;
type LetterFreq = [u8; FREQ_SIZE];

fn build_freq(letters: &str) -> (LetterFreq, usize) {
    let mut freq = [0u8; FREQ_SIZE];
    let mut count = 0usize;
    for c in letters.chars() {
        // Racks arrive from HTTP requests that validate length but not
        // characters, so an out-of-range codepoint must not index off the end.
        // It is not a tile in any language, so it is dropped rather than
        // counted -- leaving it in the total would inflate the search's budget
        // of spendable tiles without a `freq` entry to spend.
        if (c as usize) < FREQ_SIZE {
            freq[c as usize] += 1;
            count += 1;
        }
    }
    (freq, count)
}

// ---------------------------------------------------------------------------
// Languages: alphabet, point table and tile distribution, loaded at runtime
// ---------------------------------------------------------------------------

/// Most letters a language may have. A cross-check set is a `u32` with one bit
/// per letter (see `Language::letter_bit`), so this is a hard ceiling rather
/// than a tunable -- it is what restricts support to Latin-script alphabets.
/// Polish, at 32, is the largest that fits.
const ALPHABET_MAX: usize = 32;
/// All bits set -- the cross-check for a square with no perpendicular
/// neighbours, where any letter is allowed because no cross-word forms.
/// Correct for every language regardless of size: bits above its alphabet can
/// never be set by `letter_bit`, so the surplus is inert.
const CROSS_ANY: u32 = u32::MAX;

/// Delimiter separating a GADDAG entry's reversed prefix from its forward
/// suffix. `'\0'` sorts before every letter, so the builder's sorted-input
/// requirement and the runtime binary search both treat it consistently.
const SEP: char = '\0';
/// `SEP`'s slot in the child bitmap. Fixed rather than derived from the
/// alphabet's length, so the on-disk node layout is identical for every
/// language and a smaller alphabet simply leaves the bits between it and here
/// unused.
const SEP_SYMBOL: u32 = ALPHABET_MAX as u32;

/// Everything about a language that the engine needs: which letters exist, what
/// they score, and how many of each are in the bag.
///
/// This used to be six `const`s, and the same tables were transcribed by hand
/// into `web/game.py`, `web/static/js/board.js`, `board_reader`'s classifier and
/// `smart_player`'s model. They now all derive from `languages/<code>.json`;
/// `tests/test_tables_agree.py` fails if any of them drifts.
pub struct Language {
    code: String,
    /// Letters in the language's own collation order. That order fixes the
    /// cross-check bit indexes and decides `first_draw_winner`'s tiebreak.
    letters: Vec<char>,
    /// Codepoint -> (alphabet index, or -1; face value). Packed into one table
    /// because `score_word` needs both for the same character, and one indexed
    /// load into 800 L1-resident bytes beats two.
    ///
    /// The index half is lowercase-only -- the engine works in lowercase
    /// throughout -- while the points half is filled for both cases, preserving
    /// the case-insensitive scoring the old `letter_points` did with
    /// `to_uppercase()` on every call.
    tab: [(i8, u8); FREQ_SIZE],
    blank: char,
    /// The full tile distribution: letters in collation order, blanks last.
    /// Boards are seeded by shuffling this, so its order is observable -- a
    /// reordering silently changes every seeded deal.
    bag: Vec<char>,
    /// Tile symbols in codepoint order, blank first: the leave-value net's
    /// feature order. Derived from `bag` rather than declared, because a net
    /// whose input order disagrees with its caller's does not fail, it just
    /// returns confident nonsense.
    eval_letters: Vec<char>,
    eval_tab: [i8; FREQ_SIZE],
}

impl Language {
    /// Build from the plain values in a language definition. `counts` and
    /// `points` must cover every alphabet letter and the blank.
    fn new(
        code: &str,
        alphabet: &str,
        blank: char,
        counts: &HashMap<char, u32>,
        points: &HashMap<char, u32>,
    ) -> Result<Self, String> {
        let letters: Vec<char> = alphabet.chars().collect();
        if letters.is_empty() {
            return Err(format!("language '{code}': alphabet is empty"));
        }
        if letters.len() > ALPHABET_MAX {
            return Err(format!(
                "language '{code}': {} letters, but a cross-check set holds {ALPHABET_MAX}",
                letters.len()
            ));
        }
        if letters.contains(&blank) {
            return Err(format!(
                "language '{code}': blank '{blank}' is also an alphabet letter"
            ));
        }

        let mut tab = [(-1i8, 0u8); FREQ_SIZE];
        let mut eval_tab = [-1i8; FREQ_SIZE];
        for (i, &c) in letters.iter().enumerate() {
            let cp = c as usize;
            if cp >= FREQ_SIZE {
                return Err(format!(
                    "language '{code}': letter '{c}' is U+{:04X}, above the U+{FREQ_SIZE:04X} the engine indexes to",
                    c as u32
                ));
            }
            if tab[cp].0 >= 0 {
                return Err(format!("language '{code}': letter '{c}' appears twice"));
            }
            tab[cp].0 = i as i8;
        }

        for (&c, &value) in points {
            let value = u8::try_from(value)
                .map_err(|_| format!("language '{code}': '{c}' scores {value}, too much for a tile"))?;
            // Both cases carry the value: the engine plays in lowercase, but
            // callers score user input that may arrive either way.
            for variant in [c, c.to_uppercase().next().unwrap_or(c)] {
                if (variant as usize) < FREQ_SIZE {
                    tab[variant as usize].1 = value;
                }
            }
        }

        let mut bag = Vec::new();
        for &c in &letters {
            let n = *counts
                .get(&c)
                .ok_or_else(|| format!("language '{code}': letter '{c}' has no tile count"))?;
            bag.extend(std::iter::repeat_n(c, n as usize));
        }
        let blanks = *counts
            .get(&blank)
            .ok_or_else(|| format!("language '{code}': blank '{blank}' has no tile count"))?;
        bag.extend(std::iter::repeat_n(blank, blanks as usize));

        let mut eval_letters: Vec<char> = bag.clone();
        eval_letters.sort_unstable();
        eval_letters.dedup();
        for (i, &c) in eval_letters.iter().enumerate() {
            eval_tab[c as usize] = i as i8;
        }

        Ok(Self {
            code: code.to_string(),
            letters,
            tab,
            blank,
            bag,
            eval_letters,
            eval_tab,
        })
    }

    /// Alphabet index (or -1) and face value of `c` in one lookup, for callers
    /// that need both -- `score_word` does, once per letter of every candidate
    /// word it scores. Out-of-range codepoints read as "not a letter, worth 0".
    #[inline]
    fn tab_entry(&self, c: char) -> (i8, u8) {
        let i = c as usize;
        if i < FREQ_SIZE {
            self.tab[i]
        } else {
            (-1, 0)
        }
    }

    /// Bit index (0..ALPHABET_MAX) of a letter, or `None` for anything else
    /// (`SEP`, the blank, punctuation, a letter of another language).
    #[inline]
    fn letter_index(&self, c: char) -> Option<usize> {
        let i = c as usize;
        if i < FREQ_SIZE {
            let idx = self.tab[i].0;
            if idx >= 0 {
                return Some(idx as usize);
            }
        }
        None
    }

    /// Single-bit cross-check mask for `c` (0 if `c` is not an alphabet letter).
    #[inline]
    fn letter_bit(&self, c: char) -> u32 {
        self.letter_index(c).map_or(0, |i| 1u32 << i)
    }

    /// Symbol index for the flat DAWG's per-node child bitmap: the letter's own
    /// index for an alphabet letter, `SEP_SYMBOL` for the GADDAG separator.
    /// `None` for anything else -- a well-formed lexicon has no such edges, and
    /// the runtime falls back to a linear scan for them.
    #[inline]
    fn symbol_index(&self, c: char) -> Option<u32> {
        if c == SEP {
            Some(SEP_SYMBOL)
        } else {
            self.letter_index(c).map(|i| i as u32)
        }
    }

    /// Face value of a tile. The blank, and anything not in the language,
    /// scores 0 -- matching a blank's fixed in-play scoring.
    #[inline]
    fn points(&self, c: char) -> u32 {
        let i = c as usize;
        if i < FREQ_SIZE {
            self.tab[i].1 as u32
        } else {
            0
        }
    }

    /// Collation rank for `first_draw_winner`'s "closest to the first letter"
    /// tiebreak: a blank outranks every letter, unknown characters sort last.
    fn rank(&self, c: char) -> i32 {
        if c == self.blank {
            -1
        } else {
            self.letter_index(c).map_or(i32::MAX, |i| i as i32)
        }
    }

    /// Feature-vector slot for a tile symbol, or `None` for anything not in the
    /// distribution.
    #[inline]
    fn eval_index(&self, c: char) -> Option<usize> {
        let i = c as usize;
        if i < FREQ_SIZE && self.eval_tab[i] >= 0 {
            Some(self.eval_tab[i] as usize)
        } else {
            None
        }
    }

    #[inline]
    fn eval_size(&self) -> usize {
        self.eval_letters.len()
    }

    /// Feature count the leave-value net must have been trained with.
    #[inline]
    fn eval_input_dim(&self) -> usize {
        self.eval_size() + 1 + N_BOARD_FEATURES
    }

    /// Whether two definitions describe the same language, for interning. The
    /// tables are the definition; the name is only a label.
    fn same_definition(&self, other: &Language) -> bool {
        self.code == other.code
            && self.letters == other.letters
            && self.blank == other.blank
            && self.bag == other.bag
            && self.tab == other.tab
    }
}

/// A language's on-disk definition. The Python side reads the same file (see
/// `src/languages.py`) and validates it far more thoroughly; this reads only
/// the fields the engine itself needs.
#[derive(serde::Deserialize)]
struct LanguageFile {
    code: String,
    alphabet: String,
    blank: char,
    tiles: HashMap<char, TileDef>,
}

#[derive(serde::Deserialize)]
struct TileDef {
    count: u32,
    points: u32,
}

impl Language {
    /// Parse `languages/<code>.json`.
    fn from_file(path: &std::path::Path) -> Result<Self, String> {
        let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file: LanguageFile =
            serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let counts = file.tiles.iter().map(|(&c, t)| (c, t.count)).collect();
        let points = file.tiles.iter().map(|(&c, t)| (c, t.points)).collect();
        Language::new(&file.code, &file.alphabet, file.blank, &counts, &points)
    }
}

/// Interned languages. Definitions are immutable and there are only ever a
/// handful, so each is leaked once and shared as `&'static`. That keeps
/// `Board`'s clone -- which happens per endgame node and per rollout ply -- a
/// pointer copy rather than an atomic refcount bump, keeps lifetimes out of
/// `Board`/`Dawg`/`GenCtx`, and makes pointer equality a valid identity test
/// for "were these built from the same definition?".
static LANGUAGES: OnceLock<Mutex<Vec<&'static Language>>> = OnceLock::new();

fn intern_language(lang: Language) -> &'static Language {
    let registry = LANGUAGES.get_or_init(|| Mutex::new(Vec::new()));
    let mut interned = registry.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&existing) = interned.iter().find(|e| e.same_definition(&lang)) {
        return existing;
    }
    let leaked: &'static Language = Box::leak(Box::new(lang));
    interned.push(leaked);
    leaked
}

/// Locate `languages/<code>.json` by walking up from the current directory.
/// Only the CLI needs this -- the Python side passes a definition in directly.
fn find_language_file(code: &str) -> Result<std::path::PathBuf, String> {
    let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
    loop {
        let candidate = dir.join("languages").join(format!("{code}.json"));
        if candidate.exists() {
            return Ok(candidate);
        }
        if !dir.pop() {
            return Err(format!(
                "no languages/{code}.json found in the current directory or any parent"
            ));
        }
    }
}

/// Load a language by code, for the CLI.
fn load_language(code: &str) -> Result<&'static Language, String> {
    Ok(intern_language(Language::from_file(&find_language_file(code)?)?))
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
    /// The language this lexicon is in. Edge letters are decoded into symbol
    /// indexes at load, so a DAWG is bound to the alphabet it was loaded under.
    lang: &'static Language,
    /// `KIND_DAWG` or `KIND_GADDAG`, from the file header -- so loading one
    /// where the other is wanted is caught rather than silently generating
    /// nothing.
    kind: u8,
    root: u32,
    terminal: Vec<bool>,
    child_start: Vec<u32>,
    bitmap: Vec<u64>,
    children: Vec<Child>,
}

impl Dawg {
    fn load(path: &str, lang: &'static Language) -> io::Result<Self> {
        Self::from_bytes(fs::read(path)?, lang)
    }

    /// Load and check the file is the kind the caller asked for. A GADDAG
    /// opened as a DAWG finds no words at all, and a DAWG opened as a GADDAG
    /// generates no moves -- both silent failures worth naming.
    fn load_kind(path: &str, lang: &'static Language, want: u8) -> io::Result<Self> {
        let dawg = Self::load(path, lang)?;
        if dawg.kind != want {
            let name = |k: u8| if k == KIND_GADDAG { "GADDAG" } else { "DAWG" };
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("'{path}' is a {}, expected a {}", name(dawg.kind), name(want)),
            ));
        }
        Ok(dawg)
    }

    fn from_bytes(data: Vec<u8>, lang: &'static Language) -> io::Result<Self> {
        let bad = |m: String| io::Error::new(io::ErrorKind::InvalidData, m);
        if data.len() < 20 {
            return Err(bad("file too short".into()));
        }
        if &data[..8] != DAWG_MAGIC {
            return Err(bad(
                "not a SCRBDWG2 lexicon -- rebuild it with `build` / `build-gaddag`".into(),
            ));
        }
        let version = u32::from_le_bytes(data[8..12].try_into().unwrap());
        if version != DAWG_FORMAT_VERSION {
            return Err(bad(format!(
                "lexicon is format version {version}, this engine reads {DAWG_FORMAT_VERSION}"
            )));
        }
        let kind = data[12];
        if kind != KIND_DAWG && kind != KIND_GADDAG {
            return Err(bad(format!("unknown lexicon kind {kind}")));
        }

        // The stamped letters are the ones the lexicon actually contains. Every
        // one must exist in the language being loaded, which is what catches an
        // English dictionary opened as Polish (q/v/x) or the reverse (ą, ć, …).
        let n_letters = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;
        let mut pos = 20usize;
        if data.len() < pos + n_letters * 4 + 8 {
            return Err(bad("truncated header".into()));
        }
        for _ in 0..n_letters {
            let cp = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            pos += 4;
            let ch = char::from_u32(cp)
                .ok_or_else(|| bad(format!("header holds invalid codepoint U+{cp:04X}")))?;
            if lang.letter_index(ch).is_none() {
                return Err(bad(format!(
                    "lexicon uses '{ch}', which is not a letter of '{}' -- \
                     is this dictionary for a different language?",
                    lang.code
                )));
            }
        }

        let root = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        let node_count = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        pos += 8;

        let mut terminal = Vec::with_capacity(node_count);
        let mut child_start = Vec::with_capacity(node_count + 1);
        let mut bitmap = Vec::with_capacity(node_count);
        let mut children: Vec<Child> = Vec::new();

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
                children.push(Child { ch, bit: lang.letter_bit(ch), id: cid });
            }
            // Pack this node's children in symbol-index order (any non-alphabet,
            // non-SEP edge sorts after all known symbols) so the child bitmap and
            // the popcount indexing in `find_child` stay exact.
            children[start..].sort_by_key(|c| lang.symbol_index(c.ch).unwrap_or(u32::MAX));
            let mut bm = 0u64;
            for c in &children[start..] {
                if let Some(sym) = lang.symbol_index(c.ch) {
                    bm |= 1u64 << sym;
                }
            }
            bitmap.push(bm);
        }
        child_start.push(children.len() as u32);

        Ok(Self {
            lang,
            kind,
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
        match self.lang.symbol_index(c) {
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
        let lang = Language::new(code, alphabet, blank, &counts, &points)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self {
            inner: intern_language(lang),
        })
    }

    #[getter]
    fn code(&self) -> &str {
        &self.inner.code
    }

    #[getter]
    fn alphabet(&self) -> String {
        self.inner.letters.iter().collect()
    }

    #[getter]
    fn blank(&self) -> char {
        self.inner.blank
    }

    /// The full tile distribution, in the order a fresh bag holds it.
    fn fresh_tile_bag(&self) -> Vec<char> {
        self.inner.bag.clone()
    }

    /// Tile symbols in the leave-value net's feature order.
    fn eval_alphabet(&self) -> Vec<char> {
        self.inner.eval_letters.clone()
    }

    fn letter_points(&self, letter: char) -> u32 {
        self.inner.points(letter)
    }

    fn __repr__(&self) -> String {
        format!(
            "Language('{}', {} letters, {} tiles)",
            self.inner.code,
            self.inner.letters.len(),
            self.inner.bag.len()
        )
    }
}

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
    /// Load a `lang` DAWG from `path`. If `gaddag_path` is omitted, a sibling
    /// `gaddag.bin` (same directory, `dawg`→`gaddag` in the filename) is loaded
    /// when present; move generation uses it automatically. Passing a path that
    /// does not exist is an error; omitting it and finding no sibling simply
    /// leaves the GADDAG absent (legacy generation).
    #[new]
    #[pyo3(signature = (lang, path, gaddag_path=None))]
    fn new(lang: &LanguagePy, path: &str, gaddag_path: Option<&str>) -> PyResult<Self> {
        let inner = Dawg::load_kind(path, lang.inner, KIND_DAWG)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        let load_gaddag = |gp: &str| {
            Dawg::load_kind(gp, lang.inner, KIND_GADDAG).map_err(|e| PyIOError::new_err(e.to_string()))
        };
        let gaddag = match gaddag_path {
            Some(gp) => Some(load_gaddag(gp)?),
            None => match sibling_gaddag_path(path) {
                Some(gp) if std::path::Path::new(&gp).exists() => Some(load_gaddag(&gp)?),
                _ => None,
            },
        };
        Ok(DawgPy { inner, gaddag })
    }

    /// The language this lexicon was loaded under.
    #[getter]
    fn lang(&self) -> LanguagePy {
        LanguagePy {
            inner: self.inner.lang,
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

/// Shuffle in place (Fisher-Yates) using `rng`. The modulo bias here is on the
/// order of 100/2^64 and not worth a rejection loop.
fn shuffle(tiles: &mut [char], rng: &mut u64) {
    for i in (1..tiles.len()).rev() {
        xorshift(rng);
        let j = (*rng as usize) % (i + 1);
        tiles.swap(i, j);
    }
}

// `skip_from_py_object`: deriving Clone would otherwise opt `Board` into
// by-value extraction from Python. Nothing here wants that -- every Rust-side
// use borrows -- and a silent clone at an FFI boundary is exactly the kind of
// thing that makes a search layer mysteriously slow.
#[pyclass(name = "Board", skip_from_py_object)]
#[derive(Clone)]
struct Board {
    /// The language whose alphabet, points and distribution this position is
    /// played under. A pointer copy on clone, which matters: `Board` is cloned
    /// per endgame node and per rollout ply.
    lang: &'static Language,
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
    fn new(lang: &LanguagePy) -> PyResult<Self> {
        Ok(Self::with_seed(lang.inner, time_seed()))
    }

    /// A fresh board whose bag is shuffled deterministically from `seed`. Two
    /// boards built with the same seed deal the same tiles in the same order,
    /// which is what lets the arena play one bag twice with the seats swapped.
    #[staticmethod]
    fn seeded(lang: &LanguagePy, seed: u64) -> PyResult<Self> {
        Ok(Self::with_seed(lang.inner, seed))
    }

    /// The language this position is played under.
    #[getter]
    fn lang(&self) -> LanguagePy {
        LanguagePy { inner: self.lang }
    }

    /// Deep copy, including the bag and the RNG stream position. The only way
    /// to branch a position without the lossy `str()` -> `from_grid()` round
    /// trip, and the basis of make/unmake search.
    fn copy(&self) -> Board {
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
        let mut tile_bag = lang.inner.bag.clone();
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
                    let consumed = if is_blank { lang.inner.blank } else { ch };
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
            lang: lang.inner,
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

    /// This language's full tile distribution, as `Board()` and
    /// `Board.from_grid()` each start with.
    fn fresh_tile_bag(&self) -> Vec<char> {
        self.lang.bag.clone()
    }

    /// Sum of face point values of a rack (blank tiles score 0, matching
    /// their in-play scoring) -- used for the standard end-of-game scoring
    /// adjustment: the player who goes out gains this value from each
    /// opponent's rack, everyone else loses it from their own.
    fn rack_value(&self, letters: &str) -> u32 {
        letters.chars().map(|c| self.lang.points(c)).sum()
    }

    /// Face point value of a single letter. Blanks score 0 here, same as
    /// their fixed in-play scoring in `calculate_word_points`.
    fn letter_points(&self, letter: char) -> u32 {
        self.lang.points(letter)
    }

    /// Standard rule for who goes first: each player draws one tile, the one
    /// closest to the start of the alphabet goes first, and a blank beats every
    /// letter. Returns the *index* into `draws` of the winner (first index wins
    /// ties). Drawn tiles are not consumed here -- the caller is responsible for
    /// returning them to the bag before dealing real racks.
    fn first_draw_winner(&self, draws: Vec<char>) -> usize {
        draws
            .iter()
            .enumerate()
            .min_by_key(|&(_, &c)| self.lang.rank(c))
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
                        self.lang.points(ch)
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
        self.best_opening_words_inner(&dawg.inner, letters, n)
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
        self.require_same_lang(dawg.inner.lang, "dictionary")?;
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

    /// `(tw_open, dw_open, tl_open, dl_open, board_fill)` for this board --
    /// the same five scalars `smart_player/board_features.encode_board`
    /// computes, straight from the engine's own bonus layout instead of a
    /// transcription of it.
    fn board_feature_scalars(&self) -> [f32; N_BOARD_FEATURES] {
        self.board_features()
    }

    /// Best play with the bag empty, searched rather than sampled.
    ///
    /// Returns `(word, score, (row, col, horizontal), used, differential,
    /// nodes, exact)`. `differential` is the final score margin this play leads
    /// to, in points, counting the end-of-game rack adjustment; `exact` is true
    /// when the search proved it -- false means the node cap or the per-ply
    /// branching limit bit, and the move is good rather than provably optimal.
    /// An empty word means passing beats every play, which does happen when
    /// every legal move opens something worse than it scores.
    ///
    /// `opp_rack` must be the opponent's actual tiles. With the bag empty that
    /// is exactly `unseen_tile_counts(my_rack)`, which is legitimate
    /// information: it follows from the board, your own rack, and the tile
    /// distribution.
    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (dawg, my_rack, opp_rack, max_nodes=300000))]
    fn solve_endgame(
        &self,
        py: Python<'_>,
        dawg: &DawgPy,
        my_rack: &str,
        opp_rack: &str,
        max_nodes: usize,
    ) -> PyResult<(String, u32, (usize, usize, bool), Vec<char>, i32, usize, bool)> {
        self.require_same_lang(dawg.inner.lang, "dictionary")?;
        let Some(gaddag) = &dawg.gaddag else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "solve_endgame needs a GADDAG; this Dawg was loaded without one",
            ));
        };
        let mut board = self.clone();
        let (best, value, stats) = py.detach(|| {
            board.solve_endgame_inner(
                &dawg.inner, gaddag, my_rack, opp_rack, max_nodes, &EG_BRANCH, EG_MAX_DEPTH,
            )
        });
        Ok(match best {
            Some((r, c, horizontal, word, score)) => {
                let used = self.hand_tiles_for_word(&word, r, c, horizontal, my_rack);
                (
                    word,
                    score,
                    (r, c, horizontal),
                    used,
                    value,
                    stats.nodes,
                    !stats.capped && !stats.truncated,
                )
            }
            None => (
                String::new(),
                0,
                (0, 0, true),
                Vec::new(),
                value,
                stats.nodes,
                !stats.capped && !stats.truncated,
            ),
        })
    }

    /// Counts of the tiles neither on the board nor in `rack` -- the bag plus
    /// every opponent's rack -- indexed by `LeaveNet.alphabet()`.
    ///
    /// Legitimate information: a player can see the board and their own rack.
    /// It is only *correct* because the board remembers which squares hold
    /// blanks.
    fn unseen_tile_counts(&self, rack: &str) -> Vec<u8> {
        // Only the prefix this language actually uses: the buffer is sized for
        // the widest alphabet, and callers index it positionally against their
        // own tile list (see `smart_player/player.py`'s opponent-rack
        // reconstruction), so trailing slots would read past the end of it.
        self.unseen_counts(rack)[..self.lang.eval_size()].to_vec()
    }

    /// Rank this rack's plays by Monte-Carlo simulation.
    ///
    /// Returns `(word, score, (row, col, horizontal), used, equity, stderr,
    /// iterations)` per surviving candidate, best equity first. `equity` is the
    /// simulated score differential over the rollout window plus the difference
    /// in what each side is left holding, so it is on the same points scale as
    /// a move's score -- but unlike `score` it accounts for what the move hands
    /// the opponent.
    ///
    /// Candidates are seeded from the *whole* legal move list ranked by static
    /// equity, not the top-n by score. Every candidate in one iteration is
    /// rolled out against the same sampled tiles, and candidates whose interval
    /// falls clear of the leader's stop being sampled.
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
    ) -> PyResult<Vec<(String, u32, (usize, usize, bool), Vec<char>, f64, f64, usize)>> {
        self.require_same_lang(dawg.inner.lang, "dictionary")?;
        // A leave net encodes one slot per tile type, so a net for another
        // language does not merely score badly -- it reads the wrong features.
        self.require_same_lang(net.lang, "leave net")?;
        let Some(gaddag) = &dawg.gaddag else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "simulate needs a GADDAG; this Dawg was loaded without one",
            ));
        };
        let opts = SimOpts {
            candidates,
            iterations,
            batch,
            plies,
            root_eval_cap,
            leave_weight,
            // A fixed default seed would make every position in every game
            // sample the same tile orders; mix in the position so it doesn't.
            seed: if seed == 0 { time_seed() } else { seed },
        };
        // Nothing below touches Python, and a simulation is long enough that
        // holding the GIL through it would stall every other thread.
        let out = py.detach(|| {
            self.simulate_inner(&dawg.inner, gaddag, &net.inner, letters, opts)
                .into_iter()
                .map(|c| {
                    (
                        c.word.clone(),
                        c.score,
                        c.pos,
                        c.used.clone(),
                        c.mean(),
                        c.stderr(),
                        c.n,
                    )
                })
                .collect()
        });
        Ok(out)
    }
}

// Pure Rust methods — no PyO3 overhead, safe to call from rayon threads.
impl Board {
    /// Reject a board/dictionary pairing from different languages.
    ///
    /// Interning makes this a pointer comparison: two `Language` objects built
    /// from the same definition are the same object. Without it, a mismatch
    /// scores every move with the wrong point table instead of failing.
    fn require_same_lang(&self, other: &'static Language, what: &str) -> PyResult<()> {
        if !std::ptr::eq(self.lang, other) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "board is '{}' but the {what} is '{}'",
                self.lang.code, other.code
            )));
        }
        Ok(())
    }

    /// Empty board with a full bag, shuffled once from `seed`.
    fn with_seed(lang: &'static Language, seed: u64) -> Board {
        // 0 is a fixed point of xorshift64 -- it would leave the bag in
        // distribution order forever. Nudged here rather than in the Python
        // constructor so every caller, Rust included, is covered.
        let mut rng = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        let mut tile_bag = lang.bag.clone();
        shuffle(&mut tile_bag, &mut rng);
        Board {
            lang,
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
            self.lang.points(self.board[r][c])
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

    /// `best_opening_words` against a raw `Dawg`, so the simulator can call it
    /// without a `DawgPy` wrapper to hand.
    fn best_opening_words_inner(&self, dawg: &Dawg, letters: &str, n: usize) -> Vec<BestWord> {
        let mut candidates: Vec<BestWord> = Vec::new();
        for word in dawg.search_inner("*", letters) {
            for offset in 0..CENTER {
                if offset >= word.chars().count() {
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
    // Read the edge's cross-check mask straight off the decoded child: it was
    // computed once at load (see `Child.bit`), so there is nothing to redo here.
    for entry in dawg.node_children(node) {
        if entry.bit == 0 {
            continue;
        }
        let mut n = entry.id;
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
            bits |= entry.bit;
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
    /// Held directly rather than reached through `board` -- `score_word` reads
    /// it for every letter of every candidate.
    lang: &'static Language,
    gaddag: &'a Dawg,
    cross: &'a [[u32; BOARD_SIZE]; BOARD_SIZE],
    /// Sum of face values of the perpendicular neighbours at each empty square
    /// (0 where none) — the cross-word's existing contribution, tile-independent.
    cross_score: &'a [[u32; BOARD_SIZE]; BOARD_SIZE],
    /// Whether a cross-word forms at each empty square (has a perpendicular tile).
    has_cross: &'a [[bool; BOARD_SIZE]; BOARD_SIZE],
    /// Count of each real (non-blank) rack letter, indexed by `letter_index`.
    /// Precomputed once per move so `score_word`'s word-order real-vs-blank
    /// allocation is a cheap array copy, not a per-word map allocation. Sized
    /// for the widest alphabet; a smaller language leaves the tail zero.
    rack_counts: [u8; ALPHABET_MAX],
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
                // One indexed load covers both the rack slot and the face value.
                let (idx, value) = self.lang.tab_entry(ch);
                let v = if idx >= 0 && hand[idx as usize] > 0 {
                    hand[idx as usize] -= 1;
                    value as u32
                } else {
                    0
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
        let mut rack_counts = [0u8; ALPHABET_MAX];
        for ch in letters.chars() {
            if let Some(i) = self.lang.letter_index(ch) {
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
                    lang: self.lang,
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

/// Magic + version identifying a lexicon binary. Version 2 added the header;
/// version 1 files carried no identification at all, so a dictionary in the
/// wrong language loaded silently and mis-scored every move.
const DAWG_MAGIC: &[u8; 8] = b"SCRBDWG2";
const DAWG_FORMAT_VERSION: u32 = 2;
const KIND_DAWG: u8 = 0;
const KIND_GADDAG: u8 = 1;

/// The distinct letters a set of entries actually uses, ascending by codepoint,
/// with `SEP` excluded. This is the file's identity: derived from the lexicon
/// itself rather than declared, so it cannot be mislabelled. Codepoint order
/// (not collation order) keeps it a property of the file — collation order
/// belongs to the language, where it drives `first_draw_winner`.
fn lexicon_alphabet(entries: &[&str]) -> Vec<char> {
    let mut seen: Vec<char> = entries
        .iter()
        .flat_map(|w| w.chars())
        .filter(|&c| c != SEP)
        .collect();
    seen.sort_unstable();
    seen.dedup();
    seen
}

fn serialize(arena: &Arena, root: u32, kind: u8, alphabet: &[char]) -> Vec<u8> {
    let n = arena.nodes.len();
    let mut buf: Vec<u8> = Vec::new();
    buf.extend_from_slice(DAWG_MAGIC);
    buf.extend_from_slice(&DAWG_FORMAT_VERSION.to_le_bytes());
    buf.push(kind);
    buf.extend_from_slice(&[0u8; 3]); // reserved
    buf.extend_from_slice(&(alphabet.len() as u32).to_le_bytes());
    for &c in alphabet {
        buf.extend_from_slice(&(c as u32).to_le_bytes());
    }
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
// Leaf evaluation: the leave-value net, run natively
// ---------------------------------------------------------------------------
//
// Simulation evaluates a leave at every rollout ply — hundreds of thousands of
// times per move. Calling back into PyTorch for a 13k-parameter MLP would cost
// far more in FFI and GIL traffic than the ~10k multiply-adds it is asking
// for, so the engine runs the net itself. `smart_player/export_weights.py`
// writes the checkpoint into the format loaded here; the `.pt` stays the
// source of truth.

/// The feature vector's leading slots are per-tile-type counts in
/// `Language::eval_letters` order, which `smart_player/model.py` derives the
/// same way (`sorted(set(board.fresh_tile_bag()))`). Both sides deriving it
/// from the distribution is what keeps them in step -- a mismatch would not
/// fail, it would silently produce confident nonsense.
const N_BOARD_FEATURES: usize = 5;
/// Widest feature vector any supported language produces: one slot per tile
/// type (at most `ALPHABET_MAX` letters plus the blank), the unseen-tile
/// scalar, and the board-context scalars. Encoders write a fixed-size array of
/// this width and hand out the prefix the language actually uses, so the hot
/// path stays allocation-free.
const EVAL_MAX_INPUT_DIM: usize = ALPHABET_MAX + 1 + 1 + N_BOARD_FEATURES;
/// Premium squares of each type on a standard board, in `board_features`
/// order. Matches `_TOTALS` in `smart_player/board_features.py`; asserted
/// against `BONUS_TABLE` in the tests rather than trusted.
const PREMIUM_TOTALS: [f32; 4] = [8.0, 17.0, 12.0, 24.0];
/// Widest hidden layer `LeaveNet::forward` will run without allocating.
const EVAL_MAX_WIDTH: usize = 1024;

struct DenseLayer {
    in_dim: usize,
    out_dim: usize,
    /// Row-major, `out_dim` rows of `in_dim` weights.
    w: Vec<f32>,
    b: Vec<f32>,
}

/// The exported leave-value MLP. ReLU between layers, none on the output,
/// matching `LeaveValueNet.forward`.
pub struct LeaveNet {
    input_dim: usize,
    layers: Vec<DenseLayer>,
}

impl LeaveNet {
    fn load(path: &str, lang: &Language) -> io::Result<Self> {
        let data = fs::read(path)?;
        let bad = |m: &str| io::Error::new(io::ErrorKind::InvalidData, m.to_string());
        if data.len() < 16 || &data[..8] != b"SCRBNET1" {
            return Err(bad("not a SCRBNET1 file (re-run export_weights.py)"));
        }
        let u32_at = |off: usize| -> u32 {
            u32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
        };
        let input_dim = u32_at(8) as usize;
        let n_layers = u32_at(12) as usize;
        // A net trained for another language encodes a different number of tile
        // types, so the width is what catches the mismatch -- the same guard
        // `smart_player/model.py` applies to the `.pt` checkpoint.
        let expected = lang.eval_input_dim();
        if input_dim != expected {
            return Err(bad(&format!(
                "net takes {input_dim} inputs, but '{}' encodes {expected} -- \
                 is this net trained for a different language?",
                lang.code
            )));
        }
        let mut dims = Vec::with_capacity(n_layers);
        let mut off = 16;
        for _ in 0..n_layers {
            if off + 8 > data.len() {
                return Err(bad("truncated layer table"));
            }
            dims.push((u32_at(off) as usize, u32_at(off + 4) as usize));
            off += 8;
        }
        let mut layers = Vec::with_capacity(n_layers);
        for &(in_dim, out_dim) in &dims {
            if out_dim > EVAL_MAX_WIDTH {
                return Err(bad(&format!("layer wider than {EVAL_MAX_WIDTH}")));
            }
            let n_w = in_dim * out_dim;
            let need = (n_w + out_dim) * 4;
            if off + need > data.len() {
                return Err(bad("truncated weights"));
            }
            let mut read_f32s = |count: usize| -> Vec<f32> {
                let v = (0..count)
                    .map(|i| {
                        let p = off + i * 4;
                        f32::from_le_bytes([data[p], data[p + 1], data[p + 2], data[p + 3]])
                    })
                    .collect();
                off += count * 4;
                v
            };
            let w = read_f32s(n_w);
            let b = read_f32s(out_dim);
            layers.push(DenseLayer { in_dim, out_dim, w, b });
        }
        if layers.last().map(|l| l.out_dim) != Some(1) {
            return Err(bad("net must end in a scalar head"));
        }
        Ok(LeaveNet { input_dim, layers })
    }

    /// Predicted leave value for one already-encoded feature vector.
    fn forward(&self, x: &[f32]) -> f32 {
        debug_assert_eq!(x.len(), self.input_dim);
        let mut cur = [0f32; EVAL_MAX_WIDTH];
        let mut next = [0f32; EVAL_MAX_WIDTH];
        cur[..x.len()].copy_from_slice(x);
        let last = self.layers.len() - 1;
        for (li, layer) in self.layers.iter().enumerate() {
            for o in 0..layer.out_dim {
                let row = &layer.w[o * layer.in_dim..(o + 1) * layer.in_dim];
                let mut acc = layer.b[o];
                for (i, &wi) in row.iter().enumerate() {
                    acc += wi * cur[i];
                }
                next[o] = if li == last { acc } else { acc.max(0.0) };
            }
            cur[..layer.out_dim].copy_from_slice(&next[..layer.out_dim]);
        }
        cur[0]
    }
}

/// Feature vector for a leave: per-tile counts, the unseen-tile count scaled
/// by 100, then the board scalars. Mirrors `model.encode_leave`.
///
/// Returns a fixed-width buffer plus the prefix length this language actually
/// uses -- simulation calls this hundreds of thousands of times per move, so it
/// must not allocate.
fn encode_leave_features(
    lang: &Language,
    leave: &str,
    unseen_total: usize,
    feats: &[f32; N_BOARD_FEATURES],
) -> ([f32; EVAL_MAX_INPUT_DIM], usize) {
    let mut x = [0f32; EVAL_MAX_INPUT_DIM];
    for ch in leave.chars() {
        if let Some(i) = lang.eval_index(ch) {
            x[i] += 1.0;
        }
    }
    let n = lang.eval_size();
    x[n] = unseen_total as f32 / 100.0;
    x[n + 1..n + 1 + N_BOARD_FEATURES].copy_from_slice(feats);
    (x, lang.eval_input_dim())
}

/// Encode a leave and run the net over it. Every caller wants exactly this
/// pair, and keeping them together is what guarantees the net only ever sees
/// the prefix of the buffer that its language fills.
#[inline]
fn leave_value(
    net: &LeaveNet,
    lang: &Language,
    leave: &str,
    unseen_total: usize,
    feats: &[f32; N_BOARD_FEATURES],
) -> f32 {
    let (x, dim) = encode_leave_features(lang, leave, unseen_total, feats);
    net.forward(&x[..dim])
}

impl Board {
    /// `(tw_open, dw_open, tl_open, dl_open, board_fill)` — the fraction of
    /// each premium-square type still unclaimed, and how full the board is.
    /// Mirrors `board_features.encode_board`.
    fn board_features(&self) -> [f32; N_BOARD_FEATURES] {
        let mut open = [0u32; 4];
        let mut filled = 0u32;
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.board[r][c] != '-' {
                    filled += 1;
                    continue;
                }
                let (lm, wm) = quadrant_bonus(r, c);
                // Word multipliers outrank letter multipliers, same order as
                // board_features.py's `_classify`.
                if wm == 3 {
                    open[0] += 1;
                } else if wm == 2 {
                    open[1] += 1;
                } else if lm == 3 {
                    open[2] += 1;
                } else if lm == 2 {
                    open[3] += 1;
                }
            }
        }
        [
            open[0] as f32 / PREMIUM_TOTALS[0],
            open[1] as f32 / PREMIUM_TOTALS[1],
            open[2] as f32 / PREMIUM_TOTALS[2],
            open[3] as f32 / PREMIUM_TOTALS[3],
            filled as f32 / (BOARD_SIZE * BOARD_SIZE) as f32,
        ]
    }

    /// Tiles neither on the board nor in `rack`: the bag plus every opponent's
    /// rack. This is exactly what a player may legitimately deduce, and it is
    /// only correct because the board remembers which squares hold blanks — a
    /// blank counts against the `'?'` supply, not against the letter it is
    /// standing in for.
    fn unseen_counts(&self, rack: &str) -> [u8; ALPHABET_MAX + 1] {
        let lang = self.lang;
        let mut counts = [0u8; ALPHABET_MAX + 1];
        for &ch in &lang.bag {
            if let Some(i) = lang.eval_index(ch) {
                counts[i] += 1;
            }
        }
        let take = |ch: char, counts: &mut [u8; ALPHABET_MAX + 1]| {
            if let Some(i) = lang.eval_index(ch) {
                counts[i] = counts[i].saturating_sub(1);
            }
        };
        for r in 0..BOARD_SIZE {
            for c in 0..BOARD_SIZE {
                if self.board[r][c] == '-' {
                    continue;
                }
                let ch = if self.blanks[r][c] {
                    lang.blank
                } else {
                    self.board[r][c]
                };
                take(ch, &mut counts);
            }
        }
        for ch in rack.chars() {
            take(ch, &mut counts);
        }
        counts
    }

    /// Apply a generated move, recording which squares took a blank, and
    /// return the cells it filled. The Rust-side counterpart of the
    /// `place_word` pymethod; pair it with `undo_move` for make/unmake search.
    fn apply_move(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        used: &[char],
    ) -> Vec<(usize, usize)> {
        let mut filled = Vec::new();
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if self.board[r][c] == '-' {
                self.blanks[r][c] = used.get(filled.len()) == Some(&'?');
                filled.push((r, c));
            }
            self.board[r][c] = ch;
        }
        self.first = false;
        filled
    }

    /// Take a move back off the board, given the cells `apply_move` filled.
    fn undo_move(&mut self, filled: &[(usize, usize)], was_first: bool) {
        for &(r, c) in filled {
            self.board[r][c] = '-';
            self.blanks[r][c] = false;
        }
        self.first = was_first;
    }
}

impl Language {
    /// Face value of a rack, as the end-of-game adjustment counts it (blanks 0).
    fn rack_points(&self, rack: &str) -> i32 {
        rack.chars().map(|c| self.points(c) as i32).sum()
    }
}

/// Remove a move's tiles from a rack, blanks included.
fn rack_after(rack: &str, used: &[char]) -> String {
    let mut left: Vec<char> = rack.chars().collect();
    for &ch in used {
        if let Some(p) = left.iter().position(|&x| x == ch) {
            left.remove(p);
        } else if let Some(p) = left.iter().position(|&x| x == '?') {
            left.remove(p);
        }
    }
    left.into_iter().collect()
}

// ---------------------------------------------------------------------------
// Monte-Carlo simulation
// ---------------------------------------------------------------------------
//
// Static evaluation ranks a move by what it scores plus what it leaves behind.
// It cannot see what the move *gives the opponent*. Simulation answers that
// directly: play each leading candidate, deal the opponent a plausible rack
// from the unseen pool, let both sides reply with the static player, and score
// the resulting position. The candidate that survives contact wins.
//
// Two things make this affordable. Common random numbers: every candidate in
// one decision is rolled out against the *same* sampled tile sequences, so the
// comparison isolates the candidate instead of the luck. And sequential
// pruning: candidates whose confidence interval has fallen clear of the
// leader's stop being sampled.

/// How many of a rollout ply's moves get evaluated by the net. The rollout
/// policy does not need to be the strongest possible player, and evaluating
/// every generated move here would cost more than generating them.
const ROLLOUT_EVAL_CAP: usize = 10;

#[derive(Clone, Copy)]
struct SimOpts {
    /// Candidates carried into simulation, by static equity.
    candidates: usize,
    iterations: usize,
    /// Iterations between pruning passes.
    batch: usize,
    /// Half-moves simulated after the candidate, alternating opponent-first.
    ///
    /// Parity matters more than depth. Odd values leave both sides having
    /// played the same number of moves (1 = their reply; 3 = reply, our
    /// follow-up, their reply), so the differential compares like with like.
    /// An even value gives us one more scoring turn than the opponent, which
    /// biases the equity toward whatever sets up our own next move rather than
    /// toward what the candidate concedes.
    plies: usize,
    /// Root moves the net scores, by raw score. Bounds the once-per-decision
    /// cost of ranking a full move list.
    root_eval_cap: usize,
    /// How heavily a leave counts against points actually scored. Applies both
    /// to the rollout policy's own choices and to the leaf's leave difference.
    /// See `DEFAULT_LEAVE_WEIGHT` in smart_player/player.py for why this is a
    /// parameter rather than an implicit 1.0.
    leave_weight: f64,
    seed: u64,
}

/// A candidate and what simulation learned about it.
struct SimCandidate {
    word: String,
    score: u32,
    pos: (usize, usize, bool),
    used: Vec<char>,
    leave: String,
    static_equity: f64,
    n: usize,
    sum: f64,
    sum_sq: f64,
}

impl SimCandidate {
    fn mean(&self) -> f64 {
        if self.n == 0 {
            self.static_equity
        } else {
            self.sum / self.n as f64
        }
    }

    /// Standard error of the mean. `NaN` until there are two samples.
    fn stderr(&self) -> f64 {
        if self.n < 2 {
            return f64::NAN;
        }
        let n = self.n as f64;
        let var = (self.sum_sq - self.sum * self.sum / n) / (n - 1.0);
        (var.max(0.0) / n).sqrt()
    }
}

impl Board {
    /// Best move for `rack` by static equity (score + leave value), or `None`
    /// if there is no legal play. Used as the rollout policy, so it evaluates
    /// only the top `eval_cap` moves by raw score.
    fn best_static_move(
        &self,
        dawg: &Dawg,
        gaddag: &Dawg,
        net: &LeaveNet,
        rack: &str,
        eval_cap: usize,
        leave_weight: f64,
    ) -> Option<GenMove> {
        let mut moves = self.gaddag_generate(dawg, gaddag, rack, false, true);
        if moves.is_empty() {
            return None;
        }
        if moves.len() > eval_cap {
            moves.sort_unstable_by_key(|m| std::cmp::Reverse(m.4));
            moves.truncate(eval_cap);
        }
        let feats = self.board_features();
        let unseen_total: usize = self
            .unseen_counts(rack)
            .iter()
            .map(|&c| c as usize)
            .sum();
        let mut best_i = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        for (i, m) in moves.iter().enumerate() {
            let used = self.hand_tiles_for_word(&m.3, m.0, m.1, m.2, rack);
            let leave = rack_after(rack, &used);
            let v = m.4 as f64
                + leave_weight * leave_value(net, self.lang, &leave, unseen_total, &feats) as f64;
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        Some(moves.swap_remove(best_i))
    }

    /// One rollout of one candidate against one sampled tile sequence.
    ///
    /// `pool` is the unseen tiles already shuffled: the first `RACK_SIZE` are
    /// the opponent's rack, the rest are the bag both sides draw from. Sharing
    /// one shuffle across every candidate in an iteration is the common-random-
    /// numbers trick that makes the comparison between them meaningful at a
    /// few hundred iterations rather than a few thousand.
    fn rollout(
        &self,
        dawg: &Dawg,
        gaddag: &Dawg,
        net: &LeaveNet,
        cand: &SimCandidate,
        pool: &[char],
        plies: usize,
        leave_weight: f64,
    ) -> f64 {
        let mut board = self.clone();
        board.apply_move(&cand.word, cand.pos.0, cand.pos.1, cand.pos.2, &cand.used);

        let split = RACK_SIZE.min(pool.len());
        let mut opp_rack: String = pool[..split].iter().collect();
        let mut cursor = split;
        let draw = |rack: &mut String, cursor: &mut usize| {
            while rack.chars().count() < RACK_SIZE && *cursor < pool.len() {
                rack.push(pool[*cursor]);
                *cursor += 1;
            }
        };

        let mut my_rack = cand.leave.clone();
        draw(&mut my_rack, &mut cursor);

        let mut my_gain = cand.score as i64;
        let mut opp_gain = 0i64;
        let mut opponents_turn = true;

        for _ in 0..plies {
            let rack = if opponents_turn {
                opp_rack.clone()
            } else {
                my_rack.clone()
            };
            if let Some(m) =
                board.best_static_move(dawg, gaddag, net, &rack, ROLLOUT_EVAL_CAP, leave_weight)
            {
                let used = board.hand_tiles_for_word(&m.3, m.0, m.1, m.2, &rack);
                let left = rack_after(&rack, &used);
                board.apply_move(&m.3, m.0, m.1, m.2, &used);
                if opponents_turn {
                    opp_gain += m.4 as i64;
                    opp_rack = left;
                    draw(&mut opp_rack, &mut cursor);
                } else {
                    my_gain += m.4 as i64;
                    my_rack = left;
                    draw(&mut my_rack, &mut cursor);
                }
            }
            opponents_turn = !opponents_turn;
        }

        // Score differential over the simulated window, plus the difference in
        // what each side is left holding. Both leaves are valued on the final
        // board with the tiles still unaccounted for.
        let feats = board.board_features();
        let unseen_left = pool.len().saturating_sub(cursor);
        let my_leave_v = leave_value(net, board.lang, &my_rack, unseen_left, &feats) as f64;
        let opp_leave_v = leave_value(net, board.lang, &opp_rack, unseen_left, &feats) as f64;
        (my_gain - opp_gain) as f64 + leave_weight * (my_leave_v - opp_leave_v)
    }

    /// Rank `rack`'s plays by simulation. Returns the surviving candidates,
    /// best first, each with its simulated equity and standard error.
    fn simulate_inner(
        &self,
        dawg: &Dawg,
        gaddag: &Dawg,
        net: &LeaveNet,
        rack: &str,
        opts: SimOpts,
    ) -> Vec<SimCandidate> {
        // Rank the *whole* legal move list statically, not the top-50 by score:
        // a tile dump or a blocking play can be worth more than it scores.
        let mut moves = if self.first {
            Vec::new()
        } else {
            self.gaddag_generate(dawg, gaddag, rack, true, true)
        };
        if self.first {
            // The opening has no anchors to generate from; reuse the dedicated
            // centre-covering path and convert to the same shape.
            for (word, score, (r, c, h), _used) in
                self.best_opening_words_inner(dawg, rack, usize::MAX)
            {
                moves.push((r, c, h, word, score));
            }
        }
        if moves.is_empty() {
            return Vec::new();
        }
        if moves.len() > opts.root_eval_cap {
            moves.sort_unstable_by_key(|m| std::cmp::Reverse(m.4));
            moves.truncate(opts.root_eval_cap);
        }

        let feats = self.board_features();
        let unseen = self.unseen_counts(rack);
        let unseen_total: usize = unseen.iter().map(|&c| c as usize).sum();

        let mut cands: Vec<SimCandidate> = moves
            .into_iter()
            .map(|(r, c, h, word, score)| {
                let used = self.hand_tiles_for_word(&word, r, c, h, rack);
                let leave = rack_after(rack, &used);
                let static_equity = score as f64
                    + opts.leave_weight
                        * leave_value(net, self.lang, &leave, unseen_total, &feats) as f64;
                SimCandidate {
                    word,
                    score,
                    pos: (r, c, h),
                    used,
                    leave,
                    static_equity,
                    n: 0,
                    sum: 0.0,
                    sum_sq: 0.0,
                }
            })
            .collect();
        cands.sort_by(|a, b| b.static_equity.partial_cmp(&a.static_equity).unwrap());
        cands.truncate(opts.candidates.max(1));

        if cands.len() == 1 || opts.iterations == 0 || opts.plies == 0 {
            return cands;
        }

        // The pool a rollout deals from: every tile neither on the board nor in
        // our own rack.
        let mut pool: Vec<char> = Vec::with_capacity(unseen_total);
        // Only this language's slots: `unseen_counts` returns a buffer sized for
        // the widest alphabet, and the tail is not addressable in a narrower one.
        for (i, &count) in unseen.iter().take(self.lang.eval_size()).enumerate() {
            let ch = self.lang.eval_letters[i];
            for _ in 0..count {
                pool.push(ch);
            }
        }
        if pool.is_empty() {
            return cands;
        }

        let mut done = 0usize;
        while done < opts.iterations && cands.len() > 1 {
            let batch = opts.batch.max(1).min(opts.iterations - done);
            // Each iteration gets its own shuffle, shared by every candidate in
            // it, and derived from the seed so a run is reproducible.
            let results: Vec<Vec<f64>> = gen_pool().install(|| {
                (done..done + batch)
                    .into_par_iter()
                    .map(|iter| {
                        let mut rng = splitmix64(opts.seed ^ (iter as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                        let mut shuffled = pool.clone();
                        shuffle(&mut shuffled, &mut rng);
                        cands
                            .iter()
                            .map(|c| {
                                self.rollout(
                                    dawg, gaddag, net, c, &shuffled, opts.plies, opts.leave_weight,
                                )
                            })
                            .collect()
                    })
                    .collect()
            });
            for row in results {
                for (c, v) in cands.iter_mut().zip(row) {
                    c.n += 1;
                    c.sum += v;
                    c.sum_sq += v * v;
                }
            }
            done += batch;

            // Sequential pruning: drop anything whose interval has fallen clear
            // of the leader's. Sort by mean first so the "keep at least two"
            // floor keeps the two *best* rather than the first two encountered.
            //
            // Skipped once there are no iterations left to save: pruning after
            // the final batch discards measurements already paid for without
            // avoiding any work. Callers that want every candidate scored --
            // `smart_player/distill.py` harvests them all as training labels --
            // get that by setting `batch` to `iterations`.
            if done < opts.iterations && cands.len() > 2 {
                cands.sort_by(|a, b| b.mean().partial_cmp(&a.mean()).unwrap());
                let (lm, lse) = (cands[0].mean(), cands[0].stderr());
                if lse.is_finite() {
                    let floor = lm - 2.0 * lse;
                    let mut kept: Vec<SimCandidate> = Vec::new();
                    for c in cands.into_iter() {
                        let se = c.stderr();
                        if kept.len() < 2 || !se.is_finite() || c.mean() + 2.0 * se >= floor {
                            kept.push(c);
                        }
                    }
                    cands = kept;
                }
            }
        }

        cands.sort_by(|a, b| b.mean().partial_cmp(&a.mean()).unwrap());
        cands
    }
}

/// The leave-value net, loaded from `export_weights.py`'s output.
#[pyclass(name = "LeaveNet")]
struct LeaveNetPy {
    inner: LeaveNet,
    lang: &'static Language,
}

#[pymethods]
impl LeaveNetPy {
    #[new]
    fn new(lang: &LanguagePy, path: &str) -> PyResult<Self> {
        LeaveNet::load(path, lang.inner)
            .map(|inner| LeaveNetPy {
                inner,
                lang: lang.inner,
            })
            .map_err(|e| PyIOError::new_err(format!("{path}: {e}")))
    }

    /// Forward pass on an already-encoded feature vector. Exists so a test can
    /// check the engine's arithmetic against PyTorch's on identical input.
    fn raw(&self, x: Vec<f32>) -> PyResult<f32> {
        if x.len() != self.inner.input_dim {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} features, got {}",
                self.inner.input_dim,
                x.len()
            )));
        }
        Ok(self.inner.forward(&x))
    }

    /// Predicted value of holding `leave`, in `board`'s context, for a player
    /// whose full rack is `rack` (which fixes the unseen-tile count).
    fn value(&self, board: &Board, leave: &str, rack: &str) -> f32 {
        let feats = board.board_features();
        let unseen: usize = board.unseen_counts(rack).iter().map(|&c| c as usize).sum();
        leave_value(&self.inner, board.lang, leave, unseen, &feats)
    }

    /// The tile-symbol order the feature vector's leading slots use, so a test
    /// can assert it matches `model.ALPHABET`.
    fn alphabet(&self) -> Vec<char> {
        self.lang.eval_letters.clone()
    }
}

// ---------------------------------------------------------------------------
// Endgame: search once the bag is empty
// ---------------------------------------------------------------------------
//
// With an empty bag the game stops being a game of chance. Nobody draws again,
// so the opponent's rack is exactly `unseen - my rack` — computable only
// because the board now remembers which of its squares hold blanks. What is
// left is a small, deterministic, perfect-information, zero-sum game, and the
// right tool is search, not sampling: simulation is at its least useful here
// because there is nothing left to sample.
//
// This matters out of proportion to the number of turns involved. Endgames are
// where close games are decided, and close games are exactly the ones that
// convert score margin into win rate.

/// How many moves to consider at each depth. Endgames are shallow — at most
/// seven tiles a side — but an open board can still offer hundreds of plays at
/// the first ply, and the raw tree is far too wide to enumerate. Alpha-beta
/// with out-plays-first ordering does most of the work; this bounds the rest.
const EG_BRANCH: [usize; 8] = [20, 14, 10, 8, 6, 5, 4, 4];
/// Plies to search before falling back to the both-players-stop value. Racks
/// hold at most seven tiles, so this reaches the natural end of most endgames;
/// bounding by depth rather than by a node budget matters because a node cap
/// bites part-way through the tree, leaving whichever branches happened to be
/// searched first with a deeper look than the rest.
const EG_MAX_DEPTH: usize = 8;
/// Branching past the end of `EG_BRANCH`, where racks are nearly spent.
const EG_BRANCH_TAIL: usize = 5;

/// Per-depth branching limits. Passing a slice of huge values makes the search
/// exhaustive, which is how the tests prove alpha-beta agrees with brute force.
fn eg_branch_at(limits: &[usize], depth: usize) -> usize {
    *limits.get(depth).unwrap_or(&EG_BRANCH_TAIL)
}

/// Whether a search finished inside its limits, and what it cost.
struct EndgameStats {
    nodes: usize,
    /// The node backstop bit -- the tree was cut unevenly and the value is not
    /// trustworthy. Should be rare; if it fires, the limits are too loose.
    capped: bool,
    /// The depth horizon or a branching limit bit. Normal on a wide endgame:
    /// the answer is a strong move rather than a proven one.
    truncated: bool,
}

impl Board {
    /// Negamax value of the position for the side to move: the score
    /// differential (mover minus opponent) achievable from here on, assuming
    /// both sides play well.
    ///
    /// `passes` counts consecutive no-plays; two in a row ends the game with
    /// each side deducting its own rack.
    #[allow(clippy::too_many_arguments)]
    fn endgame_value(
        &mut self,
        dawg: &Dawg,
        gaddag: &Dawg,
        me: &str,
        them: &str,
        passes: u32,
        mut alpha: i32,
        beta: i32,
        depth: usize,
        stats: &mut EndgameStats,
        max_nodes: usize,
        limits: &[usize],
        max_depth: usize,
    ) -> i32 {
        // Both sides gave up: each deducts its own rack.
        if passes >= 2 {
            return self.lang.rack_points(them) - self.lang.rack_points(me);
        }
        // Horizon, or the emergency node backstop. Both fall back to the value
        // of nobody playing again -- pessimistic, but a real outcome rather
        // than a guess, and applied to both sides alike.
        if depth >= max_depth {
            stats.truncated = true;
            return self.lang.rack_points(them) - self.lang.rack_points(me);
        }
        stats.nodes += 1;
        if stats.nodes > max_nodes {
            stats.capped = true;
            return self.lang.rack_points(them) - self.lang.rack_points(me);
        }

        // Passing is always legal, and occasionally right — declining to open
        // a lane can be worth more than the points on offer.
        let mut best = -self.endgame_value(
            dawg, gaddag, them, me, passes + 1, -beta, -alpha, depth + 1, stats, max_nodes,
            limits, max_depth,
        );
        alpha = alpha.max(best);
        if alpha >= beta {
            return best;
        }

        let mut moves = self.gaddag_generate(dawg, gaddag, me, false, true);
        if moves.is_empty() {
            return best;
        }
        // Order by score, but float plays that go out to the front: ending the
        // game collects the opponent's whole rack, so they are both the most
        // likely best move and the best source of cutoffs.
        let me_len = me.chars().count();
        let out_len = |m: &GenMove| -> bool {
            self.hand_tiles_for_word(&m.3, m.0, m.1, m.2, me).len() == me_len
        };
        moves.sort_by_key(|m| (std::cmp::Reverse(out_len(m)), std::cmp::Reverse(m.4)));

        let limit = eg_branch_at(limits, depth);
        if moves.len() > limit {
            stats.truncated = true;
            moves.truncate(limit);
        }

        let was_first = self.first;
        for m in moves {
            let (r, c, horizontal, ref word, score) = m;
            let used = self.hand_tiles_for_word(word, r, c, horizontal, me);
            let left = rack_after(me, &used);
            let filled = self.apply_move(word, r, c, horizontal, &used);

            let value = if left.is_empty() {
                // Going out ends the game and collects the opponent's rack.
                score as i32 + self.lang.rack_points(them)
            } else {
                score as i32
                    - self.endgame_value(
                        dawg, gaddag, them, &left, 0, -beta, -alpha, depth + 1, stats, max_nodes,
                        limits, max_depth,
                    )
            };

            self.undo_move(&filled, was_first);

            if value > best {
                best = value;
            }
            alpha = alpha.max(best);
            if alpha >= beta {
                break;
            }
        }
        best
    }

    /// Best play for `me` with the bag empty, as
    /// `(move, differential, stats)`. An empty word means passing is best.
    fn solve_endgame_inner(
        &mut self,
        dawg: &Dawg,
        gaddag: &Dawg,
        me: &str,
        them: &str,
        max_nodes: usize,
        limits: &[usize],
        max_depth: usize,
    ) -> (Option<GenMove>, i32, EndgameStats) {
        let mut stats = EndgameStats { nodes: 0, capped: false, truncated: false };
        // Passing is the baseline every play has to beat. Usually trivially
        // beaten, but not always: declining to open the board can be worth
        // more than the points on offer.
        let mut best_value = -self.endgame_value(
            dawg,
            gaddag,
            them,
            me,
            1,
            i32::MIN + 1,
            i32::MAX - 1,
            1,
            &mut stats,
            max_nodes,
            limits,
            max_depth,
        );
        let mut best_move: Option<GenMove> = None;

        let mut moves = self.gaddag_generate(dawg, gaddag, me, false, true);
        let me_len = me.chars().count();
        let out_len = |m: &GenMove| -> bool {
            self.hand_tiles_for_word(&m.3, m.0, m.1, m.2, me).len() == me_len
        };
        moves.sort_by_key(|m| (std::cmp::Reverse(out_len(m)), std::cmp::Reverse(m.4)));
        let limit = eg_branch_at(limits, 0);
        if moves.len() > limit {
            stats.truncated = true;
            moves.truncate(limit);
        }

        let was_first = self.first;
        for m in moves {
            let (r, c, horizontal, ref word, score) = m;
            let used = self.hand_tiles_for_word(word, r, c, horizontal, me);
            let left = rack_after(me, &used);
            let filled = self.apply_move(word, r, c, horizontal, &used);

            let value = if left.is_empty() {
                score as i32 + self.lang.rack_points(them)
            } else {
                score as i32
                    - self.endgame_value(
                        dawg,
                        gaddag,
                        them,
                        &left,
                        0,
                        i32::MIN + 1,
                        i32::MAX - 1,
                        1,
                        &mut stats,
                        max_nodes,
                        limits,
                        max_depth,
                    )
            };
            self.undo_move(&filled, was_first);

            if value > best_value {
                best_value = value;
                best_move = Some(m);
            }
        }
        (best_move, best_value, stats)
    }
}

/// Scatter a counter into unrelated 64-bit values (SplitMix64's finalizer).
fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

// ---------------------------------------------------------------------------
// CLI entry points
// ---------------------------------------------------------------------------

fn usage(prog: &str) {
    eprintln!(
        "Usage:\n  {prog} build         <words.txt>     <dawg.bin>\n  \
                    {prog} build-gaddag  <words.txt>     <gaddag.bin>\n  \
                    {prog} lookup        <lang> <dawg.bin>  <word>\n  \
                    {prog} bench         <lang> <dawg.bin>  <words.txt>\n  \
                    {prog} gen-verify    <lang> <dawg.bin>  <gaddag.bin>  [games]\n  \
                    {prog} gen-bench     <lang> <dawg.bin>  <gaddag.bin>  [games]\n\n\
         <lang> is a language code with a definition in languages/<lang>.json\n\
         (e.g. `pl`), searched for in the current directory and its parents."
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

    let data = serialize(&arena, root, KIND_DAWG, &lexicon_alphabet(&words));
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

    let data = serialize(&arena, root, KIND_GADDAG, &lexicon_alphabet(&words));
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

fn cmd_lookup(lang: &'static Language, dawg_path: &str, word: &str) -> io::Result<()> {
    let dawg = Dawg::load_kind(dawg_path, lang, KIND_DAWG)?;
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

fn cmd_bench(lang: &'static Language, dawg_path: &str, words_path: &str) -> io::Result<()> {
    let dawg = Dawg::load_kind(dawg_path, lang, KIND_DAWG)?;
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
fn cmd_gen_verify(
    lang: &'static Language,
    dawg_path: &str,
    gaddag_path: &str,
    games: usize,
) -> io::Result<()> {
    let dawg = Dawg::load_kind(dawg_path, lang, KIND_DAWG)?;
    let gaddag = Dawg::load_kind(gaddag_path, lang, KIND_GADDAG)?;
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
        let mut board = Board::with_seed(lang, time_seed());
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
fn cmd_gen_bench(
    lang: &'static Language,
    dawg_path: &str,
    gaddag_path: &str,
    games: usize,
) -> io::Result<()> {
    let dawg = Dawg::load_kind(dawg_path, lang, KIND_DAWG)?;
    let gaddag = Dawg::load_kind(gaddag_path, lang, KIND_GADDAG)?;
    let dpy = DawgPy {
        inner: dawg,
        gaddag: Some(gaddag),
    };
    let gad = dpy.gaddag.as_ref().unwrap();

    let mut positions = 0usize;
    let mut leg_secs = 0.0f64;
    let mut gad_secs = 0.0f64;
    for _ in 0..games {
        let mut board = Board::with_seed(lang, time_seed());
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
    // Every command that reads a `.bin` needs the alphabet to decode its edges,
    // and the scoring commands need the point table too, so they take a language
    // code up front. `build`/`build-gaddag` derive everything from the word list
    // they are given and take none.
    let lang = |code: &str| -> io::Result<&'static Language> {
        load_language(code).map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))
    };
    match args.get(1).map(String::as_str) {
        Some("build") if args.len() == 4 => cmd_build(&args[2], &args[3]),
        Some("build-gaddag") if args.len() == 4 => cmd_build_gaddag(&args[2], &args[3]),
        Some("lookup") if args.len() == 5 => cmd_lookup(lang(&args[2])?, &args[3], &args[4]),
        Some("bench") if args.len() == 5 => cmd_bench(lang(&args[2])?, &args[3], &args[4]),
        Some("gen-verify") if args.len() == 5 || args.len() == 6 => {
            let games = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);
            cmd_gen_verify(lang(&args[2])?, &args[3], &args[4], games)
        }
        Some("gen-bench") if args.len() == 5 || args.len() == 6 => {
            let games = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(200);
            cmd_gen_bench(lang(&args[2])?, &args[3], &args[4], games)
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
    m.add_class::<LanguagePy>()?;
    m.add_class::<DawgPy>()?;
    m.add_class::<Board>()?;
    m.add_class::<LeaveNetPy>()?;
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

    /// The real Polish definition, read from `languages/pl.json` -- so these
    /// tests also fail if that file stops describing the language the engine
    /// used to have compiled in.
    fn pl() -> &'static Language {
        load_language("pl").expect("languages/pl.json")
    }

    /// A minimal English definition, for exercising the 26-letter path. Counts
    /// and points are the standard set.
    fn en() -> &'static Language {
        let counts = [
            ('a', 9), ('b', 2), ('c', 2), ('d', 4), ('e', 12), ('f', 2), ('g', 3), ('h', 2),
            ('i', 9), ('j', 1), ('k', 1), ('l', 4), ('m', 2), ('n', 6), ('o', 8), ('p', 2),
            ('q', 1), ('r', 6), ('s', 4), ('t', 6), ('u', 4), ('v', 2), ('w', 2), ('x', 1),
            ('y', 2), ('z', 1), ('?', 2),
        ];
        let points = [
            ('a', 1), ('b', 3), ('c', 3), ('d', 2), ('e', 1), ('f', 4), ('g', 2), ('h', 4),
            ('i', 1), ('j', 8), ('k', 5), ('l', 1), ('m', 3), ('n', 1), ('o', 1), ('p', 3),
            ('q', 10), ('r', 1), ('s', 1), ('t', 1), ('u', 1), ('v', 4), ('w', 4), ('x', 8),
            ('y', 4), ('z', 10), ('?', 0),
        ];
        intern_language(
            Language::new(
                "en",
                "abcdefghijklmnopqrstuvwxyz",
                '?',
                &counts.into_iter().collect(),
                &points.into_iter().collect(),
            )
            .expect("english definition"),
        )
    }

    /// Build a flat DAWG/GADDAG in memory from the given entries (any strings,
    /// including GADDAG entries containing `SEP`), via the real build pipeline.
    fn compile_in(lang: &'static Language, entries: &[&str]) -> Dawg {
        // A GADDAG's entries contain SEP, which `lexicon_alphabet` filters out;
        // the kind byte follows from that so round-trips carry a truthful header.
        let kind = if entries.iter().any(|e| e.contains(SEP)) {
            KIND_GADDAG
        } else {
            KIND_DAWG
        };
        let mut ws: Vec<&str> = entries.to_vec();
        ws.sort_unstable();
        ws.dedup();
        let (arena, root, _) = build_dawg(&ws);
        let (arena, root) = compact(&arena, root);
        let bytes = serialize(&arena, root, kind, &lexicon_alphabet(&ws));
        Dawg::from_bytes(bytes, lang).unwrap()
    }

    fn compile_gaddag_in(lang: &'static Language, words: &[&str]) -> Dawg {
        let mut entries = Vec::new();
        for w in words {
            gaddag_entries(w, &mut entries);
        }
        let refs: Vec<&str> = entries.iter().map(String::as_str).collect();
        compile_in(lang, &refs)
    }

    /// The data-driven Polish definition must reproduce, exactly, the tables
    /// that used to be `const`s in this file. Pasted here verbatim from the
    /// pre-refactor source: if `languages/pl.json` ever disagrees with them,
    /// the engine's behaviour has changed and this is where it shows up.
    #[test]
    fn polish_language_matches_the_old_constants() {
        let lang = pl();

        const OLD_ALPHABET: &str = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż";
        assert_eq!(lang.letters.iter().collect::<String>(), OLD_ALPHABET);

        // The old `fresh_tile_bag()` literal, in its original order -- boards
        // are seeded by shuffling this, so the order is observable.
        let old_bag: Vec<char> = vec![
            'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'ą', 'b', 'b', 'c', 'c', 'c', 'ć', 'd',
            'd', 'd', 'e', 'e', 'e', 'e', 'e', 'e', 'e', 'ę', 'f', 'g', 'g', 'h', 'h', 'i', 'i',
            'i', 'i', 'i', 'i', 'i', 'i', 'j', 'j', 'k', 'k', 'k', 'l', 'l', 'l', 'ł', 'ł', 'm',
            'm', 'm', 'n', 'n', 'n', 'n', 'n', 'ń', 'o', 'o', 'o', 'o', 'o', 'o', 'ó', 'p', 'p',
            'p', 'r', 'r', 'r', 'r', 's', 's', 's', 's', 'ś', 't', 't', 't', 'u', 'u', 'w', 'w',
            'w', 'w', 'y', 'y', 'y', 'y', 'z', 'z', 'z', 'z', 'z', 'ź', 'ż', '?', '?',
        ];
        assert_eq!(lang.bag, old_bag);

        // The old `letter_points` match arms, which uppercased before matching.
        let old_points = |c: char| -> u32 {
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
        };
        for c in OLD_ALPHABET.chars() {
            assert_eq!(lang.points(c), old_points(c), "points for '{c}'");
            let upper = c.to_uppercase().next().unwrap();
            assert_eq!(lang.points(upper), old_points(upper), "points for '{upper}'");
        }
        assert_eq!(lang.points('?'), 0);

        // The old `alphabet_rank`: blank first, unknown characters last.
        assert_eq!(lang.rank('?'), -1);
        assert_eq!(lang.rank('a'), 0);
        assert_eq!(lang.rank('ą'), 1);
        assert_eq!(lang.rank('ż'), 31);
        assert_eq!(lang.rank('q'), i32::MAX);
    }

    /// The leave-value net's feature order was a hand-written constant; it is
    /// now derived from the distribution. The four committed checkpoints were
    /// trained against the old string, so it must come out identical.
    #[test]
    fn eval_alphabet_is_derived_from_the_bag() {
        const OLD_EVAL_ALPHABET: &str = "?abcdefghijklmnoprstuwyzóąćęłńśźż";
        let lang = pl();
        assert_eq!(lang.eval_letters.iter().collect::<String>(), OLD_EVAL_ALPHABET);
        assert_eq!(lang.eval_size(), 33);
        assert_eq!(lang.eval_input_dim(), 39, "the committed nets take 39 inputs");
    }

    /// Seeded deals must be byte-identical to what the compiled-in bag produced,
    /// or every reproducible benchmark and every paired-seed arena result from
    /// before the refactor silently stops comparing.
    #[test]
    fn seeded_deals_match_the_pre_refactor_engine() {
        // Captured from the engine before the alphabet became runtime data.
        for (seed, expected) in [
            (1u64, "emrżewoyhpenarcćijrm"),
            (42, "peoiokbpnmreaźdąmęaa"),
            (12345, "azhnłsrichrodpidbpkm"),
        ] {
            let board = Board::with_seed(pl(), seed);
            let got: String = board.tile_bag.iter().take(20).collect();
            assert_eq!(got, expected, "seed {seed}");
        }
    }

    /// A 26-letter language must work end to end, including the bits sized by
    /// `ALPHABET_MAX` -- the cross-check bitset, `rack_counts`, and the leave
    /// encoder's fixed-width buffer.
    #[test]
    fn english_generates_and_scores() {
        let lang = en();
        assert_eq!(lang.bag.len(), 100, "a standard set is 100 tiles");
        assert_eq!(
            lang.bag.iter().map(|&c| lang.points(c)).sum::<u32>(),
            187,
            "standard English face-value total"
        );
        assert_eq!(lang.eval_input_dim(), 27 + 1 + N_BOARD_FEATURES);

        let words = ["cat", "cats", "at", "as", "a"];
        let dawg = compile_in(lang, &words);
        let gaddag = compile_gaddag_in(lang, &words);

        // Cross-checks: after "ca", only 't' completes a word.
        let bits = cross_bits(&dawg, &['c', 'a'], &[]);
        assert_ne!(bits & lang.letter_bit('t'), 0);
        assert_eq!(bits & lang.letter_bit('q'), 0);

        let mut board = Board::with_seed(lang, 7);
        board.place_word("cat", 7, 7, true, None).unwrap();
        board.first = false;
        let moves = board.gaddag_best_words(&dawg, &gaddag, "s", 5, false);
        assert!(
            moves.iter().any(|(w, ..)| w == "cats"),
            "expected the 'cats' hook, got {:?}",
            moves.iter().map(|m| &m.0).collect::<Vec<_>>()
        );
    }

    /// A lexicon stamps the letters it actually uses, so opening it under a
    /// language that lacks one of them fails loudly. Before the header existed
    /// this loaded fine and mis-scored every move.
    #[test]
    fn a_lexicon_is_rejected_by_the_wrong_language() {
        let raw = |lang: &'static Language, words: &[&str]| {
            let mut ws: Vec<&str> = words.to_vec();
            ws.sort_unstable();
            let (arena, root, _) = build_dawg(&ws);
            let (arena, root) = compact(&arena, root);
            let _ = lang;
            serialize(&arena, root, KIND_DAWG, &lexicon_alphabet(&ws))
        };

        // "quiz" needs q and z; Polish has no q.
        let english_bytes = raw(en(), &["quiz", "cat"]);
        let err = Dawg::from_bytes(english_bytes.clone(), pl())
            .err()
            .expect("polish must reject an english lexicon");
        assert!(err.to_string().contains("'q'"), "got: {err}");
        assert!(Dawg::from_bytes(english_bytes, en()).is_ok());

        // "kość" needs ć and ś; English has neither.
        let polish_bytes = raw(pl(), &["kość", "kot"]);
        let err = Dawg::from_bytes(polish_bytes.clone(), en())
            .err()
            .expect("english must reject a polish lexicon");
        assert!(err.to_string().contains("different language"), "got: {err}");
        assert!(Dawg::from_bytes(polish_bytes, pl()).is_ok());
    }

    /// A DAWG and a GADDAG are both lexicons but are not interchangeable:
    /// swapping them finds no words / generates no moves, silently.
    #[test]
    fn a_gaddag_is_not_accepted_where_a_dawg_is_wanted() {
        let lang = pl();
        let dawg_bytes = {
            let ws = vec!["kot", "kota"];
            let (arena, root, _) = build_dawg(&ws);
            let (arena, root) = compact(&arena, root);
            serialize(&arena, root, KIND_DAWG, &lexicon_alphabet(&ws))
        };
        let loaded = Dawg::from_bytes(dawg_bytes, lang).unwrap();
        assert_eq!(loaded.kind, KIND_DAWG);
        assert_eq!(compile_gaddag_in(lang, &["kot"]).kind, KIND_GADDAG);
    }

    /// A version-1 file (no magic, no header) must be refused with a message
    /// that says what to do, not decoded as garbage.
    #[test]
    fn an_unstamped_lexicon_is_refused() {
        let mut legacy = Vec::new();
        legacy.extend_from_slice(&0u32.to_le_bytes()); // root
        legacy.extend_from_slice(&1u32.to_le_bytes()); // node count
        legacy.push(1);
        legacy.extend_from_slice(&0u32.to_le_bytes());
        legacy.resize(64, 0);
        let err = Dawg::from_bytes(legacy, pl())
            .err()
            .expect("an unstamped file must be refused");
        assert!(err.to_string().contains("SCRBDWG2"), "got: {err}");
    }

    /// `unseen_counts` writes into a buffer sized for the widest alphabet, but
    /// callers index the result positionally against their own tile list -- so
    /// a language with fewer tile types must get a correspondingly shorter one.
    #[test]
    fn unseen_counts_are_sized_to_the_language() {
        for lang in [pl(), en()] {
            let board = Board::with_seed(lang, 5);
            let counts = board.unseen_counts("");
            let used = &counts[..lang.eval_size()];
            // Nothing may be recorded past the language's own tile types.
            assert!(
                counts[lang.eval_size()..].iter().all(|&c| c == 0),
                "{}: counts spill past {} slots",
                lang.code,
                lang.eval_size()
            );
            // A fresh board with no rack leaves the whole bag unseen.
            assert_eq!(
                used.iter().map(|&c| c as usize).sum::<usize>(),
                lang.bag.len(),
                "{}",
                lang.code
            );
        }
    }

    /// The rollout pool is built by walking `unseen_counts` positionally against
    /// the language's tile list. That buffer is sized for the widest alphabet,
    /// so a narrower language panicked here until the walk was bounded.
    #[test]
    fn rollout_pool_is_bounded_by_the_language() {
        for lang in [pl(), en()] {
            let board = Board::with_seed(lang, 11);
            let unseen = board.unseen_counts("");
            let mut pool = 0usize;
            for (i, &count) in unseen.iter().take(lang.eval_size()).enumerate() {
                // The indexing the simulator does; out of range would panic.
                let _ = lang.eval_letters[i];
                pool += count as usize;
            }
            assert_eq!(pool, lang.bag.len(), "{}", lang.code);
        }
    }

    /// Two boards built from the same definition share one interned language,
    /// which is what makes pointer equality a valid identity check.
    #[test]
    fn identical_definitions_are_interned_once() {
        assert!(std::ptr::eq(pl(), pl()));
        assert!(!std::ptr::eq(pl(), en()));
    }

    /// A rack containing a character outside the engine's codepoint range used
    /// to index off the end of a fixed-size array and panic the worker.
    #[test]
    fn out_of_range_rack_characters_are_ignored() {
        let lang = pl();
        let words = ["kot", "kota"];
        let dawg = compile_in(lang, &words);
        let gaddag = compile_gaddag_in(lang, &words);
        let mut board = Board::with_seed(lang, 3);
        board.place_word("kot", 7, 7, true, None).unwrap();
        board.first = false;
        // U+4E2D is far above FREQ_SIZE; U+0416 is Cyrillic, also out of range.
        let moves = board.gaddag_best_words(&dawg, &gaddag, "a\u{4E2D}\u{0416}", 5, false);
        assert!(moves.iter().any(|(w, ..)| w == "kota"));
    }

    /// Every word in the lexicon must be reconstructible from every one of its
    /// letters (anchors), following the reversed-prefix / SEP / suffix path.
    #[test]
    fn gaddag_reconstructs_every_word_from_every_anchor() {
        let words = ["kot", "kota", "koty", "as", "ma", "mama", "tom"];
        let g = compile_gaddag_in(pl(), &words);
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
        let dawg = compile_in(pl(), &["kot", "kit", "at", "ot", "ma"]);
        // prefix "k", suffix "t": allowed middles are o (kot) and i (kit).
        let bits = cross_bits(&dawg, &['k'], &['t']);
        assert_ne!(bits & pl().letter_bit('o'), 0);
        assert_ne!(bits & pl().letter_bit('i'), 0);
        assert_eq!(bits & pl().letter_bit('a'), 0, "kat is not in the lexicon");
        // prefix empty, suffix "t": allowed leaders are a (at) and o (ot).
        let bits = cross_bits(&dawg, &[], &['t']);
        assert_ne!(bits & pl().letter_bit('a'), 0);
        assert_ne!(bits & pl().letter_bit('o'), 0);
        assert_eq!(bits & pl().letter_bit('k'), 0);
    }

    /// The GADDAG generator finds a simple hook, and blanks stand in for a
    /// missing letter.
    #[test]
    fn generator_finds_hook_and_uses_blank() {
        let dawg = compile_in(pl(), &["kot", "koty", "ty", "oto"]);
        let gaddag = compile_gaddag_in(pl(), &["kot", "koty", "ty", "oto"]);
        let mut board = Board::with_seed(pl(), time_seed());
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
        let dawg = compile_in(pl(), &["kot", "koty", "ty", "oto"]);
        let gaddag = compile_gaddag_in(pl(), &["kot", "koty", "ty", "oto"]);

        // "kot" at (7,7) horizontally, with the 't' played as a blank. (7,7) is
        // the centre double-word square; k=2, o=1, t=2 at face value.
        let mut board = Board::with_seed(pl(), time_seed());
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
            let mut b = Board::with_seed(pl(), seed);
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
        let mut board = Board::with_seed(pl(), time_seed());
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

    /// Brute-force endgame value with no pruning and no branching limit, for
    /// the alpha-beta search to be checked against.
    fn brute_endgame(
        board: &mut Board,
        dawg: &Dawg,
        gaddag: &Dawg,
        me: &str,
        them: &str,
        passes: u32,
    ) -> i32 {
        if passes >= 2 {
            return board.lang.rack_points(them) - board.lang.rack_points(me);
        }
        let mut best = -brute_endgame(board, dawg, gaddag, them, me, passes + 1);
        let was_first = board.first;
        for m in board.gaddag_generate(dawg, gaddag, me, false, true) {
            let (r, c, horizontal, ref word, score) = m;
            let used = board.hand_tiles_for_word(word, r, c, horizontal, me);
            let left = rack_after(me, &used);
            let filled = board.apply_move(word, r, c, horizontal, &used);
            let v = if left.is_empty() {
                score as i32 + board.lang.rack_points(them)
            } else {
                score as i32 - brute_endgame(board, dawg, gaddag, them, &left, 0)
            };
            board.undo_move(&filled, was_first);
            best = best.max(v);
        }
        best
    }

    /// The endgame search must return the true optimal differential whenever it
    /// reports `exact`. Checked against unpruned brute force on small racks.
    #[test]
    fn endgame_search_matches_brute_force() {
        let words = ["kot", "koty", "ty", "oto", "to", "ot", "y", "tok", "kto"];
        let dawg = compile_in(pl(), &words);
        let gaddag = compile_gaddag_in(pl(), &words);

        for (me, them) in [("y", "o"), ("ty", "o"), ("to", "yk"), ("oty", "k")] {
            let mut board = Board::with_seed(pl(), time_seed());
            board.place_word("kot", 7, 7, true, None).unwrap();
            board.set_bag(Vec::new());

            let mut brute_board = board.clone();
            let expected = brute_endgame(&mut brute_board, &dawg, &gaddag, me, them, 0);
            assert_eq!(
                brute_board.__str__(),
                board.__str__(),
                "brute force left the board dirty"
            );

            let (_mv, value, stats) =
                board.solve_endgame_inner(
                &dawg, &gaddag, me, them, 10_000_000, &[usize::MAX; 16], 64,
            );
            assert!(
                !stats.capped && !stats.truncated,
                "search hit a limit on a tiny position"
            );
            assert_eq!(
                value, expected,
                "alpha-beta disagreed with brute force for me={me} them={them}"
            );
        }
    }

    /// Make/unmake must leave the position byte-identical, or a search reads a
    /// corrupted board a few plies later.
    #[test]
    fn endgame_search_restores_the_board() {
        let words = ["kot", "koty", "ty", "oto", "to", "ot", "tok", "kto"];
        let dawg = compile_in(pl(), &words);
        let gaddag = compile_gaddag_in(pl(), &words);
        let mut board = Board::with_seed(pl(), time_seed());
        board.place_word("kot", 7, 7, true, Some(vec!['k', 'o', '?'])).unwrap();
        board.set_bag(Vec::new());

        let before = board.__str__();
        let blanks_before = board.blanks;
        board.solve_endgame_inner(&dawg, &gaddag, "oty", "k", 100_000, &EG_BRANCH, EG_MAX_DEPTH);
        assert_eq!(board.__str__(), before, "search left tiles on the board");
        assert_eq!(board.blanks, blanks_before, "search left a stale blank flag");
    }

    /// `copy()` is a real deep copy, and `unplace_word` restores the position
    /// exactly -- both are load-bearing for make/unmake search.
    #[test]
    fn copy_and_unplace_restore_the_position() {
        let mut board = Board::with_seed(pl(), time_seed());
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
