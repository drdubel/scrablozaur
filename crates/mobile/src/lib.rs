//! The engine as the Android app sees it, through UniFFI.
//!
//! Like `crates/py`, a thin layer: it converts between the engine's types and
//! flat records Kotlin can hold, and maps errors. Every rule and decision
//! lives in `scrablozaur_core`; nothing here may grow logic of its own, or the
//! app and the web stop playing the same game.
//!
//! Grids cross the boundary as 225-element vectors in row-major order.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};

use scrablozaur_core::game::{
    all_levels, run_benchmark, BenchmarkControl, BenchmarkResult, ComputerMove, EngineMode,
    GameError, GameMode, GameSession, GameState, LanguagePack, PlayerConfig, Preview, SortMode,
    Suggestion, Winner,
};
use scrablozaur_core::scan::decode::{decode, CellReading, DecodeOptions, ScanBoard};
use scrablozaur_core::{square_bonus, Language, LeaveNet, Lexicon};

uniffi::setup_scaffolding!();

const CELLS: usize = 15 * 15;

// ── Errors ───────────────────────────────────────────────────────────────────

/// Loading a language failed: a missing or corrupt asset, or files from
/// different languages paired up.
///
/// Error fields are never called `message`: these become Kotlin exceptions,
/// and a field of that name collides with `Throwable.message`.
#[derive(Debug, uniffi::Error)]
pub enum LoadError {
    Failed { detail: String },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LoadError::Failed { detail } => f.write_str(detail),
        }
    }
}

impl std::error::Error for LoadError {}

/// Why a game action was refused -- `GameError`, flattened for Kotlin. The app
/// words each one in the player's language.
#[derive(Debug, uniffi::Error)]
pub enum MoveError {
    NotInDictionary {
        word: String,
    },
    FirstMoveMustCoverCentre,
    NotConnected,
    TooShort,
    OutOfBounds,
    LetterMismatch {
        row: u32,
        col: u32,
        on_board: String,
        in_word: String,
    },
    CrossWordInvalid {
        word: String,
    },
    NoNewTiles,
    WordNotWhole,
    MissingTiles,
    TilesNotOnRack,
    BagTooSmall,
    GameOver,
    NothingToUndo,
    WrongMode,
    SortUnavailable,
    InvalidPlayers {
        reason: String,
    },
    RackTooLong,
    InvalidSave {
        reason: String,
    },
    Engine {
        detail: String,
    },
}

impl fmt::Display for MoveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for MoveError {}

impl From<GameError> for MoveError {
    fn from(e: GameError) -> Self {
        match e {
            GameError::NotInDictionary(word) => MoveError::NotInDictionary { word },
            GameError::FirstMoveMustCoverCentre => MoveError::FirstMoveMustCoverCentre,
            GameError::NotConnected => MoveError::NotConnected,
            GameError::TooShort => MoveError::TooShort,
            GameError::OutOfBounds => MoveError::OutOfBounds,
            GameError::LetterMismatch {
                row,
                col,
                on_board,
                in_word,
            } => MoveError::LetterMismatch {
                row: row as u32,
                col: col as u32,
                on_board: on_board.to_string(),
                in_word: in_word.to_string(),
            },
            GameError::CrossWordInvalid(word) => MoveError::CrossWordInvalid { word },
            GameError::NoNewTiles => MoveError::NoNewTiles,
            GameError::WordNotWhole => MoveError::WordNotWhole,
            GameError::MissingTiles => MoveError::MissingTiles,
            GameError::TilesNotOnRack => MoveError::TilesNotOnRack,
            GameError::BagTooSmall => MoveError::BagTooSmall,
            GameError::GameOver => MoveError::GameOver,
            GameError::NothingToUndo => MoveError::NothingToUndo,
            GameError::WrongMode => MoveError::WrongMode,
            GameError::SortUnavailable(_) => MoveError::SortUnavailable,
            GameError::InvalidPlayers(reason) => MoveError::InvalidPlayers { reason },
            GameError::RackTooLong => MoveError::RackTooLong,
            GameError::InvalidSave(reason) => MoveError::InvalidSave { reason },
            GameError::Engine(detail) => MoveError::Engine { detail },
        }
    }
}

/// Scan readings the decoder cannot take: the wrong number of squares, or a
/// letter that is not one character of the language.
#[derive(Debug, uniffi::Error)]
pub enum ScanError {
    Invalid { reason: String },
}

impl fmt::Display for ScanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScanError::Invalid { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for ScanError {}

// ── Engine-wide settings ─────────────────────────────────────────────────────

/// What a board square multiplies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Premium {
    None,
    DoubleLetter,
    TripleLetter,
    DoubleWord,
    TripleWord,
    /// The centre: a double word the first move must cover.
    Centre,
}

/// The premium layout, 225 squares row-major, straight from the engine's
/// scoring table.
#[uniffi::export]
pub fn premium_layout() -> Vec<Premium> {
    (0..CELLS)
        .map(|i| {
            let (r, c) = (i / 15, i % 15);
            match square_bonus(r, c) {
                _ if r == 7 && c == 7 => Premium::Centre,
                (_, 3) => Premium::TripleWord,
                (_, 2) => Premium::DoubleWord,
                (3, _) => Premium::TripleLetter,
                (2, _) => Premium::DoubleLetter,
                _ => Premium::None,
            }
        })
        .collect()
}

/// Size the engine's thread pool. Only possible before the first search;
/// returns false once the pool exists.
#[uniffi::export]
pub fn set_engine_threads(threads: u32) -> bool {
    scrablozaur_core::set_num_threads(threads.max(1) as usize).is_ok()
}

#[uniffi::export]
pub fn engine_threads() -> u32 {
    scrablozaur_core::num_threads() as u32
}

// ── Languages ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, uniffi::Enum)]
pub enum Engine {
    Ranked,
    Smart,
    Sim,
}

/// One notch of the difficulty dial: the facts the app's description of it is
/// built from.
#[derive(Clone, Debug, uniffi::Record)]
pub struct LevelFacts {
    pub level: u8,
    pub engine: Engine,
    /// Rank window into the best-first move list, for ranked levels.
    pub rank_best: Option<u32>,
    pub rank_worst: Option<u32>,
    pub slow: bool,
}

/// A loaded language: its definition, dictionary and (where trained) leave net.
#[derive(uniffi::Object)]
pub struct LanguagePackHandle {
    pack: LanguagePack,
}

#[uniffi::export]
impl LanguagePackHandle {
    /// Load a language from the text of its `languages/<code>.json` and the
    /// paths of its lexicon files and optional leave net.
    #[uniffi::constructor]
    pub fn load(
        definition_json: String,
        dawg_path: String,
        gaddag_path: String,
        leave_net_path: Option<String>,
    ) -> Result<Arc<Self>, LoadError> {
        let fail = |detail: String| LoadError::Failed { detail };
        let lang: &'static Language =
            Language::from_json(&definition_json).map_err(|e| fail(e.to_string()))?;
        let lexicon =
            Lexicon::open(lang, &dawg_path, Some(&gaddag_path)).map_err(|e| fail(e.to_string()))?;
        let net = match leave_net_path {
            Some(path) => Some(Arc::new(
                LeaveNet::open(lang, &path).map_err(|e| fail(e.to_string()))?,
            )),
            None => None,
        };
        let pack = LanguagePack::new(Arc::new(lexicon), net).map_err(|e| fail(e.to_string()))?;
        Ok(Arc::new(LanguagePackHandle { pack }))
    }

    pub fn code(&self) -> String {
        self.pack.lang.code().to_string()
    }

    /// The letters in collation order.
    pub fn alphabet(&self) -> String {
        self.pack.lang.alphabet()
    }

    pub fn blank(&self) -> String {
        self.pack.lang.blank().to_string()
    }

    /// Face value of every tile, the blank included.
    pub fn letter_points(&self) -> HashMap<String, u32> {
        let lang = self.pack.lang;
        lang.alphabet()
            .chars()
            .chain(std::iter::once(lang.blank()))
            .map(|c| (c.to_string(), lang.letter_points(c)))
            .collect()
    }

    /// How many of each tile a full bag holds, the blank included.
    pub fn tile_counts(&self) -> HashMap<String, u32> {
        let mut counts = HashMap::new();
        for &c in self.pack.lang.bag() {
            *counts.entry(c.to_string()).or_insert(0) += 1;
        }
        counts
    }

    pub fn has_leave_net(&self) -> bool {
        self.pack.leave_net.is_some()
    }

    pub fn max_level(&self) -> u8 {
        self.pack.max_level()
    }

    pub fn levels(&self) -> Vec<LevelFacts> {
        all_levels(self.pack.leave_net.is_some())
            .into_iter()
            .map(|l| LevelFacts {
                level: l.level,
                engine: match l.engine {
                    EngineMode::Ranked => Engine::Ranked,
                    EngineMode::Smart => Engine::Smart,
                    EngineMode::Sim => Engine::Sim,
                },
                rank_best: l.rank_window.map(|w| w.0 as u32),
                rank_worst: l.rank_window.map(|w| w.1 as u32),
                slow: l.slow,
            })
            .collect()
    }

    pub fn is_word(&self, word: String) -> bool {
        self.pack.lexicon.contains(&word.to_lowercase())
    }

    /// The most probable legal board under `readings`, 225 squares row-major,
    /// keeping every tile of the last confirmed board -- `prior_letters` (225,
    /// a lowercase letter or "" for empty) and `prior_blanks` -- as it was.
    /// See `scan::decode` in the engine; its default options apply.
    pub fn decode_scan(
        &self,
        readings: Vec<SquareReading>,
        prior_letters: Option<Vec<String>>,
        prior_blanks: Option<Vec<bool>>,
    ) -> Result<DecodedBoard, ScanError> {
        let invalid = |reason: String| ScanError::Invalid { reason };
        let one_char = |s: &str| {
            let mut chars = s.chars();
            match (chars.next(), chars.next()) {
                (Some(ch), None) => Ok(ch),
                _ => Err(invalid(format!("'{s}' is not one letter"))),
            }
        };
        if readings.len() != CELLS {
            return Err(invalid(format!("{} readings, not {CELLS}", readings.len())));
        }
        let mut grid: [[CellReading; 15]; 15] =
            std::array::from_fn(|_| std::array::from_fn(|_| CellReading::empty()));
        for (i, r) in readings.into_iter().enumerate() {
            let letters = r
                .letters
                .iter()
                .map(|l| Ok((one_char(&l.letter)?, l.p)))
                .collect::<Result<Vec<_>, ScanError>>()?;
            grid[i / 15][i % 15] = CellReading {
                p_tile: r.p_tile,
                letters,
                p_blank: r.p_blank,
            };
        }
        let prior = match prior_letters {
            None => None,
            Some(letters) => {
                let blanks = prior_blanks.unwrap_or_else(|| vec![false; CELLS]);
                if letters.len() != CELLS || blanks.len() != CELLS {
                    return Err(invalid(format!(
                        "the previous board must have {CELLS} squares"
                    )));
                }
                let mut board = ScanBoard::empty();
                for i in 0..CELLS {
                    if !letters[i].is_empty() {
                        board.letters[i / 15][i % 15] = one_char(&letters[i])?;
                    }
                    board.blanks[i / 15][i % 15] = blanks[i];
                }
                Some(board)
            }
        };
        let out = decode(
            &self.pack.lexicon,
            &grid,
            prior.as_ref(),
            &DecodeOptions::default(),
        )
        .map_err(|e| invalid(e.to_string()))?;
        Ok(DecodedBoard {
            letters: flatten(&out.board.letters, |ch| {
                if ch == '-' {
                    String::new()
                } else {
                    ch.to_string()
                }
            }),
            blanks: flatten(&out.board.blanks, |b| b),
            corrected: flatten(&out.corrected, |b| b),
            flagged: flatten(&out.flagged, |b| b),
            legal: out.legal,
            complete: out.complete,
            cost: out.cost,
        })
    }
}

// ── Scanning ─────────────────────────────────────────────────────────────────

/// One letter a scanned square may show, and how likely.
#[derive(Clone, Debug, uniffi::Record)]
pub struct LetterChance {
    /// Lowercase.
    pub letter: String,
    pub p: f32,
}

/// What the scanner made of one square: see `CellReading` in the engine.
#[derive(Clone, Debug, uniffi::Record)]
pub struct SquareReading {
    /// Probability that the square holds a tile.
    pub p_tile: f32,
    /// The letter such a tile shows; letters left out share what these
    /// leave, and none at all says nothing about the letter.
    pub letters: Vec<LetterChance>,
    /// Probability that the tile is a blank.
    pub p_blank: f32,
}

/// The board the decoder settled on, 225 squares row-major.
#[derive(Clone, Debug, uniffi::Record)]
pub struct DecodedBoard {
    /// A letter, or "" for an empty square.
    pub letters: Vec<String>,
    /// True where a blank plays the letter shown.
    pub blanks: Vec<bool>,
    /// Where the board differs from the readings' own best guess.
    pub corrected: Vec<bool>,
    /// Squares still in a non-word or alone; only when not `legal`.
    pub flagged: Vec<bool>,
    /// Every rule of a real board holds.
    pub legal: bool,
    /// The search finished within its budget.
    pub complete: bool,
    /// How far, in nats, the rules forced the board from the readings.
    pub cost: f32,
}

// ── Games ────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum Mode {
    Sandbox,
    SandboxAuto,
    Competitive,
}

impl From<Mode> for GameMode {
    fn from(m: Mode) -> Self {
        match m {
            Mode::Sandbox => GameMode::Sandbox,
            Mode::SandboxAuto => GameMode::SandboxAuto,
            Mode::Competitive => GameMode::Competitive,
        }
    }
}

impl From<GameMode> for Mode {
    fn from(m: GameMode) -> Self {
        match m {
            GameMode::Sandbox => Mode::Sandbox,
            GameMode::SandboxAuto => Mode::SandboxAuto,
            GameMode::Competitive => Mode::Competitive,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
pub enum HintOrder {
    Score,
    Smart,
    Sim,
}

impl From<HintOrder> for SortMode {
    fn from(o: HintOrder) -> Self {
        match o {
            HintOrder::Score => SortMode::Score,
            HintOrder::Smart => SortMode::Smart,
            HintOrder::Sim => SortMode::Sim,
        }
    }
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct PlayerSetup {
    pub name: String,
    pub is_computer: bool,
    pub difficulty: u8,
    pub strength: u8,
}

impl From<PlayerSetup> for PlayerConfig {
    fn from(p: PlayerSetup) -> Self {
        PlayerConfig {
            name: p.name,
            is_computer: p.is_computer,
            difficulty: p.difficulty,
            strength: p.strength,
        }
    }
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct PlayerView {
    pub name: String,
    pub is_computer: bool,
    pub score: i32,
    /// Empty for the computer in a competitive game.
    pub letters: String,
    pub difficulty: u8,
    pub strength: u8,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct ComputerPlay {
    pub word: String,
    pub score: u32,
    pub row: u32,
    pub col: u32,
    pub horizontal: bool,
    pub passed: bool,
}

impl From<ComputerMove> for ComputerPlay {
    fn from(m: ComputerMove) -> Self {
        ComputerPlay {
            word: m.word,
            score: m.score,
            row: m.row as u32,
            col: m.col as u32,
            horizontal: m.horizontal,
            passed: m.passed,
        }
    }
}

#[derive(Clone, Debug, uniffi::Enum)]
pub enum Outcome {
    Winner { player: u32 },
    Tie,
}

/// Everything a screen needs to draw a game.
#[derive(Clone, Debug, uniffi::Record)]
pub struct GameView {
    pub language: String,
    pub mode: Mode,
    /// 225 squares, row-major: a letter, or "" for empty.
    pub cells: Vec<String>,
    /// 225 squares: true where a blank stands in for the letter shown.
    pub blanks: Vec<bool>,
    /// 225 squares: the seat that placed the tile, -1 for none.
    pub owners: Vec<i8>,
    pub players: Vec<PlayerView>,
    pub current_player: u32,
    pub is_first_move: bool,
    pub move_number: u32,
    pub can_undo: bool,
    pub tiles_remaining: u32,
    pub last_computer_move: Option<ComputerPlay>,
    pub game_over: bool,
    /// Set once the game is over.
    pub outcome: Option<Outcome>,
    /// 0-100: how the human's last move compared with what the rack allowed.
    pub last_move_rating: Option<u8>,
    /// A computer seat has the turn and the app should let it play.
    pub computer_to_move: bool,
}

fn flatten<T: Copy, U>(grid: &[[T; 15]; 15], f: impl Fn(T) -> U) -> Vec<U> {
    grid.iter().flatten().map(|&v| f(v)).collect()
}

fn game_view(state: GameState, computer_to_move: bool) -> GameView {
    GameView {
        language: state.language,
        mode: state.mode.into(),
        cells: flatten(&state.board, |c| {
            if c == '-' {
                String::new()
            } else {
                c.to_string()
            }
        }),
        blanks: flatten(&state.blanks, |b| b),
        owners: flatten(&state.tile_owners, |o| o.map_or(-1, |i| i as i8)),
        players: state
            .players
            .into_iter()
            .map(|p| PlayerView {
                name: p.name,
                is_computer: p.is_computer,
                score: p.score,
                letters: p.letters,
                difficulty: p.difficulty,
                strength: p.strength,
            })
            .collect(),
        current_player: state.current_player as u32,
        is_first_move: state.is_first_move,
        move_number: state.move_number,
        can_undo: state.can_undo,
        tiles_remaining: state.tiles_remaining as u32,
        last_computer_move: state.last_computer_move.map(Into::into),
        game_over: state.game_over,
        outcome: state.winner.map(|w| match w {
            Winner::Player(i) => Outcome::Winner { player: i as u32 },
            Winner::Tie => Outcome::Tie,
        }),
        last_move_rating: state.last_move_rating,
        computer_to_move,
    }
}

/// One ranked play.
#[derive(Clone, Debug, uniffi::Record)]
pub struct Hint {
    pub word: String,
    pub score: u32,
    pub row: u32,
    pub col: u32,
    pub horizontal: bool,
    /// The tiles it takes off the rack, `?` for a blank.
    pub used: String,
    /// What it was ranked by: score, score + leave, or simulated equity.
    pub value: f64,
}

impl From<Suggestion> for Hint {
    fn from(s: Suggestion) -> Self {
        Hint {
            word: s.word,
            score: s.score,
            row: s.row as u32,
            col: s.col as u32,
            horizontal: s.horizontal,
            used: s.used.into_iter().collect(),
            value: s.value,
        }
    }
}

#[derive(Clone, Debug, uniffi::Enum)]
pub enum PreviewResult {
    Score { points: u32 },
    NotInDictionary,
    InvalidPlacement,
    ScoreError,
}

/// One game. Every method locks the session, and the bot's methods can take a
/// second or more: call them off the main thread.
#[derive(uniffi::Object)]
pub struct Game {
    session: Mutex<GameSession>,
}

impl Game {
    fn with<T>(&self, f: impl FnOnce(&mut GameSession) -> T) -> T {
        let mut session = self.session.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut session)
    }
}

#[uniffi::export]
impl Game {
    /// Deal a new game. `seed` 0 deals from the clock.
    #[uniffi::constructor]
    pub fn new(
        pack: Arc<LanguagePackHandle>,
        mode: Mode,
        players: Vec<PlayerSetup>,
        seed: u64,
    ) -> Result<Arc<Self>, MoveError> {
        let players = players.into_iter().map(Into::into).collect();
        let session = GameSession::new(pack.pack.clone(), mode.into(), players, seed)?;
        Ok(Arc::new(Game {
            session: Mutex::new(session),
        }))
    }

    /// Restore a game saved with `save`, in `pack`'s language.
    #[uniffi::constructor]
    pub fn restore(pack: Arc<LanguagePackHandle>, saved: String) -> Result<Arc<Self>, MoveError> {
        let session = GameSession::from_json(pack.pack.clone(), &saved)?;
        Ok(Arc::new(Game {
            session: Mutex::new(session),
        }))
    }

    pub fn save(&self) -> String {
        self.with(|s| s.to_json())
    }

    pub fn state(&self) -> GameView {
        self.with(|s| game_view(s.state(), s.computer_to_move()))
    }

    pub fn computer_turn(&self) -> Result<ComputerPlay, MoveError> {
        Ok(self.with(|s| s.computer_turn())?.into())
    }

    pub fn human_move(
        &self,
        word: String,
        row: u32,
        col: u32,
        horizontal: bool,
    ) -> Result<u32, MoveError> {
        Ok(self.with(|s| s.human_move(&word, row as usize, col as usize, horizontal))?)
    }

    pub fn preview(&self, word: String, row: u32, col: u32, horizontal: bool) -> PreviewResult {
        match self.with(|s| s.preview_score(&word, row as usize, col as usize, horizontal)) {
            Preview::Score(points) => PreviewResult::Score { points },
            Preview::NotInDictionary => PreviewResult::NotInDictionary,
            Preview::InvalidPlacement => PreviewResult::InvalidPlacement,
            Preview::ScoreError => PreviewResult::ScoreError,
        }
    }

    pub fn skip(&self) -> Result<(), MoveError> {
        Ok(self.with(|s| s.skip())?)
    }

    pub fn exchange(&self, letters: String) -> Result<(), MoveError> {
        Ok(self.with(|s| s.exchange(&letters))?)
    }

    pub fn resign(&self) -> Result<(), MoveError> {
        Ok(self.with(|s| s.resign())?)
    }

    pub fn undo(&self) -> Result<(), MoveError> {
        Ok(self.with(|s| s.undo())?)
    }

    pub fn next_auto_move(&self) -> Result<ComputerPlay, MoveError> {
        Ok(self.with(|s| s.next_auto_move())?.into())
    }

    pub fn set_letters(&self, letters: String) -> Result<(), MoveError> {
        Ok(self.with(|s| s.set_letters(&letters))?)
    }

    pub fn sandbox_suggestions(&self, order: HintOrder) -> Result<Vec<Hint>, MoveError> {
        let list = self.with(|s| s.sandbox_suggestions(order.into()))?;
        Ok(list.into_iter().map(Into::into).collect())
    }

    pub fn hints(&self, order: HintOrder) -> Result<Vec<Hint>, MoveError> {
        let list = self.with(|s| s.hints(order.into()))?;
        Ok(list.into_iter().map(Into::into).collect())
    }

    pub fn place_computer_word(
        &self,
        word: String,
        row: u32,
        col: u32,
        horizontal: bool,
    ) -> Result<u32, MoveError> {
        Ok(self.with(|s| s.place_computer_word(&word, row as usize, col as usize, horizontal))?)
    }
}

// ── Benchmark ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug, uniffi::Record)]
pub struct SeatStats {
    pub name: String,
    pub difficulty: u8,
    pub strength: u8,
    pub games_played: u32,
    pub wins: u32,
    pub ties: u32,
    pub avg_score: f64,
    pub high_score: i32,
    pub low_score: i32,
    pub words_played: u32,
    pub avg_word_score: f64,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct ReplayMove {
    pub player: u32,
    pub word: String,
    pub score: u32,
    pub row: u32,
    pub col: u32,
    pub horizontal: bool,
    pub passed: bool,
    pub cells: Vec<String>,
    pub blanks: Vec<bool>,
    pub owners: Vec<i8>,
    pub scores_after: Vec<i32>,
    pub letters_after: Vec<String>,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct ReplayGame {
    pub outcome: Outcome,
    pub winner_score: i32,
    pub final_players: Vec<PlayerView>,
    pub moves: Vec<ReplayMove>,
}

#[derive(Clone, Debug, uniffi::Record)]
pub struct BenchmarkReport {
    pub games_played: u32,
    pub duration_ms: u64,
    pub seats: Vec<SeatStats>,
    pub best_game: Option<ReplayGame>,
    pub avg_game_length: f64,
    pub longest_word: Option<String>,
    pub longest_word_score: Option<u32>,
    pub highest_single_move_score: Option<u32>,
    /// Set if the run stopped on an error rather than finishing.
    pub error: Option<String>,
}

fn report(result: Result<BenchmarkResult, GameError>) -> BenchmarkReport {
    let r = match result {
        Ok(r) => r,
        Err(e) => {
            return BenchmarkReport {
                games_played: 0,
                duration_ms: 0,
                seats: Vec::new(),
                best_game: None,
                avg_game_length: 0.0,
                longest_word: None,
                longest_word_score: None,
                highest_single_move_score: None,
                error: Some(e.to_string()),
            }
        }
    };
    let players = |ps: Vec<scrablozaur_core::game::Player>| {
        ps.into_iter()
            .map(|p| PlayerView {
                name: p.name,
                is_computer: p.is_computer,
                score: p.score,
                letters: p.letters,
                difficulty: p.difficulty,
                strength: p.strength,
            })
            .collect()
    };
    BenchmarkReport {
        games_played: r.games_played as u32,
        duration_ms: r.duration_ms,
        seats: r
            .player_stats
            .iter()
            .map(|s| SeatStats {
                name: s.name.clone(),
                difficulty: s.difficulty,
                strength: s.strength,
                games_played: s.games_played as u32,
                wins: s.wins as u32,
                ties: s.ties as u32,
                avg_score: s.avg_score(),
                high_score: s.high_score,
                low_score: s.low_score,
                words_played: s.words_played as u32,
                avg_word_score: s.avg_word_score(),
            })
            .collect(),
        best_game: r.best_game.map(|g| ReplayGame {
            outcome: match g.winner {
                Winner::Player(i) => Outcome::Winner { player: i as u32 },
                Winner::Tie => Outcome::Tie,
            },
            winner_score: g.winner_score,
            final_players: players(g.final_players),
            moves: g
                .moves
                .into_iter()
                .map(|m| ReplayMove {
                    player: m.player as u32,
                    word: m.word,
                    score: m.score,
                    row: m.row as u32,
                    col: m.col as u32,
                    horizontal: m.horizontal,
                    passed: m.passed,
                    cells: flatten(&m.board, |c| {
                        if c == '-' {
                            String::new()
                        } else {
                            c.to_string()
                        }
                    }),
                    blanks: flatten(&m.blanks, |b| b),
                    owners: flatten(&m.tile_owners, |o| o.map_or(-1, |i| i as i8)),
                    scores_after: m.scores_after,
                    letters_after: m.letters_after,
                })
                .collect(),
        }),
        avg_game_length: r.avg_game_length,
        longest_word_score: r.longest_word.as_ref().map(|(_, s)| *s),
        longest_word: r.longest_word.map(|(w, _)| w),
        highest_single_move_score: r.highest_single_move_score,
        error: None,
    }
}

/// A benchmark running on a background thread: poll `games_done`, `cancel`
/// to stop early, collect `result` once `is_finished`.
#[derive(uniffi::Object)]
pub struct Benchmark {
    control: Arc<BenchmarkControl>,
    total: u32,
    result: Arc<Mutex<Option<BenchmarkReport>>>,
}

#[uniffi::export]
impl Benchmark {
    #[uniffi::constructor]
    pub fn start(
        pack: Arc<LanguagePackHandle>,
        players: Vec<PlayerSetup>,
        games: u32,
        seed: u64,
    ) -> Arc<Self> {
        let control = Arc::new(BenchmarkControl::default());
        let result = Arc::new(Mutex::new(None));
        let players: Vec<PlayerConfig> = players.into_iter().map(Into::into).collect();
        let (c, slot, pack) = (control.clone(), result.clone(), pack.pack.clone());
        std::thread::spawn(move || {
            let r = report(run_benchmark(&pack, &players, games as usize, seed, &c));
            *slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
        });
        Arc::new(Benchmark {
            control,
            total: games,
            result,
        })
    }

    pub fn games_done(&self) -> u32 {
        self.control.games_done() as u32
    }

    pub fn total_games(&self) -> u32 {
        self.total
    }

    pub fn cancel(&self) {
        self.control.cancel();
    }

    pub fn is_finished(&self) -> bool {
        self.result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    /// The report, once finished; `None` before.
    pub fn result(&self) -> Option<BenchmarkReport> {
        self.result
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }
}
