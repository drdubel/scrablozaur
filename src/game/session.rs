//! One game, start to finish: the port of `web/game.py`'s `GameSession` plus
//! what `web/routers/{game,board}.py` do to it.
//!
//! Each public method is one of the web's endpoints, with the same checks in
//! the same order and the same state changes, so that a game played through
//! the app is the game the web would have played. The places this differs are
//! marked "Deviation" and are all a rule the web gets wrong.
//!
//! The web runs a computer's reply inside the human's request. Here it is a
//! separate call, `computer_turn`, so an app can show the human's move before
//! the bot starts thinking; `computer_to_move` says when one is due.

use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::bot::{
    choose_move, choose_move_sim, move_rating, rank_suggestions, SimOptions, SortMode,
    StaticOptions, Suggestion, INTERACTIVE_ENDGAME_NODES, SUGGEST_SIM_ITERATIONS,
};
use super::difficulty::{
    clamp_level, engine_mode, max_level_for, pick_by_rank, pick_in_window, rank_window,
    strength_rank_window, EngineMode, DEFAULT_LEVEL,
};
use super::rng::GameRng;
use super::rules::{apply_end_of_game_scoring, went_out, Streaks, TurnResult};
use crate::{
    in_bounds, word_cell, Board, BoardSnapshot, EngineError, Language, LeaveNet, Lexicon,
    BOARD_SIZE, CENTER, RACK_SIZE,
};

/// Everything a game needs from its language: the definition, the dictionary,
/// and -- where one has been trained -- the leave net levels 9-10 run on.
/// Cheap to clone; sessions share one.
#[derive(Clone)]
pub struct LanguagePack {
    pub lang: &'static Language,
    pub lexicon: Arc<Lexicon>,
    pub leave_net: Option<Arc<LeaveNet>>,
}

impl LanguagePack {
    /// Bundle a lexicon and an optional net, refusing ones built for another
    /// language -- they would not fail later, they would mis-score.
    pub fn new(
        lexicon: Arc<Lexicon>,
        leave_net: Option<Arc<LeaveNet>>,
    ) -> Result<Self, EngineError> {
        let lang = lexicon.lang();
        if let Some(net) = &leave_net {
            if !std::ptr::eq(net.lang(), lang) {
                return Err(EngineError::Value(format!(
                    "the dictionary is '{}' but the leave net is '{}'",
                    lang.code(),
                    net.lang().code()
                )));
            }
        }
        Ok(LanguagePack {
            lang,
            lexicon,
            leave_net,
        })
    }

    pub fn max_level(&self) -> u8 {
        max_level_for(self.leave_net.is_some())
    }

    pub(crate) fn net(&self) -> Option<&LeaveNet> {
        self.leave_net.as_deref()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GameMode {
    /// Referee board: one computer seat whose rack the user types in, humans
    /// scored at face value, no bag.
    Sandbox,
    /// 2-4 computers, a real bag, advanced one move at a time.
    SandboxAuto,
    /// One human against one computer, real bag, hidden computer rack.
    Competitive,
}

/// A seat as requested at the start of a game.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlayerConfig {
    pub name: String,
    pub is_computer: bool,
    pub difficulty: u8,
    /// 1-100% move quality within the chosen engine's ranking.
    pub strength: u8,
}

impl PlayerConfig {
    pub fn human(name: &str) -> Self {
        PlayerConfig {
            name: name.to_string(),
            is_computer: false,
            difficulty: DEFAULT_LEVEL,
            strength: 100,
        }
    }

    pub fn computer(name: &str, difficulty: u8, strength: u8) -> Self {
        PlayerConfig {
            name: name.to_string(),
            is_computer: true,
            difficulty,
            strength,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Player {
    pub name: String,
    pub is_computer: bool,
    pub score: i32,
    pub letters: String,
    pub difficulty: u8,
    pub strength: u8,
}

/// The computer's last move, as the board shows it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComputerMove {
    pub word: String,
    pub score: u32,
    pub row: usize,
    pub col: usize,
    pub horizontal: bool,
    pub passed: bool,
}

impl ComputerMove {
    fn passed() -> Self {
        ComputerMove {
            word: String::new(),
            score: 0,
            row: 0,
            col: 0,
            horizontal: true,
            passed: true,
        }
    }
}

/// Why a move or action was refused. Variants, not messages, so the app can say
/// it in the player's language.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameError {
    /// The typed word is not in the dictionary.
    NotInDictionary(String),
    /// The first word must cover the centre square.
    FirstMoveMustCoverCentre,
    /// A later word must touch a tile already on the board.
    NotConnected,
    /// A play must be at least two letters.
    TooShort,
    /// The word runs off the board.
    OutOfBounds,
    /// The word disagrees with a tile already on the board.
    LetterMismatch {
        row: usize,
        col: usize,
        on_board: char,
        in_word: char,
    },
    /// A perpendicular word the play forms is not in the dictionary.
    CrossWordInvalid(String),
    /// The word only runs over tiles already on the board.
    NoNewTiles,
    /// The word stops short of a tile it touches along its own line, so the
    /// word actually on the board would be longer than the one submitted.
    WordNotWhole,
    /// The rack does not hold the tiles the word needs.
    MissingTiles,
    /// The tiles to exchange are not all on the rack.
    TilesNotOnRack,
    /// Exchanging needs at least a rack's worth of tiles in the bag.
    BagTooSmall,
    GameOver,
    NothingToUndo,
    /// The action does not exist in this game mode.
    WrongMode,
    /// The ordering needs a leave net this language does not have.
    SortUnavailable(SortMode),
    /// The players do not fit the mode.
    InvalidPlayers(String),
    /// A rack holds at most seven tiles.
    RackTooLong,
    /// Something the engine refused that the checks above should have caught.
    Engine(String),
    /// A saved game that cannot be restored: another format version, another
    /// language, or a position that does not hold together.
    InvalidSave(String),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for GameError {}

impl From<EngineError> for GameError {
    fn from(e: EngineError) -> Self {
        GameError::Engine(e.to_string())
    }
}

/// `preview_score`'s answer: the score, or which kind of problem the play has.
/// The web reports these as bare codes and leaves the wording to the UI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Preview {
    Score(u32),
    NotInDictionary,
    InvalidPlacement,
    ScoreError,
}

/// The tile bag a session deals from. Separate from the engine board's own
/// bag, which a session never draws from -- as in the web.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileBag {
    tiles: Vec<char>,
}

impl TileBag {
    /// A shuffled bag for `lang`'s distribution.
    fn full(lang: &Language, rng: &mut GameRng) -> Self {
        let mut tiles = lang.bag().to_vec();
        rng.shuffle(&mut tiles);
        TileBag { tiles }
    }

    /// Up to `n` tiles off the front.
    fn draw(&mut self, n: usize) -> Vec<char> {
        let n = n.min(self.tiles.len());
        self.tiles.drain(..n).collect()
    }

    pub fn remaining(&self) -> usize {
        self.tiles.len()
    }

    /// Return tiles and shuffle them in.
    fn put_back(&mut self, tiles: &[char], rng: &mut GameRng) {
        self.tiles.extend_from_slice(tiles);
        rng.shuffle(&mut self.tiles);
    }
}

/// Who won, once the game is over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Winner {
    Player(usize),
    Tie,
}

/// One seat as the board shows it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlayerView {
    pub name: String,
    pub is_computer: bool,
    pub score: i32,
    /// Empty for the computer in a competitive game: its rack is hidden.
    pub letters: String,
    pub difficulty: u8,
    pub strength: u8,
}

/// Everything a screen needs to draw the game -- the web's `BoardStateResponse`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    pub language: String,
    pub mode: GameMode,
    /// 15 rows of 15 cells, `'-'` for empty.
    pub board: [[char; BOARD_SIZE]; BOARD_SIZE],
    /// Which occupied squares hold a blank (shown as the letter it stands in
    /// for, but scoring 0 forever).
    pub blanks: [[bool; BOARD_SIZE]; BOARD_SIZE],
    /// Which seat placed each tile.
    pub tile_owners: [[Option<u8>; BOARD_SIZE]; BOARD_SIZE],
    pub players: Vec<PlayerView>,
    pub current_player: usize,
    pub is_first_move: bool,
    pub move_number: u32,
    pub can_undo: bool,
    pub tiles_remaining: usize,
    pub last_computer_move: Option<ComputerMove>,
    pub game_over: bool,
    pub winner: Option<Winner>,
    pub last_move_rating: Option<u8>,
}

type Owners = [[Option<u8>; BOARD_SIZE]; BOARD_SIZE];

/// A deep copy of the position, not a grid to rebuild from -- the `str()` ->
/// `from_grid()` round trip loses the bag and which tiles were blanks.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct UndoEntry {
    board: BoardSnapshot,
    player_scores: Vec<i32>,
    player_letters: Vec<String>,
    current_player: usize,
    is_first_move: bool,
    move_number: u32,
    bag_tiles: Option<Vec<char>>,
    last_computer_move: Option<ComputerMove>,
    tile_owners: Owners,
    streaks: Streaks,
}

pub struct GameSession {
    pack: LanguagePack,
    board: Board,
    players: Vec<Player>,
    current: usize,
    is_first_move: bool,
    move_number: u32,
    history: Vec<UndoEntry>,
    mode: GameMode,
    bag: Option<TileBag>,
    last_computer_move: Option<ComputerMove>,
    tile_owners: Owners,
    game_over: bool,
    streaks: Streaks,
    /// Seats that resigned. A set; kept sorted.
    resigned: Vec<usize>,
    last_move_rating: Option<u8>,
    rng: GameRng,
}

const MAX_PLAYERS: usize = 4;
const MAX_NAME_CHARS: usize = 20;

/// Check the seats against what the web's `_players_from_request` accepts.
fn validate_players(
    mode: GameMode,
    players: &mut [PlayerConfig],
    ceiling: u8,
) -> Result<(), GameError> {
    let bad = |m: &str| Err(GameError::InvalidPlayers(m.to_string()));
    if players.is_empty() || players.len() > MAX_PLAYERS {
        return bad("a game has 1 to 4 players");
    }
    for p in players.iter() {
        let n = p.name.chars().count();
        if n == 0 || n > MAX_NAME_CHARS {
            return bad("a player name has 1 to 20 characters");
        }
    }
    let computers = players.iter().filter(|p| p.is_computer).count();
    match mode {
        GameMode::Competitive => {
            if players.len() != 2 || computers != 1 {
                return bad("a competitive game is exactly one human against one computer");
            }
        }
        GameMode::SandboxAuto => {
            if players.len() < 2 {
                return bad("an automatic game needs at least two computers");
            }
            for p in players.iter_mut() {
                p.is_computer = true;
            }
        }
        GameMode::Sandbox => {
            if computers != 1 {
                return bad("a sandbox game has exactly one computer");
            }
        }
    }
    for p in players.iter_mut() {
        // Clamp to what this language can actually field: without a trained
        // leave net there is no level 9 or 10 to give anyone.
        p.difficulty = clamp_level(p.difficulty as i64, ceiling);
        p.strength = p.strength.clamp(1, 100);
    }
    Ok(())
}

impl GameSession {
    /// Deal a new game. Competitive and automatic games draw for first player
    /// (closest to 'A' wins, a blank beats everything), put the drawn tiles
    /// back, reshuffle, and deal seven each; the sandbox has no bag at all.
    ///
    /// In a competitive game the draw can land on the computer: check
    /// `computer_to_move` afterwards, as after every other action.
    pub fn new(
        pack: LanguagePack,
        mode: GameMode,
        mut players: Vec<PlayerConfig>,
        seed: u64,
    ) -> Result<Self, GameError> {
        validate_players(mode, &mut players, pack.max_level())?;
        let mut rng = GameRng::new(seed);
        let board = Board::new(pack.lang);
        let mut players: Vec<Player> = players
            .into_iter()
            .map(|p| Player {
                name: p.name,
                is_computer: p.is_computer,
                score: 0,
                letters: String::new(),
                difficulty: p.difficulty,
                strength: p.strength,
            })
            .collect();

        let mut bag = None;
        let mut first = 0;
        if mode != GameMode::Sandbox {
            let mut b = TileBag::full(pack.lang, &mut rng);
            let draws = b.draw(players.len());
            first = board.first_draw_winner(&draws);
            b.put_back(&draws, &mut rng);
            for p in players.iter_mut() {
                p.letters = b.draw(RACK_SIZE).into_iter().collect();
            }
            bag = Some(b);
        }

        Ok(GameSession {
            pack,
            board,
            players,
            current: first,
            is_first_move: true,
            move_number: 0,
            history: Vec::new(),
            mode,
            bag,
            last_computer_move: None,
            tile_owners: [[None; BOARD_SIZE]; BOARD_SIZE],
            game_over: false,
            streaks: Streaks::default(),
            resigned: Vec::new(),
            last_move_rating: None,
            rng,
        })
    }

    pub fn mode(&self) -> GameMode {
        self.mode
    }

    pub fn pack(&self) -> &LanguagePack {
        &self.pack
    }

    pub fn board(&self) -> &Board {
        &self.board
    }

    pub fn players(&self) -> &[Player] {
        &self.players
    }

    pub fn current_player_index(&self) -> usize {
        self.current
    }

    pub fn is_over(&self) -> bool {
        self.game_over
    }

    pub fn bag_remaining(&self) -> usize {
        self.bag.as_ref().map_or(0, TileBag::remaining)
    }

    /// Whether a computer's turn is due now. The web plays it inside the
    /// request that handed it the turn -- in a competitive game only; in the
    /// other modes the user advances the computers.
    pub fn computer_to_move(&self) -> bool {
        self.mode == GameMode::Competitive
            && !self.game_over
            && self.players[self.current].is_computer
    }

    /// The board as the web serves it (`_state_response`).
    pub fn state(&self) -> GameState {
        GameState {
            language: self.pack.lang.code().to_string(),
            mode: self.mode,
            board: self.board.board,
            blanks: self.board.blanks,
            tile_owners: self.tile_owners,
            players: self
                .players
                .iter()
                .map(|p| PlayerView {
                    name: p.name.clone(),
                    is_computer: p.is_computer,
                    score: p.score,
                    letters: if self.mode == GameMode::Competitive && p.is_computer {
                        String::new()
                    } else {
                        p.letters.clone()
                    },
                    difficulty: p.difficulty,
                    strength: p.strength,
                })
                .collect(),
            current_player: self.current,
            is_first_move: self.is_first_move,
            move_number: self.move_number,
            can_undo: !self.history.is_empty(),
            tiles_remaining: self.bag_remaining(),
            last_computer_move: self.last_computer_move.clone(),
            game_over: self.game_over,
            winner: self.game_over.then(|| self.winner()),
            last_move_rating: self.last_move_rating,
        }
    }

    fn winner(&self) -> Winner {
        let best = self.players.iter().map(|p| p.score).max().unwrap_or(0);
        let tied: Vec<usize> = (0..self.players.len())
            .filter(|&i| self.players[i].score == best)
            .collect();
        if tied.len() > 1 {
            Winner::Tie
        } else {
            Winner::Player(tied[0])
        }
    }

    // ── Bookkeeping ──────────────────────────────────────────────────────────

    fn advance_turn(&mut self) {
        self.current = (self.current + 1) % self.players.len();
        self.move_number += 1;
    }

    fn push_undo(&mut self) {
        self.history.push(UndoEntry {
            board: self.board.to_snapshot(),
            player_scores: self.players.iter().map(|p| p.score).collect(),
            player_letters: self.players.iter().map(|p| p.letters.clone()).collect(),
            current_player: self.current,
            is_first_move: self.is_first_move,
            move_number: self.move_number,
            bag_tiles: self.bag.as_ref().map(|b| b.tiles.clone()),
            last_computer_move: self.last_computer_move.clone(),
            tile_owners: self.tile_owners,
            streaks: self.streaks,
        });
    }

    /// Mark the squares a play newly covers as `player`'s. Call before placing.
    fn record_placement(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
        player: usize,
    ) {
        for i in 0..word.chars().count() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if self.board.board[r][c] == '-' {
                self.tile_owners[r][c] = Some(player as u8);
            }
        }
    }

    fn refill_rack(&mut self, player: usize) {
        let Some(bag) = self.bag.as_mut() else {
            return;
        };
        let p = &mut self.players[player];
        let needed = RACK_SIZE.saturating_sub(p.letters.chars().count());
        if needed > 0 {
            p.letters.extend(bag.draw(needed));
        }
    }

    /// End the game if `just_played` went out or the streaks ran out -- only
    /// where there is a real bag (`_check_game_over`).
    fn check_game_over(&mut self, just_played: usize) {
        if self.game_over {
            return;
        }
        let Some(bag) = &self.bag else {
            return;
        };
        if went_out(&self.players[just_played].letters, bag.remaining()) {
            apply_end_of_game_scoring(self.pack.lang, &mut self.players, Some(just_played));
            self.game_over = true;
        } else if self.streaks.exhausted() {
            apply_end_of_game_scoring(self.pack.lang, &mut self.players, None);
            self.game_over = true;
        }
    }

    // ── Move validation ──────────────────────────────────────────────────────

    /// Every check a typed word must pass before it can be scored, in the
    /// order the web makes them (dictionary, connectivity, cross-words), plus
    /// what the web leaves to its UI: the word must lie on the board, agree
    /// with the tiles it runs through, place at least one new tile, and be the
    /// whole word along its line.
    fn validate_typed_word(
        &self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> Result<(), GameError> {
        let board = &self.board;
        let len = word.chars().count();
        if !self.pack.lexicon.contains(word) {
            return Err(GameError::NotInDictionary(word.to_string()));
        }
        if len < 2 {
            return Err(GameError::TooShort);
        }
        // The web reads these cells without a bounds check (an IndexError,
        // i.e. a 500); here it is a proper refusal.
        if (0..len).any(|i| {
            let (r, c) = word_cell(row, col, horizontal, i);
            !in_bounds(r, c)
        }) {
            return Err(GameError::OutOfBounds);
        }
        self.check_connectivity(len, row, col, horizontal)?;

        let mut placed_any = false;
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            let on_board = board.board[r][c];
            if on_board != '-' {
                if on_board != ch {
                    return Err(GameError::LetterMismatch {
                        row: r,
                        col: c,
                        on_board,
                        in_word: ch,
                    });
                }
                continue;
            }
            placed_any = true;
            let cross = board.cross_word(r, c, horizontal, ch);
            if cross.len() > 1 && !self.pack.lexicon.contains(&cross) {
                return Err(GameError::CrossWordInvalid(cross));
            }
        }
        // Deviation: the web would score a "play" over existing tiles only.
        if !placed_any {
            return Err(GameError::NoNewTiles);
        }
        // Deviation: the web trusts its UI to extend the word through any
        // tile touching either end; a word that stops short would put a longer,
        // unchecked word on the board.
        let (br, bc) = if horizontal {
            (row as isize, col as isize - 1)
        } else {
            (row as isize - 1, col as isize)
        };
        let (ar, ac) = word_cell(row, col, horizontal, len);
        let occupied = |r: isize, c: isize| {
            r >= 0
                && c >= 0
                && in_bounds(r as usize, c as usize)
                && board.board[r as usize][c as usize] != '-'
        };
        if occupied(br, bc) || occupied(ar as isize, ac as isize) {
            return Err(GameError::WordNotWhole);
        }
        Ok(())
    }

    /// The first word covers the centre; every later one shares or touches a
    /// tile already down (`_check_connectivity`).
    fn check_connectivity(
        &self,
        len: usize,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> Result<(), GameError> {
        let cells = (0..len).map(|i| word_cell(row, col, horizontal, i));
        if self.is_first_move {
            if cells.clone().any(|(r, c)| r == CENTER && c == CENTER) {
                return Ok(());
            }
            return Err(GameError::FirstMoveMustCoverCentre);
        }
        let grid = &self.board.board;
        for (r, c) in cells {
            if grid[r][c] != '-' {
                return Ok(());
            }
            let neighbours = [
                (r.wrapping_sub(1), c),
                (r + 1, c),
                (r, c.wrapping_sub(1)),
                (r, c + 1),
            ];
            if neighbours
                .iter()
                .any(|&(nr, nc)| in_bounds(nr, nc) && grid[nr][nc] != '-')
            {
                return Ok(());
            }
        }
        Err(GameError::NotConnected)
    }

    /// Tiles a play takes off `rack`, one per newly covered square, real tiles
    /// first and `'?'` once a letter runs out (`_tiles_used_for_word`). `None`
    /// if the rack cannot cover the play at all (`validate_rack_for_word`).
    fn tiles_for_play(
        &self,
        rack: &str,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> Option<Vec<char>> {
        let mut remaining: Vec<char> = rack.chars().collect();
        let mut used = Vec::new();
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if self.board.board[r][c] != '-' {
                continue;
            }
            if let Some(p) = remaining.iter().position(|&x| x == ch) {
                remaining.remove(p);
                used.push(ch);
            } else {
                let p = remaining.iter().position(|&x| x == '?')?;
                remaining.remove(p);
                used.push('?');
            }
        }
        Some(used)
    }

    // ── Actions ──────────────────────────────────────────────────────────────

    /// The current (human) player types a word (`POST /board/human-move`).
    /// Returns its score. In a competitive game the computer then has the
    /// turn: see `computer_to_move`.
    pub fn human_move(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> Result<u32, GameError> {
        if self.game_over {
            // The web does not check this here; a finished game has no turn.
            return Err(GameError::GameOver);
        }
        let word = word.to_lowercase();
        self.validate_typed_word(&word, row, col, horizontal)?;

        let competitive = self.mode == GameMode::Competitive;
        let pre_letters = self.players[self.current].letters.clone();
        let used = if competitive {
            Some(
                self.tiles_for_play(&pre_letters, &word, row, col, horizontal)
                    .ok_or(GameError::MissingTiles)?,
            )
        } else {
            None
        };
        // The real rack in competitive play (so blanks score 0); the word
        // itself in the sandbox, where every tile scores at face value.
        let letters_for_scoring = if competitive {
            pre_letters.as_str()
        } else {
            word.as_str()
        };
        let score =
            self.board
                .calculate_word_points(&word, row, col, horizontal, letters_for_scoring)?;

        // Rated against the position before the move.
        self.last_move_rating = (competitive && !pre_letters.is_empty())
            .then(|| move_rating(&self.board, &self.pack.lexicon, &pre_letters, score));

        self.push_undo();
        self.record_placement(&word, row, col, horizontal, self.current);
        self.board
            .place_word(&word, row, col, horizontal, used.as_deref())?;
        self.players[self.current].score += score as i32;
        self.is_first_move = false;

        if let Some(used) = &used {
            self.players[self.current].letters = crate::rack_after(&pre_letters, used);
            self.refill_rack(self.current);
            self.streaks.record(TurnResult::Played);
            self.check_game_over(self.current);
        }
        if !self.game_over {
            self.advance_turn();
        }
        Ok(score)
    }

    /// The current player passes without playing (`POST /board/skip`). The
    /// game only ends once nobody has played for the no-play limit.
    pub fn skip(&mut self) -> Result<(), GameError> {
        if self.game_over {
            return Err(GameError::GameOver);
        }
        self.push_undo();
        if self.mode == GameMode::Competitive {
            self.streaks.record(TurnResult::NoAction);
            self.check_game_over(self.current);
        }
        if !self.game_over {
            self.advance_turn();
        }
        Ok(())
    }

    /// Swap `letters` for new tiles instead of playing (`POST /board/exchange`):
    /// competitive games only, with at least seven tiles in the bag. Never
    /// counts toward the no-play streak, but it is scoreless.
    ///
    /// Deviation: the web returns the tiles to the bag *before* drawing, so a
    /// player can draw back what they just threw away. The rule is to draw the
    /// replacements first, then mix the discards in -- as the engine's own
    /// `Board::exchange_letters` does.
    pub fn exchange(&mut self, letters: &str) -> Result<(), GameError> {
        if self.game_over {
            return Err(GameError::GameOver);
        }
        if self.mode != GameMode::Competitive || self.bag.is_none() {
            return Err(GameError::WrongMode);
        }
        let letters: Vec<char> = letters.to_lowercase().chars().collect();
        if letters.is_empty() || letters.len() > RACK_SIZE {
            return Err(GameError::TilesNotOnRack);
        }
        if self.bag_remaining() < RACK_SIZE {
            return Err(GameError::BagTooSmall);
        }
        // Exact tiles only: exchanging a blank gives up that tile, it does not
        // stand in for anything.
        let mut rack: Vec<char> = self.players[self.current].letters.chars().collect();
        for ch in &letters {
            match rack.iter().position(|x| x == ch) {
                Some(p) => {
                    rack.remove(p);
                }
                None => return Err(GameError::TilesNotOnRack),
            }
        }

        self.push_undo();
        let bag = self.bag.as_mut().expect("checked above");
        let drawn = bag.draw(letters.len());
        bag.put_back(&letters, &mut self.rng);
        let player = &mut self.players[self.current];
        player.letters = rack.into_iter().chain(drawn).collect();
        self.streaks.record(TurnResult::Exchanged);
        self.check_game_over(self.current);
        if !self.game_over {
            self.advance_turn();
        }
        Ok(())
    }

    /// The current human resigns ("Poddaj się", `POST /board/pass`): a
    /// deliberate stop, unlike `skip`. Once every human has resigned the game
    /// ends, with no end-of-game rack adjustment.
    pub fn resign(&mut self) -> Result<(), GameError> {
        if self.game_over {
            return Err(GameError::GameOver);
        }
        if !self.resigned.contains(&self.current) {
            self.resigned.push(self.current);
            self.resigned.sort_unstable();
        }
        let all_humans_resigned = (0..self.players.len())
            .filter(|&i| !self.players[i].is_computer)
            .all(|i| self.resigned.contains(&i));
        if all_humans_resigned {
            self.game_over = true;
        } else {
            self.advance_turn();
        }
        Ok(())
    }

    /// Take back the last action (`POST /board/undo`). In a competitive game
    /// that includes the computer's reply, which was never recorded separately.
    pub fn undo(&mut self) -> Result<(), GameError> {
        let entry = self.history.pop().ok_or(GameError::NothingToUndo)?;
        self.board = Board::from_snapshot(self.pack.lang, &entry.board)?;
        for (i, p) in self.players.iter_mut().enumerate() {
            p.score = entry.player_scores[i];
            p.letters = entry.player_letters[i].clone();
        }
        self.current = entry.current_player;
        self.is_first_move = entry.is_first_move;
        self.move_number = entry.move_number;
        if let (Some(tiles), Some(bag)) = (entry.bag_tiles, self.bag.as_mut()) {
            bag.tiles = tiles;
        }
        self.last_computer_move = entry.last_computer_move;
        self.tile_owners = entry.tile_owners;
        self.streaks = entry.streaks;
        self.game_over = false;
        Ok(())
    }

    /// Advance an all-computer game by one move (`POST /board/next-move`).
    pub fn next_auto_move(&mut self) -> Result<ComputerMove, GameError> {
        if self.mode != GameMode::SandboxAuto {
            return Err(GameError::WrongMode);
        }
        if self.game_over {
            return Err(GameError::GameOver);
        }
        self.push_undo();
        self.computer_turn()
    }

    /// Sandbox: type in the rack the current seat holds (`POST /board/set-letters`).
    pub fn set_letters(&mut self, letters: &str) -> Result<(), GameError> {
        if self.mode != GameMode::Sandbox {
            return Err(GameError::WrongMode);
        }
        let letters = letters.to_lowercase();
        if letters.chars().count() > RACK_SIZE {
            return Err(GameError::RackTooLong);
        }
        self.players[self.current].letters = letters;
        Ok(())
    }

    /// Whether `sort` can be served in this language.
    fn check_sort(&self, sort: SortMode) -> Result<(), GameError> {
        if sort.needs_leave_net() && self.pack.leave_net.is_none() {
            return Err(GameError::SortUnavailable(sort));
        }
        Ok(())
    }

    /// Sandbox: the ten best plays for the current seat's rack (`POST /board/suggest`).
    pub fn sandbox_suggestions(&self, sort: SortMode) -> Result<Vec<Suggestion>, GameError> {
        if self.mode != GameMode::Sandbox {
            return Err(GameError::WrongMode);
        }
        self.check_sort(sort)?;
        self.suggest(&self.players[self.current].letters, 10, sort)
    }

    /// Competitive: twenty hints for the human's rack (`GET /board/hints`).
    pub fn hints(&self, sort: SortMode) -> Result<Vec<Suggestion>, GameError> {
        if self.mode != GameMode::Competitive {
            return Err(GameError::WrongMode);
        }
        self.check_sort(sort)?;
        self.suggest(&self.players[self.current].letters, 20, sort)
    }

    fn suggest(
        &self,
        letters: &str,
        n: usize,
        sort: SortMode,
    ) -> Result<Vec<Suggestion>, GameError> {
        Ok(rank_suggestions(
            &self.board,
            &self.pack.lexicon,
            self.pack.net(),
            letters,
            n,
            sort,
            SUGGEST_SIM_ITERATIONS,
            // From the clock: asking for a hint must not change how the game
            // unfolds, which drawing from the session's stream would.
            0,
        )?)
    }

    /// Sandbox: place a chosen suggestion for the current seat
    /// (`POST /board/computer-move`). Scored against that seat's rack; only
    /// claims to know which tiles were blanks when the rack really covers it.
    pub fn place_computer_word(
        &mut self,
        word: &str,
        row: usize,
        col: usize,
        horizontal: bool,
    ) -> Result<u32, GameError> {
        if self.mode != GameMode::Sandbox {
            return Err(GameError::WrongMode);
        }
        let word = word.to_lowercase();
        self.board
            .check_word_placement(&self.pack.lexicon, &word, row, col, horizontal)
            .map_err(|e| GameError::Engine(e.to_string()))?;
        let rack = self.players[self.current].letters.clone();
        let score = self
            .board
            .calculate_word_points(&word, row, col, horizontal, &rack)?;
        let used = self.tiles_for_play(&rack, &word, row, col, horizontal);

        self.push_undo();
        self.record_placement(&word, row, col, horizontal, self.current);
        // Which tiles the play took off the rack, blanks for letters it lacks
        // -- computed before placing, from the squares still empty.
        let mut left: Vec<char> = rack.chars().collect();
        for (i, ch) in word.chars().enumerate() {
            let (r, c) = word_cell(row, col, horizontal, i);
            if self.board.board[r][c] != '-' {
                continue;
            }
            if let Some(p) = left.iter().position(|&x| x == ch) {
                left.remove(p);
            } else if let Some(p) = left.iter().position(|&x| x == '?') {
                left.remove(p);
            }
        }
        self.board
            .place_word(&word, row, col, horizontal, used.as_deref())?;
        self.players[self.current].score += score as i32;
        self.is_first_move = false;
        self.players[self.current].letters = left.into_iter().collect();
        self.advance_turn();
        Ok(score)
    }

    /// What a typed word would score, or why it cannot be played
    /// (`POST /board/preview-score`). Changes nothing.
    pub fn preview_score(&self, word: &str, row: usize, col: usize, horizontal: bool) -> Preview {
        let word = word.to_lowercase();
        match self.validate_typed_word(&word, row, col, horizontal) {
            Err(GameError::NotInDictionary(_)) => return Preview::NotInDictionary,
            Err(_) => return Preview::InvalidPlacement,
            Ok(()) => {}
        }
        let letters = if self.mode == GameMode::Competitive {
            self.players[self.current].letters.clone()
        } else {
            word.clone()
        };
        match self
            .board
            .calculate_word_points(&word, row, col, horizontal, &letters)
        {
            Ok(score) => Preview::Score(score),
            Err(_) => Preview::ScoreError,
        }
    }

    // ── The computer ─────────────────────────────────────────────────────────

    /// Play the current seat's move at its own level and advance the turn
    /// (`computer_auto_play`). Records no undo entry of its own: in a
    /// competitive game it is part of the human action that preceded it.
    pub fn computer_turn(&mut self) -> Result<ComputerMove, GameError> {
        if self.game_over {
            return Err(GameError::GameOver);
        }
        let player = self.current;
        let play = self.pick_computer_play()?;

        let Some((word, score, (row, col, horizontal), _)) = play else {
            self.streaks.record(TurnResult::NoAction);
            self.check_game_over(player);
            if !self.game_over {
                self.advance_turn();
            }
            let passed = ComputerMove::passed();
            self.last_computer_move = Some(passed.clone());
            return Ok(passed);
        };

        let rack = self.players[player].letters.clone();
        let used = self
            .tiles_for_play(&rack, &word, row, col, horizontal)
            .ok_or_else(|| {
                GameError::Engine(format!(
                    "the bot chose '{word}', which its rack cannot cover"
                ))
            })?;
        self.record_placement(&word, row, col, horizontal, player);
        self.board
            .place_word(&word, row, col, horizontal, Some(&used))?;
        self.players[player].score += score as i32;
        self.is_first_move = false;
        self.players[player].letters = crate::rack_after(&rack, &used);
        self.refill_rack(player);
        self.streaks.record(TurnResult::Played);
        self.check_game_over(player);
        if !self.game_over {
            self.advance_turn();
        }
        let mv = ComputerMove {
            word,
            score,
            row,
            col,
            horizontal,
            passed: false,
        };
        self.last_computer_move = Some(mv.clone());
        Ok(mv)
    }

    /// The current seat's choice, without playing it.
    fn pick_computer_play(&mut self) -> Result<Option<crate::BestWord>, GameError> {
        let p = &self.players[self.current];
        let (level, strength, rack) = (p.difficulty, p.strength, p.letters.clone());
        let mode = engine_mode(level);
        let lexicon = self.pack.lexicon.clone();
        let net = self.pack.leave_net.clone();
        // A ranked level, or an engine whose net this language lacks: the
        // session clamps levels at creation, so the second only arises when a
        // restored game meets a missing model.
        let mode = if net.is_none() {
            EngineMode::Ranked
        } else {
            mode
        };

        if strength < 100 {
            // Weaken the selected engine's own ranking -- neural and
            // simulated rankings included -- rather than silently swapping in
            // a different bot. Weak simulations need only enough candidates
            // for their window, and fewer rollouts as strength drops.
            let (best, worst) = strength_rank_window(strength);
            let (sort, candidates, iterations) =
                match mode {
                    EngineMode::Ranked => (SortMode::Score, worst, SUGGEST_SIM_ITERATIONS),
                    EngineMode::Smart => (SortMode::Smart, worst.max(50), SUGGEST_SIM_ITERATIONS),
                    EngineMode::Sim => (
                        SortMode::Sim,
                        worst.max(20),
                        // max(8, round(100 * strength / 100)), as the web has it.
                        (((SUGGEST_SIM_ITERATIONS * strength as usize) as f64 / 100.0)
                            .round_ties_even() as usize)
                            .max(8),
                    ),
                };
            // Only a simulation draws a seed: the ranked and neural orderings
            // use no randomness, and must leave the stream where the web has it.
            let seed = if sort == SortMode::Sim {
                self.rng.next_seed()
            } else {
                0
            };
            let ranked = rank_suggestions(
                &self.board,
                &lexicon,
                net.as_deref(),
                &rack,
                candidates,
                sort,
                iterations,
                seed,
            )?;
            return Ok(
                pick_in_window(&ranked, best, worst, &mut self.rng).map(|s| {
                    (
                        s.word.clone(),
                        s.score,
                        (s.row, s.col, s.horizontal),
                        s.used.clone(),
                    )
                }),
            );
        }

        match mode {
            EngineMode::Ranked => {
                // Every level ranks candidates as StrategicPlayer does and then
                // reaches further down the list, so a weaker bot plays a worse
                // move rather than a differently chosen one.
                let (_, worst) = rank_window(level);
                let ranked = self
                    .board
                    .get_best_words(&lexicon, &rack, worst.max(1), true);
                Ok(pick_by_rank(&ranked, level, &mut self.rng).cloned())
            }
            EngineMode::Smart => {
                let opts = StaticOptions {
                    endgame_max_nodes: INTERACTIVE_ENDGAME_NODES,
                    ..StaticOptions::default()
                };
                let net = net.as_deref().expect("mode needs a net");
                Ok(choose_move(
                    &self.board,
                    &lexicon,
                    net,
                    &rack,
                    self.bag_remaining(),
                    &opts,
                )?
                .play)
            }
            EngineMode::Sim => {
                // Deviation: the web seeds a simulation from the clock. Seeding
                // it from the session's own stream makes a game -- and a saved
                // one resumed -- replay identically, and lets a benchmark compare
                // thread counts on the same positions.
                let opts = SimOptions {
                    endgame_max_nodes: INTERACTIVE_ENDGAME_NODES,
                    seed: self.rng.next_seed(),
                    ..SimOptions::default()
                };
                let net = net.as_deref().expect("mode needs a net");
                Ok(choose_move_sim(
                    &self.board,
                    &lexicon,
                    net,
                    &rack,
                    self.bag_remaining(),
                    &opts,
                )?
                .play)
            }
        }
    }
}

/// Bumped whenever `SavedGame` changes shape; older saves are refused rather
/// than half-read.
pub const SAVE_FORMAT_VERSION: u32 = 1;

/// A session as plain data. The language pack is not part of it -- it is
/// shared, and far too big -- so restoring takes the pack back in and checks
/// the save was made in the same language.
#[derive(Serialize, Deserialize)]
struct SavedGame {
    version: u32,
    language: String,
    mode: GameMode,
    board: BoardSnapshot,
    players: Vec<Player>,
    current: usize,
    is_first_move: bool,
    move_number: u32,
    history: Vec<UndoEntry>,
    bag: Option<TileBag>,
    last_computer_move: Option<ComputerMove>,
    tile_owners: Owners,
    game_over: bool,
    streaks: Streaks,
    resigned: Vec<usize>,
    last_move_rating: Option<u8>,
    rng: GameRng,
}

impl GameSession {
    /// The whole session -- position, racks, bag, undo history and RNG stream
    /// -- as JSON. An app saves this after every move: a phone kills
    /// background apps, and a game in progress must survive that.
    pub fn to_json(&self) -> String {
        let saved = SavedGame {
            version: SAVE_FORMAT_VERSION,
            language: self.pack.lang.code().to_string(),
            mode: self.mode,
            board: self.board.to_snapshot(),
            players: self.players.clone(),
            current: self.current,
            is_first_move: self.is_first_move,
            move_number: self.move_number,
            history: self.history.clone(),
            bag: self.bag.clone(),
            last_computer_move: self.last_computer_move.clone(),
            tile_owners: self.tile_owners,
            game_over: self.game_over,
            streaks: self.streaks,
            resigned: self.resigned.clone(),
            last_move_rating: self.last_move_rating,
            rng: self.rng.clone(),
        };
        serde_json::to_string(&saved).expect("a session always serializes")
    }

    /// Restore a session saved by `to_json`, in `pack`'s language.
    pub fn from_json(pack: LanguagePack, text: &str) -> Result<Self, GameError> {
        let bad = |m: String| GameError::InvalidSave(m);
        let saved: SavedGame = serde_json::from_str(text).map_err(|e| bad(e.to_string()))?;
        if saved.version != SAVE_FORMAT_VERSION {
            return Err(bad(format!(
                "save format {} (this build reads {SAVE_FORMAT_VERSION})",
                saved.version
            )));
        }
        if saved.language != pack.lang.code() {
            return Err(bad(format!(
                "the game is in '{}', not '{}'",
                saved.language,
                pack.lang.code()
            )));
        }
        let seats = saved.players.len();
        if seats == 0 || seats > MAX_PLAYERS || saved.current >= seats {
            return Err(bad("the players do not hold together".to_string()));
        }
        let owner_ok = |owners: &Owners| {
            owners
                .iter()
                .flatten()
                .flatten()
                .all(|&o| (o as usize) < seats)
        };
        if !owner_ok(&saved.tile_owners)
            || saved.history.iter().any(|e| {
                e.player_scores.len() != seats
                    || e.player_letters.len() != seats
                    || e.current_player >= seats
                    || !owner_ok(&e.tile_owners)
                    || Board::from_snapshot(pack.lang, &e.board).is_err()
            })
        {
            return Err(bad("the undo history does not hold together".to_string()));
        }
        let board =
            Board::from_snapshot(pack.lang, &saved.board).map_err(|e| bad(e.to_string()))?;
        let mut players = saved.players;
        for p in players.iter_mut() {
            p.difficulty = clamp_level(p.difficulty as i64, pack.max_level());
        }
        Ok(GameSession {
            pack,
            board,
            players,
            current: saved.current,
            is_first_move: saved.is_first_move,
            move_number: saved.move_number,
            history: saved.history,
            mode: saved.mode,
            bag: saved.bag,
            last_computer_move: saved.last_computer_move,
            tile_owners: saved.tile_owners,
            game_over: saved.game_over,
            streaks: saved.streaks,
            resigned: saved.resigned,
            last_move_rating: saved.last_move_rating,
            rng: saved.rng,
        })
    }
}
