//! The bot-vs-bot benchmark (`run_benchmark` in `web/game.py`): many automatic
//! games between the same seats, folded into per-seat statistics plus the
//! move-by-move record of the single best game.
//!
//! Games are independent, so they run in parallel -- on the engine's own pool,
//! the one move generation and simulation already use, so a game's internal
//! parallelism shares those threads instead of oversubscribing the CPU. (The
//! web needs a process pool for this; here a thread does.)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;

use super::session::{
    GameError, GameMode, GameSession, LanguagePack, Player, PlayerConfig, Winner,
};
use crate::{gen_pool, splitmix64, time_seed, BOARD_SIZE};

/// Defensive cap on moves per game: the no-play and scoreless limits, and
/// going out with an empty bag, always end a real game long before this.
const MAX_GAME_MOVES: usize = 200;

/// Shared with whoever is watching a run: how far it has got, and a way to
/// stop it. A cancelled run returns what it finished.
#[derive(Debug, Default)]
pub struct BenchmarkControl {
    cancelled: AtomicBool,
    games_done: AtomicUsize,
}

impl BenchmarkControl {
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Relaxed)
    }

    pub fn games_done(&self) -> usize {
        self.games_done.load(Ordering::Relaxed)
    }
}

type Grid<T> = [[T; BOARD_SIZE]; BOARD_SIZE];

/// One move of the recorded game, with the position after it -- what a replay
/// steps through.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BenchmarkMove {
    pub player: usize,
    pub word: String,
    pub score: u32,
    pub row: usize,
    pub col: usize,
    pub horizontal: bool,
    pub passed: bool,
    pub board: Grid<char>,
    pub blanks: Grid<bool>,
    pub scores_after: Vec<i32>,
    pub letters_after: Vec<String>,
    pub tile_owners: Grid<Option<u8>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkPlayerStats {
    pub name: String,
    pub difficulty: u8,
    pub strength: u8,
    pub games_played: usize,
    pub wins: usize,
    pub ties: usize,
    pub total_score: i64,
    pub high_score: i32,
    pub low_score: i32,
    pub words_played: usize,
    pub total_word_score: u64,
}

impl BenchmarkPlayerStats {
    pub fn avg_score(&self) -> f64 {
        if self.games_played == 0 {
            0.0
        } else {
            self.total_score as f64 / self.games_played as f64
        }
    }

    pub fn avg_word_score(&self) -> f64 {
        if self.words_played == 0 {
            0.0
        } else {
            self.total_word_score as f64 / self.words_played as f64
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct BestGame {
    pub winner: Winner,
    pub winner_score: i32,
    pub final_players: Vec<Player>,
    pub moves: Vec<BenchmarkMove>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BenchmarkResult {
    pub games_played: usize,
    pub duration_ms: u64,
    pub player_stats: Vec<BenchmarkPlayerStats>,
    pub best_game: Option<BestGame>,
    pub avg_game_length: f64,
    /// The longest word played in any game, and what it scored.
    pub longest_word: Option<(String, u32)>,
    pub highest_single_move_score: Option<u32>,
}

struct SimulatedGame {
    players: Vec<Player>,
    moves: Vec<BenchmarkMove>,
}

fn simulate_game(
    pack: &LanguagePack,
    players: &[PlayerConfig],
    seed: u64,
    control: &BenchmarkControl,
) -> Result<SimulatedGame, GameError> {
    let mut session =
        GameSession::new(pack.clone(), GameMode::SandboxAuto, players.to_vec(), seed)?;
    let mut moves = Vec::new();
    while !session.is_over() && moves.len() < MAX_GAME_MOVES && !control.is_cancelled() {
        let player = session.current_player_index();
        let mv = session.computer_turn()?;
        let state = session.state();
        moves.push(BenchmarkMove {
            player,
            word: mv.word,
            score: mv.score,
            row: mv.row,
            col: mv.col,
            horizontal: mv.horizontal,
            passed: mv.passed,
            board: state.board,
            blanks: state.blanks,
            scores_after: session.players().iter().map(|p| p.score).collect(),
            letters_after: session
                .players()
                .iter()
                .map(|p| p.letters.clone())
                .collect(),
            tile_owners: state.tile_owners,
        });
    }
    Ok(SimulatedGame {
        players: session.players().to_vec(),
        moves,
    })
}

/// Running totals, folded in as games finish. Ties in "longest word" and
/// "best game" go to the earliest game, so a run's result does not depend on
/// which thread finished first.
struct Totals {
    stats: Vec<BenchmarkPlayerStats>,
    games: usize,
    moves: usize,
    longest: Option<(usize, usize, String, u32)>,
    highest: u32,
    best: Option<(i32, usize, BestGame)>,
}

impl Totals {
    fn fold(&mut self, index: usize, game: SimulatedGame) {
        self.games += 1;
        self.moves += game.moves.len();
        for (mi, mv) in game.moves.iter().enumerate() {
            if mv.passed {
                continue;
            }
            let s = &mut self.stats[mv.player];
            s.words_played += 1;
            s.total_word_score += mv.score as u64;
            self.highest = self.highest.max(mv.score);
            let len = mv.word.chars().count();
            let longer = match &self.longest {
                None => true,
                Some((gi, mj, w, _)) => {
                    let best_len = w.chars().count();
                    len > best_len || (len == best_len && (index, mi) < (*gi, *mj))
                }
            };
            if longer {
                self.longest = Some((index, mi, mv.word.clone(), mv.score));
            }
        }

        let top = game.players.iter().map(|p| p.score).max().unwrap_or(0);
        let winners: Vec<usize> = (0..game.players.len())
            .filter(|&i| game.players[i].score == top)
            .collect();
        for (i, p) in game.players.iter().enumerate() {
            let s = &mut self.stats[i];
            s.games_played += 1;
            s.total_score += p.score as i64;
            if s.games_played == 1 {
                s.high_score = p.score;
                s.low_score = p.score;
            } else {
                s.high_score = s.high_score.max(p.score);
                s.low_score = s.low_score.min(p.score);
            }
            if p.score == top {
                if winners.len() > 1 {
                    s.ties += 1;
                } else {
                    s.wins += 1;
                }
            }
        }

        let better = match &self.best {
            None => true,
            Some((score, gi, _)) => top > *score || (top == *score && index < *gi),
        };
        if better {
            let winner = if winners.len() == 1 {
                Winner::Player(winners[0])
            } else {
                Winner::Tie
            };
            self.best = Some((
                top,
                index,
                BestGame {
                    winner,
                    winner_score: top,
                    final_players: game.players,
                    moves: game.moves,
                },
            ));
        }
    }
}

/// Play `games` automatic games between `players` and report per-seat
/// statistics, plus the full detail of the game with the highest score for any
/// one seat. `seed` 0 deals from the clock; anything else replays the same run.
pub fn run_benchmark(
    pack: &LanguagePack,
    players: &[PlayerConfig],
    games: usize,
    seed: u64,
    control: &BenchmarkControl,
) -> Result<BenchmarkResult, GameError> {
    let start = Instant::now();
    // Validates the seats once, up front, rather than failing in every game.
    GameSession::new(pack.clone(), GameMode::SandboxAuto, players.to_vec(), 1)?;
    let base = if seed == 0 { time_seed() } else { seed };

    let totals = Mutex::new(Totals {
        stats: players
            .iter()
            .map(|p| BenchmarkPlayerStats {
                name: p.name.clone(),
                difficulty: p.difficulty.min(pack.max_level()),
                strength: p.strength.clamp(1, 100),
                games_played: 0,
                wins: 0,
                ties: 0,
                total_score: 0,
                high_score: 0,
                low_score: 0,
                words_played: 0,
                total_word_score: 0,
            })
            .collect(),
        games: 0,
        moves: 0,
        longest: None,
        highest: 0,
        best: None,
    });
    let first_error: Mutex<Option<GameError>> = Mutex::new(None);

    gen_pool().install(|| {
        (0..games).into_par_iter().for_each(|i| {
            if control.is_cancelled() {
                return;
            }
            let game_seed = splitmix64(base ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
            match simulate_game(pack, players, game_seed, control) {
                // A game cut short by cancellation is not a game played.
                Ok(_) if control.is_cancelled() => {}
                Ok(game) => {
                    totals
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .fold(i, game);
                    control.games_done.fetch_add(1, Ordering::Relaxed);
                }
                Err(e) => {
                    first_error
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .get_or_insert(e);
                    control.cancel();
                }
            }
        })
    });

    if let Some(e) = first_error.into_inner().unwrap_or_else(|e| e.into_inner()) {
        return Err(e);
    }
    let t = totals.into_inner().unwrap_or_else(|e| e.into_inner());
    Ok(BenchmarkResult {
        games_played: t.games,
        duration_ms: start.elapsed().as_millis() as u64,
        player_stats: t.stats,
        best_game: t.best.map(|(_, _, g)| g),
        avg_game_length: if t.games == 0 {
            0.0
        } else {
            t.moves as f64 / t.games as f64
        },
        longest_word: t.longest.map(|(_, _, w, s)| (w, s)),
        highest_single_move_score: (t.highest > 0).then_some(t.highest),
    })
}
