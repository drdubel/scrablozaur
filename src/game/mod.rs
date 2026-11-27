//! A game of Scrabble as the apps play it: the session state machine and its
//! rules, the bots at every difficulty level, hints, move rating, the scan
//! assistant's suggestions, and the bot-vs-bot benchmark.
//!
//! The Rust port of what the web app keeps in Python -- `web/game.py`, the
//! game, board and scan routers, `src/rules.py`, `src/strategy.py`'s rank windows,
//! `web/difficulty.py`, and `smart_player`'s `choose_move` / `choose_move_sim`.
//! The web is the specification; where this differs it says so at the spot.

mod assistant;
mod bench;
mod bot;
mod difficulty;
mod ladder;
mod rng;
mod rules;
mod session;

pub use assistant::{check_play, scan_suggestions, PlayCheck};
pub use bench::{
    run_benchmark, BenchmarkControl, BenchmarkMove, BenchmarkPlayerStats, BenchmarkResult, BestGame,
};
pub use bot::{
    choose_move, choose_move_sim, move_rating, rank_suggestions, review_equity, review_move,
    EquityCheck, MoveChoice, SimOptions, SortMode, StaticOptions, Suggestion, TakenTurn,
    DEFAULT_ENDGAME_NODES, DEFAULT_LEAVE_WEIGHT, INTERACTIVE_ENDGAME_NODES, SUGGEST_SIM_ITERATIONS,
};
pub use difficulty::{
    all_levels, clamp_level, describe_level, engine_mode, max_level_for, pick_by_rank, rank_window,
    strength_rank_window, EngineMode, LevelInfo, DEFAULT_LEVEL, MAX_LEVEL, MAX_RANKED_LEVEL,
    MIN_LEVEL, SIM_LEVEL, SMART_LEVEL,
};
pub use ladder::{
    anchor_seat, bot_rating, fit_ratings, ladder_plan, ladder_source, run_ladder, LadderMatch,
    LadderSeat, ANCHOR_RATING, LADDER_STRENGTHS,
};
pub use rng::GameRng;
pub use rules::{
    apply_end_of_game_scoring, went_out, Streaks, TurnResult, NO_PLAY_LIMIT, NO_SCORE_LIMIT,
};
pub use session::{
    ComputerMove, EndgameReview, EquityReview, GameError, GameMode, GameSession, GameState,
    LanguagePack, PlayRecord, Player, PlayerConfig, PlayerView, Preview, TileBag, TurnAction,
    TurnRecord, Winner, SAVE_FORMAT_VERSION,
};

#[cfg(test)]
mod tests;
