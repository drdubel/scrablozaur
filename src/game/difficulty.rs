//! Difficulty as one dial, level 1 (weakest) to 10 -- the port of
//! `src/strategy.py`'s rank windows and `web/difficulty.py`'s level -> engine
//! mapping.
//!
//! Levels up to `MAX_RANKED_LEVEL` play a *worse move from the same list*: every
//! level ranks candidates identically and only reaches further down it. The two
//! above swap the decision itself -- the leave-aware static player, then
//! Monte-Carlo simulation -- and so need a trained leave net. The words a
//! player reads about each level live in the app; this owns the numbers they
//! are derived from.

use serde::{Deserialize, Serialize};

use super::rng::GameRng;

pub const MIN_LEVEL: u8 = 1;
pub const MAX_LEVEL: u8 = 10;
pub const MAX_RANKED_LEVEL: u8 = 8;
/// 9: weighs the rack it leaves behind.
pub const SMART_LEVEL: u8 = MAX_RANKED_LEVEL + 1;
/// 10: simulates the opponent's reply.
pub const SIM_LEVEL: u8 = MAX_LEVEL;
/// Middle of the dial: something a decent club player has to work for.
pub const DEFAULT_LEVEL: u8 = 5;

/// How deep into the best-first candidate list `MIN_LEVEL` reaches. The window
/// shrinks geometrically from there to (1, 1) at `MAX_RANKED_LEVEL`, because
/// move quality falls off steeply at the top of the list and slowly further
/// down -- a linear ramp would make the bottom half of the dial feel identical.
const WORST_RANK_AT_MIN_LEVEL: f64 = 40.0;

/// Force `level` into `MIN_LEVEL..=ceiling`. A language with no trained leave
/// net cannot field levels 9-10, so callers that know the language pass its
/// own `max_level_for`.
pub fn clamp_level(level: i64, ceiling: u8) -> u8 {
    level.min(ceiling as i64).max(MIN_LEVEL as i64) as u8
}

/// The (best, worst) 1-based rank window `level` picks from. Levels above
/// `MAX_RANKED_LEVEL` clamp to it: what makes them stronger is a better list,
/// not a different rank.
pub fn rank_window(level: u8) -> (usize, usize) {
    let level = clamp_level(level as i64, MAX_LEVEL);
    let span = (MAX_RANKED_LEVEL - MIN_LEVEL) as f64;
    let steps_below_top = MAX_RANKED_LEVEL.saturating_sub(level) as f64;
    // Python's round(), half to even -- the windows must match the web's.
    let worst = (WORST_RANK_AT_MIN_LEVEL
        .powf(steps_below_top / span)
        .round_ties_even() as usize)
        .max(1);
    (worst.div_ceil(2).max(1), worst)
}

/// A 1-100% strength dial on top of any engine: a rank window into that
/// engine's own best-first ranking, with the same geometric falloff as the
/// level dial. Move quality, not a calibrated win probability.
pub fn strength_rank_window(strength: u8) -> (usize, usize) {
    let strength = strength.clamp(1, 100);
    let (_, weakest_rank) = rank_window(MIN_LEVEL);
    let exponent = (100 - strength) as f64 / 99.0;
    let worst = ((weakest_rank as f64).powf(exponent).round_ties_even() as usize).max(1);
    (worst.div_ceil(2).max(1), worst)
}

/// Pick from a best-first list by `level`'s rank window, uniformly inside it.
/// Clamps to the list's length, so a position offering three legal plays
/// still works (it falls back to the worst it has).
pub fn pick_by_rank<'a, T>(ranked: &'a [T], level: u8, rng: &mut GameRng) -> Option<&'a T> {
    if ranked.is_empty() {
        return None;
    }
    let (best, worst) = rank_window(level);
    pick_in_window(ranked, best, worst, rng)
}

/// `ranked[randint(lo, hi) - 1]` with the window clamped to the list.
pub(crate) fn pick_in_window<'a, T>(
    ranked: &'a [T],
    best: usize,
    worst: usize,
    rng: &mut GameRng,
) -> Option<&'a T> {
    if ranked.is_empty() {
        return None;
    }
    let hi = worst.min(ranked.len());
    let lo = best.min(hi);
    Some(&ranked[rng.randint(lo, hi) - 1])
}

/// Which decision procedure a level runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EngineMode {
    /// Best-first list, pick from a rank window.
    Ranked,
    /// Score plus the learned value of the rack left behind.
    Smart,
    /// Monte-Carlo rollouts of the opponent's reply.
    Sim,
}

pub fn engine_mode(level: u8) -> EngineMode {
    let level = clamp_level(level as i64, MAX_LEVEL);
    if level >= SIM_LEVEL {
        EngineMode::Sim
    } else if level >= SMART_LEVEL {
        EngineMode::Smart
    } else {
        EngineMode::Ranked
    }
}

/// The strongest level a language can field: levels 9 and 10 consult a learned
/// leave net, and a net is only valid for the alphabet it was trained on.
pub fn max_level_for(has_leave_net: bool) -> u8 {
    if has_leave_net {
        MAX_LEVEL
    } else {
        MAX_RANKED_LEVEL
    }
}

/// The mechanical facts about one notch of the dial, for the app to describe.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelInfo {
    pub level: u8,
    pub engine: EngineMode,
    /// Rank window into the best-first list; `None` once a level stops using one.
    pub rank_window: Option<(usize, usize)>,
    /// Spends noticeable time thinking.
    pub slow: bool,
}

pub fn describe_level(level: u8) -> LevelInfo {
    let level = clamp_level(level as i64, MAX_LEVEL);
    let engine = engine_mode(level);
    LevelInfo {
        level,
        engine,
        rank_window: (engine == EngineMode::Ranked).then(|| rank_window(level)),
        slow: engine == EngineMode::Sim,
    }
}

/// Every level a game in a language with (or without) a leave net can be
/// played at.
pub fn all_levels(has_leave_net: bool) -> Vec<LevelInfo> {
    (MIN_LEVEL..=max_level_for(has_leave_net))
        .map(describe_level)
        .collect()
}
