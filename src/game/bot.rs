//! The decisions: which move a bot plays, how hints are ordered, and how good a
//! human's move was.
//!
//! Ports of `smart_player/player.py`'s `choose_move`, `sim_player.py`'s
//! `choose_move_sim`, and `web/game.py`'s `rank_suggestions` /
//! `compute_move_rating`. The web routes its top bot levels through those
//! shared functions so the web bot and the CLI players cannot drift; these are
//! the same functions, so the apps cannot drift from either.

use serde::{Deserialize, Serialize};

use crate::{
    leave_value, rack_after, value_error, BestWord, Board, EngineResult, LeaveNet, Lexicon,
    SimParams,
};

/// How heavily the learned leave value counts against a move's own score. The
/// optimum is a property of the checkpoint, retuned with every one -- see
/// `DEFAULT_LEAVE_WEIGHT` in `smart_player/player.py` for the sweep.
pub const DEFAULT_LEAVE_WEIGHT: f64 = 1.0;
/// Candidates pulled from the engine before ranking them by leave value.
pub const DEFAULT_N_CANDIDATES: usize = 50;
/// Standard rack capacity, the sanity bound on a deduced opponent rack.
pub const MAX_RACK: usize = 7;
/// The CLI's endgame node budget: median ~16k nodes, a measured 9.4 s tail.
pub const DEFAULT_ENDGAME_NODES: usize = 300_000;
/// The budget for someone waiting on the answer: keeps essentially all of the
/// strength while bounding the tail (the web's `WEB_ENDGAME_NODES`).
pub const INTERACTIVE_ENDGAME_NODES: usize = 50_000;
/// `sim_player.py`'s defaults for a bot's own decision.
pub const SIM_ITERATIONS: usize = 200;
pub const SIM_CANDIDATES: usize = 20;
pub const SIM_PLIES: usize = 1;
pub const SIM_LEAVE_WEIGHT: f64 = 1.0;
/// Rollouts per candidate when a human asks for `sim`-ordered hints: lower than
/// the bot's own 200 because someone is waiting on the response.
pub const SUGGEST_SIM_ITERATIONS: usize = 100;

/// A chosen play, plus what the search or simulation reported about it. `play`
/// is `None` when the bot has no move -- or, in an endgame, when passing beat
/// every play.
#[derive(Clone, Debug, Default)]
pub struct MoveChoice {
    pub play: Option<BestWord>,
    pub endgame_diff: Option<i32>,
    pub endgame_exact: Option<bool>,
    pub equity: Option<f64>,
    pub stderr: Option<f64>,
}

/// `choose_move`'s knobs; `Default` is the CLI player's configuration.
#[derive(Clone, Copy, Debug)]
pub struct StaticOptions {
    pub leave_weight: f64,
    pub use_endgame: bool,
    pub endgame_max_nodes: usize,
    pub n_candidates: usize,
}

impl Default for StaticOptions {
    fn default() -> Self {
        StaticOptions {
            leave_weight: DEFAULT_LEAVE_WEIGHT,
            use_endgame: true,
            endgame_max_nodes: DEFAULT_ENDGAME_NODES,
            n_candidates: DEFAULT_N_CANDIDATES,
        }
    }
}

/// `choose_move_sim`'s knobs; `Default` is `SimPlayer`'s configuration.
#[derive(Clone, Copy, Debug)]
pub struct SimOptions {
    pub iterations: usize,
    pub candidates: usize,
    pub plies: usize,
    pub leave_weight: f64,
    pub use_endgame: bool,
    pub endgame_max_nodes: usize,
    /// 0 picks one from the clock.
    pub seed: u64,
}

impl Default for SimOptions {
    fn default() -> Self {
        SimOptions {
            iterations: SIM_ITERATIONS,
            candidates: SIM_CANDIDATES,
            plies: SIM_PLIES,
            leave_weight: SIM_LEAVE_WEIGHT,
            use_endgame: true,
            endgame_max_nodes: DEFAULT_ENDGAME_NODES,
            seed: 0,
        }
    }
}

/// The opponent's rack once the bag is empty: exactly the tiles neither on the
/// board nor in our own rack, which a player may legitimately deduce.
///
/// `None` unless it looks like a real rack. `bag_remaining` comes from the
/// caller's own bag, which can disagree with the board (a scanned position
/// claiming an empty bag with 90 tiles unseen), and handing `solve_endgame` a
/// 90-tile rack does not fail -- it simply never returns.
fn endgame_opponent_rack(board: &Board, rack: &str) -> Option<String> {
    let tiles = board.lang.eval_letters();
    let opponent: String = board
        .unseen_tile_counts(rack)
        .iter()
        .enumerate()
        .flat_map(|(i, &n)| std::iter::repeat_n(tiles[i], n as usize))
        .collect();
    let len = opponent.chars().count();
    (len > 0 && len <= MAX_RACK).then_some(opponent)
}

/// The whole static decision for `rack` on `board`: once the bag is empty,
/// search the endgame exactly; otherwise rank the top candidates by score plus
/// the learned value of the leave.
///
/// `bag_remaining` is the caller's, not the board's: a session keeps its own
/// bag and never draws from the engine's.
pub fn choose_move(
    board: &Board,
    lexicon: &Lexicon,
    net: &LeaveNet,
    rack: &str,
    bag_remaining: usize,
    opts: &StaticOptions,
) -> EngineResult<MoveChoice> {
    if opts.use_endgame && bag_remaining == 0 {
        if let Some(opponent) = endgame_opponent_rack(board, rack) {
            let out = board.solve_endgame(lexicon, rack, &opponent, opts.endgame_max_nodes)?;
            return Ok(MoveChoice {
                play: out.best,
                endgame_diff: Some(out.differential),
                endgame_exact: Some(out.exact),
                ..MoveChoice::default()
            });
        }
    }

    let candidates = board.get_best_words(lexicon, rack, opts.n_candidates, true);
    if candidates.is_empty() {
        return Ok(MoveChoice::default());
    }
    let feats = board.board_features();
    let unseen = unseen_total(board, rack);
    // Strictly greater, so an exact tie still falls to the higher-scoring
    // candidate -- the first at the maximum, as Python's `max` keeps.
    let mut best_i = 0;
    let mut best_v = f64::NEG_INFINITY;
    for (i, (_, score, _, used)) in candidates.iter().enumerate() {
        let leave = rack_after(rack, used);
        let v = *score as f64
            + opts.leave_weight * leave_value(net, board.lang, &leave, unseen, &feats) as f64;
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    Ok(MoveChoice {
        play: candidates.into_iter().nth(best_i),
        ..MoveChoice::default()
    })
}

/// Pick a move by simulation. With the bag empty there is nothing left to
/// sample, so it defers to `choose_move`'s endgame search instead.
pub fn choose_move_sim(
    board: &Board,
    lexicon: &Lexicon,
    net: &LeaveNet,
    rack: &str,
    bag_remaining: usize,
    opts: &SimOptions,
) -> EngineResult<MoveChoice> {
    if opts.use_endgame && bag_remaining == 0 {
        let static_opts = StaticOptions {
            leave_weight: opts.leave_weight,
            use_endgame: true,
            endgame_max_nodes: opts.endgame_max_nodes,
            ..StaticOptions::default()
        };
        return choose_move(board, lexicon, net, rack, bag_remaining, &static_opts);
    }
    let params = SimParams {
        candidates: opts.candidates,
        iterations: opts.iterations,
        plies: opts.plies,
        leave_weight: opts.leave_weight,
        seed: opts.seed,
        ..SimParams::default()
    };
    let best = board
        .simulate(lexicon, net, rack, params)?
        .into_iter()
        .next();
    Ok(match best {
        Some(c) => MoveChoice {
            play: Some(c.play),
            equity: Some(c.equity),
            stderr: Some(c.stderr),
            ..MoveChoice::default()
        },
        None => MoveChoice::default(),
    })
}

/// What a hint list is ordered by. They answer different questions: what
/// scores most this turn; what is worth most counting the rack it leaves; and
/// what survives the opponent's reply.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SortMode {
    Score,
    Smart,
    Sim,
}

impl SortMode {
    /// Whether this ordering consults the learned leave net, and so is only
    /// available in a language that has one trained.
    pub fn needs_leave_net(self) -> bool {
        matches!(self, SortMode::Smart | SortMode::Sim)
    }
}

/// One ranked play. `value` is the number it was ranked by -- the raw score,
/// the score plus the leave it keeps, or the simulated equity -- so a reordering
/// is explained rather than mysterious.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Suggestion {
    pub word: String,
    pub score: u32,
    pub row: usize,
    pub col: usize,
    pub horizontal: bool,
    /// One tile per newly covered square, `'?'` where a blank stands in.
    pub used: Vec<char>,
    pub value: f64,
}

impl Suggestion {
    fn from_play((word, score, (row, col, horizontal), used): BestWord, value: f64) -> Self {
        Suggestion {
            word,
            score,
            row,
            col,
            horizontal,
            used,
            value,
        }
    }

    /// Every square the word covers, existing tiles included -- what a board
    /// highlights.
    pub fn cells(&self) -> Vec<(usize, usize)> {
        (0..self.word.chars().count())
            .map(|i| crate::word_cell(self.row, self.col, self.horizontal, i))
            .collect()
    }
}

/// Python's `round(x, 1)`: half to even on the exact binary value, which is
/// what Rust's float formatting does too. The smart ordering sorts on the
/// rounded value, so this has to match for the lists to agree.
pub(crate) fn round1(x: f64) -> f64 {
    format!("{x:.1}").parse().unwrap_or(x)
}

fn unseen_total(board: &Board, rack: &str) -> usize {
    board
        .unseen_tile_counts(rack)
        .iter()
        .map(|&c| c as usize)
        .sum()
}

/// Top `n` plays for `letters`, ordered by `sort`. `seed` drives a `Sim`
/// ordering's rollouts; 0 picks one from the clock.
#[allow(clippy::too_many_arguments)]
pub fn rank_suggestions(
    board: &Board,
    lexicon: &Lexicon,
    net: Option<&LeaveNet>,
    letters: &str,
    n: usize,
    sort: SortMode,
    sim_iterations: usize,
    seed: u64,
) -> EngineResult<Vec<Suggestion>> {
    let net = match (sort.needs_leave_net(), net) {
        (true, None) => {
            return Err(value_error(format!(
                "'{sort:?}' ordering needs a trained leave net, and '{}' has none",
                board.lang.code()
            )))
        }
        (_, net) => net,
    };
    if letters.is_empty() {
        return Ok(Vec::new());
    }

    if sort == SortMode::Sim {
        // One simulation ranks the whole set; `batch == iterations` keeps every
        // candidate rather than only the survivors of pruning.
        let params = SimParams {
            candidates: n.max(1),
            iterations: sim_iterations,
            plies: 1,
            batch: sim_iterations,
            seed,
            ..SimParams::default()
        };
        let net = net.expect("checked above");
        return Ok(board
            .simulate(lexicon, net, letters, params)?
            .into_iter()
            .take(n)
            .map(|c| Suggestion::from_play(c.play, round1(c.equity)))
            .collect());
    }

    let mut suggestions: Vec<Suggestion> = board
        .get_best_words(lexicon, letters, n, true)
        .into_iter()
        .map(|play| {
            let score = play.1 as f64;
            Suggestion::from_play(play, score)
        })
        .collect();
    if sort == SortMode::Smart && !suggestions.is_empty() {
        let net = net.expect("checked above");
        // Score plus what the play leaves behind: the quantity `choose_move`
        // ranks by, so the list agrees with what the bot would do.
        let feats = board.board_features();
        let unseen = unseen_total(board, letters);
        for s in &mut suggestions {
            let leave = rack_after(letters, &s.used);
            let v = leave_value(net, board.lang, &leave, unseen, &feats) as f64;
            s.value = round1(s.score as f64 + DEFAULT_LEAVE_WEIGHT * v);
        }
        // Stable, so equal values keep their score order.
        suggestions.sort_by(|a, b| b.value.total_cmp(&a.value));
    }
    Ok(suggestions)
}

/// Rate `actual_score` 0-100 against the best and worst of the top 999 plays
/// `letters` had (`compute_move_rating`).
pub fn move_rating(board: &Board, lexicon: &Lexicon, letters: &str, actual_score: u32) -> u8 {
    review_move(board, lexicon, letters, actual_score).0
}

/// `move_rating`, and the play it measured `actual_score` against: the
/// highest-scoring one, first in score order as the hints list it. `None`
/// only when `letters` had no play at all.
pub fn review_move(
    board: &Board,
    lexicon: &Lexicon,
    letters: &str,
    actual_score: u32,
) -> (u8, Option<BestWord>) {
    let moves = board.get_best_words(lexicon, letters, 999, true);
    let (Some(best), Some(worst)) = (
        moves.iter().map(|m| m.1).max(),
        moves.iter().map(|m| m.1).min(),
    ) else {
        return (100, None);
    };
    let top = moves.iter().find(|m| m.1 == best).cloned();
    if best == worst {
        return (100, top);
    }
    let rating = (actual_score as f64 - worst as f64) / (best - worst) as f64 * 100.0;
    (rating.round_ties_even().clamp(0.0, 100.0) as u8, top)
}
