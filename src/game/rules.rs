//! When a game ends and how it is finally scored -- the port of
//! `src/rules.py`, which is the one definition every way of playing a game
//! obeys (self-play, the CLI benchmark, the web session, and now this).

use serde::{Deserialize, Serialize};

use super::Player;
use crate::Language;

/// A game ends once this many consecutive turns pass with nobody playing a
/// word. Only a *genuine* no-action turn counts: no legal play and no exchange
/// available. Exchanging is a real, repeatable action with no limit of its own.
pub const NO_PLAY_LIMIT: u32 = 2;

/// ...but exchanging being unlimited would let two players exchange at each
/// other forever, so *scoreless* turns are capped as well, as the tournament
/// rule does. An exchange is scoreless.
pub const NO_SCORE_LIMIT: u32 = 6;

/// What a player's turn actually was. Naming the three cases is what keeps
/// "exchanged" and "stuck" from being conflated -- a bug that once ended games
/// after a single round.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TurnResult {
    Played,
    Exchanged,
    NoAction,
}

/// The two counters that end a game, and the rule for reading them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Streaks {
    pub no_play: u32,
    pub no_score: u32,
}

impl Streaks {
    pub fn record(&mut self, result: TurnResult) {
        if result == TurnResult::Played {
            self.no_play = 0;
            self.no_score = 0;
            return;
        }
        // An exchange is a real action, so it never counts toward the no-play
        // streak -- but it scores nothing, so it does count toward that one.
        self.no_score += 1;
        if result == TurnResult::NoAction {
            self.no_play += 1;
        }
    }

    /// True once the game should stop, nobody having gone out.
    pub fn exhausted(&self) -> bool {
        self.no_play >= NO_PLAY_LIMIT || self.no_score >= NO_SCORE_LIMIT
    }
}

/// A player goes out by playing their last tile with the bag empty.
pub fn went_out(letters: &str, bag_remaining: usize) -> bool {
    letters.is_empty() && bag_remaining == 0
}

/// The standard final adjustment, applied in place.
///
/// Whoever goes out gains the summed rack value of everyone else and keeps
/// their own score (they hold nothing). If instead the game ran out of turns,
/// every player simply loses the value of what they are still holding.
pub fn apply_end_of_game_scoring(
    lang: &Language,
    players: &mut [Player],
    went_out_idx: Option<usize>,
) {
    match went_out_idx {
        Some(out) => {
            let others: i32 = players
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != out)
                .map(|(_, p)| lang.rack_points(&p.letters))
                .sum();
            players[out].score += others;
            for (i, p) in players.iter_mut().enumerate() {
                if i != out {
                    p.score -= lang.rack_points(&p.letters);
                }
            }
        }
        None => {
            for p in players.iter_mut() {
                p.score -= lang.rack_points(&p.letters);
            }
        }
    }
}
