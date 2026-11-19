//! What the bots are worth as opponents: an Elo rating for every engine and
//! strength a game can field, measured by playing them against each other
//! (`scrablozaur bot-ladder`), so that a player's results against them become
//! a rating of the player.
//!
//! Extension: the web keeps no ratings.

use super::bench::{run_benchmark, BenchmarkControl};
use super::difficulty::{
    clamp_level, engine_mode, max_level_for, EngineMode, MAX_RANKED_LEVEL, MIN_LEVEL, SIM_LEVEL,
    SMART_LEVEL,
};
use super::session::{GameError, LanguagePack, PlayerConfig};

/// The strengths measured for each engine; the ones between are read off the
/// straight line joining their neighbours.
pub const LADDER_STRENGTHS: [u8; 6] = [1, 10, 25, 50, 75, 100];

/// What every ladder is anchored to: the greedy engine at full strength
/// (level 8), which always plays the highest-scoring move.
pub const ANCHOR_RATING: f64 = 1500.0;

/// One language's ladder: the Elo of each engine at each of
/// `LADDER_STRENGTHS`. An engine the language cannot field has none.
struct Ladder {
    code: &'static str,
    ranked: [f64; 6],
    smart: Option<[f64; 6]>,
    sim: Option<[f64; 6]>,
}

/// Measured by `bot-ladder` (see `run_ladder`), 200 games a pairing;
/// re-measure after retraining a leave net or rebuilding a dictionary, since
/// both change how the bots play.
static LADDERS: &[Ladder] = &[
    Ladder {
        code: "pl",
        ranked: [230.0, 351.0, 648.0, 995.0, 1197.0, 1500.0],
        smart: Some([209.0, 392.0, 716.0, 1100.0, 1287.0, 1537.0]),
        sim: Some([286.0, 432.0, 744.0, 1107.0, 1310.0, 1583.0]),
    },
    Ladder {
        code: "en",
        ranked: [336.0, 441.0, 722.0, 1051.0, 1231.0, 1500.0],
        smart: Some([307.0, 504.0, 803.0, 1154.0, 1318.0, 1606.0]),
        sim: Some([381.0, 527.0, 815.0, 1200.0, 1340.0, 1645.0]),
    },
];

/// Elo of a computer at `level` and `strength` in language `code`, or `None`
/// where no ladder has been measured. Levels are clamped as a game clamps
/// them, so a language without a leave net rates its level 10 as level 8.
pub fn bot_rating(code: &str, has_leave_net: bool, level: u8, strength: u8) -> Option<f64> {
    let ladder = LADDERS.iter().find(|l| l.code == code)?;
    let level = clamp_level(level as i64, max_level_for(has_leave_net));
    let strength = strength.clamp(1, 100);
    let (row, strength) = match engine_mode(level) {
        // Below full strength a ranked bot picks from the strength's window,
        // whatever its level; at full strength from the level's -- which is
        // the window level 8 uses at the strength found here.
        EngineMode::Ranked if strength == 100 => (&ladder.ranked, equivalent_strength(level)),
        EngineMode::Ranked => (&ladder.ranked, strength as f64),
        EngineMode::Smart => (ladder.smart.as_ref()?, strength as f64),
        EngineMode::Sim => (ladder.sim.as_ref()?, strength as f64),
    };
    Some(interpolate(row, strength))
}

/// The level-8 strength whose rank window is ranked `level`'s at full
/// strength: both windows end at 40 to a power, `(100 - s) / 99` for a
/// strength and `(8 - level) / 7` for a level.
fn equivalent_strength(level: u8) -> f64 {
    let steps = MAX_RANKED_LEVEL.saturating_sub(level) as f64;
    100.0 - 99.0 * steps / (MAX_RANKED_LEVEL - MIN_LEVEL) as f64
}

fn interpolate(row: &[f64; 6], strength: f64) -> f64 {
    let s = LADDER_STRENGTHS.map(|s| s as f64);
    let i = s
        .windows(2)
        .position(|w| strength <= w[1])
        .unwrap_or(s.len() - 2);
    let t = ((strength - s[i]) / (s[i + 1] - s[i])).clamp(0.0, 1.0);
    row[i] + t * (row[i + 1] - row[i])
}

/// A computer the ladder measures: `(level, strength)`.
pub type LadderSeat = (u8, u8);

/// What one pairing of seats came to.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LadderMatch {
    pub a: usize,
    pub b: usize,
    pub a_wins: usize,
    pub b_wins: usize,
    pub ties: usize,
}

/// The seats to measure and the pairings to play between them: neighbouring
/// strengths of one engine, and one strength across neighbouring engines --
/// close enough that a result says something, rather than a foregone
/// conclusion a fit cannot use.
pub fn ladder_plan(has_leave_net: bool) -> (Vec<LadderSeat>, Vec<(usize, usize)>) {
    let levels: &[u8] = if has_leave_net {
        &[MAX_RANKED_LEVEL, SMART_LEVEL, SIM_LEVEL]
    } else {
        &[MAX_RANKED_LEVEL]
    };
    let k = LADDER_STRENGTHS.len();
    let seats = levels
        .iter()
        .flat_map(|&l| LADDER_STRENGTHS.map(|s| (l, s)))
        .collect();
    let mut pairs = Vec::new();
    for (li, _) in levels.iter().enumerate() {
        for si in 1..k {
            pairs.push((li * k + si - 1, li * k + si));
        }
        if li > 0 {
            for si in 0..k {
                pairs.push(((li - 1) * k + si, li * k + si));
            }
        }
    }
    (seats, pairs)
}

/// The index of the anchor seat in `ladder_plan`'s list.
pub fn anchor_seat(seats: &[LadderSeat]) -> usize {
    seats
        .iter()
        .position(|&s| s == (MAX_RANKED_LEVEL, 100))
        .expect("every ladder measures level 8 at full strength")
}

/// Play `games` automatic games for every pairing of `ladder_plan`, each run
/// seeded by its index so a ladder replays exactly. `progress` hears of each
/// pairing as it finishes.
pub fn run_ladder(
    pack: &LanguagePack,
    games: usize,
    mut progress: impl FnMut(&LadderMatch, &[LadderSeat]),
) -> Result<(Vec<LadderSeat>, Vec<LadderMatch>), GameError> {
    let (seats, pairs) = ladder_plan(pack.leave_net.is_some());
    let mut matches = Vec::new();
    for (i, &(a, b)) in pairs.iter().enumerate() {
        let seat = |n: usize, name: &str| PlayerConfig::computer(name, seats[n].0, seats[n].1);
        let players = [seat(a, "A"), seat(b, "B")];
        let result = run_benchmark(
            pack,
            &players,
            games,
            0x1ADD_E400 + i as u64,
            &BenchmarkControl::default(),
        )?;
        let m = LadderMatch {
            a,
            b,
            a_wins: result.player_stats[0].wins,
            b_wins: result.player_stats[1].wins,
            ties: result.player_stats[0].ties,
        };
        progress(&m, &seats);
        matches.push(m);
    }
    Ok((seats, matches))
}

/// Elo for `n` seats from their pairings: a Bradley-Terry fit (Hunter's MM
/// iteration), ties as half a win to each side and half a win and a loss
/// added to every pairing so a clean sweep does not drive a rating to
/// infinity. Shifted so `anchor` is `ANCHOR_RATING`.
pub fn fit_ratings(n: usize, matches: &[LadderMatch], anchor: usize) -> Vec<f64> {
    let mut wins = vec![0.0f64; n];
    let mut games = vec![vec![0.0f64; n]; n];
    for m in matches {
        let total = (m.a_wins + m.b_wins + m.ties) as f64 + 1.0;
        wins[m.a] += m.a_wins as f64 + 0.5 * m.ties as f64 + 0.5;
        wins[m.b] += m.b_wins as f64 + 0.5 * m.ties as f64 + 0.5;
        games[m.a][m.b] += total;
        games[m.b][m.a] += total;
    }
    let mut strength = vec![1.0f64; n];
    for _ in 0..2000 {
        let next: Vec<f64> = (0..n)
            .map(|i| {
                let denom: f64 = (0..n)
                    .filter(|&j| games[i][j] > 0.0)
                    .map(|j| games[i][j] / (strength[i] + strength[j]))
                    .sum();
                if denom > 0.0 {
                    wins[i] / denom
                } else {
                    strength[i]
                }
            })
            .collect();
        // Only ratios mean anything: keep the scale from drifting.
        let mean_log = next.iter().map(|g| g.ln()).sum::<f64>() / n as f64;
        strength = next.iter().map(|g| (g.ln() - mean_log).exp()).collect();
    }
    let elo: Vec<f64> = strength.iter().map(|g| 400.0 * g.log10()).collect();
    let shift = ANCHOR_RATING - elo[anchor];
    elo.iter().map(|r| r + shift).collect()
}

/// The Rust source of a `Ladder` entry for `code`, from `fit_ratings`' output
/// over `ladder_plan`'s seats -- what `bot-ladder` prints for pasting into
/// `LADDERS`.
pub fn ladder_source(code: &str, seats: &[LadderSeat], ratings: &[f64]) -> String {
    let row = |level: u8| -> Option<String> {
        let values: Vec<String> = seats
            .iter()
            .zip(ratings)
            .filter(|(s, _)| s.0 == level)
            .map(|(_, r)| format!("{r:.0}.0"))
            .collect();
        (!values.is_empty()).then(|| format!("[{}]", values.join(", ")))
    };
    let optional = |r: Option<String>| r.map_or("None".to_string(), |r| format!("Some({r})"));
    format!(
        "    Ladder {{\n        code: \"{code}\",\n        ranked: {},\n        smart: {},\n        sim: {},\n    }},",
        row(MAX_RANKED_LEVEL).expect("level 8 is always measured"),
        optional(row(SMART_LEVEL)),
        optional(row(SIM_LEVEL)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fit_orders_seats_by_their_results_and_keeps_the_anchor() {
        // 0 beats 1 three games in four, 1 beats 2 likewise.
        let m = |a, b, a_wins, b_wins| LadderMatch {
            a,
            b,
            a_wins,
            b_wins,
            ties: 0,
        };
        let r = fit_ratings(3, &[m(0, 1, 75, 25), m(1, 2, 75, 25)], 1);
        assert_eq!(r[1], ANCHOR_RATING);
        assert!(r[0] > r[1] && r[1] > r[2]);
        // 3:1 odds are about 190 Elo; the added half games pull it in a little.
        assert!((r[0] - r[1] - 190.0).abs() < 15.0, "{r:?}");
    }

    #[test]
    fn a_plan_measures_every_engine_and_links_them() {
        let (seats, pairs) = ladder_plan(true);
        assert_eq!(seats.len(), 18);
        assert_eq!(seats[anchor_seat(&seats)], (8, 100));
        // Every seat is in some pairing.
        for i in 0..seats.len() {
            assert!(pairs.iter().any(|&(a, b)| a == i || b == i), "seat {i}");
        }
        let (seats, _) = ladder_plan(false);
        assert!(seats.iter().all(|&(level, _)| level == 8));
    }

    #[test]
    fn every_measured_ladder_is_anchored_and_climbs_with_strength() {
        for ladder in LADDERS {
            assert_eq!(ladder.ranked[5], ANCHOR_RATING, "{}", ladder.code);
            for row in [Some(ladder.ranked), ladder.smart, ladder.sim]
                .into_iter()
                .flatten()
            {
                assert!(
                    row.windows(2).all(|w| w[0] < w[1]),
                    "{}: {row:?}",
                    ladder.code
                );
            }
            // Read through bot_rating: level 8 at full strength is the anchor,
            // a weaker ranked level sits below it, and an unknown code has none.
            let rating = |level, strength| bot_rating(ladder.code, true, level, strength).unwrap();
            assert_eq!(rating(8, 100), ANCHOR_RATING);
            assert!(rating(3, 100) < rating(8, 100));
            assert!(rating(10, 50) < rating(10, 100));
        }
        assert_eq!(bot_rating("xx", true, 8, 100), None);
    }

    #[test]
    fn a_full_strength_ranked_level_rates_as_the_matching_strength() {
        assert_eq!(equivalent_strength(8), 100.0);
        assert_eq!(equivalent_strength(1), 1.0);
        let row = [0.0, 10.0, 25.0, 50.0, 75.0, 100.0];
        assert_eq!(interpolate(&row, 100.0), 100.0);
        assert_eq!(interpolate(&row, 1.0), 0.0);
        assert!((interpolate(&row, 5.5) - 5.0).abs() < 1e-9);
    }
}
