use serde::{Deserialize, Serialize};

use crate::{shuffle, time_seed, xorshift};

/// The randomness a game consumes: shuffling the tile bag, and a weakened
/// bot's pick from its rank window.
///
/// Its own stream, held in the session rather than drawn from a global, so a
/// saved game resumes the same sequence it would have had -- and so a test can
/// replay the web app's exact deal by feeding Python the same algorithm. The
/// shuffle is the engine's own Fisher-Yates, which walks the list the same way
/// Python's `random.shuffle` does.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GameRng {
    state: u64,
}

impl GameRng {
    /// A stream started from `seed`; 0 picks one from the clock.
    pub fn new(seed: u64) -> Self {
        // 0 is xorshift64's fixed point: it would never move.
        let state = if seed == 0 { time_seed() } else { seed };
        GameRng { state }
    }

    pub fn shuffle(&mut self, tiles: &mut [char]) {
        shuffle(tiles, &mut self.state);
    }

    /// A seed for a computation that keeps its own generator, such as a
    /// simulation's rollouts: one step of this stream, never 0 (which the
    /// engine reads as "seed from the clock").
    pub fn next_seed(&mut self) -> u64 {
        xorshift(&mut self.state);
        self.state
    }

    /// Uniform in `lo..=hi`, like Python's `random.randint`. Always consumes
    /// one step, even when `lo == hi`.
    pub fn randint(&mut self, lo: usize, hi: usize) -> usize {
        debug_assert!(lo <= hi);
        xorshift(&mut self.state);
        lo + (self.state % (hi - lo + 1) as u64) as usize
    }
}
