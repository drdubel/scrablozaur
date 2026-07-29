# smart_player

A learned rack-leave evaluator for `StrategicPlayer`, aimed at win rate
rather than raw per-move score.

## Why

`StrategicPlayer.evaluate_word` (`src/strategy.py`) picks among the top 50
scoring candidates using `points` plus `sum(letter_points(ch) for ch in
get_letters_left())`. `get_letters_left()` returns the tiles not yet on the
board or in the player's own hand -- i.e. still in the bag or the
opponent's rack -- which is identical for every candidate word being
compared in a single decision. That heuristic term is therefore a constant
offset per turn: it never actually changes which word gets picked, only raw
`points` does. `StrategicPlayer` is, in effect, greedy. (The code now makes
this explicit: `get_best_word` precomputes the term once into
`self._leave_points` and `evaluate_word` returns `points +
self._leave_points` -- argmax-equivalent, just without recomputing a
constant ~50 times per turn.)

`SmartPlayer` (`player.py`) replaces that heuristic with a small learned
model of the *leave* -- the rack a candidate move would actually leave
behind -- trained on self-play outcomes. This mirrors how the strongest
classical Scrabble engines (Maven, Quackle) get their edge: not from
generating moves differently (the existing Rust DAWG engine already finds
every legal play and its exact score, fast), but from evaluating the
already-enumerated candidates by more than just their immediate score.

## Pipeline

```
python smart_player/generate_data.py 200000  # self-play -> _leave_dataset.npz (~20-40 min)
python smart_player/train.py                 # -> models/leave_value.pt
python smart_player/evaluate.py 2000         # SmartPlayer vs StrategicPlayer win rate
```

- **generate_data.py** plays `StrategicPlayer` vs `StrategicPlayer` games to
  completion (via `simulate.play_game`, which applies the standard
  end-of-game rack-value adjustment), recording every leave a player held
  during the game. This is the same idea real leave tables are built from --
  simulate lots of games, correlate leave with outcome -- automated
  end-to-end instead of hand-computed.
- **train.py** fits `model.LeaveValueNet`, an MLP (no convolution -- a leave
  has no spatial structure) on `(leave, unseen_tiles) -> target`, and saves
  a checkpoint with the tile alphabet and hidden layer sizes embedded next
  to the weights (`model.get_model()` reconstructs whatever architecture a
  checkpoint actually was trained with, falling back to the original 64/32
  for older checkpoints that predate storing this). Uses CUDA/MPS
  automatically when available (`--device` to override), and encodes +
  keeps the whole dataset resident on-device as plain tensors rather than
  going through `Dataset`/`DataLoader` -- at tens-of-millions-of-samples
  scale, per-sample Python `__getitem__` calls (and the small batch size
  that went with them) were the actual bottleneck, not compute. Concretely:
  re-encoding the existing 4.8M-sample dataset the old way took ~53s/epoch
  on CPU; vectorized + on an Apple GPU it's ~1-2s/epoch, and a report from a
  47.9M-sample dataset on an RTX 4090 (after the same fix) went from
  ~1190s/epoch to ~2s/epoch. `--batch` default bumped from 256 to 8192
  accordingly -- small batches on a dataset this size mostly measure Python
  loop overhead, not anything the model needs. `--hidden1`/`--hidden2`
  (default 128/64, up from the original 64/32) size the two hidden layers --
  now that training is ~free, there's no reason to stay at a size picked
  back when every epoch was expensive.
- **evaluate.py** plays `SmartPlayer` against the existing baselines and
  reports win rate, mirroring `src/main.py`'s `benchmark()`.

## Credit assignment: bounded lookahead, not the whole game

The first version labelled every recorded leave with that player's *final*
score margin -- simple, but a single leave is only one of ~15-20 decisions
in a game, so most of that label's variance comes from everything that
happens *after* the leave, not the leave itself. A 50k-game dataset built
that way only explained ~5.6% of final-margin variance and produced a
win rate statistically indistinguishable from 50%; 4x the data barely
moved that (~6.2% explained variance), because more samples just average
down noise that was never going away -- the target itself was the problem.

`generate_data.py` now labels each leave with an **n-step return**
(`--lookahead`, default 4): the change in (this player's score - opponent's
score) between the moment the leave was held and `lookahead` of that
player's *own* turns later (or the actual final differential, for leaves
within `lookahead` turns of the game's end). This is the standard
Monte-Carlo/TD middle ground -- truncate the return horizon to cut
irrelevant long-range variance, without needing full TD bootstrapping off
the value network's own (still-training) predictions. `--lookahead 0`
recovers the old unbounded/whole-game behavior.

Effect on a 200k-game / ~4.8M-sample dataset: baseline target variance
dropped ~4.4x (10013 -> 2253) and, more importantly, the resulting
`SmartPlayer` got meaningfully stronger, not just lower-variance:

| Opponent | Games | Whole-game margin | Bounded lookahead (k=4) |
|---|---|---|---|
| `StrategicPlayer` | 4000 | 55.7% (387.3 / 374.1 avg) | **61.9%** (410.3 / 381.5 avg) |
| `SimplePlayer` | 3000 | 54.9% (387.2 / 375.4 avg) | **63.4%** (410.9 / 377.1 avg) |

(win rate / avg score, SmartPlayer vs. opponent; std error ~0.8pp at these
sample sizes, so both jumps are well outside noise). **The checkpoint
committed in this repo uses `--lookahead 4`** (200k games).

A larger run -- 2M games, `--lookahead 8`, trained post-GPU-fix on an
RTX 4090 -- was evaluated at n=10000 (much tighter, ~0.5pp std error) and
scored:

| Opponent | Games | `--lookahead 4` (200k games, committed) | `--lookahead 8` (2M games) |
|---|---|---|---|
| `StrategicPlayer` | 10000 | 61.9% | **63.5%** (407.4 / 374.9 avg) |
| `SimplePlayer` | 10000 | 63.4% | 61.9% (409.3 / 377.5 avg) |

Read this carefully: it's *better* vs. `StrategicPlayer` but *worse* vs.
`SimplePlayer` than the committed checkpoint, and the two roughly average
out (62.65% vs. 62.7%) -- so despite 10x the games and 2x the lookahead,
this is closer to a lateral move than a clean win, once measured with a
large enough sample to trust both numbers. It also bundles two changes at
once (lookahead *and* dataset size), so which one is actually responsible
for the vs-strategic gain -- or whether it's just which-opponent-is-
harder-to-generalize-against noise -- is unresolved. That checkpoint itself
isn't in this repo (it lives on the machine it was trained on); the
committed one is still the `--lookahead 4` / 200k-game checkpoint above.
Worth an ablation (same game count, sweep only `--lookahead`) before
treating 8 as a better default than 4.

## Policy iteration (`iterate.py`)

The natural next step after a static regression: close the self-play loop
so each round's training data comes from the *current best* `SmartPlayer`
instead of the fixed `StrategicPlayer` baseline, the same generate ->
improve -> generate-with-the-improved-version loop AlphaZero/TD-Gammon-style
self-play RL uses. `generate_data.py --player smart --model-path PATH`
does the self-play half of this (`--player strategic`, the default,
unchanged); `iterate.py` wraps the full loop with champion/challenger
gating so a bad round can't silently regress the deployed checkpoint:

```
python smart_player/iterate.py --rounds 5 --games 60000
```

Each round: generate `--games` self-play games with the current champion,
train a candidate on them, play candidate vs. champion over `--eval-games`
games, and promote (overwrite the champion checkpoint) only if the
candidate's win rate clears `--promote-threshold` (default 0.52). Each
round's scratch dataset is deleted after training; only metrics and, if
promoted, the checkpoint persist.

**Result of a 5-round / 60k-games-per-round run: 0/5 promoted**, every
candidate landing in a 48.8%-50.8% band against the champion -- a
statistical tie, not an improvement or a regression. The likely reason:
each round trains a *fresh* `LeaveValueNet` from random init rather than
fine-tuning the champion's weights, so with nothing carried over between
rounds, a round is really "retrain an equally-powerful model on
similarly-distributed data" -- expected to land near a tie rather than
compound, especially since `SmartPlayer`'s choices only diverge from
`StrategicPlayer`'s on candidates that were already close to tied in raw
score, so self-play under either doesn't visit dramatically different
positions. Untried fixes that would more plausibly produce real compounding
gains: warm-starting each round from the champion's weights (so training
actually refines rather than re-rolls), or a bigger lever entirely (richer
features, `--lookahead` sweep, a bigger dataset per round).

## Board-aware features

Every ablation above (data volume, `--lookahead`, model capacity, learning
rate) plateaued around the same ~62-63% win rate -- strong evidence the
ceiling wasn't optimization or capacity, but the inputs: `SmartPlayer` only
ever saw the rack leave plus one scalar (unseen-tile count). It had zero
visibility into the board, even though a leave's real value depends on
board context (an S is worth more when bingo lanes are open).

`board_features.py` adds 5 scalars, computed from the board at the moment a
leave is held: `tw_open`/`dw_open`/`tl_open`/`dl_open` (fraction of that
premium-square type still unclaimed -- 8 TW / 17 DW / 12 TL / 24 DL exist
total) and `board_fill` (fraction of all 225 cells occupied). The bonus
layout isn't exposed to Python by the engine (`BONUS_TABLE` in
`src/lib.rs`), so it's ported from the already-validated copy at
`web/static/js/board.js:4-32` rather than re-transcribed from Rust.

Input width: 33 letter counts + `unseen_tiles` + these 5 = **39 dims**
(was 34). Old datasets/checkpoints are incompatible with the new encoding
-- `model.get_model()` now stores and validates `input_dim` in the
checkpoint and fails loudly on a mismatch rather than silently loading a
wrong-shaped model.

`SmartPlayer.get_best_word` (`player.py`) computes board features once per
turn, from the board as it exists *before* any candidate is placed -- that
match matters: `generate_data.py`'s training-data capture uses the exact
same pre-move snapshot (via `_RecordingMixin.get_best_word`, not
`draw_letters`, which runs after `place_word`). Getting this wrong the
first time around trained on a systematically more-filled board than
inference ever sees; caught and fixed within this session before any real
training ran on the mismatched version.

**Result of this feature is not yet validated against the established
61.9%/63.4% (`--lookahead 4`, 200k games) baseline** -- see the module
docstrings for the ablation plan (regenerate at matching game
count/lookahead, train, evaluate at n=10000). A quick unrelated data point:
a 200k-game/`--lookahead 8` run trained on the *already-committed*
board-aware checkpoint scored 61.3% vs `StrategicPlayer` -- in the same
range as the pre-board-features baseline, not a clear win, though not a
controlled comparison (different lookahead, and confounded with whatever
else changed in that run).

## Smart exchanging

`StrategicPlayer.play_word` (`src/strategy.py`) decides whether to exchange
via a fixed rule (`points < 6`) and picks *which* tiles via a hand-tuned
vowel/consonant-balance heuristic (`get_letters_to_exchange`) -- neither
uses the leave-value model at all. `SmartPlayer` now overrides `play_word`
to make both decisions with the same learned value function used for move
ranking: the tiles you *keep* when exchanging are a "leave" in exactly the
same sense as a leave-after-a-move (what you draw to replace the discarded
tiles is random regardless of what you choose to discard), so
`_best_exchange` brute-forces all `2**7 - 1` possible keep-subsets of the
rack (cheap, no need to approximate) and compares the best one's predicted
value against the best available word's `points + leave_value` -- playing
whichever is actually higher, not a fixed threshold.

**This surfaced a real, pre-existing engine bug, since fixed at the
source.** `Board.get_best_words` (`src/lib.rs`) used to flip its internal
first-move flag the moment it was *called* while the board was empty, not
when a word was actually placed. Turned out to be broader than just the
exchange case: *any* search on an empty board retired first-move
eligibility, including one that legitimately found nothing (e.g. an
all-consonant, no-blank opening rack -- rare but not that rare) -- so even
two players just playing normally could occasionally get stuck at 0 points
forever if whoever went first happened to draw an unplayable rack. Fixed
by moving the flag flip from `get_best_words` into `place_word`
(`src/lib.rs`, ~10 lines) -- first-move eligibility now only retires on an
actual placement, exactly as it should. Verified directly against the raw
Rust method before and after (a second search on a still-empty board used
to always return nothing regardless of rack; now works correctly), and via
a 500-game benchmark through `src/main.py` (0 degenerate 0-0 games,
vs. a nonzero rate before). `SmartPlayer.play_word` still never offers to
exchange on an empty board as a harmless extra safety net, and
`StrategicPlayer`/`SimplePlayer` (`src/strategy.py`) gained the same
`_board_is_empty` guard, but the actual fix means none of that is load-
bearing anymore.

This also means `src/main.py`'s benchmark (`graj()`) had two more,
separate problems worth knowing about even though they're not
`smart_player/`-scoped: it ended the game the moment *either* player
failed to play *once* (should be two consecutive no-plays, matching
`web/game.py`/`simulate.py`'s `CONSECUTIVE_NO_PLAY_LIMIT`), and it never
applied the standard end-of-game rack-value score adjustment at all. Both
fixed alongside the engine bug above, since the symptom that surfaced all
of this was `graj()` reporting near-0-0 games.

One more related fix: `StrategicPlayer`/`SmartPlayer` used to refuse to
exchange on consecutive turns (`not self.last_exchanged` gating the
exchange decision) -- not a real Scrabble rule (exchanging has no
restriction on repeated use, as long as `can_exchange()` holds), and it
meant a player stuck with bad tiles for several turns in a row would
alternate between a productive exchange and a wasted do-nothing pass
instead of just exchanging again. Removed; `last_exchanged` is still
tracked (used for transcript labeling) but no longer blocks anything.
`SimplePlayer` never had this restriction to begin with, so all three
player classes are now consistent.

**The `_board_is_empty` guard (both files) was then removed entirely.**
It had already gone from load-bearing to "harmless" once the engine fix
landed -- but harmless turned out to be wrong. A larger `src/main.py`
benchmark (3000 games) surfaced a rarer real case: a player draws a
genuinely unplayable opening rack (e.g. all consonants, no blank -- rare
but not that rare), can't play, and -- because the guard still refused to
exchange on an empty board -- couldn't recover either, forced into a
useless pass instead of the fresh rack that would have fixed it. Verified
directly that exchanging on a still-empty board now works correctly
(post engine-fix) before removing the guard, and confirmed removing it
increased *and then further reduced* negative-score games once the next
fix below landed (see that section for the numbers).

**`play_word`'s return value now distinguishes exchanging from genuinely
having no action available**, across `SimplePlayer`/`StrategicPlayer`
(`src/strategy.py`) and `SmartPlayer` (`player.py`): returns the played
word, `None` if tiles were exchanged instead, or `""` only when no legal
word exists *and* exchanging isn't possible either. Previously both
"exchanged" and "genuinely stuck" returned `""`, indistinguishable without
separately checking `.last_exchanged`. This mattered for a real bug: every
game-ending loop (`src/main.py`'s `graj()`, `simulate.py`'s `play_game()`,
`web/game.py`'s `consecutive_no_play`) counted *both* toward the same
"no play" streak that ends the game -- so two players independently
choosing to exchange (a completely normal strategic choice, not being
stuck) could end the game after just one round, producing exactly the
near-0 and negative final scores this whole thread started from. Fixed to
only count genuine no-action turns; exchanging is unlimited and never
contributes to ending the game on its own. Result, 1000 fresh games
through `graj()`: 0 negative-score games (down from ~0.3%, itself down
from the pre-engine-fix rate), min score 240/241 (up from -19/-25), no
change to average scores (~392/421) or game length (min 19 moves, vs. 2
before any of these fixes).

**Added tile exchange as a human action in the web app.** The backend
endpoint (`POST /api/board/exchange`, `web/routers/board.py`) already existed
but had no frontend at all -- exploring the codebase to add one surfaced
a second, real, previously-untested bug: it called `Board.can_exchange`
(an *instance* method reading the Rust engine's own internal bag) as if
it were a staticmethod taking a tile count, which the web app doesn't
even use for bag tracking (it has its own `TileBag`/`session.tile_bag`).
Always raised `TypeError` -- 500 Internal Server Error on every call,
caught by actually driving the new UI in a browser (Playwright) rather
than just adding it and moving on. Fixed by inlining the same >=7-tiles
rule directly against `session.tile_bag.remaining()`. Also brought
`consecutive_no_play` in `web/game.py`/`board.py`'s human exchange path
in line with the same "unlimited, doesn't count" rule as the bot side.
Frontend: rack tiles are now clickable (toggle a `.selected` highlight),
a "Wymień litery" button submits whichever are selected via a new
`ApiClient.exchangeTiles()`, matching the existing skip/pass button
patterns in `web/static/js/game.js`.

**Cost**: `_best_exchange`'s 127-candidate search runs on most turns now
(no pre-filter), roughly doubling `--player smart` self-play throughput
(measured: ~10 games/s -> ~5 games/s, single-process). Only affects bulk
`--player smart` data generation (`generate_data.py`, `iterate.py`) -- a
real game is ~15-20 turns, so the extra cost is imperceptible for actual
gameplay (`evaluate.py`, the web app). A safe pre-filter (skip the full
search when the best word's raw score is high enough that no realistic
exchange could beat it) would recover most of this if `--player smart`
generation throughput becomes a bottleneck; not added since it wasn't
proven necessary yet.

## Files

| File | Purpose |
|---|---|
| `model.py` | `LeaveValueNet`, rack + board-feature encoding, multi-checkpoint-aware loading |
| `board_features.py` | Bonus-square layout + `encode_board()` (board-state summary scalars) |
| `simulate.py` | Shared self-play game loop + end-of-game scoring |
| `player.py` | `SmartPlayer` (StrategicPlayer + learned leave evaluator + learned exchange decision) |
| `generate_data.py` | Self-play data generation CLI (StrategicPlayer or SmartPlayer) |
| `train.py` | Training CLI (also importable as `train()`) |
| `evaluate.py` | Win-rate benchmark CLI: vs. baselines, or candidate vs. champion |
| `iterate.py` | Policy-iteration orchestrator (generate -> train -> gate -> promote) |
| `models/leave_value.pt` | Trained checkpoint (committed, like `board_reader`'s CNN weights) |
