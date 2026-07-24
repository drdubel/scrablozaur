# smart_player

A learned rack-leave evaluator for `StrategicPlayer`, aimed at win rate
rather than raw per-move score.

## Why

`StrategicPlayer.evaluate_word` (`src/strategy.py`) picks among the top 50
scoring candidates using `points + sum(letter_points(ch) for ch in
get_letters_left())`. `get_letters_left()` returns the tiles not yet on the
board or in the player's own hand -- i.e. still in the bag or the
opponent's rack -- which is identical for every candidate word being
compared in a single decision. That heuristic term is therefore a constant
offset per turn: it never actually changes which word gets picked, only raw
`points` does. `StrategicPlayer` is, in effect, greedy.

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
- **train.py** fits `model.LeaveValueNet`, a tiny MLP (~3.5k params, no
  convolution -- a leave has no spatial structure) on `(leave, unseen_tiles)
  -> target`, and saves a checkpoint with the tile alphabet embedded next to
  the weights. Uses CUDA/MPS automatically when available (`--device` to
  override), and encodes + keeps the whole dataset resident on-device as
  plain tensors rather than going through `Dataset`/`DataLoader` -- at
  tens-of-millions-of-samples scale, per-sample Python `__getitem__` calls
  (and the small batch size that went with them) were the actual
  bottleneck, not compute. Concretely: re-encoding the existing 4.8M-sample
  dataset the old way took ~53s/epoch on CPU; vectorized + on an Apple GPU
  it's ~1-2s/epoch. `--batch` default bumped from 256 to 8192 accordingly --
  small batches on a dataset this size mostly measure Python loop overhead,
  not anything the model needs.
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
sample sizes, so both jumps are well outside noise). The committed
checkpoint uses `--lookahead 4`.

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

## Files

| File | Purpose |
|---|---|
| `model.py` | `LeaveValueNet`, rack encoding, multi-checkpoint-aware loading |
| `simulate.py` | Shared self-play game loop + end-of-game scoring |
| `player.py` | `SmartPlayer` (StrategicPlayer + learned leave evaluator) |
| `generate_data.py` | Self-play data generation CLI (StrategicPlayer or SmartPlayer) |
| `train.py` | Training CLI (also importable as `train()`) |
| `evaluate.py` | Win-rate benchmark CLI: vs. baselines, or candidate vs. champion |
| `iterate.py` | Policy-iteration orchestrator (generate -> train -> gate -> promote) |
| `models/leave_value.pt` | Trained checkpoint (committed, like `board_reader`'s CNN weights) |
