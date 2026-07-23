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
  during the game labelled with that player's *final score margin*. This is
  the same idea real leave tables are built from -- simulate lots of games,
  correlate leave with outcome -- automated instead of hand-computed.
- **train.py** fits `model.LeaveValueNet`, a tiny MLP (~3.5k params, no
  convolution -- a leave has no spatial structure) on `(leave, unseen_tiles)
  -> final_margin`, and saves a checkpoint with the tile alphabet embedded
  next to the weights.
- **evaluate.py** plays `SmartPlayer` against the existing baselines and
  reports win rate, mirroring `src/main.py`'s `benchmark()`.

## Current status

The committed checkpoint was trained on a 200k-game / ~4.8M-sample
self-play dataset (~40 min to generate + train on this machine). A first
attempt at 50k games / 1.2M samples only explained ~5.6% of the variance in
final score margin and produced a statistically insignificant win rate --
expected, since a single leave is one of ~15-20 decisions in a game, so its
effect on the *final* margin is a small signal buried in a lot of noise
from everything that happens afterward. 4x the data pushed explained
variance to ~6.2% (val MSE ~9394 vs. ~10013 baseline variance), which was
enough for the signal to show up in actual play:

| Opponent | Games | SmartPlayer record | Win rate | Avg score (Smart / opp) |
|---|---|---|---|---|
| `StrategicPlayer` | 4000 | 2220W 1763L 17T | **55.7%** | 387.3 / 374.1 |
| `SimplePlayer` | 3000 | 1638W 1344L 18T | **54.9%** | 387.2 / 375.4 |

Both are several standard errors above the 50% no-improvement baseline
(std error ~0.8pp at these sample sizes). More self-play data is still the
first lever to pull for further gains; see the module docstrings for other
tuning ideas (e.g. crediting a leave with a bounded-lookahead score delta
instead of the full final margin, which would cut a lot of the irrelevant
late-game variance out of the training target).

## Files

| File | Purpose |
|---|---|
| `model.py` | `LeaveValueNet`, rack encoding, checkpoint loading |
| `simulate.py` | Shared self-play game loop + end-of-game scoring |
| `player.py` | `SmartPlayer` (StrategicPlayer + learned leave evaluator) |
| `generate_data.py` | Self-play data generation CLI |
| `train.py` | Training CLI |
| `evaluate.py` | Win-rate benchmark CLI |
| `models/leave_value.pt` | Trained checkpoint (committed, like `board_reader`'s CNN weights) |
