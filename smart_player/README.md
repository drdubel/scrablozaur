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

## Re-baselined: every number below this line predates two engine fixes

**All win rates further down were measured on a bag that was not being
shuffled fairly, and against a baseline that is not doing what its name
suggests.** They are kept for continuity, but the current numbers are here.

`give_letters` used to re-seed from the wall clock on every call and index the
bag with `% bag.len()`. Over 60000 opening racks that came out at
chi-square/dof = **9.06** against the true tile distribution (a fair draw
gives ~1.0), systematically under-dealing the *rare, high-value* Polish
letters -- `ź` -8.5%, `ę` -8.5%, `ó` -8.1%, `ł` -7.2%, `ć` -6.7%, each worth
5-9 points. The bag is now shuffled once with Fisher-Yates and drawn from the
end: chi-square/dof = 0.93. Separately, a blank played onto the board used to
keep scoring at the face value of the letter it stood in for, for the rest of
the game.

Rebuilding the old commits and re-running `evaluate.py` at n=2000 separates
the two (the checkpoint is identical throughout -- nothing was retrained):

| Build | vs `StrategicPlayer` | Avg scores |
|---|---|---|
| before both fixes | 63.3% | 427.3 / 390.4 |
| + fair draw | 66.6% | 430.7 / 387.8 |
| + blank scoring fix | **65.7%** | 426.5 / 387.6 |

So the biased bag was *understating* `SmartPlayer`'s edge -- unsurprisingly,
since it suppressed exactly the tiles that reward leave management. Nothing
got stronger here; the measurement got fairer. Read the plateau discussion
below with that in mind: it was a plateau at ~62-63% *on a biased bag*.

Current ladder, via `arena.py`:

| A | B | Pairs | A's match score | Elo | Mean margin |
|---|---|---|---|---|---|
| `sim` | `strategic` | 250 | 71.10% +/- 2.05pp | +156 | +52.7 +/- 3.8 |
| `smart` | `strategic` | 2000 | 66.86% +/- 0.70pp | +122 | +44.0 +/- 1.4 |
| `strategic` | `simple` | 1000 | 50.00% +/- 0.00pp | +0 | **+0.1 +/- 0.1** |

What each piece contributes, measured against the same player without it:

| Change | Pairs | Match score | Mean margin | Elo |
|---|---|---|---|---|
| simulation (`sim` vs `smart`) | 150 | 53.33% +/- 2.74pp | +10.7 +/- 4.8 | +23 |
| 2M-game checkpoint at its own weight | 1200 | 51.67% +/- 0.94pp | +4.8 +/- 1.7 | +12 |
| endgame search (`smart` vs `smart!noeg`) | 1200 | 51.08% +/- 0.24pp | +2.7 +/- 0.2 | +8 |

The checkpoint upgrade shows up cleanly in the static player: `smart` vs
`strategic` moved from 66.07% (+41.2 pts) to 66.86% (+44.0 pts). It does *not*
show up in `sim` vs `strategic` — 71.10% +/- 2.05 against 72.00% +/- 1.95
before, with the margin up from +51.2 to +52.7. Those are the same number at
250 pairs, so simulation neither gained nor lost from the better evaluator as
far as this sample can tell; resolving a 5-point effect there needs roughly
1200 pairs, which at ~9 minutes per 250 is a much longer run than it is worth
right now.

The `strategic` vs `simple` row in the first table is the one to read carefully.
The two differ only in when they exchange, and across 2000 games that difference
is worth **a tenth of a point per game** — they are the same player to within
measurement error.
The README already said `StrategicPlayer` is "in effect, greedy"; this puts a
number on it. So the headline "vs `StrategicPlayer`" has always meant "vs a
greedy player", and `smart` beating `strategic` and `simple` by the same margin
is one fact stated twice, not two results.

That last row is the one to look at. `StrategicPlayer` and `SimplePlayer`
differ only in when they exchange (`points < 6` vs. only-when-stuck), and
across 2000 games that difference is worth **a tenth of a point per game** --
they are the same player to within measurement error. The README already said
`StrategicPlayer` is "in effect, greedy"; this puts a number on it. The
pairing cancels 24x on that row precisely *because* the two play nearly
identical games.

Which means the headline "vs `StrategicPlayer`" number has always been "vs a
greedy player", and `smart` beating `strategic` and `simple` by the same
65.7% is the same fact stated twice, not two independent results.

## Pipeline

```
python smart_player/generate_data.py 200000  # self-play -> _leave_dataset.npz (~20-40 min)
python smart_player/train.py                 # -> models/leave_value.pt
python smart_player/arena.py --a smart --b strategic --pairs 1000   # paired benchmark
python smart_player/evaluate.py 2000         # older unpaired benchmark
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
  reports win rate, mirroring `src/main.py`'s `benchmark()`. Unseeded and
  unpaired; kept because every historical number in this file came from it.
- **arena.py** is the replacement: it plays each seeded bag *twice* with the
  seats swapped, and reports a per-pair match score with a paired standard
  error (plus the Elo that implies, and the ties-dropped win rate so the two
  stay comparable). Seeding also makes a run reproducible, which `evaluate.py`
  never was. Two honest caveats: the pairing only cuts the margin std error
  ~1.1x -- the seat advantage cancels exactly, but the two games diverge as
  soon as the players choose different moves, and that is most of the variance
  -- and a run of `strategic` vs `strategic` returns exactly 50.00% with zero
  variance, which is a wiring check rather than a result.

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

## Simulation (`sim_player.py`)

Every plateau above shares one cause: `SmartPlayer` ranks a move by
`score + leave_value`, and no function of (my score, my leave) can express what
the move *hands the opponent*. A play that scores four more points while opening
the triple-word lane is a bad play, and static evaluation cannot say so.

`SimPlayer` asks directly. For each leading candidate it plays the move, deals
the opponent a rack from the tiles neither on the board nor in its own rack,
lets both sides reply with the static player, and scores the resulting position.
`Board.simulate` (`src/lib.rs`) does the whole rollout natively; two things make
it affordable:

- **Common random numbers.** Every candidate in one iteration is rolled out
  against the *same* shuffled tile sequence, so the comparison isolates the
  candidate instead of the luck.
- **Sequential pruning.** Candidates whose interval falls clear of the leader's
  stop being sampled. In practice 20 candidates collapse to 3 within a couple of
  hundred iterations, so the nominal cost is rarely paid.

The leave net runs *in Rust* (`export_weights.py` -> `models/leave_value.bin`,
`LeaveNet` in `src/lib.rs`). At hundreds of thousands of evaluations per move,
calling back into PyTorch for a 13k-parameter MLP would cost far more in FFI and
GIL traffic than its ~10k multiply-adds. The `.pt` stays the source of truth and
`sim_player.get_net` re-exports automatically when it is newer than the `.bin`.
Parity against PyTorch is 2e-6 on real feature vectors.

**Ply parity matters more than depth.** Measured against `smart`:

| Config | Pairs | Match score | Mean margin | Relative cost |
|---|---|---|---|---|
| `plies=1` (their reply — balanced 1v1) | 150 | 53.33% +/- 2.74pp | **+10.7 +/- 4.8** | 1x |
| `plies=3` (reply, ours, reply — balanced 2v2) | 120 | 51.67% +/- 2.77pp | +8.8 +/- 5.4 | ~2.5x |
| `plies=2` (reply, ours — **unbalanced 2v1**) | 20 | 45.00% +/- 6.18pp | -1.6 +/- 12.2 | ~1.7x |

An even ply count gives us one more scoring turn than the opponent ever gets,
which rewards setting up our own follow-up over noticing what the candidate
concedes. `plies=1` is the default: same as the deeper balanced window within
error, at a fraction of the cost.

**Honest accounting.** Simulation is worth about **+10 points/game, ~+25 Elo**
over `SmartPlayer` — real (the margin is significant at t=2.2, and the two
measurements agree: +10.7 head-to-head, and +7.6 inferred from 70.12% vs 65.70%
against the same baseline) but far short of the +60-90 Elo simulation is worth
in engines like Quackle. The reason is visible in the design: the rollout policy
*is* the static player, so the rollouts inherit its judgement. A sim can only
distinguish candidates as well as the evaluator scoring its leaves, and this
evaluator was trained on greedy self-play against an n-step return. There is
also a scale mismatch — the net predicts a 4-turn score-differential return, and
that gets added to a realised 1-ply differential, so the leave term double-counts
future scoring.

Which re-orders what to do next. Simulation was supposed to be the big lever;
measurement says its ceiling is set by the leaf evaluator. The two things that
should now come first are the **exact endgame solver** (with the bag empty the
position is perfect information, and endgames decide exactly the close games
where win rate is won) and **distilling sim output back into the static net**,
which lifts the rollout policy and the ranking together.

Cost: ~0.9 CPU-seconds per decision at the defaults, ~110 ms wall on 8 threads.
Fine for real games; a benchmark needs `set_num_threads(1)` in each worker, which
`arena.py` does — without it, one worker process per core each spinning up the
engine's 8-thread pool turns a one-minute run into ten minutes of thrashing.

## Checkpoints, and the weight that goes with them

`models/leave_value.pt` is the champion. The rest are the named sources behind
the results below, kept because they are the controls any future comparison
needs.

| File | Games | Lookahead | Best weight | Margin vs `leave_v1` |
|---|---|---|---|---|
| **`leave_value.pt`** (= `leave_v2.pt`) | **2M** | **4** | **1.0** | **+5.4 +/- 1.6** |
| `leave_v1.pt` (previous champion) | ? | ~8, see below | 0.8 | — |
| `leave_k2.pt` | 200k | 2 | ~1.0 | −0.3 +/- 1.5 |
| `leave_k4.pt` | 200k | 4 | ~1.0 | −1.1 +/- 1.6 |
| `leave_k6.pt` | 200k | 6 | ~0.83 | −1.4 +/- 1.5 |
| `leave_k8.pt` | 200k | 8 | ~0.82 | +1.1 +/- 1.5 |

Measured at 1500 seeded pairs each, endgame search off on both sides — it is
checkpoint-independent, so it only added cost (9 minutes a run instead of one).

**The lookahead horizon does not matter.** All four 200k checkpoints land within
±2 points of the old champion. That closes the question this file previously
left open — and it was only answerable with the weight swept, because the
horizon mechanically sets the model's output *scale*: prediction mean runs
−3.83 at k2 to −13.43 at k8, std 8.53 to 12.02. Comparing at one fixed weight
measures scale, not horizon.

**Data volume is not saturated after all.** An earlier entry in this file, and
an earlier round of this work, concluded it was. Both were wrong for the same
reason: the comparison ran the new checkpoint at the incumbent's weight. With
each at its own optimum, the clean 10x ablation (`leave_v2` 2M vs `leave_k4`
200k, same lookahead) is **+3.0 +/- 1.5**, and in the deployable configuration
`leave_v2` beats `leave_v1` by **+4.8 +/- 1.7 points/game**. The two models'
predictions still correlate +0.977 — 10x the data barely changes the *function*,
it changes how far the function can be trusted.

**Which is the practical lesson: always sweep the leave weight per checkpoint.**

| weight | `leave_v2` (2M) | `leave_k4` (200k) |
|---|---|---|
| 0.80 | +0.7 +/- 1.5 | — |
| ~0.90 | +3.8 +/- 1.5 | −2.0 +/- 1.5 |
| **1.00** | **+5.4 +/- 1.6** | −1.1 +/- 1.6 |
| 1.10 | +4.9 +/- 1.6 | — |
| 1.25 | +1.2 +/- 1.7 | −3.1 +/- 1.6 |
| 1.50 | −13.6 +/- 1.8 | — |

Same architecture, same target, same horizon — only the data differs, and the
peaks land in different places. Scored at the incumbent's 0.8, `leave_v2` reads
+0.7 +/- 1.5: a tie, and a real five-point improvement nearly discarded. The
optimum is a better quality signal than validation MSE, which cannot compare two
models whose targets sit on different scales.

**`leave_v1` was probably not the lookahead-4 model this file used to claim.**
Its predictions correlate +0.966 with `leave_k8` and only +0.830 with
`leave_k4`; correlation is scale-invariant, so that is about ranking behaviour,
not magnitude. Its output scale also matches k8's (mean −13.66/std 12.37 against
k8's −13.43/12.02) rather than k4's (−6.94/10.94). Treat the older
"lookahead 4 vs 8" numbers here as unreliable. The current champion genuinely is
lookahead 4, on 2M games, generated after the engine fixes.

## How the weight got tuned (against `leave_v1`)

Kept because it is where the parameter came from, and because reading it next to
the section above shows how a per-model constant gets mistaken for a universal
one.

`points + leave_value` weighted the model as heavily as the points themselves:
over ~30k leaves from real self-play positions, `leave_v1`'s predictions have a
spread of **12.33** against a candidate-score spread of **12.8**. That looked
like a scale artefact — the net regresses a 4-turn score-differential return
(std 47.5), so it inherits that scale. So the weight became a parameter (`~w` in
an arena spec) and was swept at 2500 pairs — 5000 games — per point:

| w | 0.1 | 0.25 | 0.5 | 0.75 | **0.8** | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| pts vs w=1.0 | −25.0 | −14.2 | −2.1 | +2.0 | **+2.5** | −5.5 | −32.1 | −153.8 |

The term is clearly load-bearing — w=2.0 collapses to an 18% match score, w=0.1
to 38.7% — and 0.8 was worth a real but small **+2.5 ± 1.1 points/game** over
1.0. A 600-pair sweep that had shown +7.7 at w=0.75 was noise; at 2500 pairs
0.75 and 0.8 are a dead tie (−0.1 ± 0.7).

**The mistake was concluding 0.8 was a property of the *player*.** It is a
property of the *checkpoint*: `leave_v1` peaked at 0.8 because its predictions
were noisy enough that leaning harder on them cost points. The 2M-game
checkpoint peaks at 1.0. Adopting 0.8 as a default then briefly scored its
replacement at +0.7 ± 1.5 — a tie — and nearly threw away five points. The
weight is now retuned with every checkpoint, and `DEFAULT_LEAVE_WEIGHT`'s
comment says so.

## Endgame search

Once the bag is empty the game stops being a game of chance. Nobody draws again,
so the opponent holds exactly the tiles neither on the board nor in our own rack
— computable only because the board now remembers which squares hold blanks
(`Board.unseen_tile_counts`, checked against the real rack on 120 endgames and
correct every time). Simulation is at its *least* useful here: there is nothing
left to sample.

`Board.solve_endgame` runs negamax alpha-beta over both racks, passing included,
terminal values carrying the standard rack adjustment, and out-plays ordered
first — they collect the opponent's whole rack, so they are both the likely best
move and the best source of cutoffs. Make/unmake, not cloning.

It is bounded by **depth, not by a node budget**. A node cap bites part-way
through the tree and leaves whichever branches were searched first with a deeper
look than the rest, which makes the result depend on move ordering in a way that
is not a search property. A ply horizon cuts every branch alike. Verified
against unpruned brute force: with the branching limits lifted, alpha-beta
returns the identical differential on small positions.

**Worth +2.7 ± 0.2 points/game (+8 Elo)** over the same player without it —
solid at t = 13.5, but well short of the +25-40 estimated. Endgames are two to
four moves of a twenty-five move game, and playing the highest score is usually
already right; though not always, since the search picks a different move in
**59%** of endgames. ~490 ms median per endgame decision.

That comparison is also the clearest demonstration of why the arena is paired:
the two players play identically until the bag empties, so the pairs cancel and
the margin std error drops **9.91x**. A 2.7-point effect is not resolvable
otherwise.

## Measured: the model cannot be given the simulator's judgement

The distillation result above said the model was being asked to predict
something its inputs could not distinguish. That suggested an obvious fix --
give it per-candidate inputs -- so it was gated before building, and **the gate
failed twice.**

The setup: ~123,000 sim-labelled candidates across 10,338 decisions, the same
small MLP fitted twice, scored on the **within-decision centred** target, which
is the only part that can reorder anything. Bar set in advance at +10
percentage points of R².

| features added | R² | gain |
|---|---|---|
| baseline: leave + unseen + pre-move board | 0.271 | — |
| \+ whole-board aggregates after the move | 0.276 | +0.9 pp |
| \+ placement-local features | 0.313 | **+4.2 pp** |

The first attempt was a bad experiment, not a bad idea: whole-board aggregates
(reachable premiums, anchor totals, board fill) have a within-decision standard
deviation of ~0.03 against the leave's 0.176 — one word barely moves a
225-square average, so they were per-candidate in name only. The retest used
features measured relative to the squares the move covered (premium value newly
opened, biggest hook conceded, new anchors adjacent to the placement, how far
from the edge), which vary as much as the leave does (0.15-0.35). They tripled
the gain and still came in at less than half the bar.

**What that rules out.** Within-decision label noise accounts for only ~8% of
the variance, so roughly 60% of the ranking-relevant signal is real and remains
unexplained by leave, position and placement together. Hand-designed static
features do not reach it. Whatever decides between two candidates lives in the
specific interaction between the resulting board and the racks the opponent
might hold — which is what generating the opponent's replies *is*. You cannot
approximate the simulator with features; if you want its judgement, you have to
run it.

Which makes rollout throughput, not model capacity, the thing worth working on:
a simulation costs ~2175 move generations (263 ms against 0.12 ms), and every
one of them rebuilds both 15x15 cross-check tables from scratch even though a
rollout only disturbs the squares near its last placement.

## What to try next

Three rounds of measurement now point the same way. Simulation (+25 Elo), the
leave weight (+6), and endgame search (+8) were each smaller than estimated, and
each for the same underlying reason: **everything downstream is limited by how
good the leave evaluator is, and it can explain at most 6.75% of its target's
variance.** Tuning around it is finished.

Step 1 below is now **done** — regenerating on the fixed engine at 2M games is
what produced the current champion, worth +4.8 ± 1.7 points/game. Note that this
partly contradicts the paragraph above: "change the target, not the volume" was
too strong, and volume did buy something. What survives is that it bought less
than the label quality would, and that the 6.75% ceiling still binds.

What is left, in order:

1. ~~**Regenerate the training data.**~~ Done — see the checkpoint table at the
   top. The old data came off a bag that under-dealt ź/ę/ó/ł/ć by 7-8.5% and
   scored blanks at face value forever.
2. **Change the target.** Subtract a position-only baseline (fit on the board
   features and unseen count) so the net learns leave equity rather than game
   phase; the baseline is constant across a decision's candidates, so dropping
   it at inference changes no argmax. Measured on fresh data, position alone
   explains only **1.58%** of target variance, so expect this to fix the output
   *scale* (and with it the need to retune the weight every time) rather than to
   move Elo much. Report R², not raw MSE, which would have made the 6% ceiling
   obvious from the first run.
3. **Distil the simulation.** A sim equity has a standard error of ~1-1.5
   points; the n-step return has a standard deviation of 47.5. Per sample a sim
   label carries roughly 25-30x less noise, which is the direct fix for a 6.75%
   ceiling in a way more n-step samples are not. `distill.py` does not exist
   yet.

**Two process rules earned the hard way**, both of which nearly cost a real
result:

- **Sweep the leave weight for every new checkpoint.** The optimum is a property
  of the model, not the player.
- **Benchmark checkpoints with the endgame search off.** It is
  checkpoint-independent, so it contributes nothing to the comparison and turns
  a one-minute run into nine.

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
| `sim_player.py` | `SimPlayer` (picks its word by Monte-Carlo simulation) |
| `export_weights.py` | Export a `.pt` checkpoint to the flat binary the Rust engine loads |
| `board_features.py` | Bonus-square layout + `encode_board()` (board-state summary scalars) |
| `simulate.py` | Shared self-play game loop + end-of-game scoring |
| `player.py` | `SmartPlayer` (StrategicPlayer + learned leave evaluator + learned exchange decision) |
| `generate_data.py` | Self-play data generation CLI (StrategicPlayer or SmartPlayer) |
| `train.py` | Training CLI (also importable as `train()`) |
| `arena.py` | Paired-seed benchmark CLI: same bag twice, seats swapped |
| `evaluate.py` | Older unpaired win-rate benchmark: vs. baselines, or candidate vs. champion |
| `iterate.py` | Policy-iteration orchestrator (generate -> train -> gate -> promote) |
| `models/leave_value.pt` | Trained checkpoint (committed, like `board_reader`'s CNN weights) |
