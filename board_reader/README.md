# board_reader

Computer-vision pipeline that turns a photo of a physical Scrabble board into a
15x15 board state (letters + confidence + ranked alternatives).
It's a standalone, script-style package (no `__init__.py`, no package build) --
`web/scan.py` imports directly from `board_reader/src` by inserting it onto
`sys.path`, wraps it with a dictionary-driven correction pass, and exposes it
as the "scan board" feature of the web app.

## Pipeline

A photo goes through six stages, each owned by one `src/` module:

| Stage | Module | Input -> Output |
|---|---|---|
| 1. Find the board | `detect_board.py` | photo -> 4 corners, perspective-warped square crop |
| 2. Fix orientation | `rotate_board.py` | warped crop -> rotated so the red orientation marker sits top-left |
| 3. Locate the grid | `grid_detector.py` | rotated crop -> 16x16 grid-line intersection mesh |
| 4. Detect tiles | `tile_detector.py` (via `read_board.py`) | mesh -> per-square tile/empty verdict |
| 5. Normalise glyphs | `glyph_normalizer.py` | tile crop -> centred, binarised letter (+ digit) image |
| 6. Classify letters | `letter_classifier.py` (via `read_letters.py`) | glyph -> letter, confidence, ranked alternatives |

`read_board.py` orchestrates stages 1-4 (`read_board()`), and `read_letters.py`
orchestrates 5-6 across every detected tile (`classify_tiles()` /
`classify_board()`). Everything else supports one of these stages:

- `premium_layout.py` -- the fixed (not tunable) 15x15 premium-square layout, used to
  register the grid and to group cells by class for tile detection's per-class colour model.
- `hsv_config.py` -- shared load/save for every stage's tunable parameters, persisted to
  `src/hsv_config.json`. Each stage exposes a `_params(overrides=None)` + `PARAM_DEFAULTS`
  pair following the same convention, so a tuned preset is picked up automatically and an
  untuned stage just falls back to its hardcoded defaults.
- `cv_utils.py` -- generic OpenCV display helpers plus the shared interactive-trackbar
  framework (`run_tuner()`) that three of `tuner.py`'s four subcommands build on.
- `parallel_utils.py` -- one process-wide `ThreadPoolExecutor`, shared by every stage
  with independent per-tile work (`tile_detector.py`, `read_letters.py`) instead of each
  spinning up its own. Deliberately layered *on top of* OpenCV's own internal threading
  rather than replacing it -- see the module docstring for the measurement behind that.
- `tuner.py` -- interactive tuners for stages 1-2 and 4-6 (see [Tuning](#tuning)).

### Detection philosophy

Nothing assumes a fixed crop geometry, a fixed colour threshold, or equal
cell spacing -- lighting, board edition, and photo angle all vary too much
for constants tuned against one photo to generalise. Instead:

- The board's own colour (teal + black bezel) is segmented per photo, falling back to
  Canny edges if that finds nothing (`detect_board.py`).
- Every grid line is located individually via a coverage-profile comb fit + per-line/
  per-intersection refinement, not a fixed offset/pitch formula (`grid_detector.py`).
- Tile presence is judged by a **per-photo, self-calibrating** colour model: empty
  squares of each premium class set the "empty" baseline, and outliers that agree with
  each other across classes set the "tile" colour -- so a photo's own lighting and board
  edition never need to be known in advance (`tile_detector.py`).
- Letter identity fuses three independent, weighted sources -- a CNN, template matching,
  and the tile's own printed point-value digit -- rather than trusting any one of them
  (`letter_classifier.py`).

## Quick usage

```python
import sys
sys.path.insert(0, "board_reader/src")

from read_board import read_board
from read_letters import classify_board

rotated, mesh, _cells, verdicts, shift = read_board("photo.jpg", show=False)
if verdicts is not None:
    board = classify_board(rotated, mesh, verdicts, global_shift=shift)
    # board: 15x15 list of single-character strings ('-' empty, '?' unrecognised)
```

`classify_tiles()` (used by `classify_board()` internally) returns the richer
`{(row, col): (letter, confidence, ranked_alternatives)}` shape that
`web/scan.py` builds its dictionary-correction pass on top of.

## Accuracy

Measured against the hand-labelled ground truth in `test/out/` (89 photos across
three difficulty tiers -- 44 easy, 28 medium, 17 hard):

```
tests/eval_tile_detection.py   (default: easy + medium)
  precision 99.3%  recall 98.0%  cell accuracy 99.5%

tests/eval_letters.py          (default: easy + medium)
  precision 99.3%  recall 98.0%  letter accuracy 98.9%  cell accuracy 99.3%
```

Hard-difficulty photos are excluded by default (extreme angle/lighting that would
mostly just add noise to the aggregate score) -- pass `-d emh` to include them.
Both scripts also report board-detection failures separately (currently 0).

## Tuning

Every detection stage's parameters can be tuned interactively against
`test/in/`'s photos and saved back to `src/hsv_config.json`:

```bash
cd board_reader
python src/tuner.py board [-d em] [pattern]           # teal colour range + Canny/quad params
python src/tuner.py red_rectangle [-d em] [pattern]   # orientation-marker detection
python src/tuner.py tile_detector [-d em] [pattern]   # tile-presence colour/glyph model
python src/tuner.py letters [-d em] [pattern]         # CNN/template fusion weights
```

`-d`/`--difficulty` selects which `test/in/imgN_<difficulty>.jpg` photos to load
(any combination of `e`/`m`/`h`, default `e`); an explicit glob `pattern`
overrides it. See each subcommand's `--help`, or `tuner.py`'s module
docstring, for its keybindings.

## Training the letter/digit classifiers

Real-photo glyph crops feed both CNNs, always harvested from ground truth and
manually reviewed before they're trusted -- a tile only needs to be correctly
*detected* to be harvested, not correctly *classified*, and glyph extraction
can still crop badly even when detection and position are right.

Everything below is per language: `--lang <code>` picks a definition from
`languages/<code>.json`, models live in `src/models/<code>/`, and training data
in `src/data_train/<code>/`. The classifier refuses a checkpoint whose classes
are not letters of the language being loaded, so a mixed-up model fails loudly
instead of predicting confidently from the wrong alphabet.

```bash
cd board_reader

# letters
python scripts/harvest_templates.py --lang pl        # crop real glyphs at their ground-truth position -> staging/
python scripts/review_templates.py --lang pl        # manually accept/reject each crop
python scripts/clean_templates.py --lang pl          # auto-strip stray noise components from accepted crops
python scripts/generate_synthetic_dataset.py --lang pl --per-letter 400
python scripts/train_classifier.py --lang pl

# point-value digits (helps disambiguate accented/unaccented pairs, e.g. A vs A-ogonek).
# Only for a language whose tiles print a readable value -- `train_digit_classifier.py`
# refuses one with "use_point_prior": false, such as English.
python scripts/harvest_digit_templates.py --lang pl
python scripts/review_templates.py --lang pl --digits
python scripts/train_digit_classifier.py --lang pl --epochs 12
```

### Language support, honestly

| | Polish | English |
|---|---|---|
| Letter CNN | 32 classes, real + synthetic glyphs | 26 classes, **synthetic only** |
| Digit reader | yes | **off** |
| Validated against photos | 89 fixtures, 98.9% letter accuracy | **never** |

**English OCR is unvalidated and should be treated as experimental.** Its model
was trained purely on letters rendered from system fonts, because every photo
fixture in `test/` is of a Polish board and there is no English one to measure
against. That is not a gap you can close with code — it needs photographs.

Two things make it less bad than it sounds. The review UI already flags
low-confidence cells, so a weaker classifier degrades into "more cells to fix by
hand" rather than a silently wrong board; and the dictionary-correction pass in
`web/scan.py` fixes many misreads regardless of how they arose.

The digit channel is off for English because it cannot work there. Every tile
prints its point value, and reading it is a strong prior *in Polish*, where it
separates A/Ą and Z/Ź/Ż — pairs that differ by a diacritic and by points.
English has no diacritics, so the prior buys much less; and its 10-point tiles
(Q, Z) print a **two-glyph** value that `glyph_normalizer`'s single-component
digit extraction cannot represent at all. Enabling it would need either a `"10"`
class with two-component extraction (touching the most delicate CV code here) or
a width heuristic on the digit band. Neither is worth doing before there is a
photo set to measure the result against.

Board detection is a separate matter and orthogonal to language: `premium_layout.py`
and `hsv_config.json` are tuned to one *physical board edition*, so a differently
coloured board needs `tuner.py` regardless of which language is printed on it.

The digit flow has no `generate_synthetic_dataset.py` step of its own: unlike the
letter CNN, `train_digit_classifier.py` synthesises its training set in memory
each run, mixing it with the reviewed real crops in
`src/data/<code>/real_digit_templates/`.

Both CNNs are optional at inference time: if `torch` or the `.pt` weights
aren't available, `letter_classifier.py` degrades to template matching alone
(and skips digit-based disambiguation) rather than failing.

## Directory layout

```
board_reader/
├── src/                  # the pipeline itself (flat modules, no package/__init__.py)
│   ├── data/<code>/      # harvested real glyph/digit crops (staging/accepted/rejected) [not in git]
│   ├── data_train/<code>/         # generated CNN training set, letters                  [not in git]
│   ├── data_train_digits/<code>/  # generated CNN training set, point digits             [not in git]
│   ├── models/<code>/    # trained CNN weights (letter_cnn.pt, digit_cnn.pt)             [in git]
│   ├── data_paths.py     # where all of the above live -- one definition, see below
│   └── hsv_config.json   # tuned parameter presets, see Tuning                           [in git]
├── scripts/              # offline tooling: harvest -> review -> clean -> generate -> train,
│                         #   plus benchmark_pipeline.py (stage-by-stage timing)
├── tests/                # eval_*.py (accuracy against ground truth) + ground_truth.py loader
├── test/                 # fixtures: in/imgN_<difficulty>.jpg, out/boardN.txt ground truth [not in git]
└── ruff.toml             # lint config for this subtree
```

Note the two similarly-named directories: `tests/` holds runnable eval
*scripts*; `test/` holds the *data* they evaluate against.

Every path above is derived in `src/data_paths.py` rather than spelled out in
each script. That matters because these directories are handed between five tools
in sequence (harvest → review → clean → generate → train), and a disagreement
between any two of them is silent: the producer writes somewhere the consumer
never looks, so the workflow appears to run and simply has no effect.

**What a fresh clone does and doesn't get.** The repo root's `.gitignore` is a
strict whitelist, and it does include `*.pt` and `*.json` -- so both trained CNN
checkpoints (`src/models/<code>/letter_cnn.pt`, plus `digit_cnn.pt` where the
language uses one) and the tuned
`src/hsv_config.json` *are* committed. A fresh clone can therefore run the
pipeline at full accuracy with no training step.

What is *not* committed is the training and evaluation material:
`src/data/<code>/` (harvested glyph crops), `src/data_train/<code>/` (the
generated training set), and all of `test/` (the photos and ground truth). So a fresh clone can *run* the reader
but cannot reproduce the [accuracy](#accuracy) numbers above or retrain the CNNs
without a copy of those local files.

## Requirements

No separate `pyproject.toml` -- dependencies (`opencv-python`, `numpy`,
`torch`, `pillow`) are managed by the repo root's `pyproject.toml`/`uv.lock`,
alongside the `web/` app that consumes this pipeline.
