# board_reader

Computer-vision pipeline that turns a photo of a physical Polish Scrabble
board into a 15x15 board state (letters + confidence + ranked alternatives).
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

```bash
cd board_reader

# letters
python scripts/harvest_templates.py                 # crop real glyphs at their ground-truth position -> staging/
python scripts/review_templates.py                  # manually accept/reject each crop
python scripts/clean_templates.py                    # auto-strip stray noise components from accepted crops
python scripts/generate_synthetic_dataset.py --out src/data_train --per-letter 400
python scripts/train_classifier.py --data src/data_train --out src/models/letter_cnn.pt

# point-value digits (helps disambiguate accented/unaccented pairs, e.g. A vs A-ogonek)
python scripts/harvest_digit_templates.py
python scripts/review_templates.py --digits
python scripts/train_digit_classifier.py --epochs 12
```

Both CNNs are optional at inference time: if `torch` or the `.pt` weights
aren't available, `letter_classifier.py` degrades to template matching alone
(and skips digit-based disambiguation) rather than failing.

## Directory layout

```
board_reader/
├── src/                  # the pipeline itself (flat modules, no package/__init__.py)
│   ├── data/             # harvested + reviewed real glyph/digit templates (staging/accepted/rejected)
│   ├── data_train/       # generated CNN training set (letters)
│   ├── data_train_digits/ # generated CNN training set (digits)
│   ├── models/           # trained CNN weights (letter_cnn.pt, digit_cnn.pt)
│   └── hsv_config.json   # tuned parameter presets (see Tuning)
├── scripts/              # offline tooling: harvest -> review -> clean -> generate -> train
├── tests/                # eval_*.py (accuracy against ground truth) + ground_truth.py loader
└── test/                 # test fixtures: in/imgN_<difficulty>.jpg photos, out/boardN.txt ground truth
```

Note the two similarly-named directories: `tests/` holds runnable eval
*scripts*; `test/` holds the *data* they evaluate against.

**Only the `.py` files above are in version control.** The repo root's
`.gitignore` is a strict whitelist (`*.txt`, `.in`, `.rs`, `.py`, `.pyi` plus
a few named files); everything else -- `src/data/`, `src/data_train*/`,
`src/models/*.pt`, `src/hsv_config.json`, and all of `test/` -- is local-disk
only, gitignored, and absent from a fresh clone. The [accuracy](#accuracy)
numbers above assume this working copy's local models/data are present; a
fresh clone starts with no trained CNN weights, no harvested templates, and
no test fixtures, and needs either a copy of those local files or a full run
through [Training](#training-the-letterdigit-classifiers) (synthetic font
templates alone still let `letter_classifier.py` degrade gracefully, just at
lower accuracy than the numbers above).

## Requirements

No separate `pyproject.toml` -- dependencies (`opencv-python`, `numpy`,
`torch`, `pillow`) are managed by the repo root's `pyproject.toml`/`uv.lock`,
alongside the `web/` app that consumes this pipeline.
