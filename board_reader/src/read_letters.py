"""Classify every detected tile's letter and assemble the final 15x15
board state.

Ported from ocr/scrabble_reader/pipeline.py's `_recognize()`/`_fast_sources()`
(the orchestration) plus `board_builder.py`'s `BoardBuilder.build()` (the
final assembly), adapted to this project's house style: free functions
reading a `_params()` preset, no dataclass Config/BoardBuilder instance.

`extract_tile_patches()`/`_cell_quad`/`_sample_quad` live in read_board.py
(it already owns all the perspective-sampling geometry); this module owns
everything past "here is a 160px BGR crop of a detected tile": glyph
normalisation (glyph_normalizer.py), letter classification
(letter_classifier.py), and assembling the results into a board.
"""


import cv2
import numpy as np
import glyph_normalizer as gn
from letter_classifier import (
    POLISH_ALPHABET,
    _params,
    classify_cnn_batch,
    classify_digit_cnn_batch,
    classify_digit_templates,
    classify_templates,
    fuse_predictions,
    points_distribution,
)
from parallel_utils import get_executor
from premium_layout import GRID
from read_board import _cell_quad, extract_tile_patches


def _fast_sources(glyphs, p):
    """Run CNN (batched) + template matcher + the tile's own point-value digit.

    Template matching only runs for a glyph when the CNN's own top
    probability is below template_trigger_confidence -- measured on the
    test set, this actually triggers on a majority of photos (each tile
    independently gated on its own CNN confidence), so the per-glyph
    matchTemplate cost -- the biggest single contributor to classify_tiles()'s
    wall time after gn.normalize -- is run across the shared thread pool
    below: each glyph's cnn/template/digit lookup is independent of every
    other glyph's (the batched CNN/digit-CNN calls above already produced
    every glyph's distribution), so this is a plain parallel map, order
    preserved via executor.map so the returned list still lines up
    positionally with `glyphs`.
    """
    cnn = classify_cnn_batch(glyphs)
    digit_cnn = classify_digit_cnn_batch(glyphs)

    def _one(glyph, cnn_dist, digit_dist):
        pr = {}
        cnn_top = 0.0
        if cnn_dist:
            pr["cnn"] = (cnn_dist, p["weight_cnn"])
            cnn_top = max(cnn_dist.values())
        if cnn_top < p["template_trigger_confidence"]:
            tm = classify_templates(glyph)
            if tm:
                pr["template"] = (tm, p["weight_template"])
        if digit_dist is None:
            digit_dist = classify_digit_templates(glyph)
        if digit_dist:
            pts = points_distribution(digit_dist)
            if pts:
                pr["points"] = (pts, p["weight_points"])
        return pr

    return list(get_executor().map(_one, glyphs, cnn, digit_cnn))


ROTATION_SUBSET_SIZE = 10
ROTATION_SUBSET_MIN_TILES = 12  # at or below this many tiles, search rotation on every tile instead of a subset


def _resolve_rotation(tile_verdicts, patches):
    """Global rotation decision for the whole board (not per-tile -- see
    module docstring): the ~10 tiles with the clearest glyphs (by
    tile_detector's own glyph_score, already computed) settle it as
    reliably as the whole board, at a fraction of the normalisation cost.
    Uses the CNN alone -- a relative comparison across rotations doesn't
    need the template matcher.

    This is a defensive backstop for rotate_board.py's silent-failure path
    (no red rectangle marker found -> board returned unrotated), not a
    redundant re-solve of an already-solved problem: rotate_board.py plus
    grid_detector.py's premium-pattern registration already resolve whole-
    board orientation independently, so k=0 should win on nearly every
    photo where those succeeded.

    Benchmarked cost: this runs on nearly every photo (resolve_rotation
    defaults on) and normalizes up to 4*ROTATION_SUBSET_SIZE glyphs --
    roughly half of classify_tiles()'s total gn.normalize call volume on a
    typical photo, despite the "only ~10 tiles" framing above. The 4
    rotations' glyphs are therefore normalized as one flattened, threaded
    batch (independent per (tile, k) pair) and classified with a single
    classify_cnn_batch call instead of 4 separate ones -- a pure batching
    change, not a behaviour change: the model runs in eval() mode, so
    BatchNorm uses frozen running stats and a glyph's predicted
    distribution can't depend on what else shares its batch.
    """
    subset = (
        sorted(tile_verdicts, key=lambda v: -v.glyph_score)[:ROTATION_SUBSET_SIZE]
        if len(tile_verdicts) > ROTATION_SUBSET_MIN_TILES
        else tile_verdicts
    )
    pairs = [(v, k) for k in range(4) for v in subset]
    glyphs = list(get_executor().map(lambda vk: gn.normalize(patches[(vk[0].row, vk[0].col)], rotation_k=vk[1]), pairs))
    cnn = classify_cnn_batch(glyphs)

    best_k, best_score = 0, -1.0
    for k in range(4):
        idx = [i for i, (_, kk) in enumerate(pairs) if kk == k]
        confs = [max(cnn[i].values()) for i in idx if cnn[i]]
        conf_score = float(np.mean(confs)) if confs else 0.0
        # The score digit only ever sits bottom-right on a real tile: its
        # position is a letter-independent orientation cue that breaks ties
        # between rotation-symmetric-looking letters (I, Z, N, O, H...).
        digits = [glyphs[i].digit_score for i in idx if glyphs[i].has_glyph]
        digit = float(np.mean(digits)) if digits else 0.0
        score = conf_score + 0.35 * digit
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def _rotate_position(row, col, k):
    """Where (row, col) lands after rotating the whole board k*90 degrees
    clockwise -- the inverse of the correction _resolve_rotation() found
    needs applying, so words in the returned board still read left-to-right/
    top-to-bottom instead of only their individual letters being upright."""
    for _ in range(k % 4):
        row, col = col, GRID - 1 - row
    return row, col


def classify_tiles(rotated, mesh, verdicts, global_shift=None, **param_overrides):
    """Per-tile-position -> (letter, confidence, ranked alternatives).
    Only cells `verdicts` flags is_tile are classified. Returns {} if there
    are no tiles. Richer than classify_board()'s flattened output -- the
    tuner's overlay and a future dictionary-correction pass both want the
    confidence/alternatives, classify_board() just discards them.

    NOTE on coordinates: if rotation resolves to a non-zero k (see
    _resolve_rotation()), every returned position is rotated to match --
    the coordinate frame of this function's *output* is not guaranteed to
    be the same as `mesh`'s/`verdicts`' own frame when that happens. Expect
    k=0 (no remapping) on nearly every photo.
    """
    p = _params(param_overrides)
    tile_verdicts = [v for v in verdicts if v.is_tile]
    if not tile_verdicts:
        return {}

    patches = extract_tile_patches(rotated, mesh, verdicts, global_shift=global_shift)
    rotation_k = _resolve_rotation(tile_verdicts, patches) if p["resolve_rotation"] else 0
    if rotation_k:
        print(
            f"read_letters: board rotated {rotation_k * 90} degrees for recognition (rotate_board.py may have missed it)"
        )

    glyphs = list(get_executor().map(lambda v: gn.normalize(patches[(v.row, v.col)], rotation_k=rotation_k), tile_verdicts))
    preds = _fast_sources(glyphs, p)

    # Reprocess weak glyphs with alternative binarisation variants.
    for i, (v, glyph, pred) in enumerate(zip(tile_verdicts, glyphs, preds)):
        _, conf, _, _ = fuse_predictions(pred)
        if glyph.has_glyph and conf >= p["reprocess_confidence"]:
            continue
        for variant in (1, 2):
            alt = gn.normalize(patches[(v.row, v.col)], rotation_k=rotation_k, variant=variant)
            if not alt.has_glyph:
                continue
            alt_pred = _fast_sources([alt], p)[0]
            _, alt_conf, _, _ = fuse_predictions(alt_pred)
            if alt_conf > conf:
                glyphs[i], preds[i], conf = alt, alt_pred, alt_conf

    readings = {}
    for v, pred in zip(tile_verdicts, preds):
        letter, conf, alts, _tops = fuse_predictions(pred)
        r, c = _rotate_position(v.row, v.col, rotation_k)
        readings[(r, c)] = (letter, conf, alts)
    return readings


def classify_board(rotated, mesh, verdicts, global_shift=None, **param_overrides):
    """The pipeline's final, minimal-shape output: 15x15 list of
    single-character strings ('-' empty, '?' unrecognised letter)."""
    readings = classify_tiles(rotated, mesh, verdicts, global_shift, **param_overrides)
    board = [["-"] * GRID for _ in range(GRID)]
    for (r, c), (letter, _conf, _alts) in readings.items():
        board[r][c] = letter if letter and letter in POLISH_ALPHABET else "?"
    return board


def draw_letter_overlay(rotated, mesh, verdicts, board):
    """Debug image: green quad per tile with its classified letter drawn
    large and centred, red quad for empty cells -- like read_board.py's
    draw_tile_overlay() but showing the letter instead of a confidence
    number."""
    out = rotated.copy()
    for v in verdicts:
        quad = _cell_quad(mesh, v.row, v.col).astype(np.int32)
        color = (0, 200, 0) if v.is_tile else (0, 0, 200)
        cv2.polylines(out, [quad], True, color, 3)
        if v.is_tile:
            letter = board[v.row][v.col]
            center = quad.mean(axis=0).astype(int)
            (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
            cv2.putText(
                out, letter, (center[0] - tw // 2, center[1] + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3
            )
    return out
