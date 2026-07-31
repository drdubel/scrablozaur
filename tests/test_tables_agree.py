"""The anti-drift test: every consumer's alphabet / point / distribution table
must agree with `languages/<code>.json`.

This is what makes "one source of truth" true rather than aspirational. While a
consumer still carries its own hardcoded copy, this test proves the JSON is
equivalent to it -- which is the licence to delete that copy. Once a consumer
reads the JSON, the same assertion becomes a regression guard for free.

Everything here is checked against Polish specifically: the Rust engine, the JS
renderer and the OCR classifier are all still compiled or written against one
hardcoded language, and this file is what lets them stop being.
"""

import os
import re

import pytest
from languages import engine_language as _engine_lang
from scrablozaur import Board, LeaveNet

_ROOT = os.path.join(os.path.dirname(__file__), "..")


# ── The engine (src/lib.rs) ──────────────────────────────────────────────────


def test_engine_tile_bag_matches_the_json(pl):
    """Order and multiplicity both. `Board.seeded()` shuffles this list, so a
    reordering would silently change every seeded deal in the benchmarks."""
    assert Board(_engine_lang(pl)).fresh_tile_bag() == pl.bag


def test_engine_letter_points_match_the_json(pl):
    board = Board(_engine_lang(pl))
    for letter, points in pl.points.items():
        assert board.letter_points(letter) == points, f"{letter!r}"


def test_engine_scores_the_alphabet_case_insensitively(pl):
    """`letter_points` uppercases before matching, and the web layer lowercases
    at the HTTP boundary -- so both cases must land on the same value."""
    board = Board(_engine_lang(pl))
    for letter in pl.alphabet:
        assert board.letter_points(letter.upper()) == pl.points[letter]


def test_leave_net_feature_order_matches_the_derived_eval_alphabet(pl):
    """Both the engine and `smart_player/model.py` derive this order from the
    tile distribution. They must agree, or the committed checkpoints are being
    fed a permuted feature vector -- which does not fail, it just predicts
    nonsense."""
    assert pl.leave_net is not None
    net = LeaveNet(_engine_lang(pl), str(pl.leave_net.weights))
    assert net.alphabet() == pl.eval_alphabet


# ── The web app (web/game.py) ────────────────────────────────────────────────


def test_web_tile_counts_match_the_json(pl):
    from web.game import TILE_COUNTS

    assert TILE_COUNTS == pl.counts


# ── The strategy heuristic (src/strategy.py) ─────────────────────────────────


def test_strategy_vowels_and_consonants_match_the_json(pl):
    import strategy

    assert set(strategy.VOWELS) == set(pl.vowels)
    assert set(strategy.CONSONANTS) == set(pl.consonants)


# ── The leave-value model (smart_player/model.py) ────────────────────────────


def test_smart_player_alphabet_matches_the_json(pl):
    model = pytest.importorskip("model", reason="smart_player needs torch + numpy")

    assert model.ALPHABET == pl.eval_alphabet
    assert model.INPUT_DIM == len(pl.eval_alphabet) + 1 + model.N_BOARD_FEATURES


# ── The board renderer (web/static/js/board.js) ──────────────────────────────


def test_js_letter_values_match_the_json(pl):
    """The JS copy exists because the renderer draws the point value in the
    corner of every tile. Parsed out of the source rather than trusted, since
    nothing else would notice it drifting."""
    source = open(os.path.join(_ROOT, "web", "static", "js", "board.js"), encoding="utf-8").read()
    body = re.search(r"const LETTER_VALUES = \{(.*?)\};", source, re.S)
    assert body, "board.js no longer defines LETTER_VALUES -- update or drop this test"

    js_values = {ch: int(n) for ch, n in re.findall(r"'(.)'\s*:\s*(\d+)", body.group(1))}
    assert js_values == pl.points


# ── The board scanner (board_reader/src/letter_classifier.py) ────────────────


def test_ocr_tables_match_the_json(pl):
    classifier = pytest.importorskip(
        "letter_classifier", reason="board_reader needs cv2 + torch"
    )

    assert classifier.POLISH_ALPHABET == pl.alphabet.upper()
    assert classifier.LETTER_POINTS == {ch.upper(): pts for ch, pts in pl.points.items() if ch != pl.blank}
    # The physical tile set prints no 0 and has no blanks, so the digit reader
    # is deliberately trained only on the values that can actually appear.
    expected_digits = sorted({pts for ch, pts in pl.points.items() if ch != pl.blank})
    assert classifier.VALID_DIGITS == "".join(str(d) for d in expected_digits)


# ── The definition file's own paths ──────────────────────────────────────────


def test_declared_artifacts_are_the_ones_actually_loaded(pl):
    """The JSON claims where the dictionary and models live. Until every loader
    reads it, this proves the claim matches the paths they hardcode."""
    from web.engine import DAWG_PATH, GADDAG_PATH

    assert DAWG_PATH.resolve() == pl.dawg.resolve()
    assert GADDAG_PATH.resolve() == pl.gaddag.resolve()

    model = pytest.importorskip("model", reason="smart_player needs torch + numpy")
    assert pl.leave_net is not None
    assert os.path.realpath(model._WEIGHTS_PATH) == os.path.realpath(pl.leave_net.checkpoint)

    classifier = pytest.importorskip("letter_classifier", reason="board_reader needs cv2 + torch")
    assert pl.ocr is not None
    assert os.path.realpath(classifier.CNN_WEIGHTS) == os.path.realpath(pl.ocr.letter_cnn)
    assert os.path.realpath(classifier.DIGIT_CNN_WEIGHTS) == os.path.realpath(pl.ocr.digit_cnn)
    assert os.path.realpath(classifier.REAL_TEMPLATES_DIR) == os.path.realpath(pl.ocr.real_templates)
