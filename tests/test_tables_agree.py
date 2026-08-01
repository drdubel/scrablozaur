"""The anti-drift test: every consumer's alphabet / point / distribution table
must agree with `languages/<code>.json`.

This is what makes "one source of truth" true rather than aspirational. While a
consumer still carries its own hardcoded copy, this test proves the JSON is
equivalent to it -- which is the licence to delete that copy. Once a consumer
reads the JSON, the same assertion becomes a regression guard for free.

Checks taking the `spec` fixture run for every installed language; those taking
`pl` are ones where only Polish has the artifact to compare against (the
committed leave net, the photo-trained OCR models).

Several tests here now assert the *absence* of a table rather than its
contents -- once a consumer reads the JSON, the thing worth guarding is that
nobody pastes a literal back in.
"""

import os

import pytest
import languages
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


def test_web_tile_bag_is_dealt_from_the_json(pl):
    """`web/game.py` used to hold its own 100-tile table, separate from the
    engine's. The bag a game deals from is now built from the definition file,
    so the visible bag and the engine's can no longer disagree."""
    from collections import Counter

    from web.game import TileBag

    assert Counter(TileBag.full(pl).tiles) == Counter(pl.counts)


def test_web_game_no_longer_hardcodes_a_tile_table(pl):
    import web.game

    assert not hasattr(web.game, "TILE_COUNTS"), (
        "a hardcoded tile table is back in web/game.py -- it belongs in languages/*.json"
    )


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


def test_a_leave_net_from_another_language_is_refused(pl):
    """A net encodes one input slot per tile type. Fed another language's rack
    it does not fail -- it drops the letters that alphabet lacks and scores the
    rest against the wrong distribution, which is worse than an error."""
    pytest.importorskip("model", reason="smart_player needs torch + numpy")

    codes = [c for c in languages.available() if c != pl.code]
    if not codes:
        pytest.skip("needs a second language")

    # The engine-side check is on feature width, which differs whenever the two
    # languages have different numbers of tile types.
    other = languages.load(codes[0])
    if len(other.eval_alphabet) == len(pl.eval_alphabet):
        pytest.skip("languages happen to encode the same width")
    assert pl.leave_net is not None

    with pytest.raises(OSError, match="inputs"):
        LeaveNet(_engine_lang(other), str(pl.leave_net.weights))


# ── The board renderer (web/static/js/board.js) ──────────────────────────────


def _js_sources() -> dict[str, str]:
    import glob

    return {
        os.path.basename(path): open(path, encoding="utf-8").read()
        for path in glob.glob(os.path.join(_ROOT, "web", "static", "js", "*.js"))
    }


def test_no_js_file_references_a_global_point_table(pl):
    """`LETTER_VALUES` was a global in board.js that three files read. Deleting
    it without checking the other two shipped a ReferenceError to the rack
    renderer and the scan grid -- checking only its own file is what let that
    through, so this sweeps every script."""
    for name, source in _js_sources().items():
        assert "LETTER_VALUES" not in source, (
            f"{name} references the removed LETTER_VALUES global -- point values come "
            f"from Languages.letterValues() / setLetterValues() now"
        )


def test_no_js_file_has_a_hardcoded_letter_class(pl):
    """Which keys count as letters follows the language now, via
    `Languages.isLetter` -- a fixed character class silently rejects a letter
    another alphabet has. Prose in Polish is fine and stays; a *character
    class* is the language treated as data.
    """
    for name, source in _js_sources().items():
        assert "a-ząćęłńóśźż" not in source, f"{name} has a hardcoded Polish letter class"
        assert "a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ" not in source, f"{name} has a hardcoded Polish letter class"
    assert "Languages.isLetter" in _js_sources()["game.js"]


# ── The board scanner (board_reader/src/letter_classifier.py) ────────────────


def test_ocr_tables_match_the_json(spec):
    """The classifier's alphabet and point tables are derived from the language
    definition now, so this holds for every language rather than just Polish."""
    classifier = pytest.importorskip(
        "letter_classifier", reason="board_reader needs cv2 + torch"
    )

    classifier.set_language(spec)
    assert classifier.alphabet() == spec.alphabet.upper()
    assert classifier.letter_points() == {
        ch.upper(): pts for ch, pts in spec.points.items() if ch != spec.blank
    }
    # The tile set prints no 0 (a blank has no digit), so the digit reader is
    # deliberately limited to values that can actually appear.
    expected = sorted({pts for ch, pts in spec.points.items() if ch != spec.blank})
    assert classifier.valid_digits() == tuple(str(d) for d in expected)
    # Restore the default so test order cannot matter.
    classifier.set_language(classifier.DEFAULT_LANGUAGE)


def test_ocr_model_classes_match_the_language(spec):
    """A checkpoint trained for another alphabet does not fail -- it predicts
    confidently from the wrong letter set. The loader refuses it; this proves
    the shipped models are the right ones."""
    classifier = pytest.importorskip(
        "letter_classifier", reason="board_reader needs cv2 + torch"
    )
    if not spec.has_ocr:
        pytest.skip(f"{spec.code} has no OCR models")

    classifier.set_language(spec)
    _model, classes = classifier._get_cnn()
    assert classes, f"{spec.code}: letter model did not load"
    assert set(classes) <= set(spec.alphabet.upper())
    classifier.set_language(classifier.DEFAULT_LANGUAGE)


# ── The definition file's own paths ──────────────────────────────────────────


def test_declared_artifacts_are_the_ones_actually_loaded(pl):
    """The JSON claims where the dictionary and models live. Until every loader
    reads it, this proves the claim matches the paths they hardcode."""
    from web.engine import get_pack

    pack = get_pack(pl.code)
    assert pack.spec.dawg.resolve() == pl.dawg.resolve()
    assert pack.spec.gaddag.resolve() == pl.gaddag.resolve()
    # And the loaded dictionary really is in this language.
    assert pack.dawg.lang.alphabet == pl.alphabet

    model = pytest.importorskip("model", reason="smart_player needs torch + numpy")
    assert pl.leave_net is not None
    assert os.path.realpath(model._WEIGHTS_PATH) == os.path.realpath(pl.leave_net.checkpoint)

    assert pl.ocr is not None
    assert pl.has_ocr, "the Polish letter model should be committed"
    assert pl.ocr.digit_cnn is not None and pl.ocr.digit_cnn.exists()
