"""Validation of the language definition files themselves.

Pure data checks, no engine, no dictionary load -- these run in milliseconds and
are what catches a hand-edit that puts a letter in `alphabet` but forgets it in
`tiles`, or gives a letter a count of zero, or drifts a language past the
engine's 32-letter ceiling.
"""

import json

import languages
import pytest


def test_at_least_one_language_is_defined():
    assert languages.available(), "languages/ has no definition files"


def test_load_rejects_unknown_code():
    with pytest.raises(ValueError, match="unknown language"):
        languages.load("definitely-not-a-language")


def test_spec_parses_and_validates(spec):
    """`load()` does the validating, so reaching here at all means the file is
    structurally sound. These assertions cover what it deliberately does not
    reject outright."""
    assert spec.name
    assert spec.alphabet
    assert spec.blank not in spec.alphabet


def test_bag_expands_to_the_declared_counts(spec):
    from collections import Counter

    assert Counter(spec.bag) == Counter(spec.counts)
    assert len(spec.bag) == spec.total_tiles


def test_bag_is_in_collation_order_with_blanks_last(spec):
    """Boards are seeded by shuffling this list, so its order is part of the
    engine's observable behaviour -- a reorder silently changes every seeded
    deal and every reproducible benchmark."""
    letters = [ch for ch in spec.bag if ch != spec.blank]
    assert letters == sorted(letters, key=spec.alphabet.index)
    assert set(spec.bag[len(letters):]) <= {spec.blank}


def test_eval_alphabet_is_codepoint_order_with_the_blank_first(spec):
    """The leave-value net's feature order. A net whose input order disagrees
    with its caller's does not fail -- it returns confident nonsense."""
    assert spec.eval_alphabet == sorted(set(spec.bag))
    assert spec.eval_alphabet[0] == spec.blank
    assert len(spec.eval_alphabet) == len(spec.alphabet) + 1


def test_a_real_scrabble_set_has_100_tiles(spec):
    """Every official distribution is 100 tiles with 2 blanks. Not a law of
    nature, but a typo in a count is far more likely than a deliberate variant,
    and this catches it instantly."""
    assert spec.total_tiles == 100
    assert spec.counts[spec.blank] == 2


def test_vowels_and_consonants_partition_the_alphabet(spec):
    assert set(spec.vowels) | set(spec.consonants) == set(spec.alphabet)
    assert not (set(spec.vowels) & set(spec.consonants))


def test_declared_dictionary_files_exist(spec):
    """Optional artifacts (OCR templates, an untrained leave net) are allowed to
    be absent, but the dictionary a language is *for* must be present."""
    for path in (spec.words, spec.dawg, spec.gaddag):
        assert path.exists(), f"{spec.code}: {path} is declared but missing"


def test_json_keeps_tiles_in_alphabet_order():
    """Purely a readability contract, but the tile table is the block most
    likely to be edited by hand and a stable order makes a diff reviewable."""
    for code in languages.available():
        raw = json.loads((languages.LANGUAGES_DIR / f"{code}.json").read_text(encoding="utf-8"))
        listed = [ch for ch in raw["tiles"] if ch != raw["blank"]]
        assert listed == list(raw["alphabet"]), f"{code}.json: tiles are not in alphabet order"
