"""`decode_scan` through the Python bindings. The engine's own tests cover the
decoding; these pin the shapes, the prior handling and the errors that
`scrablozaur.pyi` promises."""

import pytest
from languages import engine_language
from scrablozaur import Dawg, decode_scan

GRID = 15
Reading = tuple[float, list[tuple[str, float]], float]
EMPTY: Reading = (0.01, [], 0.0)


@pytest.fixture(scope="module")
def dawg(pl):
    return Dawg(engine_language(pl), str(pl.dawg), str(pl.gaddag))


def sure(words) -> list[list[Reading]]:
    """Readings sure of every square of `words`, given as (row, col, word, across)."""
    readings = [[EMPTY] * GRID for _ in range(GRID)]
    for r, c, word, across in words:
        for i, ch in enumerate(word):
            rr, cc = (r, c + i) if across else (r + i, c)
            readings[rr][cc] = (0.99, [(ch, 0.95)], 0.001)
    return readings


def letters(readings):
    return [
        ["-" if p < 0.5 else (ls[0][0] if ls else "?") for p, ls, _ in row]
        for row in readings
    ]


def test_a_misread_comes_back_corrected(dawg):
    truth = letters(sure([(7, 6, "nos", True), (7, 7, "oko", False)]))
    readings = sure([(7, 6, "nos", True), (7, 7, "oko", False)])
    readings[7][7] = (0.99, [("ę", 0.95)], 0.001)
    out = decode_scan(dawg, readings)
    assert out.legal and out.complete
    assert out.letters == truth
    assert out.corrected[7][7] and sum(map(sum, out.corrected)) == 1
    assert not any(map(any, out.flagged))
    assert out.cost > 0 and out.work > 0


def test_the_previous_board_and_its_blanks_are_kept(dawg):
    prior = letters(sure([(7, 6, "kot", True)]))
    prior_blanks = [[False] * GRID for _ in range(GRID)]
    prior_blanks[7][8] = True
    readings = sure([(7, 6, "kota", True)])
    readings[7][7] = (0.99, [("a", 0.95)], 0.001)  # "kata" is a word too
    out = decode_scan(dawg, readings, prior, prior_blanks)
    assert "".join(out.letters[7][6:10]) == "kota"
    assert out.blanks[7][8] and sum(map(sum, out.blanks)) == 1


def test_bad_input_is_a_value_error(dawg):
    with pytest.raises(ValueError, match="15x15"):
        decode_scan(dawg, sure([])[:3])
    prior = [["-"] * GRID for _ in range(GRID)]
    prior[7][7] = "q"
    with pytest.raises(ValueError, match="not a letter"):
        decode_scan(dawg, sure([]), prior)
