"""The web's photo scan through the engine's scanner, on a test photo.

The photos are private data (`board_reader/test`, a link into the main
checkout): without them these skip.
"""

import json
from pathlib import Path

import pytest

from web.scan import scan_board_image

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "board_reader" / "test" / "derived" / "scan_dataset"
# A medium photo whose layout is legal but for one deliberate non-word.
BOARD = 62


@pytest.fixture(scope="module")
def photo():
    record = DATASET / f"board{BOARD}.json"
    if not record.exists():
        pytest.skip("no test photos here")
    rec = json.loads(record.read_text(encoding="utf-8"))
    truth = [[ch.lower() for ch in row] for row in rec["truth"]]
    return str(ROOT / "board_reader" / "test" / "in" / rec["photo"]), truth


def test_a_photo_reads_as_the_page_expects(photo):
    path, truth = photo
    out = scan_board_image(path, None, "pl")
    assert set(out) == {"cells", "board"}
    assert len(out["cells"]) == 15 and all(len(row) == 15 for row in out["cells"])
    assert set(out["cells"][7][7]) == {
        "letter",
        "confidence",
        "alternatives",
        "flagged",
        "carried_over",
    }
    # Every square the decoder took away from what the photo shows is
    # flagged for the player: here, the non-word's.
    differs = {
        (r, c)
        for r in range(15)
        for c in range(15)
        if out["board"][r][c] != truth[r][c]
    }
    flagged = {
        (r, c) for r in range(15) for c in range(15) if out["cells"][r][c]["flagged"]
    }
    assert differs and differs == flagged


def test_the_last_confirmed_board_is_kept(photo):
    path, truth = photo
    out = scan_board_image(path, truth, "pl")
    assert out["board"] == truth
    assert not any(cell["flagged"] for row in out["cells"] for cell in row)


def test_a_confirmed_tile_the_photo_reads_otherwise_is_marked(photo):
    path, truth = photo
    prior = [row[:] for row in truth]
    r, c = next((r, c) for r in range(15) for c in range(15) if truth[r][c] not in "-?")
    prior[r][c] = "ź" if truth[r][c] != "ź" else "ż"
    out = scan_board_image(path, prior, "pl")
    # Tiles never change once played: the confirmed letter stays, marked.
    assert out["board"][r][c] == prior[r][c]
    carried = [
        (rr, cc)
        for rr in range(15)
        for cc in range(15)
        if out["cells"][rr][cc]["carried_over"]
    ]
    assert carried == [(r, c)]
