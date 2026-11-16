"""The engine's cell reader against PyTorch: the same exported file, the
same crops, the same scores -- and a whole board read through
`CellReader.read_board` against PyTorch on the crops it should cut. Skipped
until a reader has been trained (`train_cell_reader.py --final`)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from languages import engine_language
from scrablozaur import CellReader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board_reader" / "scripts"))
import train_cell_reader as training  # noqa: E402

MODEL = ROOT / "board_reader" / "src" / "models" / "pl" / "cell_reader.pt"


@pytest.fixture(scope="module")
def readers(pl):
    if not MODEL.exists():
        pytest.skip("no trained cell reader")
    checkpoint = torch.load(MODEL, weights_only=True)
    model = training.CellReader(2 + len(checkpoint["alphabet"]))
    model.load_state_dict(checkpoint["state"])
    model.eval()
    return (
        model,
        CellReader(engine_language(pl), str(MODEL.with_suffix(".bin"))),
        checkpoint,
    )


def test_scores_match_pytorch(readers):
    model, engine, checkpoint = readers
    crop = checkpoint["crop"]
    rng = np.random.default_rng(0)
    crops = rng.integers(0, 256, (4, crop, crop, 3)).astype(np.float32)
    with torch.no_grad():
        want = model(torch.from_numpy(crops).permute(0, 3, 1, 2) / 255).numpy()
    for one, expected in zip(crops, want):
        np.testing.assert_allclose(
            engine.raw(one.reshape(-1).tolist()), expected, rtol=1e-3, atol=1e-3
        )


def test_a_board_reads_as_pytorch_reads_its_crops(readers):
    model, engine, checkpoint = readers
    cell, crop, temperature = (
        engine.cell_size,
        checkpoint["crop"],
        checkpoint["temperature"],
    )
    side = 17 * cell
    rng = np.random.default_rng(1)
    # Smooth noise, nearer to a photo than independent pixels are.
    board = (
        rng.integers(0, 256, (side // 4 + 1, side // 4 + 1, 3))
        .repeat(4, 0)
        .repeat(4, 1)[:side, :side]
    )
    board = board.astype(np.uint8)
    read = engine.read_board(board.tobytes(), side)
    for r, c in [(0, 0), (7, 7), (14, 3), (5, 14)]:
        y, x = (2 * r + 3) * cell // 2 - crop // 2, (2 * c + 3) * cell // 2 - crop // 2
        pixels = (
            torch.from_numpy(
                board[y : y + crop, x : x + crop].astype(np.float32)
            ).permute(2, 0, 1)[None]
            / 255
        )
        with torch.no_grad():
            p = torch.softmax(model(pixels)[0] / temperature, 0).numpy()
        p_tile, letters, p_blank = read[r][c]
        assert p_tile == pytest.approx(1 - p[0], abs=1e-4)
        assert p_blank == pytest.approx(p[1] / (1 - p[0]), abs=1e-3)
        best = max(letters, key=lambda lp: lp[1])[0]
        assert best == checkpoint["alphabet"][int(np.argmax(p[2:]))]
