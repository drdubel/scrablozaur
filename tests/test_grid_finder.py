"""The engine's grid finder against PyTorch: the same exported file, the
same heatmap -- and, where the private test photos are present, the same
corners found by the whole two-pass search. Skipped until a grid finder has
been trained (`train_grid_finder.py --final`)."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scrablozaur import GridFinder

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "board_reader" / "scripts"))
import train_grid_finder as training  # noqa: E402

MODEL = ROOT / "board_reader" / "src" / "models" / "pl" / "grid_finder.pt"


@pytest.fixture(scope="module")
def finders():
    if not MODEL.exists():
        pytest.skip("no trained grid finder")
    model = training.GridFinder()
    model.load_state_dict(torch.load(MODEL, weights_only=True)["state"])
    return model.eval(), GridFinder(str(MODEL.with_suffix(".bin")))


def test_heatmap_matches_pytorch(finders):
    model, engine = finders
    side = training.SIDE
    rng = np.random.default_rng(0)
    image = (
        rng.integers(0, 256, (side // 8, side // 8, 3))
        .repeat(8, 0)
        .repeat(8, 1)
        .astype(np.float32)
    )
    with torch.no_grad():
        want = (
            model(torch.from_numpy(image).permute(2, 0, 1)[None] / 255)[0]
            .numpy()
            .ravel()
        )
    got = np.array(engine.raw(image.ravel().tolist()))
    np.testing.assert_allclose(got, want, rtol=1e-3, atol=1e-3)


def test_corners_match_the_python_search_on_a_photo(finders):
    import cv2

    photos = sorted((ROOT / "board_reader" / "test" / "in").glob("img*_e.jpg"))[:3]
    if not photos:
        pytest.skip("the private test photos are not here")
    model, engine = finders
    for path in photos:
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        want = training.find(
            model, torch.from_numpy(rgb).permute(2, 0, 1).float() / 255
        )
        got = engine.find(rgb.tobytes(), w, h)
        assert (want is None) == (got is None)
        if got is not None:
            np.testing.assert_allclose(np.array(got), want, atol=0.05 * max(w, h) / 256)
