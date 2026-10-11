"""Export and load several languages in the same process, as the web app does."""

import shutil

import pytest
import torch

from export_weights import export
from languages import engine_language
from model import N_BOARD_FEATURES, encode_leave, get_model, model_alphabet
from scrablozaur import LeaveNet
from sim_player import get_net


def test_missing_binary_is_exported_with_the_checkpoints_encoding(spec, tmp_path):
    if spec.leave_net is None:
        pytest.skip(f"{spec.code} has no leave model")
    checkpoint = str(spec.leave_net.checkpoint)
    binary = tmp_path / f"{spec.code}.bin"
    lang = engine_language(spec)
    net = get_net(checkpoint, str(binary), lang=lang)
    assert binary.exists()
    assert net.alphabet() == spec.eval_alphabet

    # Loading successfully is insufficient: verify that the exported weights
    # evaluate the same real feature vectors as the source PyTorch checkpoint.
    alphabet = model_alphabet(checkpoint)
    for leave in ("", "ae?", spec.alphabet[-1] * 2):
        features = encode_leave(leave, 43, (0.5, 0.25, 0.75, 1.0, 0.4), alphabet)
        assert len(features) == len(alphabet) + 1 + N_BOARD_FEATURES
        with torch.no_grad():
            expected = get_model(checkpoint)(features).item()
        assert net.raw(features.tolist()) == pytest.approx(expected, abs=1e-4)


def test_custom_checkpoint_defaults_to_its_own_binary(spec, tmp_path):
    if spec.leave_net is None:
        pytest.skip(f"{spec.code} has no leave model")
    checkpoint = tmp_path / f"{spec.code}.pt"
    shutil.copyfile(spec.leave_net.checkpoint, checkpoint)
    binary = export(str(checkpoint))
    assert binary == str(checkpoint.with_suffix(".bin"))
    assert LeaveNet(engine_language(spec), binary).alphabet() == spec.eval_alphabet
