"""Export a trained LeaveValueNet checkpoint to a flat binary the Rust engine
can load.

    python smart_player/export_weights.py                     # -> models/<lang>/leave_value.bin
    python smart_player/export_weights.py --out other.bin --model-path other.pt

Why. The simulation player evaluates a leave at every rollout ply -- hundreds
of thousands of times per move. Crossing back into Python for a 13k-parameter
MLP would cost far more in FFI and GIL traffic than the ~10k multiply-adds it
is actually asking for, so the engine runs the net itself (see `LeaveNet` in
src/lib.rs). This writes the weights in the layout it expects.

The `.pt` checkpoint stays the source of truth: it is what `train.py` writes
and what `SmartPlayer` loads. Re-run this after training.

Format (little-endian throughout):

    magic      8 bytes   b"SCRBNET1"
    input_dim  u32       must match the alphabet + scalars the engine encodes
    n_layers   u32
    per layer: in_dim u32, out_dim u32
    per layer: weights (out_dim * in_dim f32, row-major), bias (out_dim f32)

ReLU is applied between layers and not after the last one, matching
LeaveValueNet.forward.
"""

import argparse
import os
import struct

import torch

from model import ALPHABET, DEFAULT_WEIGHTS_PATH, INPUT_DIM, LANGUAGE, _SPEC, get_model

MAGIC = b"SCRBNET1"
# Per-language, alongside the `.pt` it is exported from. The engine checks the
# feature width when it loads one, so a `.bin` from another language is caught
# rather than run against the wrong tile alphabet.
DEFAULT_BIN_PATH = (
    str(_SPEC.leave_net.weights)
    if _SPEC.leave_net
    else os.path.join(os.path.dirname(__file__), "models", LANGUAGE, "leave_value.bin")
)


def export(model_path: str = DEFAULT_WEIGHTS_PATH, out_path: str = DEFAULT_BIN_PATH) -> str:
    model = get_model(model_path)

    layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]
    if not layers:
        raise ValueError(f"{model_path} has no Linear layers to export")
    if layers[0].in_features != INPUT_DIM:
        raise ValueError(
            f"{model_path} takes {layers[0].in_features} inputs but the current "
            f"encoding produces {INPUT_DIM} -- retrain before exporting."
        )
    if layers[-1].out_features != 1:
        raise ValueError(f"{model_path} must end in a scalar head, got {layers[-1].out_features}")

    with open(out_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", INPUT_DIM, len(layers)))
        for layer in layers:
            f.write(struct.pack("<II", layer.in_features, layer.out_features))
        for layer in layers:
            w = layer.weight.detach().cpu().numpy().astype("<f4")
            b = layer.bias.detach().cpu().numpy().astype("<f4")
            f.write(w.tobytes())
            f.write(b.tobytes())

    shapes = " -> ".join([str(layers[0].in_features)] + [str(x.out_features) for x in layers])
    print(f"exported {model_path}")
    print(f"      -> {out_path}  ({shapes}, {len(ALPHABET)}-tile alphabet)")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", default=DEFAULT_WEIGHTS_PATH)
    ap.add_argument("--out", default=DEFAULT_BIN_PATH)
    args = ap.parse_args()
    export(args.model_path, args.out)
