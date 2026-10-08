"""Concatenate several `generate_data.py` outputs into one dataset.

`generate_data.py` holds every sample in memory and then concatenates, so its
peak is roughly twice the final size. A 2M-game run is ~50M samples, and that
peak is enough to get the process killed on a machine that comfortably handles
the finished array -- which looks like a silent death at ~90% with no traceback,
because a kill leaves none.

Generating in batches and merging keeps the peak to one batch:

    for i in 1 2 3 4; do
      SCRABLOZAUR_LANGUAGE=en python smart_player/generate_data.py 500000 \\
        --lookahead 4 --out smart_player/_en_part$i.npz
    done
    python smart_player/merge_datasets.py smart_player/_en_part*.npz \\
        --out smart_player/_en_leave_dataset.npz

The parts must come from the same language and the same `--lookahead`; nothing
here can check that, so keep them in one directory per run.
"""

import argparse
import os

import numpy as np

# The columns generate_data.py writes. Listed explicitly so a part missing one
# fails here rather than producing a dataset train.py reads as truncated.
COLUMNS = ("leaves", "unseen", "margins", "tw_open", "dw_open", "tl_open", "dl_open", "board_fill")


def merge(paths: list[str], out_path: str, quiet: bool = False) -> int:
    columns: dict[str, list[np.ndarray]] = {name: [] for name in COLUMNS}
    total = 0
    for path in paths:
        with np.load(path) as part:
            missing = [c for c in COLUMNS if c not in part.files]
            if missing:
                raise SystemExit(f"{path}: missing column(s) {missing}")
            n = len(part["leaves"])
            total += n
            if not quiet:
                print(f"  {os.path.basename(path)}: {n:,} samples")
            for name in COLUMNS:
                columns[name].append(part[name])

    merged = {}
    for name in COLUMNS:
        # Freed as we go: holding both the per-part list and the concatenated
        # result for every column at once is the peak this script exists to avoid.
        merged[name] = np.concatenate(columns[name])
        columns[name].clear()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, **merged)
    if not quiet:
        print(f"{total:,} samples from {len(paths)} part(s) -> {out_path}")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", help="npz files from generate_data.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    merge(args.parts, args.out, quiet=args.quiet)


if __name__ == "__main__":
    main()
