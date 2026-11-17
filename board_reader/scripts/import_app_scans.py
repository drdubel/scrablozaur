"""Bring the scans the app saved for training into board_reader's data.

Ticking "Zapisz do treningu OCR" before confirming a scan makes the app keep
the camera frame (scan_<ms>.jpg) and what the player confirmed it shows, with
where the grid finder put the grid's corners and the lens's focal length
(scan_<ms>.json), in its external files. This copies them into test/app/,
checking each, skipping any already there; export_scan_dataset.py then
exports them with the photos, and the next training run learns from real
camera frames.

    python board_reader/scripts/import_app_scans.py              # from the phone, over adb
    python board_reader/scripts/import_app_scans.py --from DIR   # from a folder already pulled
    python board_reader/scripts/import_app_scans.py --clear-phone

then

    python board_reader/scripts/export_scan_dataset.py
    python board_reader/scripts/train_cell_reader.py --dump _cell_reader_oof.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "board_reader" / "test" / "app"
PHONE_DIR = "/sdcard/Android/data/pl.scrablozaur/files/scans"
CELLS = 15 * 15


def adb_path() -> str:
    found = shutil.which("adb")
    if found:
        return found
    sdk = os.environ.get("ANDROID_HOME") or os.path.expanduser("~/Library/Android/sdk")
    return os.path.join(sdk, "platform-tools", "adb")


def problem(scan: dict, alphabet: str) -> str | None:
    """What is wrong with a scan's JSON, if anything."""
    try:
        letters, blanks = scan["letters"], scan["blanks"]
        corners, focal = scan["corners"], float(scan["focal_px"])
        width, height = int(scan["width"]), int(scan["height"])
    except (KeyError, TypeError, ValueError) as e:
        return f"incomplete ({e})"
    if scan.get("language") != "pl":
        return f"language {scan.get('language')!r}: the dataset is Polish"
    if len(letters) != CELLS or len(blanks) != CELLS:
        return "not 15x15"
    if any(ch and (len(ch) != 1 or ch not in alphabet) for ch in letters):
        return "a square that is not one of the alphabet's letters"
    if len(corners) != 8 or not all(isinstance(v, (int, float)) for v in corners):
        return "no grid corners"
    if not 0 < focal < 10 * max(width, height):
        return f"focal length {focal}"
    if sum(1 for ch in letters if ch) < 2:
        return "an empty board teaches nothing"
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", type=Path, help="a folder of scan_*.json/.jpg (default: the phone)")
    ap.add_argument("--clear-phone", action="store_true", help="delete the scans from the phone once copied")
    ap.add_argument("--serial", help="the phone's adb serial, if more than one is connected")
    args = ap.parse_args()

    sys.path.insert(0, str(ROOT / "src"))
    from languages import load

    alphabet = load("pl").alphabet.lower()
    adb = [adb_path()] + (["-s", args.serial] if args.serial else [])
    with tempfile.TemporaryDirectory() as tmp:
        source = args.source
        if source is None:
            pulled = subprocess.run([*adb, "pull", PHONE_DIR, tmp], capture_output=True, text=True)
            if pulled.returncode != 0:
                sys.exit(f"adb pull failed: {pulled.stderr.strip() or pulled.stdout.strip()}")
            source = Path(tmp) / "scans"
        names = sorted(p.stem for p in source.glob("scan_*.json"))
        APP_DIR.mkdir(parents=True, exist_ok=True)
        added, copied = 0, []
        for name in names:
            frame = source / f"{name}.jpg"
            if (APP_DIR / f"{name}.json").exists():
                copied.append(name)
                continue
            try:
                scan = json.loads((source / f"{name}.json").read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                print(f"{name}: unreadable ({e})")
                continue
            why = problem(scan, alphabet) or (None if frame.exists() else "no frame")
            if why:
                print(f"{name}: skipped, {why}")
                continue
            shutil.copy2(frame, APP_DIR / frame.name)
            shutil.copy2(source / f"{name}.json", APP_DIR / f"{name}.json")
            tiles = sum(1 for ch in scan["letters"] if ch)
            print(f"{name}: {tiles} tiles, {scan['width']}x{scan['height']}")
            added += 1
            copied.append(name)
        print(f"{added} new scans in {APP_DIR} ({len(names) - added} already there or skipped)")
        if args.clear_phone and args.source is None and copied:
            paths = [f"{PHONE_DIR}/{n}.{ext}" for n in copied for ext in ("jpg", "json")]
            subprocess.run([*adb, "shell", "rm", "-f", *paths], check=True)
            print(f"deleted {len(copied)} copied scans from the phone")
    if added:
        print("next: python board_reader/scripts/export_scan_dataset.py, then retrain")


if __name__ == "__main__":
    main()
