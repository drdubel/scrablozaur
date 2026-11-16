"""Time the bots on a real phone, to size the engine's thread pool for it.

Cross-compiles the `scrablozaur` CLI for the device with the same `mobile`
profile the app's library uses, pushes it together with one language's data,
and runs `bot-bench` -- automatic games through the same `GameSession` the app
plays, every move timed -- for each level and thread count asked for.

    python3 android/tools/device_bench.py                    # pl, levels 9-10, 1/2/4/8 threads
    python3 android/tools/device_bench.py --levels 10 --threads 2,4,6,8 --games 2
    python3 android/tools/device_bench.py --serial emulator-5584

The NDK, the rustup toolchain and the data paths are read from the Android
project and `languages/<code>.json`, so this never disagrees with the app build.
A phone heats up over a long run, which slows later configurations: compare
runs made in one session, and use --cooldown on long ones.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANDROID = ROOT / "android"
DEVICE_DIR = "/data/local/tmp/scrablozaur"
TRIPLE = "aarch64-linux-android"


def properties(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def build(min_sdk: int) -> Path:
    """The CLI for the device, built as the app's library is."""
    gradle = properties(ANDROID / "gradle.properties")
    local = properties(ANDROID / "local.properties")
    build_script = (ANDROID / "app" / "build.gradle.kts").read_text(encoding="utf-8")
    ndk_version = re.search(r'ndkVersion\s*=\s*"([^"]+)"', build_script).group(1)
    host = "darwin-x86_64" if sys.platform == "darwin" else "linux-x86_64"
    bin_dir = (
        Path(local["sdk.dir"])
        / "ndk"
        / ndk_version
        / "toolchains"
        / "llvm"
        / "prebuilt"
        / host
        / "bin"
    )
    clang = bin_dir / f"{TRIPLE}{min_sdk}-clang"
    key = TRIPLE.upper().replace("-", "_")
    env = dict(os.environ)
    env.update(
        {
            # Homebrew's rustc has no Android std; rustup's toolchain does.
            "PATH": gradle.get("scrablozaur.rustupBin", str(Path.home() / ".cargo/bin"))
            + os.pathsep
            + env["PATH"],
            "RUSTUP_TOOLCHAIN": gradle.get("scrablozaur.rustToolchain", "stable"),
            f"CARGO_TARGET_{key}_LINKER": str(clang),
            f"CC_{TRIPLE.replace('-', '_')}": str(clang),
            f"AR_{TRIPLE.replace('-', '_')}": str(bin_dir / "llvm-ar"),
            f"CARGO_TARGET_{key}_RUSTFLAGS": "-C link-arg=-Wl,-z,max-page-size=16384",
        }
    )
    subprocess.run(
        [
            "cargo",
            "build",
            "-p",
            "scrablozaur-core",
            "--bin",
            "scrablozaur",
            "--profile",
            "mobile",
            "--target",
            TRIPLE,
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return ROOT / "target" / TRIPLE / "mobile" / "scrablozaur"


def data_files(code: str) -> tuple[list[str], str, str, str | None]:
    """Repo-relative paths of what bot-bench reads, from the language definition."""
    spec = json.loads((ROOT / "languages" / f"{code}.json").read_text(encoding="utf-8"))
    net = (spec.get("leave_net") or {}).get("weights")
    files = [f"languages/{code}.json", spec["dawg"], spec["gaddag"]] + (
        [net] if net else []
    )
    return files, spec["dawg"], spec["gaddag"], net


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--serial", help="adb device (default: the only one connected)")
    ap.add_argument("--lang", default="pl")
    ap.add_argument("--levels", default="9,10")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--games", type=int, default=1, help="games per configuration")
    ap.add_argument(
        "--cooldown",
        type=float,
        default=0,
        help="seconds to rest between configurations",
    )
    ap.add_argument("--min-sdk", type=int, default=26)
    args = ap.parse_args()

    adb = ["adb"] + (["-s", args.serial] if args.serial else [])
    model = subprocess.run(
        adb + ["shell", "getprop", "ro.product.model"],
        capture_output=True,
        text=True,
        check=True,
    )
    print(f"device: {model.stdout.strip()}")

    binary = build(args.min_sdk)
    files, dawg, gaddag, net = data_files(args.lang)
    subprocess.run(adb + ["shell", "mkdir", "-p", DEVICE_DIR], check=True)
    for rel in files:
        subprocess.run(
            adb + ["shell", "mkdir", "-p", f"{DEVICE_DIR}/{Path(rel).parent}"],
            check=True,
        )
        subprocess.run(
            adb + ["push", "--sync", str(ROOT / rel), f"{DEVICE_DIR}/{rel}"],
            check=True,
            capture_output=True,
        )
    subprocess.run(
        adb + ["push", str(binary), f"{DEVICE_DIR}/scrablozaur"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        adb + ["shell", "chmod", "755", f"{DEVICE_DIR}/scrablozaur"], check=True
    )

    rows = []
    for level in [int(x) for x in args.levels.split(",")]:
        for threads in [int(x) for x in args.threads.split(",")]:
            command = (
                f"cd {DEVICE_DIR} && RAYON_NUM_THREADS={threads} ./scrablozaur bot-bench "
                f"{args.lang} {dawg} {gaddag} {net or '-'} {level} {args.games}"
            )
            out = subprocess.run(
                adb + ["shell", command], capture_output=True, text=True, check=True
            ).stdout
            print(out.strip())
            stats = {}
            for label, key in (("bag not empty", "mid"), ("bag empty", "end")):
                m = re.search(
                    rf"{label}\s*:\s*(\d+) moves \| mean\s+([\d.]+) ms \| p50\s+([\d.]+) \| p90\s+([\d.]+) \| max\s+([\d.]+)",
                    out,
                )
                if m:
                    stats[key] = [float(v) for v in m.groups()[1:]]
            rows.append((level, threads, stats))
            if args.cooldown:
                time.sleep(args.cooldown)

    print(
        "\nlevel threads | bag not empty: mean   p90    max  | bag empty: mean    max   (ms)"
    )
    for level, threads, stats in rows:
        mid = stats.get("mid", [float("nan")] * 4)
        end = stats.get("end", [float("nan")] * 4)
        print(
            f"{level:5} {threads:7} | {mid[0]:19.1f} {mid[2]:6.1f} {mid[3]:6.1f} | {end[0]:15.1f} {end[3]:6.1f}"
        )


if __name__ == "__main__":
    main()
