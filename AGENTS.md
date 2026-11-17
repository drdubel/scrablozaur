# Scrablozaur — agent guide

Multi-language Scrabble engine (Polish + English): **Rust core** (DAWG + GADDAG + Rayon move search) exposed to
**Python** via PyO3, plus a FastAPI web app, a CV board scanner, and a learned
rack-leave evaluator.

Read `README.md` for engine internals, binary formats, and the Python API.
Sub-projects have their own READMEs: `board_reader/README.md`, `smart_player/README.md`.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | The engine (`scrablozaur-core`): DAWG/GADDAG, `Board`, move search, scoring, CLI subcommands. Plain Rust, no bindings |
| `src/main.rs` | Thin binary entry point, delegates to `lib.rs` |
| `crates/py/` | The `scrablozaur` Python module: a thin PyO3 wrapper over the engine |
| `crates/mobile/` | The Android bindings: a thin UniFFI wrapper over the engine and `src/game/` (Kotlin package `pl.scrablozaur.engine`) |
| `android/` | The sideloaded Android app (Kotlin + Compose). Gradle cross-compiles the engine, generates the bindings and packs the dictionaries from this repo |
| `src/game/` | Game layer for the apps: session state machine, rules, bots (levels 1–10), hints, move rating, the scan assistant's suggestions, benchmark — a Rust port of the web's Python game logic |
| `src/scan.rs` | The scanner's dictionary pass (correction + flagging), ported from `web/scan.py` |
| `src/scan/reader.rs` | The learned cell reader (a small CNN the engine runs itself, `scrablozaur.CellReader`): a straightened board → per-square readings, and which way up the board is (each letter is four classes, one per quarter turn). Trained by `board_reader/scripts/train_cell_reader.py`; `tests/test_cell_reader.py` checks it against PyTorch |
| `src/scan/grid.rs`, `src/scan/geometry.rs`, `src/scan/frame.rs` | The rest of the live scanner, in the engine: the learned grid finder (`scrablozaur.GridFinder`, trained by `board_reader/scripts/train_grid_finder.py`), straightening the tiles' tops from the camera's pose, and the whole frame → board pipeline (`scrablozaur.scan_frame`). `src/scan/live.rs` is the app camera's rule for when to read a frame and when a scan is done. `src/scan/net.rs` is the small-CNN runtime both nets run on |
| `src/scan/decode.rs` | The legal-board decoder: the most probable board that obeys the rules, given per-square readings (`scrablozaur.decode_scan`). `board_reader/scripts/eval_decoder.py` measures it — tune against that |
| `tests/web_parity.py` | Records the web app's behaviour into `tests/fixtures/web_parity.json`, which `cargo test` replays against `src/game/` and `src/scan.rs` |
| `src/main.py`, `src/strategy.py`, `src/verify_engine.py` | Self-play benchmark, bot classes, sanity checks |
| `web/` | FastAPI app (`web.main:app`), routes under `web/routers/`, vanilla-JS frontend in `web/static/` |
| `board_reader/` | Photo → board-state OpenCV/torch pipeline |
| `smart_player/` | Leave-value model, per language (`models/<code>/leave_value.pt`, exported to `.bin`) |
| `words/<code>/` | Per language: `words.txt`, pre-built `dawg.bin` + `gaddag.bin` (pl ~3/36 MiB, en ~1/8 MiB) |
| `languages/` | One JSON per language — alphabet, points, distribution, artifact paths. **The source of truth.** |
| `tests/cli_build.rs` | `cargo test` integration test for the CLI |
| `tests/test_strategy.py` | Player-logic tests |
| `test/*.in` | Sample board states for manual testing |
| `scrablozaur.pyi` | **Generated-by-hand** type stubs; installed as `scrablozaur/__init__.pyi` |

## Commands

```bash
# Python deps (uv-managed; pyproject.toml is NOT a real package — `package = false`)
uv sync

# Build/rebuild the Python extension — REQUIRED after any change to src/lib.rs or crates/py
uv run maturin develop --release

# Rust (a workspace: the engine at the root is the default member)
cargo build --release          # engine + CLI binary
cargo test                     # engine tests, includes tests/cli_build.rs
cargo clippy --workspace --all-targets
cargo fmt --all

# Python tests / lint
uv run pytest tests/
uv run ruff check .            # line-length = 120 (board_reader/ruff.toml)
uv run mypy .

# Web app
uv run uvicorn web.main:app --reload

# Android app (JDK 17, Android SDK 37 + NDK 28.2, rustup toolchain with the android targets)
cd android && ./gradlew :app:assembleDebug        # → app/build/outputs/apk/debug/app-debug.apk
cd android && ./gradlew :app:testDebugUnitTest
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
python3 android/tools/device_bench.py --levels 9,10 --threads 2,4,8   # bot ms/move on a connected phone

# Engine CLI (see README "CLI" section)
make dicts  LANG=pl            # rebuild both binaries (build + build-gaddag)
make verify LANG=pl            # GADDAG vs legacy parity  (gen-verify)
make bench  LANG=pl            # speed comparison         (gen-bench)
#   LANG is a language code with a definition in languages/<code>.json (pl, en)
```

## Conventions & gotchas

- **`maturin develop --release` after every `src/lib.rs` edit.** Python will silently
  keep importing the stale `.so` otherwise. Always use `--release`; debug builds of the
  move generator are unusably slow.
- The engine has **no binding code**. Python's module is `crates/py` (built by maturin
  via `manifest-path`); it only converts arguments and maps `EngineError` onto
  `ValueError`/`OSError`/`RuntimeError`. Put behaviour in the engine, never in a wrapper,
  or the bindings drift apart.
- `extension-module` is deliberately **off** in `crates/py/Cargo.toml` so
  `cargo build --workspace` can link libpython normally. Maturin turns it on via
  `[tool.maturin] features`.
- `scrablozaur.pyi` is maintained by hand — **update it whenever the PyO3 API changes**,
  or type checking of `web/`, `smart_player/`, and `src/*.py` goes stale.
- **The web app's Python game logic is the specification for `src/game/`.** Change
  `web/game.py`, a router, `src/rules.py`, `src/strategy.py`'s windows or `smart_player`'s
  `choose_move`, and `tests/test_web_parity.py` fails until you regenerate the fixture
  (`uv run python tests/web_parity.py`) — after which `cargo test` fails until the Rust
  port matches again. Deliberate deviations (rule fixes) are marked "Deviation" in
  `src/game/session.rs`; the fixture only exercises behaviour both sides share.
- **Alphabet, point values and tile distribution live in `languages/<code>.json`** — never
  hardcode a letter table again. `src/languages.py` parses and validates it; the engine
  reads it too (`Language::from_file`). `tests/test_tables_agree.py` fails if any consumer
  drifts from it, and that test is the reason a hardcoded copy can be deleted safely.
- Every `Board`, `Dawg` and `LeaveNet` is bound to a `Language`, deliberately with no
  default — a silent fallback to Polish is the exact bug class this design prevents.
  CLI commands that read a `.bin` take a language code as their first argument.
- **Per-language artifacts** live under a `<code>/` directory: `words/<code>/`,
  `smart_player/models/<code>/`, `board_reader/src/models/<code>/`. Each is declared in
  `languages/<code>.json`; nullable entries mean "this language does not have one yet",
  which degrades a feature rather than breaking it (no leave net → difficulty caps at 8
  and the `smart`/`sim` orderings are refused; no OCR models → no scan tab).
- **Never point a language at another's model to fill a gap.** A net or classifier
  trained on a different alphabet does not fail, it returns confident nonsense. The
  loaders refuse a mismatch (`get_model`, `_check_classes`, `LeaveNet::load`) — that is
  a guard to keep, not an obstacle to work around.
- **The Android app plays through `crates/mobile`, which only converts types.** Rules,
  bots and wording-free refusals live in `src/game/`; the app words them (Polish strings in
  `android/app/src/main/res/values/strings.xml`, matching the web's). UniFFI error fields
  are never named `message` — they become Kotlin exceptions and collide with
  `Throwable.message`.
- The APK's engine data is staged from this repo by `android/app/build.gradle.kts`
  (`languages/`, `words/<code>/`, `smart_player/models/<code>/`, with a SHA-256 manifest).
  Never copy a dictionary into `android/` by hand.
- Gradle builds the Android library with rustup's toolchain (`scrablozaur.rustToolchain` /
  `scrablozaur.rustupBin` in `android/gradle.properties`): Homebrew's `rustc` has no Android
  standard library. `android/local.properties` (SDK path) and `android/keystore.properties`
  (release signing) are machine-local and git-ignored; a sideloaded update must be signed
  with the same key as the installed app.
- `smart_player`'s pipeline picks its language from the `SCRABLOZAUR_LANGUAGE` env var,
  not a flag, because its process-pool workers re-import `model.py`.
- Rayon pool is capped at `min(8, cores)`; override with `set_num_threads(n)` or
  `RAYON_NUM_THREADS`. Don't "fix" the cap — a single move is too cheap to scale past it.
- Correctness bar for move-gen changes: `cargo run --release -- gen-verify` must show
  move-for-move parity with the legacy pattern search. Run it before claiming a win.
- Performance claims need `gen-bench` numbers, not intuition.
- `.gitignore` is allowlist-style (`*` then `!*.rs` etc.). New file types are ignored by
  default — check `git status` and add a `!` rule if a new extension should be tracked.
- Files/dirs prefixed with `_` are scratch and git-ignored (`_best_game.txt`,
  `smart_player/_leave_dataset.npz`).

## Do not touch

`target/`, `android/**/build/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, generated profile
artifacts (`profile*.json.gz`, `profile.svg`, `*.pstats`), and the committed binaries
`words/<code>/dawg.bin` / `words/<code>/gaddag.bin` (regenerate with `make dicts` instead
of editing).
