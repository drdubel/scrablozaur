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
| `src/game/ladder.rs` | The bots' Elo, measured by `scrablozaur bot-ladder` and kept as a table (`LADDERS`); the app rates players against it |
| `tests/web_parity.py` | Records the web app's behaviour into `tests/fixtures/web_parity.json`, which `cargo test` replays against `src/game/` and `src/scan.rs` |
| `src/main.py`, `src/strategy.py`, `src/verify_engine.py` | Self-play benchmark, bot classes, sanity checks |
| `web/` | FastAPI app (`web.main:app`), routes under `web/routers/`, vanilla-JS frontend in `web/static/`. `web/leaderboard.py` is the shared ranking, off unless `SCRABLOZAUR_LEADERBOARD` names a file |
| `.github/workflows/ci.yml` | CI: the engine, the web app and bindings, the Android app (see README "Continuous integration") |
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
# Debug builds: the scanner fed recorded frames (frame-bench's SCRBFRM1) instead of the camera,
# one held still per scan -- tests everything past the lens with no board in front of the phone
adb push frames.bin /sdcard/Android/data/pl.scrablozaur/files/
adb shell am start -n pl.scrablozaur/.ui.MainActivity --es scan_replay /sdcard/Android/data/pl.scrablozaur/files/frames.bin
# Scans the app saved for training ("Zapisz do treningu OCR") into board_reader's data, then re-export and retrain
uv run python board_reader/scripts/import_app_scans.py && uv run python board_reader/scripts/export_scan_dataset.py

# Engine CLI (see README "CLI" section)
make dicts  LANG=pl            # rebuild both binaries (build + build-gaddag)
make verify LANG=pl            # GADDAG vs legacy parity  (gen-verify)
make bench  LANG=pl            # speed comparison         (gen-bench)
cargo run --release -- bot-ladder pl words/pl/dawg.bin words/pl/gaddag.bin \
    smart_player/models/pl/leave_value.bin 200   # the bots' Elo, for src/game/ladder.rs
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
- **A game's turns are logged by the engine** (`TurnRecord`, `GameView.turns`), so undo and
  saves keep them right. The app's history (`files/history.jsonl`), statistics, ranking and
  game review all read that log, and the in-game rating card does too: never keep a second
  move list app-side. The web keeps the same log (`TurnRecord` in `web/game.py`), so the
  parity fixture checks it, reviews included -- equity to within 0.1, since the web values
  leaves with PyTorch and the port with its own copy of the net.
- `GameMode::Multiplayer` (2-4 seats passing one phone) is an extension the web does not
  have, as is everything that reads its `is_match()`; the parity fixture never plays one.
  It shares competitive play's rules except that a resigned seat leaves and the rest play on.
- **Re-measure `LADDERS` (`src/game/ladder.rs`) after retraining a leave net or rebuilding a
  dictionary**: `scrablozaur bot-ladder <lang> ... 200` prints the entry (30-40 min a language
  on 18 cores). Players' Elo in the app is only as right as that table.
- The Android UI is Polish in `values/` (`tools:locale="pl"`) and English in `values-en/`,
  which must carry every resource with the same format arguments: a missing one is a lint
  error. Word new strings in both.
- The app's engine data lives in `noBackupFilesDir`, out of Auto Backup's 25 MB; what is
  backed up is listed in `res/xml/backup_rules.xml` and `data_extraction_rules.xml`. A new
  file in `filesDir` that should survive a phone change goes into both.
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
