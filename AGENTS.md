# Scrablozaur — agent guide

Polish Scrabble engine: **Rust core** (DAWG + GADDAG + Rayon move search) exposed to
**Python** via PyO3, plus a FastAPI web app, a CV board scanner, and a learned
rack-leave evaluator.

Read `README.md` for engine internals, binary formats, and the Python API.
Sub-projects have their own READMEs: `board_reader/README.md`, `smart_player/README.md`.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | The engine: DAWG/GADDAG, `Board`, move search, scoring, CLI subcommands |
| `src/main.rs` | Thin binary entry point, delegates to `lib.rs` |
| `src/main.py`, `src/strategy.py`, `src/verify_engine.py` | Self-play benchmark, bot classes, sanity checks |
| `web/` | FastAPI app (`web.main:app`), routes under `web/routers/`, vanilla-JS frontend in `web/static/` |
| `board_reader/` | Photo → board-state OpenCV/torch pipeline |
| `smart_player/` | Leave-value model (`models/leave_value.pt`, exported to `.bin`) |
| `words/` | `words.txt` (2.58M words), pre-built `dawg.bin` (~3 MiB), `gaddag.bin` (~36 MiB) |
| `tests/cli_build.rs` | `cargo test` integration test for the CLI |
| `tests/test_strategy.py` | Player-logic tests |
| `test/*.in` | Sample board states for manual testing |
| `scrablozaur.pyi` | **Generated-by-hand** type stubs; installed as `scrablozaur/__init__.pyi` |

## Commands

```bash
# Python deps (uv-managed; pyproject.toml is NOT a real package — `package = false`)
uv sync

# Build/rebuild the Python extension — REQUIRED after any change to src/lib.rs
uv run maturin develop --release

# Rust
cargo build --release          # CLI binary (extension-module feature OFF by default)
cargo test                     # includes tests/cli_build.rs
cargo clippy --all-targets
cargo fmt

# Python tests / lint
uv run pytest tests/
uv run ruff check .            # line-length = 120 (board_reader/ruff.toml)
uv run mypy .

# Web app
uv run uvicorn web.main:app --reload

# Engine CLI (see README "CLI" section)
cargo run --release -- build         words/words.txt words/dawg.bin
cargo run --release -- build-gaddag  words/words.txt words/gaddag.bin
cargo run --release -- gen-verify    pl words/dawg.bin words/gaddag.bin 200   # GADDAG vs legacy parity
cargo run --release -- gen-bench     pl words/dawg.bin words/gaddag.bin 200   # speed comparison
```

## Conventions & gotchas

- **`maturin develop --release` after every `src/lib.rs` edit.** Python will silently
  keep importing the stale `.so` otherwise. Always use `--release`; debug builds of the
  move generator are unusably slow.
- `extension-module` is deliberately **off** in `Cargo.toml` so `cargo run`/`cargo test`
  can link libpython normally. Maturin turns it on via `[tool.maturin] features`.
- `scrablozaur.pyi` is maintained by hand — **update it whenever the PyO3 API changes**,
  or type checking of `web/`, `smart_player/`, and `src/*.py` goes stale.
- **Alphabet, point values and tile distribution live in `languages/<code>.json`** — never
  hardcode a letter table again. `src/languages.py` parses and validates it; the engine
  reads it too (`Language::from_file`). `tests/test_tables_agree.py` fails if any consumer
  drifts from it, and that test is the reason a hardcoded copy can be deleted safely.
- Every `Board`, `Dawg` and `LeaveNet` is bound to a `Language`, deliberately with no
  default — a silent fallback to Polish is the exact bug class this design prevents.
  CLI commands that read a `.bin` take a language code as their first argument.
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

`target/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, generated profile
artifacts (`profile*.json.gz`, `profile.svg`, `*.pstats`), and the committed binaries
`words/dawg.bin` / `words/gaddag.bin` (regenerate via the CLI instead of editing).
