# Scrablozaur — agent guide

Multi-language Scrabble engine (Polish + English): **Rust core** (DAWG + GADDAG + Rayon move search) exposed to
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
make dicts  LANG=pl            # rebuild both binaries (build + build-gaddag)
make verify LANG=pl            # GADDAG vs legacy parity  (gen-verify)
make bench  LANG=pl            # speed comparison         (gen-bench)
#   LANG is a language code with a definition in languages/<code>.json (pl, en)
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
- **Per-language artifacts** live under a `<code>/` directory: `words/<code>/`,
  `smart_player/models/<code>/`, `board_reader/src/models/<code>/`. Each is declared in
  `languages/<code>.json`; nullable entries mean "this language does not have one yet",
  which degrades a feature rather than breaking it (no leave net → difficulty caps at 8
  and the `smart`/`sim` orderings are refused; no OCR models → no scan tab).
- **Never point a language at another's model to fill a gap.** A net or classifier
  trained on a different alphabet does not fail, it returns confident nonsense. The
  loaders refuse a mismatch (`get_model`, `_check_classes`, `LeaveNet::load`) — that is
  a guard to keep, not an obstacle to work around.
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

`target/`, `.venv/`, `__pycache__/`, `.mypy_cache/`, `.ruff_cache/`, generated profile
artifacts (`profile*.json.gz`, `profile.svg`, `*.pstats`), and the committed binaries
`words/<code>/dawg.bin` / `words/<code>/gaddag.bin` (regenerate with `make dicts` instead
of editing).
