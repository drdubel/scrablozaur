---
name: engine-change
description: Checklist for changing the Rust engine in src/lib.rs (DAWG, GADDAG, move generation, scoring) and keeping the PyO3 bindings, .pyi stubs, and Python callers in sync. Use whenever editing src/lib.rs, changing the PyO3 API surface, or when Python behaviour disagrees with the Rust code.
---

# Changing the Scrablozaur engine

The Rust engine is consumed by Python through a compiled PyO3 extension. Editing
`src/lib.rs` alone changes nothing that Python sees until the extension is rebuilt.
Most "impossible" bugs in this repo are a stale `.so`.

## 1. Rebuild, always

```bash
uv run maturin develop --release
```

- Run this after **every** `src/lib.rs` edit, before any Python test or benchmark.
- Always `--release`. A debug build of the move generator is one to two orders of
  magnitude slower and will make any timing look catastrophic.
- `maturin develop` installs into the active `.venv`; if imports still look stale,
  confirm you are in that venv (`uv run python -c "import scrablozaur; print(scrablozaur.__file__)"`).

`extension-module` is deliberately **off** in `Cargo.toml` so `cargo run` / `cargo test`
can link libpython normally for the CLI binary. Maturin re-enables it via
`[tool.maturin] features`. Don't "fix" this by flipping the default.

## 2. Keep the API surface in sync

If you add, remove, or change the signature of anything exposed with `#[pyfunction]`,
`#[pymethods]`, or `#[pyclass]`:

1. Update `scrablozaur.pyi` by hand — it is not generated. It ships as
   `scrablozaur/__init__.pyi`, and it is the only thing giving `web/`, `smart_player/`,
   and `src/*.py` type information.
2. Grep for callers before changing a signature:
   ```bash
   rg -n 'scrablozaur\.|from scrablozaur' --glob '*.py'
   ```
   Main consumers: `src/main.py`, `src/strategy.py`, `src/verify_engine.py`,
   `web/engine.py`, `web/game.py`, `smart_player/`.
3. Re-check types: `uv run mypy .`

## 3. Verify correctness before performance

Move generation has a legacy pattern-search implementation kept specifically as an
oracle. Any change to the GADDAG generator must stay move-for-move identical:

```bash
make verify LANG=pl && make verify LANG=en
```

Both languages: the alphabet, point table and tile distribution are runtime data
now, so a change can be correct for one and wrong for the other.

Also run:

```bash
cargo test                       # includes tests/cli_build.rs
uv run pytest tests/             # player-logic tests
uv run python -m src.verify_engine
```

If you changed scoring, cross-word validation, blank handling, or the bag, exercise the
sample positions in `test/*.in` too — those are the fixtures for hand-checking boards.

## 4. Only then measure

```bash
make bench LANG=pl
cargo run --release -- bench pl words/pl/dawg.bin words/pl/words.txt
```

Pin `RAYON_NUM_THREADS` when comparing runs; the pool defaults to `min(8, cores)`
because a single move is too cheap to scale past that. Report medians of at least 3
runs. The `/bench` prompt template in `.pi/prompts/bench.md` spells out the full loop.

## 5. Dictionary binaries

`words/<code>/dawg.bin` and `words/<code>/gaddag.bin` are committed build artifacts,
one pair per language in `languages/`. Never edit them; regenerate if the binary
format or a lexicon changes:

```bash
make dicts LANG=pl
make dicts LANG=en
```

Each file's header stamps the letters its lexicon actually uses, and the engine
refuses to load one whose letters the language does not have — so a rebuild that
misses a language fails loudly rather than mis-scoring.

Changing the on-disk format means both files must be rebuilt together, and
`gen-verify` must pass afterwards.
