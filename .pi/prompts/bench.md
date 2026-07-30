Benchmark and profile the engine, focusing on: {{focus}}

Follow this loop and report numbers, not impressions:

1. **Baseline.** Make sure the extension is current:
   `uv run maturin develop --release`
2. **Correctness gate first.** Run
   `cargo run --release -- gen-verify words/dawg.bin words/gaddag.bin 200`
   and confirm move-for-move parity with the legacy pattern search. If parity fails,
   stop and fix that before measuring anything.
3. **Measure.**
   - Rust move-gen: `cargo run --release -- gen-bench words/dawg.bin words/gaddag.bin 200`
   - DAWG lookups: `cargo run --release -- bench words/dawg.bin words/words.txt`
   - Python-side self-play: `uv run python -m src.main` (`benchmark()` in `src/main.py`)
   - For Python hot spots, cProfile into a `_`-prefixed scratch file so it stays git-ignored.
4. **Control the variables.** Pin threads with `RAYON_NUM_THREADS` (or `set_num_threads`)
   when comparing runs; the default pool is `min(8, cores)`. Run each configuration at
   least 3 times and report median, not best-of.
5. **Report** a before/after table: configuration, metric, median, spread, and the exact
   commands used. Call out any change that is within noise as "no measurable effect".

Do not commit profile artifacts (`profile*.json.gz`, `profile.svg`, `*.pstats`).
