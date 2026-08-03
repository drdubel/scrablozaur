# Scrablozaur

A high-performance multi-language Scrabble engine written in Rust, exposed to Python via [PyO3](https://pyo3.rs). Polish and English ship; a language is one JSON file (see [Languages](#languages)). Scrablozaur combines a minimized DAWG dictionary, a GADDAG move generator, and Rayon-parallel search to find the highest-scoring legal play from any board position in milliseconds.

---

## Features

- **Minimized DAWG** — multi-language (2.58M-word Polish, 168k-word English) compressed into a compact binary format; sub-microsecond lookups via a binary-searched, flat node layout
- **GADDAG move generation** — the primary generator: anchor-based bidirectional search over a GADDAG of the same lexicon, ~7× faster per position than the legacy pattern search it falls back to (measured by `gen-bench`, verified move-for-move by `gen-verify`)
- **Pattern search** — flexible wildcard syntax (`-` one letter, `*` any number) with blank-tile support
- **Board-aware scoring** — all bonus squares (Double/Triple Letter and Word), bingo bonus for using all 7 tiles
- **Cross-word validation** — every candidate placement is checked against all perpendicular words it creates
- **Rayon parallelism** — anchors are searched concurrently on a dedicated pool capped at `min(8, cores)` by default (a single move is too cheap to scale past that); override with `set_num_threads(n)` or `RAYON_NUM_THREADS`
- **O(1) letter lookup** — the letter bag is represented as a frequency array, eliminating linear scans during DAWG traversal
- **Python bindings** — clean PyO3 API with bundled `.pyi` stubs for full type-checker support

---

## Board

Standard 15 × 15 Scrabble layout, the same in every language. The first word must cover the centre square (⭐).

```
🟥⬜⬜🟩⬜⬜⬜🟥⬜⬜⬜🟩⬜⬜🟥
⬜🟧⬜⬜⬜🟦⬜⬜⬜🟦⬜⬜⬜🟧⬜
⬜⬜🟧⬜⬜⬜🟩⬜🟩⬜⬜⬜🟧⬜⬜
🟩⬜⬜🟧⬜⬜⬜🟩⬜⬜⬜🟧⬜⬜🟩
⬜⬜⬜⬜🟧⬜⬜⬜⬜⬜🟧⬜⬜⬜⬜
⬜🟦⬜⬜⬜🟦⬜⬜⬜🟦⬜⬜⬜🟦⬜
⬜⬜🟩⬜⬜⬜🟩⬜🟩⬜⬜⬜🟩⬜⬜
🟥⬜⬜🟩⬜⬜⬜⭐⬜⬜⬜🟩⬜⬜🟥
⬜⬜🟩⬜⬜⬜🟩⬜🟩⬜⬜⬜🟩⬜⬜
⬜🟦⬜⬜⬜🟦⬜⬜⬜🟦⬜⬜⬜🟦⬜
⬜⬜⬜⬜🟧⬜⬜⬜⬜⬜🟧⬜⬜⬜⬜
🟩⬜⬜🟧⬜⬜⬜🟩⬜⬜⬜🟧⬜⬜🟩
⬜⬜🟧⬜⬜⬜🟩⬜🟩⬜⬜⬜🟧⬜⬜
⬜🟧⬜⬜⬜🟦⬜⬜⬜🟦⬜⬜⬜🟧⬜
🟥⬜⬜🟩⬜⬜⬜🟥⬜⬜⬜🟩⬜⬜🟥
```

| Symbol | Bonus         | Effect                          |
|:------:|:--------------|:--------------------------------|
| 🟥     | Triple Word   | word score × 3                  |
| 🟧     | Double Word   | word score × 2                  |
| 🟦     | Triple Letter | that letter's score × 3         |
| 🟩     | Double Letter | that letter's score × 2         |
| ⭐     | Centre (DW)   | word score × 2, first move only |
| ⬜     | —             | plain square                    |

Multipliers apply only to tiles placed on that square during the current move; tiles already on the board always score at face value. (This exact layout is also encoded programmatically in `board_reader/src/premium_layout.py`, used to register the grid detector against a photographed board.)

---

## Tile Distribution

Each language declares its own bag in `languages/<code>.json`, and that file is
the single source of truth — the engine, the web app, the board renderer, the OCR
classifier and the leave model all read it.

Polish, shown below, is **100 tiles**: 98 lettered tiles across the 32 letters of
the alphabet, plus 2 blanks. English is also 100 tiles, over 26 letters, with a
different point scale (its Q and Z are worth 10).

| Letter | Count | Points | &nbsp; | Letter | Count | Points | &nbsp; | Letter | Count | Points |
|:------:|------:|-------:|--------|:------:|------:|-------:|--------|:------:|------:|-------:|
| A      |     9 |      1 |        | J      |     2 |      3 |        | S      |     4 |      1 |
| Ą      |     1 |      5 |        | K      |     3 |      2 |        | Ś      |     1 |      5 |
| B      |     2 |      3 |        | L      |     3 |      2 |        | T      |     3 |      2 |
| C      |     3 |      2 |        | Ł      |     2 |      3 |        | U      |     2 |      3 |
| Ć      |     1 |      6 |        | M      |     3 |      2 |        | W      |     4 |      1 |
| D      |     3 |      2 |        | N      |     5 |      1 |        | Y      |     4 |      2 |
| E      |     7 |      1 |        | Ń      |     1 |      7 |        | Z      |     5 |      1 |
| Ę      |     1 |      5 |        | O      |     6 |      1 |        | Ź      |     1 |      9 |
| F      |     1 |      5 |        | Ó      |     1 |      5 |        | Ż      |     1 |      5 |
| G      |     2 |      3 |        | P      |     3 |      2 |        |        |       |        |
| H      |     2 |      3 |        | R      |     4 |      1 |        |        |       |        |
| I      |     8 |      1 |        |        |       |        |        |        |       |        |

A blank tile (`?`) may substitute for any letter during a search but scores 0 points. `board.fresh_tile_bag()` returns whichever distribution that board's language declares.

---

## Requirements

| Dependency     | Version | Purpose                                          |
|:---------------|:--------|:--------------------------------------------------|
| Rust toolchain | ≥ 1.83  | Compiling the engine (pyo3 0.28's MSRV)          |
| Python         | ≥ 3.12  | Running game logic, the `web/` app, `board_reader/` (`.python-version` pins 3.14) |
| [uv](https://docs.astral.sh/uv/) | latest | Managing the Python environment (`.venv`) and dependencies |
| maturin        | ≥ 1.14  | Building the Python extension (a `uv` dev-dependency, see `pyproject.toml`) |
| rayon          | 1.12    | Parallel pattern evaluation (bundled)            |
| pyo3           | 0.28    | Python bindings (bundled)                        |

---

## Installation

### Build the Python extension

```bash
uv sync                            # installs Python deps (web/ + board_reader/) into .venv
uv run maturin develop --release   # compiles the Rust extension into that same .venv
```

`maturin develop` installs the extension straight into the active `.venv`
(`.venv/lib/python3.*/site-packages/scrablozaur/`, alongside the bundled
`__init__.pyi` stubs), so `import scrablozaur` works immediately. Re-run
`maturin develop --release` after any change to `src/lib.rs`.

`target/release/` is Cargo's own output directory — it holds the CLI binary
(`scrablozaur`) and the `libscrablozaur.dylib`/`.so` cdylib, neither of which
Python imports.

### Languages

A language is one file, `languages/<code>.json`, holding its alphabet (in
collation order), tile distribution, point values, vowel/consonant split, and
the paths to its dictionary and models. It is the **single source of truth**:
the engine, the web app, the board renderer, the OCR classifier and the leave
model all read it, and `tests/test_tables_agree.py` fails if any of them drifts.

Two ship: **Polish** (`pl`, 2.58M words) and **English** (`en`, ENABLE, 168k
words). The picker is on the new-game screen; each game stores its own language,
so two sessions can run different ones side by side.

To add a third: drop in `languages/xx.json`, put a word list at
`words/xx/words.txt`, run `make dicts LANG=xx`, and it appears in the picker —
no code change. Constraints are ≤32 letters (a cross-check set is a `u32`) and
every letter below U+0190 (letters are indexed by codepoint), which means Latin
script. `src/languages.py` validates all of it at load, naming the offending
letter.

Optional artifacts degrade rather than break. A language with no trained
leave-value net (`"leave_net": null`) caps at difficulty 8 and hides the
`smart`/`sim` suggestion orderings — a net is only valid for the tile alphabet
it was trained on, so borrowing another language's would silently produce
confident nonsense. A language with no OCR models (`"ocr": null`) simply has no
board scanner.

### Rebuild the dictionaries

Each language keeps its lexicon under `words/<code>/`, and both binaries are
pre-built and committed. To recompile them:

```bash
make dicts LANG=pl
```

or by hand:

```bash
cargo run --release -- build        words/pl/words.txt words/pl/dawg.bin
cargo run --release -- build-gaddag words/pl/words.txt words/pl/gaddag.bin
```

To check the pipeline quickly with a tiny input:

```bash
cargo run --release -- build words/en/smoke.txt /tmp/dawg.bin
```

`Board.get_best_words` uses the GADDAG for fast anchor-based move generation;
`Dawg(lang, "words/pl/dawg.bin")` auto-loads the sibling `gaddag.bin`. The
GADDAG is much larger than the DAWG (~36 MiB vs ~3 MiB for the full Polish
lexicon) and its build peaks around ~7 GB RAM. If no `gaddag.bin` is present,
move generation transparently falls back to the legacy DAWG pattern search.

**Rebuild both together.** Each file's header stamps the letters its lexicon
actually uses, and the engine refuses to load one whose letters the language
does not have — which is what stops a dictionary in the wrong language from
loading silently and mis-scoring every move.

Two commands validate and benchmark the generator against that fallback:

```bash
make verify LANG=pl   # best-move parity  (gen-verify)
make bench  LANG=pl   # single-thread speedup (gen-bench)
```

`gen-verify` must report zero score mismatches; `gen-bench` reports the
per-position generation time for each and their ratio — currently ~1.35 ms/position
for the legacy pattern search vs ~0.19 ms for GADDAG generation, a **7.0×**
single-threaded speedup. Both take an optional trailing game count (default 200).

---

## Python API

Every `Board`, `Dawg` and `LeaveNet` is bound to a `Language` — there is no
default, because silently falling back to one language's point table is exactly
the bug this prevents. Build one first; every snippet below reuses it.

```python
import sys
sys.path.insert(0, "src")            # `languages` is a script-style module
from languages import engine_language, load

spec = load("pl")                    # parsed + validated languages/pl.json
lang = engine_language(spec)         # the engine-side Language
```

### `Dawg`

```python
from scrablozaur import Dawg

d = Dawg(lang, "words/pl/dawg.bin")

# membership test
d.contains("hamulec")        # True
"hamulec" in d               # True  (same via __contains__)
d.contains("xyzzy")          # False

# pattern search (see Pattern Syntax below)
d.search("ha-ulec", "m")     # ['hamulec']
d.search("*", "aekrtu")      # all words buildable from these letters
d.search("k-t", "oar?")      # k + one letter + t, '?' = blank tile

# diagnostics
d.node_count()               # number of DAWG nodes after minimization (116734)
d.has_gaddag()               # True when the sibling gaddag.bin was auto-loaded
```

### `Board`

```python
from scrablozaur import Board, Dawg

d = Dawg(lang, "words/pl/dawg.bin")
b = Board(lang)                        # empty 15x15 board + full 100-tile bag
# Board.from_grid(lang, [["-"] * 15 for _ in range(15)])  # or start from a grid

# draw letters from the bag (fills hand up to 7 tiles)
hand = b.give_letters("")              # e.g. "aeimnrt"

# find the best move — the board tracks first-move state itself, so the opening
# move (which must cover the centre square) needs no special flag
word, score, (row, col, horizontal), used = b.get_best_word(d, hand)

# score and validate before committing — calculate_word_points must be called
# before place_word; after placement the tiles are no longer on empty squares
# and bonus multipliers no longer apply.
pts = b.calculate_word_points(word, row=row, col=col, horizontal=horizontal, letters=hand)
b.check_word_placement(d, word, row=row, col=col, horizontal=horizontal)  # raises on invalid

b.place_word(word, row, col, horizontal)   # this is what retires first-move eligibility
for ch in used:
    hand = hand.replace(ch, "", 1)
hand += b.give_letters(hand)

# subsequent moves — identical call
word, score, (row, col, horizontal), used = b.get_best_word(d, hand)

# top-N candidates instead of just the best
b.get_best_words(d, hand, 10)   # [(word, score, (row, col, horizontal), used), ...]

# inspect candidate patterns
b.get_all_patterns()     # list of (index, start, end, horizontal)
b.get_row_patterns(7)    # (start, end) column spans in row 7
b.get_col_patterns(4)    # (start, end) row spans in column 4

print(b)                 # pretty-print the board
```

`used` lists one entry per newly placed tile, with `'?'` wherever a blank had
to stand in for a letter the hand had run out of.

### Full two-player simulation

```python
from scrablozaur import Board, Dawg

d = Dawg(lang, "words/pl/dawg.bin")


class Player:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = board.give_letters("")
        self.score = 0

    def play(self) -> str:
        word, points, (row, col, horiz), used = self.board.get_best_word(d, self.letters)
        if not word:
            return ""
        self.score += points
        self.board.place_word(word, row, col, horiz)
        for ch in used:
            self.letters = self.letters.replace(ch, "", 1)
        self.letters += self.board.give_letters(self.letters)
        return word


b = Board(lang)
players = [Player(b), Player(b)]

# A game ends when both players fail to play in a row, or when someone goes
# out — not on the first single no-play, which would cut most games short.
no_play_streak = 0
turn = 0
while no_play_streak < len(players):
    p = players[turn % len(players)]
    no_play_streak = 0 if p.play() else no_play_streak + 1
    if not p.letters:          # went out: bag and rack are both empty
        break
    turn += 1

# Standard end-of-game rack adjustment: everyone loses their own rack's value,
# and a player who went out also gains every opponent's.
for p in players:
    penalty = b.rack_value(p.letters)
    p.score -= penalty
    if not p.letters:
        continue
    for other in players:
        if other is not p and not other.letters:
            other.score += penalty

print(f"Player 1: {players[0].score}  Player 2: {players[1].score}")
print(b)
```

Ready-made players wrap this loop with tile exchanging, and all of them end a
game by the same rules (`src/rules.py`):

| Player | Where | Picks its move by |
|---|---|---|
| `SimplePlayer` | `src/strategy.py` | highest score |
| `StrategicPlayer` | `src/strategy.py` | highest score (its leave term is constant, so it is greedy in practice) |
| `RankedPlayer` | `src/strategy.py` | a rank window below the best — `strategy.rank_window(level)`, the custom difficulty dial |
| `SmartPlayer` | `smart_player/player.py` | score plus a learned value for the rack it leaves behind |
| `SimPlayer` | `smart_player/sim_player.py` | Monte-Carlo simulation of what each candidate concedes |

`SmartPlayer` and `SimPlayer` also search the endgame exactly once the bag is
empty. See [`smart_player/README.md`](smart_player/README.md) for how they are
trained and what each piece is worth.

---

## Pattern Search Syntax

`Dawg.search(pattern, letters)` traverses the DAWG matching a positional pattern against the player's hand.

| Token   | Meaning                                             |
|:-------:|:----------------------------------------------------|
| `a`–`ż` | Fixed letter — must appear exactly at this position |
| `-`     | Exactly one letter consumed from the hand           |
| `*`     | Zero or more letters consumed from the hand         |
| `?`     | Blank tile in the hand — matches any letter, scores 0 |

```python
d.search("l--y",  "oadn")    # ['lady', 'lany', 'lody']
d.search("l*y",   "oadn")    # ['lady', 'landy', 'lany', 'lny', 'lody']
d.search("k-t",   "oar?")    # ['kat', 'ket', 'kit', 'kot', 'kąt'] — '?' fills 'e'/'i'
d.search("*",     "aeimnrt") # every word buildable from those 7 tiles
d.search("ham*",  "ulec")    # ['hamulce', 'hamulec']
```

Note that `*` is not a free-form suffix: like `-`, every letter it matches is
consumed from the hand. `d.search("ham*", "lec")` returns `[]`, because
completing `hamulec` also needs a `u` the hand does not hold.

The hand is treated as a **multiset**: `"aab"` allows `a` twice but `b` only once across all wildcard positions combined. Duplicates caused by blank-tile substitution are removed automatically.

---

## Scoring

### Letter point values

```
1 pt  — A  E  I  N  O  R  S  W  Z
2 pt  — C  D  K  L  M  P  T  Y
3 pt  — B  G  H  J  Ł  U
5 pt  — Ą  Ę  F  Ó  Ś  Ż
6 pt  — Ć
7 pt  — Ń
9 pt  — Ź
```

### Score calculation

1. Sum the point values of all **newly placed** tiles, applying Letter multipliers from the squares they land on.
2. Add the face values of all tiles **already on the board** that become part of the word.
3. Multiply the subtotal by all Word multipliers covered by newly placed tiles.
4. Repeat steps 1–3 for each perpendicular **cross-word** created by the move.
5. Add a **+50 bingo bonus** if all 7 tiles in the hand were used in a single move.

When a word crosses multiple bonus squares, all active Word multipliers stack multiplicatively. Bonus squares are neutralized once a tile has been placed on them.

---

## Architecture

### DAWG binary format

```
header:
  [8] magic "SCRBDWG2"
  [4] format version (2)
  [1] kind: 0 = DAWG, 1 = GADDAG
  [3] reserved (zero)
  [4] number of distinct letters N
  [4N] their Unicode codepoints, ascending
  [4] root node ID
  [4] total node count

per node:
  [1] is_terminal flag
  [4] number of children
  per child:
    [4] Unicode codepoint of the edge label (sorted ascending)
    [4] child node ID
```

The stamped letters are the ones the lexicon actually contains, computed from
the word list at build time rather than declared — so it cannot be mislabelled.
At load, every stamped letter must exist in the language being loaded: an
English dictionary opened as Polish fails on `q`, and a Polish one opened as
English fails on `ą`. The `kind` byte catches the other easy mistake, loading a
DAWG where a GADDAG is wanted or vice versa. Version 1 files carried none of
this and are refused; rebuild them with `make dicts`.

Child edges are stored sorted by codepoint, enabling binary search in O(log k) where k ≤ 32 (Polish: 23 Latin letters + 9 diacritics). The offset of each node is precomputed into a flat table at load time so every node access is a direct array index with no pointer chasing.

`gaddag.bin` uses the identical format and loader — it is simply a different
lexicon, one entry per (word, anchor) pair, with a `\0` separator edge between
each entry's reversed prefix and forward suffix (so k ≤ 33 there).

### Pattern matching

`match_pattern` is a recursive DAWG traversal. At each `-` slot it iterates over child edges and checks the player's letter frequency array in O(1):

```
freq[c as usize] > 0   →  use the tile
freq['?' as usize] > 0 →  use a blank as this letter
```

A `mandatory_slots` counter tracks how many `-` tokens remain in the unprocessed pattern, preventing `*` expansions from starving them. Results are deduplicated with `sort + dedup` to handle the case where the same word is reachable via both a regular tile and a blank.

### Move search

`get_best_words` (and its `n == 1` wrapper `get_best_word`) dispatches to one of
three routines depending on board state and what the `Dawg` carries:

1. **Opening move** (`best_opening_words`) — the board is empty, so there are no
   anchors to hook onto. Every word buildable from the hand is enumerated with
   `search("*", …)` and scored at every offset that still covers the centre
   square.

2. **GADDAG generation** (`gaddag_best_words`) — the normal path once a tile is
   on the board. Each anchor square is expanded bidirectionally through the
   GADDAG, with per-square cross-check bitsets (one `u32` bit per alphabet
   letter) pruning any letter that would form an illegal perpendicular word
   before the traversal ever reaches it. This enumerates *every* legal play, so
   the top `n` are the true top `n`.

3. **Legacy pattern search** (`best_words_from_patterns`) — the fallback when no
   `gaddag.bin` was loaded. Every row and column is scanned for contiguous spans
   containing both placed tiles and empty squares, each span is searched against
   the DAWG, cross-word-validated and scored, and the results are ranked. It
   keeps only the single best word per span, so for `n > 1` it can under-report a
   span's other high-scoring plays; the best move is identical either way.

Paths 2 and 3 both fan their per-anchor / per-span work out across a Rayon
work-stealing pool with no shared mutable state between threads, then reduce to
the global top `n`. `gen-verify` checks the two agree move-for-move.

### Bonus table

`calculate_word_points` uses a precomputed **8 × 8 static table** that exploits the board's four-fold reflective symmetry:

```rust
let (r2, c2) = (r.min(14 - r), c.min(14 - c));   // fold into quadrant
let (letter_mul, word_mul) = BONUS_TABLE[r2][c2];  // O(1) lookup
```

---

## CLI

The crate includes these diagnostic commands:

```bash
cargo run --release -- build        words/pl/words.txt words/pl/dawg.bin    # compile DAWG
cargo run --release -- build-gaddag words/pl/words.txt words/pl/gaddag.bin  # compile GADDAG (move gen)
cargo run --release -- lookup       pl words/pl/dawg.bin hamulec             # single lookup
cargo run --release -- bench        pl words/pl/dawg.bin words/pl/words.txt  # lookup throughput
cargo run --release -- gen-verify   pl words/pl/dawg.bin words/pl/gaddag.bin [games]  # GADDAG vs legacy parity
cargo run --release -- gen-bench    pl words/pl/dawg.bin words/pl/gaddag.bin [games]  # GADDAG vs legacy speed

# `build`/`build-gaddag` derive everything from the word list. The rest read a
# `.bin`, so they need the alphabet: they take a language code (a definition in
# `languages/<code>.json`) as their first argument.
```

Sample `bench` output (measured against the current `words.txt`/`dawg.bin`):

```
Results (5 × 2584337 = 12921685 lookups):
  total time  : 401.829ms
  throughput  : 32157174 lookups/s
  per lookup  : 31.1 ns
  hits        : 12921685/12921685 (100.0%)
```

---

## Web App & Board Scanner

This engine also powers a FastAPI web app at `web/` (vanilla-JS/HTML frontend
under `web/static/`, Polish UI). All routers are mounted under `/api`:

| Router | Prefix | What it serves |
|:---|:---|:---|
| `routers/game.py` | `/api/game` | session lifecycle — new game, player types, state |
| `routers/board.py` | `/api/board` | moves: human/computer plays, exchange, skip, pass, undo, hints |
| `routers/scan.py` | `/api/scan` | "scan board" assistant — photo → board state → suggested move |
| `routers/benchmark.py` | `/api/benchmark` | run bot-vs-bot batches from the browser and chart the results |

The scan assistant is backed by the computer-vision pipeline in
[`board_reader/`](board_reader/README.md); the playable board and its bots are
backed by `src/strategy.py` and [`smart_player/`](smart_player/README.md) — the
web bot calls the same decision functions the CLI players do, and
`tests/test_web_agrees_with_cli.py` asserts the two never diverge.

### Difficulty

There are no named tiers. Difficulty is one custom level, an integer `1..10`,
set with a slider in the setup dialog (and per bot in automatic sandbox):

| Level | How the move is chosen |
|:---|:---|
| 1–8 | `strategy.rank_window(level)` — pick uniformly from that window of the best-first candidate list; the window shrinks geometrically from `(20, 40)` to `(1, 1)` |
| 9 | `smart_player.player.choose_move` — score plus the learned value of the rack left behind |
| 10 | `smart_player.sim_player.choose_move_sim` — Monte-Carlo rollouts (~110 ms/move) |

`src/strategy.py` owns the level → rank-window maths so `RankedPlayer`, the web
bot and the benchmark all weaken identically; `web/difficulty.py` owns the level
→ engine mapping plus the descriptions served by
`GET /api/game/difficulty-levels`, which is what the slider shows as feedback
(what the bot will do, and what to expect from it) — so the text can't drift
from the windows the bot actually plays by.

An 8-game 4-bot benchmark (`/api/benchmark`) at levels 1/3/6/8 averaged
82 / 128 / 197 / 229 points: the dial is monotone in real play, not just on
paper.

Benchmark games run in a process pool. Each worker re-imports the app (torch)
and loads its own DAWG + GADDAG, i.e. ~300 MB RSS, so the pool is deliberately
small and is shut down once it has been idle for a while instead of sitting on
the memory for the lifetime of the server:

| Env var | Default | What |
|:---|:---|:---|
| `SCRABLOZAUR_BENCH_WORKERS` | `min(4, cores)` | benchmark worker processes (~300 MB each) |
| `SCRABLOZAUR_BENCH_THREADS` | `min(4, cores/workers)` | rayon threads inside each worker |
| `SCRABLOZAUR_BENCH_POOL_IDLE` | `120` | seconds of idleness before the pool is reaped (`0` = immediately) |

```bash
uv sync
uv run maturin develop --release
uv run uvicorn web.main:app --reload
```

## Project Structure

```
scrablozaur/
├── src/
│   ├── lib.rs           # Rust engine: DAWG/GADDAG, Board, move search, scoring
│   ├── main.rs          # thin binary entry point (delegates to lib.rs's CLI)
│   ├── main.py          # Python self-play benchmark script (`graj()`, `benchmark()`)
│   ├── strategy.py      # SimplePlayer / StrategicPlayer / RankedPlayer bot classes
│   ├── rules.py         # when a game ends and how it is finally scored (one definition)
│   ├── languages.py     # loads + validates languages/*.json (the source of truth)
│   └── verify_engine.py # engine sanity checks
├── languages/           # one JSON per language: alphabet, points, distribution, paths
│   ├── pl.json
│   └── en.json
├── web/                 # FastAPI web app (game UI + board scanner + benchmark UI)
│   ├── main.py          # app entry point (`uvicorn web.main:app`)
│   ├── game.py, scan.py, engine.py, models.py
│   ├── difficulty.py    # custom difficulty level 1-10: engine per level + UI descriptions
│   ├── routers/         # game/board/scan/benchmark API routes (mounted under /api)
│   └── static/          # HTML/CSS/vanilla-JS frontend
├── board_reader/        # photo -> board-state CV pipeline (see its own README)
├── smart_player/        # learned rack-leave evaluator / SmartPlayer (see its own README)
├── words/               # one directory per language, all pre-built and committed
│   ├── pl/              # words.txt (2.58 M), dawg.bin ~3 MiB, gaddag.bin ~36 MiB
│   └── en/              # words.txt (168 k, ENABLE), dawg.bin ~1 MiB, gaddag.bin ~8 MiB
├── test/                # sample board states (.in files) for manual testing
├── tests/
│   ├── cli_build.rs     # `cargo test` integration test for the CLI
│   ├── test_strategy.py # player-logic tests
│   └── test_web_agrees_with_cli.py  # the web app and the CLI must choose the same move
├── scrablozaur.pyi      # Python type stubs (installed as scrablozaur/__init__.pyi)
├── Makefile             # dictionary builds + verification, per language
├── pyproject.toml       # uv-managed Python dependencies (web/ + board_reader/ + smart_player/)
├── uv.lock
└── Cargo.toml           # Rust package manifest
```
