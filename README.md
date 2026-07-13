# Scrablozaur

A high-performance Polish-language Scrabble engine written in Rust, exposed to Python via [PyO3](https://pyo3.rs). Scrablozaur combines a minimized DAWG dictionary, a board-aware pattern matcher, and Rayon-parallel move search to find the highest-scoring legal play from any board position in milliseconds.

---

## Features

- **Minimized DAWG** — 3.17 million-word Polish dictionary compressed into a compact binary format; sub-microsecond lookups via a binary-searched, flat node layout
- **Pattern search** — flexible wildcard syntax (`-` one letter, `*` any number) with blank-tile support
- **Board-aware scoring** — all bonus squares (Double/Triple Letter and Word), bingo bonus for using all 7 tiles
- **Cross-word validation** — every candidate placement is checked against all perpendicular words it creates
- **Rayon parallelism** — board patterns are evaluated concurrently across all logical CPU cores
- **O(1) letter lookup** — the letter bag is represented as a frequency array, eliminating linear scans during DAWG traversal
- **Python bindings** — clean PyO3 API with bundled `.pyi` stubs for full type-checker support

---

## Board

Standard 15 × 15 Polish Scrabble layout. The first word must cover the centre square (★).

```
      0    1    2    3    4    5    6    7    8    9   10   11   12   13   14
 0  [ TW]  ·    ·  [ DL]  ·    ·    ·  [ TW]  ·    ·    ·  [ DL]  ·    ·  [ TW]
 1    ·  [ DW]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ DW]  ·
 2    ·    ·  [ DW]  ·    ·    ·  [ DL]  ·  [ DL]  ·    ·    ·  [ DW]  ·    ·
 3  [ DL]  ·    ·  [ DW]  ·    ·    ·  [ DL]  ·    ·    ·  [ DW]  ·    ·  [ DL]
 4    ·    ·    ·    ·  [ DW]  ·    ·    ·    ·    ·  [ DW]  ·    ·    ·    ·
 5    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·
 6    ·    ·  [ DL]  ·    ·    ·  [ DL]  ·  [ DL]  ·    ·    ·  [ DL]  ·    ·
 7  [ TW]  ·    ·  [ DL]  ·    ·    ·  [ ★ ]  ·    ·    ·  [ DL]  ·    ·  [ TW]
 8    ·    ·  [ DL]  ·    ·    ·  [ DL]  ·  [ DL]  ·    ·    ·  [ DL]  ·    ·
 9    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·
10    ·    ·    ·    ·  [ DW]  ·    ·    ·    ·    ·  [ DW]  ·    ·    ·    ·
11  [ DL]  ·    ·  [ DW]  ·    ·    ·  [ DL]  ·    ·    ·  [ DW]  ·    ·  [ DL]
12    ·    ·  [ DW]  ·    ·    ·  [ DL]  ·  [ DL]  ·    ·    ·  [ DW]  ·    ·
13    ·  [ DW]  ·    ·    ·  [ TL]  ·    ·    ·  [ TL]  ·    ·    ·  [ DW]  ·
14  [ TW]  ·    ·  [ DL]  ·    ·    ·  [ TW]  ·    ·    ·  [ DL]  ·    ·  [ TW]
```

| Symbol | Bonus         | Effect                          |
|:------:|:--------------|:--------------------------------|
| TW     | Triple Word   | word score × 3                  |
| DW     | Double Word   | word score × 2                  |
| TL     | Triple Letter | that letter's score × 3         |
| DL     | Double Letter | that letter's score × 2         |
| ★      | Centre (DW)   | word score × 2, first move only |

Multipliers apply only to tiles placed on that square during the current move; tiles already on the board always score at face value.

---

## Tile Distribution

The bag contains **98 tiles** across 32 letters of the Polish alphabet.

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

A blank tile (`?`) may substitute for any letter during a search but scores 0 points.

---

## Requirements

| Dependency     | Version | Purpose                              |
|:---------------|:--------|:-------------------------------------|
| Rust toolchain | ≥ 1.70  | Compiling the engine                 |
| Python         | ≥ 3.10  | Running game logic                   |
| maturin        | ≥ 1.0   | Building the Python extension        |
| rayon          | 1.12    | Parallel pattern evaluation (bundled)|
| pyo3           | 0.28    | Python bindings (bundled)            |

---

## Installation

### Build the Python extension

```bash
# install maturin if needed
pip install maturin

# build and install into the current Python environment
maturin develop --release
```

The compiled `.so` lands in `target/release/` and is registered in your environment automatically.

### Rebuild the DAWG dictionary

A pre-built `words/dawg.bin` is included. To recompile from a word list:

```bash
cargo run --release -- build words/words.txt words/dawg.bin
```

To verify quickly with a smaller input file:

```bash
cargo run --release -- build words/sth.txt /tmp/dawg.bin
```

---

## Python API

### `Dawg`

```python
from scrablozaur import Dawg

d = Dawg("words/dawg.bin")

# membership test
d.contains("hamulec")        # True
"hamulec" in d               # True  (same via __contains__)
d.contains("xyzzy")          # False

# pattern search (see Pattern Syntax below)
d.search("ha-ulec", "m")     # ['hamulec']
d.search("*", "aekrtu")      # all words buildable from these letters
d.search("k-t", "oar?")      # k + one letter + t, '?' = blank tile

# diagnostic
d.node_count()               # number of DAWG nodes after minimization
```

### `Board`

```python
from scrablozaur import Board, Dawg

d = Dawg("words/dawg.bin")
b = Board([["-"] * 15 for _ in range(15)])

# draw letters from the bag (fills hand up to 7 tiles)
hand = b.give_letters("")              # e.g. "aeimnrt"

# find the best first move (must cover the centre square)
word, score, (row, col, horizontal), used = b.get_best_word(d, hand, first=True)

# score and validate before committing — calculate_word_points must be called
# before place_word; after placement the tiles are no longer on empty squares
# and bonus multipliers no longer apply.
pts = b.calculate_word_points(word, row=row, col=col, horizontal=horizontal, letters=hand)
b.check_word_placement(d, word, row=row, col=col, horizontal=horizontal)  # raises on invalid

b.place_word(word, row, col, horizontal)
for ch in used:
    hand = hand.replace(ch, "", 1)
hand += b.give_letters(hand)

# subsequent moves
word, score, (row, col, horizontal), used = b.get_best_word(d, hand, first=False)

# inspect candidate patterns
b.get_all_patterns()     # list of (index, start, end, horizontal)
b.get_row_patterns(7)    # patterns in row 7
b.get_col_patterns(4)    # patterns in column 4

print(b)                 # pretty-print the board
```

### Full two-player simulation

```python
from scrablozaur import Board, Dawg

d = Dawg("words/dawg.bin")


class Player:
    def __init__(self, board: Board) -> None:
        self.board = board
        self.letters = board.give_letters("")
        self.score = 0

    def play(self, first: bool = False) -> str:
        word, points, (row, col, horiz), used = self.board.get_best_word(
            d, self.letters, first
        )
        if not word:
            return ""
        self.score += points
        self.board.place_word(word, row, col, horiz)
        for ch in used:
            self.letters = self.letters.replace(ch, "", 1)
        self.letters += self.board.give_letters(self.letters)
        return word


b = Board([["-"] * 15 for _ in range(15)])
p1, p2 = Player(b), Player(b)

p1.play(first=True)
while True:
    if not p2.play():
        break
    if not p1.play():
        break

print(f"Player 1: {p1.score}  Player 2: {p2.score}")
print(b)
```

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
d.search("l--y",  "oadn")    # 4-letter words: l, 2 from hand, y
d.search("l*y",   "oadn")    # l, any count from hand, y
d.search("k-t",   "oar?")    # k, one letter, t  — blank tile available
d.search("*",     "aeimnrt") # every word buildable from 7 tiles
d.search("ham*",  "lec")     # words starting with 'ham', hand extends freely
```

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
header (8 bytes):
  [4] root node ID
  [4] total node count

per node:
  [1] is_terminal flag
  [4] number of children
  per child:
    [4] Unicode codepoint of the edge label (sorted ascending)
    [4] child node ID
```

Child edges are stored sorted by codepoint, enabling binary search in O(log k) where k ≤ 33 (Polish alphabet). The offset of each node is precomputed into a flat table at load time so every node access is a direct array index with no pointer chasing.

### Pattern matching

`match_pattern` is a recursive DAWG traversal. At each `-` slot it iterates over child edges and checks the player's letter frequency array in O(1):

```
freq[c as usize] > 0   →  use the tile
freq['?' as usize] > 0 →  use a blank as this letter
```

A `mandatory_slots` counter tracks how many `-` tokens remain in the unprocessed pattern, preventing `*` expansions from starving them. Results are deduplicated with `sort + dedup` to handle the case where the same word is reachable via both a regular tile and a blank.

### Move search

`get_best_word` works in three phases:

1. **Pattern generation** — every row and column is scanned for contiguous spans that contain both placed tiles (anchors) and empty squares; those are the positions where a new word could legally connect to the board.

2. **Parallel evaluation** — patterns are dispatched to a Rayon work-stealing pool. Each thread independently runs `best_word_from_pattern_inner`: DAWG search → cross-word validation → scoring, with no shared mutable state between threads. Patterns whose empty-slot count exceeds the hand size are skipped before entering the DAWG.

3. **Global reduction** — `max_by_key` selects the highest-scoring result across all threads.

### Bonus table

`calculate_word_points` uses a precomputed **8 × 8 static table** that exploits the board's four-fold reflective symmetry:

```rust
let (r2, c2) = (r.min(14 - r), c.min(14 - c));   // fold into quadrant
let (letter_mul, word_mul) = BONUS_TABLE[r2][c2];  // O(1) lookup
```

---

## CLI

The crate includes three diagnostic commands:

```bash
cargo run --release -- build  words/words.txt  words/dawg.bin   # compile DAWG
cargo run --release -- lookup words/dawg.bin   hamulec          # single lookup
cargo run --release -- bench  words/dawg.bin   words/words.txt  # throughput benchmark
```

Sample `bench` output:

```
Results (5 × 317162 = 1585810 lookups):
  total time  : 183.441ms
  throughput  : 8647532 lookups/s
  per lookup  : 115.7 ns
  hits        : 1585810/1585810 (100.0%)
```

---

## Project Structure

```
scrablozaur/
├── src/
│   ├── lib.rs           # Rust engine: DAWG, Board, pattern search, scoring
│   └── main.py          # Python game loop (two-player simulation)
├── words/
│   ├── words.txt        # 3.17 M-word Polish dictionary (source)
│   └── dawg.bin         # compiled DAWG (pre-built)
├── test/                # sample board states for manual testing
├── scrablozaur.pyi      # Python type stubs
├── pyrightconfig.json   # Pyright / type-checker configuration
└── Cargo.toml           # Rust package manifest
```
