"""Score the scan decoder against the dictionary pass it would replace.

The labelled test photos are mostly deliberate non-word layouts, so a decoder
that insists on real words cannot be scored on them. Instead, self-play games
make legal boards -- blanks included -- and every square of them gets a reading
the classical pipeline really produced on a test photo of the chosen
difficulty: a tile gets a real reading of a tile showing the same letter (the
detector's real misses among them), an empty square a real empty square's
(false alarms among them). Blanks, which no test photo shows, read as tiles
with no legible letter. Squares are drawn independently, which is kinder than
a real photo, where glare and parallax spoil whole regions.

Each board is decoded twice: "cold", the first scan of a game, with no
previous board; and "warm", with the board before the last move already
confirmed, as on every later scan.

Compared, per difficulty: the readings as they are, the web's single-letter
correction (`web/scan.py`), and the decoder. The classical scores are
calibrated into probabilities on the same export first.

Needs the dataset `scripts/export_scan_dataset.py` writes. From the repository
root:

    uv run python board_reader/scripts/eval_decoder.py [--games 30] [--seed 1]
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import scrablozaur  # noqa: E402
from languages import engine_language, load  # noqa: E402
from web.scan import _MIN_ALTERNATIVE_PROB, _verify_and_correct  # noqa: E402

DATASET = ROOT / "board_reader" / "test" / "derived" / "scan_dataset"
GRID = 15
TIERS = "emh"
# A tile whose letter was read is almost surely not a blank, which has no
# glyph to read; one with no legible glyph usually is, since real letters are
# rarely illegible; a tile the detector missed has no reading, so it gets the
# share of blanks among a board's tiles.
P_BLANK_READ = 1e-5
P_BLANK_ILLEGIBLE = 0.85
P_BLANK_UNSEEN = 0.03

# (detector says tile, its confidence, (letter, confidence, alternatives) or None)
Raw = tuple[bool, float, tuple | None]


def lower_reading(reading):
    if reading is None:
        return None
    letter, conf, alts = reading
    return (letter.lower() if letter else None, conf, [(a.lower(), p) for a, p in alts])


def load_records(path: Path) -> list[dict]:
    files = sorted(glob.glob(str(path / "board*.json")))
    if not files:
        sys.exit(f"no dataset at {path}: run board_reader/scripts/export_scan_dataset.py first")
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]


def cells(rec):
    for r in range(GRID):
        for c in range(GRID):
            key = f"{r},{c}"
            is_tile, conf = rec["tiles"][key]
            yield r, c, rec["truth"][r][c].lower(), (is_tile, conf, lower_reading(rec["readings"].get(key)))


def conf_bin(x: float) -> int:
    return min(19, int(x * 20))


def occupancy_key(is_tile: bool, conf: float, reading) -> tuple:
    """What decides how likely a square holds a tile: the detector's verdict
    and confidence, and -- for a detected tile -- whether the classifier read
    a letter with conviction. False alarms rarely look like a letter (median
    letter confidence 0.40 against 0.62 for real tiles)."""
    legible = None if reading is None or reading[0] is None else reading[1] >= 0.5
    return (is_tile, conf_bin(conf), legible)


class Calibration:
    """Classical scores mapped to probabilities by counting outcomes."""

    def __init__(self, records):
        self.occ = collections.defaultdict(lambda: [0, 0])  # occupancy_key -> [tiles, squares]
        self.top = collections.defaultdict(lambda: [0, 0])  # bin -> [top letter right, readings]
        ranks = [0] * 5
        wrong = 0
        for rec in records:
            for _, _, truth, (is_tile, conf, reading) in cells(rec):
                slot = self.occ[occupancy_key(is_tile, conf, reading)]
                slot[0] += truth != "-"
                slot[1] += 1
                if truth == "-" or reading is None or reading[0] is None:
                    continue
                letter, lconf, alts = reading
                hit = self.top[conf_bin(lconf)]
                hit[0] += letter == truth
                hit[1] += 1
                if letter != truth:
                    wrong += 1
                    for k, (alt, _) in enumerate(alts):
                        if alt == truth:
                            ranks[k] += 1
        # P(truth is the k-th alternative | the top letter is wrong).
        self.rank = [(ranks[k] + 0.5) / (wrong + 1) for k in range(5)]

    def p_tile(self, is_tile: bool, conf: float, reading) -> float:
        tiles, n = self.occ[occupancy_key(is_tile, conf, reading)]
        return (tiles + 1) / (n + 2)

    def p_top(self, conf: float) -> float:
        right, n = self.top[conf_bin(conf)]
        return (right + 1) / (n + 2)

    def reading(self, raw: Raw):
        is_tile, conf, reading = raw
        p_tile = self.p_tile(is_tile, conf, reading)
        if reading is None:
            return (p_tile, [], P_BLANK_UNSEEN)
        letter, lconf, alts = reading
        if letter is None:
            return (p_tile, [], P_BLANK_ILLEGIBLE)
        p_top = self.p_top(lconf)
        letters = [(letter, p_top)]
        for k, (alt, _) in enumerate(alts):
            if alt != letter:
                letters.append((alt, (1 - p_top) * self.rank[k]))
        return (p_tile, letters, P_BLANK_READ)


class Pools:
    """Real readings to draw from, per difficulty: tiles by letter, and empty squares."""

    def __init__(self, records):
        self.tiles = {t: collections.defaultdict(list) for t in TIERS}
        self.empty = {t: [] for t in TIERS}
        for rec in records:
            t = rec["difficulty"]
            for _, _, truth, raw in cells(rec):
                if truth == "-":
                    self.empty[t].append(raw)
                else:
                    self.tiles[t][truth].append(raw)
        self.any_tile = {t: [raw for pool in self.tiles[t].values() for raw in pool] for t in TIERS}

    def tile(self, letter: str, tier: str, rng: random.Random) -> Raw:
        pool = self.tiles[tier][letter]
        if len(pool) >= 3:
            return rng.choice(pool)
        # Too rare in this tier's photos: borrow another letter's reading
        # and swap the two letters throughout, which keeps its error.
        donor = rng.choice([x for x, p in self.tiles[tier].items() if len(p) >= 3])
        is_tile, conf, reading = rng.choice(self.tiles[tier][donor])

        def swap(ch):
            return letter if ch == donor else donor if ch == letter else ch

        if reading is not None and reading[0] is not None:
            reading = (swap(reading[0]), reading[1], [(swap(a), p) for a, p in reading[2]])
        return (is_tile, conf, reading)

    def blank(self, tier: str, rng: random.Random) -> Raw:
        is_tile, conf, _ = rng.choice(self.any_tile[tier])
        return (is_tile, conf, (None, 0.0, []) if is_tile else None)


class Classical:
    """Readings the classical pipeline produced, calibrated."""

    def __init__(self, records):
        self.calibration = Calibration(records)
        self.pools = Pools(records)

    def tile(self, letter, tier, rng):
        return self.pools.tile(letter, tier, rng)

    def blank(self, tier, rng):
        return self.pools.blank(tier, rng)

    def empty(self, tier, rng):
        pool = self.pools.empty[tier]
        return pool[rng.randrange(len(pool))]

    def reading(self, raw):
        return self.calibration.reading(raw)

    def read(self, raw) -> str:
        """The square as read: '-', '?' for a tile with no legible letter, or the letter."""
        is_tile, _, rd = raw
        return "-" if not is_tile else (rd[0] if rd and rd[0] else "?")

    def alternatives(self, raw):
        is_tile, _, rd = raw
        return rd[2] if is_tile and rd else []


class Learned:
    """Readings the cell reader made of the test photos while they were held
    out of its training: `train_cell_reader.py --dump`. A reading is the
    softmax over (empty, blank, every letter), already calibrated."""

    def __init__(self, records, dump_path: Path):
        dump = json.loads(dump_path.read_text(encoding="utf-8"))
        self.letters = dump["alphabet"]
        probs = dict(zip(dump["boards"], dump["probs"]))
        self.tiles = {t: collections.defaultdict(list) for t in TIERS}
        self.empties = {t: [] for t in TIERS}
        self.blanks = [np.array(p) for p in dump["blank_probs"]]
        for rec in records:
            board = int(rec["photo"][3:].split("_")[0])
            if board not in probs:
                continue
            tier = rec["difficulty"]
            for r in range(GRID):
                for c in range(GRID):
                    p = np.array(probs[board][r * GRID + c])
                    truth = rec["truth"][r][c].lower()
                    if truth == "-":
                        self.empties[tier].append(p)
                    elif truth == "?":
                        self.blanks.append(p)
                    else:
                        self.tiles[tier][truth].append(p)

    def tile(self, letter, tier, rng):
        pool = self.tiles[tier][letter]
        if len(pool) >= 3:
            return rng.choice(pool)
        # Too rare in this tier: another letter's reading, the two swapped.
        donor = rng.choice([x for x, p in self.tiles[tier].items() if len(p) >= 3])
        p = rng.choice(self.tiles[tier][donor]).copy()
        i, j = 2 + self.letters.index(letter), 2 + self.letters.index(donor)
        p[i], p[j] = p[j], p[i]
        return p

    def blank(self, tier, rng):
        return rng.choice(self.blanks)

    def empty(self, tier, rng):
        pool = self.empties[tier]
        return pool[rng.randrange(len(pool))]

    def reading(self, p):
        p_tile = max(1e-6, 1 - float(p[0]))
        lettered = max(1e-6, p_tile - float(p[1]))
        best = np.argsort(p[2:])[::-1][:8]
        return (p_tile, [(self.letters[i], float(p[2 + i]) / lettered) for i in best], float(p[1]) / p_tile)

    def read(self, p) -> str:
        k = int(np.argmax(p))
        return "-" if k == 0 else "?" if k == 1 else self.letters[k - 2]

    def alternatives(self, p):
        _, letters, _ = self.reading(p)
        return letters


def parse_board(board) -> list[list[str]]:
    return [line.split() for line in str(board).splitlines()]


def self_play(dawg, lang, games: int, seed: int):
    """Legal boards, every position of `games` self-play games: (letters, blanks) after each move."""
    rng = random.Random(seed)
    out = []
    for g in range(games):
        board = scrablozaur.Board.seeded(lang, seed * 1000 + g)
        racks, passes, turn = ["", ""], 0, 0
        history: list[tuple[list[list[str]], list[list[bool]]]] = []
        while passes < 2 and len(history) < 60:
            rack = racks[turn] + board.give_letters(racks[turn])
            moves = board.get_best_words(dawg, rack, 4)
            if not moves:
                passes += 1
                racks[turn] = rack
                turn ^= 1
                continue
            passes = 0
            word, _, (r, c, horizontal), used = moves[0] if rng.random() < 0.5 else rng.choice(moves)
            board.place_word(word, r, c, horizontal, used)
            for tile in used:
                rack = rack.replace(tile, "", 1)
            racks[turn] = rack
            history.append((parse_board(board), board.blank_mask()))
            if not rack and board.bag_remaining() == 0:
                break
            turn ^= 1
        out.append(history)
    return out


def cases(histories, every: int):
    """(previous board, board) pairs: every `every`-th move of each game, and its last."""
    for history in histories:
        picks = set(range(every - 1, len(history), every)) | {len(history) - 1}
        for m in sorted(picks):
            if m >= 1:
                yield history[m - 1], history[m]


def synthesize(board, blanks, tier, source, rng):
    return [
        [
            source.empty(tier, rng)
            if board[r][c] == "-"
            else source.blank(tier, rng)
            if blanks[r][c]
            else source.tile(board[r][c], tier, rng)
            for c in range(GRID)
        ]
        for r in range(GRID)
    ]


def as_read(raw, source) -> list[list[str]]:
    return [[source.read(x) for x in row] for row in raw]


def greedy(dawg, raw, prior, letters, source):
    """`web/scan.py`'s `scan_board_image` after the vision stage."""
    grid = as_read(raw, source)
    grid = [[ch if ch in letters or ch == "-" else "?" for ch in row] for row in grid]
    alternatives = [
        [[a for a, p in source.alternatives(x) if p >= _MIN_ALTERNATIVE_PROB and a in letters] for x in row]
        for row in raw
    ]
    locked = set()
    if prior is not None:
        for r in range(GRID):
            for c in range(GRID):
                if prior[r][c] != "-":
                    locked.add((r, c))
                    grid[r][c] = prior[r][c]
    _verify_and_correct(dawg, grid, alternatives, locked=locked)
    return grid


def errors(truth, grid):
    return sum(truth[r][c] != grid[r][c] for r in range(GRID) for c in range(GRID))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=DATASET)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--every", type=int, default=4, help="score every n-th position of a game")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--tiers", default=TIERS)
    ap.add_argument(
        "--reader",
        default="classical",
        help="'classical', or the held-out readings `train_cell_reader.py --dump` wrote",
    )
    for name in ("min_p_tile", "max_p_tile", "letter_floor", "blank_floor"):
        ap.add_argument(f"--{name.replace('_', '-')}", dest=name, type=float)
    ap.add_argument("--max-flips", type=int)
    ap.add_argument("--budget", type=int)
    ap.add_argument(
        "--dump", type=Path, help="also write every decoded case here, for `scrablozaur scan-bench` on a phone"
    )
    args = ap.parse_args()
    options = {
        k: getattr(args, k)
        for k in ("min_p_tile", "max_p_tile", "letter_floor", "blank_floor", "max_flips", "budget")
        if getattr(args, k) is not None
    }

    spec = load("pl")
    lang = engine_language(spec)
    dawg = scrablozaur.Dawg(lang, str(spec.dawg), str(spec.gaddag))
    letters = set(spec.alphabet)
    records = load_records(args.dataset)
    source = Classical(records) if args.reader == "classical" else Learned(records, Path(args.reader))
    start = time.time()
    histories = self_play(dawg, lang, args.games, args.seed)
    pairs = list(cases(histories, args.every))
    tiles = [sum(ch != "-" for row in b[0] for ch in row) for _, b in pairs]
    print(
        f"{len(pairs)} boards from {args.games} self-play games in {time.time() - start:.0f} s, "
        f"{min(tiles)}-{max(tiles)} tiles (median {statistics.median(tiles):.0f}); options {options or 'default'}"
    )

    rng = random.Random(args.seed)
    dumped = []
    for tier in args.tiers:
        for scenario in ("cold", "warm"):
            stats = collections.defaultdict(list)
            for (prev, prev_blanks), (board, blanks) in pairs:
                raw = synthesize(board, blanks, tier, source, rng)
                prior = prev if scenario == "warm" else None
                read = as_read(raw, source)
                if prior is not None:
                    read = [
                        [prior[r][c] if prior[r][c] != "-" else read[r][c] for c in range(GRID)] for r in range(GRID)
                    ]
                fixed = greedy(dawg, raw, prior, letters, source)
                readings = [[source.reading(x) for x in row] for row in raw]
                if args.dump:
                    dumped.append(
                        {
                            "label": f"{tier} {scenario}",
                            "readings": readings,
                            "prior": prior,
                            "prior_blanks": prev_blanks if prior is not None else None,
                        }
                    )
                t0 = time.perf_counter()
                out = scrablozaur.decode_scan(
                    dawg, readings, prior, prev_blanks if prior is not None else None, **options
                )
                stats["ms"].append((time.perf_counter() - t0) * 1000)
                stats["work"].append(out.work)
                stats["legal"].append(out.legal)
                stats["complete"].append(out.complete)
                for name, grid in (("read", read), ("greedy", fixed), ("decoder", out.letters)):
                    e = errors(board, grid)
                    stats[name].append(e)
                    # Squares the method got wrong that the readings had right.
                    stats[name + "_broke"].append(
                        sum(read[r][c] == board[r][c] != grid[r][c] for r in range(GRID) for c in range(GRID))
                    )
            n = len(pairs)
            line = [f"{tier} {scenario}:"]
            for name in ("read", "greedy", "decoder"):
                e = stats[name]
                line.append(
                    f"{name} {statistics.mean(e):5.2f} wrong/board, {sum(x == 0 for x in e) / n:5.1%} exact"
                    + (f" (broke {statistics.mean(stats[name + '_broke']):.2f})" if name != "read" else "")
                )
            ms = sorted(stats["ms"])
            line.append(
                f"| legal {sum(stats['legal']) / n:.0%}, complete {sum(stats['complete']) / n:.0%}, "
                f"{statistics.mean(ms):.1f} ms mean, p90 {ms[int(0.9 * (n - 1))]:.1f}, max {ms[-1]:.0f}, "
                f"work max {max(stats['work'])}"
            )
            print(" | ".join(line[:4]), line[4])
    if args.dump:
        args.dump.write_text(json.dumps(dumped, ensure_ascii=False), encoding="utf-8")
        print(f"{len(dumped)} cases written to {args.dump}")


if __name__ == "__main__":
    main()
