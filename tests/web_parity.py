"""A golden record of the web app's game logic, for the Rust port to replay.

The Android app plays through `src/game/` -- a Rust port of `web/game.py`, the
game and board routers, `src/rules.py`, `src/strategy.py`'s rank windows and
`smart_player`'s `choose_move` -- and `src/scan.rs`, the port of `web/scan.py`'s
dictionary pass. A port that is not checked against its original drifts, which
is how the web bot once ended up weaker than the CLI one. So this records what
the *Python* implementation does, and `cargo test` asserts the port does the
same: the same deals, bot moves, scores, hints, ratings, errors and
corrections.

Games are played through the web's real HTTP API (FastAPI's TestClient), with
its randomness swapped for the engine's own xorshift shuffle and randint --
the algorithm `src/game/rng.rs` uses -- so both sides deal identical bags.

    uv run python tests/web_parity.py     # rewrite tests/fixtures/web_parity.json

`tests/test_web_parity.py` fails if the Python side has changed since the
fixture was written; regenerate it, then make `cargo test` pass again.
"""

from __future__ import annotations

import json
import random as _stdlib_random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
for _p in (
    _ROOT,
    _ROOT / "src",
    _ROOT / "smart_player",
    _ROOT / "board_reader" / "src",
):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FIXTURE = _ROOT / "tests" / "fixtures" / "web_parity.json"
FIXTURE_VERSION = 1
# The interactive endgame budget both sides use (web/game.py's WEB_ENDGAME_NODES).
ENDGAME_NODES = 50_000
_MASK = (1 << 64) - 1


class XorshiftRandom:
    """The engine's RNG, standing in for the `random` module in `web.game` and
    `strategy`. `shuffle` is `src/lib.rs`'s Fisher-Yates step for step, and
    `randint` always consumes one draw, as `GameRng::randint` does."""

    def __init__(self, seed: int) -> None:
        self.state = seed & _MASK

    def _next(self) -> int:
        s = self.state
        s ^= (s << 13) & _MASK
        s ^= s >> 7
        s ^= (s << 17) & _MASK
        self.state = s
        return s

    def shuffle(self, x: list) -> None:
        for i in range(len(x) - 1, 0, -1):
            j = self._next() % (i + 1)
            x[i], x[j] = x[j], x[i]

    def randint(self, a: int, b: int) -> int:
        return a + self._next() % (b - a + 1)


@contextmanager
def engine_randomness(seed: int):
    import strategy
    import web.game

    shim = XorshiftRandom(seed)
    saved = (web.game.random, strategy.random)
    web.game.random = shim
    strategy.random = shim
    try:
        yield shim
    finally:
        web.game.random, strategy.random = saved


# ── Shared encoding ──────────────────────────────────────────────────────────


def _rows(grid) -> list[str]:
    return ["".join(row) for row in grid]


def _blank_cells(mask) -> list[list[int]]:
    return [[r, c] for r in range(15) for c in range(15) if mask[r][c]]


def _owners(owners) -> list[str]:
    return ["".join("." if o is None else str(o) for o in row) for row in owners]


def _suggestion(s: dict) -> list:
    return [s["word"], s["score"], s["row"], s["col"], s["horizontal"], s["value"]]


def _board_rows(board) -> list[str]:
    return [row.replace(" ", "") for row in str(board).strip().split("\n")]


def _state(body: dict) -> dict:
    """The fields of a `BoardStateResponse` the port must reproduce."""
    last = body["last_computer_move"]
    return {
        "board": _rows(body["board"]),
        "blanks": _blank_cells(body["board_blanks"]),
        "owners": _owners(body["tile_owners"]),
        "players": [
            [p["name"], p["is_computer"], p["score"], p["letters"]]
            for p in body["players"]
        ],
        "current": body["current_player_idx"],
        "first": body["is_first_move"],
        "move_number": body["move_number"],
        "can_undo": body["can_undo"],
        "mode": body["game_mode"],
        "tiles_remaining": body["tiles_remaining"],
        "last_computer_move": (
            None
            if last is None
            else [
                last[k] for k in ("word", "score", "row", "col", "horizontal", "passed")
            ]
        ),
        "game_over": body["game_over"],
        "winner": body["winner_name"],
        "rating": body["last_move_rating"],
    }


# What the web's 400 messages mean, as the port's `GameError` names them.
_ERRORS = [
    ("nie ma w słowniku", "NotInDictionary"),
    ("Pierwsze słowo musi przechodzić przez środek", "FirstMoveMustCoverCentre"),
    ("Słowo musi być połączone", "NotConnected"),
    ("cross-word", "CrossWordInvalid"),
    ("Nie masz wymaganych liter", "MissingTiles"),
    ("Gra już się zakończyła", "GameOver"),
    ("Brak ruchów do cofnięcia", "NothingToUndo"),
    ("only available in sandbox mode", "WrongMode"),
    ("Dostępne tylko w trybie automatycznym", "WrongMode"),
    ("Podpowiedzi dostępne tylko w trybie rywalizacji", "WrongMode"),
]


def _error_kind(detail: str) -> str:
    for needle, kind in _ERRORS:
        if needle in detail:
            return kind
    raise ValueError(f"unmapped web error: {detail!r}")


# ── Whole games through the HTTP API ─────────────────────────────────────────


class Recorder:
    def __init__(self, client, name: str, lang: str, seed: int) -> None:
        self.client = client
        self.trace = {"name": name, "lang": lang, "seed": seed, "steps": []}
        self.state: dict | None = None

    def call(self, action: str, method: str, path: str, *, json_body=None, params=None):
        resp = self.client.request(method, f"/api{path}", json=json_body, params=params)
        step = {"action": action, "args": json_body or params or {}}
        body = resp.json()
        if resp.status_code != 200:
            step["error"] = _error_kind(body["detail"])
        elif "suggestions" in body:
            step["suggestions"] = [_suggestion(s) for s in body["suggestions"]]
        elif "board" in body:
            self.state = body
            step["state"] = _state(body)
        else:
            step["preview"] = body
        self.trace["steps"].append(step)
        return resp.status_code, body

    def move(
        self, action: str, path: str, word: str, row: int, col: int, horizontal: bool
    ):
        body = {"word": word, "row": row, "col": col, "horizontal": horizontal}
        return self.call(action, "POST", path, json_body=body)


def _new_game(
    rec: Recorder, mode: str, players: list[dict], difficulty=8, strength=100
):
    body = {
        "players": players,
        "game_mode": mode,
        "difficulty": difficulty,
        "strength": strength,
        "language": rec.trace["lang"],
    }
    return rec.call("new", "POST", "/game/new", json_body=body)


def _competitive(client, name, lang, seed, difficulty, strength, dawg, *, max_turns=30):
    """A human who always plays the top hint (and probes a few mistakes) against
    a bot at `difficulty`/`strength`."""

    rec = Recorder(client, name, lang, seed)
    with engine_randomness(seed):
        _new_game(rec, "competitive", [{"name": "Gracz"}], difficulty, strength)
        for turn in range(max_turns):
            state = rec.state
            if state["game_over"]:
                break
            if state["is_first_move"] and turn == 0:
                # A real word that misses the centre square.
                rec.move(
                    "human_move",
                    "/board/human-move",
                    "kot" if lang == "pl" else "cat",
                    0,
                    0,
                    True,
                )
            if turn == 2:
                rec.move("human_move", "/board/human-move", "qxzv", 7, 7, True)
            _, hints = rec.call(
                "hints", "GET", "/board/hints", params={"sort": "score"}
            )
            if turn % 5 == 3:
                rec.call("hints", "GET", "/board/hints", params={"sort": "smart"})
            if turn == 4:
                rec.call("skip", "POST", "/board/skip")
                continue
            if turn == 6:
                rec.call("undo", "POST", "/board/undo")
            suggestions = hints.get("suggestions", [])
            if not suggestions:
                rec.call("skip", "POST", "/board/skip")
                continue
            top = suggestions[0]
            placement = {k: top[k] for k in ("word", "row", "col", "horizontal")}
            rec.call("preview", "POST", "/board/preview-score", json_body=placement)
            letters = state["players"][state["current_player_idx"]]["letters"]
            if turn == 3 and "?" not in letters:
                # A real, well-placed word the rack cannot cover: the best play
                # for a rack of letters this one lacks.
                other = "".join(ch for ch in dawg.lang.alphabet if ch not in letters)[
                    :7
                ]
                board = _board_from_state(dawg, state)
                word, _s, (row, col, horizontal), _u = board.get_best_word(dawg, other)
                if word:
                    rec.move(
                        "human_move", "/board/human-move", word, row, col, horizontal
                    )
            rec.move(
                "human_move",
                "/board/human-move",
                top["word"],
                top["row"],
                top["col"],
                top["horizontal"],
            )
        rec.call("undo", "POST", "/board/undo")
        rec.call("resign", "POST", "/board/pass")
    return rec.trace


def _board_from_state(dawg, state: dict):
    """The engine position a state shows. Built by writing tiles rather than
    with `Board.from_grid`, which (rightly) refuses a grid holding more copies
    of a letter than the set has -- and a sandbox, whose computer rack is typed
    in rather than drawn, can produce exactly that."""
    from scrablozaur import Board

    board = Board(dawg.lang)
    for r, row in enumerate(state["board"]):
        for c, ch in enumerate(row):
            if ch != "-":
                board.place_word(ch, r, c, True, ["?"] if state["board_blanks"][r][c] else None)
    return board


def _sandbox_auto(client, name, lang, seed, players, *, max_moves=120):
    rec = Recorder(client, name, lang, seed)
    with engine_randomness(seed):
        _new_game(rec, "sandbox_auto", players)
        for i in range(max_moves):
            if rec.state["game_over"]:
                break
            rec.call("next_move", "POST", "/board/next-move")
            if i in (5, 17):
                rec.call("undo", "POST", "/board/undo")
        rec.call("next_move", "POST", "/board/next-move")  # refused once over
    return rec.trace


def _sandbox(client, name, lang, seed, racks, dawg, *, turns=12):
    """Referee mode: a human types words (found here with the engine) and the
    computer seat plays the top suggestion for a typed-in rack."""

    rec = Recorder(client, name, lang, seed)
    with engine_randomness(seed):
        _new_game(
            rec, "sandbox", [{"name": "Ala"}, {"name": "Bot", "is_computer": True}]
        )
        for turn in range(turns):
            state = rec.state
            rack = racks[turn % len(racks)]
            if state["players"][state["current_player_idx"]]["is_computer"]:
                rec.call(
                    "set_letters",
                    "POST",
                    "/board/set-letters",
                    json_body={"letters": rack},
                )
                _, sug = rec.call(
                    "suggest", "POST", "/board/suggest", params={"sort": "score"}
                )
                if turn == 3:
                    rec.call(
                        "suggest", "POST", "/board/suggest", params={"sort": "smart"}
                    )
                best = sug["suggestions"][0] if sug.get("suggestions") else None
                if best is None:
                    rec.call("skip", "POST", "/board/skip")
                    continue
                rec.call(
                    "computer_move",
                    "POST",
                    "/board/computer-move",
                    json_body={
                        k: best[k]
                        for k in ("word", "row", "col", "horizontal", "score")
                    },
                )
            else:
                board = _board_from_state(dawg, state)
                word, _score, (row, col, horizontal), _used = board.get_best_word(
                    dawg, rack
                )
                if not word:
                    rec.call("skip", "POST", "/board/skip")
                    continue
                rec.move("human_move", "/board/human-move", word, row, col, horizontal)
            if turn == 5:
                rec.call("undo", "POST", "/board/undo")
        rec.call(
            "hints", "GET", "/board/hints", params={"sort": "score"}
        )  # refused: not competitive
    return rec.trace


# ── Decisions on fixed positions ─────────────────────────────────────────────


def _positions(lang_code: str, seeds, every: int = 2):
    """Real positions, reached by greedy self-play from seeded bags, through to
    the endgame. Yields (dawg, board, rack, bag_remaining) for every `every`-th
    turn, and for every turn once the bag is empty."""
    from languages import engine_language, load
    from rules import Streaks, classify_turn
    from scrablozaur import Board, Dawg
    from strategy import StrategicPlayer

    spec = load(lang_code)
    lang = engine_language(spec)
    dawg = Dawg(lang, str(spec.dawg), str(spec.gaddag))
    for seed in seeds:
        board = Board.seeded(lang, seed)
        players = [StrategicPlayer(board), StrategicPlayer(board)]
        streaks = Streaks()
        for ply in range(100):
            p = players[ply % 2]
            if ply % every == 0 or board.bag_remaining() == 0:
                yield dawg, board, p.letters, board.bag_remaining()
            streaks.record(classify_turn(p.play_word(dawg)))
            if streaks.exhausted or (not p.letters and board.bag_remaining() == 0):
                break


def _decisions(lang_code: str, seeds) -> list[dict]:
    from languages import load
    from player import choose_move
    from web.game import compute_move_rating, rank_suggestions

    net = str(load(lang_code).leave_net.checkpoint)
    out = []
    for dawg, board, rack, bag in _positions(lang_code, seeds):
        if not rack:
            continue
        choice = choose_move(
            board, dawg, rack, bag, model_path=net, endgame_max_nodes=ENDGAME_NODES
        )
        row, col, horizontal = choice.position
        score_hints = rank_suggestions(board, dawg, rack, 10, "score")
        smart_hints = rank_suggestions(board, dawg, rack, 10, "smart")
        actual = sorted({0, *(s["score"] for s in score_hints[::3])})
        shim = SimpleNamespace(board=board)
        out.append(
            {
                "lang": lang_code,
                "rows": _board_rows(board),
                "blanks": _blank_cells(board.blank_mask()),
                "rack": rack,
                "bag_remaining": bag,
                "choose_move": {
                    "word": choice.word,
                    "score": choice.score,
                    "row": row,
                    "col": col,
                    "horizontal": horizontal,
                    "endgame_diff": choice.endgame_diff,
                    "endgame_exact": choice.endgame_exact,
                },
                "score_hints": [_suggestion(s) for s in score_hints],
                "smart_hints": [_suggestion(s) for s in smart_hints],
                "ratings": [
                    [a, compute_move_rating(shim, dawg, rack, a)] for a in actual
                ],
            }
        )
    return out


# ── The scan dictionary pass ─────────────────────────────────────────────────


def _scan_cases(lang_code: str, seeds) -> list[dict]:
    """Real boards with a few tiles misread, each misread cell offered ranked
    alternatives (the true letter among decoys, or missing), some cells locked
    as if carried over from an earlier confirmed scan."""
    from web.scan import _verify_and_correct, flag_invalid

    grids, dawg = [], None
    for dawg, board, _rack, _bag in _positions(lang_code, seeds, every=12):
        grids.append(_board_rows(board))
    alphabet = dawg.lang.alphabet
    cases = []
    for n, final in enumerate(grids):
        rng = _stdlib_random.Random(1000 + n)
        grid = [list(row) for row in final]
        tiles = [(r, c) for r in range(15) for c in range(15) if grid[r][c] != "-"]
        if len(tiles) < 6:
            continue
        locked = set(rng.sample(tiles, len(tiles) // 3))
        alternatives = {}
        for r, c in rng.sample(tiles, min(6, len(tiles))):
            truth = grid[r][c]
            decoys = rng.sample([ch for ch in alphabet if ch != truth], 3)
            if (r, c) not in locked:
                grid[r][c] = decoys[0]
            choice = decoys[1:] + ([truth] if rng.random() < 0.8 else [])
            rng.shuffle(choice)
            alternatives[(r, c)] = choice
        for r, c in rng.sample(tiles, min(3, len(tiles))):
            alternatives.setdefault((r, c), rng.sample(list(alphabet), 2))
        if n % 3 == 0:
            r, c = rng.choice(tiles)
            if (r, c) not in locked:
                grid[r][c] = "?"
        before = ["".join(row) for row in grid]
        only_flagged = flag_invalid(dawg, [row[:] for row in grid], locked=locked)
        alt_grid = [
            [alternatives.get((r, c), []) for c in range(15)] for r in range(15)
        ]
        flagged = _verify_and_correct(dawg, grid, alt_grid, locked=locked)
        cases.append(
            {
                "lang": lang_code,
                "rows": before,
                "alternatives": [
                    [r, c, "".join(a)] for (r, c), a in sorted(alternatives.items())
                ],
                "locked": sorted([list(p) for p in locked]),
                "flag_only": sorted([list(p) for p in only_flagged]),
                "corrected": ["".join(row) for row in grid],
                "flagged": sorted([list(p) for p in flagged]),
            }
        )
    return cases


def build() -> dict:
    from fastapi.testclient import TestClient

    from languages import engine_language, load
    from scrablozaur import Dawg
    from strategy import rank_window
    from web.difficulty import strength_rank_window
    from web.main import app

    pl, en = load("pl"), load("en")
    pl_dawg = Dawg(engine_language(pl), str(pl.dawg), str(pl.gaddag))
    en_dawg = Dawg(engine_language(en), str(en.dawg), str(en.gaddag))
    traces = []
    with TestClient(app) as client:
        traces.append(
            _competitive(
                client, "competitive_pl_level8", "pl", 0x5EED01, 8, 100, pl_dawg
            )
        )
        traces.append(
            _competitive(
                client,
                "competitive_pl_level3_strength40",
                "pl",
                0x5EED02,
                3,
                40,
                pl_dawg,
            )
        )
        traces.append(
            _competitive(
                client, "competitive_en_level8", "en", 0x5EED03, 8, 100, en_dawg
            )
        )
        traces.append(
            _sandbox_auto(
                client,
                "auto_pl_three_bots",
                "pl",
                0x5EED04,
                [
                    {
                        "name": "A",
                        "is_computer": True,
                        "difficulty": 8,
                        "strength": 100,
                    },
                    {
                        "name": "B",
                        "is_computer": True,
                        "difficulty": 4,
                        "strength": 100,
                    },
                    {"name": "C", "is_computer": True, "difficulty": 8, "strength": 55},
                ],
            )
        )
        traces.append(
            _sandbox(
                client,
                "sandbox_pl",
                "pl",
                0x5EED05,
                ["kotawie", "rzeczna", "mąkałby", "siostra"],
                pl_dawg,
            )
        )
    return {
        "version": FIXTURE_VERSION,
        "rank_windows": [list(rank_window(level)) for level in range(1, 11)],
        "strength_windows": [list(strength_rank_window(s)) for s in range(1, 101)],
        "decisions": _decisions("pl", range(3)) + _decisions("en", range(1)),
        "scan_cases": _scan_cases("pl", range(4)),
        "traces": traces,
    }


def dumps(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(dumps(build()), encoding="utf-8")
    print(f"wrote {FIXTURE.relative_to(_ROOT)} ({FIXTURE.stat().st_size // 1024} KiB)")
