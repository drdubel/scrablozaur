"""The shared leaderboard: finished games from this server's own competitive
play and from the Android app's history (uploaded on the player's request),
ranked together.

Off unless the server's operator names a file for it in
`SCRABLOZAUR_LEADERBOARD`: a server keeps nobody's games by default. Records
are the app's history format (`GameRecord.toJson` in
`android/app/src/main/java/pl/scrablozaur/core/GameHistory.kt`), one JSON object
a line. A game is keyed by where it came from and its id, so uploading a
history again replaces rather than duplicates. Uploads need the token in
`SCRABLOZAUR_LEADERBOARD_TOKEN` when one is set.

The standings follow the app's own (`core/Stats.kt`): every human seat of a
game is one result, a resignation is a loss whatever the score, accuracy is
the mean move rating of a game's rated plays.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

RECORD_VERSION = 1
MAX_UPLOAD_GAMES = 2000
TOP_SCORES = 10

_lock = threading.Lock()


def _path() -> Path | None:
    path = os.environ.get("SCRABLOZAUR_LEADERBOARD")
    return Path(path) if path else None


def enabled() -> bool:
    return _path() is not None


def token_required() -> str | None:
    return os.environ.get("SCRABLOZAUR_LEADERBOARD_TOKEN") or None


def _read() -> dict[tuple[str, str], dict]:
    path = _path()
    games: dict[tuple[str, str], dict] = {}
    if path is None or not path.exists():
        return games
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
            games[(entry["source"], entry["game"]["id"])] = entry
        except (ValueError, KeyError, TypeError):
            continue  # a line a crash cut short
    return games


def _write(games: dict[tuple[str, str], dict]) -> None:
    path = _path()
    if path is None:
        return
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in games.values()), encoding="utf-8")
    tmp.replace(path)


def valid_record(r) -> bool:
    """What the standings read off a record, there and well-formed."""
    try:
        players = r["players"]
        return (
            r["v"] == RECORD_VERSION
            and isinstance(r["id"], str)
            and 0 < len(r["id"]) <= 100
            and isinstance(r["language"], str)
            and isinstance(r["finished"], int)
            and isinstance(r["winner"], int)
            and isinstance(r["turns"], list)
            and 1 < len(players) <= 4
            and all(
                isinstance(p["name"], str)
                and 0 < len(p["name"]) <= 20
                and isinstance(p["computer"], bool)
                and isinstance(p["score"], int)
                for p in players
            )
            and any(not p["computer"] for p in players)
        )
    except (KeyError, TypeError):
        return False


def add(source: str, records: list[dict]) -> int:
    """Keep `records` from `source`, replacing the ones already kept. Returns
    how many were valid and kept."""
    kept = [r for r in records if valid_record(r)]
    with _lock:
        games = _read()
        for r in kept:
            games[(source, r["id"])] = {"source": source, "received": int(time.time() * 1000), "game": r}
        _write(games)
    return len(kept)


def records(language: str | None = None) -> list[dict]:
    with _lock:
        games = [e["game"] for e in _read().values()]
    return [g for g in games if language is None or g["language"] == language]


# ── Standings (core/Stats.kt's rules) ─────────────────────────────────────────


def _resigned(game: dict, seat: int) -> bool:
    return any(t.get("kind") == "RESIGN" and t.get("p") == seat for t in game["turns"])


def result(game: dict, seat: int) -> str:
    """"win", "loss" or "tie" for one seat: a resignation is a loss; otherwise
    the game's outcome, a tie being shared by the seats level at the top."""
    if _resigned(game, seat):
        return "loss"
    if game["winner"] == seat:
        return "win"
    if game["winner"] < 0:
        standing = [p["score"] for i, p in enumerate(game["players"]) if not _resigned(game, i)]
        return "tie" if standing and game["players"][seat]["score"] == max(standing) else "loss"
    return "loss"


def accuracy(game: dict, seat: int) -> float | None:
    ratings = [t["rating"] for t in game["turns"] if t.get("p") == seat and t.get("rating") is not None]
    return sum(ratings) / len(ratings) if ratings else None


def standings(language: str | None = None) -> dict:
    games = sorted(records(language), key=lambda g: g["finished"])
    by_player: dict[str, dict] = {}
    entries = []
    for g in games:
        for seat, p in enumerate(g["players"]):
            if p["computer"]:
                continue
            res = result(g, seat)
            acc = accuracy(g, seat)
            others = [o for i, o in enumerate(g["players"]) if i != seat]
            opponent = max(others, key=lambda o: o["score"])
            entries.append((g, seat, res, opponent))
            row = by_player.setdefault(
                p["name"].strip().lower(),
                {"games": 0, "wins": 0, "losses": 0, "ties": 0, "scores": [], "accuracies": []},
            )
            row["name"] = p["name"]  # the latest game's spelling
            row["games"] += 1
            row[{"win": "wins", "loss": "losses", "tie": "ties"}[res]] += 1
            row["scores"].append(p["score"])
            if acc is not None:
                row["accuracies"].append(acc)
    players = [
        {
            "name": r["name"],
            "games": r["games"],
            "wins": r["wins"],
            "losses": r["losses"],
            "ties": r["ties"],
            "win_rate": r["wins"] / r["games"],
            "avg_score": sum(r["scores"]) / r["games"],
            "best_score": max(r["scores"]),
            "accuracy": sum(r["accuracies"]) / len(r["accuracies"]) if r["accuracies"] else None,
        }
        for r in by_player.values()
    ]
    players.sort(key=lambda r: (-r["win_rate"], -r["games"], r["name"]))
    top = sorted(entries, key=lambda e: (-e[0]["players"][e[1]]["score"], e[0]["finished"]))[:TOP_SCORES]
    return {
        "games": len(games),
        "languages": sorted({g["language"] for g in records()}),
        "players": players,
        "top_scores": [
            {
                "name": g["players"][seat]["name"],
                "score": g["players"][seat]["score"],
                "opponent": opponent["name"],
                "opponent_computer": opponent["computer"],
                "opponent_level": opponent.get("level"),
                "opponent_strength": opponent.get("strength"),
                "result": res,
                "language": g["language"],
                "finished": g["finished"],
            }
            for g, seat, res, opponent in top
        ],
    }


# ── This server's own games ───────────────────────────────────────────────────


def _play(p) -> dict | None:
    if p is None:
        return None
    return {"word": p.word, "score": p.score, "row": p.row, "col": p.col, "h": p.horizontal, "used": p.used}


def record_of(session) -> dict:
    """A finished competitive web game in the app's record format."""
    grid = session.board_grid()
    blanks = session.board.blank_mask()
    scores = [p.score for p in session.players]
    top = max(scores)
    winner = scores.index(top) if scores.count(top) == 1 else -1
    turns = []
    for t in session.turns:
        turn = {
            "p": t.player,
            "kind": t.action.upper(),
            "word": t.word,
            "score": t.score,
            "row": t.row,
            "col": t.col,
            "h": t.horizontal,
            "tiles": t.tiles,
            "rack": t.rack,
        }
        if t.rating is not None:
            turn["rating"] = t.rating
        if t.best is not None:
            turn["best"] = _play(t.best)
        if t.equity is not None:
            turn["eq"] = {"played": t.equity.played, "best": t.equity.best}
            if t.equity.best_play is not None:
                turn["eq"]["play"] = _play(t.equity.best_play)
            if t.equity.best_exchange is not None:
                turn["eq"]["exchange"] = t.equity.best_exchange
        if t.endgame is not None:
            turn["eg"] = {"played": t.endgame.played, "best": t.endgame.best, "exact": t.endgame.exact}
            if t.endgame.best_play is not None:
                turn["eg"]["play"] = _play(t.endgame.best_play)
        turns.append(turn)
    return {
        "v": RECORD_VERSION,
        "id": session.session_id,
        "mode": "COMPETITIVE",
        "language": session.language,
        "started": int(session.started_at * 1000),
        "finished": int(time.time() * 1000),
        "players": [
            {"name": p.name, "computer": p.is_computer, "level": p.difficulty, "strength": p.strength, "score": p.score}
            for p in session.players
        ],
        "human": next(i for i, p in enumerate(session.players) if not p.is_computer),
        "winner": winner,
        "turns": turns,
        "cells": "".join("." if ch == "-" else ch for row in grid for ch in row),
        "blanks": [r * 15 + c for r in range(15) for c in range(15) if blanks[r][c]],
        "owners": "".join("." if o is None else str(o) for row in session.tile_owners for o in row),
        "hints": len(session.hint_turns),
        "undos": session.undos,
    }
