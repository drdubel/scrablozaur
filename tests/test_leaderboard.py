"""The shared leaderboard (`web/leaderboard.py`): off unless the server names a
file for it, the web's own finished games recorded once, the app's uploads
checked and kept per device, and the standings scored as the app scores its
own (`core/Stats.kt`)."""

import json

import pytest
from fastapi.testclient import TestClient

from web import leaderboard
from web.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture
def board_file(tmp_path, monkeypatch):
    path = tmp_path / "leaderboard.jsonl"
    monkeypatch.setenv("SCRABLOZAUR_LEADERBOARD", str(path))
    monkeypatch.delenv("SCRABLOZAUR_LEADERBOARD_TOKEN", raising=False)
    return path


def _app_record(game_id, name="Ala", score=300, opponent_score=250, *, resigned=False, winner=None, ratings=()):
    """A record as the app's `GameRecord.toJson` writes it."""
    turns = [
        {"p": 0, "kind": "PLAY", "word": "kot", "score": 10, "row": 7, "col": 7, "h": True, "tiles": 3, "rack": "kotabcd",
         "rating": r}
        for r in ratings
    ]
    if resigned:
        turns.append({"p": 0, "kind": "RESIGN", "word": "", "score": 0, "row": 0, "col": 0, "h": True, "tiles": 0,
                      "rack": "abc"})
    if winner is None:
        winner = 0 if score > opponent_score else 1 if opponent_score > score else -1
    return {
        "v": 1,
        "id": game_id,
        "mode": "COMPETITIVE",
        "language": "pl",
        "started": 1_700_000_000_000,
        "finished": 1_700_000_600_000,
        "players": [
            {"name": name, "computer": False, "level": 5, "strength": 100, "score": score},
            {"name": "Komputer", "computer": True, "level": 8, "strength": 100, "score": opponent_score},
        ],
        "human": 0,
        "winner": winner,
        "turns": turns,
        "cells": "." * 225,
        "blanks": [],
        "owners": "." * 225,
        "hints": 0,
        "undos": 0,
    }


def _upload(client, games, source="phone-1", token=None):
    headers = {"X-Leaderboard-Token": token} if token else {}
    return client.post("/api/leaderboard/games", json={"source": source, "games": games}, headers=headers)


def test_the_leaderboard_is_off_unless_the_server_names_a_file(client, monkeypatch):
    monkeypatch.delenv("SCRABLOZAUR_LEADERBOARD", raising=False)
    assert client.get("/api/leaderboard").json()["enabled"] is False
    assert _upload(client, [_app_record("a")]).status_code == 404


def test_a_finished_web_game_is_recorded_once(client, board_file):
    new = {"players": [{"name": "Ola"}], "game_mode": "competitive", "difficulty": 3, "language": "pl"}
    assert client.post("/api/game/new", json=new).status_code == 200
    client.get("/api/board/hints")
    state = client.post("/api/board/pass").json()
    assert state["game_over"]
    lines = board_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["source"] == "web"
    record = entry["game"]
    assert leaderboard.valid_record(record)
    assert record["players"][0]["name"] == "Ola" and record["hints"] == 1
    assert [t["kind"] for t in record["turns"]][-1] == "RESIGN"
    # Asking for the state again changes nothing: the same ending is not kept twice.
    before = board_file.read_text(encoding="utf-8")
    client.get("/api/game/state")
    assert board_file.read_text(encoding="utf-8") == before
    # A resignation is a loss, whatever the score.
    ola = next(p for p in client.get("/api/leaderboard").json()["players"] if p["name"] == "Ola")
    assert (ola["games"], ola["losses"]) == (1, 1)


def test_uploads_are_checked_and_replace_a_devices_earlier_ones(client, board_file, monkeypatch):
    monkeypatch.setenv("SCRABLOZAUR_LEADERBOARD_TOKEN", "sekret")
    games = [_app_record("g1", ratings=(80, 60)), _app_record("g2", score=200), {"v": 1, "id": "broken"}]
    assert _upload(client, games).status_code == 403
    assert _upload(client, games, token="nie").status_code == 403
    assert _upload(client, games, token="sekret").json() == {"received": 3, "kept": 2}
    # The same games again, one now with a better score: replaced, not added.
    games[1] = _app_record("g2", score=400)
    assert _upload(client, games[:2], token="sekret").json()["kept"] == 2
    # Another device's game of the same id is its own.
    assert _upload(client, [_app_record("g1")], source="phone-2", token="sekret").json()["kept"] == 1
    board = client.get("/api/leaderboard", params={"language": "pl"}).json()
    assert board["games"] == 3
    ala = next(p for p in board["players"] if p["name"] == "Ala")
    assert (ala["games"], ala["wins"], ala["best_score"]) == (3, 3, 400)
    assert ala["accuracy"] == 70
    assert board["top_scores"][0]["score"] == 400
    assert client.get("/api/leaderboard", params={"language": "en"}).json()["games"] == 0


def test_standings_score_resignations_and_ties_as_the_app_does():
    won_but_resigned = _app_record("r", score=300, opponent_score=100, resigned=True, winner=0)
    tie = _app_record("t", score=250, opponent_score=250)
    loss = _app_record("l", score=100, opponent_score=250)
    assert leaderboard.result(won_but_resigned, 0) == "loss"
    assert leaderboard.result(tie, 0) == "tie"
    assert leaderboard.result(loss, 0) == "loss"
    assert leaderboard.accuracy(_app_record("a", ratings=(100, 50)), 0) == 75
    assert leaderboard.accuracy(loss, 0) is None
    # Only what the standings read has to be there and well-formed.
    assert not leaderboard.valid_record({**tie, "players": [tie["players"][1], tie["players"][1]]})
    assert not leaderboard.valid_record({**tie, "v": 2})
