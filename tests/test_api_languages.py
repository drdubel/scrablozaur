"""End-to-end checks of the language wiring, through the real app.

The first FastAPI `TestClient` tests in the repo, and deliberately narrow: the
routing/session/serialisation layer between a language code arriving in a
request and the right dictionary being used is exactly the part unit tests
cannot see.
"""

import pytest
from fastapi.testclient import TestClient

import languages
from web.main import app


@pytest.fixture(scope="module")
def client():
    # `with` runs the lifespan, which warms the default language's dictionary.
    with TestClient(app) as c:
        yield c


def _new_game(client, language=None, mode="sandbox", difficulty=5):
    body = {
        "players": [
            {"name": "Ja", "is_computer": False},
            {"name": "Bot", "is_computer": True, "difficulty": difficulty},
        ],
        "game_mode": mode,
        "difficulty": difficulty,
    }
    if language is not None:
        body["language"] = language
    return client.post("/api/game/new", json=body)


# ── The language list ────────────────────────────────────────────────────────


def test_languages_endpoint_lists_every_definition_file(client):
    data = client.get("/api/game/languages").json()
    assert {lang["code"] for lang in data["languages"]} == set(languages.available())
    assert data["default"] in languages.available()


def test_languages_endpoint_serves_the_tables_the_client_needs(client):
    """`board.js` used to carry its own copy of the point values. It now takes
    them from here, so this payload is what keeps the rendered tiles honest."""
    data = client.get("/api/game/languages").json()
    for info in data["languages"]:
        spec = languages.load(info["code"])
        assert info["letter_values"] == spec.points
        assert info["tile_counts"] == spec.counts
        assert info["alphabet"] == spec.alphabet
        assert info["total_tiles"] == spec.total_tiles == 100


def test_languages_endpoint_does_not_load_the_dictionaries(client):
    """Building a dropdown must not pay ~60-80 MB per language. Only the
    default should be resident, warmed by the app's own startup."""
    from web.engine import DEFAULT_LANGUAGE, loaded_codes

    client.get("/api/game/languages")
    assert loaded_codes() == [DEFAULT_LANGUAGE] or DEFAULT_LANGUAGE in loaded_codes()


# ── Starting a game in a language ────────────────────────────────────────────


def test_new_game_defaults_to_the_default_language(client):
    state = _new_game(client).json()
    assert state["language"] == languages.load("pl").code


def test_new_game_honours_an_explicit_language(client):
    for code in languages.available():
        state = _new_game(client, language=code).json()
        assert state["language"] == code, f"asked for {code}"


def test_the_session_remembers_its_language(client):
    for code in languages.available():
        _new_game(client, language=code)
        assert client.get("/api/game/state").json()["language"] == code


def test_an_unknown_language_is_rejected(client):
    response = _new_game(client, language="xx")
    assert response.status_code == 400
    assert "xx" in response.json()["detail"]


# ── The dictionary actually follows the language ─────────────────────────────


def test_a_word_is_judged_by_its_own_language(client):
    """The point of the whole exercise: the lexicon consulted must be the one
    the game was started in, not whichever happened to be loaded first."""
    codes = languages.available()
    if len(codes) < 2:
        pytest.skip("needs two languages installed")

    # A word each language has and the other cannot (different alphabets).
    probes = {"pl": "kot", "en": "quiz"}
    for code in codes:
        word = probes.get(code)
        if word is None:
            continue
        _new_game(client, language=code)
        ok = client.get(f"/api/board/definition/{word}")
        assert ok.status_code == 200, f"{word} under {code}"


def test_placing_a_word_uses_the_session_language(client):
    _new_game(client, language="pl")
    response = client.post(
        "/api/board/human-move",
        json={"word": "kot", "row": 7, "col": 7, "horizontal": True},
    )
    assert response.status_code == 200, response.text
    state = response.json()
    assert state["language"] == "pl"
    assert state["players"][0]["score"] > 0


# ── Difficulty follows the language ──────────────────────────────────────────


def test_difficulty_levels_stop_where_a_language_has_no_leave_net(client):
    """Levels 9-10 need a trained rack-leave evaluator. A language without one
    serves a shorter list, and the slider honours it without knowing why."""
    for code in languages.available():
        spec = languages.load(code)
        data = client.get(f"/api/game/difficulty-levels?language={code}").json()
        expected = 10 if spec.leave_net is not None else 8
        assert data["max_level"] == expected, code
        assert [lvl["level"] for lvl in data["levels"]] == list(range(1, expected + 1))


def test_leave_net_orderings_work_where_a_net_exists(client):
    """`smart` and `sim` both consult the learned leave evaluator."""
    for code in languages.available():
        spec = languages.load(code)
        if spec.leave_net is None:
            continue
        _new_game(client, language=code)
        client.post("/api/board/set-letters", json={"letters": "aeiorst"})
        for sort in ("smart", "sim"):
            response = client.post(f"/api/board/suggest?sort={sort}")
            assert response.status_code == 200, f"{code}/{sort}: {response.text}"


def test_leave_net_orderings_are_refused_without_a_net(client, monkeypatch):
    """Fed another language's rack, a net does not fail -- it drops the letters
    its own alphabet lacks and scores the rest against the wrong distribution.
    So the orderings are refused outright rather than answered with nonsense.

    Every installed language currently has a net, which would make this
    vacuous, so the refusal path is driven from a spec with `leave_net=None`
    rather than from whichever language happens to lack one.
    """
    import dataclasses

    from web import engine, game

    code = languages.available()[0]
    _new_game(client, language=code)
    client.post("/api/board/set-letters", json={"letters": "aeiorst"})

    real = engine.get_pack(code)
    stripped = dataclasses.replace(real, spec=dataclasses.replace(real.spec, leave_net=None))
    monkeypatch.setitem(engine._packs, code, stripped)
    # `has_leave_net` reads through the same registry the routes do.
    assert not game.has_leave_net(code)

    for sort in ("smart", "sim"):
        response = client.post(f"/api/board/suggest?sort={sort}")
        assert response.status_code == 400, f"{sort} should be refused: {response.text}"
        assert "modelu" in response.json()["detail"]

    # Plain score order must remain available regardless.
    assert client.post("/api/board/suggest?sort=score").status_code == 200


def test_score_ordering_always_works(client):
    """Whatever a language lacks, plain score order must still be available."""
    for code in languages.available():
        _new_game(client, language=code)
        client.post("/api/board/set-letters", json={"letters": "aeiorst"})
        response = client.post("/api/board/suggest?sort=score")
        assert response.status_code == 200, f"{code}: {response.text}"
        assert response.json()["suggestions"], code


def test_a_level_above_a_languages_ceiling_is_clamped(client):
    for code in languages.available():
        spec = languages.load(code)
        state = _new_game(client, language=code, mode="competitive", difficulty=10).json()
        bot = next(p for p in state["players"] if p["is_computer"])
        assert bot["difficulty"] <= (10 if spec.leave_net is not None else 8), code
