"""Browser smoke test: drive the real UI and fail on any console error.

The only layer that catches a JavaScript `ReferenceError`. Two shipped bugs
proved the point -- deleting a global from `board.js` broke the rack renderer
and the scan grid, and every server-side test stayed green because the API
returned 200 throughout. A user hit both.

Deliberately shallow. It asserts that each view *renders*, not that the game is
correct -- correctness is `tests/test_web_agrees_with_cli.py`'s job, and a
browser test that checks scores would be slow and flaky for no gain.

Skipped unless Playwright and a Chromium build are both present, so it never
blocks a normal `pytest` run:

    uv run playwright install chromium
"""

import contextlib
import os
import socket
import subprocess
import sys
import time

import pytest

pytest.importorskip("playwright", reason="pip install playwright")
from playwright.sync_api import sync_playwright  # noqa: E402

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# A Polish fixture; the scan pipeline is Polish-only in practice (see
# board_reader/README.md), so this exercises the default language.
_SCAN_IMAGE = os.path.join(_ROOT, "board_reader", "test", "in", "img11_e.jpg")


def _free_port() -> int:
    with contextlib.closing(socket.socket()) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def server():
    """The real app on its own port, so the test is self-contained."""
    port = _free_port()
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "web.main:app", "--port", str(port)],
        cwd=_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}/"
    try:
        # Startup loads a dictionary eagerly, so allow a generous window.
        for _ in range(120):
            if proc.poll() is not None:
                pytest.fail("uvicorn exited during startup")
            with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.5):
                break
            time.sleep(0.5)
        else:
            pytest.fail("uvicorn did not start")
        yield url
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def page(server):
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - any launch failure means "not installed"
            pytest.skip(f"chromium unavailable: run `playwright install chromium` ({exc})")
        pg = browser.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        pg.on(
            "console",
            lambda m: errors.append(f"console.{m.type}: {m.text}") if m.type == "error" else None,
        )
        pg.goto(server)
        pg.wait_for_timeout(2000)
        # The first load probes /api/game/state with no cookie and gets an
        # expected 401; anything after this point is a real error.
        errors.clear()
        pg.console_errors = errors  # type: ignore[attr-defined]
        yield pg
        browser.close()


def _errors(page) -> list[str]:
    return page.console_errors  # type: ignore[attr-defined]


def _open_setup(page):
    """Ensure the setup dialog is open, whether or not it already was.

    Without this, one failing test leaves the dialog open and every later click
    is intercepted by it -- turning a single failure into a cascade that hides
    which test actually broke.
    """
    if page.locator("#dialog-setup[open]").count() == 0:
        page.click("#btn-new-game")
    page.wait_for_selector("#dialog-setup[open]", timeout=15000)


def _start_game(page, mode="competitive"):
    """Start a game and wait for the board to actually be dealt.

    Waits on the condition, not a fixed delay: starting a competitive game can
    include the bot's opening move, whose duration depends on its level and on
    how loaded the machine is. A sleep long enough on an idle laptop is a flaky
    test on a busy one.
    """
    _open_setup(page)
    page.click(f'.mode-card[data-mode="{mode}"]')
    page.wait_for_selector("#btn-start-game:visible", timeout=15000)
    page.click("#btn-start-game")
    page.wait_for_selector("#tile-rack .rack-tile", timeout=60000)


def test_competitive_renders_a_rack_with_point_values(page):
    """The exact bug that shipped: `_renderTileRack` threw, so competitive mode
    looked broken. Only this mode renders a human rack from the bag."""
    _start_game(page)
    # A full rack is seven tiles; wait for the last one rather than assuming
    # they all render in the same frame.
    page.wait_for_function(
        "document.querySelectorAll('#tile-rack .rack-tile').length === 7", timeout=30000
    )
    _assert_strongest_bots(page)
    assert not _errors(page), _errors(page)
    # The point values are what the removed global used to supply. A blank
    # renders as a star with no value, so compare against the non-blank tiles
    # rather than assuming all seven carry a number.
    tiles = page.locator("#tile-rack .rack-tile").count()
    blanks = page.locator("#tile-rack .rack-tile.blank").count()
    assert page.locator("#tile-rack .tile-val").count() == tiles - blanks


def test_switching_language_still_renders(page):
    _open_setup(page)
    picker = page.locator("#setup-language")
    if not picker.count():
        pytest.skip("only one language installed")
    picker.select_option("en")
    # The highest difficulty must follow the language immediately.
    page.wait_for_function(
        "typeof Languages !== 'undefined' && Languages.loaded && Difficulty.loaded",
        timeout=30000,
    )
    page.click("#btn-start-game")
    page.wait_for_selector("#dialog-setup", state="hidden", timeout=60000)
    page.wait_for_function(
        "document.querySelectorAll('#tile-rack .rack-tile').length === 7", timeout=60000
    )
    _assert_strongest_bots(page)
    assert not _errors(page), _errors(page)


def _assert_strongest_bots(page):
    state = page.evaluate("async () => (await fetch('/api/game/state')).json()")
    maximum = page.evaluate("code => Languages.maxLevel(code)", state["language"])
    bots = [p for p in state["players"] if p["is_computer"]]
    assert bots
    assert all(p["difficulty"] == maximum for p in bots)
    assert page.locator("#setup-strength").count() == 1
    assert not page.locator("#setup-strength").is_visible()


@pytest.mark.parametrize("submode", ["manual", "auto"])
def test_sandbox_modes_use_strongest_bots(page, submode):
    _open_setup(page)
    page.click('.mode-card[data-mode="sandbox"]')
    page.click(f'.sub-toggle-btn[data-sandbox-sub="{submode}"]')
    page.select_option("#setup-count", "4")
    page.click("#btn-start-game")
    page.wait_for_selector("#dialog-setup", state="hidden", timeout=60000)
    assert page.locator("#board .cell").count() == 225
    _assert_strongest_bots(page)
    if submode == "auto":
        page.wait_for_selector("#panel-auto:visible")
        page.click("#btn-auto-next")
        page.wait_for_selector("#auto-move-log li", timeout=60000)
    assert not _errors(page), _errors(page)


def test_advanced_options_change_strength_and_bot_type(page):
    for strength, neural, simulation, level in [(1, False, False, 8), (55, True, False, 9), (37, True, True, 10)]:
        _open_setup(page)
        assert not page.locator("#setup-strength").is_visible()
        page.click('.mode-card[data-mode="sandbox"]')
        page.click('.sub-toggle-btn[data-sandbox-sub="auto"]')
        page.select_option("#setup-count", "2")
        page.click("#setup-advanced summary")
        page.locator("#setup-strength").evaluate(
            "(el, value) => { el.value = value; el.dispatchEvent(new Event('input', {bubbles: true})); }",
            str(strength),
        )
        page.locator("#setup-smart").set_checked(neural)
        if neural:
            page.locator("#setup-simulation").set_checked(simulation)
        else:
            assert page.locator("#setup-simulation").is_disabled()
        assert page.locator("#setup-strength-value").inner_text() == f"{strength}%"
        page.click("#btn-start-game")
        page.wait_for_selector("#dialog-setup", state="hidden")
        state = page.evaluate("async () => (await fetch('/api/game/state')).json()")
        assert all(p["strength"] == strength and p["difficulty"] == level for p in state["players"])
        page.click("#btn-auto-next")
        page.wait_for_selector("#auto-move-log li", timeout=60000)
        assert not _errors(page), _errors(page)
    _open_setup(page)
    assert not page.locator("#setup-strength").is_visible()
    assert page.locator("#setup-strength").input_value() == "37"
    # A complete benchmark must receive the same options as a live game.
    page.fill("#setup-benchmark-games", "1")
    with page.expect_request("**/api/benchmark/start") as request:
        page.click("#btn-run-benchmark")
    assert all(p["strength"] == 37 and p["difficulty"] == 10 for p in request.value.post_data_json["players"])
    page.wait_for_selector("#benchmark-results:visible", timeout=120000)
    assert not _errors(page), _errors(page)


def test_benchmark_customizes_each_player_independently(page):
    _open_setup(page)
    page.click('.mode-card[data-mode="sandbox"]')
    page.click('.sub-toggle-btn[data-sandbox-sub="auto"]')
    page.select_option("#setup-count", "3")
    rows = page.locator(".setup-auto-player")
    expected = [(25, False, False, 8), (70, True, False, 9), (100, True, True, 10)]
    for i, (strength, neural, simulation, _level) in enumerate(expected):
        row = rows.nth(i)
        assert not row.locator(".player-strength").is_visible()
        row.locator("input[type=text]").fill(f"Bot {i + 1}")
        row.locator("summary").click()
        row.locator(".player-strength").evaluate(
            "(el, value) => { el.value = value; el.dispatchEvent(new Event('input', {bubbles: true})); }",
            str(strength),
        )
        row.locator(".player-smart").set_checked(neural)
        if neural:
            row.locator(".player-simulation").set_checked(simulation)
        else:
            assert row.locator(".player-simulation").is_disabled()
    # Rebuilding rows must retain each player's independent settings and name.
    page.select_option("#setup-count", "4")
    page.select_option("#setup-count", "3")
    for i, (strength, neural, simulation, _level) in enumerate(expected):
        row = rows.nth(i)
        assert row.locator("input[type=text]").input_value() == f"Bot {i + 1}"
        assert row.locator(".player-strength").input_value() == str(strength)
        assert row.locator(".player-smart").is_checked() == neural
        assert row.locator(".player-simulation").is_checked() == (simulation if neural else True)
    page.fill("#setup-benchmark-games", "1")
    with page.expect_request("**/api/benchmark/start") as request:
        page.click("#btn-run-benchmark")
    players = request.value.post_data_json["players"]
    assert [(p["strength"], p["difficulty"]) for p in players] == [(s, level) for s, _, _, level in expected]
    page.wait_for_selector("#benchmark-results:visible", timeout=120000)
    labels = page.locator(".benchmark-stat-diff").all_text_contents()
    assert len(labels) == 3
    assert all(f"{strength}%" in labels[i] for i, (strength, _, _, _) in enumerate(expected))
    _open_setup(page)
    assert rows.nth(0).locator(".player-strength").input_value() == "25"
    assert not rows.nth(0).locator(".player-strength").is_visible()
    rows.nth(0).locator("summary").click()
    rows.nth(0).locator(".player-defaults").click()
    assert rows.nth(0).locator(".player-strength").input_value() == page.locator("#setup-strength").input_value()
    assert rows.nth(1).locator(".player-strength").input_value() == "70"
    assert not _errors(page), _errors(page)


def test_scan_renders_a_board(page):
    """The other shipped bug: the scan grid read the same removed global.

    Starts a Polish game first, deliberately. A scan with no scan-session of
    its own borrows the running game's language (`web/routers/scan.py`'s
    `_scan_language`), and the fixture is a photo of a Polish board -- left in
    the English game the previous test starts, this would read it with the
    English models. That is correct behaviour, not a bug, but it makes the
    fixture the wrong input.
    """
    if not os.path.isfile(_SCAN_IMAGE):
        pytest.skip("board_reader/test fixtures are not committed")

    # Start from a clean slate: dropping the session cookie and reloading means
    # this test does not inherit the previous one's language or view.
    page.context.clear_cookies()
    page.reload()
    page.wait_for_function(
        "typeof Languages !== 'undefined' && Languages.loaded", timeout=30000
    )
    _errors(page).clear()  # the fresh load re-probes /game/state and gets its 401

    _open_setup(page)
    # The scan card is bound by ScanController and opens its view directly --
    # there is no start button in that flow.
    page.click('.mode-card[data-mode="scan"]')
    # The file input is `hidden` behind a styled button, so wait for it to be
    # attached rather than visible -- `set_input_files` works on it either way.
    page.wait_for_selector("#scan-file-input", state="attached", timeout=15000)
    page.set_input_files("#scan-file-input", _SCAN_IMAGE)
    # Reading a photo runs the whole CV pipeline; wait for a rendered tile
    # rather than guessing how long that takes.
    page.wait_for_selector("#scan-board .cell.placed", timeout=180000)
    assert not _errors(page), _errors(page)
