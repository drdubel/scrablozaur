"""The fixture the Rust game layer is checked against must still describe the
web app.

`cargo test` replays `tests/fixtures/web_parity.json` against `src/game/` and
`src/scan.rs`. That only proves anything while the fixture is what the Python
implementation does *now*: change `web/game.py`, a router, `src/rules.py`,
`src/strategy.py` or `smart_player`'s decision functions without regenerating
it, and the port would keep passing against a web app that no longer exists.
"""

import web_parity


def test_web_parity_fixture_is_current():
    fresh = web_parity.dumps(web_parity.build())
    committed = web_parity.FIXTURE.read_text(encoding="utf-8")
    assert fresh == committed, (
        "the web app's behaviour changed: run `uv run python tests/web_parity.py`, "
        "then make `cargo test` pass against the new fixture"
    )
