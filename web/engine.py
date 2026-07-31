import sys
from pathlib import Path

from scrablozaur import Board, Dawg, Language

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from languages import LanguageSpec, engine_language, load  # noqa: E402

# The language every session currently plays in. A per-session choice replaces
# this constant once the API and UI carry one.
DEFAULT_LANGUAGE = "pl"

SPEC: LanguageSpec = load(DEFAULT_LANGUAGE)
LANG: Language = engine_language(SPEC)
DAWG_PATH = SPEC.dawg
# GADDAG for fast move generation; build with `cargo run --release -- build-gaddag`.
GADDAG_PATH = SPEC.gaddag

_dawg: Dawg | None = None


def get_dawg() -> Dawg:
    global _dawg
    if _dawg is None:
        _dawg = Dawg(LANG, str(DAWG_PATH), str(GADDAG_PATH))
    return _dawg


__all__ = [
    "Board",
    "Dawg",
    "Language",
    "get_dawg",
    "DAWG_PATH",
    "GADDAG_PATH",
    "LANG",
    "SPEC",
    "DEFAULT_LANGUAGE",
]
