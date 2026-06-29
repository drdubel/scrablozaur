from pathlib import Path

from scrablozaur import Board, Dawg

PROJECT_ROOT = Path(__file__).parent.parent
DAWG_PATH = PROJECT_ROOT / "words" / "dawg.bin"

_dawg: Dawg | None = None


def get_dawg() -> Dawg:
    global _dawg
    if _dawg is None:
        _dawg = Dawg(str(DAWG_PATH))
    return _dawg


__all__ = ["Board", "Dawg", "get_dawg"]
