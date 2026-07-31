"""Per-language dictionary registry.

A `Dawg` decodes into roughly 60-80 MB resident (larger than the file, since
each edge becomes a 12-byte struct), so loading one is worth doing once and
never again. Packs are built lazily on first use and kept for the process
lifetime -- deliberately without an LRU: sessions are process-local and never
expire, so evicting a dictionary mid-game would cost seconds on the next
request, and an in-flight request pins the object anyway.
"""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path

from scrablozaur import Board, Dawg, Language

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import languages  # noqa: E402
from languages import LanguageSpec  # noqa: E402

# Language a session gets when it does not ask for one. Kept as the historical
# default so existing links and clients that predate the picker still work.
DEFAULT_LANGUAGE = "pl"


@dataclass(frozen=True)
class LanguagePack:
    """Everything the app needs to run a game in one language."""

    spec: LanguageSpec
    lang: Language
    dawg: Dawg

    @property
    def code(self) -> str:
        return self.spec.code


_packs: dict[str, LanguagePack] = {}
# Two requests for the same unloaded language would otherwise each pay the load.
_lock = threading.Lock()


def available() -> list[str]:
    """Language codes with a definition file. Adding a language is dropping a
    JSON file in `languages/` -- nothing here knows the set in advance."""
    return languages.available()


def is_available(code: str) -> bool:
    return code in available()


def get_pack(code: str | None = None) -> LanguagePack:
    """The pack for `code`, loading it on first use.

    Raises `ValueError` for an unknown code, which the API layer turns into a
    400 rather than letting it surface as a 500.
    """
    code = code or DEFAULT_LANGUAGE
    pack = _packs.get(code)
    if pack is not None:
        return pack
    with _lock:
        # Another thread may have finished while this one waited.
        if code in _packs:
            return _packs[code]
        spec = languages.load(code)  # raises ValueError on an unknown code
        lang = languages.engine_language(spec)
        dawg = Dawg(lang, str(spec.dawg), str(spec.gaddag))
        _packs[code] = LanguagePack(spec=spec, lang=lang, dawg=dawg)
        return _packs[code]


def loaded_codes() -> list[str]:
    """Which languages are resident. Only useful for diagnostics."""
    return sorted(_packs)


__all__ = [
    "Board",
    "Dawg",
    "Language",
    "LanguagePack",
    "LanguageSpec",
    "DEFAULT_LANGUAGE",
    "available",
    "is_available",
    "get_pack",
    "loaded_codes",
]
