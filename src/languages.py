"""Language definitions: the single source of truth for a language's alphabet,
tile distribution and point values.

Every one of those three tables used to exist in several hand-maintained
copies -- in `src/lib.rs` (twice for the alphabet), `web/game.py`,
`web/static/js/board.js`, `board_reader/src/letter_classifier.py`,
`src/strategy.py` and `smart_player/model.py`. They agreed only because
somebody kept checking. Now they are read from `languages/<code>.json`, and
`tests/test_tables_agree.py` fails if any consumer drifts from it.

The engine's own limits are mirrored here as MAX_ALPHABET / MAX_CODEPOINT so a
language file that the Rust side could not represent is rejected when it loads,
with a message naming the offending letter, rather than at some later point
deep inside move generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES_DIR = _ROOT / "languages"

# Cross-check sets are a u32 with one bit per letter (src/lib.rs's `letter_bit`),
# so 32 letters is a hard ceiling, not a tunable.
MAX_ALPHABET = 32
# Letter lookups are direct codepoint indexes into a fixed-size table
# (src/lib.rs's FREQ_SIZE), so every letter must fall below it. This is what
# restricts languages to the Latin script -- Cyrillic and Greek start at U+0370.
MAX_CODEPOINT = 400

__all__ = [
    "LanguageSpec",
    "LeaveNetPaths",
    "OcrPaths",
    "MAX_ALPHABET",
    "MAX_CODEPOINT",
    "available",
    "load",
    "load_all",
]


@dataclass(frozen=True)
class LeaveNetPaths:
    """Where a language's learned rack-leave evaluator lives. `None` on a
    language that has no trained net yet, which caps it at difficulty 8."""

    checkpoint: Path
    weights: Path


@dataclass(frozen=True)
class OcrPaths:
    """Where a language's board-photo models live.

    `digit_cnn` and `real_templates` are optional. The digit reader turns the
    point value printed on a tile into a letter prior, which is worth a lot in
    Polish (it separates A/Ą and Z/Ź/Ż) and little in an alphabet with no
    diacritics -- and cannot represent a two-glyph value like English's 10 at
    all. `real_templates` holds glyphs harvested from photos of real tiles; a
    language trained on rendered fonts alone simply has none.
    """

    letter_cnn: Path
    digit_cnn: Path | None
    real_templates: Path | None
    use_point_prior: bool


@dataclass(frozen=True)
class LanguageSpec:
    code: str
    name: str
    flag: str
    #: Letters in collation order. This order is the language's own, and it is
    #: load-bearing twice: it fixes the cross-check bit indexes, and it decides
    #: `first_draw_winner`'s "closest to A" tiebreak.
    alphabet: str
    blank: str
    counts: dict[str, int]
    points: dict[str, int]
    vowels: str
    consonants: str
    words: Path
    dawg: Path
    gaddag: Path
    leave_net: LeaveNetPaths | None
    ocr: OcrPaths | None
    definitions: tuple[str, ...]

    @property
    def bag(self) -> list[str]:
        """The full tile distribution, letters in collation order followed by
        the blanks -- byte-for-byte the order `fresh_tile_bag()` produced when
        it was a Rust literal. Boards are seeded by shuffling this, so a change
        in order silently changes every seeded deal."""
        tiles = [ch for ch in self.alphabet for _ in range(self.counts[ch])]
        tiles.extend(self.blank * self.counts[self.blank])
        return tiles

    @property
    def eval_alphabet(self) -> list[str]:
        """Tile types in codepoint order, blank first -- the feature order of
        the leave-value net. Derived from the bag rather than declared, because
        a net whose input order disagrees with the caller's does not fail, it
        just returns confident nonsense."""
        return sorted(set(self.bag))

    @property
    def total_tiles(self) -> int:
        return sum(self.counts.values())

    @property
    def has_ocr(self) -> bool:
        """Whether the board scanner can actually run for this language.

        The config block existing is not enough -- the checkpoint has to be on
        disk, or the UI would offer a scan tab that fails on first use.
        """
        return self.ocr is not None and self.ocr.letter_cnn.exists()

    @property
    def ocr_is_experimental(self) -> bool:
        """Whether this language's scanner has ever been checked against photos.

        Derived, not declared: `real_templates` holds glyphs harvested from
        photographs of actual tiles, so a language without them was trained on
        rendered fonts alone and its accuracy on real boards is unmeasured.
        """
        return self.has_ocr and self.ocr is not None and self.ocr.real_templates is None

    def missing_artifacts(self) -> list[Path]:
        """Declared files that are not on disk. The dictionary is committed, but
        OCR templates are harvested locally and a language may ship before its
        leave net is trained, so this reports rather than raises."""
        paths = [self.words, self.dawg, self.gaddag]
        if self.leave_net is not None:
            paths += [self.leave_net.checkpoint, self.leave_net.weights]
        if self.ocr is not None:
            paths += [p for p in (self.ocr.letter_cnn, self.ocr.digit_cnn) if p is not None]
        return [p for p in paths if not p.exists()]


def _require(cond: bool, code: str, message: str) -> None:
    if not cond:
        raise ValueError(f"languages/{code}.json: {message}")


def _parse(code: str, raw: dict) -> LanguageSpec:
    missing = {"code", "name", "alphabet", "blank", "tiles", "vowels", "consonants"} - raw.keys()
    _require(not missing, code, f"missing required key(s): {', '.join(sorted(missing))}")
    _require(raw["code"] == code, code, f"declares code {raw['code']!r} but is named {code}.json")

    alphabet: str = raw["alphabet"]
    blank: str = raw["blank"]
    letters = set(alphabet)

    _require(len(blank) == 1, code, f"blank must be a single character, got {blank!r}")
    _require(len(letters) == len(alphabet), code, "alphabet contains a duplicate letter")
    _require(
        len(alphabet) <= MAX_ALPHABET,
        code,
        f"alphabet has {len(alphabet)} letters; the engine's cross-check bitset holds {MAX_ALPHABET}",
    )
    _require(blank not in letters, code, f"blank {blank!r} must not also be an alphabet letter")
    over = [ch for ch in alphabet + blank if ord(ch) >= MAX_CODEPOINT]
    _require(
        not over,
        code,
        f"letter(s) {over!r} are above U+{MAX_CODEPOINT:04X}; the engine indexes letters by codepoint",
    )
    _require(alphabet == alphabet.lower(), code, "alphabet must be lowercase; the engine works in lowercase")

    tiles: dict[str, dict] = raw["tiles"]
    expected = letters | {blank}
    _require(
        tiles.keys() == expected,
        code,
        "tiles must have exactly one entry per alphabet letter plus the blank; "
        f"missing {sorted(expected - tiles.keys())!r}, unexpected {sorted(tiles.keys() - expected)!r}",
    )
    for ch, tile in tiles.items():
        _require({"count", "points"} <= tile.keys(), code, f"tile {ch!r} needs both 'count' and 'points'")
        _require(tile["count"] >= 1, code, f"tile {ch!r} has count {tile['count']}; every tile type needs at least one")
        _require(tile["points"] >= 0, code, f"tile {ch!r} has negative points")
    _require(tiles[blank]["points"] == 0, code, "the blank must be worth 0 points")

    vowels, consonants = raw["vowels"], raw["consonants"]
    _require(
        set(vowels) | set(consonants) == letters,
        code,
        "vowels + consonants must cover the alphabet exactly; "
        f"uncovered {sorted(letters - set(vowels) - set(consonants))!r}, "
        f"unknown {sorted((set(vowels) | set(consonants)) - letters)!r}",
    )
    _require(
        not (set(vowels) & set(consonants)),
        code,
        f"letter(s) {sorted(set(vowels) & set(consonants))!r} are listed as both vowel and consonant",
    )

    leave_net = raw.get("leave_net")
    ocr = raw.get("ocr")
    return LanguageSpec(
        code=code,
        name=raw["name"],
        flag=raw.get("flag", ""),
        alphabet=alphabet,
        blank=blank,
        counts={ch: t["count"] for ch, t in tiles.items()},
        points={ch: t["points"] for ch, t in tiles.items()},
        vowels=vowels,
        consonants=consonants,
        words=_ROOT / raw["words"],
        dawg=_ROOT / raw["dawg"],
        gaddag=_ROOT / raw["gaddag"],
        leave_net=None
        if leave_net is None
        else LeaveNetPaths(_ROOT / leave_net["checkpoint"], _ROOT / leave_net["weights"]),
        ocr=None
        if ocr is None
        else OcrPaths(
            _ROOT / ocr["letter_cnn"],
            _ROOT / ocr["digit_cnn"] if ocr.get("digit_cnn") else None,
            _ROOT / ocr["real_templates"] if ocr.get("real_templates") else None,
            bool(ocr.get("use_point_prior", True)),
        ),
        definitions=tuple(raw.get("definitions", ())),
    )


_cache: dict[str, LanguageSpec] = {}


def available() -> list[str]:
    """Language codes with a definition file, alphabetically. Adding a language
    is dropping a file in `languages/` -- no code knows the set in advance."""
    return sorted(p.stem for p in LANGUAGES_DIR.glob("*.json"))


def load(code: str) -> LanguageSpec:
    """Parse and validate `languages/<code>.json`. Cached: the result is
    immutable and several packages load the same language independently."""
    if code not in _cache:
        path = LANGUAGES_DIR / f"{code}.json"
        if not path.exists():
            raise ValueError(f"unknown language {code!r}; available: {', '.join(available()) or 'none'}")
        _cache[code] = _parse(code, json.loads(path.read_text(encoding="utf-8")))
    return _cache[code]


def load_all() -> list[LanguageSpec]:
    return [load(code) for code in available()]


def engine_language(code: str | LanguageSpec):
    """The engine-side `scrablozaur.Language` for a definition.

    Imported lazily so this module stays usable without the compiled extension
    (`board_reader` wants the tables, not the engine). The engine interns
    definitions, so two calls return the same underlying object either way --
    the cache here just avoids rebuilding the argument dicts.
    """
    from scrablozaur import Language

    spec = code if isinstance(code, LanguageSpec) else load(code)
    if spec.code not in _engine_cache:
        _engine_cache[spec.code] = Language(
            spec.code, spec.alphabet, spec.counts, spec.points, spec.blank
        )
    return _engine_cache[spec.code]


_engine_cache: dict = {}
