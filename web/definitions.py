"""Word-definition lookup, one provider set per language.

These scrapers used to live in `web/routers/board.py`, which made a router the
home of several hundred lines of HTML parsing. The route now just picks the
providers its language declares (`"definitions"` in `languages/<code>.json`) and
tries them in order -- structurally the same as the old `sjp or pwn` fallback,
but with the order coming from data rather than being spelled into the handler.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Callable

#: A provider returns `(lemma, definition)` pairs, or an empty list if it has
#: nothing -- including when it is simply unreachable. A definition is a nicety;
#: no lookup should ever fail a request.
Provider = Callable[[str], list[tuple[str, str]]]


_SKIP_P = re.compile(
    r"^(SŁOWNIK SJP|KOMENTARZE|PROSIMY|POWIĄZANE|dopuszczal|niedopuszczal|function |-$|\(brak\)|dodaj$|OK$|nazwisko$|imię$)",
    re.IGNORECASE,
)
_HTML_TAG = re.compile(r"<[^>]+>")
_ENTITIES = {"&quot;": '"', "&amp;": "&", "&nbsp;": " ", "&lt;": "<", "&gt;": ">"}
_SENSE_BREAK = re.compile(r"\s*;?\s*(?=\d{1,2}\.\s)")


def _clean_html(s: str) -> str:
    # A space (not "") for stripped tags -- source markup uses bare `<br />`
    # between numbered senses with no surrounding whitespace, so dropping tags
    # outright glues the tail of one sense to the next one's digit, e.g.
    # "...kotwica<br />5. ..." would collapse into "...kotwica5. ...".
    s = _HTML_TAG.sub(" ", s)
    for ent, ch in _ENTITIES.items():
        s = s.replace(ent, ch)
    return " ".join(s.split())


def _break_senses(text: str) -> str:
    """Put each numbered sense ("1. ...", "2. ...") on its own line.

    Source dictionaries separate senses with `;` or a lone `<br>` in the raw
    HTML, which `_clean_html` flattens to plain whitespace -- this restores
    readable line breaks without touching unnumbered single-sense entries.
    """
    parts = [p for p in _SENSE_BREAK.split(text) if p]
    return "<br>".join(parts)


def _fetch_sjp(word: str) -> list[tuple[str, str]]:
    """Fetch entries from sjp.pl for *word*.

    Returns list of (lemma, definition) tuples — sjp.pl transparently maps
    inflected forms to their base form, so "zwie" yields ("zwać", "...") etc.
    """
    url = f"https://sjp.pl/{urllib.parse.quote(word)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scrablozaur/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8")
    except Exception:
        return []

    raw = [
        _clean_html(p)
        for p in re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL)
    ]
    # Stop at comments section — everything after it is user noise
    paragraphs = []
    for p in raw:
        if re.match(r"^KOMENTARZE", p, re.IGNORECASE) or re.match(r"^POWIĄZANE", p, re.IGNORECASE):
            break
        if p and not _SKIP_P.search(p):
            paragraphs.append(p)

    entries: list[tuple[str, str]] = []
    i = 0
    while i < len(paragraphs):
        t = paragraphs[i]
        is_lemma = (
            not t.startswith("znaczenie")
            and len(t) < 60
            and not re.search(r"\d\.", t)
        )
        if is_lemma:
            # Skip proper nouns (capitalized lemmas not matching the searched word)
            if t[0].isupper() and t.lower() != word:
                i += 1
                continue
            # Skip the "znaczenie: info (N)" line if present
            j = i + 1
            if j < len(paragraphs) and paragraphs[j].startswith("znaczenie"):
                j += 1
            if j < len(paragraphs):
                defn = paragraphs[j]
                if not _SKIP_P.search(defn):
                    entries.append((t, defn))
                    i = j + 1
                    continue
        i += 1
    return entries[:3]


_PWN_HEADWORD = re.compile(
    r'<span class="tytul"><a[^>]*title="([^"]*)"[^>]*>.*?</a></span>.*?<li[^>]*>(.*?)</li>',
    re.DOTALL,
)
_PWN_SENSE = re.compile(r'<div class="znacz">(.*?)</div>', re.DOTALL)


def _fetch_pwn(word: str) -> list[tuple[str, str]]:
    """Fetch entries from sjp.pwn.pl for *word* -- fallback for words sjp.pl has no entry for.

    A word absent from PWN's dictionary responds with a 307 that has no
    `Location` header (a same-URL "not found" render, not a real redirect),
    which urllib surfaces as an HTTPError -- treated the same as any other
    fetch failure, i.e. no entries.
    """
    url = f"https://sjp.pwn.pl/slowniki/{urllib.parse.quote(word)}.html"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scrablozaur/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode("utf-8")
    except Exception:
        return []

    entries: list[tuple[str, str]] = []
    for m in _PWN_HEADWORD.finditer(html):
        lemma = _clean_html(m.group(1))
        li = m.group(2)
        senses = _PWN_SENSE.findall(li)
        if senses:
            # Leave line-breaking to _break_senses (applied uniformly in
            # get_definition) instead of inserting <br> here directly.
            defn = " ".join(_clean_html(s) for s in senses)
        else:
            body = re.split(r'<br|<div class="s-przykl"', li)[0]
            defn = _clean_html(body)
        if defn:
            entries.append((lemma, defn))
    return entries[:3]


# ── English ──────────────────────────────────────────────────────────────────


def _fetch_json(url: str):
    """Shared fetch for the JSON providers. Any failure -- offline, 404, rate
    limit, malformed body -- is simply "no definition", never an error the
    caller has to handle."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "scrablozaur/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8", "ignore"))
    except Exception:
        return None


def _fetch_wiktionary(word: str) -> list[tuple[str, str]]:
    """Wikimedia's own REST endpoint. Best coverage of the obscure words a
    tournament word list turns up, which is exactly where a player most wants
    to know whether something is real.

    The response is keyed by language code; `en` is the English section. Its
    definitions contain HTML, so they go through the same `_clean_html` the
    Polish scrapers use. Content is CC-BY-SA -- attributed in the UI.
    """
    quoted = urllib.parse.quote(word)
    data = _fetch_json(f"https://en.wiktionary.org/api/rest_v1/page/definition/{quoted}")
    if not isinstance(data, dict):
        return []

    entries: list[tuple[str, str]] = []
    for section in data.get("en", []):
        part = _clean_html(section.get("partOfSpeech", ""))
        for item in section.get("definitions", []):
            defn = _clean_html(item.get("definition", ""))
            if not defn:
                continue
            entries.append((part.lower() if part else word, defn))
            if len(entries) >= 3:
                return entries
    return entries


def _fetch_dictionaryapi(word: str) -> list[tuple[str, str]]:
    """dictionaryapi.dev -- free and clean, but a hobby service with no uptime
    guarantee and thin coverage of rare words, so it sits behind Wiktionary."""
    quoted = urllib.parse.quote(word)
    data = _fetch_json(f"https://api.dictionaryapi.dev/api/v2/entries/en/{quoted}")
    if not isinstance(data, list):
        return []

    entries: list[tuple[str, str]] = []
    for entry in data:
        for meaning in entry.get("meanings", []):
            part = meaning.get("partOfSpeech", "")
            for item in meaning.get("definitions", []):
                defn = (item.get("definition") or "").strip()
                if not defn:
                    continue
                entries.append((part or word, defn))
                if len(entries) >= 3:
                    return entries
    return entries


PROVIDERS: dict[str, Provider] = {
    "sjp": _fetch_sjp,
    "pwn": _fetch_pwn,
    "wiktionary": _fetch_wiktionary,
    "dictionaryapi": _fetch_dictionaryapi,
}


def lookup(word: str, providers: tuple[str, ...]) -> list[str]:
    """Definitions of `word`, formatted for display. First provider with
    anything to say wins; an empty list means nobody knew it.

    Formatting lives here rather than in the route so a provider's raw
    `(lemma, definition)` shape stays this module's business.
    """
    for name in providers:
        provider = PROVIDERS.get(name)
        if provider is None:
            continue
        entries = provider(word)
        if entries:
            return [
                # A lemma only earns a prefix when it differs from what was
                # asked -- dictionaries map inflected forms to a base form, and
                # "zwie -> zwać" is worth showing while "kot -> kot" is noise.
                _break_senses(defn) if lemma.lower() == word else f"{lemma} — {_break_senses(defn)}"
                for lemma, defn in entries
            ]
    return []
