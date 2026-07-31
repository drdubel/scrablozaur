# English word list

| | |
|---|---|
| **File** | `words.txt` |
| **Words** | 168,551 |
| **sha256** | `9e26bdd69fd492b4f62ecd11ab179874f01db712493f49312e459f8f51ac6f81` |
| **Source** | ENABLE (Enhanced North American Benchmark Lexicon), Alan Beale, 1997 |
| **Obtained from** | <https://raw.githubusercontent.com/dolph/dictionary/master/enable1.txt> (172,823 words) |
| **Licence** | **Public domain** — Beale released ENABLE explicitly into the public domain |

## Why ENABLE

It is a genuine word-game list, not a spellcheck dump: no proper nouns, no
abbreviations, no hyphens or apostrophes. The alternatives were worse:

- **TWL/OWL (NASPA)** and **Collins/SOWPODS** are the lists tournaments actually
  use, but both are proprietary and neither may be redistributed. Not an option
  for a committed file.
- **`dwyl/english-words`** is MIT-licensed but is a spellcheck list — it carries
  abbreviations, junk strings and invalid inflections, which make for worse
  gameplay than a smaller, curated list.

## Filter applied

```bash
grep -xE '[a-z]{2,15}' enable1.txt | sort -u > words.txt
```

This dropped 4,272 entries: single letters (unplayable — a move is at least two
tiles) and words longer than the 15-square board.

## Invariants

Matching `words/pl/words.txt`:

- one word per line, UTF-8, **lowercase**, byte-sorted, deduplicated
- exactly the 26 letters of `languages/en.json`'s alphabet

`cargo run --release -- build` stamps the distinct letters it finds into the
`.bin` header, so a list violating the last point yields a dictionary the `en`
language then refuses to load.

## Rebuilding

```bash
make dicts LANG=en
```

The English GADDAG is ~8 MiB and builds in under a second — nothing like the
Polish one's ~7 GB peak, because the lexicon is ~15× smaller.
