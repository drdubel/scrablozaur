# Polish word list

| | |
|---|---|
| **File** | `words.txt` |
| **Words** | 2,584,336 |
| **sha256** | `ff70a8807fc98c3db579b5832da74c4f188674650204111afa80f935848b0737` |
| **Source** | Combined from multiple sources (close to osps)  |
| **Scraper** | `_scraper.py` at the repo root — git-ignored by the `_*` rule, so it is not part of the tree |

## Invariants

The build pipeline and the engine both rely on these:

- one word per line, UTF-8, **lowercase**
- byte-sorted (`sort_unstable` in `cmd_build` re-sorts anyway, but the DAWG
  builder's incremental minimisation requires sorted input)
- exactly the 32 letters of `languages/pl.json`'s alphabet — no `q`, `v` or `x`,
  which do not appear on the Polish tile set

`cargo run --release -- build` stamps the distinct letters it actually finds into
the `.bin` header, so a word list that violates the last point produces a
dictionary the `pl` language then refuses to load.

## Rebuilding

```bash
make dicts LANG=pl
```

Both binaries must be rebuilt together — see the note in the `Makefile`.
