# Dictionary builds. These were prose in the README, which meant the GADDAG's
# ~7 GB peak and the DAWG-before-GADDAG ordering had to be remembered rather
# than encoded.
#
#   make dicts LANG=pl     rebuild both binaries for one language
#   make verify LANG=pl    move-for-move parity against the legacy generator
#   make bench LANG=pl     generation speed
#
# Both binaries must be rebuilt together after a format change: they carry a
# header stamping the lexicon's own alphabet, and the engine refuses a file
# whose letters the language does not have.

LANG ?= pl
CARGO ?= cargo run --release -q --
WORDS := words/$(LANG)/words.txt
DAWG  := words/$(LANG)/dawg.bin
GADDAG := words/$(LANG)/gaddag.bin
GAMES ?= 200

.PHONY: dicts dawg gaddag verify verify-all bench test

dicts: dawg gaddag

dawg: $(DAWG)
$(DAWG): $(WORDS)
	$(CARGO) build $(WORDS) $(DAWG)

# Peaks around 7 GB for a 2.5M-word lexicon; far less for a smaller one.
gaddag: $(GADDAG)
$(GADDAG): $(WORDS)
	$(CARGO) build-gaddag $(WORDS) $(GADDAG)

verify: $(DAWG) $(GADDAG)
	$(CARGO) gen-verify $(LANG) $(DAWG) $(GADDAG) $(GAMES)

bench: $(DAWG) $(GADDAG)
	$(CARGO) gen-bench $(LANG) $(DAWG) $(GADDAG) $(GAMES)

# Every installed language, from the definition files themselves -- adding one
# should not require editing this.
LANGS := $(basename $(notdir $(wildcard languages/*.json)))

test:
	cargo test --release
	uv run pytest tests/
	@for l in $(LANGS); do \
		echo "--- verify_engine: $$l ---"; \
		uv run python src/verify_engine.py --lang $$l || exit 1; \
	done

# Move-for-move parity in every language. The alphabet and point table are
# runtime data, so a change can be correct for one language and wrong for another.
verify-all:
	@for l in $(LANGS); do $(MAKE) --no-print-directory verify LANG=$$l || exit 1; done
