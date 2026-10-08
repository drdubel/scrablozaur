"""Classify a normalised glyph's letter -- CNN primary, template matching
as a confidence-gated fallback, weighted-vote fusion between them.

Ported from ocr/scrabble_reader/letter_classifier.py + template_matcher.py +
board_builder.py's fuse_predictions(), adapted to this project's house
style: no dataclass Config, tunables read from hsv_config.json via
_params()/PARAM_DEFAULTS like every other stage in this codebase. Combined
into one file (rather than kept as three, like ocr/) because these sources
are always invoked together and share one param preset -- the same
reasoning tile_detector.py already applies to its own several sub-models.

Model/template loading is expensive (a ~5MB checkpoint + device transfer,
building several hundred template images) and must happen once per process,
not once per call -- eval_letters.py alone calls into this ~90 times. Since
this codebase has no persistent instances for callers to hold across calls
(everything is free functions), that one-time setup is memoised behind
lazily-initialised module-level singletons instead.
"""

import glob
import os
import sys
import unicodedata

import cv2
import numpy as np
from glyph_normalizer import DIGIT_SIZE, GLYPH_SIZE
from hsv_config import load_params

try:
    import torch
    import torch.nn as nn

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

try:
    from PIL import Image, ImageDraw, ImageFont

    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# ---------------------------------------------------------------------------
# Active language
#
# Which letters exist, what each is worth, and where this language's models
# live all come from `languages/<code>.json` (see src/languages.py) -- the same
# file the engine reads. These used to be hardcoded Polish tables that had to
# be kept in step with src/lib.rs by hand.
#
# The language is process-global rather than a per-call argument because every
# expensive thing here is a lazily-built module-level singleton (see the module
# docstring); the caches below are keyed by language code so switching back and
# forth costs nothing after the first use of each.

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_PKG_DIR, "..", "..", "src"))

import languages  # noqa: E402

DEFAULT_LANGUAGE = "pl"
_spec = languages.load(DEFAULT_LANGUAGE)


def set_language(spec):
    """Point the classifier at a language, by code or `LanguageSpec`."""
    global _spec
    _spec = spec if hasattr(spec, "code") else languages.load(spec)
    return _spec


def language():
    return _spec


def alphabet():
    """The letters this language's tiles can show, uppercase -- tiles are
    printed in capitals, while the engine works in lowercase."""
    return _spec.alphabet.upper()


def letter_points():
    """{LETTER: points}, blank excluded (a blank prints no digit)."""
    return {ch.upper(): pts for ch, pts in _spec.points.items() if ch != _spec.blank}


def point_groups():
    """{points: "LETTERS"} -- the inverse of letter_points().

    Every tile prints its point value in a corner, and several letters share
    each value, so the digit alone rarely identifies a letter. It is a strong
    *exclusionary* signal for the confusions this classifier actually makes:
    in Polish, A/Ą and Z/Ź/Ż differ in points as well as in a diacritic.
    """
    groups = {}
    for ch, pts in letter_points().items():
        groups.setdefault(pts, []).append(ch)
    return {pts: "".join(sorted(chars)) for pts, chars in groups.items()}


def valid_digits():
    """The point values that can actually be printed on a tile of this set.

    Training or matching the digit reader against values that never appear
    would only spend model capacity and confidence mass on impossible answers.

    A tuple rather than a concatenated string, because a value above 9 is two
    characters -- English's 10 -- and joining them would silently produce the
    classes "1" and "0". The single-glyph digit reader cannot represent such a
    value at all, which is why those languages turn the channel off entirely
    (see `use_point_prior()`).
    """
    return tuple(str(p) for p in sorted(point_groups()))


def use_point_prior():
    """Whether the printed point digit is worth reading for this language.

    It earns its keep where it separates letters the shape-based sources
    confuse -- Polish's diacritic pairs. In an alphabet without diacritics it
    buys much less, and where a value needs two glyphs the reader cannot read
    it at all.
    """
    return _spec.ocr is not None and _spec.ocr.use_point_prior


def _ocr_paths():
    if _spec.ocr is None:
        raise ValueError(
            f"language '{_spec.code}' has no OCR models configured "
            f"(\"ocr\" is null in languages/{_spec.code}.json)"
        )
    return _spec.ocr


def has_models():
    """Whether this language has a board scanner at all."""
    return _spec.ocr is not None and os.path.isfile(str(_spec.ocr.letter_cnn))
# Candidate font files for synthetic templates -- first ones found are used.
TEMPLATE_FONTS = (
    "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia Bold.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)

PARAM_DEFAULTS = {
    "weight_cnn": 1.0,  # fusion weight for the CNN's vote
    "weight_template": 0.75,  # fusion weight for the template matcher's vote
    "weight_points": 1.0,  # fusion weight for the tile's own printed point-value digit (see points_distribution())
    "template_trigger_confidence": 0.85,  # template matcher only runs if the CNN's own top prob is below this
    "reprocess_confidence": 0.50,  # fused confidence below this triggers a re-binarize retry (read_letters.py)
    "resolve_rotation": 1,  # bool-as-0/1 (cv2 trackbars are int-only, same convention as every other slider)
}


def _params(overrides=None):
    """Merge hsv_config.json's saved "letter_recognition_params" preset
    with any explicit overrides -- same convention as every other stage."""
    merged = load_params("letter_recognition_params", PARAM_DEFAULTS)
    if overrides:
        merged.update({k: v for k, v in overrides.items() if v is not None})
    return merged


# ---------------------------------------------------------------------------
# CNN (primary source)

if _HAS_TORCH:

    class LetterCNN(nn.Module):
        """4 conv blocks + global average pooling; ~1M parameters."""

        def __init__(self, n_classes):
            super().__init__()

            def block(cin, cout):
                return nn.Sequential(
                    nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                    nn.BatchNorm2d(cout),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )

            self.features = nn.Sequential(block(1, 32), block(32, 64), block(64, 128), block(128, 256))
            self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3), nn.Linear(256, n_classes))

        def forward(self, x):
            return self.head(self.features(x))


# Keyed by language code: loading a ~5 MB checkpoint and moving it to the
# device is expensive, and a process that scans two languages should pay it
# once each rather than on every switch.
_cnn_cache: dict = {}
_cnn_device = "cpu"


def _check_classes(code, classes, kind):
    """Refuse a checkpoint trained for a different alphabet.

    Without this a Polish model loaded under English produces confident
    predictions drawn entirely from the wrong letter set -- garbage that looks
    like a working scanner. Mirrors the guard `smart_player/model.py` applies
    to the leave net.
    """
    expected = set(alphabet()) if kind == "letter" else set(valid_digits())
    got = set(classes)
    if not got <= expected:
        raise ValueError(
            f"{kind} model for '{code}' predicts {sorted(got - expected)!r}, "
            f"which are not in that language -- is this checkpoint from another language?"
        )


def _get_cnn():
    """Lazily load the active language's CNN checkpoint. Returns (model,
    classes) or (None, []) if torch or the weights file is unavailable --
    callers degrade gracefully rather than crash."""
    global _cnn_device
    code = _spec.code
    if code in _cnn_cache:
        return _cnn_cache[code]
    weights = str(_spec.ocr.letter_cnn) if _spec.ocr else ""
    if not (_HAS_TORCH and weights and os.path.isfile(weights)):
        _cnn_cache[code] = (None, [])
        return _cnn_cache[code]
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    _check_classes(code, classes, "letter")
    model = LetterCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    # Apple-GPU inference roughly halves the per-batch latency; tiny
    # transfer sizes (N x 64 x 64) make the copies negligible.
    _cnn_device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(_cnn_device)
    _cnn_cache[code] = (model, classes)
    return _cnn_cache[code]


def classify_cnn_batch(glyphs):
    """Per-glyph {letter: probability}; None for unavailable/blank glyphs."""
    model, classes = _get_cnn()
    if model is None:
        return [None] * len(glyphs)
    idx = [i for i, g in enumerate(glyphs) if g.has_glyph]
    out = [None] * len(glyphs)
    if not idx:
        return out
    batch = np.stack([glyphs[i].gray for i in idx]).astype(np.float32)
    batch = (255.0 - batch) / 255.0  # ink -> 1, background -> 0
    x = torch.from_numpy(batch).unsqueeze(1).to(_cnn_device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
    for row, i in enumerate(idx):
        letters = alphabet()
        out[i] = {ch: float(p) for ch, p in zip(classes, probs[row]) if ch in letters}
    return out


# ---------------------------------------------------------------------------
# Template matching (confidence-gated fallback)
#
# Two template sources: synthetic (every alphabet letter rendered with
# several fonts, missing font files skipped) and real (glyph crops
# harvested from labelled photos of actual tiles, dominate the score when
# available since they capture the exact tile font). Matching is a Dice
# overlap of binary ink masks over a small translation search; a sharpened
# softmax turns Dice scores into a confidence distribution.


def normalize_template(mask, out_size=GLYPH_SIZE, margin=0.12):
    """Centre/scale a binary ink mask exactly like glyph_normalizer._compose."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    crop = mask[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    box = int(out_size * (1 - 2 * margin))
    h, w = crop.shape
    s = box / max(h, w)
    rm = cv2.resize(crop, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((out_size, out_size), np.uint8)
    oy, ox = (out_size - rm.shape[0]) // 2, (out_size - rm.shape[1]) // 2
    canvas[oy : oy + rm.shape[0], ox : ox + rm.shape[1]] = (rm > 127) * 255
    return canvas


# Keyed by language code, like the model caches -- building several hundred
# template images is the other expensive one-time cost in this module.
_templates_cache: dict = {}  # code -> {letter: [(mask, is_real), ...]}


def _build_synthetic(templates):
    if not _HAS_PIL:
        return
    fonts = [p for p in TEMPLATE_FONTS if os.path.isfile(p)]
    for path in fonts:
        try:
            font = ImageFont.truetype(path, 160)
        except OSError:
            continue
        for ch in alphabet():
            img = Image.new("L", (260, 260), 0)
            d = ImageDraw.Draw(img)
            d.text((130, 130), ch, fill=255, font=font, anchor="mm")
            mask = np.asarray(img)
            for scale_y in (1.0, 1.15):  # tile fonts are often condensed
                m = (
                    mask
                    if scale_y == 1.0
                    else cv2.resize(mask, None, fx=1.0, fy=scale_y, interpolation=cv2.INTER_LINEAR)
                )
                t = normalize_template((m > 127).astype(np.uint8) * 255)
                if t is not None and t.any():
                    templates[ch].append((t, False))


def _load_real(templates):
    real_dir = str(_spec.ocr.real_templates) if _spec.ocr and _spec.ocr.real_templates else ""
    if not real_dir or not os.path.isdir(real_dir):
        return
    for letter_dir in sorted(glob.glob(os.path.join(real_dir, "*"))):
        ch = unicodedata.normalize("NFC", os.path.basename(letter_dir))
        if ch not in templates:
            continue
        files = sorted(glob.glob(os.path.join(letter_dir, "*.png")))
        # Cap per-letter real templates: beyond ~a dozen they only add
        # matching cost, not discrimination. Spread the selection across
        # the directory to keep source diversity.
        if len(files) > 12:
            step = len(files) / 12.0
            files = [files[int(i * step)] for i in range(12)]
        for f in files:
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            t = normalize_template((img > 127).astype(np.uint8) * 255)
            if t is not None and t.any():
                templates[ch].append((t, True))


def _get_templates():
    """Lazily build the active language's synthetic+real template set."""
    code = _spec.code
    if code not in _templates_cache:
        templates = {ch: [] for ch in alphabet()}
        _build_synthetic(templates)
        _load_real(templates)
        _templates_cache[code] = templates
    return _templates_cache[code]


def _pair_score(glyph_mask, tmpl):
    """Best Dice overlap over +-3 px shifts.

    Both operands went through the same size normalisation, so scale
    search adds cost, not discrimination -- a small shift search is all
    that residual centring differences need.
    """
    g = (glyph_mask > 127).astype(np.float32)
    t = (tmpl > 127).astype(np.float32)
    th, tw = t.shape
    gh, gw = g.shape
    pad = 3
    canvas = np.zeros((gh + 2 * pad, gw + 2 * pad), np.float32)
    oy, ox = (canvas.shape[0] - th) // 2, (canvas.shape[1] - tw) // 2
    if oy < 0 or ox < 0:
        return 0.0
    canvas[oy : oy + th, ox : ox + tw] = t
    # Correlation of binary images relates linearly to intersection;
    # matchTemplate slides the glyph for the best alignment.
    res = cv2.matchTemplate(canvas, g, cv2.TM_CCORR)
    inter = float(res.max())
    return 2.0 * inter / (float(g.sum()) + float(t.sum()) + 1e-9)


def classify_templates(glyph):
    """Return {letter: confidence} or None when matching is unavailable."""
    templates = _get_templates()
    if not glyph.has_glyph or not any(templates.values()):
        return None
    raw = {}
    for ch, tmpls in templates.items():
        if not tmpls:
            continue
        score = 0.0
        for t, is_real in tmpls:
            s = _pair_score(glyph.mask, t)
            if is_real:
                s = min(1.0, s * 1.10)  # trust real-tile templates more
            score = max(score, s)
        raw[ch] = score
    if not raw:
        return None
    # Sharpened softmax turns Dice scores into a confidence distribution.
    letters = list(raw)
    v = np.array([raw[c] for c in letters])
    e = np.exp((v - v.max()) * 14.0)
    p = e / e.sum()
    return {c: float(pi) for c, pi in zip(letters, p)}


# ---------------------------------------------------------------------------
# Score-digit reading -- reads the tile's own printed point value to help
# disambiguate the letter. Same CNN-primary / template-fallback shape as
# letter classification above: a first cut using pure Dice template
# matching (classify_digit_templates()) measured only 82.9% accurate even
# on cleanly-extracted crops (all 10 digits' Dice scores bunch within
# ~0.15 of each other -- simple blocky digit shapes just don't separate
# well from a handful of synthetic templates alone, the same reason
# letters needed a CNN rather than template matching alone). A small CNN
# (identical architecture to LetterCNN, just 10 classes) fixes that;
# template matching stays as the fallback for when the digit CNN weights
# are missing or torch isn't available.

_digit_cnn_cache: dict = {}
_digit_cnn_device = "cpu"


def _get_digit_cnn():
    """Lazily load the active language's digit CNN. Returns (model, classes)
    or (None, []) if unavailable or if this language does not use the printed
    point value at all -- mirrors _get_cnn()."""
    global _digit_cnn_device
    code = _spec.code
    if code in _digit_cnn_cache:
        return _digit_cnn_cache[code]
    weights = str(_spec.ocr.digit_cnn) if _spec.ocr and _spec.ocr.digit_cnn else ""
    if not (use_point_prior() and _HAS_TORCH and weights and os.path.isfile(weights)):
        _digit_cnn_cache[code] = (None, [])
        return _digit_cnn_cache[code]
    ckpt = torch.load(weights, map_location="cpu", weights_only=True)
    classes = list(ckpt["classes"])
    _check_classes(code, classes, "digit")
    model = LetterCNN(len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _digit_cnn_device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(_digit_cnn_device)
    _digit_cnn_cache[code] = (model, classes)
    return _digit_cnn_cache[code]


def classify_digit_cnn_batch(glyphs):
    """Per-glyph {digit_str: probability}; None for glyphs with no isolated
    digit crop or when the digit CNN is unavailable. Mirrors
    classify_cnn_batch()'s batching shape exactly."""
    model, classes = _get_digit_cnn()
    if model is None:
        return [None] * len(glyphs)
    idx = [i for i, g in enumerate(glyphs) if g.digit_gray is not None]
    out = [None] * len(glyphs)
    if not idx:
        return out
    batch = np.stack([glyphs[i].digit_gray for i in idx]).astype(np.float32)
    batch = (255.0 - batch) / 255.0
    x = torch.from_numpy(batch).unsqueeze(1).to(_digit_cnn_device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()
    for row, i in enumerate(idx):
        out[i] = {d: float(p) for d, p in zip(classes, probs[row])}
    return out


_digit_templates_cache: dict = {}  # code -> {digit_str: [mask, ...]}


def _get_digit_templates():
    """Lazily build the active language's synthetic digit templates (only the
    point values its tiles can actually print)."""
    code = _spec.code
    if code in _digit_templates_cache:
        return _digit_templates_cache[code]
    # A value needing two glyphs cannot be matched by a single-glyph template,
    # so those languages get no digit templates at all.
    templates = {d: [] for d in valid_digits() if len(d) == 1} if use_point_prior() else {}
    if _HAS_PIL:
        fonts = [p for p in TEMPLATE_FONTS if os.path.isfile(p)]
        for path in fonts:
            try:
                font = ImageFont.truetype(path, 80)
            except OSError:
                continue
            for d in templates:
                img = Image.new("L", (140, 140), 0)
                ImageDraw.Draw(img).text((70, 70), d, fill=255, font=font, anchor="mm")
                mask = np.asarray(img)
                t = normalize_template((mask > 127).astype(np.uint8) * 255, out_size=DIGIT_SIZE)
                if t is not None and t.any():
                    templates[d].append(t)
    _digit_templates_cache[code] = templates
    return templates


def classify_digit_templates(glyph):
    """Return {digit_str: confidence} from the tile's isolated score-digit
    crop (glyph_normalizer.py's digit_mask) via Dice template matching, or
    None if no digit was found or matching is unavailable. Fallback for
    when the digit CNN isn't available -- see classify_digit_cnn_batch()."""
    if glyph.digit_mask is None:
        return None
    templates = _get_digit_templates()
    if not any(templates.values()):
        return None
    raw = {}
    for d, tmpls in templates.items():
        if not tmpls:
            continue
        raw[d] = max(_pair_score(glyph.digit_mask, t) for t in tmpls)
    if not raw:
        return None
    digits = list(raw)
    v = np.array([raw[d] for d in digits])
    e = np.exp((v - v.max()) * 14.0)
    p = e / e.sum()
    return {d: float(pi) for d, pi in zip(digits, p)}


def points_distribution(digit_dist):
    """Turn a {digit_str: confidence} reading into a {letter: confidence}
    distribution for fuse_predictions(): each point value's probability
    mass is split evenly across every letter carrying that value (e.g. a
    confident "9" reading concentrates almost entirely on Z-acute, the only
    9-point letter; a confident "1" reading only rules out the 2/3/5/6/7/9
    point letters, since nine different letters share 1 point). Returns
    None if the digit reading names no known point value (0 = blank tile,
    or a misread digit outside the real 1-9 range this game uses)."""
    out = {}
    for digit_str, p in digit_dist.items():
        letters = point_groups().get(int(digit_str), "")
        if not letters:
            continue
        share = p / len(letters)
        for ch in letters:
            out[ch] = out.get(ch, 0.0) + share
    return out or None


# ---------------------------------------------------------------------------
# Fusion


def fuse_predictions(preds):
    """Combine {source: (distribution, weight)} into a single ranking.

    Returns (letter, confidence, ranked alternatives, per-source winners).
    Confidence blends the fused posterior with cross-source agreement.
    """
    votes = {}
    tops = {}
    total_w = 0.0
    for name, (dist, w) in preds.items():
        if not dist:
            continue
        total_w += w
        top = max(dist, key=dist.get)
        tops[name] = top
        for ch, p in dist.items():
            votes[ch] = votes.get(ch, 0.0) + w * p
    if not votes or total_w == 0.0:
        return None, 0.0, [], tops

    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    norm = sum(votes.values())
    best, second = ranked[0][1] / norm, (ranked[1][1] / norm if len(ranked) > 1 else 0.0)

    n_sources = len(tops)
    agree = sum(1 for t in tops.values() if t == ranked[0][0])
    agreement = agree / n_sources if n_sources else 0.0

    margin = best - second
    confidence = float(np.clip(0.55 * best + 0.25 * agreement + 0.20 * min(1.0, margin * 3.0), 0, 1))
    alternatives = [(ch, v / norm) for ch, v in ranked[:5]]
    return ranked[0][0], confidence, alternatives, tops
