"""Where board_reader keeps its per-language data and models on disk.

One module, because these directories are handed between five scripts in
sequence -- harvest -> review -> clean -> generate dataset -> train -- and a
disagreement between any two of them is *silent*: the producer writes somewhere
the consumer never looks, so the workflow appears to run and simply has no
effect. That is exactly what happened when the artifacts first moved under a
per-language directory and the scripts kept their own copies of the old paths.

The language definition stays authoritative for the artifacts it declares
(`real_templates`, `letter_cnn`, `digit_cnn` in `languages/<code>.json`); the
working directories around them -- staging, rejected, pre-cleanup backup -- are
derived here, since nothing outside this pipeline needs to know about them.

Layout:

    board_reader/src/data/<code>/real_templates/            (declared)
                                real_templates_staging/
                                real_templates_rejected/
                                real_templates_before_cleanup/
                                real_digit_templates/
                                real_digit_templates_staging/
                                real_digit_templates_rejected/
    board_reader/src/data_train/<code>/                     letter CNN training set
    board_reader/src/data_train_digits/<code>/              digit CNN training set
    board_reader/src/models/<code>/                         (declared)
"""

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SRC_DIR, "..", "..", "src"))

import languages  # noqa: E402

__all__ = [
    "spec_for",
    "data_root",
    "letter_dirs",
    "digit_dirs",
    "train_dir",
    "digit_train_dir",
    "letter_cnn_path",
    "digit_cnn_path",
]


def spec_for(code):
    """Accepts a code or an already-loaded `LanguageSpec`."""
    return code if hasattr(code, "code") else languages.load(code)


def data_root(code) -> str:
    return os.path.join(SRC_DIR, "data", spec_for(code).code)


def letter_dirs(code) -> dict[str, str]:
    """The four letter-glyph directories: accepted (the one the classifier
    actually reads), plus staging, rejected and the pre-cleanup backup."""
    spec = spec_for(code)
    root = data_root(spec)
    accepted = (
        str(spec.ocr.real_templates)
        if spec.ocr is not None and spec.ocr.real_templates is not None
        else os.path.join(root, "real_templates")
    )
    return {
        "accepted": accepted,
        "staging": os.path.join(root, "real_templates_staging"),
        "rejected": os.path.join(root, "real_templates_rejected"),
        "backup": os.path.join(root, "real_templates_before_cleanup"),
    }


def digit_dirs(code) -> dict[str, str]:
    """The point-value digit equivalents. Only meaningful for a language whose
    tiles print a readable value -- see `letter_classifier.use_point_prior`."""
    root = data_root(code)
    return {
        "accepted": os.path.join(root, "real_digit_templates"),
        "staging": os.path.join(root, "real_digit_templates_staging"),
        "rejected": os.path.join(root, "real_digit_templates_rejected"),
    }


def train_dir(code) -> str:
    return os.path.join(SRC_DIR, "data_train", spec_for(code).code)


def digit_train_dir(code) -> str:
    return os.path.join(SRC_DIR, "data_train_digits", spec_for(code).code)


def letter_cnn_path(code) -> str:
    spec = spec_for(code)
    if spec.ocr is not None:
        return str(spec.ocr.letter_cnn)
    return os.path.join(SRC_DIR, "models", spec.code, "letter_cnn.pt")


def digit_cnn_path(code) -> str:
    """Where the digit model belongs. Taken from the language file when it
    declares one, so a retrain lands exactly where the classifier will load it
    from -- writing elsewhere is a retrain that silently never takes effect."""
    spec = spec_for(code)
    if spec.ocr is not None and spec.ocr.digit_cnn is not None:
        return str(spec.ocr.digit_cnn)
    return os.path.join(SRC_DIR, "models", spec.code, "digit_cnn.pt")
