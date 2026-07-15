"""Shared thread pool for board_reader's independent per-tile/per-cell work
(glyph normalization, glyph scoring, parallax-shift candidates, template
matching) -- one pool reused across every photo and every call site,
instead of each optimization spinning up its own (pool creation has real
overhead, paid repeatedly across the 89-photo test set and every request in
production).

OpenCV's own C++ calls already fan out across every core by default
(confirmed on this project's dev machine: cv2.getNumThreads() == 18 out of
the box, via macOS's GCD backend). The obvious worry is that an *outer*
thread pool doing many small per-tile cv2 calls per worker would then
oversubscribe the same cores twice over -- measured directly (via
scripts/benchmark_pipeline.py) by pinning OpenCV to a single internal
thread and comparing: that made the *single*-threaded, full-image stages
(find_board_quad, rotate_board, detect_grid -- all real cv2.getNumThreads()
beneficiaries on 1000px+ images) 2-4x slower, for far less than that back
in the newly-threaded per-tile stages, whose patches (64-96px) are small
enough that OpenCV's own parallel_for doesn't bother splitting them across
cores anyway. Net effect of forcing single-threaded OpenCV was a *slower*
pipeline overall, not a faster one -- so it deliberately is NOT done here;
OpenCV keeps its own internal threading, and the pool below only adds
Python-level concurrency on top of it for the per-tile loops.
"""

import atexit
import os
from concurrent.futures import ThreadPoolExecutor

_executor = None


def get_executor():
    """Lazily create the process-wide pool on first use (same lazy-init
    convention as letter_classifier.py's model/template singletons)."""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        atexit.register(_executor.shutdown, wait=False)
    return _executor
