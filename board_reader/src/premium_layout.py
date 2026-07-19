"""The standard Scrabble premium-square layout -- fixed board geometry, not
a detection parameter, so (unlike everything in hsv_config.json) none of
this is tunable.
"""

GRID = 15

# The layout has 4-fold rotational symmetry, so it is valid regardless of
# which side of the (already-normalized) board ends up "top".
#   T = triple word, D = double word, t = triple letter, d = double letter,
#   * = centre star (double word), . = plain square.
PREMIUM_LAYOUT = [
    "T..d...T...d..T",
    ".D...t...t...D.",
    "..D...d.d...D..",
    "d..D...d...D..d",
    "....D.....D....",
    ".t...t...t...t.",
    "..d...d.d...d..",
    "T..d...*...d..T",
    "..d...d.d...d..",
    ".t...t...t...t.",
    "....D.....D....",
    "d..D...d...D..d",
    "..D...d.d...D..",
    ".D...t...t...D.",
    "T..d...T...d..T",
]


def premium_class(row: int, col: int) -> str:
    """Premium class character for a square ('.', 'd', 't', 'D', 'T', '*')."""
    return PREMIUM_LAYOUT[row][col]
