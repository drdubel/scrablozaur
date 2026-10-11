"""Percentage strength must monotonically narrow the selected engine's rank window."""

from web.difficulty import strength_rank_window


def test_strength_rank_windows_are_monotone_and_reach_the_endpoints():
    windows = [strength_rank_window(strength) for strength in range(1, 101)]
    assert windows[0] == (20, 40)
    assert windows[-1] == (1, 1)
    assert all(1 <= best <= worst <= 40 for best, worst in windows)
    assert all(b2 <= b1 and w2 <= w1 for (b1, w1), (b2, w2) in zip(windows, windows[1:]))
