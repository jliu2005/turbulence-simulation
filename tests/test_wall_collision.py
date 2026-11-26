"""Checks for the matched vortex-wall experiment."""

import numpy as np

from simulations.run_2d_wall_collision import matched_pair, snapshots


def test_initial_velocity_and_vorticity_match_through_the_walls():
    free, rigid = matched_pair(65)
    assert free.walls["top"].kind == "free_slip"
    assert rigid.walls["top"].kind == "no_slip"
    for side in ("bottom", "left", "right"):
        assert free.walls[side] == rigid.walls[side]
    assert free.nu == rigid.nu
    np.testing.assert_allclose(free.velocity(), rigid.velocity(), atol=1e-12)
    omega = free.vorticity()
    np.testing.assert_allclose(omega, rigid.vorticity(), atol=1e-12)
    np.testing.assert_allclose(omega, -omega[::-1, :], atol=1e-11)


def test_pair_reaches_top_and_separates_farther_with_free_slip():
    n = 65
    for t, free, rigid in snapshots(n):
        if t >= 0.9:
            break

    def left_core(field):
        # Locate the original positive core on the left, away from the wall nodes.
        window = field[1 : n // 2, 1:-2]
        i, j = np.unravel_index(window.argmax(), window.shape)
        return (np.array([i, j]) + 1) / (n - 1)

    x_free, y_free = left_core(free)
    x_rigid, y_rigid = left_core(rigid)
    assert min(y_free, y_rigid) > 0.8
    assert x_free + 0.05 < x_rigid
    assert y_free > y_rigid
    np.testing.assert_array_equal(free[:, -1], 0)
    assert np.max(np.abs(rigid[:, -1])) > 1
