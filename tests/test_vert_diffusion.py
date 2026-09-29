"""Tests for implicit vertical diffusion."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pymini_cloud.main import run
from pymini_cloud.vert_diffusion import vert_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestVerticalDiffusion(unittest.TestCase):
  """Exercise diffusion from a fixed lower vapour boundary."""

  def test_bottom_vapour_source_diffuses_upward(self) -> None:
    atmosphere = run(SETUP_PATH, n_steps=0, write_output=False).atmosphere
    q_bottom = np.array([1.17e-7])
    q = np.full((1, atmosphere.n_layers), 1.0e-30)
    initial = q.copy()
    bottom_history = []
    column_history = []

    for _ in range(10):
      q = vert_diffusion(
        q,
        q_bottom,
        100.0,
        atmosphere.p_edge,
        atmosphere.T,
        atmosphere.mu,
        atmosphere.Kzz,
        atmosphere.rho,
        atmosphere.z,
        atmosphere.dz,
      )
      bottom_history.append(q[0, -1])
      column_history.append(np.sum(atmosphere.rho * atmosphere.dz * q[0]))

    self.assertTrue(np.array_equal(initial, np.full_like(initial, 1.0e-30)))
    self.assertTrue(np.all(np.isfinite(q)))
    self.assertTrue(np.all(q >= 1.0e-99))
    self.assertTrue(np.all(np.diff(bottom_history) > 0.0))
    self.assertTrue(np.all(np.diff(column_history) > 0.0))
    self.assertGreater(q[0, -2], 1.0e-30)
    self.assertLess(q[0, -1], q_bottom[0])
    active_layers = np.flatnonzero(q[0] > 10.0 * initial[0, 0])
    self.assertGreater(active_layers.size, 1)
    self.assertTrue(np.all(np.diff(q[0, active_layers[0]:]) > 0.0))


if __name__ == "__main__":
  unittest.main()
