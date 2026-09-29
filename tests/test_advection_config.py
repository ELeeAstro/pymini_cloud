"""Tests for YAML-driven settling-advection configuration."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pymini_cloud.io_read import AdvectionOptions, read_advection_options, read_setup
from pymini_cloud.main import advance_settling, run
from pymini_cloud.vert_advection import vert_advection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestAdvectionConfiguration(unittest.TestCase):
  """Validate configuration parsing and the main-code transport wrapper."""

  def test_example_setup_selects_koren_muscl(self) -> None:
    options = read_setup(SETUP_PATH).advection
    self.assertEqual(options.method, "muscl_ssprk3")
    self.assertEqual(options.limiter, "koren")
    self.assertEqual(options.cfl_limit, 0.8)
    self.assertEqual(options.top_boundary, "zero_inflow")
    self.assertEqual(options.bottom_boundary, "outflow")
    np.testing.assert_array_equal(options.top_values(3), np.zeros(3))

  def test_fixed_top_value_can_be_set_per_moment(self) -> None:
    options = read_advection_options(
      {
        "mini_cloud": {
          "vertical_advection": {
            "top_boundary": "fixed_value",
            "top_value": [1.0e-8, 2.0e-8],
            "bottom_boundary": "zero_flux",
          }
        }
      }
    )
    np.testing.assert_array_equal(
      options.top_values(2),
      np.array([1.0e-8, 2.0e-8]),
    )
    with self.assertRaisesRegex(ValueError, "expected 3, got 2"):
      options.top_values(3)

  def test_main_wrapper_passes_all_transport_options(self) -> None:
    atmosphere = run(SETUP_PATH, n_steps=0, write_output=False).atmosphere
    q = np.zeros((2, atmosphere.n_layers))
    velocity = np.vstack(
      (
        np.full(atmosphere.n_layers, 20.0),
        np.full(atmosphere.n_layers, 40.0),
      )
    )
    options = AdvectionOptions(
      limiter="mc",
      top_boundary="fixed_value",
      top_value=(1.0e-8, 2.0e-8),
      bottom_boundary="zero_flux",
    )
    result = advance_settling(atmosphere, q, velocity, 100.0, options)
    expected = vert_advection(
      q,
      velocity,
      100.0,
      atmosphere.rho,
      atmosphere.dz,
      q_top=np.array([1.0e-8, 2.0e-8]),
      limiter="mc",
      bottom_boundary="zero_flux",
    )
    np.testing.assert_array_equal(result, expected)
    self.assertTrue(np.all(result[:, 0] > 0.0))

  def test_unknown_option_is_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "unknown.*option"):
      read_advection_options(
        {"mini_cloud": {"vertical_advection": {"limter": "koren"}}}
      )


if __name__ == "__main__":
  unittest.main()
