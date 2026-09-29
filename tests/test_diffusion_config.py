"""Tests for YAML-driven vertical-diffusion configuration."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pymini_cloud.io_read import (
  DiffusionOptions,
  read_diffusion_options,
  read_setup,
)
from pymini_cloud.main import advance_diffusion, run
from pymini_cloud.vert_diffusion import vert_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestDiffusionConfiguration(unittest.TestCase):
  """Validate configuration parsing and the main-code diffusion wrapper."""

  def test_example_setup_uses_species_bottom_value(self) -> None:
    options = read_setup(SETUP_PATH).diffusion
    self.assertEqual(options.theta, 0.5)
    self.assertEqual(options.top_boundary, "zero_flux")
    self.assertEqual(options.bottom_boundary, "fixed_value")
    self.assertEqual(options.bottom_value, "from_species")
    np.testing.assert_array_equal(options.top_flux_values(2), np.zeros(2))
    np.testing.assert_array_equal(
      options.bottom_values(2, np.array([1.17e-7, 0.0])),
      np.array([1.17e-7, 0.0]),
    )

  def test_fixed_fluxes_can_be_set_per_tracer(self) -> None:
    options = read_diffusion_options(
      {
        "mini_cloud": {
          "vertical_diffusion": {
            "theta": 1.0,
            "top_boundary": "fixed_flux",
            "top_flux": [1.0e-14, -2.0e-14],
            "bottom_boundary": "fixed_flux",
            "bottom_flux": 3.0e-14,
          }
        }
      }
    )
    self.assertEqual(options.theta, 1.0)
    np.testing.assert_array_equal(
      options.top_flux_values(2),
      np.array([1.0e-14, -2.0e-14]),
    )
    np.testing.assert_array_equal(
      options.bottom_flux_values(2),
      np.full(2, 3.0e-14),
    )
    with self.assertRaisesRegex(ValueError, "expected 3"):
      options.top_flux_values(3)

  def test_main_wrapper_passes_all_transport_options(self) -> None:
    atmosphere = run(SETUP_PATH, n_steps=0, write_output=False).atmosphere
    q = np.full((2, atmosphere.n_layers), 1.0e-10)
    q_bottom = np.array([1.17e-7, 2.0e-9])
    options = DiffusionOptions(
      theta=1.0,
      top_boundary="fixed_flux",
      top_flux=(1.0e-14, 2.0e-14),
      bottom_boundary="fixed_value",
      bottom_value="from_species",
    )
    result = advance_diffusion(
      atmosphere,
      q,
      50.0,
      options,
      bottom_value=q_bottom,
      source=1.0e-18,
      loss_rate=1.0e-6,
    )
    expected = vert_diffusion(
      q,
      q_bottom,
      50.0,
      atmosphere.p_edge,
      atmosphere.T,
      atmosphere.mu,
      atmosphere.Kzz,
      atmosphere.rho,
      atmosphere.z,
      atmosphere.dz,
      theta=1.0,
      top_boundary="fixed_flux",
      top_flux=np.array([1.0e-14, 2.0e-14]),
      bottom_boundary="dirichlet",
      source=1.0e-18,
      loss_rate=1.0e-6,
    )
    np.testing.assert_array_equal(result, expected)

  def test_missing_runtime_species_values_are_rejected(self) -> None:
    options = DiffusionOptions()
    with self.assertRaisesRegex(ValueError, "runtime bottom_value"):
      options.bottom_values(2)

  def test_invalid_or_unknown_options_are_rejected(self) -> None:
    with self.assertRaisesRegex(ValueError, "theta must lie"):
      read_diffusion_options(
        {"mini_cloud": {"vertical_diffusion": {"theta": 0.49}}}
      )
    with self.assertRaisesRegex(ValueError, "top_flux is required"):
      read_diffusion_options(
        {
          "mini_cloud": {
            "vertical_diffusion": {"top_boundary": "fixed_flux"}
          }
        }
      )
    with self.assertRaisesRegex(ValueError, "unknown.*option"):
      read_diffusion_options(
        {"mini_cloud": {"vertical_diffusion": {"theeta": 0.5}}}
      )


if __name__ == "__main__":
  unittest.main()
