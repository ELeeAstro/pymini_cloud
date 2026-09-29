"""Checks specific to the hydrostatic non-uniform atmospheric grid."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pymini_cloud.main import run
from pymini_cloud.vert_diffusion import vert_diffusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestNonuniformAtmosphericGrid(unittest.TestCase):
  """Verify finite-volume weights and transport on the production grid."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.state = run(SETUP_PATH, n_steps=0, write_output=False)

  def test_hydrostatic_cell_mass_matches_pressure_difference(self) -> None:
    atmosphere = self.state.atmosphere
    geometric_mass = atmosphere.rho * atmosphere.dz
    hydrostatic_mass = np.diff(atmosphere.p_edge) / atmosphere.g
    self.assertGreater(np.ptp(atmosphere.dz), 0.0)
    np.testing.assert_allclose(
      geometric_mass,
      hydrostatic_mass,
      # R_gas versus k_B/amu introduces a ~1e-9 physical-constant mismatch.
      rtol=2.0e-9,
      atol=0.0,
    )

  def test_constant_mixing_ratio_has_zero_diffusive_flux(self) -> None:
    atmosphere = self.state.atmosphere
    initial = np.full((1, atmosphere.n_layers), 2.0e-8)
    result = vert_diffusion(
      initial,
      0.0,
      1.0e4,
      atmosphere.p_edge,
      atmosphere.T,
      atmosphere.mu,
      atmosphere.Kzz,
      atmosphere.rho,
      atmosphere.z,
      atmosphere.dz,
      top_boundary="zero_flux",
      bottom_boundary="zero_flux",
    )
    np.testing.assert_allclose(result, initial, rtol=2.0e-13, atol=0.0)

  def test_kcl_bottom_vmr_is_converted_to_fortran_mass_ratio(self) -> None:
    species = self.state.cloud.species[0]
    expected = 1.17e-7 * 74.5513 / 2.33
    self.assertAlmostEqual(
      species.bottom_mass_mixing_ratio(self.state.atmosphere.mu[-1]),
      expected,
      places=18,
    )


if __name__ == "__main__":
  unittest.main()
