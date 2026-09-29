"""Tests for the first diffusion-only main-loop implementation."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from pymini_cloud.data_cld_sp import CloudSpecies, CloudState
from pymini_cloud.io_read import read_setup
from pymini_cloud.main import advance_model_step, run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestCloudState(unittest.TestCase):
  """Check the canonical q_v, q_0, q_1, q_2 transport layout."""

  def test_transport_pack_and_bottom_order(self) -> None:
    species = (
      CloudSpecies.from_fortran_defaults("KCl", 1.0e-7),
      CloudSpecies.from_fortran_defaults("ZnS", 2.0e-7),
    )
    state = CloudState.initialise(species, n_layers=3)
    state.q_v[:] = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    state.q_0[:] = 7.0
    state.q_1[:] = np.array([[8.0, 9.0, 10.0], [11.0, 12.0, 13.0]])
    state.q_2[:] = 14.0

    np.testing.assert_array_equal(
      state.pack_transport(),
      np.array(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
          [7.0, 7.0, 7.0],
          [8.0, 9.0, 10.0],
          [11.0, 12.0, 13.0],
          [14.0, 14.0, 14.0],
        ]
      ),
    )
    np.testing.assert_array_equal(
      state.transport_bottom_values(2.33),
      np.array(
        [
          1.0e-7 * 74.551 / 2.33,
          2.0e-7 * 65.38 / 2.33,
          1.0e-30,
          1.0e-30,
          1.0e-30,
          1.0e-30,
        ]
      ),
    )


class TestMainDiffusionLoop(unittest.TestCase):
  """Exercise setup parsing and several complete outer timesteps."""

  def test_example_species_and_integrator_are_read(self) -> None:
    setup = read_setup(SETUP_PATH)
    species = setup.species
    integrator = setup.integrator
    self.assertEqual(
      species[0],
      CloudSpecies.from_fortran_defaults(
        "KCl",
        1.17e-7,
        condensate_molar_mass=74.5513,
        vapour_molar_mass=74.5513,
        optical_constants=str(PROJECT_ROOT / "data" / "nk" / "KCl[s].dat"),
      ),
    )
    self.assertEqual(integrator.t_step, 100.0)
    self.assertEqual(integrator.n_step, 10000)

  def test_bottom_vapour_source_leaves_cloud_moments_at_floor(self) -> None:
    state = run(SETUP_PATH, n_steps=10, write_output=False)
    self.assertEqual(state.step, 10)
    self.assertEqual(state.time, 1000.0)
    self.assertGreater(state.cloud.q_v[0, -1], state.cloud.q_floor)
    self.assertGreater(state.cloud.q_v[0, -1], state.cloud.q_v[0, 0])
    bottom_mass = state.cloud.species[0].bottom_mass_mixing_ratio(
      state.atmosphere.mu[-1]
    )
    self.assertLess(state.cloud.q_v[0, -1], bottom_mass)
    np.testing.assert_allclose(
      state.cloud.q_0,
      state.cloud.q_floor,
      rtol=1.0e-13,
      atol=0.0,
    )
    np.testing.assert_allclose(
      state.cloud.q_1,
      state.cloud.q_floor,
      rtol=1.0e-13,
      atol=0.0,
    )
    np.testing.assert_allclose(
      state.cloud.q_2,
      state.cloud.q_floor,
      rtol=1.0e-13,
      atol=0.0,
    )

  def test_settling_option_applies_two_half_steps(self) -> None:
    state = run(SETUP_PATH, n_steps=0, write_output=False)
    state.do_diffusion = False
    state.do_settling = True
    state.cloud.q_0[10] = 1.0e-8
    state.settling_velocity = np.full(
      (state.cloud.n_species + 2, state.atmosphere.n_layers),
      100.0,
    )
    initial = state.cloud.q_0.copy()
    advance_model_step(state)
    self.assertFalse(np.array_equal(state.cloud.q_0, initial))
    self.assertEqual(state.step, 1)
    self.assertEqual(state.time, state.integrator.t_step)


if __name__ == "__main__":
  unittest.main()
