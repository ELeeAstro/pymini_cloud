"""Tests for complete NetCDF model-state output."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from netCDF4 import Dataset
import numpy as np
import yaml

from pymini_cloud.io_read import read_setup
from pymini_cloud.io_write import write_model_output
from pymini_cloud.main import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestNetcdfOutput(unittest.TestCase):
  """Verify restart fields, dimensions, metadata, and configured naming."""

  def test_complete_state_round_trip(self) -> None:
    state = run(SETUP_PATH, n_steps=2, write_output=False)
    setup = read_setup(SETUP_PATH)
    with tempfile.TemporaryDirectory() as temporary_directory:
      expected_path = (
        Path(temporary_directory).resolve() / "nested" / "final_state.nc"
      )
      output_path = write_model_output(
        state,
        setup,
        output_path=expected_path,
      )
      self.assertEqual(output_path, expected_path)
      self.assertTrue(output_path.is_file())

      with Dataset(output_path, "r") as dataset:
        self.assertEqual(dataset.schema_version, "1.1")
        self.assertEqual(dataset.model_step, 2)
        self.assertEqual(dataset.model_time, 200.0)
        self.assertIn("mini_cloud:", dataset.setup_yaml)
        self.assertEqual(len(dataset.dimensions["layer"]), 96)
        self.assertEqual(len(dataset.dimensions["edge"]), 97)
        self.assertEqual(len(dataset.dimensions["cloud_species"]), 1)

        atmosphere = dataset.groups["atmosphere"]
        cloud = dataset.groups["cloud"]
        configuration = dataset.groups["configuration"]
        np.testing.assert_array_equal(
          atmosphere.variables["pressure"][:],
          state.atmosphere.p,
        )
        np.testing.assert_array_equal(
          atmosphere.variables["thermal_conductivity"][:],
          state.atmosphere.kappa,
        )
        np.testing.assert_array_equal(
          cloud.variables["q_v"][:],
          state.cloud.q_v,
        )
        np.testing.assert_array_equal(
          cloud.variables["q_0"][:],
          state.cloud.q_0,
        )
        np.testing.assert_array_equal(
          cloud.variables["q_1"][:],
          state.cloud.q_1,
        )
        np.testing.assert_array_equal(
          cloud.variables["q_2"][:],
          state.cloud.q_2,
        )
        self.assertEqual(cloud.variables["species_name"][0], "KCl")
        self.assertAlmostEqual(cloud.variables["vmr_bottom"][0], 1.17e-7)
        self.assertEqual(configuration.diffusion_theta, 0.5)

  def test_output_can_restart_and_continue(self) -> None:
    initial = run(SETUP_PATH, n_steps=2, write_output=False)
    initial.settling_velocity = np.full(
      (initial.cloud.n_species + 2, initial.atmosphere.n_layers),
      12.5,
    )
    setup = read_setup(SETUP_PATH)
    with tempfile.TemporaryDirectory() as temporary_directory:
      directory = Path(temporary_directory)
      checkpoint = write_model_output(
        initial,
        setup,
        output_path=directory / "checkpoint.nc",
      )

      restart_setup = setup.raw_copy()
      restart_setup["mini_cloud"]["restart"] = True
      restart_setup["mini_cloud"]["restart_file"] = str(checkpoint)
      restart_setup["mini_cloud"]["output_file"] = "continued.nc"
      restart_yaml = directory / "restart.yaml"
      with restart_yaml.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(restart_setup, stream, sort_keys=False)

      continued = run(restart_yaml, n_steps=1, write_output=False)
      self.assertEqual(continued.step, 3)
      self.assertEqual(continued.time, 300.0)
      np.testing.assert_array_equal(
        continued.atmosphere.p,
        initial.atmosphere.p,
      )
      np.testing.assert_array_equal(
        continued.settling_velocity,
        initial.settling_velocity,
      )
      self.assertGreater(continued.cloud.q_v[0, -1], initial.cloud.q_v[0, -1])


if __name__ == "__main__":
  unittest.main()
