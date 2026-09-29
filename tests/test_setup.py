"""Tests for the validated dot-access setup tree."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
import tempfile
import unittest

import yaml

from pymini_cloud.io_read import Setup, read_setup
from pymini_cloud.main import run


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SETUP_PATH = PROJECT_ROOT / "experiments" / "Y_400K_KCl" / "setup.yaml"


class TestSetup(unittest.TestCase):
  """Validate grouped access, resolved paths, and strict schema checks."""

  def test_complete_setup_uses_grouped_dot_access(self) -> None:
    setup = read_setup(SETUP_PATH)
    self.assertIsInstance(setup, Setup)
    self.assertFalse(setup.restart.enabled)
    self.assertEqual(setup.grid.n_layers, 96)
    self.assertEqual(setup.physics.size_distribution, "gamma")
    self.assertEqual(setup.advection.method, "muscl_ssprk3")
    self.assertEqual(setup.diffusion.theta, 0.5)
    self.assertEqual(setup.integrator.t_step, 100.0)
    self.assertEqual(setup.species[0].name, "KCl")
    self.assertEqual(
      setup.output.file,
      SETUP_PATH.parent / "Y_400K_mc_out.nc",
    )

  def test_option_groups_are_immutable(self) -> None:
    setup = read_setup(SETUP_PATH)
    with self.assertRaises(FrozenInstanceError):
      setup.grid.n_layers = 48  # type: ignore[misc]

  def test_validated_setup_can_be_passed_to_run(self) -> None:
    setup = read_setup(SETUP_PATH)
    state = run(setup, n_steps=0, write_output=False)
    self.assertEqual(state.atmosphere.n_layers, setup.grid.n_layers)

  def test_raw_copy_is_independent(self) -> None:
    setup = read_setup(SETUP_PATH)
    copied = setup.raw_copy()
    copied["mini_cloud"]["n_layers"] = 48
    self.assertEqual(setup.grid.n_layers, 96)
    self.assertEqual(setup.raw["mini_cloud"]["n_layers"], 96)

  def test_unknown_model_option_is_rejected(self) -> None:
    setup = read_setup(SETUP_PATH).raw_copy()
    setup["mini_cloud"]["n_layres"] = 96
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "bad.yaml"
      with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(setup, stream, sort_keys=False)
      with self.assertRaisesRegex(ValueError, "n_layres"):
        read_setup(path)


if __name__ == "__main__":
  unittest.main()
