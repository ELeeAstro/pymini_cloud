"""Check decay toward a mass-conserving, well-mixed column."""

from pathlib import Path

import numpy as np

from _common import (
  COLUMN_DEPTH,
  KZZ,
  Q_SCALE,
  TIME,
  DiffusionCase,
  constant_density_profiles,
  errors,
  evolve,
  plot_case,
)


def calculate_comparison() -> tuple[DiffusionCase, float]:
  """Calculate the double-Neumann mode decay and mass residual."""

  profiles = constant_density_profiles()
  depth, _, _, _, _, rho, _, dz = profiles
  wave_number = np.pi / COLUMN_DEPTH
  initial = Q_SCALE * (1.0 + 0.5 * np.cos(wave_number * depth))
  numerical = evolve(
    initial,
    profiles,
    bottom_boundary="zero_flux",
  )[0]
  analytical = Q_SCALE * (
    1.0 + 0.5 * np.cos(wave_number * depth) * np.exp(-KZZ * wave_number**2 * TIME)
  )
  initial_mass = np.sum(rho * dz * initial)
  final_mass = np.sum(rho * dz * numerical)
  mass_residual = float((final_mass - initial_mass) / initial_mass)
  return (
    DiffusionCase("Well-mixed double-Neumann column", depth, initial, numerical, analytical),
    mass_residual,
  )


def main() -> None:
  """Run the well-mixed comparison and save its plot."""

  case, mass_residual = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:             {l2_error:.6e}")
  print(f"Linf error:           {max_error:.6e}")
  print(f"relative mass change: {mass_residual:.6e}")
  output = plot_case(case, Path(__file__).with_name("diffusion_well_mixed.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
