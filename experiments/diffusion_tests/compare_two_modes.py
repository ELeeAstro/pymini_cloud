"""Compare diffusion with a constant-density two-mode solution."""

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


def calculate_comparison() -> DiffusionCase:
  """Calculate numerical and exact two-mode decay."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  first_wave_number = 0.5 * np.pi / COLUMN_DEPTH
  second_wave_number = 2.5 * np.pi / COLUMN_DEPTH
  first_mode = np.cos(first_wave_number * depth)
  second_mode = np.cos(second_wave_number * depth)
  initial = Q_SCALE * (first_mode + 0.1 * second_mode)
  numerical = evolve(initial, profiles)[0]
  analytical = Q_SCALE * (
    first_mode * np.exp(-KZZ * first_wave_number**2 * TIME)
    + 0.1 * second_mode * np.exp(-KZZ * second_wave_number**2 * TIME)
  )
  return DiffusionCase(
    "Constant-density two-mode solution",
    depth,
    initial,
    numerical,
    analytical,
  )


def main() -> None:
  """Run and plot the two-mode comparison."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output_path = plot_case(
    case,
    Path(__file__).resolve().with_name("diffusion_two_modes.png"),
  )
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
