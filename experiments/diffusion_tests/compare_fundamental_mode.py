"""Compare diffusion with a constant-density fundamental eigenmode."""

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
  """Calculate numerical and exact fundamental-mode decay."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  wave_number = 0.5 * np.pi / COLUMN_DEPTH
  initial = Q_SCALE * np.cos(wave_number * depth)
  numerical = evolve(initial, profiles)[0]
  analytical = initial * np.exp(-KZZ * wave_number**2 * TIME)
  return DiffusionCase(
    "Constant-density fundamental mode",
    depth,
    initial,
    numerical,
    analytical,
  )


def main() -> None:
  """Run and plot the fundamental-mode comparison."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output_path = plot_case(
    case,
    Path(__file__).resolve().with_name("diffusion_fundamental_mode.png"),
  )
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
