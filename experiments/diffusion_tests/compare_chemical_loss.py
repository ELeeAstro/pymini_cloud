"""Compare diffusion plus first-order chemical loss with an exact mode."""

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


LOSS_TIMESCALE = 0.2 * COLUMN_DEPTH**2 / KZZ


def calculate_comparison() -> DiffusionCase:
  """Calculate a mixed-boundary eigenmode with uniform linear loss."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  wave_number = 0.5 * np.pi / COLUMN_DEPTH
  initial = Q_SCALE * np.cos(wave_number * depth)
  loss_rate = 1.0 / LOSS_TIMESCALE
  numerical = evolve(initial, profiles, loss_rate=loss_rate)[0]
  analytical = initial * np.exp(-(KZZ * wave_number**2 + loss_rate) * TIME)
  return DiffusionCase(
    "Diffusion with first-order chemical loss",
    depth,
    initial,
    numerical,
    analytical,
  )


def main() -> None:
  """Run the diffusion-loss comparison and save its plot."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("diffusion_chemical_loss.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
