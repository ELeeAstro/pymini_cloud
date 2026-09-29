"""Compare diffusion with the complementary-error-function step solution."""

from pathlib import Path

import numpy as np
from scipy.special import erfc

from _common import (
  COLUMN_DEPTH,
  KZZ,
  Q_SCALE,
  DiffusionCase,
  constant_density_profiles,
  errors,
  evolve,
  plot_case,
)


STEP_DEPTH = 0.5 * COLUMN_DEPTH
INITIAL_TIME = 5.0e2
FINAL_TIME = 2.0e3
DT = 1.0e1


def diffused_step(depth: np.ndarray, time: float) -> np.ndarray:
  """Return the infinite-domain solution for a downward-increasing step."""

  argument = (STEP_DEPTH - depth) / np.sqrt(4.0 * KZZ * time)
  return 0.5 * Q_SCALE * erfc(argument)


def calculate_comparison() -> DiffusionCase:
  """Calculate numerical and analytical smoothed-step profiles."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  initial = diffused_step(depth, INITIAL_TIME)
  numerical = evolve(
    initial,
    profiles,
    time=FINAL_TIME - INITIAL_TIME,
    dt=DT,
    bottom_boundary="zero_flux",
  )[0]
  analytical = diffused_step(depth, FINAL_TIME)
  return DiffusionCase(
    "Complementary-error-function step",
    depth,
    initial,
    numerical,
    analytical,
    time=FINAL_TIME,
  )


def main() -> None:
  """Run the erfc comparison and save its plot."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("diffusion_erfc_step.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
