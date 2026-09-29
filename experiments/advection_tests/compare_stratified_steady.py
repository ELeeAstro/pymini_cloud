"""Check the exact finite-volume steady state in a stratified column."""

from pathlib import Path

import numpy as np

from _common import (
  COLUMN_DEPTH,
  Q_SCALE,
  VELOCITY,
  AdvectionCase,
  errors,
  evolve,
  plot_case,
  vertical_grid,
)


SCALE_HEIGHT = 0.35 * COLUMN_DEPTH
TIME = 0.50 * COLUMN_DEPTH / VELOCITY
DT = TIME / 500


def calculate_comparison() -> AdvectionCase:
  """Construct and evolve the continuum constant-mass-flux solution."""

  depth, dz = vertical_grid()
  rho = 1.0e-6 * np.exp(depth / SCALE_HEIGHT)
  steady = Q_SCALE * np.exp(-depth / SCALE_HEIGHT)
  profiles = (depth, rho, dz)
  numerical = evolve(
    steady,
    VELOCITY,
    profiles,
    time=TIME,
    dt=DT,
    q_top=Q_SCALE,
  )[0]
  return AdvectionCase(
    "Stratified analytical constant-flux state",
    depth,
    steady,
    numerical,
    steady,
    TIME,
    VELOCITY,
  )


def main() -> None:
  """Run the stratified steady-state comparison."""

  case = calculate_comparison()
  l1_error, l2_error, max_error = errors(case)
  print(f"L1 error:   {l1_error:.6e}")
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("advection_stratified_steady.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
