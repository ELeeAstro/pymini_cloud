"""Compare free lower-boundary outflow with a translated Gaussian."""

from math import erf, sqrt
from pathlib import Path

import numpy as np

from _common import (
  COLUMN_DEPTH,
  Q_SCALE,
  RHO,
  VELOCITY,
  AdvectionCase,
  constant_density_profiles,
  errors,
  evolve,
  gaussian,
  plot_case,
)


CENTRE = 0.70 * COLUMN_DEPTH
WIDTH = 0.06 * COLUMN_DEPTH
TIME = 0.25 * COLUMN_DEPTH / VELOCITY
DT = TIME / 500


def gaussian_column(centre: float) -> float:
  """Return the exact tracer column retained between zero and ``L``."""

  factor = sqrt(np.pi / 2.0) * WIDTH
  upper = erf((COLUMN_DEPTH - centre) / (sqrt(2.0) * WIDTH))
  lower = erf((0.0 - centre) / (sqrt(2.0) * WIDTH))
  return RHO * Q_SCALE * factor * (upper - lower)


def calculate_comparison() -> tuple[AdvectionCase, float]:
  """Calculate the profile and retained-column comparison."""

  profiles = constant_density_profiles()
  depth, rho, dz = profiles
  initial = gaussian(depth, CENTRE, WIDTH)
  numerical = evolve(initial, VELOCITY, profiles, time=TIME, dt=DT)[0]
  final_centre = CENTRE + VELOCITY * TIME
  analytical = gaussian(depth, final_centre, WIDTH)
  numerical_column = float(np.sum(rho * dz * numerical))
  exact_column = gaussian_column(final_centre)
  relative_column_error = (numerical_column - exact_column) / exact_column
  return (
    AdvectionCase(
      "Gaussian leaving through the free lower boundary",
      depth,
      initial,
      numerical,
      analytical,
      TIME,
      VELOCITY,
    ),
    relative_column_error,
  )


def main() -> None:
  """Run the lower-outflow comparison and save its plot."""

  case, column_error = calculate_comparison()
  l1_error, l2_error, max_error = errors(case)
  print(f"L1 profile error:      {l1_error:.6e}")
  print(f"L2 profile error:      {l2_error:.6e}")
  print(f"Linf profile error:    {max_error:.6e}")
  print(f"relative column error: {column_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("advection_outflow_mass.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
