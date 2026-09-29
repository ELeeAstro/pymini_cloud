"""Compare upper-boundary inflow with an analytical translating step."""

from pathlib import Path

import numpy as np

from _common import (
  COLUMN_DEPTH,
  Q_SCALE,
  VELOCITY,
  AdvectionCase,
  constant_density_profiles,
  errors,
  evolve,
  plot_case,
)


TIME = 0.35 * COLUMN_DEPTH / VELOCITY
DT = TIME / 400


def calculate_comparison() -> AdvectionCase:
  """Calculate numerical and exact profiles for constant top inflow."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  initial = np.zeros_like(depth)
  numerical = evolve(
    initial,
    VELOCITY,
    profiles,
    time=TIME,
    dt=DT,
    q_top=Q_SCALE,
  )[0]
  analytical = np.where(depth < VELOCITY * TIME, Q_SCALE, 0.0)
  return AdvectionCase(
    "Constant top-inflow step",
    depth,
    initial,
    numerical,
    analytical,
    TIME,
    VELOCITY,
  )


def main() -> None:
  """Run the inflow-step comparison and save its plot."""

  case = calculate_comparison()
  l1_error, l2_error, max_error = errors(case)
  print(f"L1 error:   {l1_error:.6e}")
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("advection_inflow_step.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
