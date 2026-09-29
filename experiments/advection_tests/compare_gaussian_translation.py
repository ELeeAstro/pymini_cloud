"""Compare constant-speed advection with a translating Gaussian pulse."""

from pathlib import Path

from _common import (
  COLUMN_DEPTH,
  VELOCITY,
  AdvectionCase,
  constant_density_profiles,
  errors,
  evolve,
  gaussian,
  plot_case,
)


CENTRE = 0.30 * COLUMN_DEPTH
WIDTH = 0.06 * COLUMN_DEPTH
TIME = 0.20 * COLUMN_DEPTH / VELOCITY
DT = TIME / 400


def calculate_comparison() -> AdvectionCase:
  """Calculate numerical and exact translated Gaussian profiles."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  initial = gaussian(depth, CENTRE, WIDTH)
  numerical = evolve(initial, VELOCITY, profiles, time=TIME, dt=DT)[0]
  analytical = gaussian(depth, CENTRE + VELOCITY * TIME, WIDTH)
  return AdvectionCase(
    "Constant-speed Gaussian translation",
    depth,
    initial,
    numerical,
    analytical,
    TIME,
    VELOCITY,
  )


def main() -> None:
  """Run the Gaussian translation comparison and save its plot."""

  case = calculate_comparison()
  l1_error, l2_error, max_error = errors(case)
  print(f"L1 error:   {l1_error:.6e}")
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("advection_gaussian.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
