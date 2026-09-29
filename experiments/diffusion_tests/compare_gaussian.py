"""Compare diffusion with a freely spreading Gaussian pulse."""

from pathlib import Path

import numpy as np

from _common import (
  KZZ,
  Q_SCALE,
  DiffusionCase,
  constant_density_profiles,
  errors,
  evolve,
  plot_case,
)


INITIAL_TIME = 2.0e3
FINAL_TIME = 1.0e4
DT = 2.0e1


def gaussian(depth: np.ndarray, time: float) -> np.ndarray:
  """Return a top-centred Gaussian with conserved half-column integral."""

  return Q_SCALE * np.sqrt(INITIAL_TIME / time) * np.exp(
    -depth**2 / (4.0 * KZZ * time)
  )


def calculate_comparison() -> DiffusionCase:
  """Calculate the finite-volume and infinite-domain Gaussian solutions."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  initial = gaussian(depth, INITIAL_TIME)
  numerical = evolve(
    initial,
    profiles,
    time=FINAL_TIME - INITIAL_TIME,
    dt=DT,
    bottom_boundary="zero_flux",
  )[0]
  analytical = gaussian(depth, FINAL_TIME)
  return DiffusionCase(
    "Top-centred Gaussian pulse",
    depth,
    initial,
    numerical,
    analytical,
    time=FINAL_TIME,
  )


def main() -> None:
  """Run the Gaussian comparison and save its plot."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("diffusion_gaussian.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
