"""Compare a stratified column with its constant-flux steady solution."""

from pathlib import Path

import numpy as np

from _common import (
  COLUMN_DEPTH,
  KZZ,
  Q_SCALE,
  DiffusionCase,
  errors,
  evolve,
  plot_case,
  stratified_profiles,
)
from pymini_cloud.vert_diffusion import R_gas


SCALE_HEIGHT = 0.5 * COLUMN_DEPTH
Q_BOTTOM = 0.25 * Q_SCALE


def calculate_comparison() -> DiffusionCase:
  """Evolve an exact steady state under a prescribed downward top flux."""

  profiles = stratified_profiles(SCALE_HEIGHT)
  depth, p_edge, temperature, mu, _, _, _, _ = profiles
  top_density = p_edge[0] * mu[0] / (R_gas * temperature[0])
  top_excess = 0.5 * Q_SCALE
  top_flux = (
    top_excess
    * top_density
    * KZZ
    / (SCALE_HEIGHT * (1.0 - np.exp(-COLUMN_DEPTH / SCALE_HEIGHT)))
  )
  steady = Q_BOTTOM + top_flux * SCALE_HEIGHT / (top_density * KZZ) * (
    np.exp(-depth / SCALE_HEIGHT) - np.exp(-COLUMN_DEPTH / SCALE_HEIGHT)
  )
  numerical = evolve(
    steady,
    profiles,
    q_bottom=Q_BOTTOM,
    top_boundary="fixed_flux",
    top_flux=top_flux,
  )[0]
  return DiffusionCase(
    "Constant-flux stratified steady state",
    depth,
    steady,
    numerical,
    steady,
  )


def main() -> None:
  """Run the constant-flux comparison and save its plot."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output = plot_case(case, Path(__file__).with_name("diffusion_constant_flux.png"))
  print(f"saved {output}")


if __name__ == "__main__":
  main()
