"""Compare diffusion with an eigenmode in an exponential atmosphere."""

from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from _common import (
  COLUMN_DEPTH,
  KZZ,
  Q_SCALE,
  TIME,
  DiffusionCase,
  errors,
  evolve,
  plot_case,
  stratified_profiles,
)


def calculate_comparison() -> DiffusionCase:
  """Calculate numerical and exact density-weighted eigenmode decay."""

  scale_height = 0.5 * COLUMN_DEPTH
  profiles = stratified_profiles(scale_height)
  depth = profiles[0]

  def boundary_equation(dimensionless_wave_number: float) -> float:
    coefficient = COLUMN_DEPTH / (
      2.0 * scale_height * dimensionless_wave_number
    )
    return (
      np.cos(dimensionless_wave_number)
      + coefficient * np.sin(dimensionless_wave_number)
    )

  root = brentq(boundary_equation, 0.5 * np.pi, np.pi)
  wave_number = root / COLUMN_DEPTH
  coefficient = 1.0 / (2.0 * scale_height * wave_number)
  eigenmode = np.exp(-depth / (2.0 * scale_height)) * (
    np.cos(wave_number * depth)
    + coefficient * np.sin(wave_number * depth)
  )
  initial = Q_SCALE * eigenmode
  numerical = evolve(initial, profiles)[0]
  decay_rate = KZZ * (
    wave_number**2 + 1.0 / (4.0 * scale_height**2)
  )
  analytical = initial * np.exp(-decay_rate * TIME)
  return DiffusionCase(
    "Exponentially stratified fundamental mode",
    depth,
    initial,
    numerical,
    analytical,
  )


def main() -> None:
  """Run and plot the stratified-mode comparison."""

  case = calculate_comparison()
  l2_error, max_error = errors(case)
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")
  output_path = plot_case(
    case,
    Path(__file__).resolve().with_name("diffusion_stratified_mode.png"),
  )
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
