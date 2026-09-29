"""Shared utilities for analytical diffusion comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.vert_diffusion import R_gas, vert_diffusion


FloatArray = NDArray[np.float64]

N_LAYERS = 96
COLUMN_DEPTH = 1.0e7       # cm
KZZ = 1.0e8                # cm^2 s^-1
Q_SCALE = 1.0e-7
MU = 2.33                  # g mol^-1
DT = 1.0e2                 # s
TIME = 0.10 * COLUMN_DEPTH**2 / KZZ


@dataclass(frozen=True)
class DiffusionCase:
  """Numerical and analytical results for one diffusion benchmark."""

  name: str
  depth: FloatArray
  initial: FloatArray
  numerical: FloatArray
  analytical: FloatArray
  time: float = TIME


def vertical_coordinates(
  n_layers: int = N_LAYERS,
  stretch: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return downward depth, upward altitude, and cell thickness.

  ``stretch`` is the exponent applied to uniformly spaced edge indices. A
  value above one concentrates cells near the top boundary.
  """

  if n_layers < 2:
    raise ValueError("n_layers must be at least two")
  if stretch <= 0.0:
    raise ValueError("stretch must be positive")
  edge_fraction = np.linspace(0.0, 1.0, n_layers + 1)**stretch
  edge_depth = COLUMN_DEPTH * edge_fraction
  dz = np.diff(edge_depth)
  depth = 0.5 * (edge_depth[:-1] + edge_depth[1:])
  z = COLUMN_DEPTH - depth
  return depth, z, dz


def constant_density_profiles(
  n_layers: int = N_LAYERS,
  stretch: float = 1.0,
) -> tuple[FloatArray, ...]:
  """Construct profiles giving constant centre and interface density."""

  depth, z, dz = vertical_coordinates(n_layers, stretch)
  rho_value = 1.0e-5
  p_edge = np.linspace(1.0e6, 2.0e6, n_layers + 1)
  rho = np.full(n_layers, rho_value)
  Kzz = np.full(n_layers, KZZ)
  mu = np.full(n_layers, MU)

  factor = MU / (R_gas * rho_value)
  T = np.empty(n_layers)
  T[-1] = factor * p_edge[-1]
  for layer in range(n_layers - 1, 0, -1):
    T[layer - 1] = 2.0 * factor * p_edge[layer] - T[layer]

  return depth, p_edge, T, mu, Kzz, rho, z, dz


def stratified_profiles(
  scale_height: float,
  n_layers: int = N_LAYERS,
) -> tuple[FloatArray, ...]:
  """Construct an isothermal atmosphere with exponential density."""

  depth, z, dz = vertical_coordinates(n_layers)
  edge_depth = np.linspace(0.0, COLUMN_DEPTH, n_layers + 1)
  temperature = 1500.0
  p_edge = 1.0e6 * np.exp(edge_depth / scale_height)
  p = np.diff(p_edge) / np.diff(np.log(p_edge))
  T = np.full(n_layers, temperature)
  mu = np.full(n_layers, MU)
  Kzz = np.full(n_layers, KZZ)
  rho = p * MU / (R_gas * temperature)
  return depth, p_edge, T, mu, Kzz, rho, z, dz


def evolve(
  q: FloatArray,
  profiles: tuple[FloatArray, ...],
  *,
  time: float = TIME,
  dt: float = DT,
  q_bottom: float | FloatArray = 0.0,
  **diffusion_options: object,
) -> FloatArray:
  """Evolve one or more profiles for a requested duration."""

  _, p_edge, T, mu, Kzz, rho, z, dz = profiles
  evolved = np.atleast_2d(np.asarray(q, dtype=np.float64)).copy()
  n_steps = int(round(time / dt))
  if not np.isclose(n_steps * dt, time):
    raise ValueError("time must be an integer multiple of dt")
  for _ in range(n_steps):
    evolved = vert_diffusion(
      evolved,
      q_bottom,
      dt,
      p_edge,
      T,
      mu,
      Kzz,
      rho,
      z,
      dz,
      **diffusion_options,
    )
  return evolved


def errors(case: DiffusionCase) -> tuple[float, float]:
  """Return normalized L2 and maximum errors."""

  difference = (case.numerical - case.analytical) / Q_SCALE
  return float(np.sqrt(np.mean(difference**2))), float(np.max(np.abs(difference)))


def plot_case(case: DiffusionCase, output_path: str | Path) -> Path:
  """Plot one analytical comparison and its normalized error."""

  depth = case.depth / COLUMN_DEPTH
  initial = case.initial / Q_SCALE
  numerical = case.numerical / Q_SCALE
  analytical = case.analytical / Q_SCALE
  error = np.abs(numerical - analytical)
  l2_error, max_error = errors(case)

  figure, (profile_axis, error_axis) = plt.subplots(
    1,
    2,
    figsize=(10.5, 5.2),
    sharey=True,
  )
  profile_axis.plot(initial, depth, color="0.65", linestyle="--", label="initial")
  profile_axis.plot(analytical, depth, linewidth=2.0, label="analytical")
  profile_axis.plot(
    numerical,
    depth,
    linestyle="none",
    marker="o",
    markersize=3.2,
    markevery=3,
    label="numerical",
  )
  profile_axis.set_xlabel(r"Normalized mixing ratio $q/q_0$")
  profile_axis.set_ylabel(r"Normalized depth $x/L$")
  profile_axis.set_title("Profiles")
  profile_axis.grid(alpha=0.25)
  profile_axis.legend()

  error_axis.plot(np.maximum(error, 1.0e-16), depth, linewidth=2.0)
  error_axis.set_xscale("log")
  error_axis.set_xlabel("Absolute normalized error")
  error_axis.set_title("Numerical error")
  error_axis.grid(alpha=0.25, which="both")
  error_axis.invert_yaxis()

  time_ratio = case.time * KZZ / COLUMN_DEPTH**2
  figure.suptitle(
    f"{case.name}\n"
    f"$t/t_D={time_ratio:g}$, "
    f"$L_2={l2_error:.2e}$, $L_\\infty={max_error:.2e}$"
  )
  figure.tight_layout()

  output_path = Path(output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=200)
  plt.close(figure)
  return output_path
