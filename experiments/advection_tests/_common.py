"""Shared utilities for analytical settling-advection comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import NDArray

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.vert_advection import vert_advection


FloatArray = NDArray[np.float64]

N_LAYERS = 96
COLUMN_DEPTH = 1.0e7  # cm
VELOCITY = 1.0e2      # cm s^-1, positive downward
Q_SCALE = 1.0e-7
RHO = 1.0e-5          # g cm^-3


@dataclass(frozen=True)
class AdvectionCase:
  """Numerical and analytical profiles for one advection benchmark."""

  name: str
  depth: FloatArray
  initial: FloatArray
  numerical: FloatArray
  analytical: FloatArray
  time: float
  velocity: float


def vertical_grid(
  n_layers: int = N_LAYERS,
  stretch: float = 1.0,
) -> tuple[FloatArray, FloatArray]:
  """Return downward cell-centre depths and layer thicknesses."""

  if n_layers < 2:
    raise ValueError("n_layers must be at least two")
  if stretch <= 0.0:
    raise ValueError("stretch must be positive")
  edge_depth = COLUMN_DEPTH * np.linspace(0.0, 1.0, n_layers + 1)**stretch
  dz = np.diff(edge_depth)
  depth = 0.5 * (edge_depth[:-1] + edge_depth[1:])
  return depth, dz


def constant_density_profiles(
  n_layers: int = N_LAYERS,
  stretch: float = 1.0,
) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Return depth, density, and layer thickness for a uniform-density column."""

  depth, dz = vertical_grid(n_layers, stretch)
  return depth, np.full(n_layers, RHO), dz


def evolve(
  q: FloatArray,
  velocity: float | FloatArray,
  profiles: tuple[FloatArray, FloatArray, FloatArray],
  *,
  time: float,
  dt: float,
  q_top: float | FloatArray = 0.0,
  **advection_options: object,
) -> FloatArray:
  """Advance one or more moment profiles for an integer number of steps."""

  _, rho, dz = profiles
  evolved = np.atleast_2d(np.asarray(q, dtype=np.float64)).copy()
  velocity_array = np.asarray(velocity, dtype=np.float64)
  if velocity_array.ndim == 0:
    velocity_array = np.full_like(evolved, float(velocity_array))
  elif velocity_array.shape == (evolved.shape[0],):
    velocity_array = np.broadcast_to(velocity_array[:, None], evolved.shape).copy()
  elif velocity_array.shape != evolved.shape:
    raise ValueError("velocity must be scalar, per-moment, or match q")

  n_steps = int(round(time / dt))
  if not np.isclose(n_steps * dt, time):
    raise ValueError("time must be an integer multiple of dt")
  for _ in range(n_steps):
    evolved = vert_advection(
      evolved,
      velocity_array,
      dt,
      rho,
      dz,
      q_top,
      **advection_options,
    )
  return evolved


def gaussian(depth: FloatArray, centre: float, width: float) -> FloatArray:
  """Return a Gaussian profile with peak ``Q_SCALE``."""

  return Q_SCALE * np.exp(-0.5 * ((depth - centre) / width)**2)


def errors(case: AdvectionCase) -> tuple[float, float, float]:
  """Return normalized L1, L2, and maximum errors."""

  difference = np.abs(case.numerical - case.analytical) / Q_SCALE
  return (
    float(np.mean(difference)),
    float(np.sqrt(np.mean(difference**2))),
    float(np.max(difference)),
  )


def plot_case(case: AdvectionCase, output_path: str | Path) -> Path:
  """Plot one analytical profile comparison and its normalized error."""

  normalized_depth = case.depth / COLUMN_DEPTH
  normalized_error = np.abs(case.numerical - case.analytical) / Q_SCALE
  l1_error, l2_error, max_error = errors(case)

  figure, (profile_axis, error_axis) = plt.subplots(
    1, 2, figsize=(10.5, 5.2), sharey=True
  )
  profile_axis.plot(
    case.initial / Q_SCALE,
    normalized_depth,
    color="0.65",
    linestyle="--",
    label="initial",
  )
  profile_axis.plot(
    case.analytical / Q_SCALE,
    normalized_depth,
    linewidth=2.0,
    label="analytical",
  )
  profile_axis.plot(
    case.numerical / Q_SCALE,
    normalized_depth,
    linestyle="none",
    marker="o",
    markersize=3.2,
    markevery=max(1, case.depth.size // 32),
    label="numerical",
  )
  profile_axis.set_xlabel(r"Normalized mixing ratio $q/q_0$")
  profile_axis.set_ylabel(r"Normalized downward depth $x/L$")
  profile_axis.grid(alpha=0.25)
  profile_axis.legend()

  error_axis.plot(np.maximum(normalized_error, 1.0e-16), normalized_depth)
  error_axis.set_xscale("log")
  error_axis.set_xlabel("Absolute normalized error")
  error_axis.grid(alpha=0.25, which="both")
  error_axis.invert_yaxis()

  crossing_time = COLUMN_DEPTH / case.velocity
  figure.suptitle(
    f"{case.name}\n"
    f"$t/(L/v)={case.time / crossing_time:.3g}$, "
    f"$L_1={l1_error:.2e}$, $L_2={l2_error:.2e}$, "
    f"$L_\\infty={max_error:.2e}$"
  )
  figure.tight_layout()

  output_path = Path(output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=200)
  plt.close(figure)
  return output_path
