"""Compare vertical diffusion with a constant-coefficient analytical solution."""

from __future__ import annotations

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
Q_BOTTOM = 1.17e-7
RHO = 1.0e-5               # g cm^-3
MU = 2.33                  # g mol^-1
P_TOP = 1.0e6              # dyne cm^-2
P_BOTTOM = 2.0e6           # dyne cm^-2
DT = 1.0e2                 # s
NORMALIZED_TIMES = (0.01, 0.05, 0.20)


def analytical_solution(
  depth: FloatArray,
  time: float,
  n_terms: int = 512,
) -> FloatArray:
  """Return the mixed-boundary constant-diffusion series solution."""

  mode = np.arange(n_terms, dtype=np.float64)
  wave_number = (mode + 0.5) * np.pi / COLUMN_DEPTH
  coefficient = 2.0 * (-1.0)**mode / ((mode + 0.5) * np.pi)
  remaining_fraction = np.sum(
    coefficient[:, None]
    * np.cos(wave_number[:, None] * depth[None, :])
    * np.exp(-KZZ * wave_number[:, None]**2 * time),
    axis=0,
  )
  return Q_BOTTOM * (1.0 - remaining_fraction)


def constant_density_column(
  n_layers: int = N_LAYERS,
) -> tuple[FloatArray, ...]:
  """Construct profiles giving constant centre and interface densities."""

  dz_value = COLUMN_DEPTH / n_layers
  depth = (np.arange(n_layers, dtype=np.float64) + 0.5) * dz_value
  z = COLUMN_DEPTH - depth
  dz = np.full(n_layers, dz_value)
  rho = np.full(n_layers, RHO)
  Kzz = np.full(n_layers, KZZ)
  mu = np.full(n_layers, MU)
  p_edge = np.linspace(P_TOP, P_BOTTOM, n_layers + 1)

  # Match the Fortran interface-density rule exactly. Interior interfaces use
  # the average temperature of their neighboring cells; the bottom interface
  # uses the final cell temperature.
  density_temperature_factor = MU / (R_gas * RHO)
  T = np.empty(n_layers)
  T[-1] = density_temperature_factor * p_edge[-1]
  for layer in range(n_layers - 1, 0, -1):
    T[layer - 1] = (
      2.0 * density_temperature_factor * p_edge[layer] - T[layer]
    )

  return depth, p_edge, T, mu, Kzz, rho, z, dz


def calculate_comparison() -> tuple[
  FloatArray,
  dict[float, FloatArray],
  dict[float, FloatArray],
]:
  """Evolve the numerical column and evaluate matching analytical profiles."""

  depth, p_edge, T, mu, Kzz, rho, z, dz = constant_density_column()
  diffusion_time = COLUMN_DEPTH**2 / KZZ
  comparison_times = tuple(value * diffusion_time for value in NORMALIZED_TIMES)
  comparison_steps = {int(round(time / DT)): time for time in comparison_times}

  q = np.zeros((1, N_LAYERS))
  numerical = {}
  analytical = {}
  for step in range(1, max(comparison_steps) + 1):
    q = vert_diffusion(
      q,
      Q_BOTTOM,
      DT,
      p_edge,
      T,
      mu,
      Kzz,
      rho,
      z,
      dz,
    )
    if step in comparison_steps:
      time = comparison_steps[step]
      numerical[time] = q[0].copy()
      analytical[time] = analytical_solution(depth, time)

  return depth, numerical, analytical


def plot_comparison(output_path: str | Path) -> Path:
  """Run the comparison and save numerical profiles and their errors."""

  depth, numerical, analytical = calculate_comparison()
  normalized_depth = depth / COLUMN_DEPTH
  diffusion_time = COLUMN_DEPTH**2 / KZZ

  figure, (profile_axis, error_axis) = plt.subplots(
    1,
    2,
    figsize=(11.0, 5.4),
    sharey=True,
  )
  colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

  print("t/t_diff       L2 error     Linf error")
  for color, (time, q_numerical) in zip(colors, numerical.items()):
    q_analytical = analytical[time]
    normalized_numerical = q_numerical / Q_BOTTOM
    normalized_analytical = q_analytical / Q_BOTTOM
    error = np.abs(normalized_numerical - normalized_analytical)
    l2_error = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    time_ratio = time / diffusion_time
    print(f"{time_ratio:8.3f}  {l2_error:13.6e}  {max_error:13.6e}")

    profile_axis.plot(
      normalized_analytical,
      normalized_depth,
      color=color,
      linewidth=2.0,
      label=fr"analytical, $t/t_D={time_ratio:g}$",
    )
    profile_axis.plot(
      normalized_numerical,
      normalized_depth,
      color=color,
      linestyle="none",
      marker="o",
      markersize=3.2,
      markevery=3,
      label=fr"numerical, $t/t_D={time_ratio:g}$",
    )
    error_axis.plot(
      np.maximum(error, 1.0e-16),
      normalized_depth,
      color=color,
      linewidth=2.0,
      label=fr"$t/t_D={time_ratio:g}$",
    )

  profile_axis.set_xlabel(r"Normalized mixing ratio $q/q_{\mathrm{bottom}}$")
  profile_axis.set_ylabel(r"Normalized depth $x/L$")
  profile_axis.set_title("Profiles")
  profile_axis.grid(alpha=0.25)
  profile_axis.legend(fontsize=8)

  error_axis.set_xscale("log")
  error_axis.set_xlabel("Absolute normalized error")
  error_axis.set_title("Numerical error")
  error_axis.grid(alpha=0.25, which="both")
  error_axis.legend(fontsize=8)
  error_axis.invert_yaxis()

  figure.suptitle("Implicit diffusion compared with the analytical solution")
  figure.tight_layout()

  output_path = Path(output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=200)
  plt.close(figure)
  return output_path


def main() -> None:
  """Run the analytical comparison experiment."""

  output_path = plot_comparison(
    Path(__file__).resolve().with_name("diffusion_analytical_comparison.png")
  )
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
