"""Check the finite-volume column budget against the boundary flux."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import KZZ, MU, Q_SCALE, constant_density_profiles
from pymini_cloud.vert_diffusion import R_gas, vert_diffusion


DT = 100.0
N_STEPS = 100
Q_BOTTOM = Q_SCALE


def main() -> None:
  """Integrate a bottom-forced column and compare both budget sides."""

  profiles = constant_density_profiles()
  _, p_edge, temperature, mu, kzz, rho, z, dz = profiles
  q = np.full((1, rho.size), 1.0e-12)
  boundary_density = p_edge[-1] * MU / (R_gas * temperature[-1])
  bottom_conductance = boundary_density * KZZ / (0.5 * dz[-1])

  initial_mass = float(np.sum(rho * dz * q[0]))
  cumulative_influx = 0.0
  rows = [(0.0, initial_mass, initial_mass, 0.0)]
  for step in range(1, N_STEPS + 1):
    previous_bottom = q[0, -1]
    q = vert_diffusion(
      q,
      Q_BOTTOM,
      DT,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
    )
    average_bottom = 0.5 * (previous_bottom + q[0, -1])
    cumulative_influx += DT * bottom_conductance * (Q_BOTTOM - average_bottom)
    mass = float(np.sum(rho * dz * q[0]))
    expected_mass = initial_mass + cumulative_influx
    rows.append((step * DT, mass, expected_mass, mass - expected_mass))

  scale = max(abs(rows[-1][2] - initial_mass), np.finfo(float).tiny)
  relative_residual = abs(rows[-1][3]) / scale
  print(f"final relative budget residual: {relative_residual:.6e}")

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "diffusion_flux_budget.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("time_s", "numerical_mass", "budget_mass", "residual"))
    writer.writerows(rows)

  values = np.asarray(rows)
  figure, (mass_axis, residual_axis) = plt.subplots(2, 1, figsize=(7.0, 7.0))
  mass_axis.plot(values[:, 0], values[:, 1], label="summed column mass")
  mass_axis.plot(values[:, 0], values[:, 2], "--", label="integrated boundary flux")
  mass_axis.set_ylabel(r"Tracer column [g cm$^{-2}$]")
  mass_axis.grid(alpha=0.25)
  mass_axis.legend()
  residual_axis.plot(values[:, 0], values[:, 3])
  residual_axis.set_xlabel("Time [s]")
  residual_axis.set_ylabel("Budget residual")
  residual_axis.grid(alpha=0.25)
  figure.tight_layout()
  plot_path = output_directory / "diffusion_flux_budget.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
