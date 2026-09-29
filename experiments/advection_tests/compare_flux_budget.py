"""Check conservative redistribution in a column with no boundary flux."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, Q_SCALE, VELOCITY, constant_density_profiles, gaussian
from pymini_cloud.vert_advection import vert_advection


DT = 1.0e2
N_STEPS = 500


def main() -> None:
  """Evolve a variable-speed closed column and track its tracer mass."""

  depth, rho, dz = constant_density_profiles(stretch=1.4)
  initial = 1.0e-12 + gaussian(depth, 0.35 * COLUMN_DEPTH, 0.08 * COLUMN_DEPTH)
  q = initial[None, :]
  velocity_profile = VELOCITY * (
    0.25 + np.sin(np.pi * depth / COLUMN_DEPTH)**2
  )
  velocity_profile[-1] = 0.0
  velocity = velocity_profile[None, :]
  initial_mass = float(np.sum(rho * dz * initial))
  rows = [(0.0, initial_mass, 0.0)]

  for step in range(1, N_STEPS + 1):
    q = vert_advection(q, velocity, DT, rho, dz, q_top=0.0)
    mass = float(np.sum(rho * dz * q[0]))
    rows.append((step * DT, mass, (mass - initial_mass) / initial_mass))

  print(f"final relative mass residual: {rows[-1][2]:.6e}")
  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "advection_flux_budget.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("time_s", "column_mass", "relative_residual"))
    writer.writerows(rows)

  values = np.asarray(rows)
  figure, (mass_axis, residual_axis) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
  mass_axis.plot(values[:, 0], values[:, 1])
  mass_axis.set_ylabel(r"Tracer column [g cm$^{-2}$]")
  mass_axis.grid(alpha=0.25)
  residual_axis.plot(values[:, 0], values[:, 2])
  residual_axis.set_xlabel("Time [s]")
  residual_axis.set_ylabel("Relative mass residual")
  residual_axis.grid(alpha=0.25)
  figure.tight_layout()
  plot_path = output_directory / "advection_flux_budget.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
