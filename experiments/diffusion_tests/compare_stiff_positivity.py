"""Probe Crank-Nicolson positivity for increasingly stiff diffusion steps."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, KZZ, Q_SCALE, constant_density_profiles
from pymini_cloud.vert_diffusion import Q_MIN, vert_diffusion


DIFFUSION_NUMBERS = (0.05, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0)


def main() -> None:
  """Advance a sharp step once at each diffusion number."""

  profiles = constant_density_profiles()
  _, p_edge, temperature, mu, kzz, rho, z, dz = profiles
  initial = np.full(rho.size, 1.0e-12)
  initial[rho.size // 2] = Q_SCALE
  initial_mass = float(np.sum(rho * dz * initial))
  dx = COLUMN_DEPTH / initial.size
  rows = []

  for diffusion_number in DIFFUSION_NUMBERS:
    dt = diffusion_number * dx**2 / KZZ
    numerical = vert_diffusion(
      initial[None, :],
      0.0,
      dt,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
      bottom_boundary="zero_flux",
    )[0]
    clipped_cells = int(np.count_nonzero(numerical <= 10.0 * Q_MIN))
    undershoot = max(0.0, 1.0e-12 - float(np.min(numerical))) / Q_SCALE
    overshoot = max(0.0, float(np.max(numerical)) - Q_SCALE) / Q_SCALE
    mass_error = (float(np.sum(rho * dz * numerical)) - initial_mass) / initial_mass
    rows.append((diffusion_number, dt, clipped_cells, undershoot, overshoot, mass_error))
    print(
      f"r={diffusion_number:6.2f}: clipped={clipped_cells:2d}, "
      f"undershoot={undershoot:.3e}, overshoot={overshoot:.3e}, "
      f"mass error={mass_error:.3e}"
    )

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "diffusion_stiff_positivity.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      ("diffusion_number", "dt_s", "clipped_cells", "undershoot", "overshoot", "mass_error")
    )
    writer.writerows(rows)

  values = np.asarray(rows)
  figure, (bounds_axis, mass_axis) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
  bounds_axis.semilogx(values[:, 0], values[:, 2], "o-", label="floor-clipped cells")
  bounds_axis.set_ylabel("Cell count")
  bounds_axis.grid(alpha=0.25, which="both")
  bounds_axis.legend()
  mass_axis.semilogx(values[:, 0], np.abs(values[:, 5]), "o-")
  mass_axis.set_yscale("log")
  mass_axis.set_xlabel(r"Diffusion number $K\Delta t/\Delta x^2$")
  mass_axis.set_ylabel("Absolute relative mass error")
  mass_axis.grid(alpha=0.25, which="both")
  figure.tight_layout()
  plot_path = output_directory / "diffusion_stiff_positivity.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
