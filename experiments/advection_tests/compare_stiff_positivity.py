"""Compare MUSCL subcycling and legacy TR-BDF2 at large Courant number."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import Q_SCALE, VELOCITY, constant_density_profiles
from pymini_cloud.vert_advection import QMIN, vert_advection


N_LAYERS = 192
PULSE_CELL = 48
COURANT_NUMBERS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
BACKGROUND = 1.0e-12


def translated_cell_average(
  cell_edges: np.ndarray,
  cell_width: float,
  courant: float,
) -> np.ndarray:
  """Return exact cell averages for a translated one-cell top-hat pulse."""

  pulse_left = (PULSE_CELL + courant) * cell_width
  pulse_right = pulse_left + cell_width
  overlap = np.maximum(
    0.0,
    np.minimum(cell_edges[1:], pulse_right)
    - np.maximum(cell_edges[:-1], pulse_left),
  )
  return BACKGROUND + Q_SCALE * overlap / cell_width


def main() -> None:
  """Compare accuracy, positivity, and mass conservation for one large step."""

  depth, rho, dz = constant_density_profiles(N_LAYERS)
  cell_width = dz[0]
  cell_edges = np.linspace(0.0, np.sum(dz), N_LAYERS + 1)
  initial = np.full(N_LAYERS, BACKGROUND)
  initial[PULSE_CELL] += Q_SCALE
  velocity = np.full((1, N_LAYERS), VELOCITY)
  exact_excess_mass = rho[0] * cell_width * Q_SCALE
  rows = []

  for courant in COURANT_NUMBERS:
    dt = courant * cell_width / VELOCITY
    analytical = translated_cell_average(cell_edges, cell_width, courant)
    muscl = vert_advection(
      initial[None, :], velocity, dt, rho, dz, q_top=BACKGROUND
    )[0]
    legacy = vert_advection(
      initial[None, :],
      velocity,
      dt,
      rho,
      dz,
      q_top=BACKGROUND,
      method="upwind_trbdf2",
    )[0]

    muscl_error = np.abs(muscl - analytical) / Q_SCALE
    legacy_error = np.abs(legacy - analytical) / Q_SCALE
    muscl_mass_error = (
      np.sum(rho * dz * (muscl - BACKGROUND)) - exact_excess_mass
    ) / exact_excess_mass
    legacy_mass_error = (
      np.sum(rho * dz * (legacy - BACKGROUND)) - exact_excess_mass
    ) / exact_excess_mass
    rows.append(
      (
        courant,
        dt,
        int(np.count_nonzero(muscl < 0.0)),
        int(np.count_nonzero(legacy <= 10.0 * QMIN)),
        float(np.mean(muscl_error)),
        float(np.mean(legacy_error)),
        float(muscl_mass_error),
        float(legacy_mass_error),
      )
    )
    print(
      f"C={courant:5.1f}: MUSCL L1={rows[-1][4]:.3e}, "
      f"mass={muscl_mass_error:.3e}, negative={rows[-1][2]}; "
      f"legacy L1={rows[-1][5]:.3e}, mass={legacy_mass_error:.3e}, "
      f"clipped={rows[-1][3]}"
    )

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "advection_stiff_positivity.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(
      (
        "courant",
        "dt_s",
        "muscl_negative_cells",
        "legacy_clipped_cells",
        "muscl_l1_error",
        "legacy_l1_error",
        "muscl_mass_error",
        "legacy_mass_error",
      )
    )
    writer.writerows(rows)

  values = np.asarray(rows)
  figure, (error_axis, mass_axis) = plt.subplots(2, 1, figsize=(7.0, 7.0), sharex=True)
  error_axis.loglog(values[:, 0], values[:, 4], "o-", label="MUSCL--SSP-RK2")
  error_axis.loglog(values[:, 0], values[:, 5], "s-", label="legacy TR-BDF2")
  error_axis.set_ylabel(r"Normalized $L_1$ error")
  error_axis.grid(alpha=0.25, which="both")
  error_axis.legend()
  mass_axis.loglog(values[:, 0], np.maximum(np.abs(values[:, 6]), 1.0e-16), "o-")
  mass_axis.loglog(values[:, 0], np.maximum(np.abs(values[:, 7]), 1.0e-16), "s-")
  mass_axis.set_xlabel(r"Outer-step Courant number $v\Delta t/\Delta x$")
  mass_axis.set_ylabel("Absolute excess-mass error")
  mass_axis.grid(alpha=0.25, which="both")
  figure.tight_layout()
  plot_path = output_directory / "advection_stiff_positivity.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
