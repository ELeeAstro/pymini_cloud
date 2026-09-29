"""Measure SSP-RK2 time convergence against a fine-step reference."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, Q_SCALE, VELOCITY, constant_density_profiles, evolve, gaussian


N_LAYERS = 96
STEP_COUNTS = (40, 80, 160, 320, 640)
REFERENCE_STEPS = 5120
TIME = 0.10 * COLUMN_DEPTH / VELOCITY


def main() -> None:
  """Run the timestep-refinement comparison and write CSV and PNG outputs."""

  profiles = constant_density_profiles(N_LAYERS)
  depth, _, dz = profiles
  initial = gaussian(depth, 0.30 * COLUMN_DEPTH, 0.06 * COLUMN_DEPTH)
  reference = evolve(
    initial,
    VELOCITY,
    profiles,
    time=TIME,
    dt=TIME / REFERENCE_STEPS,
  )[0]

  rows = []
  for n_steps in STEP_COUNTS:
    dt = TIME / n_steps
    numerical = evolve(initial, VELOCITY, profiles, time=TIME, dt=dt)[0]
    normalized_error = np.abs(numerical - reference) / Q_SCALE
    rows.append(
      (
        n_steps,
        dt,
        VELOCITY * dt / dz[0],
        float(np.mean(normalized_error)),
        float(np.sqrt(np.mean(normalized_error**2))),
      )
    )

  dt_values = np.array([row[1] for row in rows])
  l1_errors = np.array([row[3] for row in rows])
  l2_errors = np.array([row[4] for row in rows])
  l1_order = float(np.polyfit(np.log(dt_values), np.log(l1_errors), 1)[0])
  l2_order = float(np.polyfit(np.log(dt_values), np.log(l2_errors), 1)[0])
  print(f"L1 temporal order: {l1_order:.3f}")
  print(f"L2 temporal order: {l2_order:.3f}")

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "advection_temporal_convergence.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("n_steps", "dt_s", "courant", "l1_error", "l2_error"))
    writer.writerows(rows)

  figure, axis = plt.subplots(figsize=(6.8, 5.2))
  axis.loglog(dt_values, l1_errors, "o-", label=f"$L_1$: order {l1_order:.2f}")
  axis.loglog(dt_values, l2_errors, "s-", label=f"$L_2$: order {l2_order:.2f}")
  second_order = l1_errors[-1] * (dt_values / dt_values[-1])**2
  axis.loglog(dt_values, second_order, "k--", alpha=0.6, label="second order")
  axis.set_xlabel(r"Timestep $\Delta t$ [s]")
  axis.set_ylabel("Normalized error against fine-step reference")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()
  plot_path = output_directory / "advection_temporal_convergence.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
