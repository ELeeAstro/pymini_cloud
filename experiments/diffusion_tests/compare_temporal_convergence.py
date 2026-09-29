"""Measure Crank-Nicolson temporal convergence at fixed spatial resolution."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, KZZ, Q_SCALE, TIME, constant_density_profiles, evolve


N_LAYERS = 384
STEP_COUNTS = (5, 10, 20, 40, 80)


def main() -> None:
  """Run the timestep refinement study and write a plot and CSV table."""

  profiles = constant_density_profiles(N_LAYERS)
  depth = profiles[0]
  dx = COLUMN_DEPTH / N_LAYERS
  wave_number = 0.5 * np.pi / COLUMN_DEPTH
  initial = Q_SCALE * np.cos(wave_number * depth)
  discrete_rate = -4.0 * KZZ / dx**2 * np.sin(0.5 * wave_number * dx)**2
  analytical = initial * np.exp(discrete_rate * TIME)

  rows = []
  for n_steps in STEP_COUNTS:
    dt = TIME / n_steps
    numerical = evolve(initial, profiles, time=TIME, dt=dt)[0]
    normalized_error = (numerical - analytical) / Q_SCALE
    rows.append(
      (
        n_steps,
        dt,
        float(np.sqrt(np.mean(normalized_error**2))),
        float(np.max(np.abs(normalized_error))),
      )
    )

  dt_values = np.array([row[1] for row in rows])
  l2_errors = np.array([row[2] for row in rows])
  max_errors = np.array([row[3] for row in rows])
  l2_order = float(np.polyfit(np.log(dt_values), np.log(l2_errors), 1)[0])
  max_order = float(np.polyfit(np.log(dt_values), np.log(max_errors), 1)[0])
  print(f"L2 temporal order:   {l2_order:.3f}")
  print(f"Linf temporal order: {max_order:.3f}")

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "diffusion_temporal_convergence.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("n_steps", "dt_s", "l2_error", "linf_error"))
    writer.writerows(rows)

  figure, axis = plt.subplots(figsize=(6.8, 5.2))
  axis.loglog(dt_values, l2_errors, "o-", label=f"$L_2$: order {l2_order:.2f}")
  axis.loglog(dt_values, max_errors, "s-", label=rf"$L_\infty$: order {max_order:.2f}")
  reference = l2_errors[-1] * (dt_values / dt_values[-1])**2
  axis.loglog(dt_values, reference, "k--", alpha=0.6, label="second order")
  axis.set_xlabel(r"Timestep $\Delta t$ [s]")
  axis.set_ylabel("Normalized error")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()
  plot_path = output_directory / "diffusion_temporal_convergence.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
