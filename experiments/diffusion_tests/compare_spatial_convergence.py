"""Measure spatial convergence on uniform and stretched grids."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, KZZ, Q_SCALE, TIME, constant_density_profiles, evolve


LAYER_COUNTS = (24, 48, 96, 192)
STRETCHES = {"uniform": 1.0, "stretched": 1.5}


def run_series(stretch: float) -> list[tuple[int, float, float, float]]:
  """Return layer count, representative spacing, L2 error, and max error."""

  rows = []
  wave_number = 0.5 * np.pi / COLUMN_DEPTH
  for n_layers in LAYER_COUNTS:
    profiles = constant_density_profiles(n_layers, stretch)
    depth, _, _, _, _, _, _, dz = profiles
    initial = Q_SCALE * np.cos(wave_number * depth)
    dt = TIME / (4 * n_layers)
    numerical = evolve(initial, profiles, time=TIME, dt=dt)[0]
    analytical = initial * np.exp(-KZZ * wave_number**2 * TIME)
    normalized_error = (numerical - analytical) / Q_SCALE
    weights = dz / np.sum(dz)
    l2_error = float(np.sqrt(np.sum(weights * normalized_error**2)))
    max_error = float(np.max(np.abs(normalized_error)))
    rows.append((n_layers, COLUMN_DEPTH / n_layers, l2_error, max_error))
  return rows


def convergence_order(rows: list[tuple[int, float, float, float]], column: int) -> float:
  """Fit the log-log convergence order for one error column."""

  spacing = np.array([row[1] for row in rows])
  error = np.array([row[column] for row in rows])
  return float(np.polyfit(np.log(spacing), np.log(error), 1)[0])


def main() -> None:
  """Run the spatial refinement study and write a plot and CSV table."""

  output_directory = Path(__file__).resolve().parent
  results = {name: run_series(stretch) for name, stretch in STRETCHES.items()}

  csv_path = output_directory / "diffusion_spatial_convergence.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("grid", "n_layers", "mean_dz_cm", "l2_error", "linf_error"))
    for name, rows in results.items():
      for row in rows:
        writer.writerow((name, *row))

  figure, axis = plt.subplots(figsize=(6.8, 5.2))
  for name, rows in results.items():
    spacing = np.array([row[1] for row in rows])
    l2_error = np.array([row[2] for row in rows])
    order = convergence_order(rows, 2)
    axis.loglog(spacing, l2_error, "o-", label=f"{name}: order {order:.2f}")
    print(
      f"{name:9s}: L2 order={order:.3f}, "
      f"Linf order={convergence_order(rows, 3):.3f}"
    )
  reference_spacing = np.array([COLUMN_DEPTH / LAYER_COUNTS[-1], COLUMN_DEPTH / LAYER_COUNTS[0]])
  reference_error = results["uniform"][-1][2] * (reference_spacing / reference_spacing[0])**2
  axis.loglog(reference_spacing, reference_error, "k--", alpha=0.6, label="second order")
  axis.set_xlabel(r"Mean layer thickness $L/N$ [cm]")
  axis.set_ylabel(r"Normalized weighted $L_2$ error")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()
  plot_path = output_directory / "diffusion_spatial_convergence.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
