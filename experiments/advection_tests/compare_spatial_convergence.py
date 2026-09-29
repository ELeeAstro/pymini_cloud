"""Measure the upwind scheme's spatial convergence rate."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import COLUMN_DEPTH, Q_SCALE, VELOCITY, constant_density_profiles, evolve


LAYER_COUNTS = (96, 192, 384, 768)
STRETCHES = {"uniform": 1.0, "stretched": 1.25}
CENTRE = 0.30 * COLUMN_DEPTH
HALF_WIDTH = 0.18 * COLUMN_DEPTH
TIME = 0.20 * COLUMN_DEPTH / VELOCITY
COURANT_TARGET = 0.25


def compact_bump(depth: np.ndarray, centre: float) -> np.ndarray:
  """Return a smooth compactly supported tracer pulse."""

  coordinate = (depth - centre) / HALF_WIDTH
  result = np.zeros_like(depth)
  inside = np.abs(coordinate) < 1.0
  result[inside] = Q_SCALE * (1.0 - coordinate[inside]**2)**4
  return result


def run_series(stretch: float) -> list[tuple[int, float, int, float, float]]:
  """Return grid size, mean spacing, steps, L1 error, and L2 error."""

  rows = []
  for n_layers in LAYER_COUNTS:
    profiles = constant_density_profiles(n_layers, stretch)
    depth, _, dz = profiles
    initial = compact_bump(depth, CENTRE)
    analytical = compact_bump(depth, CENTRE + VELOCITY * TIME)
    n_steps = int(np.ceil(VELOCITY * TIME / (COURANT_TARGET * np.min(dz))))
    numerical = evolve(
      initial,
      VELOCITY,
      profiles,
      time=TIME,
      dt=TIME / n_steps,
    )[0]
    normalized_error = np.abs(numerical - analytical) / Q_SCALE
    weights = dz / np.sum(dz)
    l1_error = float(np.sum(weights * normalized_error))
    l2_error = float(np.sqrt(np.sum(weights * normalized_error**2)))
    rows.append((n_layers, COLUMN_DEPTH / n_layers, n_steps, l1_error, l2_error))
  return rows


def order(rows: list[tuple[int, float, int, float, float]], column: int) -> float:
  """Fit a log-log convergence order for one error column."""

  spacing = np.array([row[1] for row in rows])
  error = np.array([row[column] for row in rows])
  return float(np.polyfit(np.log(spacing), np.log(error), 1)[0])


def main() -> None:
  """Run uniform and stretched-grid refinement studies."""

  results = {name: run_series(stretch) for name, stretch in STRETCHES.items()}
  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "advection_spatial_convergence.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("grid", "n_layers", "mean_dz_cm", "n_steps", "l1_error", "l2_error"))
    for name, rows in results.items():
      for row in rows:
        writer.writerow((name, *row))

  figure, axis = plt.subplots(figsize=(6.8, 5.2))
  for name, rows in results.items():
    spacing = np.array([row[1] for row in rows])
    l1_error = np.array([row[3] for row in rows])
    fitted_order = order(rows, 3)
    axis.loglog(spacing, l1_error, "o-", label=f"{name}: order {fitted_order:.2f}")
    print(
      f"{name:9s}: L1 order={fitted_order:.3f}, "
      f"L2 order={order(rows, 4):.3f}"
    )

  reference_spacing = np.array([COLUMN_DEPTH / LAYER_COUNTS[-1], COLUMN_DEPTH / LAYER_COUNTS[0]])
  reference_error = results["uniform"][-1][3] * (
    reference_spacing / reference_spacing[0]
  )**2
  axis.loglog(reference_spacing, reference_error, "k--", alpha=0.6, label="second order")
  axis.set_xlabel(r"Mean layer thickness $L/N$ [cm]")
  axis.set_ylabel(r"Normalized weighted $L_1$ error")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()
  plot_path = output_directory / "advection_spatial_convergence.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
