"""Compare Koren and monotonized-central MUSCL limiters."""

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
  COLUMN_DEPTH,
  Q_SCALE,
  VELOCITY,
  constant_density_profiles,
  evolve,
  gaussian,
)


LIMITERS = ("koren", "mc")
TIME = 0.20 * COLUMN_DEPTH / VELOCITY
DT = TIME / 400


def normalized_errors(
  numerical: np.ndarray,
  analytical: np.ndarray,
) -> tuple[float, float, float]:
  """Return normalized L1, L2, and maximum errors."""

  difference = np.abs(numerical - analytical) / Q_SCALE
  return (
    float(np.mean(difference)),
    float(np.sqrt(np.mean(difference**2))),
    float(np.max(difference)),
  )


def main() -> None:
  """Compare both limiters for a smooth pulse and a discontinuous front."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  gaussian_initial = gaussian(
    depth,
    0.30 * COLUMN_DEPTH,
    0.06 * COLUMN_DEPTH,
  )
  gaussian_exact = gaussian(
    depth,
    0.30 * COLUMN_DEPTH + VELOCITY * TIME,
    0.06 * COLUMN_DEPTH,
  )
  step_initial = np.zeros_like(depth)
  step_exact = np.where(depth < VELOCITY * TIME, Q_SCALE, 0.0)

  rows = []
  results: dict[tuple[str, str], np.ndarray] = {}
  for limiter in LIMITERS:
    gaussian_numerical = evolve(
      gaussian_initial,
      VELOCITY,
      profiles,
      time=TIME,
      dt=DT,
      limiter=limiter,
    )[0]
    step_numerical = evolve(
      step_initial,
      VELOCITY,
      profiles,
      time=TIME,
      dt=DT,
      q_top=Q_SCALE,
      limiter=limiter,
    )[0]
    results[("gaussian", limiter)] = gaussian_numerical
    results[("step", limiter)] = step_numerical
    for case_name, numerical, analytical in (
      ("gaussian", gaussian_numerical, gaussian_exact),
      ("step", step_numerical, step_exact),
    ):
      l1_error, l2_error, max_error = normalized_errors(numerical, analytical)
      rows.append((case_name, limiter, l1_error, l2_error, max_error))
      print(
        f"{case_name:8s} {limiter:5s}: L1={l1_error:.6e}, "
        f"L2={l2_error:.6e}, Linf={max_error:.6e}"
      )

  output_directory = Path(__file__).resolve().parent
  csv_path = output_directory / "advection_limiter_comparison.csv"
  with csv_path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("case", "limiter", "l1_error", "l2_error", "linf_error"))
    writer.writerows(rows)

  figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
  cases = (
    ("Gaussian pulse", "gaussian", gaussian_exact),
    ("Inflow step", "step", step_exact),
  )
  for axis, (title, case_name, analytical) in zip(axes, cases, strict=True):
    axis.plot(analytical / Q_SCALE, depth / COLUMN_DEPTH, "k--", label="analytical")
    for limiter in LIMITERS:
      axis.plot(
        results[(case_name, limiter)] / Q_SCALE,
        depth / COLUMN_DEPTH,
        marker="o",
        markersize=2.5,
        markevery=3,
        label=limiter,
      )
    axis.set_title(title)
    axis.set_xlabel(r"Normalized mixing ratio $q/q_0$")
    axis.grid(alpha=0.25)
    axis.invert_yaxis()
  axes[0].set_ylabel(r"Normalized downward depth $x/L$")
  axes[0].legend()
  figure.suptitle("MUSCL limiter comparison")
  figure.tight_layout()
  plot_path = output_directory / "advection_limiter_comparison.png"
  figure.savefig(plot_path, dpi=200)
  plt.close(figure)
  print(f"saved {csv_path}")
  print(f"saved {plot_path}")


if __name__ == "__main__":
  main()
