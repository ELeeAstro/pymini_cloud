"""Verify that each cloud moment is advected at its own settling speed."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import (
  COLUMN_DEPTH,
  Q_SCALE,
  AdvectionCase,
  constant_density_profiles,
  errors,
  evolve,
  gaussian,
)


VELOCITIES = np.array([25.0, 75.0, 150.0])
CENTRE = 0.25 * COLUMN_DEPTH
WIDTH = 0.05 * COLUMN_DEPTH
TIME = 0.20 * COLUMN_DEPTH / np.max(VELOCITIES)
DT = TIME / 400


def main() -> None:
  """Advect three identical profiles and compare their distinct translations."""

  profiles = constant_density_profiles()
  depth = profiles[0]
  one_profile = gaussian(depth, CENTRE, WIDTH)
  initial = np.broadcast_to(one_profile, (VELOCITIES.size, depth.size)).copy()
  numerical = evolve(initial, VELOCITIES, profiles, time=TIME, dt=DT)

  figure, axis = plt.subplots(figsize=(7.0, 5.6))
  for moment, velocity in enumerate(VELOCITIES):
    analytical = gaussian(depth, CENTRE + velocity * TIME, WIDTH)
    case = AdvectionCase(
      f"moment {moment}", depth, one_profile, numerical[moment], analytical, TIME, velocity
    )
    l1_error, l2_error, max_error = errors(case)
    print(
      f"v={velocity:6.1f} cm s^-1: L1={l1_error:.6e}, "
      f"L2={l2_error:.6e}, Linf={max_error:.6e}"
    )
    color = f"C{moment}"
    axis.plot(
      analytical / Q_SCALE,
      depth / COLUMN_DEPTH,
      color=color,
      label=rf"exact, $v={velocity:g}$ cm s$^{{-1}}$",
    )
    axis.plot(
      numerical[moment] / Q_SCALE,
      depth / COLUMN_DEPTH,
      linestyle="none",
      marker="o",
      markersize=3,
      markevery=3,
      color=color,
    )

  axis.set_xlabel(r"Normalized mixing ratio $q/q_0$")
  axis.set_ylabel(r"Normalized downward depth $x/L$")
  axis.invert_yaxis()
  axis.grid(alpha=0.25)
  axis.legend()
  figure.suptitle("Independent settling speeds for three cloud moments")
  figure.tight_layout()
  output = Path(__file__).with_name("advection_multiple_velocities.png")
  figure.savefig(output, dpi=200)
  plt.close(figure)
  print(f"saved {output.resolve()}")


if __name__ == "__main__":
  main()
