"""Manufactured-solution test on the configured nonuniform atmosphere."""

from pathlib import Path

import matplotlib
import numpy as np
from scipy.interpolate import PchipInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.main import run
from pymini_cloud.vert_diffusion import vert_diffusion


Q_SCALE = 1.0e-7
AMPLITUDE = 0.2
DECAY_TIME = 5.0e3
FINAL_TIME = 1.0e3
DT = 1.0e1


def exact_solution(z: np.ndarray | float, column_height: float, time: float) -> np.ndarray:
  """Return the positive manufactured mixing-ratio profile."""

  return Q_SCALE * (
    1.0 + AMPLITUDE * np.cos(np.pi * np.asarray(z) / column_height)
  ) * np.exp(-time / DECAY_TIME)


def manufactured_source(
  z: np.ndarray,
  rho: np.ndarray,
  kzz: np.ndarray,
  column_height: float,
  time: float,
) -> np.ndarray:
  """Return the source making ``exact_solution`` satisfy the continuum PDE."""

  q = exact_solution(z, column_height, time)
  phase = np.pi * z / column_height
  common = Q_SCALE * AMPLITUDE * np.exp(-time / DECAY_TIME)
  q_z = -common * np.pi / column_height * np.sin(phase)
  q_zz = -common * (np.pi / column_height)**2 * np.cos(phase)

  increasing = np.argsort(z)
  diffusivity_density = rho * kzz
  interpolator = PchipInterpolator(z[increasing], diffusivity_density[increasing])
  coefficient_gradient = interpolator.derivative()(z)
  diffusion_tendency = (coefficient_gradient * q_z + diffusivity_density * q_zz) / rho
  return -q / DECAY_TIME - diffusion_tendency


def main() -> None:
  """Run the manufactured solution on the project atmosphere and plot errors."""

  setup_path = Path(__file__).resolve().parents[1] / "Y_400K_KCl" / "setup.yaml"
  atmosphere = run(setup_path, n_steps=0, write_output=False).atmosphere
  if atmosphere.z is None or atmosphere.z_edge is None or atmosphere.dz is None:
    raise RuntimeError("atmospheric vertical grid was not initialized")
  if atmosphere.rho is None:
    raise RuntimeError("atmospheric density was not initialized")

  column_height = float(atmosphere.z_edge[0])
  q = exact_solution(atmosphere.z, column_height, 0.0)[None, :]
  n_steps = int(round(FINAL_TIME / DT))
  for step in range(n_steps):
    time = step * DT
    next_time = (step + 1) * DT
    source = manufactured_source(
      atmosphere.z, atmosphere.rho, atmosphere.Kzz, column_height, time
    )
    source_next = manufactured_source(
      atmosphere.z, atmosphere.rho, atmosphere.Kzz, column_height, next_time
    )
    q = vert_diffusion(
      q,
      exact_solution(0.0, column_height, time),
      DT,
      atmosphere.p_edge,
      atmosphere.T,
      atmosphere.mu,
      atmosphere.Kzz,
      atmosphere.rho,
      atmosphere.z,
      atmosphere.dz,
      q_bottom_next=exact_solution(0.0, column_height, next_time),
      source=source,
      source_next=source_next,
    )

  analytical = exact_solution(atmosphere.z, column_height, FINAL_TIME)
  normalized_error = (q[0] - analytical) / Q_SCALE
  l2_error = float(np.sqrt(np.mean(normalized_error**2)))
  max_error = float(np.max(np.abs(normalized_error)))
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")

  pressure_bar = atmosphere.p / 1.0e6
  figure, (profile_axis, error_axis) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
  profile_axis.plot(analytical / Q_SCALE, pressure_bar, label="manufactured exact")
  profile_axis.plot(q[0] / Q_SCALE, pressure_bar, "o", markersize=3, markevery=3, label="numerical")
  profile_axis.set_xlabel(r"Normalized mixing ratio $q/q_0$")
  profile_axis.set_ylabel("Pressure [bar]")
  profile_axis.legend()
  profile_axis.grid(alpha=0.25)
  error_axis.plot(np.maximum(np.abs(normalized_error), 1.0e-16), pressure_bar)
  error_axis.set_xscale("log")
  error_axis.set_xlabel("Absolute normalized error")
  error_axis.grid(alpha=0.25, which="both")
  for axis in (profile_axis, error_axis):
    axis.set_yscale("log")
    axis.invert_yaxis()
  figure.suptitle(
    f"Manufactured solution on the configured atmosphere\n"
    f"$L_2={l2_error:.2e}$, $L_\\infty={max_error:.2e}$"
  )
  figure.tight_layout()
  output = Path(__file__).with_name("diffusion_manufactured.png")
  figure.savefig(output, dpi=200)
  plt.close(figure)
  print(f"saved {output.resolve()}")


if __name__ == "__main__":
  main()
