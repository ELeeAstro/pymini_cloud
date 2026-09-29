"""Check steady settling flux on the configured 96-layer atmosphere."""

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.main import run
from pymini_cloud.vert_advection import vert_advection


Q_TOP = 1.0e-7
DT = 100.0
N_STEPS = 100


def main() -> None:
  """Test an analytical constant-flux state on the atmospheric grid."""

  setup_path = Path(__file__).resolve().parents[1] / "Y_400K_KCl" / "setup.yaml"
  atmosphere = run(setup_path, n_steps=0, write_output=False).atmosphere
  if atmosphere.rho is None or atmosphere.z is None or atmosphere.z_edge is None:
    raise RuntimeError("atmospheric density and vertical grid were not initialized")
  if atmosphere.dz is None:
    raise RuntimeError("atmospheric layer thickness was not initialized")

  depth_fraction = (atmosphere.z_edge[0] - atmosphere.z) / atmosphere.z_edge[0]
  velocity = 20.0 + 180.0 * depth_fraction
  steady = Q_TOP * atmosphere.rho[0] * velocity[0] / (
    atmosphere.rho * velocity
  )
  q = steady[None, :]
  for _ in range(N_STEPS):
    q = vert_advection(
      q,
      velocity[None, :],
      DT,
      atmosphere.rho,
      atmosphere.dz,
      q_top=Q_TOP,
    )

  normalized_error = np.abs(q[0] - steady) / Q_TOP
  l2_error = float(np.sqrt(np.mean(normalized_error**2)))
  max_error = float(np.max(normalized_error))
  print(f"L2 error:   {l2_error:.6e}")
  print(f"Linf error: {max_error:.6e}")

  pressure_bar = atmosphere.p / 1.0e6
  figure, (profile_axis, error_axis) = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
  profile_axis.plot(steady / Q_TOP, pressure_bar, label="analytical constant-flux state")
  profile_axis.plot(q[0] / Q_TOP, pressure_bar, "o", markersize=3, markevery=3, label="numerical")
  profile_axis.set_xlabel(r"Normalized mixing ratio $q/q_{\rm top}$")
  profile_axis.set_ylabel("Pressure [bar]")
  profile_axis.grid(alpha=0.25)
  profile_axis.legend()
  error_axis.plot(np.maximum(normalized_error, 1.0e-18), pressure_bar)
  error_axis.set_xscale("log")
  error_axis.set_xlabel("Absolute normalized error")
  error_axis.grid(alpha=0.25, which="both")
  for axis in (profile_axis, error_axis):
    axis.set_yscale("log")
    axis.invert_yaxis()
  figure.suptitle(
    "Constant settling flux on the configured atmosphere\n"
    rf"$L_2={l2_error:.2e}$, $L_\infty={max_error:.2e}$"
  )
  figure.tight_layout()
  output = Path(__file__).with_name("advection_atmospheric_steady.png")
  figure.savefig(output, dpi=200)
  plt.close(figure)
  print(f"saved {output.resolve()}")


if __name__ == "__main__":
  main()
