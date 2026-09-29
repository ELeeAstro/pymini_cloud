"""Demonstrate diffusion from the fixed lower KCl vapour boundary."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.main import (
  ModelState,
  advance_model_step,
  run,
)


bar = 1.0e6  # Bar to dyne cm^-2
SNAPSHOT_STEPS = (0, 1, 2, 5, 10)


def calculate_snapshots(
  setup_path: str | Path,
) -> tuple[ModelState, dict[int, np.ndarray]]:
  """Run the diffusion-only main loop and retain vapour snapshots."""

  setup_path = Path(setup_path).resolve()
  state = run(setup_path, n_steps=0, write_output=False)
  snapshots = {0: state.cloud.q_v.copy()}
  for step in range(1, SNAPSHOT_STEPS[-1] + 1):
    advance_model_step(state)
    if step in SNAPSHOT_STEPS:
      snapshots[step] = state.cloud.q_v.copy()

  return state, snapshots


def plot_snapshots(
  setup_path: str | Path,
  output_path: str | Path,
) -> tuple[Path, ModelState]:
  """Plot vapour profiles from the diffusion demonstration."""

  state, snapshots = calculate_snapshots(setup_path)
  pressure_bar = state.atmosphere.p / bar

  figure, axis = plt.subplots(figsize=(6.4, 5.2))
  for species_index, species in enumerate(state.cloud.species):
    for step, q_v in snapshots.items():
      axis.plot(
        q_v[species_index],
        pressure_bar,
        linewidth=2.0,
        label=(
          f"{species.name}, step {step} "
          f"({step * state.integrator.t_step:g} s)"
        ),
      )
    axis.axvline(
      species.bottom_mass_mixing_ratio(state.atmosphere.mu[-1]),
      color="black",
      linestyle="--",
      linewidth=1.2,
      label=rf"{species.name} $q_{{v,\mathrm{{bot}}}}$",
    )
  axis.set_xscale("log")
  axis.set_yscale("log")
  axis.invert_yaxis()
  axis.set_xlabel("KCl vapour mass mixing ratio")
  axis.set_ylabel("Pressure [bar]")
  axis.set_title("Vertical diffusion from the lower boundary")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()

  output_path = Path(output_path).resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=200)
  plt.close(figure)
  return output_path, state


def main() -> None:
  """Run the demonstration and save its plot beside this script."""

  experiment_directory = Path(__file__).resolve().parent
  setup_path = experiment_directory / "setup.yaml"
  output_path, state = plot_snapshots(
    setup_path,
    experiment_directory / "diffusion_test.png",
  )
  print(
    "cloud moment maxima after "
    f"{state.step} steps: q_0={np.max(state.cloud.q_0):.6e}, "
    f"q_1={np.max(state.cloud.q_1):.6e}, "
    f"q_2={np.max(state.cloud.q_2):.6e}"
  )
  print(f"saved {output_path}")


if __name__ == "__main__":
  main()
