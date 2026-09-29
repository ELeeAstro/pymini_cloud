"""Plot vapour mixing ratios from a pymini_cloud NetCDF output file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib
import numpy as np
from netCDF4 import Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pymini_cloud.io_read import read_setup


BAR = 1.0e6  # dyne cm^-2


def output_path_from_setup(setup_path: str | Path) -> Path:
  """Resolve ``mini_cloud.output_file`` relative to its YAML file."""

  setup_path = Path(setup_path).expanduser().resolve()
  setup = read_setup(setup_path)
  return setup.output.file


def read_vapour_profiles(
  netcdf_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...], float, int]:
  """Read pressure, vapour profiles, names, time, and step from output."""

  path = Path(netcdf_path).expanduser().resolve()
  if not path.is_file():
    raise FileNotFoundError(f"NetCDF output file not found: {path}")

  with Dataset(path, "r") as dataset:
    try:
      pressure_bar = np.asarray(
        dataset.groups["atmosphere"].variables["pressure"][:],
        dtype=np.float64,
      ) / BAR
      q_v = np.asarray(
        dataset.groups["cloud"].variables["q_v"][:],
        dtype=np.float64,
      )
      raw_names = dataset.groups["cloud"].variables["species_name"][:]
    except KeyError as error:
      raise ValueError(
        f"{path} is not a supported pymini_cloud output file"
      ) from error
    names = tuple(
      value.decode("utf-8") if isinstance(value, bytes) else str(value)
      for value in raw_names
    )
    model_time = float(dataset.getncattr("model_time"))
    model_step = int(dataset.getncattr("model_step"))

  if q_v.shape != (len(names), pressure_bar.size):
    raise ValueError("q_v dimensions do not match species and pressure")
  return pressure_bar, q_v, names, model_time, model_step


def plot_vapour_profiles(
  netcdf_path: str | Path,
  figure_path: str | Path,
) -> Path:
  """Plot every final vapour profile against logarithmic pressure."""

  pressure_bar, q_v, names, model_time, model_step = read_vapour_profiles(
    netcdf_path
  )
  figure, axis = plt.subplots(figsize=(6.4, 5.2))
  for species_index, name in enumerate(names):
    axis.plot(q_v[species_index], pressure_bar, linewidth=2.0, label=name)

  axis.set_xscale("log")
  axis.set_yscale("log")
  axis.invert_yaxis()
  axis.set_xlabel(r"Vapour mass mixing ratio $q_v$")
  axis.set_ylabel("Pressure [bar]")
  axis.set_title(f"Vapour profiles: step {model_step}, t = {model_time:g} s")
  axis.grid(alpha=0.25, which="both")
  axis.legend()
  figure.tight_layout()

  output_path = Path(figure_path).expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=200)
  plt.close(figure)
  return output_path


def main(argv: Sequence[str] | None = None) -> int:
  """Command-line entry point."""

  experiment_directory = Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--setup",
    type=Path,
    default=experiment_directory / "setup.yaml",
    help="setup YAML used to locate output_file",
  )
  parser.add_argument(
    "--input",
    type=Path,
    default=None,
    help="NetCDF file override",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=experiment_directory / "q_v_pressure.png",
    help="output figure path",
  )
  arguments = parser.parse_args(argv)

  netcdf_path = (
    output_path_from_setup(arguments.setup)
    if arguments.input is None
    else arguments.input
  )
  figure_path = plot_vapour_profiles(netcdf_path, arguments.output)
  print(f"read {Path(netcdf_path).resolve()}")
  print(f"saved {figure_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
