"""NetCDF output routines for model diagnostics and restart state."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING, Any
import warnings

from netCDF4 import Dataset
import numpy as np
import yaml


if TYPE_CHECKING:
  from .io_read import Setup
  from .main import ModelState


def _write_array(
  group: Any,
  name: str,
  dimensions: tuple[str, ...],
  values: np.ndarray,
  *,
  units: str,
  long_name: str,
) -> None:
  """Create one compressed double-precision NetCDF variable."""

  variable = group.createVariable(
    name,
    "f8",
    dimensions,
    compression="zlib",
    complevel=4,
    shuffle=True,
  )
  variable.units = units
  variable.long_name = long_name
  # netCDF4 currently triggers this NumPy 2.5 deprecation internally while
  # assigning an array; the public slice-assignment API remains correct.
  with warnings.catch_warnings():
    warnings.filterwarnings(
      "ignore",
      message="Setting the shape on a NumPy array has been deprecated.*",
      category=DeprecationWarning,
    )
    variable[:] = np.asarray(values, dtype=np.float64)


def _write_string_array(
  group: Any,
  name: str,
  dimension: str,
  values: tuple[str, ...],
  *,
  long_name: str,
) -> None:
  """Create one variable-length UTF-8 string variable."""

  variable = group.createVariable(name, str, (dimension,))
  variable.long_name = long_name
  variable[:] = np.asarray(values, dtype=object)


def _json_value(value: Any) -> str:
  """Serialize scalar-or-vector configuration values consistently."""

  if isinstance(value, tuple):
    value = list(value)
  return json.dumps(value)


def _write_dataset(
  path: Path,
  state: ModelState,
  setup: Setup,
) -> None:
  """Write the complete final state to an already selected path."""

  atmosphere = state.atmosphere
  cloud = state.cloud
  required_profiles = {
    "z_edge": atmosphere.z_edge,
    "z": atmosphere.z,
    "dz": atmosphere.dz,
    "nd": atmosphere.nd,
    "rho": atmosphere.rho,
    "cT": atmosphere.cT,
    "eta": atmosphere.eta,
    "nu": atmosphere.nu,
    "mfp": atmosphere.mfp,
    "kappa": atmosphere.kappa,
  }
  missing = [name for name, value in required_profiles.items() if value is None]
  if missing:
    raise RuntimeError(
      "cannot write output before atmospheric initialization; missing "
      + ", ".join(missing)
    )

  with Dataset(path, "w", format="NETCDF4") as dataset:
    dataset.title = "pymini_cloud final atmospheric state"
    dataset.schema_version = "1.1"
    dataset.model_time = np.float64(state.time)
    dataset.model_step = np.int64(state.step)
    dataset.q_floor = np.float64(cloud.q_floor)
    dataset.setup_yaml = yaml.safe_dump(setup.raw, sort_keys=False)
    dataset.setup_file = str(setup.path)

    dataset.createDimension("layer", atmosphere.n_layers)
    dataset.createDimension("edge", atmosphere.n_layers + 1)
    dataset.createDimension("background_species", len(atmosphere.bg_species))
    dataset.createDimension("cloud_species", cloud.n_species)
    dataset.createDimension("settling_moment", cloud.n_species + 2)

    atmosphere_group = dataset.createGroup("atmosphere")
    _write_array(
      atmosphere_group,
      "pressure",
      ("layer",),
      atmosphere.p,
      units="dyn cm-2",
      long_name="layer pressure",
    )
    _write_array(
      atmosphere_group,
      "pressure_edge",
      ("edge",),
      atmosphere.p_edge,
      units="dyn cm-2",
      long_name="pressure at layer interfaces",
    )
    _write_array(
      atmosphere_group,
      "temperature",
      ("layer",),
      atmosphere.T,
      units="K",
      long_name="layer temperature",
    )
    _write_array(
      atmosphere_group,
      "gravity",
      ("layer",),
      atmosphere.g,
      units="cm s-2",
      long_name="gravitational acceleration",
    )
    _write_array(
      atmosphere_group,
      "eddy_diffusion_coefficient",
      ("layer",),
      atmosphere.Kzz,
      units="cm2 s-1",
      long_name="eddy-diffusion coefficient",
    )
    _write_array(
      atmosphere_group,
      "mean_molar_mass",
      ("layer",),
      atmosphere.mu,
      units="g mol-1",
      long_name="mean atmospheric molar mass",
    )
    _write_string_array(
      atmosphere_group,
      "background_species_name",
      "background_species",
      atmosphere.bg_species,
      long_name="background gas species name",
    )
    _write_array(
      atmosphere_group,
      "background_vmr",
      ("background_species", "layer"),
      atmosphere.VMR,
      units="1",
      long_name="background gas volume mixing ratio",
    )
    _write_array(
      atmosphere_group,
      "altitude_edge",
      ("edge",),
      required_profiles["z_edge"],
      units="cm",
      long_name="altitude at layer interfaces",
    )
    _write_array(
      atmosphere_group,
      "altitude",
      ("layer",),
      required_profiles["z"],
      units="cm",
      long_name="layer-center altitude",
    )
    _write_array(
      atmosphere_group,
      "layer_thickness",
      ("layer",),
      required_profiles["dz"],
      units="cm",
      long_name="geometric layer thickness",
    )
    _write_array(
      atmosphere_group,
      "number_density",
      ("layer",),
      required_profiles["nd"],
      units="cm-3",
      long_name="atmospheric number density",
    )
    _write_array(
      atmosphere_group,
      "mass_density",
      ("layer",),
      required_profiles["rho"],
      units="g cm-3",
      long_name="atmospheric mass density",
    )
    _write_array(
      atmosphere_group,
      "thermal_velocity",
      ("layer",),
      required_profiles["cT"],
      units="cm s-1",
      long_name="mean atmospheric thermal velocity",
    )
    _write_array(
      atmosphere_group,
      "dynamic_viscosity",
      ("layer",),
      required_profiles["eta"],
      units="g cm-1 s-1",
      long_name="background-gas dynamic viscosity",
    )
    _write_array(
      atmosphere_group,
      "kinematic_viscosity",
      ("layer",),
      required_profiles["nu"],
      units="cm2 s-1",
      long_name="background-gas kinematic viscosity",
    )
    _write_array(
      atmosphere_group,
      "mean_free_path",
      ("layer",),
      required_profiles["mfp"],
      units="cm",
      long_name="background-gas molecular mean free path",
    )
    _write_array(
      atmosphere_group,
      "thermal_conductivity",
      ("layer",),
      required_profiles["kappa"],
      units="erg s-1 cm-1 K-1",
      long_name="background-gas thermal conductivity",
    )

    cloud_group = dataset.createGroup("cloud")
    _write_string_array(
      cloud_group,
      "species_name",
      "cloud_species",
      tuple(item.name for item in cloud.species),
      long_name="condensate and vapour species name",
    )
    _write_array(
      cloud_group,
      "vmr_bottom",
      ("cloud_species",),
      np.asarray([item.vmr_bottom for item in cloud.species]),
      units="1",
      long_name="lower-boundary vapour volume mixing ratio",
    )
    _write_array(
      cloud_group,
      "bottom_mass_mixing_ratio",
      ("cloud_species",),
      np.asarray(
        [
          item.bottom_mass_mixing_ratio(atmosphere.mu[-1])
          for item in cloud.species
        ]
      ),
      units="1",
      long_name="lower-boundary vapour mass mixing ratio",
    )
    species_arrays = {
      "condensate_density": (
        [item.condensate_density for item in cloud.species],
        "g cm-3",
        "bulk condensate density",
      ),
      "condensate_molar_mass": (
        [item.condensate_molar_mass for item in cloud.species],
        "g mol-1",
        "condensate formula-unit molar mass",
      ),
      "vapour_molar_mass": (
        [item.vapour_molar_mass for item in cloud.species],
        "g mol-1",
        "vapour or key-reactant molar mass",
      ),
      "vapour_to_condensate_mass_ratio": (
        [item.vapour_to_condensate_mass_ratio for item in cloud.species],
        "1",
        "vapour-to-condensate mass conversion factor",
      ),
      "condensation_efficiency": (
        [item.condensation_efficiency for item in cloud.species],
        "1",
        "condensation sticking efficiency",
      ),
      "nucleation_efficiency": (
        [item.nucleation_efficiency for item in cloud.species],
        "1",
        "nucleation efficiency",
      ),
      "nucleation_cluster_size": (
        [item.nucleation_cluster_size for item in cloud.species],
        "1",
        "modified classical nucleation cluster size",
      ),
      "seed_radius": (
        [item.seed_radius for item in cloud.species],
        "cm",
        "minimum seed radius",
      ),
      "monomer_mass": (
        [item.monomer_mass for item in cloud.species],
        "g",
        "condensate monomer mass",
      ),
      "monomer_volume": (
        [item.monomer_volume for item in cloud.species],
        "cm3",
        "condensate monomer volume",
      ),
      "monomer_radius": (
        [item.monomer_radius for item in cloud.species],
        "cm",
        "equivalent spherical monomer radius",
      ),
      "seed_mass": (
        [item.seed_mass for item in cloud.species],
        "g",
        "seed particle mass",
      ),
    }
    for name, (values, units, long_name) in species_arrays.items():
      _write_array(
        cloud_group,
        name,
        ("cloud_species",),
        np.asarray(values),
        units=units,
        long_name=long_name,
      )
    _write_string_array(
      cloud_group,
      "nucleation_model",
      "cloud_species",
      tuple(item.nucleation_model for item in cloud.species),
      long_name="nucleation model selection",
    )
    _write_string_array(
      cloud_group,
      "optical_constants",
      "cloud_species",
      tuple(item.optical_constants or "" for item in cloud.species),
      long_name="refractive-index data path",
    )
    _write_array(
      cloud_group,
      "q_v",
      ("cloud_species", "layer"),
      cloud.q_v,
      units="1",
      long_name="vapour mass mixing ratio",
    )
    _write_array(
      cloud_group,
      "q_0",
      ("layer",),
      cloud.q_0,
      units="1",
      long_name="cloud zeroth moment mixing ratio",
    )
    _write_array(
      cloud_group,
      "q_1",
      ("cloud_species", "layer"),
      cloud.q_1,
      units="1",
      long_name="cloud species mass mixing ratio",
    )
    _write_array(
      cloud_group,
      "q_2",
      ("layer",),
      cloud.q_2,
      units="1",
      long_name="cloud second moment mixing ratio",
    )
    if state.settling_velocity is not None:
      settling_velocity = np.asarray(state.settling_velocity, dtype=np.float64)
      expected_shape = (cloud.n_species + 2, atmosphere.n_layers)
      if settling_velocity.shape != expected_shape:
        raise ValueError(
          "settling_velocity must have shape "
          f"{expected_shape}, got {settling_velocity.shape}"
        )
      _write_array(
        cloud_group,
        "settling_velocity",
        ("settling_moment", "layer"),
        settling_velocity,
        units="cm s-1",
        long_name="downward settling velocity in q0, q1(:), q2 order",
      )

    configuration = dataset.createGroup("configuration")
    configuration.do_diffusion = np.int8(state.do_diffusion)
    configuration.do_settling = np.int8(state.do_settling)
    configuration.do_opacity = np.int8(state.model_options.do_opacity)
    configuration.metallicity_log10 = np.float64(
      state.model_options.metallicity_log10
    )
    configuration.size_distribution = state.model_options.size_distribution
    configuration.include_condensation = np.int8(
      state.model_options.include_condensation
    )
    configuration.include_nucleation = np.int8(
      state.model_options.include_nucleation
    )
    configuration.include_coagulation = np.int8(
      state.model_options.include_coagulation
    )
    configuration.include_coalescence = np.int8(
      state.model_options.include_coalescence
    )
    configuration.wavelengths = state.model_options.wavelengths or ""
    configuration.t_step = np.float64(state.integrator.t_step)
    configuration.n_step = np.int64(state.integrator.n_step)
    configuration.rtol = np.float64(state.integrator.rtol)
    configuration.atol = np.float64(state.integrator.atol)
    configuration.advection_method = state.advection.method
    configuration.advection_limiter = state.advection.limiter
    configuration.advection_cfl_limit = np.float64(state.advection.cfl_limit)
    configuration.advection_top_boundary = state.advection.top_boundary
    configuration.advection_top_value = _json_value(state.advection.top_value)
    configuration.advection_bottom_boundary = state.advection.bottom_boundary
    configuration.diffusion_theta = np.float64(state.diffusion.theta)
    configuration.diffusion_top_boundary = state.diffusion.top_boundary
    configuration.diffusion_top_flux = _json_value(state.diffusion.top_flux)
    configuration.diffusion_bottom_boundary = state.diffusion.bottom_boundary
    configuration.diffusion_bottom_value = _json_value(
      state.diffusion.bottom_value
    )
    configuration.diffusion_bottom_flux = _json_value(
      state.diffusion.bottom_flux
    )


def write_model_output(
  state: ModelState,
  setup: Setup,
  *,
  output_path: str | Path | None = None,
) -> Path:
  """Write the final model state to ``mini_cloud.output_file``.

  ``output_path`` may override the validated YAML destination, primarily for
  tests and checkpoints. Relative overrides resolve against the setup-file
  directory. The file is written atomically.
  """

  if output_path is None:
    resolved_output = setup.output.file
  else:
    resolved_output = Path(output_path).expanduser()
    if not resolved_output.is_absolute():
      resolved_output = setup.directory / resolved_output
    resolved_output = resolved_output.resolve()
  resolved_output.parent.mkdir(parents=True, exist_ok=True)

  temporary_name: str | None = None
  try:
    with tempfile.NamedTemporaryFile(
      prefix=f".{resolved_output.name}.",
      suffix=".tmp",
      dir=resolved_output.parent,
      delete=False,
    ) as temporary:
      temporary_name = temporary.name
    _write_dataset(
      Path(temporary_name),
      state,
      setup,
    )
    os.replace(temporary_name, resolved_output)
  finally:
    if temporary_name is not None:
      Path(temporary_name).unlink(missing_ok=True)
  return resolved_output
