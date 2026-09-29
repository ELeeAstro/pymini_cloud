"""Restore a complete pymini_cloud state from a NetCDF checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from netCDF4 import Dataset
import numpy as np

from .data_atmosphere import Atmosphere
from .data_cld_sp import CloudSpecies, CloudState, FORTRAN_SPECIES_DATA


@dataclass(frozen=True, slots=True)
class RestartState:
  """Atmosphere, cloud tracers, and clock restored from one checkpoint."""

  atmosphere: Atmosphere
  cloud: CloudState
  settling_velocity: np.ndarray | None
  time: float
  step: int


def _array(group: Any, name: str) -> np.ndarray:
  try:
    values = group.variables[name][:]
  except KeyError as error:
    raise ValueError(f"restart file is missing variable {group.path}/{name}") from error
  return np.ascontiguousarray(np.asarray(values, dtype=np.float64))


def _strings(group: Any, name: str) -> tuple[str, ...]:
  try:
    values = group.variables[name][:]
  except KeyError as error:
    raise ValueError(f"restart file is missing variable {group.path}/{name}") from error
  return tuple(str(value) for value in np.asarray(values).tolist())


def _optional_array(group: Any, name: str) -> np.ndarray | None:
  if name not in group.variables:
    return None
  return _array(group, name)


def read_restart(path: str | Path) -> RestartState:
  """Read atmosphere, cloud state, species properties, time, and step."""

  restart_path = Path(path).expanduser()
  if not restart_path.is_file():
    raise FileNotFoundError(f"restart file not found: {restart_path}")

  with Dataset(restart_path, "r") as dataset:
    try:
      atmosphere_group = dataset.groups["atmosphere"]
      cloud_group = dataset.groups["cloud"]
    except KeyError as error:
      raise ValueError("restart file must contain atmosphere and cloud groups") from error

    atmosphere = Atmosphere(
      T=_array(atmosphere_group, "temperature"),
      p=_array(atmosphere_group, "pressure"),
      p_edge=_array(atmosphere_group, "pressure_edge"),
      g=_array(atmosphere_group, "gravity"),
      Kzz=_array(atmosphere_group, "eddy_diffusion_coefficient"),
      mu=_array(atmosphere_group, "mean_molar_mass"),
      VMR=_array(atmosphere_group, "background_vmr"),
      bg_species=_strings(atmosphere_group, "background_species_name"),
    )
    atmosphere.set_vertical_grid(
      _array(atmosphere_group, "altitude_edge"),
      _array(atmosphere_group, "altitude"),
      _array(atmosphere_group, "layer_thickness"),
    )
    atmosphere.set_layer_densities(
      _array(atmosphere_group, "number_density"),
      _array(atmosphere_group, "mass_density"),
      _array(atmosphere_group, "thermal_velocity"),
    )
    atmosphere.set_viscosity(_array(atmosphere_group, "dynamic_viscosity"))
    atmosphere.set_gas_transport(
      _array(atmosphere_group, "kinematic_viscosity"),
      _array(atmosphere_group, "mean_free_path"),
    )
    atmosphere.set_conductivity(
      _array(atmosphere_group, "thermal_conductivity")
    )

    names = _strings(cloud_group, "species_name")
    vmr_bottom = _optional_array(cloud_group, "vmr_bottom")
    if vmr_bottom is None:
      # Schema 1.0 stored a mass mixing ratio under q_v_bottom. Convert it
      # back to VMR so old output remains usable as a restart.
      legacy_bottom = _array(cloud_group, "q_v_bottom")
      vmr_bottom = np.empty_like(legacy_bottom)
      for index, name in enumerate(names):
        try:
          vapour_mass = FORTRAN_SPECIES_DATA[name][2]
        except KeyError as error:
          raise ValueError(
            f"legacy restart lacks material data for species {name!r}"
          ) from error
        vmr_bottom[index] = (
          legacy_bottom[index] * atmosphere.mu[-1] / vapour_mass
        )

    property_names = {
      "condensate_density": "condensate_density",
      "condensate_molar_mass": "condensate_molar_mass",
      "vapour_molar_mass": "vapour_molar_mass",
      "vapour_to_condensate_mass_ratio": "vapour_to_condensate_mass_ratio",
      "condensation_efficiency": "condensation_efficiency",
      "nucleation_efficiency": "nucleation_efficiency",
      "nucleation_cluster_size": "nucleation_cluster_size",
      "seed_radius": "seed_radius",
    }
    stored = {
      field: _optional_array(cloud_group, variable)
      for field, variable in property_names.items()
    }
    nucleation_models = (
      _strings(cloud_group, "nucleation_model")
      if "nucleation_model" in cloud_group.variables
      else ()
    )
    optical_constants = (
      _strings(cloud_group, "optical_constants")
      if "optical_constants" in cloud_group.variables
      else ()
    )

    species: list[CloudSpecies] = []
    for index, name in enumerate(names):
      overrides: dict[str, object] = {}
      for field, values in stored.items():
        if values is not None:
          overrides[field] = float(values[index])
      if nucleation_models:
        overrides["nucleation_model"] = nucleation_models[index]
      if optical_constants and optical_constants[index]:
        overrides["optical_constants"] = optical_constants[index]
      species.append(
        CloudSpecies.from_fortran_defaults(
          name,
          float(vmr_bottom[index]),
          **overrides,
        )
      )

    cloud = CloudState(
      species=tuple(species),
      q_v=_array(cloud_group, "q_v"),
      q_0=_array(cloud_group, "q_0"),
      q_1=_array(cloud_group, "q_1"),
      q_2=_array(cloud_group, "q_2"),
      q_floor=float(getattr(dataset, "q_floor", 1.0e-30)),
    )
    settling_velocity = _optional_array(cloud_group, "settling_velocity")
    if settling_velocity is not None:
      expected_shape = (cloud.n_species + 2, atmosphere.n_layers)
      if settling_velocity.shape != expected_shape:
        raise ValueError(
          "restart settling_velocity must have shape "
          f"{expected_shape}, got {settling_velocity.shape}"
        )
    time = float(getattr(dataset, "model_time", 0.0))
    step = int(getattr(dataset, "model_step", 0))

  if not np.isfinite(time) or time < 0.0 or step < 0:
    raise ValueError("restart model_time and model_step must be non-negative")
  return RestartState(
    atmosphere=atmosphere,
    cloud=cloud,
    settling_velocity=settling_velocity,
    time=time,
    step=step,
  )
