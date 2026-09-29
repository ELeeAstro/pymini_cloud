"""Input routines for model configuration and atmospheric profiles."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.interpolate import PchipInterpolator

from .data_atmosphere import Atmosphere
from .data_cld_sp import CloudSpecies


bar = 1.0e6  # Bar to dyne cm^-2


@dataclass(frozen=True, slots=True)
class RestartOptions:
  """Restart selection and resolved checkpoint path."""

  enabled: bool
  file: Path | None


@dataclass(frozen=True, slots=True)
class GridOptions:
  """Fresh-run atmospheric profile and pressure-grid controls."""

  atmosphere_file: Path | None
  p_top: float | None
  p_bottom: float | None
  n_layers: int | None


@dataclass(frozen=True, slots=True)
class OutputOptions:
  """Resolved model-output path."""

  file: Path


@dataclass(frozen=True, slots=True)
class ModelOptions:
  """Validated model-level process and composition controls."""

  metallicity_log10: float = 0.0
  size_distribution: str = "gamma"
  do_opacity: bool = False
  do_diffusion: bool = True
  do_settling: bool = False
  include_condensation: bool = True
  include_nucleation: bool = True
  include_coagulation: bool = True
  include_coalescence: bool = True
  wavelengths: str | None = None


def read_model_options(
  setup: dict[str, Any],
  setup_directory: str | Path | None = None,
) -> ModelOptions:
  """Validate model switches whose physics stages are being built."""

  try:
    raw = setup["mini_cloud"]
  except (KeyError, TypeError) as error:
    raise ValueError("setup must contain a 'mini_cloud' mapping") from error
  deprecated = {
    "metallicity_factor": "metallicity_log10",
    "size_distriution": "size_distribution",
    "inc_cond": "include_condensation",
    "inc_nuc": "include_nucleation",
    "inc_coag": "include_coagulation",
    "inc_coal": "include_coalescence",
  }
  present = [name for name in deprecated if name in raw]
  if present:
    replacements = ", ".join(
      f"{name} -> {deprecated[name]}" for name in present
    )
    raise ValueError(f"deprecated mini_cloud option(s): {replacements}")
  try:
    metallicity = float(raw.get("metallicity_log10", 0.0))
  except (TypeError, ValueError) as error:
    raise ValueError("mini_cloud.metallicity_log10 must be numeric") from error
  if not np.isfinite(metallicity):
    raise ValueError("mini_cloud.metallicity_log10 must be finite")

  distribution = raw.get("size_distribution", "gamma")
  if distribution not in {"gamma", "monodisperse", "lognormal", "exponential"}:
    raise ValueError(
      "mini_cloud.size_distribution must be 'gamma', 'monodisperse', "
      "'lognormal', or 'exponential'"
    )
  switches: dict[str, bool] = {}
  for name, default in (
    ("do_opacity", False),
    ("do_diffusion", True),
    ("do_settling", False),
    ("include_condensation", True),
    ("include_nucleation", True),
    ("include_coagulation", True),
    ("include_coalescence", True),
  ):
    value = raw.get(name, default)
    if not isinstance(value, bool):
      raise ValueError(f"mini_cloud.{name} must be either true or false")
    switches[name] = value

  wavelengths: str | None = None
  if "wavelengths" in raw:
    value = raw["wavelengths"]
    if not isinstance(value, str) or not value.strip():
      raise ValueError("mini_cloud.wavelengths must be a non-empty path")
    path = Path(value.strip()).expanduser()
    if setup_directory is not None and not path.is_absolute():
      path = Path(setup_directory) / path
    wavelengths = str(path.resolve())
  if switches["do_opacity"] and wavelengths is None:
    raise ValueError("mini_cloud.wavelengths is required when do_opacity is true")
  if switches["do_opacity"] and not Path(wavelengths).is_file():
    raise FileNotFoundError(f"wavelength file not found: {wavelengths}")

  return ModelOptions(
    metallicity_log10=metallicity,
    size_distribution=distribution,
    wavelengths=wavelengths,
    **switches,
  )


@dataclass(frozen=True, slots=True)
class AdvectionOptions:
  """Validated configuration for downward settling advection."""

  method: str = "muscl_ssprk3"
  limiter: str = "koren"
  cfl_limit: float = 0.8
  top_boundary: str = "zero_inflow"
  top_value: float | tuple[float, ...] = 0.0
  bottom_boundary: str = "outflow"

  def top_values(self, n_moments: int) -> np.ndarray:
    """Return one upper-boundary mixing ratio per advected moment."""

    if n_moments < 1:
      raise ValueError("n_moments must be positive")
    if self.top_boundary == "zero_inflow":
      return np.zeros(n_moments, dtype=np.float64)
    if isinstance(self.top_value, tuple):
      if len(self.top_value) != n_moments:
        raise ValueError(
          "vertical_advection.top_value must contain one value per "
          f"advected moment; expected {n_moments}, got {len(self.top_value)}"
        )
      return np.asarray(self.top_value, dtype=np.float64)
    return np.full(n_moments, self.top_value, dtype=np.float64)


def read_advection_options(setup: dict[str, Any]) -> AdvectionOptions:
  """Validate and return ``mini_cloud.vertical_advection`` settings."""

  try:
    model_setup = setup["mini_cloud"]
  except (KeyError, TypeError) as error:
    raise ValueError("setup must contain a 'mini_cloud' mapping") from error
  if not isinstance(model_setup, dict):
    raise ValueError("setup must contain a 'mini_cloud' mapping")

  raw_options = model_setup.get("vertical_advection", {})
  if not isinstance(raw_options, dict):
    raise ValueError("mini_cloud.vertical_advection must be a mapping")
  allowed_keys = {
    "method",
    "limiter",
    "cfl_limit",
    "top_boundary",
    "top_value",
    "bottom_boundary",
  }
  unknown_keys = sorted(set(raw_options) - allowed_keys)
  if unknown_keys:
    raise ValueError(
      "unknown mini_cloud.vertical_advection option(s): "
      + ", ".join(unknown_keys)
    )

  method = raw_options.get("method", "muscl_ssprk3")
  if method not in {"muscl_ssprk3", "muscl_ssprk2", "upwind_trbdf2"}:
    raise ValueError(
      "mini_cloud.vertical_advection.method must be 'muscl_ssprk3', "
      "'muscl_ssprk2', or 'upwind_trbdf2'"
    )
  limiter = raw_options.get("limiter", "koren")
  if limiter not in {"koren", "mc"}:
    raise ValueError(
      "mini_cloud.vertical_advection.limiter must be 'koren' or 'mc'"
    )
  try:
    default_cfl = 0.45 if method == "muscl_ssprk2" else 0.8
    cfl_limit = float(raw_options.get("cfl_limit", default_cfl))
  except (TypeError, ValueError) as error:
    raise ValueError(
      "mini_cloud.vertical_advection.cfl_limit must be numeric"
    ) from error
  maximum_cfl = 0.5 if method == "muscl_ssprk2" else 0.85
  if not np.isfinite(cfl_limit) or not 0.0 < cfl_limit <= maximum_cfl:
    raise ValueError(
      "mini_cloud.vertical_advection.cfl_limit must lie in "
      f"(0, {maximum_cfl:g}] for {method}"
    )

  top_boundary = raw_options.get("top_boundary", "zero_inflow")
  if top_boundary not in {"zero_inflow", "fixed_value"}:
    raise ValueError(
      "mini_cloud.vertical_advection.top_boundary must be 'zero_inflow' "
      "or 'fixed_value'"
    )
  if top_boundary == "fixed_value" and "top_value" not in raw_options:
    raise ValueError(
      "mini_cloud.vertical_advection.top_value is required for a "
      "fixed_value top boundary"
    )
  raw_top_value = raw_options.get("top_value", 0.0)
  if isinstance(raw_top_value, bool):
    raise ValueError(
      "mini_cloud.vertical_advection.top_value must be numeric or a list"
    )
  if isinstance(raw_top_value, (list, tuple)):
    if not raw_top_value:
      raise ValueError(
        "mini_cloud.vertical_advection.top_value list cannot be empty"
      )
    try:
      top_array = np.asarray(raw_top_value, dtype=np.float64)
    except (TypeError, ValueError) as error:
      raise ValueError(
        "mini_cloud.vertical_advection.top_value must contain numeric values"
      ) from error
    if top_array.ndim != 1:
      raise ValueError(
        "mini_cloud.vertical_advection.top_value must be a one-dimensional list"
      )
    if np.any(~np.isfinite(top_array)) or np.any(top_array < 0.0):
      raise ValueError(
        "mini_cloud.vertical_advection.top_value must be finite and non-negative"
      )
    top_value: float | tuple[float, ...] = tuple(float(value) for value in top_array)
  else:
    try:
      top_value = float(raw_top_value)
    except (TypeError, ValueError) as error:
      raise ValueError(
        "mini_cloud.vertical_advection.top_value must be numeric or a list"
      ) from error
    if not np.isfinite(top_value) or top_value < 0.0:
      raise ValueError(
        "mini_cloud.vertical_advection.top_value must be finite and non-negative"
      )

  bottom_boundary = raw_options.get("bottom_boundary", "outflow")
  if bottom_boundary not in {"outflow", "zero_flux"}:
    raise ValueError(
      "mini_cloud.vertical_advection.bottom_boundary must be 'outflow' "
      "or 'zero_flux'"
    )
  return AdvectionOptions(
    method=method,
    limiter=limiter,
    cfl_limit=cfl_limit,
    top_boundary=top_boundary,
    top_value=top_value,
    bottom_boundary=bottom_boundary,
  )


BoundaryValue = float | tuple[float, ...]


def _read_boundary_value(
  name: str,
  value: Any,
  *,
  nonnegative: bool,
) -> BoundaryValue:
  """Validate a scalar or one-dimensional boundary-value list."""

  if isinstance(value, bool):
    raise ValueError(f"{name} must be numeric or a list")
  if isinstance(value, (list, tuple)):
    if not value:
      raise ValueError(f"{name} list cannot be empty")
    try:
      array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
      raise ValueError(f"{name} must contain numeric values") from error
    if array.ndim != 1:
      raise ValueError(f"{name} must be a one-dimensional list")
    if np.any(~np.isfinite(array)):
      raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
      raise ValueError(f"{name} must be non-negative")
    return tuple(float(item) for item in array)

  try:
    result = float(value)
  except (TypeError, ValueError) as error:
    raise ValueError(f"{name} must be numeric or a list") from error
  if not np.isfinite(result):
    raise ValueError(f"{name} must be finite")
  if nonnegative and result < 0.0:
    raise ValueError(f"{name} must be non-negative")
  return result


def _expand_boundary_value(
  name: str,
  value: BoundaryValue | np.ndarray,
  n_tracers: int,
  *,
  nonnegative: bool,
) -> np.ndarray:
  """Broadcast a boundary setting to one value per transported tracer."""

  if n_tracers < 1:
    raise ValueError("n_tracers must be positive")
  array = np.asarray(value, dtype=np.float64)
  if array.ndim == 0:
    result = np.full(n_tracers, float(array), dtype=np.float64)
  elif array.shape == (n_tracers,):
    result = np.ascontiguousarray(array)
  else:
    raise ValueError(
      f"{name} must be a scalar or contain one value per transported "
      f"tracer; expected {n_tracers}, got shape {array.shape}"
    )
  if np.any(~np.isfinite(result)):
    raise ValueError(f"{name} must contain only finite values")
  if nonnegative and np.any(result < 0.0):
    raise ValueError(f"{name} must be non-negative")
  return result


@dataclass(frozen=True, slots=True)
class DiffusionOptions:
  """Validated configuration for implicit vertical diffusion."""

  theta: float = 0.5
  top_boundary: str = "zero_flux"
  top_flux: BoundaryValue = 0.0
  bottom_boundary: str = "fixed_value"
  bottom_value: str | BoundaryValue = "from_species"
  bottom_flux: BoundaryValue = 0.0

  def top_flux_values(self, n_tracers: int) -> np.ndarray:
    """Return one signed, downward-positive top flux per tracer."""

    if self.top_boundary == "zero_flux":
      return np.zeros(n_tracers, dtype=np.float64)
    return _expand_boundary_value(
      "vertical_diffusion.top_flux",
      self.top_flux,
      n_tracers,
      nonnegative=False,
    )

  def bottom_flux_values(self, n_tracers: int) -> np.ndarray:
    """Return one signed, downward-positive bottom flux per tracer."""

    if self.bottom_boundary != "fixed_flux":
      return np.zeros(n_tracers, dtype=np.float64)
    return _expand_boundary_value(
      "vertical_diffusion.bottom_flux",
      self.bottom_flux,
      n_tracers,
      nonnegative=False,
    )

  def bottom_values(
    self,
    n_tracers: int,
    runtime_value: BoundaryValue | np.ndarray | None = None,
  ) -> np.ndarray:
    """Return lower mixing ratios, resolving ``from_species`` at runtime."""

    if self.bottom_boundary != "fixed_value":
      return np.zeros(n_tracers, dtype=np.float64)
    if self.bottom_value == "from_species":
      if runtime_value is None:
        raise ValueError(
          "a runtime bottom_value is required when "
          "vertical_diffusion.bottom_value is 'from_species'"
        )
      value = runtime_value
    else:
      value = self.bottom_value
    return _expand_boundary_value(
      "vertical_diffusion.bottom_value",
      value,
      n_tracers,
      nonnegative=True,
    )


def read_diffusion_options(setup: dict[str, Any]) -> DiffusionOptions:
  """Validate and return ``mini_cloud.vertical_diffusion`` settings."""

  try:
    model_setup = setup["mini_cloud"]
  except (KeyError, TypeError) as error:
    raise ValueError("setup must contain a 'mini_cloud' mapping") from error
  if not isinstance(model_setup, dict):
    raise ValueError("setup must contain a 'mini_cloud' mapping")

  raw_options = model_setup.get("vertical_diffusion", {})
  if not isinstance(raw_options, dict):
    raise ValueError("mini_cloud.vertical_diffusion must be a mapping")
  allowed_keys = {
    "theta",
    "top_boundary",
    "top_flux",
    "bottom_boundary",
    "bottom_value",
    "bottom_flux",
  }
  unknown_keys = sorted(set(raw_options) - allowed_keys)
  if unknown_keys:
    raise ValueError(
      "unknown mini_cloud.vertical_diffusion option(s): "
      + ", ".join(unknown_keys)
    )

  try:
    theta = float(raw_options.get("theta", 0.5))
  except (TypeError, ValueError) as error:
    raise ValueError(
      "mini_cloud.vertical_diffusion.theta must be numeric"
    ) from error
  if not np.isfinite(theta) or not 0.5 <= theta <= 1.0:
    raise ValueError(
      "mini_cloud.vertical_diffusion.theta must lie in [0.5, 1.0]"
    )

  top_boundary = raw_options.get("top_boundary", "zero_flux")
  if top_boundary not in {"zero_flux", "fixed_flux"}:
    raise ValueError(
      "mini_cloud.vertical_diffusion.top_boundary must be 'zero_flux' "
      "or 'fixed_flux'"
    )
  if top_boundary == "fixed_flux" and "top_flux" not in raw_options:
    raise ValueError(
      "mini_cloud.vertical_diffusion.top_flux is required for a "
      "fixed_flux top boundary"
    )
  top_flux = _read_boundary_value(
    "mini_cloud.vertical_diffusion.top_flux",
    raw_options.get("top_flux", 0.0),
    nonnegative=False,
  )

  bottom_boundary = raw_options.get("bottom_boundary", "fixed_value")
  if bottom_boundary not in {"fixed_value", "zero_flux", "fixed_flux"}:
    raise ValueError(
      "mini_cloud.vertical_diffusion.bottom_boundary must be 'fixed_value', "
      "'zero_flux', or 'fixed_flux'"
    )
  if bottom_boundary == "fixed_flux" and "bottom_flux" not in raw_options:
    raise ValueError(
      "mini_cloud.vertical_diffusion.bottom_flux is required for a "
      "fixed_flux bottom boundary"
    )

  raw_bottom_value = raw_options.get("bottom_value", "from_species")
  if raw_bottom_value == "from_species":
    bottom_value: str | BoundaryValue = "from_species"
  else:
    bottom_value = _read_boundary_value(
      "mini_cloud.vertical_diffusion.bottom_value",
      raw_bottom_value,
      nonnegative=True,
    )
  bottom_flux = _read_boundary_value(
    "mini_cloud.vertical_diffusion.bottom_flux",
    raw_options.get("bottom_flux", 0.0),
    nonnegative=False,
  )

  return DiffusionOptions(
    theta=theta,
    top_boundary=top_boundary,
    top_flux=top_flux,
    bottom_boundary=bottom_boundary,
    bottom_value=bottom_value,
    bottom_flux=bottom_flux,
  )


@dataclass(frozen=True, slots=True)
class IntegratorOptions:
  """Validated controls for the outer model timestep loop."""

  t_step: float
  n_step: int
  rtol: float
  atol: float


def read_integrator_options(setup: dict[str, Any]) -> IntegratorOptions:
  """Validate and return ``mini_cloud.integrator`` settings."""

  try:
    raw_options = setup["mini_cloud"]["integrator"]
  except (KeyError, TypeError) as error:
    raise ValueError("setup must contain a mini_cloud.integrator mapping") from error
  if not isinstance(raw_options, dict):
    raise ValueError("mini_cloud.integrator must be a mapping")

  try:
    t_step = float(raw_options["t_step"])
    rtol = float(raw_options["rtol"])
    atol = float(raw_options["atol"])
  except KeyError as error:
    raise ValueError(
      f"mini_cloud.integrator.{error.args[0]} is required"
    ) from error
  except (TypeError, ValueError) as error:
    raise ValueError(
      "mini_cloud.integrator t_step, rtol, and atol must be numeric"
    ) from error
  n_step = raw_options.get("n_step")
  if isinstance(n_step, bool) or not isinstance(n_step, int):
    raise ValueError("mini_cloud.integrator.n_step must be an integer")
  if not np.isfinite(t_step) or t_step <= 0.0:
    raise ValueError("mini_cloud.integrator.t_step must be finite and positive")
  if n_step < 1:
    raise ValueError("mini_cloud.integrator.n_step must be positive")
  if not np.isfinite(rtol) or rtol <= 0.0:
    raise ValueError("mini_cloud.integrator.rtol must be finite and positive")
  if not np.isfinite(atol) or atol < 0.0:
    raise ValueError(
      "mini_cloud.integrator.atol must be finite and non-negative"
    )
  return IntegratorOptions(
    t_step=t_step,
    n_step=n_step,
    rtol=rtol,
    atol=atol,
  )


def read_cloud_species(
  setup: dict[str, Any],
  setup_directory: str | Path | None = None,
) -> tuple[CloudSpecies, ...]:
  """Read cloud properties and bottom VMRs using Fortran defaults."""

  try:
    raw_species = setup["mini_cloud"]["species"]
  except (KeyError, TypeError) as error:
    raise ValueError("setup must contain a mini_cloud.species list") from error
  if not isinstance(raw_species, list) or not raw_species:
    raise ValueError("mini_cloud.species must be a non-empty list")

  species: list[CloudSpecies] = []
  for index, raw_item in enumerate(raw_species):
    location = f"mini_cloud.species[{index}]"
    if not isinstance(raw_item, dict):
      raise ValueError(f"{location} must be a mapping")
    allowed_keys = {
      "name",
      "vmr_bottom",
      "condensate_density",
      "condensate_molar_mass",
      "vapour_molar_mass",
      "vapour_to_condensate_mass_ratio",
      "condensation_efficiency",
      "nucleation_model",
      "nucleation_efficiency",
      "nucleation_cluster_size",
      "optical_constants",
      "seed_radius",
    }
    unknown_keys = sorted(set(raw_item) - allowed_keys)
    if unknown_keys:
      raise ValueError(
        f"unknown {location} option(s): " + ", ".join(unknown_keys)
      )
    name = raw_item.get("name")
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f"{location}.name must be a non-empty species name")
    name = name.strip()
    raw_bottom = raw_item.get("vmr_bottom")
    if isinstance(raw_bottom, bool):
      raise ValueError(f"{location}.vmr_bottom must be numeric")
    try:
      vmr_bottom = float(raw_bottom)
    except (TypeError, ValueError) as error:
      raise ValueError(f"{location}.vmr_bottom must be numeric") from error
    if not np.isfinite(vmr_bottom) or vmr_bottom < 0.0:
      raise ValueError(f"{location}.vmr_bottom must be finite and non-negative")

    overrides: dict[str, object] = {}
    numeric_keys = (
      "condensate_density",
      "condensate_molar_mass",
      "vapour_molar_mass",
      "vapour_to_condensate_mass_ratio",
      "condensation_efficiency",
      "nucleation_efficiency",
      "nucleation_cluster_size",
      "seed_radius",
    )
    for key in numeric_keys:
      if key not in raw_item:
        continue
      if isinstance(raw_item[key], bool):
        raise ValueError(f"{location}.{key} must be numeric")
      try:
        overrides[key] = float(raw_item[key])
      except (TypeError, ValueError) as error:
        raise ValueError(f"{location}.{key} must be numeric") from error

    if "nucleation_model" in raw_item:
      model = raw_item["nucleation_model"]
      if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{location}.nucleation_model must be a non-empty string")
      overrides["nucleation_model"] = model.strip()
    if "optical_constants" in raw_item:
      optical_name = raw_item["optical_constants"]
      if not isinstance(optical_name, str) or not optical_name.strip():
        raise ValueError(f"{location}.optical_constants must be a file path")
      optical_path = Path(optical_name.strip()).expanduser()
      if setup_directory is not None and not optical_path.is_absolute():
        optical_path = Path(setup_directory) / optical_path
      overrides["optical_constants"] = str(optical_path.resolve())

    species.append(
      CloudSpecies.from_fortran_defaults(name, vmr_bottom, **overrides)
    )

  names = [item.name for item in species]
  if len(set(names)) != len(names):
    raise ValueError("mini_cloud.species names must be unique")
  return tuple(species)


@dataclass(frozen=True, slots=True)
class Setup:
  """Fully validated model configuration with grouped dot access."""

  path: Path
  restart: RestartOptions
  grid: GridOptions
  physics: ModelOptions
  advection: AdvectionOptions
  diffusion: DiffusionOptions
  integrator: IntegratorOptions
  species: tuple[CloudSpecies, ...]
  output: OutputOptions
  _raw: dict[str, Any] = field(repr=False)

  @property
  def directory(self) -> Path:
    """Directory containing the input YAML file."""

    return self.path.parent

  @property
  def raw(self) -> dict[str, Any]:
    """Return an isolated copy of the original YAML mapping."""

    return copy.deepcopy(self._raw)

  def raw_copy(self) -> dict[str, Any]:
    """Return a mutable copy of the original YAML mapping."""

    return copy.deepcopy(self._raw)


def _resolved_path(value: Any, name: str, directory: Path) -> Path:
  """Validate and resolve one path relative to the setup file."""

  if not isinstance(value, str) or not value.strip():
    raise ValueError(f"{name} must be a non-empty path")
  result = Path(value.strip()).expanduser()
  if not result.is_absolute():
    result = directory / result
  return result.resolve()


def read_setup(path: str | Path) -> Setup:
  """Read and fully validate a YAML setup into nested option dataclasses."""

  setup_path = Path(path).expanduser().resolve()
  if not setup_path.is_file():
    raise FileNotFoundError(f"setup file not found: {setup_path}")

  with setup_path.open(encoding="utf-8") as stream:
    raw_setup = yaml.safe_load(stream)

  if not isinstance(raw_setup, dict):
    raise ValueError("setup file must contain a YAML mapping")
  if not isinstance(raw_setup.get("mini_cloud"), dict):
    raise ValueError("setup file must contain a 'mini_cloud' mapping")
  unknown_top_level = sorted(set(raw_setup) - {"mini_cloud"})
  if unknown_top_level:
    raise ValueError(
      "unknown top-level setup option(s): " + ", ".join(unknown_top_level)
    )
  model = raw_setup["mini_cloud"]
  directory = setup_path.parent
  allowed_model_keys = {
    "restart",
    "restart_file",
    "output_file",
    "T_p_grav_Kzz_mu_VMR",
    "p_top",
    "p_bottom",
    "n_layers",
    "metallicity_log10",
    "size_distribution",
    "wavelengths",
    "do_opacity",
    "do_diffusion",
    "do_settling",
    "vertical_advection",
    "vertical_diffusion",
    "include_condensation",
    "include_nucleation",
    "include_coagulation",
    "include_coalescence",
    # Accepted here only so read_model_options can emit migration guidance.
    "metallicity_factor",
    "size_distriution",
    "inc_cond",
    "inc_nuc",
    "inc_coag",
    "inc_coal",
    "species",
    "integrator",
  }
  unknown_model_keys = sorted(set(model) - allowed_model_keys)
  if unknown_model_keys:
    raise ValueError(
      "unknown mini_cloud option(s): " + ", ".join(unknown_model_keys)
    )

  restart_enabled = model.get("restart")
  if not isinstance(restart_enabled, bool):
    raise ValueError("mini_cloud.restart must be either true or false")
  restart_file = None
  if "restart_file" in model:
    restart_file = _resolved_path(
      model["restart_file"],
      "mini_cloud.restart_file",
      directory,
    )
  if restart_enabled and restart_file is None:
    raise ValueError("mini_cloud.restart_file is required when restart is true")

  atmosphere_file: Path | None = None
  p_top: float | None = None
  p_bottom: float | None = None
  n_layers: int | None = None
  grid_keys = {"T_p_grav_Kzz_mu_VMR", "p_top", "p_bottom", "n_layers"}
  if not restart_enabled or any(name in model for name in grid_keys):
    try:
      atmosphere_file = _resolved_path(
        model["T_p_grav_Kzz_mu_VMR"],
        "mini_cloud.T_p_grav_Kzz_mu_VMR",
        directory,
      )
      p_top = float(model["p_top"])
      p_bottom = float(model["p_bottom"])
      n_layers = model["n_layers"]
    except KeyError as error:
      raise ValueError(
        f"mini_cloud.{error.args[0]} is required for a fresh run"
      ) from error
    except (TypeError, ValueError) as error:
      raise ValueError(
        "mini_cloud.p_top and p_bottom must be numeric"
      ) from error
    if not np.isfinite(p_top) or p_top <= 0.0:
      raise ValueError("mini_cloud.p_top must be finite and positive")
    if not np.isfinite(p_bottom) or p_bottom <= p_top:
      raise ValueError(
        "mini_cloud.p_bottom must be finite and greater than p_top"
      )
    if isinstance(n_layers, bool) or not isinstance(n_layers, int):
      raise ValueError("mini_cloud.n_layers must be an integer")
    if n_layers < 2:
      raise ValueError("mini_cloud.n_layers must be at least 2")

  output_file = _resolved_path(
    model.get("output_file"),
    "mini_cloud.output_file",
    directory,
  )
  if "species" in model:
    species = read_cloud_species(raw_setup, directory)
  elif restart_enabled:
    species = ()
  else:
    raise ValueError("mini_cloud.species is required for a fresh run")

  return Setup(
    path=setup_path,
    restart=RestartOptions(enabled=restart_enabled, file=restart_file),
    grid=GridOptions(
      atmosphere_file=atmosphere_file,
      p_top=p_top,
      p_bottom=p_bottom,
      n_layers=n_layers,
    ),
    physics=read_model_options(raw_setup, directory),
    advection=read_advection_options(raw_setup),
    diffusion=read_diffusion_options(raw_setup),
    integrator=read_integrator_options(raw_setup),
    species=species,
    output=OutputOptions(file=output_file),
    _raw=copy.deepcopy(raw_setup),
  )


def read_atmosphere(
  path: str | Path,
  p_top: float,
  p_bottom: float,
  n_layers: int,
  bg_species: tuple[str, ...] = ("H2", "He"),
) -> Atmosphere:
  """Read and interpolate a one-dimensional atmospheric profile.

  The source columns are pressure [bar], temperature [K], gravity [cm s^-2],
  Kzz [cm^2 s^-1], mean molar mass [g mol^-1], followed by one VMR column
  for each entry in ``bg_species``. The model pressure interfaces are
  logarithmically spaced between ``p_top`` and ``p_bottom``. Source fields
  are PCHIP-interpolated in log pressure to logarithmic-mean layer pressures.
  The nearest source value is held constant outside the source pressure
  range, matching the boundary behavior of the Fortran test model.
  """

  if not np.isfinite(p_top) or p_top <= 0.0:
    raise ValueError("p_top must be finite and positive")
  if not np.isfinite(p_bottom) or p_bottom <= p_top:
    raise ValueError("p_bottom must be finite and greater than p_top")
  if isinstance(n_layers, bool) or not isinstance(n_layers, int):
    raise ValueError("n_layers must be an integer")
  if n_layers < 2:
    raise ValueError("n_layers must be at least 2")

  profile_path = Path(path).expanduser()
  if not profile_path.is_file():
    raise FileNotFoundError(f"atmosphere profile not found: {profile_path}")

  try:
    data = np.loadtxt(
      profile_path,
      delimiter=",",
      comments="#",
      dtype=np.float64,
      ndmin=2,
    )
  except ValueError as error:
    raise ValueError(
      f"could not parse atmosphere profile {profile_path}: {error}"
    ) from error

  expected_columns = 5 + len(bg_species)
  if data.shape[1] != expected_columns:
    raise ValueError(
      f"atmosphere profile {profile_path} has {data.shape[1]} columns; "
      f"expected {expected_columns}"
    )
  if data.shape[0] < 2:
    raise ValueError("atmosphere profile must contain at least two pressures")

  p_source = data[:, 0]
  source_fields = data[:, 1:].T

  pressure_step = np.diff(p_source)
  if not (np.all(pressure_step > 0.0) or np.all(pressure_step < 0.0)):
    raise ValueError("atmospheric pressure must be strictly monotonic")
  if pressure_step[0] < 0.0:
    p_source = p_source[::-1]
    source_fields = source_fields[:, ::-1]
  if np.any(~np.isfinite(data)) or np.any(p_source <= 0.0):
    raise ValueError("atmosphere profile must contain finite, positive pressure")

  p_edge = np.geomspace(p_top * bar, p_bottom * bar, n_layers + 1)
  log_p_edge = np.log(p_edge)
  p = np.diff(p_edge) / np.diff(log_p_edge)

  log_p_source = np.log(p_source)
  log_p_target = np.clip(
    np.log(p / bar),
    log_p_source[0],
    log_p_source[-1],
  )
  interpolated = PchipInterpolator(
    log_p_source,
    source_fields,
    axis=1,
    extrapolate=False,
  )(log_p_target)

  T, g, Kzz, mu = interpolated[:4]
  VMR = np.clip(interpolated[4:], 0.0, 1.0)
  vmr_sum = VMR.sum(axis=0)
  if np.any(vmr_sum <= 0.0):
    raise ValueError("interpolated background-gas VMR must have a positive sum")
  VMR /= vmr_sum

  return Atmosphere(
    T=T,
    p=p,
    p_edge=p_edge,
    g=g,
    Kzz=Kzz,
    mu=mu,
    VMR=VMR,
    bg_species=bg_species,
  )
