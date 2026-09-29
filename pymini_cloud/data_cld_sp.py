"""Cloud-species configuration and evolving tracer state."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
AMU = 1.66053906892e-24  # g
DEFAULT_SEED_RADIUS = 1.0e-7  # cm


# condensate molar mass, bulk density, vapour/key-reactant molar mass,
# vapour-to-condensate mass factor, homogeneous nucleation, MCNT cluster size.
# These values reproduce cloud_sp in mini_cloud_3_gamma_mix_mod.f90.
FORTRAN_SPECIES_DATA: dict[str, tuple[float, float, float, float, bool, float]] = {
  "C": (12.0107, 2.27, 12.0107, 1.0, True, 5.0),
  "TiC": (59.8777, 4.93, 47.8670, 47.8670 / 59.8777, False, 0.0),
  "SiC": (40.0962, 3.21, 68.1817, 68.1817 / 40.0962, False, 0.0),
  "CaTiO3": (135.943, 3.98, 40.0780, 40.0780 / 135.943, False, 0.0),
  "Al2O3": (101.961, 3.986, 26.98153860, 2.0 * 26.98153860 / 101.961, False, 0.0),
  "TiO2": (79.866, 4.23, 79.866, 1.0, True, 0.0),
  "VO": (66.94090, 5.76, 66.94090, 1.0, False, 0.0),
  "Fe": (55.845, 7.87, 55.845, 1.0, False, 0.0),
  "FeO": (71.8444, 5.99, 55.845, 55.845 / 71.8444, False, 0.0),
  "Fe2O3": (159.6882, 5.25, 55.845, 2.0 * 55.845 / 159.6882, False, 0.0),
  "FeS": (87.91, 4.84, 55.845, 55.845 / 87.91, False, 0.0),
  "Mg2SiO4": (140.693, 3.21, 24.305, 2.0 * 24.305 / 140.693, False, 0.0),
  "MgSiO3": (100.389, 3.19, 24.305, 24.305 / 100.389, False, 0.0),
  "MgO": (40.3044, 3.6, 24.305, 24.305 / 40.3044, False, 0.0),
  "MgS": (56.37, 2.84, 24.305, 24.305 / 56.37, False, 0.0),
  "SiO2": (60.084, 2.648, 44.085, 44.085 / 60.084, False, 0.0),
  "SiO": (44.085, 2.18, 44.085, 1.0, True, 0.0),
  "SiS": (60.1505, 2.18, 60.1505, 1.0, False, 0.0),
  "Cr": (51.996, 7.19, 51.996, 1.0, False, 0.0),
  "MnS": (87.003, 4.08, 54.938045, 54.938045 / 87.003, False, 0.0),
  "Na2S": (78.0445, 1.856, 22.98976928, 2.0 * 22.98976928 / 78.0445, False, 0.0),
  "ZnS": (97.445, 4.09, 65.38, 65.38 / 97.445, False, 0.0),
  "KCl": (74.551, 1.99, 74.551, 1.0, True, 5.0),
  "NaCl": (58.443, 2.165, 58.443, 1.0, True, 5.0),
  "NH4Cl": (53.4915, 1.52, 36.4609, 36.4609 / 53.4915, False, 0.0),
  "H2O": (18.015, 0.93, 18.015, 1.0, False, 0.0),
  "NH3": (17.031, 0.87, 17.031, 1.0, False, 0.0),
  "CH4": (16.043, 0.425, 16.043, 1.0, False, 0.0),
  "CO": (28.0101, 1.14, 28.0101, 1.0, False, 0.0),
  "CO2": (44.0095, 1.98, 44.0095, 1.0, False, 0.0),
  "H2SO4": (98.0785, 1.8302, 80.0632, 80.0632 / 98.0785, False, 0.0),
  "NH4SH": (51.1114, 1.17, 34.08, 34.08 / 51.1114, False, 0.0),
  "H2S": (34.08, 1.12, 34.08, 1.0, False, 0.0),
  "O2": (31.9988, 1.141, 31.9988, 1.0, False, 0.0),
}


@dataclass(frozen=True, slots=True)
class CloudSpecies:
  """Constant material, boundary, nucleation, and optical species data."""

  name: str
  vmr_bottom: float
  condensate_density: float
  condensate_molar_mass: float
  vapour_molar_mass: float
  vapour_to_condensate_mass_ratio: float
  condensation_efficiency: float = 1.0
  nucleation_model: str = "none"
  nucleation_efficiency: float = 1.0
  nucleation_cluster_size: float = 0.0
  optical_constants: str | None = None
  seed_radius: float = DEFAULT_SEED_RADIUS

  def __post_init__(self) -> None:
    if not self.name.strip():
      raise ValueError("cloud species name must be non-empty")
    nonnegative = {
      "vmr_bottom": self.vmr_bottom,
      "nucleation_cluster_size": self.nucleation_cluster_size,
    }
    positive = {
      "condensate_density": self.condensate_density,
      "condensate_molar_mass": self.condensate_molar_mass,
      "vapour_molar_mass": self.vapour_molar_mass,
      "vapour_to_condensate_mass_ratio": self.vapour_to_condensate_mass_ratio,
      "seed_radius": self.seed_radius,
    }
    for name, value in nonnegative.items():
      if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    for name, value in positive.items():
      if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    if self.vmr_bottom > 1.0:
      raise ValueError("vmr_bottom must not exceed one")
    for name, value in (
      ("condensation_efficiency", self.condensation_efficiency),
      ("nucleation_efficiency", self.nucleation_efficiency),
    ):
      if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and lie in [0, 1]")
    if not self.nucleation_model.strip():
      raise ValueError("nucleation_model must be non-empty")

  @classmethod
  def from_fortran_defaults(
    cls,
    name: str,
    vmr_bottom: float,
    **overrides: object,
  ) -> CloudSpecies:
    """Construct a species from the Fortran material-property catalogue."""

    try:
      molar_mass, density, vapour_mass, mass_ratio, nucleates, cluster = (
        FORTRAN_SPECIES_DATA[name]
      )
    except KeyError as error:
      supported = ", ".join(sorted(FORTRAN_SPECIES_DATA))
      raise ValueError(
        f"no default material data for cloud species {name!r}; "
        f"supported species: {supported}"
      ) from error
    values: dict[str, object] = {
      "name": name,
      "vmr_bottom": vmr_bottom,
      "condensate_density": density,
      "condensate_molar_mass": molar_mass,
      "vapour_molar_mass": vapour_mass,
      "vapour_to_condensate_mass_ratio": mass_ratio,
      "nucleation_model": "MCNT" if nucleates else "none",
      "nucleation_cluster_size": cluster,
    }
    values.update(overrides)
    return cls(**values)  # type: ignore[arg-type]

  @property
  def monomer_mass(self) -> float:
    return self.condensate_molar_mass * AMU

  @property
  def monomer_volume(self) -> float:
    return self.monomer_mass / self.condensate_density

  @property
  def monomer_radius(self) -> float:
    return (3.0 * self.monomer_volume / (4.0 * math.pi)) ** (1.0 / 3.0)

  @property
  def seed_mass(self) -> float:
    if self.nucleation_model.lower() == "none":
      return 0.0
    volume = 4.0 * math.pi * self.seed_radius**3 / 3.0
    return volume * self.condensate_density

  def bottom_mass_mixing_ratio(self, atmospheric_mu: float) -> float:
    """Convert the configured bottom VMR to vapour mass mixing ratio."""

    if not np.isfinite(atmospheric_mu) or atmospheric_mu <= 0.0:
      raise ValueError("atmospheric_mu must be finite and positive")
    return self.vmr_bottom * self.vapour_molar_mass / atmospheric_mu


@dataclass(slots=True, eq=False)
class CloudState:
  """Vapour and cloud moments on the atmospheric layer grid."""

  species: tuple[CloudSpecies, ...]
  q_v: FloatArray
  q_0: FloatArray
  q_1: FloatArray
  q_2: FloatArray
  q_floor: float = 1.0e-30

  def __post_init__(self) -> None:
    self.species = tuple(self.species)
    if not self.species:
      raise ValueError("at least one cloud species is required")
    names = [item.name for item in self.species]
    if len(set(names)) != len(names):
      raise ValueError("cloud species names must be unique")
    if not np.isfinite(self.q_floor) or self.q_floor <= 0.0:
      raise ValueError("q_floor must be finite and positive")
    self.q_0 = self._as_nonnegative("q_0", self.q_0, 1)
    n_layers = self.q_0.size
    n_species = len(self.species)
    self.q_v = self._as_nonnegative("q_v", self.q_v, 2)
    self.q_1 = self._as_nonnegative("q_1", self.q_1, 2)
    self.q_2 = self._as_nonnegative("q_2", self.q_2, 1)
    if self.q_v.shape != (n_species, n_layers):
      raise ValueError(f"q_v must have shape ({n_species}, {n_layers})")
    if self.q_1.shape != (n_species, n_layers):
      raise ValueError(f"q_1 must have shape ({n_species}, {n_layers})")
    if self.q_2.shape != (n_layers,):
      raise ValueError(f"q_2 must have shape ({n_layers},)")

  @staticmethod
  def _as_nonnegative(name: str, values: FloatArray, ndim: int) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
      raise ValueError(f"{name} must be {ndim}-dimensional")
    if np.any(~np.isfinite(array)) or np.any(array < 0.0):
      raise ValueError(f"{name} must contain finite, non-negative values")
    return np.ascontiguousarray(array)

  @classmethod
  def initialise(
    cls,
    species: tuple[CloudSpecies, ...],
    n_layers: int,
    q_floor: float = 1.0e-30,
  ) -> CloudState:
    """Create a floor-valued state for a fresh model run."""

    if isinstance(n_layers, bool) or not isinstance(n_layers, int):
      raise ValueError("n_layers must be an integer")
    if n_layers < 2:
      raise ValueError("n_layers must be at least 2")
    n_species = len(species)
    return cls(
      species=species,
      q_v=np.full((n_species, n_layers), q_floor),
      q_0=np.full(n_layers, q_floor),
      q_1=np.full((n_species, n_layers), q_floor),
      q_2=np.full(n_layers, q_floor),
      q_floor=q_floor,
    )

  @property
  def n_species(self) -> int:
    return len(self.species)

  @property
  def n_layers(self) -> int:
    return self.q_0.size

  def pack_transport(self) -> FloatArray:
    return np.vstack((self.q_v, self.q_0, self.q_1, self.q_2))

  def pack_settling(self) -> FloatArray:
    """Pack only settling cloud moments as q0, each q1, then q2."""

    return np.vstack((self.q_0, self.q_1, self.q_2))

  def update_from_transport(self, q: FloatArray) -> None:
    transported = np.asarray(q, dtype=np.float64)
    expected_shape = (2 * self.n_species + 2, self.n_layers)
    if transported.shape != expected_shape:
      raise ValueError(
        f"transported q must have shape {expected_shape}, got {transported.shape}"
      )
    if np.any(~np.isfinite(transported)) or np.any(transported < 0.0):
      raise ValueError("transported q must contain finite, non-negative values")
    vapour_end = self.n_species
    q_0_row = vapour_end
    q_1_start = q_0_row + 1
    q_1_end = q_1_start + self.n_species
    self.q_v = np.ascontiguousarray(transported[:vapour_end])
    self.q_0 = np.ascontiguousarray(transported[q_0_row])
    self.q_1 = np.ascontiguousarray(transported[q_1_start:q_1_end])
    self.q_2 = np.ascontiguousarray(transported[q_1_end])

  def update_from_settling(self, q: FloatArray) -> None:
    """Update q0, q1, and q2 from their settling-only packed order."""

    moments = np.asarray(q, dtype=np.float64)
    expected_shape = (self.n_species + 2, self.n_layers)
    if moments.shape != expected_shape:
      raise ValueError(
        f"settled q must have shape {expected_shape}, got {moments.shape}"
      )
    if np.any(~np.isfinite(moments)) or np.any(moments < 0.0):
      raise ValueError("settled q must contain finite, non-negative values")
    self.q_0 = np.ascontiguousarray(moments[0])
    self.q_1 = np.ascontiguousarray(moments[1:-1])
    self.q_2 = np.ascontiguousarray(moments[-1])

  def enforce_floor(self) -> None:
    np.maximum(self.q_v, self.q_floor, out=self.q_v)
    np.maximum(self.q_0, self.q_floor, out=self.q_0)
    np.maximum(self.q_1, self.q_floor, out=self.q_1)
    np.maximum(self.q_2, self.q_floor, out=self.q_2)

  def transport_bottom_values(self, atmospheric_mu: float) -> FloatArray:
    """Build mass-mixing-ratio lower boundaries in transport order."""

    q_v_bottom = np.asarray(
      [item.bottom_mass_mixing_ratio(atmospheric_mu) for item in self.species],
      dtype=np.float64,
    )
    cloud_bottom = np.full(self.n_species + 2, self.q_floor)
    return np.concatenate((q_v_bottom, cloud_bottom))
