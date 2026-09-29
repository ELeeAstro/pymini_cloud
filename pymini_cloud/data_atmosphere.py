"""Atmospheric state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


class AtmosphereState(NamedTuple):
  """Numba-compatible view of an atmospheric column."""

  T: FloatArray
  p: FloatArray
  g: FloatArray
  Kzz: FloatArray
  mu: FloatArray
  VMR: FloatArray
  p_edge: FloatArray
  z_edge: FloatArray
  z: FloatArray
  dz: FloatArray
  nd: FloatArray
  rho: FloatArray
  cT: FloatArray
  eta: FloatArray
  nu: FloatArray
  mfp: FloatArray
  kappa: FloatArray


@dataclass(slots=True, eq=False)
class Atmosphere:
  """One-dimensional atmospheric structure and composition.

  Parameters
  ----------
  T : array-like
      Temperature profile in K, ordered consistently by atmospheric layer.
  p : array-like
      Pressure profile in dyne cm^-2, with one value per atmospheric layer.
  p_edge : array-like
      Pressure interfaces in dyne cm^-2, with ``n_layers + 1`` values.
  g : array-like
      Gravitational acceleration in cm s^-2.
  Kzz : array-like
      Eddy-diffusion coefficient in cm^2 s^-1.
  mu : array-like
      Mean molar mass in g mol^-1.
  VMR : array-like
      Background-gas volume mixing ratios with shape ``(n_gas, n_layers)``.
  bg_species : tuple of str
      Species corresponding to the first dimension of ``VMR``.

  Notes
  -----
  Inputs are stored as contiguous, double-precision NumPy arrays. Pass the
  result of :meth:`as_numba` to Numba-compiled functions; ordinary Python
  dataclass instances are not native Numba values.
  """

  T: FloatArray = field(repr=False)
  p: FloatArray = field(repr=False)
  p_edge: FloatArray = field(repr=False)
  g: FloatArray = field(repr=False)
  Kzz: FloatArray = field(repr=False)
  mu: FloatArray = field(repr=False)
  VMR: FloatArray = field(repr=False)
  bg_species: tuple[str, ...] = ("H2", "He")
  z_edge: FloatArray | None = field(default=None, init=False, repr=False)
  z: FloatArray | None = field(default=None, init=False, repr=False)
  dz: FloatArray | None = field(default=None, init=False, repr=False)
  nd: FloatArray | None = field(default=None, init=False, repr=False)
  rho: FloatArray | None = field(default=None, init=False, repr=False)
  cT: FloatArray | None = field(default=None, init=False, repr=False)
  eta: FloatArray | None = field(default=None, init=False, repr=False)
  nu: FloatArray | None = field(default=None, init=False, repr=False)
  mfp: FloatArray | None = field(default=None, init=False, repr=False)
  kappa: FloatArray | None = field(default=None, init=False, repr=False)

  def __post_init__(self) -> None:
    self.T = self._as_profile("T", self.T)
    self.p = self._as_profile("p", self.p)
    self.p_edge = self._as_pressure_edges(self.p_edge, self.p.size)
    self.g = self._as_profile("g", self.g)
    self.Kzz = self._as_profile("Kzz", self.Kzz)
    self.mu = self._as_profile("mu", self.mu)
    self.VMR = self._as_vmr(self.VMR)
    self.bg_species = tuple(self.bg_species)

    profiles = {
      "p": self.p,
      "g": self.g,
      "Kzz": self.Kzz,
      "mu": self.mu,
    }
    mismatched = [
      name for name, profile in profiles.items()
      if profile.shape != self.T.shape
    ]
    if mismatched:
      raise ValueError(
        "all atmospheric profiles must contain the same number of layers; "
        f"mismatched profiles: {', '.join(mismatched)}"
      )
    if self.VMR.shape[1] != self.n_layers:
      raise ValueError("VMR must contain one value per atmospheric layer")
    if np.any(self.p <= self.p_edge[:-1]) or np.any(self.p >= self.p_edge[1:]):
      raise ValueError("each layer pressure must lie between its interfaces")
    if len(self.bg_species) != self.VMR.shape[0]:
      raise ValueError("bg_species must contain one name per VMR profile")
    if not self.bg_species or len(set(self.bg_species)) != len(self.bg_species):
      raise ValueError("bg_species must contain unique species names")

  @staticmethod
  def _as_profile(name: str, values: FloatArray) -> FloatArray:
    profile = np.asarray(values, dtype=np.float64)
    if profile.ndim != 1:
      raise ValueError(f"{name} must be a one-dimensional profile")
    if profile.size == 0:
      raise ValueError(f"{name} must contain at least one atmospheric layer")
    if not np.all(np.isfinite(profile)):
      raise ValueError(f"{name} must contain only finite values")
    if np.any(profile <= 0.0):
      raise ValueError(f"{name} must contain only positive values")
    return np.ascontiguousarray(profile)

  @staticmethod
  def _as_pressure_edges(values: FloatArray, n_layers: int) -> FloatArray:
    pressure_edges = np.asarray(values, dtype=np.float64)
    if pressure_edges.shape != (n_layers + 1,):
      raise ValueError("p_edge must contain n_layers + 1 values")
    if np.any(~np.isfinite(pressure_edges)) or np.any(pressure_edges <= 0.0):
      raise ValueError("p_edge must contain only finite, positive values")
    if np.any(np.diff(pressure_edges) <= 0.0):
      raise ValueError("p_edge must be strictly increasing")
    return np.ascontiguousarray(pressure_edges)

  @staticmethod
  def _as_vmr(values: FloatArray) -> FloatArray:
    vmr = np.asarray(values, dtype=np.float64)
    if vmr.ndim != 2:
      raise ValueError("VMR must have shape (n_gas, n_layers)")
    if not np.all(np.isfinite(vmr)):
      raise ValueError("VMR must contain only finite values")
    if np.any((vmr < 0.0) | (vmr > 1.0)):
      raise ValueError("VMR values must lie between zero and one")
    if not np.allclose(vmr.sum(axis=0), 1.0, rtol=1.0e-8, atol=1.0e-12):
      raise ValueError("background-gas VMR values must sum to one in each layer")
    return np.ascontiguousarray(vmr)

  @property
  def n_layers(self) -> int:
    """Number of layers in the atmospheric column."""

    return self.T.size

  def set_vertical_grid(
    self,
    z_edge: FloatArray,
    z: FloatArray,
    dz: FloatArray,
  ) -> None:
    """Store pressure interfaces and hydrostatic altitude coordinates."""

    altitude_edges = np.asarray(z_edge, dtype=np.float64)
    altitude = np.asarray(z, dtype=np.float64)
    layer_thickness = np.asarray(dz, dtype=np.float64)

    if altitude_edges.shape != (self.n_layers + 1,):
      raise ValueError("z_edge must contain n_layers + 1 values")
    if altitude.shape != self.T.shape or layer_thickness.shape != self.T.shape:
      raise ValueError("z and dz must contain one value per atmospheric layer")
    if any(
      np.any(~np.isfinite(values))
      for values in (altitude_edges, altitude, layer_thickness)
    ):
      raise ValueError("vertical-grid coordinates must contain finite values")
    if np.any(altitude_edges < 0.0) or np.any(np.diff(altitude_edges) >= 0.0):
      raise ValueError("z_edge must be non-negative and decrease downward")
    if np.any(altitude <= 0.0) or np.any(layer_thickness <= 0.0):
      raise ValueError("z and dz must contain only positive values")
    if not np.allclose(
      altitude_edges[:-1] - altitude_edges[1:],
      layer_thickness,
      rtol=1.0e-12,
      atol=0.0,
    ):
      raise ValueError("dz must equal the difference between adjacent z_edge values")

    self.z_edge = np.ascontiguousarray(altitude_edges)
    self.z = np.ascontiguousarray(altitude)
    self.dz = np.ascontiguousarray(layer_thickness)

  def set_layer_densities(
    self,
    nd: FloatArray,
    rho: FloatArray,
    cT: FloatArray,
  ) -> None:
    """Store derived density and thermal-velocity profiles."""

    number_density = self._as_profile("nd", nd)
    mass_density = self._as_profile("rho", rho)
    thermal_velocity = self._as_profile("cT", cT)
    if (
      number_density.shape != self.T.shape
      or mass_density.shape != self.T.shape
      or thermal_velocity.shape != self.T.shape
    ):
      raise ValueError(
        "nd, rho, and cT must contain one value per atmospheric layer"
      )
    self.nd = number_density
    self.rho = mass_density
    self.cT = thermal_velocity

  def set_viscosity(self, eta: FloatArray) -> None:
    """Store the derived dynamic-viscosity profile."""

    viscosity = self._as_profile("eta", eta)
    if viscosity.shape != self.T.shape:
      raise ValueError("eta must contain one value per atmospheric layer")
    self.eta = viscosity

  def set_gas_transport(
    self,
    nu: FloatArray,
    mfp: FloatArray,
  ) -> None:
    """Store kinematic-viscosity and mean-free-path profiles."""

    kinematic_viscosity = self._as_profile("nu", nu)
    mean_free_path = self._as_profile("mfp", mfp)
    if (
      kinematic_viscosity.shape != self.T.shape
      or mean_free_path.shape != self.T.shape
    ):
      raise ValueError("nu and mfp must contain one value per atmospheric layer")
    self.nu = kinematic_viscosity
    self.mfp = mean_free_path

  def set_conductivity(self, kappa: FloatArray) -> None:
    """Store the derived thermal-conductivity profile."""

    conductivity = self._as_profile("kappa", kappa)
    if conductivity.shape != self.T.shape:
      raise ValueError("kappa must contain one value per atmospheric layer")
    self.kappa = conductivity

  def as_numba(self) -> AtmosphereState:
    """Return a named-tuple view accepted by Numba nopython functions."""

    if (
      self.z_edge is None
      or self.z is None
      or self.dz is None
      or self.nd is None
      or self.rho is None
      or self.cT is None
      or self.eta is None
      or self.nu is None
      or self.mfp is None
      or self.kappa is None
    ):
      raise RuntimeError(
        "static atmosphere values must be calculated before calling as_numba"
      )

    return AtmosphereState(
      T=self.T,
      p=self.p,
      g=self.g,
      Kzz=self.Kzz,
      mu=self.mu,
      VMR=self.VMR,
      p_edge=self.p_edge,
      z_edge=self.z_edge,
      z=self.z,
      dz=self.dz,
      nd=self.nd,
      rho=self.rho,
      cT=self.cT,
      eta=self.eta,
      nu=self.nu,
      mfp=self.mfp,
      kappa=self.kappa,
    )

  def __repr__(self) -> str:
    return (
      f"Atmosphere(n_layers={self.n_layers}, "
      f"bg_species={self.bg_species!r})"
    )
