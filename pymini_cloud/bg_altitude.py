"""Hydrostatic pressure interfaces, altitude, and layer thickness."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

R_gas = 8.31446261815324e7  # Ideal gas constant [erg mol^-1 K^-1]


def _as_positive_profile(
  name: str,
  values: FloatArray,
  n_layers: int | None = None,
) -> FloatArray:
  """Validate and return a contiguous one-dimensional profile."""

  profile = np.asarray(values, dtype=np.float64)
  if profile.ndim != 1 or profile.size < 2:
    raise ValueError(f"{name} must be a one-dimensional profile of 2+ layers")
  if n_layers is not None and profile.size != n_layers:
    raise ValueError(f"{name} must contain one value per atmospheric layer")
  if np.any(~np.isfinite(profile)) or np.any(profile <= 0.0):
    raise ValueError(f"{name} must contain only finite, positive values")
  return np.ascontiguousarray(profile)


def bg_altitude(
  p: FloatArray,
  p_edge: FloatArray,
  T: FloatArray,
  mu: FloatArray,
  g: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Construct a hydrostatic vertical grid from layer-centred profiles.

  Parameters
  ----------
  p : ndarray, shape (n_layers,)
      Layer-centred pressure in dyne cm^-2, increasing from top to bottom.
  p_edge : ndarray, shape (n_layers + 1,)
      Pressure at layer interfaces in dyne cm^-2.
  T : ndarray, shape (n_layers,)
      Layer-centred temperature in K.
  mu : ndarray, shape (n_layers,)
      Layer-centred mean molar mass in g mol^-1.
  g : ndarray, shape (n_layers,)
      Layer-centred gravitational acceleration in cm s^-2.

  Returns
  -------
  z_edge : ndarray, shape (n_layers + 1,)
      Interface altitude in cm, measured from zero at the bottom boundary.
  z : ndarray, shape (n_layers,)
      Layer-centre altitude in cm.
  dz : ndarray, shape (n_layers,)
      Hydrostatic layer thickness in cm.

  Notes
  -----
  The hypsometric equation is evaluated using the layer-centred scale height,
  ``R_gas*T/(mu*g)``. The bottom interface defines zero altitude, and layer
  centres are placed halfway between their altitude interfaces, matching the
  grid construction in the Fortran test model.
  """

  pressure = _as_positive_profile("p", p)
  n_layers = pressure.size
  pressure_edges = np.asarray(p_edge, dtype=np.float64)
  if pressure_edges.shape != (n_layers + 1,):
    raise ValueError("p_edge must contain n_layers + 1 values")
  if np.any(~np.isfinite(pressure_edges)) or np.any(pressure_edges <= 0.0):
    raise ValueError("p_edge must contain only finite, positive values")
  if np.any(np.diff(pressure_edges) <= 0.0):
    raise ValueError("p_edge must be strictly increasing")
  if np.any(pressure <= pressure_edges[:-1]) or np.any(
    pressure >= pressure_edges[1:]
  ):
    raise ValueError("each layer pressure must lie between its interfaces")
  temperature = _as_positive_profile("T", T, n_layers)
  molar_mass = _as_positive_profile("mu", mu, n_layers)
  gravity = _as_positive_profile("g", g, n_layers)

  scale_height = R_gas * temperature / (molar_mass * gravity)
  dz = scale_height * np.log(pressure_edges[1:] / pressure_edges[:-1])

  if np.any(~np.isfinite(dz)) or np.any(dz <= 0.0):
    raise ValueError("hypsometric integration produced invalid layer thicknesses")

  z_edge = np.empty(n_layers + 1, dtype=np.float64)
  z_edge[-1] = 0.0
  for layer in range(n_layers - 1, -1, -1):
    z_edge[layer] = z_edge[layer + 1] + dz[layer]

  z = 0.5 * (z_edge[:-1] + z_edge[1:])

  return z_edge, z, dz
