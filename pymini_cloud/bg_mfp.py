"""Background-gas kinematic-viscosity and mean-free-path calculations."""

from __future__ import annotations

from numba import njit
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]

R_gas = 8.31446261815324e7  # Ideal gas constant [erg mol^-1 K^-1]


@njit(cache=True)
def bg_mfp(
  eta: FloatArray,
  rho: FloatArray,
  mu: FloatArray,
  T: FloatArray,
) -> tuple[FloatArray, FloatArray]:
  """Calculate kinematic viscosity and molecular mean free path.

  Parameters
  ----------
  eta : ndarray, shape (n_layers,)
      Dynamic viscosity in g cm^-1 s^-1.
  rho : ndarray, shape (n_layers,)
      Gas mass density in g cm^-3.
  mu : ndarray, shape (n_layers,)
      Mean molar mass in g mol^-1.
  T : ndarray, shape (n_layers,)
      Gas temperature in K.

  Returns
  -------
  nu : ndarray, shape (n_layers,)
      Kinematic viscosity in cm^2 s^-1.
  mfp : ndarray, shape (n_layers,)
      Molecular mean free path in cm.
  """

  nu = eta / rho
  mfp = 2.0 * nu * np.sqrt((np.pi * mu) / (8.0 * R_gas * T))
  return nu, mfp
