"""Background-atmosphere number- and mass-density calculations."""

from __future__ import annotations

from numba import njit
from numpy.typing import NDArray
import numpy as np

FloatArray = NDArray[np.float64]

kb = 1.380649e-16          # Boltzmann constant [erg K^-1]
amu = 1.66053906892e-24    # Atomic mass unit [g]


@njit(cache=True)
def bg_nd_rho(
  p: FloatArray,
  T: FloatArray,
  mu: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
  """Calculate layer number density, mass density, and thermal velocity.

  Parameters
  ----------
  p : ndarray, shape (n_layers,)
      Gas pressure in dyne cm^-2.
  T : ndarray, shape (n_layers,)
      Gas temperature in K.
  mu : ndarray, shape (n_layers,)
      Mean molar mass in g mol^-1, numerically equivalent to the mean
      molecular mass in atomic mass units.

  Returns
  -------
  nd : ndarray, shape (n_layers,)
      Total gas number density in cm^-3.
  rho : ndarray, shape (n_layers,)
      Total gas mass density in g cm^-3.
  cT : ndarray, shape (n_layers,)
      Background-gas thermal velocity in cm s^-1.

  Notes
  -----
  Pressure has already been converted to cgs by the input reader, so no
  additional pascal-to-dyne conversion is applied here.
  """

  nd = p / (kb * T)
  rho = nd * mu * amu
  cT = np.sqrt((2.0 * kb * T) / (mu * amu))
  return nd, rho, cT
