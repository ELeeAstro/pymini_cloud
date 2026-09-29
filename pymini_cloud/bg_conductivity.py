"""Background-gas thermal conductivity and mixture calculation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

kb = 1.380649e-16          # Boltzmann constant [erg K^-1]
amu = 1.66053906892e-24    # Atomic mass unit [g]

_W_M_K_TO_CGS = 1.0e5
_MW_M_K_TO_CGS = 1.0e2

# Molecular diameter [cm], Lennard-Jones energy [erg], and molar mass
# [g mol^-1]. These are kept local so this routine has no call-order or
# mutable-state dependency on bg_viscosity.
_SPECIES_DATA = {
  "H": (2.500e-8, 30.0 * kb, 1.00794),
  "O": (2.660e-8, 70.0 * kb, 15.99940),
  "H2": (2.827e-8, 59.7 * kb, 2.01588),
  "He": (2.511e-8, 10.22 * kb, 4.002602),
  "H2O": (2.641e-8, 809.1 * kb, 18.01528),
  "N2": (3.798e-8, 71.4 * kb, 28.0134),
  "NH3": (2.900e-8, 558.3 * kb, 17.03052),
  "CO2": (3.941e-8, 195.2 * kb, 44.0095),
  "CO": (3.690e-8, 91.7 * kb, 28.0101),
  "CH4": (3.758e-8, 148.6 * kb, 16.0425),
  "C2H2": (4.033e-8, 231.8 * kb, 26.0373),
}

_H2_NUMERATOR = np.array([
  -3.40976e-1,
  4.5882,
  -1.4508,
  3.26394e-1,
  3.16939e-3,
  1.90592e-4,
  -1.13900e-6,
])
_H2_DENOMINATOR = np.array([1.38497e2, -2.21878e1, 4.57151, 1.0])


def _species_conductivity(species: str, T: FloatArray) -> FloatArray:
  """Return a pure-species thermal conductivity in cgs units."""

  if species == "H":
    return 263.5 * T**0.751 * (1.0 - 6.689e-5 * T + 3.350e-8 * T**2)
  if species == "O":
    return 46.70 * T**0.77 * (1.0 - 2.228e-5 * T + 5.545e-8 * T**2)
  if species == "H2":
    reduced_temperature = T / 33.145
    numerator = np.polynomial.polynomial.polyval(
      reduced_temperature,
      _H2_NUMERATOR,
    )
    denominator = np.polynomial.polynomial.polyval(
      reduced_temperature,
      _H2_DENOMINATOR,
    )
    return (numerator / denominator) * _W_M_K_TO_CGS
  if species == "He":
    return (
      -688.0
      + 591.4 * np.sqrt(T)
      + 23.64 * T
      - 0.2095 * T**1.5
      + 1.0979e-3 * T**2
    )
  if species == "H2O":
    reduced_temperature = T / 647.27
    conductivity = np.sqrt(reduced_temperature) / (
      2.02223
      + 14.11166 / reduced_temperature
      + 5.25597 / reduced_temperature**2
      - 2.01870 / reduced_temperature**3
    )
    return conductivity * _W_M_K_TO_CGS
  if species == "N2":
    reduced_temperature = T / 126.192
    conductivity = (
      -50.4059
      + 1457.09 * reduced_temperature
      - 2580.5 * reduced_temperature**2
      + 3616.59 * reduced_temperature**3
      - 695.367 * reduced_temperature**4
      + 129.796 * reduced_temperature**5
      + 4.03114 * reduced_temperature**6
    ) / (
      99.3852
      - 151.972 * reduced_temperature
      + 229.102 * reduced_temperature**2
      - 22.9451 * reduced_temperature**3
      + 6.0789 * reduced_temperature**4
      + reduced_temperature**5
    )
    return conductivity * _MW_M_K_TO_CGS
  if species == "NH3":
    reduced_temperature = T / 405.56
    conductivity = (
      86.9294
      - 170.5502 * reduced_temperature
      + 608.0287 * reduced_temperature**2
      - 100.9764 * reduced_temperature**3
      + 85.1986 * reduced_temperature**4
    ) / (
      4.68994
      + 9.21307 * reduced_temperature
      - 1.53637 * reduced_temperature**2
      + reduced_temperature**3
    )
    return conductivity * _MW_M_K_TO_CGS
  if species == "CO2":
    reduced_temperature = T / 304.1282
    conductivity = np.sqrt(reduced_temperature) / (
      1.51874307e-2
      + 2.80674040e-2 / reduced_temperature
      + 2.28564190e-2 / reduced_temperature**2
      - 7.41624210e-3 / reduced_temperature**3
    )
    return conductivity * _MW_M_K_TO_CGS
  if species == "CO":
    return np.full_like(T, 25.0 * _MW_M_K_TO_CGS)
  if species == "CH4":
    return np.full_like(T, 34.4 * _MW_M_K_TO_CGS)
  if species == "C2H2":
    return np.full_like(T, 21.4 * _MW_M_K_TO_CGS)
  raise ValueError(f"bg_conductivity: no data for species {species}")


def bg_conductivity(
  bg_species: tuple[str, ...] | list[str],
  bg_VMR: FloatArray,
  T: FloatArray,
) -> FloatArray:
  """Calculate background-gas mixture thermal conductivity.

  Parameters
  ----------
  bg_species : sequence of str, shape (n_bg,)
      Background species names.
  bg_VMR : ndarray, shape (n_bg, n_layers)
      Background-species volume mixing ratios.
  T : ndarray, shape (n_layers,)
      Gas temperature in K.

  Returns
  -------
  kappa : ndarray, shape (n_layers,)
      Mixture thermal conductivity in erg s^-1 cm^-1 K^-1.

  Notes
  -----
  Pure-species viscosities are recalculated locally so this function is
  independent of calls to :func:`pymini_cloud.bg_viscosity.bg_viscosity`.
  The mixture uses the Wilke (1950) mixing rule from mini-cloud.
  """

  species = tuple(bg_species)
  temperature = np.atleast_1d(np.asarray(T, dtype=np.float64))
  vmr = np.asarray(bg_VMR, dtype=np.float64)

  if not species:
    raise ValueError("bg_species must contain at least one species")
  if temperature.ndim != 1 or np.any(~np.isfinite(temperature)):
    raise ValueError("T must be a finite one-dimensional profile")
  if np.any(temperature <= 0.0):
    raise ValueError("T must contain only positive values")
  if vmr.shape != (len(species), temperature.size):
    raise ValueError("bg_VMR must have shape (n_bg, n_layers)")
  if np.any(~np.isfinite(vmr)) or np.any(vmr < 0.0):
    raise ValueError("bg_VMR must contain finite, non-negative values")
  if not np.allclose(vmr.sum(axis=0), 1.0, rtol=1.0e-8, atol=1.0e-12):
    raise ValueError("background-gas VMR values must sum to one in each layer")

  n_bg = len(species)
  diameter = np.empty(n_bg)
  lj_energy = np.empty(n_bg)
  molar_mass = np.empty(n_bg)
  conductivity = np.empty((n_bg, temperature.size))

  for index, name in enumerate(species):
    try:
      diameter[index], lj_energy[index], molar_mass[index] = _SPECIES_DATA[name]
    except KeyError as error:
      raise ValueError(f"bg_conductivity: no data for species {name}") from error
    conductivity[index] = _species_conductivity(name, temperature)

  active = vmr > 1.0e-20
  if np.any(active & ((conductivity <= 0.0) | ~np.isfinite(conductivity))):
    raise ValueError(
      "pure-species conductivity is non-positive or non-finite within the "
      "requested temperature range"
    )

  kT = kb * temperature[None, :]
  species_viscosity = (
    (5.0 / 16.0)
    * np.sqrt(np.pi * (molar_mass[:, None] * amu) * kT)
    / (np.pi * diameter[:, None]**2)
    * ((kT / lj_energy[:, None])**0.16 / 1.22)
  )

  viscosity_ratio = (
    species_viscosity[:, None, :] / species_viscosity[None, :, :]
  )
  mass_ratio_ji = molar_mass[None, :, None] / molar_mass[:, None, None]
  mass_ratio_ij = molar_mass[:, None, None] / molar_mass[None, :, None]
  phi = (
    (1.0 + np.sqrt(viscosity_ratio) * mass_ratio_ji**0.25)**2
    / (np.sqrt(8.0) * np.sqrt(1.0 + mass_ratio_ij))
  )

  mixing_denominator = np.einsum("jl,ijl->il", vmr, phi)
  kappa = np.sum(vmr * conductivity / mixing_denominator, axis=0)
  return np.ascontiguousarray(kappa)
