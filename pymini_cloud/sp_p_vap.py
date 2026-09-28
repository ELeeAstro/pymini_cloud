"""
sp_p_vap.py
==========

Calclate the vapour pressure [dyne cm-2] for a species
"""

import numpy as np
from data_global import bar, atm, pa, mmHg

def _p_H2O(T):
  # Murphy & Koop (2005) saturation vapour pressure over ice/liquid [Pa]
  # Liquid expression is held constant above 1048 K
  Tl = np.minimum(T, 1048.0)
  ice = np.exp(9.550426 - 5723.265/T + 3.53068*np.log(T) - 0.00728332*T)
  liq = np.exp(54.842763 - 6763.22/Tl - 4.210*np.log(Tl) + 0.000367*Tl
    + np.tanh(0.0415*(Tl - 218.8))
    * (53.878 - 1331.22/Tl - 9.44523*np.log(Tl) + 0.014025*Tl))
  return np.where(T <= 273.16, ice, liq) * pa


def _p_CO(T):
  # Fray & Schmitt (2009)
  lo = np.exp(1.043e1 - 7.213e2/T - 1.074e4/T**2 + 2.341e5/T**3 - 2.392e6/T**4 + 9.478e6/T**5)
  hi = np.exp(1.025e1 - 7.482e2/T - 5.843e3/T**2 + 3.939e4/T**3)
  return np.where(T < 61.55, lo, hi) * bar


def _p_CO2(T):
  # Fray & Schmitt (2009)
  lo = np.exp(1.476e1 - 2.571e3/T - 7.781e4/T**2 + 4.325e6/T**3 - 1.207e8/T**4 + 1.350e9/T**5)
  hi = np.exp(1.861e1 - 4.154e3/T + 1.041e5/T**2)
  return np.where(T < 194.7, lo, hi) * bar


# Each entry: f(T, p, met) -> p_vap [dyne cm-2]
# p is the gas pressure [dyne cm-2], met is [M/H] in dex
_p_vap_funcs = {

  # Gail & Sedlmayr (2013) - I think...
  'C': lambda T, p, met: np.exp(3.27860e1 - 8.65139e4/(T + 4.80395e-1)),

  # Kimura et al. (2023)
  'TiC': lambda T, p, met: 10.0**(-33600.0/T + 7.652) * atm,

  # Elspeth 5 polynomial JANAF-NIST fit
  'SiC': lambda T, p, met: np.exp(-9.51431385e4/T + 3.72019157e1 + 1.09809718e-3*T
    - 5.63629542e-7*T**2 + 6.97886017e-11*T**3),

  # Wakeford et al. (2017) - taken from VIRGA
  'CaTiO3': lambda T, p, met: 10.0**(-72160.0/T + 30.24 - np.log10(p/1e6) - 2.0*met) * bar,

  # GGChem 5 polynomial NIST fit
  'TiO2': lambda T, p, met: np.exp(-7.70443e4/T + 4.03144e1 - 2.59140e-3*T
    + 6.02422e-7*T**2 - 6.86899e-11*T**3),

  # NIST 5 param fit
  'VO': lambda T, p, met: np.exp(-6.74603e4/T + 3.82717e1 - 2.78551e-3*T
    + 5.72078e-7*T**2 - 7.41840e-11*T**3),

  # Wakeford et al. (2017) - taken from CARMA
  'Al2O3': lambda T, p, met: 10.0**(17.7 - 45892.6/T - 1.66*met) * bar,

  # Visscher et al. (2010) - taken from CARMA
  'Fe': lambda T, p, met: 10.0**(7.23 - 20995.0/T) * bar,

  # GGChem 5 polynomial NIST fit
  'FeS': lambda T, p, met: np.exp(-5.69922e4/T + 3.86753e1 - 4.68301e-3*T
    + 1.03559e-6*T**2 - 8.42872e-11*T**3),

  # GGChem 5 polynomial NIST fit
  'FeO': lambda T, p, met: np.exp(-6.30018e4/T + 3.66364e1 - 2.42990e-3*T
    + 3.18636e-7*T**2),

  # Visscher et al. (2010)/Visscher notes - taken from CARMA
  'Mg2SiO4': lambda T, p, met: 10.0**(14.88 - 32488.0/T - 1.4*met - 0.2*np.log10(p/1e6)) * bar,

  # Visscher - taken from VIRGA
  'MgSiO3': lambda T, p, met: 10.0**(13.43 - 28665.0/T - met) * bar,

  # Corrected expression using NASA9 coefficents
  'MgO': lambda T, p, met: np.exp(-8.44591675e4/T + 4.58952742e1 + 4.38797163e-3*T
    - 9.84437451e-8*T**2 - 7.30855823e-11*T**3),

  # GGChem 5 polynomial NIST fit
  'SiO2': lambda T, p, met: np.exp(-7.28086e4/T + 3.65312e1 - 2.56109e-4*T
    - 5.24980e-7*T**2 + 1.53343e-10*T**3),

  # Gail et al. (2013)
  'SiO': lambda T, p, met: np.exp(-49520.0/T + 32.52),

  # GGChem 5 polynomial NIST fit
  'Cr': lambda T, p, met: np.exp(-4.78455e4/T + 3.22423e1 - 5.28710e-4*T
    - 6.17347e-8*T**2 + 2.88469e-12*T**3),

  # Morley et al. (2012)
  'MnS': lambda T, p, met: 10.0**(11.532 - 23810.0/T - met) * bar,

  # Morley et al. (2012)
  'Na2S': lambda T, p, met: 10.0**(8.550 - 13889.0/T - 0.5*met) * bar,

  # Elspeth 5 polynomial Barin data fit
  'ZnS': lambda T, p, met: np.exp(-4.75507888e4/T + 3.66993865e1 - 2.49490016e-3*T
    + 7.29116854e-7*T**2 - 1.12734453e-10*T**3),

  # GGChem 5 polynomial NIST fit
  'KCl': lambda T, p, met: np.exp(-2.69250e4/T + 3.39574e1 - 2.04903e-3*T
    - 2.83957e-7*T**2 + 1.82974e-10*T**3),

  # GGChem 5 polynomial NIST fit
  # NOTE: Fortran has -3.11287e3*T, assumed typo for e-3
  'NaCl': lambda T, p, met: np.exp(-2.79146e4/T + 3.46023e1 - 3.11287e-3*T
    + 5.30965e-7*T**2 - 2.59584e-12*T**3),

  # Zahnle et al. (2016)
  'S2': lambda T, p, met: np.where(T < 413.0,
    np.exp(27.0 - 18500.0/T), np.exp(16.1 - 14000.0/T)) * bar,

  # Zahnle et al. (2016)
  'S8': lambda T, p, met: np.where(T < 413.0,
    np.exp(20.0 - 11800.0/T), np.exp(9.6 - 7510.0/T)) * bar,

  # Unknown - I think I fit this?
  'NH4Cl': lambda T, p, met: 10.0**(7.0220 - 4302.0/T) * bar,

  # Murphy & Koop (2005)
  'H2O': lambda T, p, met: _p_H2O(T),

  # Fray & Schmitt (2009)
  'NH3': lambda T, p, met: np.exp(15.96 - 3537.0/T - 3.310e4/T**2
    + 1.742e6/T**3 - 2.995e7/T**4) * bar,

  # Fray & Schmitt (2009)
  'CH4': lambda T, p, met: np.exp(1.051e1 - 1.110e3/T - 4.341e3/T**2
    + 1.035e5/T**3 - 7.910e5/T**4) * bar,

  # E.Lee's fit to Walker & Lumsden (1897)
  'NH4SH': lambda T, p, met: 10.0**(7.8974 - 2409.4/T) * bar,

  # Fray & Schmitt (2009)
  'H2S': lambda T, p, met: np.exp(12.98 - 2.707e3/T) * bar,

  # GGChem 5 polynomial NIST fit
  'H2SO4': lambda T, p, met: np.exp(-1.01294e4/T + 3.55465e1 - 8.34848e-3*T),

  # Fray & Schmitt (2009)
  'CO': lambda T, p, met: _p_CO(T),
  'CO2': lambda T, p, met: _p_CO2(T),

  # Blakley et al. (2024) - experimental to low T and pressure (beta O2)
  'O2': lambda T, p, met: np.exp(15.29 - 1166.2/T - 0.75587*np.log(T) + 0.14188*T
    - 1.8665e-3*T**2 + 7.582e-6*T**3) * bar,
}


def sp_p_vap(c_sp, T, p, met):
  """
    Parameters
    ----------
    c_sp : list of str, shape (n_sp)
        Condensate species names.
    T : array, shape (n_lay)
        Temperature [K].
    p : array, shape (n_lay)
        Gas pressure [dyne cm-2]
    met : float or array, shape (n_lay)
        Metallicity [M/H] in dex.

    Returns
    -------
    p_vap : array, shape (n_sp, n_lay)
        Vapour pressure [dyne cm-2].
  """

  c_sp = np.atleast_1d(c_sp)
  T = np.atleast_1d(np.asarray(T, dtype=float))
  p = np.asarray(p, dtype=float)
  met = np.asarray(met, dtype=float)

  n_sp = len(c_sp)
  n_lay = len(T)

  # Check all species exist (and have what they need) before computing
  for sp in c_sp:
    if sp not in _p_vap_funcs:
      raise ValueError(f'sp_p_vap: vapour pressure species not found: {sp}')

  # Array declarations
  p_vap = np.zeros((n_sp, n_lay))

  # Each species evaluated over all layers at once
  for n, sp in enumerate(c_sp):
    p_vap[n, :] = _p_vap_funcs[sp](T, p, met)
    p_vap[n,:] = np.maximum(p_vap[n,:],1e-199)

  return p_vap


def test_p_vap():

  # Sanity check: H2O triple point (Murphy & Koop -> ~611.657 Pa)
  p_tp = sp_p_vap(['H2O'], [273.16], [1e6], [0.0])[0, 0] / pa
  print(f'H2O at 273.16 K: {p_tp:.3f} Pa (expect ~611.657)')

  # Continuity across piecewise boundaries
  for sp, Tb in [('H2O', 273.16), ('CO', 61.55), ('CO2', 194.7), ('S2', 413.0), ('S8', 413.0)]:
    Tc = np.array([Tb - 1e-6, Tb + 1e-6])
    lo, hi = sp_p_vap([sp], Tc, np.full(2, 1e6), np.zeros(2))[0]
    print(f'{sp:>5s} jump at {Tb} K: {abs(hi/lo - 1.0):.2e}')

  # All species over a 1D T profile in one call -> (n_sp, n_lay)
  c_sp = list(_p_vap_funcs.keys())
  T = np.array([100.0, 300.0, 1000.0, 2000.0])
  p = np.full(T.shape, 1e6)     # 1 bar
  met = np.zeros(T.shape)       # solar
  with np.errstate(over='ignore', under='ignore'):
    p_vap = sp_p_vap(c_sp, T, p, met)
  print(f'\np_vap shape: {p_vap.shape}')
  print(f'{"species":>14s}' + ''.join(f'{t:>12.0f}' for t in T))
  for n, sp in enumerate(c_sp):
    print(f'{sp:>14s}' + ''.join(f'{v:>12.3e}' for v in p_vap[n]))

if __name__ == '__main__':
  test_p_vap()