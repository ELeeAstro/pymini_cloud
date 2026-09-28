"""
sp_surface_tension.py
==================

Calculate the surface tension [erg cm-2] for a species
"""

import numpy as np

# Floor value for surface tension [erg cm-2]
sig_min = 10.0

# Each entry: f(T, TC) -> sig [erg cm-2], T in K, TC in C
_sig_funcs = {

  # Tabak et al. (1995)
  'C': lambda T, TC: 1400.0,

  # Chigai et al. (1999)
  'TiC': lambda T, TC: 1242.0,

  # Nozawa et al. (2003)
  'SiC': lambda T, TC: 1800.0,

  # Kozasa et al. (1987)
  'CaTiO3': lambda T, TC: 494.0,

  # Sindel et al. (2022)
  'TiO2': lambda T, TC: 589.79 - 0.0708*T,

  # http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_5.html
  # Pradhan et al. (2009): 2858.0 - 0.51*T
  'Fe': lambda T, TC: 1862.0 - 0.39*(TC - 1530.0),

  'Fe2O3': lambda T, TC: 410.0,

  # Janz 1992 - https://data.nist.gov/od/id/mds2-2298
  'FeO': lambda T, TC: 585.0,

  # Pradhan et al. (2009)
  # Kozasa et al. (1989): 690.0
  'Al2O3': lambda T, TC: 1024.0 - 0.177*T,

  # Janz 1992 - https://data.nist.gov/od/id/mds2-2298
  'MgSiO3': lambda T, TC: 197.3 + 0.098*T,

  # Kozasa et al. (1989)
  'Mg2SiO4': lambda T, TC: 436.0,

  # Gail and Sedlmayr (1986)
  'SiO': lambda T, TC: 500.0,

  # Janz 1992 - https://data.nist.gov/od/id/mds2-2298
  # Pradhan et al. (2009): 243.2 - 0.013*T
  'SiO2': lambda T, TC: 243.2 + 0.031*T,

  # Pradhan et al. (2009)
  'MgO': lambda T, TC: 1170.0 - 0.636*T,

  # http://www.kayelaby.npl.co.uk/general_physics/2_2/2_2_5.html
  'Cr': lambda T, TC: 1642.0 - 0.20*(TC - 1860.0),

  # Gail and Sedlmayr (1986)
  'MgS': lambda T, TC: 800.0,

  'MnS': lambda T, TC: 2326.0,

  'Na2S': lambda T, TC: 1033.0,

  # Janz 1992 - https://data.nist.gov/od/id/mds2-2298
  'KCl': lambda T, TC: 175.57 - 0.07321*T,

  # Janz 1992 - https://data.nist.gov/od/id/mds2-2298
  'NaCl': lambda T, TC: 191.16 - 0.07188*T,

  'ZnS': lambda T, TC: 860.0,

  # Hale and Plummer (1974)
  'H2O': lambda T, TC: 141.0 - 0.15*TC,

  # Weast et al. (1988)
  'NH3': lambda T, TC: 23.4,

  'NH4Cl': lambda T, TC: 56.0,

  'NH4SH': lambda T, TC: 50.0,

  # USCG (1984) - can be updated to more accurate expression
  'CH4': lambda T, TC: 14.0,

  # Nehb and Vydra (2006)
  'H2S': lambda T, TC: 58.1,

  # Fanelli (1950)
  'S2': lambda T, TC: 60.8,

  # Fanelli (1950)
  'S8': lambda T, TC: 60.8,  
}

# Pradhan et al. (2009) - not currently used:
# Si : 732 - 0.086*(T - 1685.0)
# CaO : 791 - 0.0935*T


def sp_surface_tension(c_sp, T):
  """
    Parameters
    ----------
    c_sp : list of str, shape (n_sp)
        Condensate species names.
    T : array, shape (n_lay)
        Temperature [K].

    Returns
    -------
    sig : array, shape (n_sp, n_lay)
        Surface tension [erg cm-2], floored at sig_min.
  """

  c_sp = np.atleast_1d(c_sp)
  T = np.atleast_1d(np.asarray(T, dtype=float))

  n_sp = len(c_sp)
  n_lay = len(T)

  # Check all species exist before computing
  for sp in c_sp:
    if sp not in _sig_funcs:
      raise ValueError(f'sp_surface_tension: species surface tension not found: {sp}')

  # Temperature in Celsius
  TC = T - 273.15

  # Array declarations
  sig = np.zeros((n_sp, n_lay))

  # Each species evaluated over all layers at once (constants broadcast)
  for n, sp in enumerate(c_sp):
    sig[n, :] = _sig_funcs[sp](T, TC)

  # Floor to avoid zero or negative surface tension at high T
  sig = np.maximum(sig_min, sig)

  return sig


def test_sig():

  # Sanity check: H2O at 0 C (Hale & Plummer -> 141 erg cm-2)
  sig_H2O = sp_surface_tension(['H2O'], [273.15])[0, 0]
  print(f'H2O at 273.15 K: {sig_H2O:.2f} erg cm-2 (expect 141.00)')

  # Floor check: H2O at high T should hit sig_min
  sig_hot = sp_surface_tension(['H2O'], [2000.0])[0, 0]
  print(f'H2O at 2000 K: {sig_hot:.2f} erg cm-2 (expect floor {sig_min})')

  # All species over a 1D T profile in one call -> (n_sp, n_lay)
  c_sp = list(_sig_funcs.keys())
  T = np.array([100.0, 300.0, 1000.0, 2000.0])
  sig = sp_surface_tension(c_sp, T)
  print(f'\nsig shape: {sig.shape}')
  print(f'{"species":>10s}' + ''.join(f'{t:>12.0f}' for t in T))
  for n, sp in enumerate(c_sp):
    print(f'{sp:>10s}' + ''.join(f'{v:>12.2f}' for v in sig[n]))


if __name__ == '__main__':
  test_sig()