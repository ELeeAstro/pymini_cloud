"""
bg_viscosity.py
================

Background gas mixture dynamic viscosity using the Rosner (2012) species
viscosity and the Davidson (1993) mixing rule.
"""

import numpy as np
from data_global import kb, amu


# Species data: molecular diameter [cm], LJ potential depth [erg], molecular weight [g mol-1]
sp_data = {
  'OH':   (3.06e-8,  100.0 * kb, 17.00734),  # estimate
  'H2':   (2.827e-8,  59.7 * kb,  2.01588),
  'H2O':  (2.641e-8, 809.1 * kb, 18.01528),
  'H':    (2.5e-8,    30.0 * kb,  1.00794),
  'CO':   (3.690e-8,  91.7 * kb, 28.0101),
  'CO2':  (3.941e-8, 195.2 * kb, 44.0095),
  'O':    (2.66e-8,   70.0 * kb, 15.99940),
  'CH4':  (3.758e-8, 148.6 * kb, 16.0425),
  'C2H2': (4.033e-8, 231.8 * kb, 26.0373),
  'NH3':  (2.900e-8, 558.3 * kb, 17.03052),
  'N2':   (3.798e-8,  71.4 * kb, 28.0134),
  'HCN':  (3.630e-8, 569.1 * kb, 27.0253),
  'He':   (2.511e-8,  10.22 * kb, 4.002602),
}


def bg_viscosity(bg_sp, bg_VMR, T):
  """
    Parameters
    ----------
    bg_sp : list of str, shape (n_bg)
        Background species names (keys of sp_data).
    bg_VMR : array, shape (n_bg, n_lay)
        Volume mixing ratios of the background species.
    T : array, shape (n_lay)
        Temperature [K].

    Returns
    -------
    eta_mix : array, shape (n_lay)
        Mixture dynamic viscosity [g cm-1 s-1].
  """

  bg_sp = np.atleast_1d(bg_sp)
  T = np.atleast_1d(np.asarray(T, dtype=float))
  bg_VMR = np.asarray(bg_VMR, dtype=float).reshape(len(bg_sp), -1)
  n_bg = len(bg_sp)

  # Unpack species data into the g arrays
  g_d = np.zeros(n_bg)
  g_LJ = np.zeros(n_bg)
  g_molw = np.zeros(n_bg)
  for n, sp in enumerate(bg_sp):
    try:
      g_d[n], g_LJ[n], g_molw[n] = sp_data[sp]
    except KeyError:
      raise ValueError(f'bg_viscosity: no data for species {sp}')

  # Broadcast species (n_bg, 1) against layers (1, n_lay)
  d = g_d[:, None]
  LJ = g_LJ[:, None]
  molw = g_molw[:, None]
  kT = kb * T[None, :]

  # First calculate each species eta following Rosner (2012) -> (n_bg, n_lay)
  g_eta = (5.0/16.0) * (np.sqrt(np.pi*(molw*amu)*kT)/(np.pi*d**2)) \
    * ((kT/LJ)**0.16/1.22)

  # Davidson (1993) viscosity mixing rule
  # y values (momentum fractions) -> (n_bg, n_lay)
  top = bg_VMR * np.sqrt(molw)
  y = top/np.sum(top, axis=0)

  # Eij efficiency matrix -> (n_bg, n_bg)
  Eij = ((2.0*np.sqrt(np.outer(g_molw, g_molw)))/(g_molw[:, None] + g_molw[None, :]))**0.375

  # Fluidity: sum_ij y_i y_j Eij / sqrt(eta_i eta_j) -> (n_lay)
  z = y/np.sqrt(g_eta)
  fluid = np.einsum('il,ij,jl->l', z, Eij, z)

  # Viscosity is inverse fluidity
  eta_mix = 1.0/fluid

  return eta_mix


def test_eta():

  # Pure H2 (single species: VMR normalises out through y)
  eta_H2 = bg_viscosity(['H2'], [[0.85]], 1000.0)
  print('H2 only,  T=1000 K :', eta_H2)

  # H2/He mixture over a few layers
  T = np.array([500.0, 1000.0, 2000.0])
  bg_sp = ['H2', 'He']
  bg_VMR = np.array([[0.85, 0.85, 0.85],
                     [0.15, 0.15, 0.15]])
  eta_mix = bg_viscosity(bg_sp, bg_VMR, T)
  print('H2/He mix, T =', T, ':', eta_mix)


if __name__ == '__main__':
  test_eta()