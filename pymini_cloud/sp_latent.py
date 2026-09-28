"""
sp_l_heat.py
==========

Calculate the latent heat [erg g-1] for a species, using Clausius-Clapeyron
on the active vapour pressure expression in sp_p_vap.py:

  L = (R_gas/mol_w) * T**2 * d(ln p_vap)/dT
"""

import numpy as np
from data_global import R_gas

ln10 = np.log(10.0)

def _L_H2O(T):
  # Murphy & Koop (2005) latent heat [J mol-1]: Eq. (5) ice, Eq. (9) liquid
  ice = 46782.5 + 35.8925*T - 0.07414*T**2 + 541.5*np.exp(-(T/123.75)**2)
  liq = 56579.0 - 42.212*T + np.exp(0.1149*(281.6 - T))
  return np.where(T <= 273.16, ice, liq) * 1.0e7   # -> erg mol-1


def _L_CO(T):
  # Fray & Schmitt piecewise CO fit in inverse powers of T
  lo = 7.213e2 + 2.0*1.074e4/T - 3.0*2.341e5/T**2 + 4.0*2.392e6/T**3 - 5.0*9.478e6/T**4
  hi = 7.482e2 + 2.0*5.843e3/T - 3.0*3.939e4/T**2
  return np.where(T < 61.55, lo, hi) * R_gas


def _L_CO2(T):
  # Fray & Schmitt piecewise CO2 fit in inverse powers of T
  lo = 2.571e3 + 2.0*7.781e4/T - 3.0*4.325e6/T**2 + 4.0*1.207e8/T**3 - 5.0*1.350e9/T**4
  hi = 4.154e3 - 2.0*1.041e5/T
  return np.where(T < 194.7, lo, hi) * R_gas


# Each entry: f(T) -> molar latent heat [erg mol-1]
# (divided by mol_w [g mol-1] in sp_l_heat to give erg g-1)
_L_funcs = {

  # Gail & Sedlmayr shifted inverse-temperature carbon fit
  'C': lambda T: R_gas * 8.65139e4 * (T/(T + 4.80395e-1))**2,

  # Kimura et al. base-10 inverse-temperature TiC fit
  'TiC': lambda T: R_gas * 33600.0 * ln10,

  # JANAF/NIST polynomial SiC fit in ln(p_vap)
  'SiC': lambda T: R_gas * (9.51431385e4 + 1.09809718e-3*T**2
    - 2.0*5.63629542e-7*T**3 + 3.0*6.97886017e-11*T**4),

  # Wakeford/VIRGA base-10 CaTiO3 fit (p, met terms have no T dependence)
  'CaTiO3': lambda T: R_gas * 72160.0 * ln10,

  # Wakeford/CARMA base-10 Al2O3 fit (met term has no T dependence)
  'Al2O3': lambda T: R_gas * 45892.6 * ln10,

  # GGChem/NIST polynomial TiO2 fit in ln(p_vap)
  'TiO2': lambda T: R_gas * (7.70443e4 - 2.59140e-3*T**2
    + 2.0*6.02422e-7*T**3 - 3.0*6.86899e-11*T**4),

  # NIST polynomial VO fit in ln(p_vap)
  'VO': lambda T: R_gas * (6.74603e4 - 2.78551e-3*T**2
    + 2.0*5.72078e-7*T**3 - 3.0*7.41840e-11*T**4),

  # Visscher/CARMA base-10 inverse-temperature Fe fit
  'Fe': lambda T: R_gas * 20995.0 * ln10,

  # GGChem/NIST polynomial FeS fit in ln(p_vap)
  'FeS': lambda T: R_gas * (5.69922e4 - 4.68301e-3*T**2
    + 2.0*1.03559e-6*T**3 - 3.0*8.42872e-11*T**4),

  # GGChem/NIST polynomial FeO fit in ln(p_vap)
  'FeO': lambda T: R_gas * (6.30018e4 - 2.42990e-3*T**2
    + 2.0*3.18636e-7*T**3),

  # Visscher/CARMA base-10 Mg2SiO4 fit (p, met terms have no T dependence)
  'Mg2SiO4': lambda T: R_gas * 32488.0 * ln10,

  # Visscher/VIRGA base-10 MgSiO3 fit (met term has no T dependence)
  'MgSiO3': lambda T: R_gas * 28665.0 * ln10,

  # Corrected NASA9/GGChem polynomial MgO fit in ln(p_vap)
  'MgO': lambda T: R_gas * (8.44591675e4 + 4.38797163e-3*T**2
    - 2.0*9.84437451e-8*T**3 - 3.0*7.30855823e-11*T**4),

  # GGChem/NIST polynomial SiO2 fit in ln(p_vap)
  'SiO2': lambda T: R_gas * (7.28086e4 - 2.56109e-4*T**2
    - 2.0*5.24980e-7*T**3 + 3.0*1.53343e-10*T**4),

  # Gail et al. inverse-temperature SiO fit
  'SiO': lambda T: R_gas * 49520.0,

  # GGChem/NIST polynomial Cr fit in ln(p_vap)
  'Cr': lambda T: R_gas * (4.78455e4 - 5.28710e-4*T**2
    - 2.0*6.17347e-8*T**3 + 3.0*2.88469e-12*T**4),

  # Morley base-10 inverse-temperature MnS fit (met term has no T dependence)
  'MnS': lambda T: R_gas * 23810.0 * ln10,

  # Morley base-10 inverse-temperature Na2S fit (met term has no T dependence)
  'Na2S': lambda T: R_gas * 13889.0 * ln10,

  # Barin/GGChem polynomial ZnS fit in ln(p_vap)
  'ZnS': lambda T: R_gas * (4.75507888e4 - 2.49490016e-3*T**2
    + 2.0*7.29116854e-7*T**3 - 3.0*1.12734453e-10*T**4),

  # GGChem/NIST polynomial KCl fit in ln(p_vap)
  'KCl': lambda T: R_gas * (2.69250e4 - 2.04903e-3*T**2
    - 2.0*2.83957e-7*T**3 + 3.0*1.82974e-10*T**4),

  # GGChem/NIST polynomial NaCl fit in ln(p_vap)
  # NOTE: Fortran has -3.11287e3*T**2, assumed typo for e-3 (as in p_vap)
  'NaCl': lambda T: R_gas * (2.79146e4 - 3.11287e-3*T**2
    + 2.0*5.30965e-7*T**3 - 3.0*2.59584e-12*T**4),

  # Base-10 inverse-temperature NH4Cl fit
  'NH4Cl': lambda T: R_gas * 4302.0 * ln10,

  # Murphy & Koop (2005)
  'H2O': _L_H2O,

  # Fray & Schmitt polynomial NH3 fit in inverse powers of T
  'NH3': lambda T: R_gas * (3537.0 + 2.0*3.310e4/T
    - 3.0*1.742e6/T**2 + 4.0*2.995e7/T**3),

  # Fray & Schmitt polynomial CH4 fit in inverse powers of T
  'CH4': lambda T: R_gas * (1.110e3 + 2.0*4.341e3/T
    - 3.0*1.035e5/T**2 + 4.0*7.910e5/T**3),

  # Walker & Lumsden base-10 inverse-temperature NH4SH fit
  'NH4SH': lambda T: R_gas * 2409.4 * ln10,

  # Fray & Schmitt inverse-temperature H2S fit
  'H2S': lambda T: R_gas * 2.707e3,

  # Zahnle piecewise inverse-temperature S2 fit
  'S2': lambda T: R_gas * np.where(T < 413.0, 18500.0, 14000.0),

  # Zahnle piecewise inverse-temperature S8 fit
  'S8': lambda T: R_gas * np.where(T < 413.0, 11800.0, 7510.0),

  # Fray & Schmitt piecewise fits
  'CO': _L_CO,
  'CO2': _L_CO2,

  # GGChem/NIST polynomial H2SO4 fit in ln(p_vap)
  'H2SO4': lambda T: R_gas * (1.01294e4 - 8.34848e-3*T**2),

  # Blakley beta-O2 polynomial fit in ln(p_vap)
  'O2': lambda T: R_gas * (1166.2 - 0.75587*T + 0.14188*T**2
    - 2.0*1.8665e-3*T**3 + 3.0*7.582e-6*T**4),
}

# Species with a surface tension but no active vapour pressure expression
_no_p_vap = {'MgS'}


def sp_l_heat(c_sp, T, mol_w):
  """
    Parameters
    ----------
    c_sp : list of str, shape (n_sp)
        Condensate species names.
    T : array, shape (n_lay)
        Temperature [K].
    mol_w : array, shape (n_sp)
        Molecular weight of each condensate species [g mol-1].

    Returns
    -------
    L_heat : array, shape (n_sp, n_lay)
        Latent heat [erg g-1].
  """

  c_sp = np.atleast_1d(c_sp)
  T = np.atleast_1d(np.asarray(T, dtype=float))
  mol_w = np.atleast_1d(np.asarray(mol_w, dtype=float))

  n_sp = len(c_sp)
  n_lay = len(T)

  if len(mol_w) != n_sp:
    raise ValueError(f'sp_l_heat: mol_w has {len(mol_w)} entries for {n_sp} species')

  # Check all species exist before computing
  for sp in c_sp:
    if sp in _no_p_vap:
      raise ValueError(f'sp_l_heat: no active vapour pressure expression for {sp}')
    if sp not in _L_funcs:
      raise ValueError(f'sp_l_heat: latent heat species not found: {sp}')

  # Array declarations
  L_heat = np.zeros((n_sp, n_lay))

  # Each species evaluated over all layers at once (constants broadcast)
  for n, sp in enumerate(c_sp):
    L_heat[n, :] = _L_funcs[sp](T) / mol_w[n]

  return L_heat


def test_l_heat():

  from sp_p_vap import sp_p_vap

  # Molecular weights from formulae for the test [g mol-1]
  amass = {'H': 1.00794, 'C': 12.0107, 'N': 14.0067, 'O': 15.9994,
    'Na': 22.98977, 'Mg': 24.3050, 'Al': 26.98154, 'Si': 28.0855,
    'S': 32.065, 'Cl': 35.453, 'K': 39.0983, 'Ca': 40.078,
    'Ti': 47.867, 'V': 50.9415, 'Cr': 51.9961, 'Mn': 54.93805,
    'Fe': 55.845, 'Zn': 65.38}

  def formula_mw(sp):
    import re
    sp = sp.replace('_amorph', '')
    return sum(amass[el] * (int(n) if n else 1)
      for el, n in re.findall(r'([A-Z][a-z]?)(\d*)', sp))

  # Sanity check: H2O at 273.16 K, Murphy & Koop Eq. (5) -> ~51.06 kJ mol-1 (ice)
  mw_H2O = formula_mw('H2O')
  L = sp_l_heat(['H2O'], [273.16], [mw_H2O])[0, 0]
  print(f'H2O at 273.16 K: {L:.4e} erg g-1 = {L*mw_H2O/1e10:.2f} kJ mol-1 (expect ~51.06)')

  # Consistency with sp_p_vap: L_num = R/mw * T^2 * d(ln p_vap)/dT (central difference)
  c_sp = list(_L_funcs.keys())
  mol_w = np.array([formula_mw(sp) for sp in c_sp])
  T = np.array([60.0, 150.0, 300.0, 1000.0, 2000.0])
  n_lay = len(T)
  p = np.full(n_lay, 1e6)
  met = np.zeros(n_lay)

  L_heat = sp_l_heat(c_sp, T, mol_w)

  h = 1e-5 * T
  with np.errstate(over='ignore', under='ignore', divide='ignore', invalid='ignore'):
    lp_hi = np.log(sp_p_vap(c_sp, T + h, p, met))
    lp_lo = np.log(sp_p_vap(c_sp, T - h, p, met))
    L_num = R_gas/mol_w[:, None] * T**2 * (lp_hi - lp_lo)/(2.0*h)
    rel = np.abs(L_heat/L_num - 1.0)

  print(f'\nL_heat shape: {L_heat.shape}')
  print('Latent heat [erg g-1] / relative difference to numerical d(ln p_vap)/dT')
  print('(-- = p_vap under/overflows at this T, no numerical comparison)')
  print(f'{"species":>14s}' + ''.join(f'{t:>21.0f}' for t in T))
  for n, sp in enumerate(c_sp):
    row = ''
    for l in range(n_lay):
      r = f'{rel[n, l]:.1e}' if np.isfinite(rel[n, l]) else '--'
      row += f'{L_heat[n, l]:>12.3e} {r:>8s}'
    print(f'{sp:>14s}' + row)

  # MgS should raise an error
  try:
    sp_l_heat(['MgS'], T, [56.37])
  except ValueError as e:
    print(f'\nMgS check: {e}')


if __name__ == '__main__':
  test_l_heat()