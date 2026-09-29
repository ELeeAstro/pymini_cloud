"""
phys_settling_v.py
==================
"""

import numpy as np


v_floor = 1e-30

def v_ohno():

  # Knudsen number
  Kn = mfp/np.maximum(r, r_seed)

  # Difference in density
  delta_rho = np.maximum(rho_d_m - rho, 0.0)

  # Cunningham slip factor, Jung et al. (2012)
  beta = 1.0 + Kn*(1.165 + 0.480*np.exp(-0.101/Kn))

  # Reynolds factor, Ohno & Okuzumi (2017)
  reynolds_fac = (1.0 + \
    ((0.45*grav*r**3*rho*rho_d_m)/(54.0*eta**2))**0.4)**(-1.25)

  # Settling velocity cm s^-1
  v = (2.0*beta*grav*r**2*delta_rho)/(9.0*eta) * reynolds_fac
  v = np.maximum(v, v_floor)

  return v



def v_int_mono():


   return v_net


def v_int_lognormal():

  return v_net


def v_int_exp():

  return v_net


def v_int_gamma():

  return v_net