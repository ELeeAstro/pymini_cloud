
'''
phys_nucleation.py
==================
'''

import numpy as np

kb = 1.380649e-16           # Boltzmann constant [erg K^-1]
amu = 1.66053906892e-24     # Atomic mass unit [g]
R = 8.31446261815324e7      # Ideal gas constant [erg mol^-1 K^-1]

third = 1.0/3.0
twothird = 2.0/3.0

# Function for classical nucleation theory
def CNT(T, p, sig, S, n_v):

  # T [K], p [dyn cm^-2], sig [erg cm^-2], S [dimensionless],
  # and monomer vapour number density n_v [cm^-3]

  if (S <= 1.0):
    # Unsaturated, J_s not defined
    J_s = 0.0
  else:

    # Critical cluster radius [cm].
    a_c = (2.0 * mw * sig) / (rho_d * R * T * np.log(S))

    # Continuous number of monomer molecules, gm, in a spherical critical cluster volume [cm^3].
    cluster_volume = 4.0/3.0 * np.pi * a_c**3
    gm = cluster_volume / V0

    # Critical-cluster formation energy [erg]
    F = 4.0/3.0 * np.pi * sig * a_c**2

    # Monomer molecular flux onto a unit surface [cm^-2 s^-1]
    # Correct flux using the monomer vapour partial pressure, not the atmospheric pressure.
    p_par = n_v*kb*T
    phi = p_par/np.sqrt(2.0*np.pi*m0*kb*T)

    # Zeldovich factor [dimensionless]
    Z = np.sqrt(F / (3.0 * np.pi * kb * T * gm**2))

    # Nucleation rate, J_s [cm^-3 s^-1]
    J_s = 4.0 * np.pi * a_c**2 * phi * Z * n_v \
      * np.exp(-F / (kb * T))

  return J_s

# Function for modified classical nucleation theory
def MCNT(T, sig, S, n_v):

  # T [K], sig [erg cm^-2], S [-], 
  # and monomer vapour number density n_v [cm^-3]

  if (S <= 1.0):
    # Unsaturated, J_s not defined
    J_s = 0.0
  else:

    # Dimensionless thermodynamic and geometric quantities
    ln_ss = np.log(S)          # Natural log of saturation ratio [dimensionless]
    f0 = 4.0 * np.pi * r0**2   # Spherical monomer surface area [cm^2]
    kbT = kb * T               # Thermal energy per molecule [erg]

    alpha = 1.0 # Sticking probability [dimensionless]

    # Dimensionless surface-energy parameter, theta_inf = Delta G_surface/(kb*T)
    theta_inf = (f0 * sig)/(kbT)

    # Classical critical cluster size excluding the reference monomer [monomers]
    N_inf = (((twothird) * theta_inf) / ln_ss)**3

    # MCNT critical cluster size [monomers].
    # Gail et al. (2014); note the typo in Lee et al. (2015a).
    N_star = 1.0 + (N_inf / 8.0) \
      * (1.0 + np.sqrt(1.0 + 2.0*(Nf/N_inf)**third) \
      - 2.0*(Nf/N_inf)**third)**3
    N_star = max(1.00001, N_star) # Ensure no div 0
    N_star_1 = N_star - 1.0

    # Critical-cluster surface free energy divided by kb*T [dimensionless]
    dg_kbt = theta_inf * (N_star_1 / (N_star_1**third + Nf**third))

    # Zeldovich factor at N_star [dimensionless]
    Zel = np.sqrt((theta_inf / (9.0 * np.pi * (N_star_1)**(4.0/3.0))) \
      * ((1.0 + 2.0*(Nf/N_star_1)**third)/(1.0 + (Nf/N_star_1)**third)**3))

    # Calculate the inverse growth timescale, tau_gr^-1 [s^-1]
    tau_gr_inv = (f0 * N_star**(twothird)) * alpha * np.sqrt(kbT \
       / (2.0 * np.pi * mw * amu)) * n_v

    # Boltzmann exponent for critical-cluster formation [dimensionless]
    exponent = N_star_1*ln_ss - dg_kbt

    # Nucleation rate [cm^-3 s^-1]
    J_s = n_v * tau_gr_inv * Zel * np.exp(exponent)

  return J_s

# Function for non-classical nucleation theory
def NCNT(T, S, n_v, dfG):


  # T [K], S [-], 
  # and monomer vapour number density n_v [cm^-3]
  # and Gibbs formation energy of clusters dfG [erg mol^-1]

  if (S <= 1.0):
    # Unsaturated, J_s not defined
    return 0.0

  ncl = len(dfG)
  N = np.arange(1, ncl+1, dtype=float)
  p_ref = 1.0e6 # Reference pressure: 1 bar [dyn cm^-2]
  p_par = n_v * kb * T # Monomer partial pressure [dyn cm^-2]

  # Equilibrium cluster partial pressures from the law of mass action:
  # p_N = p_ref * (p_1/p_ref)^N * exp[-(Delta_f G_N - N Delta_f G_1)/(R T)].
  p_par_n = p_ref * (p_par/p_ref)**N * np.exp(-(dfG - N*dfG[0])/(R*T))

  # Equilibrium cluster number densities [cm^-3].
  n_v_cl = p_par_n / (kb*T)

  # Monomer attachment timescale for each cluster.
  alpha = 1.0
  f0 = 4.0 * np.pi * r0**2
  tau_gr = 1.0 / (f0 * N**twothird * alpha \
    * np.sqrt(kb*T/(2.0*np.pi*m0)) * n_v)

  # J_s^-1 = sum_N tau_gr(N)/n_eq(N).
  resistance = tau_gr / n_v_cl
  
  # Nucleation rate, J_s [cm^-3 s^-1]
  J_s = 1.0 / np.sum(resistance)

  return J_s
