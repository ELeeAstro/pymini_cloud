# Diffusion tests

This directory contains independent numerical experiments for the
density-weighted vertical diffusion equation

```math
\frac{\partial q}{\partial t}
=\frac{1}{\rho}\frac{\partial}{\partial z}
\left(\rho K_{zz}\frac{\partial q}{\partial z}\right)+S-\lambda q.
```

Run any script from the repository root. Each script prints its diagnostics
and writes a PNG; convergence and budget studies also write CSV data.

## Analytical profiles

### Bottom-source series

`compare_analytical.py` uses a constant-density column with zero flux at the
top, fixed mixing ratio at the bottom, and initially zero tracer. Its series
solution is

```math
\frac{q(x,t)}{q_{\mathrm{bottom}}}=1-
\sum_{n=0}^{\infty}\frac{4(-1)^n}{(2n+1)\pi}
\cos\left(\frac{(2n+1)\pi x}{2L}\right)
\exp\left[-K_{zz}\left(\frac{(2n+1)\pi}{2L}\right)^2t\right].
```

### Eigenmodes

- `compare_fundamental_mode.py`: one mixed-boundary eigenmode.
- `compare_two_modes.py`: two modes with distinct decay rates.
- `compare_stratified_mode.py`: density-weighted mode in an isothermal,
  exponentially stratified atmosphere.
- `compare_well_mixed.py`: double-zero-flux mode decay toward a conserved,
  uniform mixing ratio.
- `compare_chemical_loss.py`: an eigenmode with uniform first-order loss.

### Infinite-domain solutions

- `compare_gaussian.py`: spreading and amplitude decay of a Gaussian pulse.
- `compare_erfc_step.py`: broadening of a sharp step according to the
  complementary error function.

The profiles are placed far enough from the unused boundary that the
finite-column correction is negligible over the comparison time.

### Forced solutions

- `compare_constant_flux.py` checks the steady analytical solution in a
  stratified atmosphere with a prescribed downward flux at the top and a
  fixed mixing ratio at the bottom.
- `compare_manufactured.py` runs on the configured 96-layer atmosphere. It
  supplies the exact source required by a chosen time-dependent solution,
  exercising the real pressure grid, variable density, and configured
  diffusion-coefficient profile together.

## Convergence and diagnostics

- `compare_spatial_convergence.py` refines both uniform and stretched grids.
  The expected finite-volume convergence rate is second order.
- `compare_temporal_convergence.py` compares against the semi-discrete
  eigenmode so that the expected second-order Crank--Nicolson time accuracy
  is isolated from spatial error.
- `compare_flux_budget.py` compares the tracer-column change with the
  time-integrated lower-boundary flux.
- `compare_stiff_positivity.py` advances a one-cell pulse over a range of
  diffusion numbers. It exposes the point at which Crank--Nicolson produces
  an oscillatory negative value and the positivity floor consequently breaks
  exact mass conservation.

## Commands

```console
python experiments/diffusion_tests/compare_analytical.py
python experiments/diffusion_tests/compare_fundamental_mode.py
python experiments/diffusion_tests/compare_two_modes.py
python experiments/diffusion_tests/compare_stratified_mode.py
python experiments/diffusion_tests/compare_gaussian.py
python experiments/diffusion_tests/compare_erfc_step.py
python experiments/diffusion_tests/compare_spatial_convergence.py
python experiments/diffusion_tests/compare_temporal_convergence.py
python experiments/diffusion_tests/compare_flux_budget.py
python experiments/diffusion_tests/compare_stiff_positivity.py
python experiments/diffusion_tests/compare_well_mixed.py
python experiments/diffusion_tests/compare_constant_flux.py
python experiments/diffusion_tests/compare_chemical_loss.py
python experiments/diffusion_tests/compare_manufactured.py
```
