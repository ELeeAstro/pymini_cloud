# Advection tests

These experiments verify downward settling advection of cloud moments. The
scheme advances the conservative equation

```math
\frac{\partial(\rho q)}{\partial t}
+\frac{\partial(\rho v q)}{\partial x}=0,
```

where downward depth $x$ and settling velocity $v$ are positive. The default
scheme uses Koren-limited piecewise-linear MUSCL reconstruction and SSP-RK2
time integration. It evolves the conserved layer tracer mass directly and
automatically subcycles to a density-weighted Courant number of at most 0.45.
The monotonized-central limiter remains available as `limiter="mc"`.
The original first-order-upwind TR--BDF2 scheme remains available as
`method="upwind_trbdf2"`. The top boundary specifies incoming mixing ratio;
the bottom boundary supports free outflow or zero flux. These choices are
available under `mini_cloud.vertical_advection` in the model YAML file.

Run each script from the repository root. Every comparison writes a PNG, and
the convergence, conservation, and stiffness studies also write CSV data.

## Analytical profile comparisons

- `compare_gaussian_translation.py` compares a smooth pulse with
  $q(x,t)=q(x-vt,0)$ before it reaches either boundary.
- `compare_inflow_step.py` compares constant upper-boundary inflow with the
  translating Heaviside solution and exercises the limiter at a discontinuity.
- `compare_outflow_mass.py` follows a Gaussian through the free lower boundary
  and compares both its remaining profile and integrated column with the
  truncated analytical Gaussian.
- `compare_multiple_velocities.py` advects three otherwise identical moments
  at different speeds in one solver call.
- `compare_stratified_steady.py` checks the analytical state
  $q\propto(\rho v)^{-1}$ that carries constant downward tracer flux in a
  stratified atmosphere.
- `compare_atmospheric_steady.py` repeats the constant-flux check on the
  configured 96-layer pressure grid with its derived density and layer
  thickness profiles.
- `compare_limiters.py` compares Koren and monotonized-central reconstruction
  directly for a smooth Gaussian and a discontinuous inflow front.

## Convergence and diagnostics

- `compare_spatial_convergence.py` refines uniform and stretched grids using a
  smooth compact pulse and measures the nominally second-order MUSCL spatial
  convergence.
- `compare_temporal_convergence.py` compares SSP-RK2 with a fine-step
  reference on a fixed grid, isolating time error from spatial error.
- `compare_flux_budget.py` closes the lower boundary by setting its velocity
  to zero and checks conservation on a stretched, variable-speed column.
- `compare_stiff_positivity.py` translates an exact one-cell top-hat over
  outer-step Courant numbers from 0.1 to 50. It compares positive MUSCL
  subcycling with the legacy TR--BDF2 floor and its associated conservation
  error.

## Commands

```console
python experiments/advection_tests/compare_gaussian_translation.py
python experiments/advection_tests/compare_inflow_step.py
python experiments/advection_tests/compare_outflow_mass.py
python experiments/advection_tests/compare_multiple_velocities.py
python experiments/advection_tests/compare_stratified_steady.py
python experiments/advection_tests/compare_atmospheric_steady.py
python experiments/advection_tests/compare_limiters.py
python experiments/advection_tests/compare_spatial_convergence.py
python experiments/advection_tests/compare_temporal_convergence.py
python experiments/advection_tests/compare_flux_budget.py
python experiments/advection_tests/compare_stiff_positivity.py
```
