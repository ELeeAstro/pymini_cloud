# Getting started

The model grid is configured under `mini_cloud` in the setup YAML:

```yaml
mini_cloud:
  restart: false
  T_p_grav_Kzz_mu_VMR: Gao_2018_400_425.txt
  p_top: 3.0e-3       # bar
  p_bottom: 300.0     # bar
  n_layers: 96

  vertical_advection:
    method: muscl_ssprk3
    limiter: koren
    cfl_limit: 0.8
    top_boundary: zero_inflow
    bottom_boundary: outflow

  vertical_diffusion:
    theta: 0.5
    top_boundary: zero_flux
    bottom_boundary: fixed_value
    bottom_value: from_species
```

`p_top` and `p_bottom` define `n_layers + 1` logarithmically spaced pressure
interfaces. Layer pressures are logarithmic means of adjacent interfaces. The
input atmosphere is PCHIP-interpolated in log pressure onto those layer
pressures. Beyond the input profile, the nearest endpoint value is held
constant rather than extrapolating the cubic polynomial.

The atmospheric text file contains comma-separated columns for pressure
[bar], temperature [K], gravity [cm s^-2], Kzz [cm^2 s^-1], mean molar mass
[g mol^-1], and one VMR column for each background gas.

The YAML is parsed once into an immutable, validated setup tree. Configuration
is accessed by group rather than by dictionary keys:

```python
from pymini_cloud.io_read import read_setup

setup = read_setup("experiments/Y_400K_KCl/setup.yaml")
print(setup.grid.n_layers)
print(setup.physics.do_diffusion)
print(setup.advection.method)
print(setup.species[0].vmr_bottom)
print(setup.output.file)

# The validated object can be passed directly without rereading the YAML.
from pymini_cloud.main import run
state = run(setup, n_steps=10)
```

All configured paths are resolved relative to the YAML file. Unknown options
are rejected during parsing. `setup.raw_copy()` is available when a mutable
copy of the original YAML mapping is needed to generate another setup file.

## Settling advection

The default `muscl_ssprk3` method uses Koren-limited reconstruction and one
shared substep for all cloud moments at the configured density-weighted
Courant limit. `cfl_limit` must be greater than zero and no larger than 0.85.
The alternatives are `limiter: mc`, `method: muscl_ssprk2` with a maximum
CFL of 0.5, and the legacy `method: upwind_trbdf2`.

Because settling velocities are always downward, only an incoming value is
needed at the top. `top_boundary: zero_inflow` supplies no cloud from above.
To prescribe cloud entering the column, use a scalar value for every moment,

```yaml
    top_boundary: fixed_value
    top_value: 1.0e-30
```

or a list containing one value for each cloud moment, in the same order as
the moment array passed to the transport routine.

The default `bottom_boundary: outflow` lets particles settle out of the
column. `bottom_boundary: zero_flux` closes the lower boundary. A fixed value
is not imposed at a downward outflow boundary because information travels out
of, rather than into, the model domain there.

## Vertical diffusion

Vertical diffusion uses an implicit theta method. The default `theta: 0.5`
is second-order Crank--Nicolson. Values up to `1.0` add damping, with `1.0`
selecting backward Euler; this can be useful for very stiff diffusion steps.

The upper boundary may be `zero_flux` or `fixed_flux`. For a fixed flux,
provide `top_flux` as either one value for every tracer or a list in the same
row order as the transported array:

```yaml
  vertical_diffusion:
    top_boundary: fixed_flux
    top_flux: 1.0e-14
```

Fluxes have units of g cm^-2 s^-1 and are positive downward, so a positive
top flux enters the column while a positive bottom flux leaves it.

At the bottom, `fixed_value` prescribes the mixing ratio, `zero_flux` closes
the column, and `fixed_flux` prescribes `bottom_flux`. With the default
`bottom_value: from_species`, the main loop supplies a boundary vector built
from each vapour's `vmr_bottom`, converted to mass mixing ratio using its
vapour molar mass and the bottom atmospheric `mu`, plus the chosen values for
cloud moments. A numeric scalar applies to every tracer, while a list must contain one value
per tracer in transported-array row order. Source and chemical loss terms
are calculated at runtime and are passed directly to the diffusion step;
they are not setup options.

## First diffusion-only loop

The current main-loop scaffold initializes every `q_v`, `q_0`, `q_1`, and
`q_2` profile to `1e-30`. Its packed transport order is
`q_v(:), q_0, q_1(:), q_2`, matching the Fortran model. For a
`bottom_value: from_species` boundary, each vapour receives its converted
bottom mass mixing ratio while every cloud-moment boundary remains at the
abundance floor.

Each outer timestep performs optional settling and diffusion half-steps,
leaves a placeholder for microphysics, then performs the second diffusion and
settling half-steps. Until settling velocities are implemented, keep
`do_settling: false`; enabling it without supplying velocities raises an
explicit error. A short run can be executed programmatically with:

```python
from pymini_cloud.main import run

state = run("experiments/Y_400K_KCl/setup.yaml", n_steps=10)
```

The example plot is generated with:

```console
python experiments/Y_400K_KCl/diffusion_test.py
```

## NetCDF output

At the end of `run`, the final state is written to the path given by
`mini_cloud.output_file`. Relative paths are resolved from the setup YAML
directory. The NetCDF file contains the complete atmospheric grid and static
profiles, all `q_v`, `q_0`, `q_1`, and `q_2` values, species names and lower
boundaries, model time and step, transport settings, and a copy of the input
YAML.

To continue from that file, set `restart: true` and point `restart_file` at
it. `n_step` or `--steps` then means additional steps; the saved model time
and step counter are retained.

For a short KCl run followed by plotting:

```console
python -m pymini_cloud.main experiments/Y_400K_KCl/setup.yaml --steps 10
python experiments/Y_400K_KCl/plot_output.py
```

The first command writes `Y_400K_mc_out.nc` using the YAML setting. The
second reads that file and writes `q_v_pressure.png`.
