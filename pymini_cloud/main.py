"""Top-level execution order for the mini-cloud model."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .bg_altitude import bg_altitude
from .bg_conductivity import bg_conductivity
from .bg_mfp import bg_mfp
from .bg_nd_rho import bg_nd_rho
from .bg_viscosity import bg_viscosity
from .data_atmosphere import Atmosphere
from .data_cld_sp import CloudState
from .io_read import (
  AdvectionOptions,
  DiffusionOptions,
  IntegratorOptions,
  ModelOptions,
  Setup,
  read_atmosphere,
  read_setup,
)
from .io_write import write_model_output
from .init_restart import read_restart
from .vert_advection import vert_advection
from .vert_diffusion import vert_diffusion


bar = 1.0e6  # Bar to dyne cm^-2
FloatArray = NDArray[np.float64]


@dataclass(slots=True, eq=False)
class ModelState:
  """Atmospheric, cloud, and time-integration state for a model run."""

  atmosphere: Atmosphere
  cloud: CloudState
  advection: AdvectionOptions
  diffusion: DiffusionOptions
  integrator: IntegratorOptions
  model_options: ModelOptions
  do_diffusion: bool
  do_settling: bool
  settling_velocity: FloatArray | None = None
  time: float = 0.0
  step: int = 0
  output_path: Path | None = None


def advance_settling(
  atmosphere: Atmosphere,
  q: FloatArray,
  settling_velocity: FloatArray,
  dt: float,
  options: AdvectionOptions,
) -> FloatArray:
  """Apply configured settling advection using the atmospheric grid.

  Only cloud moments should be passed to this routine; vapour mixing ratios
  do not settle. Moment rows and velocity rows must use the same order,
  expected to be ``q0``, each species' ``q1``, then ``q2``. In a
  Strang-split model step, call it with half the main timestep before and
  after diffusion and microphysics.
  """

  if atmosphere.rho is None or atmosphere.dz is None:
    raise RuntimeError(
      "atmospheric density and vertical grid must be initialized before "
      "settling advection"
    )
  moments = np.asarray(q)
  if moments.ndim != 2:
    raise ValueError("q must have shape (n_moments, n_layers)")
  q_top = options.top_values(moments.shape[0])
  return vert_advection(
    q,
    settling_velocity,
    dt,
    atmosphere.rho,
    atmosphere.dz,
    q_top=q_top,
    method=options.method,
    limiter=options.limiter,
    cfl_limit=options.cfl_limit,
    bottom_boundary=options.bottom_boundary,
  )


def advance_diffusion(
  atmosphere: Atmosphere,
  q: FloatArray,
  dt: float,
  options: DiffusionOptions,
  *,
  bottom_value: float | FloatArray | None = None,
  bottom_value_next: float | FloatArray | None = None,
  source: float | FloatArray | None = None,
  source_next: float | FloatArray | None = None,
  loss_rate: float | FloatArray | None = None,
) -> FloatArray:
  """Apply configured diffusion to vapour and cloud mixing ratios.

  Rows in ``q`` define the tracer order used by any boundary-value or flux
  lists. When ``bottom_value: from_species`` is configured, the caller must
  assemble and pass ``bottom_value`` in that same order. This will normally
  contain each species' bottom VMR converted to a mass mixing ratio, plus the
  selected lower cloud boundary values. Sources and loss rates are runtime
  physics and remain function arguments rather than static YAML settings.
  """

  if atmosphere.rho is None or atmosphere.z is None or atmosphere.dz is None:
    raise RuntimeError(
      "atmospheric density and vertical grid must be initialized before "
      "vertical diffusion"
    )
  tracers = np.asarray(q)
  if tracers.ndim != 2:
    raise ValueError("q must have shape (n_tracers, n_layers)")
  n_tracers = tracers.shape[0]
  q_bottom = options.bottom_values(n_tracers, bottom_value)
  if bottom_value_next is None:
    q_bottom_next = q_bottom
  else:
    q_bottom_next = options.bottom_values(n_tracers, bottom_value_next)

  bottom_boundary = (
    "dirichlet"
    if options.bottom_boundary == "fixed_value"
    else options.bottom_boundary
  )
  return vert_diffusion(
    tracers,
    q_bottom,
    dt,
    atmosphere.p_edge,
    atmosphere.T,
    atmosphere.mu,
    atmosphere.Kzz,
    atmosphere.rho,
    atmosphere.z,
    atmosphere.dz,
    theta=options.theta,
    top_boundary=options.top_boundary,
    top_flux=options.top_flux_values(n_tracers),
    bottom_boundary=bottom_boundary,
    bottom_flux=options.bottom_flux_values(n_tracers),
    q_bottom_next=q_bottom_next,
    source=source,
    source_next=source_next,
    loss_rate=loss_rate,
  )


def initialise_atmosphere(
  setup: Setup,
) -> Atmosphere:
  """Create the atmospheric state for a fresh model run."""

  grid = setup.grid
  if (
    grid.atmosphere_file is None
    or grid.p_top is None
    or grid.p_bottom is None
    or grid.n_layers is None
  ):
    raise ValueError("fresh-run atmospheric grid options are unavailable")
  return read_atmosphere(
    grid.atmosphere_file,
    grid.p_top,
    grid.p_bottom,
    grid.n_layers,
  )


def advance_model_step(state: ModelState) -> None:
  """Advance one outer timestep in the Fortran driver's operation order.

  Two half-steps bracket the future microphysics call, preserving the intended
  Strang-splitting layout. For now microphysics is a no-op, so only vapour
  supplied by ``vmr_bottom`` changes from its initialized abundance floor.
  """

  half_step = 0.5 * state.integrator.t_step

  # Settling velocities are supplied by the future settling-physics routine.
  # Keeping them as explicit state makes transport testable independently.
  if state.do_settling:
    if state.settling_velocity is None:
      raise RuntimeError(
        "do_settling is true but settling velocities have not been initialized"
      )
    settled = advance_settling(
      state.atmosphere,
      state.cloud.pack_settling(),
      state.settling_velocity,
      half_step,
      state.advection,
    )
    state.cloud.update_from_settling(settled)

  if state.do_diffusion:
    q_bottom = state.cloud.transport_bottom_values(state.atmosphere.mu[-1])
    transported = advance_diffusion(
      state.atmosphere,
      state.cloud.pack_transport(),
      half_step,
      state.diffusion,
      bottom_value=q_bottom,
    )
    state.cloud.update_from_transport(transported)

  # Microphysics will update q_v, q_0, q_1, and q_2 here.

  if state.do_diffusion:
    transported = advance_diffusion(
      state.atmosphere,
      state.cloud.pack_transport(),
      half_step,
      state.diffusion,
      bottom_value=q_bottom,
    )
    state.cloud.update_from_transport(transported)
  # Microphysics will eventually recalculate moment-dependent velocities here.
  if state.do_settling:
    settled = advance_settling(
      state.atmosphere,
      state.cloud.pack_settling(),
      state.settling_velocity,
      half_step,
      state.advection,
    )
    state.cloud.update_from_settling(settled)

  state.cloud.enforce_floor()
  # Calculate cloud opacity, diagnostics, output, and convergence.

  state.step += 1
  state.time += state.integrator.t_step


def run(
  setup_file: str | Path | Setup,
  n_steps: int | None = None,
  *,
  write_output: bool = True,
) -> ModelState:
  """Initialize and run the model in the same order as the Fortran driver.

  ``n_steps`` overrides ``mini_cloud.integrator.n_step`` when supplied. Pass
  zero to initialize all model state without entering the timestep loop.
  """

  # 1. Read the model setup.
  setup = (
    setup_file
    if isinstance(setup_file, Setup)
    else read_setup(setup_file)
  )
  advection_options = setup.advection
  diffusion_options = setup.diffusion
  integrator_options = setup.integrator
  model_options = setup.physics
  do_diffusion = model_options.do_diffusion
  do_settling = model_options.do_settling

  if setup.restart.enabled:
    if setup.restart.file is None:
      raise RuntimeError("validated restart path is unexpectedly unavailable")
    restored = read_restart(setup.restart.file)
    atmosphere = restored.atmosphere
    cloud = restored.cloud
    settling_velocity = restored.settling_velocity
    initial_time = restored.time
    initial_step = restored.step
  else:
    # 2. Read and grid a fresh atmospheric profile.
    atmosphere = initialise_atmosphere(setup)

    # 3. Calculate static values derived from the background atmosphere.
    z_edge, z, dz = bg_altitude(
      atmosphere.p,
      atmosphere.p_edge,
      atmosphere.T,
      atmosphere.mu,
      atmosphere.g,
    )
    atmosphere.set_vertical_grid(z_edge, z, dz)

    nd, rho, cT = bg_nd_rho(atmosphere.p, atmosphere.T, atmosphere.mu)
    atmosphere.set_layer_densities(nd, rho, cT)
    eta = bg_viscosity(atmosphere.bg_species, atmosphere.VMR, atmosphere.T)
    atmosphere.set_viscosity(eta)
    nu, mfp = bg_mfp(atmosphere.eta, atmosphere.rho, atmosphere.mu, atmosphere.T)
    atmosphere.set_gas_transport(nu, mfp)
    kappa = bg_conductivity(
      atmosphere.bg_species,
      atmosphere.VMR,
      atmosphere.T,
    )
    atmosphere.set_conductivity(kappa)

    # 4. Initialize species and transported quantities.
    cloud = CloudState.initialise(
      setup.species,
      atmosphere.n_layers,
    )
    initial_time = 0.0
    initial_step = 0
    settling_velocity = None
  state = ModelState(
    atmosphere=atmosphere,
    cloud=cloud,
    advection=advection_options,
    diffusion=diffusion_options,
    integrator=integrator_options,
    model_options=model_options,
    do_diffusion=do_diffusion,
    do_settling=do_settling,
    settling_velocity=settling_velocity,
    time=initial_time,
    step=initial_step,
  )
  if model_options.do_opacity:
    missing_optics = [
      item.name
      for item in cloud.species
      if item.optical_constants is None
      or not Path(item.optical_constants).is_file()
    ]
    if missing_optics:
      raise FileNotFoundError(
        "missing optical-constant file for cloud species: "
        + ", ".join(missing_optics)
      )

  # 5. Run the column model. The step routine currently contains diffusion
  # and optional settling around a no-op microphysics stage.
  steps = integrator_options.n_step if n_steps is None else n_steps
  if isinstance(steps, bool) or not isinstance(steps, int):
    raise ValueError("n_steps must be an integer")
  if steps < 0:
    raise ValueError("n_steps must be non-negative")
  for _ in range(steps):
    advance_model_step(state)
  if write_output:
    state.output_path = write_model_output(
      state,
      setup,
    )
  return state


def print_atmosphere(atmosphere: Atmosphere) -> None:
  """Print input and derived atmospheric profiles for inspection."""

  profile_header = [
    f"{'layer':>5}",
    f"{'P [bar]':>14}",
    f"{'T [K]':>14}",
    f"{'g [cm s^-2]':>14}",
    f"{'Kzz [cm^2 s^-1]':>16}",
    f"{'mu [g mol^-1]':>19}",
  ]
  profile_header.extend(f"{species + ' VMR':>12}" for species in atmosphere.bg_species)

  print("\nInput atmosphere profile")
  print(" ".join(profile_header))
  for layer in range(atmosphere.n_layers):
    values = [
      f"{layer:5d}",
      f"{atmosphere.p[layer] / bar:14.6e}",
      f"{atmosphere.T[layer]:14.6e}",
      f"{atmosphere.g[layer]:14.6e}",
      f"{atmosphere.Kzz[layer]:16.6e}",
      f"{atmosphere.mu[layer]:19.6e}",
    ]
    values.extend(f"{vmr:12.6e}" for vmr in atmosphere.VMR[:, layer])
    print(" ".join(values))

  if any(
    profile is None
    for profile in (
      atmosphere.p_edge,
      atmosphere.z_edge,
      atmosphere.z,
      atmosphere.dz,
      atmosphere.nd,
      atmosphere.rho,
      atmosphere.cT,
      atmosphere.eta,
      atmosphere.nu,
      atmosphere.mfp,
      atmosphere.kappa,
    )
  ):
    raise RuntimeError("derived atmosphere profiles have not been calculated")

  print("\nDerived atmosphere profile")
  print(
    f"{'layer':>5} "
    f"{'z [cm]':>14} "
    f"{'dz [cm]':>14} "
    f"{'nd [cm^-3]':>17} "
    f"{'rho [g cm^-3]':>17} "
    f"{'cT [cm s^-1]':>16} "
    f"{'eta [g cm^-1 s^-1]':>20} "
    f"{'nu [cm^2 s^-1]':>17} "
    f"{'mfp [cm]':>14} "
    f"{'kappa [erg s^-1 cm^-1 K^-1]':>31}"
  )
  for layer in range(atmosphere.n_layers):
    print(
      f"{layer:5d} "
      f"{atmosphere.z[layer]:14.6e} "
      f"{atmosphere.dz[layer]:14.6e} "
      f"{atmosphere.nd[layer]:17.6e} "
      f"{atmosphere.rho[layer]:17.6e} "
      f"{atmosphere.cT[layer]:16.6e} "
      f"{atmosphere.eta[layer]:20.6e} "
      f"{atmosphere.nu[layer]:17.6e} "
      f"{atmosphere.mfp[layer]:14.6e} "
      f"{atmosphere.kappa[layer]:31.6e}"
    )


def main(argv: Sequence[str] | None = None) -> int:
  """Command-line entry point."""

  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("setup", type=Path, help="path to the YAML setup file")
  parser.add_argument(
    "--steps",
    type=int,
    default=None,
    help="override mini_cloud.integrator.n_step (use 0 for initialization only)",
  )
  arguments = parser.parse_args(argv)

  state = run(arguments.setup, n_steps=arguments.steps)
  print_atmosphere(state.atmosphere)
  print(f"\nCompleted {state.step} timestep(s); model time = {state.time:g} s")
  if state.output_path is not None:
    print(f"Wrote {state.output_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
