"""Conservative vertical advection for settling cloud moments.

The atmospheric column is ordered from low pressure (top) to high pressure
(bottom). Settling velocities are positive in the downward direction.
"""

from __future__ import annotations

import numpy as np
from numba import njit
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

QMIN = 1.0e-30
GAMMA = 2.0 - np.sqrt(2.0)
BDF_STAGE = 1.0 / (GAMMA * (2.0 - GAMMA))
BDF_OLD = (1.0 - GAMMA)**2 / (GAMMA * (2.0 - GAMMA))


def _as_positive_profile(
  name: str,
  values: FloatArray,
  n_layers: int,
) -> FloatArray:
  """Validate and return a contiguous atmospheric profile."""

  profile = np.asarray(values, dtype=np.float64)
  if profile.shape != (n_layers,):
    raise ValueError(f"{name} must have shape (n_layers,)")
  if np.any(~np.isfinite(profile)) or np.any(profile <= 0.0):
    raise ValueError(f"{name} must contain only finite, positive values")
  return np.ascontiguousarray(profile)


@njit(cache=True)
def _tr_bdf2_downward_solve(
  q: FloatArray,
  settling_velocity: FloatArray,
  rho: FloatArray,
  dz: FloatArray,
  q_top: FloatArray,
  dt: float,
  bottom_boundary_code: int,
) -> FloatArray:
  """Apply the two TR-BDF2 stages to all cloud moments."""

  n_moments, n_layers = q.shape
  q_new = np.empty_like(q)
  q_stage = np.empty(n_layers, dtype=np.float64)
  rhs = np.empty(n_layers, dtype=np.float64)
  rho_edge = np.empty(n_layers + 1, dtype=np.float64)
  velocity_edge = np.empty(n_layers + 1, dtype=np.float64)
  mass_flux_coefficient = np.empty(n_layers + 1, dtype=np.float64)
  implicit_weight = np.empty(n_layers, dtype=np.float64)

  rho_edge[0] = rho[0]
  for layer in range(1, n_layers):
    total_distance = dz[layer - 1] + dz[layer]
    rho_edge[layer] = (
      dz[layer] * rho[layer - 1] + dz[layer - 1] * rho[layer]
    ) / total_distance
  rho_edge[n_layers] = rho[n_layers - 1]
  for layer in range(n_layers):
    implicit_weight[layer] = (
      0.5 * GAMMA * dt / (rho[layer] * dz[layer])
    )

  for moment in range(n_moments):
    velocity_edge[0] = settling_velocity[moment, 0]
    for layer in range(1, n_layers):
      total_distance = dz[layer - 1] + dz[layer]
      velocity_edge[layer] = (
        dz[layer] * settling_velocity[moment, layer - 1]
        + dz[layer - 1] * settling_velocity[moment, layer]
      ) / total_distance
    if bottom_boundary_code == 0:
      velocity_edge[n_layers] = settling_velocity[moment, n_layers - 1]
    else:
      velocity_edge[n_layers] = 0.0

    for layer in range(n_layers + 1):
      mass_flux_coefficient[layer] = (
        rho_edge[layer] * velocity_edge[layer]
      )

    # TR-BDF2 stage 1:
    # (I - gamma*dt/2 L) q_stage = (I + gamma*dt/2 L) q_old
    rhs[0] = q[moment, 0] + implicit_weight[0] * (
      mass_flux_coefficient[0] * q_top[moment]
      - mass_flux_coefficient[1] * q[moment, 0]
    )
    for layer in range(1, n_layers):
      rhs[layer] = q[moment, layer] + implicit_weight[layer] * (
        mass_flux_coefficient[layer] * q[moment, layer - 1]
        - mass_flux_coefficient[layer + 1] * q[moment, layer]
      )

    q_stage[0] = (
      rhs[0]
      + implicit_weight[0] * mass_flux_coefficient[0] * q_top[moment]
    ) / (
      1.0 + implicit_weight[0] * mass_flux_coefficient[1]
    )
    q_stage[0] = max(q_stage[0], QMIN)
    for layer in range(1, n_layers):
      q_stage[layer] = (
        rhs[layer]
        + implicit_weight[layer]
        * mass_flux_coefficient[layer]
        * q_stage[layer - 1]
      ) / (
        1.0
        + implicit_weight[layer] * mass_flux_coefficient[layer + 1]
      )
      q_stage[layer] = max(q_stage[layer], QMIN)

    # TR-BDF2 stage 2:
    # (I - gamma*dt/2 L) q_new = c_stage*q_stage - c_old*q_old
    for layer in range(n_layers):
      rhs[layer] = (
        BDF_STAGE * q_stage[layer]
        - BDF_OLD * q[moment, layer]
      )

    q_new[moment, 0] = (
      rhs[0]
      + implicit_weight[0] * mass_flux_coefficient[0] * q_top[moment]
    ) / (
      1.0 + implicit_weight[0] * mass_flux_coefficient[1]
    )
    q_new[moment, 0] = max(q_new[moment, 0], QMIN)
    for layer in range(1, n_layers):
      q_new[moment, layer] = (
        rhs[layer]
        + implicit_weight[layer]
        * mass_flux_coefficient[layer]
        * q_new[moment, layer - 1]
      ) / (
        1.0
        + implicit_weight[layer] * mass_flux_coefficient[layer + 1]
      )
      q_new[moment, layer] = max(q_new[moment, layer], QMIN)

  return q_new


@njit(cache=True)
def _minmod3(first: float, second: float, third: float) -> float:
  """Return the three-argument minmod slope limiter."""

  if first > 0.0 and second > 0.0 and third > 0.0:
    return min(first, second, third)
  if first < 0.0 and second < 0.0 and third < 0.0:
    return max(first, second, third)
  return 0.0


@njit(cache=True)
def _fill_muscl_fluxes(
  q: FloatArray,
  mass: FloatArray,
  mass_flux_coefficient: FloatArray,
  dz: FloatArray,
  q_top: float,
  substep: float,
  limiter_code: int,
  flux: FloatArray,
  slope: FloatArray,
) -> None:
  """Construct positive, downward MUSCL fluxes for one moment."""

  n_layers = q.size
  left_gradient = (q[0] - q_top) / (0.5 * dz[0])
  right_gradient = 2.0 * (q[1] - q[0]) / (dz[0] + dz[1])
  if limiter_code == 0:
    centred_gradient = (q[1] - q_top) / (0.5 * (dz[0] + dz[1]))
    slope[0] = _minmod3(
      centred_gradient,
      2.0 * left_gradient,
      2.0 * right_gradient,
    )
  elif left_gradient * right_gradient > 0.0:
    ratio = left_gradient / right_gradient
    limiter = max(
      0.0,
      min(2.0 * ratio, (1.0 + 2.0 * ratio) / 3.0, 2.0),
    )
    slope[0] = limiter * right_gradient
  else:
    slope[0] = 0.0
  slope[n_layers - 1] = 0.0
  for layer in range(1, n_layers - 1):
    left_gradient = 2.0 * (q[layer] - q[layer - 1]) / (
      dz[layer - 1] + dz[layer]
    )
    right_gradient = 2.0 * (q[layer + 1] - q[layer]) / (
      dz[layer] + dz[layer + 1]
    )
    centred_gradient = 2.0 * (q[layer + 1] - q[layer - 1]) / (
      dz[layer - 1] + 2.0 * dz[layer] + dz[layer + 1]
    )
    if limiter_code == 0:
      limited = _minmod3(
        centred_gradient,
        2.0 * left_gradient,
        2.0 * right_gradient,
      )
    elif left_gradient * right_gradient > 0.0:
      ratio = left_gradient / right_gradient
      limiter = max(
        0.0,
        min(2.0 * ratio, (1.0 + 2.0 * ratio) / 3.0, 2.0),
      )
      limited = limiter * right_gradient
    else:
      limited = 0.0
    positivity_limit = 2.0 * q[layer] / dz[layer]
    slope[layer] = min(max(limited, -positivity_limit), positivity_limit)

  flux[0] = mass_flux_coefficient[0] * q_top
  for layer in range(n_layers):
    right_state = max(q[layer] + 0.5 * dz[layer] * slope[layer], 0.0)
    if layer < n_layers - 1:
      right_state = min(
        max(right_state, min(q[layer], q[layer + 1])),
        max(q[layer], q[layer + 1]),
      )
    outgoing_flux = mass_flux_coefficient[layer + 1] * right_state

    # This bound is normally inactive because the call is subcycled to a
    # safe Courant number. It guarantees that a forward-Euler building block
    # cannot export more tracer than its donor cell contains.
    maximum_flux = mass[layer] / substep
    flux[layer + 1] = min(outgoing_flux, maximum_flux)


@njit(cache=True)
def _muscl_ssprk2_downward_solve(
  q: FloatArray,
  settling_velocity: FloatArray,
  rho: FloatArray,
  dz: FloatArray,
  q_top: FloatArray,
  dt: float,
  cfl_limit: float,
  limiter_code: int,
  bottom_boundary_code: int,
) -> FloatArray:
  """Apply conservative MUSCL reconstruction with SSP-RK2 subcycling."""

  n_moments, n_layers = q.shape
  q_new = np.empty_like(q)
  rho_edge = np.empty(n_layers + 1, dtype=np.float64)
  velocity_edge = np.empty(n_layers + 1, dtype=np.float64)
  mass_flux_coefficient = np.empty(n_layers + 1, dtype=np.float64)
  cell_mass_coefficient = rho * dz
  mass = np.empty(n_layers, dtype=np.float64)
  stage_mass = np.empty(n_layers, dtype=np.float64)
  next_mass = np.empty(n_layers, dtype=np.float64)
  stage_q = np.empty(n_layers, dtype=np.float64)
  work_q = np.empty(n_layers, dtype=np.float64)
  flux = np.empty(n_layers + 1, dtype=np.float64)
  slope = np.empty(n_layers, dtype=np.float64)

  rho_edge[0] = rho[0]
  for layer in range(1, n_layers):
    total_distance = dz[layer - 1] + dz[layer]
    rho_edge[layer] = (
      dz[layer] * rho[layer - 1] + dz[layer - 1] * rho[layer]
    ) / total_distance
  rho_edge[n_layers] = rho[n_layers - 1]

  for moment in range(n_moments):
    velocity_edge[0] = settling_velocity[moment, 0]
    for layer in range(1, n_layers):
      total_distance = dz[layer - 1] + dz[layer]
      velocity_edge[layer] = (
        dz[layer] * settling_velocity[moment, layer - 1]
        + dz[layer - 1] * settling_velocity[moment, layer]
      ) / total_distance
    if bottom_boundary_code == 0:
      velocity_edge[n_layers] = settling_velocity[moment, n_layers - 1]
    else:
      velocity_edge[n_layers] = 0.0
    for edge in range(n_layers + 1):
      mass_flux_coefficient[edge] = rho_edge[edge] * velocity_edge[edge]

    maximum_courant = 0.0
    for layer in range(n_layers):
      local_courant = (
        dt * mass_flux_coefficient[layer + 1] / cell_mass_coefficient[layer]
      )
      maximum_courant = max(maximum_courant, local_courant)
    n_substeps = max(1, int(np.ceil(maximum_courant / cfl_limit)))
    substep = dt / n_substeps

    for layer in range(n_layers):
      mass[layer] = cell_mass_coefficient[layer] * q[moment, layer]

    for _ in range(n_substeps):
      for layer in range(n_layers):
        work_q[layer] = mass[layer] / cell_mass_coefficient[layer]
      _fill_muscl_fluxes(
        work_q,
        mass,
        mass_flux_coefficient,
        dz,
        q_top[moment],
        substep,
        limiter_code,
        flux,
        slope,
      )
      for layer in range(n_layers):
        stage_mass[layer] = mass[layer] + substep * (
          flux[layer] - flux[layer + 1]
        )
        stage_q[layer] = stage_mass[layer] / cell_mass_coefficient[layer]

      _fill_muscl_fluxes(
        stage_q,
        stage_mass,
        mass_flux_coefficient,
        dz,
        q_top[moment],
        substep,
        limiter_code,
        flux,
        slope,
      )
      for layer in range(n_layers):
        next_mass[layer] = 0.5 * (
          mass[layer]
          + stage_mass[layer]
          + substep * (flux[layer] - flux[layer + 1])
        )
        mass[layer] = max(next_mass[layer], 0.0)

    for layer in range(n_layers):
      q_new[moment, layer] = mass[layer] / cell_mass_coefficient[layer]

  return q_new


@njit(cache=True)
def _muscl_ssprk3_downward_solve(
  q: FloatArray,
  settling_velocity: FloatArray,
  rho: FloatArray,
  dz: FloatArray,
  q_top: FloatArray,
  dt: float,
  cfl_limit: float,
  limiter_code: int,
  bottom_boundary_code: int,
) -> FloatArray:
  """Apply synchronized conservative MUSCL/SSPRK3 substeps."""

  n_moments, n_layers = q.shape
  rho_edge = np.empty(n_layers + 1, dtype=np.float64)
  velocity_edge = np.empty((n_moments, n_layers + 1), dtype=np.float64)
  flux_coefficient = np.empty((n_moments, n_layers + 1), dtype=np.float64)
  cell_mass_coefficient = rho * dz
  mass = np.empty_like(q)
  stage1_mass = np.empty_like(q)
  stage2_mass = np.empty_like(q)
  work_q = np.empty(n_layers, dtype=np.float64)
  flux = np.empty(n_layers + 1, dtype=np.float64)
  slope = np.empty(n_layers, dtype=np.float64)

  rho_edge[0] = rho[0]
  for layer in range(1, n_layers):
    total_distance = dz[layer - 1] + dz[layer]
    rho_edge[layer] = (
      dz[layer] * rho[layer - 1] + dz[layer - 1] * rho[layer]
    ) / total_distance
  rho_edge[n_layers] = rho[n_layers - 1]

  maximum_courant = 0.0
  for moment in range(n_moments):
    velocity_edge[moment, 0] = settling_velocity[moment, 0]
    for layer in range(1, n_layers):
      total_distance = dz[layer - 1] + dz[layer]
      velocity_edge[moment, layer] = (
        dz[layer] * settling_velocity[moment, layer - 1]
        + dz[layer - 1] * settling_velocity[moment, layer]
      ) / total_distance
    if bottom_boundary_code == 0:
      velocity_edge[moment, n_layers] = settling_velocity[moment, n_layers - 1]
    else:
      velocity_edge[moment, n_layers] = 0.0
    for edge in range(n_layers + 1):
      flux_coefficient[moment, edge] = (
        rho_edge[edge] * velocity_edge[moment, edge]
      )
    for layer in range(n_layers):
      local_courant = (
        dt * flux_coefficient[moment, layer + 1]
        / cell_mass_coefficient[layer]
      )
      maximum_courant = max(maximum_courant, local_courant)

  n_substeps = max(1, int(np.ceil(maximum_courant / cfl_limit)))
  substep = dt / n_substeps
  for moment in range(n_moments):
    for layer in range(n_layers):
      mass[moment, layer] = cell_mass_coefficient[layer] * q[moment, layer]

  for _ in range(n_substeps):
    # Stage 1: U1 = U + dt L(U)
    for moment in range(n_moments):
      for layer in range(n_layers):
        work_q[layer] = mass[moment, layer] / cell_mass_coefficient[layer]
      _fill_muscl_fluxes(
        work_q,
        mass[moment],
        flux_coefficient[moment],
        dz,
        q_top[moment],
        substep,
        limiter_code,
        flux,
        slope,
      )
      for layer in range(n_layers):
        stage1_mass[moment, layer] = max(
          mass[moment, layer]
          + substep * (flux[layer] - flux[layer + 1]),
          0.0,
        )

    # Stage 2: U2 = 3/4 U + 1/4 (U1 + dt L(U1))
    for moment in range(n_moments):
      for layer in range(n_layers):
        work_q[layer] = (
          stage1_mass[moment, layer] / cell_mass_coefficient[layer]
        )
      _fill_muscl_fluxes(
        work_q,
        stage1_mass[moment],
        flux_coefficient[moment],
        dz,
        q_top[moment],
        substep,
        limiter_code,
        flux,
        slope,
      )
      for layer in range(n_layers):
        euler_mass = stage1_mass[moment, layer] + substep * (
          flux[layer] - flux[layer + 1]
        )
        stage2_mass[moment, layer] = max(
          0.75 * mass[moment, layer] + 0.25 * euler_mass,
          0.0,
        )

    # Stage 3: U = 1/3 U + 2/3 (U2 + dt L(U2))
    for moment in range(n_moments):
      for layer in range(n_layers):
        work_q[layer] = (
          stage2_mass[moment, layer] / cell_mass_coefficient[layer]
        )
      _fill_muscl_fluxes(
        work_q,
        stage2_mass[moment],
        flux_coefficient[moment],
        dz,
        q_top[moment],
        substep,
        limiter_code,
        flux,
        slope,
      )
      for layer in range(n_layers):
        euler_mass = stage2_mass[moment, layer] + substep * (
          flux[layer] - flux[layer + 1]
        )
        mass[moment, layer] = max(
          mass[moment, layer] / 3.0 + 2.0 * euler_mass / 3.0,
          0.0,
        )

  q_new = np.empty_like(q)
  for moment in range(n_moments):
    for layer in range(n_layers):
      q_new[moment, layer] = mass[moment, layer] / cell_mass_coefficient[layer]
  return q_new


def vert_advection(
  q: FloatArray,
  settling_velocity: FloatArray,
  dt: float,
  rho: FloatArray,
  dz: FloatArray,
  q_top: float | FloatArray = 0.0,
  *,
  method: str = "muscl_ssprk3",
  cfl_limit: float | None = None,
  limiter: str = "koren",
  bottom_boundary: str = "outflow",
) -> FloatArray:
  """Advect cloud moments downward with conservative finite-volume fluxes.

  Parameters
  ----------
  q : ndarray, shape (n_moments, n_layers)
      Cell-centred cloud moment mixing ratios.
  settling_velocity : ndarray, shape (n_moments, n_layers)
      Downward settling velocity for each moment in cm s^-1. Values must be
      non-negative.
  dt : float
      Advection interval in s.
  rho : ndarray, shape (n_layers,)
      Cell-centred gas mass density in g cm^-3.
  dz : ndarray, shape (n_layers,)
      Hydrostatic layer thickness in cm.
  q_top : float or ndarray, shape (n_moments,), optional
      Mixing ratio entering through the top boundary. The default of zero
      gives no incoming cloud from above.
  method : {"muscl_ssprk3", "muscl_ssprk2", "upwind_trbdf2"}, optional
      Numerical scheme. The default uses conservative MUSCL reconstruction,
      synchronized subcycling, and SSP-RK3. SSP-RK2 and the original implicit
      first-order-upwind scheme remain available for comparisons.
  cfl_limit : float, optional
      Maximum density-weighted face Courant number for each MUSCL substep.
      SSP-RK3 accepts values through 0.85; SSP-RK2 is restricted to 0.5.
      It is ignored by the legacy TR-BDF2 method.
  limiter : {"koren", "mc"}, optional
      Slope limiter used by the MUSCL reconstruction. The Koren limiter is
      the default; ``"mc"`` selects the monotonized-central limiter. It is
      ignored by the legacy TR-BDF2 method.
  bottom_boundary : {"outflow", "zero_flux"}, optional
      Lower boundary condition. ``"outflow"`` lets tracer settle out of the
      model column and is the default. ``"zero_flux"`` closes the lower
      boundary. An outflow boundary value is neither needed nor used because
      all settling velocities point downward.

  Returns
  -------
  q_new : ndarray, shape (n_moments, n_layers)
      Moment mixing ratios after the requested advection interval.

  Notes
  -----
  The default lower boundary is free outflow. The default method evolves the
  layer tracer mass ``rho*q*dz`` directly, uses limited piecewise-linear
  reconstruction, and uses one CFL-limited substep shared by all moments. A
  conservative flux cap provides a final positivity safeguard; no post-step
  abundance floor is used.

  The legacy TR-BDF2 method is second-order accurate and L-stable in time,
  first-order upwind in space, and has no settling CFL stability restriction.
  It retains mini-cloud's ``1e-30`` post-stage floor, which can violate exact
  conservation when a large Courant number produces an undershoot.
  """

  moments = np.asarray(q, dtype=np.float64)
  velocity = np.asarray(settling_velocity, dtype=np.float64)

  if moments.ndim != 2:
    raise ValueError("q must have shape (n_moments, n_layers)")
  if moments.shape[0] == 0 or moments.shape[1] < 2:
    raise ValueError("q must contain at least one moment and two layers")
  if velocity.shape != moments.shape:
    raise ValueError("settling_velocity must have the same shape as q")
  if np.any(~np.isfinite(moments)) or np.any(moments < 0.0):
    raise ValueError("q must contain finite, non-negative values")
  if np.any(~np.isfinite(velocity)) or np.any(velocity < 0.0):
    raise ValueError(
      "settling_velocity must contain finite, non-negative values"
    )
  if not np.isfinite(dt) or dt < 0.0:
    raise ValueError("dt must be finite and non-negative")
  methods = {"muscl_ssprk3", "muscl_ssprk2", "upwind_trbdf2"}
  if method not in methods:
    raise ValueError(
      "method must be 'muscl_ssprk3', 'muscl_ssprk2', or 'upwind_trbdf2'"
    )
  if cfl_limit is None:
    cfl_limit = 0.45 if method == "muscl_ssprk2" else 0.8
  maximum_cfl = 0.5 if method == "muscl_ssprk2" else 0.85
  if method != "upwind_trbdf2" and (
    not np.isfinite(cfl_limit) or not 0.0 < cfl_limit <= maximum_cfl
  ):
    raise ValueError(
      f"cfl_limit must be finite and lie in (0, {maximum_cfl:g}]"
    )
  limiters = {"mc": 0, "koren": 1}
  if method != "upwind_trbdf2" and limiter not in limiters:
    raise ValueError("limiter must be either 'koren' or 'mc'")
  bottom_boundaries = {"outflow": 0, "zero_flux": 1}
  if bottom_boundary not in bottom_boundaries:
    raise ValueError("bottom_boundary must be either 'outflow' or 'zero_flux'")
  if dt == 0.0:
    return np.array(moments, copy=True, order="C")

  n_moments, n_layers = moments.shape
  density = _as_positive_profile("rho", rho, n_layers)
  layer_thickness = _as_positive_profile("dz", dz, n_layers)

  top_boundary = np.asarray(q_top, dtype=np.float64)
  if top_boundary.ndim == 0:
    top_boundary = np.full(n_moments, top_boundary, dtype=np.float64)
  if top_boundary.shape != (n_moments,):
    raise ValueError("q_top must be a scalar or have shape (n_moments,)")
  if np.any(~np.isfinite(top_boundary)) or np.any(top_boundary < 0.0):
    raise ValueError("q_top must contain finite, non-negative values")

  moments = np.ascontiguousarray(moments)
  velocity = np.ascontiguousarray(velocity)
  top_boundary = np.ascontiguousarray(top_boundary)
  if method == "muscl_ssprk3":
    return _muscl_ssprk3_downward_solve(
      moments,
      velocity,
      density,
      layer_thickness,
      top_boundary,
      float(dt),
      float(cfl_limit),
      limiters[limiter],
      bottom_boundaries[bottom_boundary],
    )
  if method == "muscl_ssprk2":
    return _muscl_ssprk2_downward_solve(
      moments,
      velocity,
      density,
      layer_thickness,
      top_boundary,
      float(dt),
      float(cfl_limit),
      limiters[limiter],
      bottom_boundaries[bottom_boundary],
    )
  return _tr_bdf2_downward_solve(
    moments,
    velocity,
    density,
    layer_thickness,
    top_boundary,
    float(dt),
    bottom_boundaries[bottom_boundary],
  )
