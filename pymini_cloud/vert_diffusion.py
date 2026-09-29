"""Implicit vertical diffusion for atmospheric tracers."""

from __future__ import annotations

import numpy as np
from numba import njit


THETA = 0.5
R_GAS = 8.31446261815324e7  # erg mol^-1 K^-1
# Backwards-compatible public name used by the analytical experiments.
R_gas = R_GAS
Q_MIN = 1.0e-99
EPS = 1.0e-300


def _as_positive_profile(name: str, values: np.ndarray, size: int) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return array


def _as_boundary_vector(
    name: str,
    values: float | np.ndarray,
    n_tracers: int,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(n_tracers, float(array), dtype=np.float64)
    elif array.shape == (n_tracers,):
        array = np.ascontiguousarray(array)
    else:
        raise ValueError(f"{name} must be a scalar or have shape ({n_tracers},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return array


def _as_tracer_field(
    name: str,
    values: float | np.ndarray | None,
    n_tracers: int,
    n_layers: int,
    *,
    nonnegative: bool = False,
) -> np.ndarray:
    if values is None:
        return np.zeros((n_tracers, n_layers), dtype=np.float64)

    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        result = np.full((n_tracers, n_layers), float(array), dtype=np.float64)
    elif array.shape == (n_layers,):
        result = np.broadcast_to(array, (n_tracers, n_layers)).copy()
    elif array.shape == (n_tracers,):
        result = np.broadcast_to(array[:, None], (n_tracers, n_layers)).copy()
    elif array.shape == (n_tracers, n_layers):
        result = np.ascontiguousarray(array)
    else:
        raise ValueError(
            f"{name} must be a scalar or have shape ({n_layers},), "
            f"({n_tracers},), or ({n_tracers}, {n_layers})"
        )
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(result < 0.0):
        raise ValueError(f"{name} must be non-negative")
    return result


@njit(cache=True)
def _solve_tridiagonal(lower, diagonal, upper, rhs):
    """Solve one tridiagonal system using the Thomas algorithm."""
    n = diagonal.size
    c_prime = np.empty(n, dtype=np.float64)
    d_prime = np.empty(n, dtype=np.float64)

    denominator = diagonal[0]
    c_prime[0] = upper[0] / denominator
    d_prime[0] = rhs[0] / denominator

    for i in range(1, n):
        denominator = diagonal[i] - lower[i] * c_prime[i - 1]
        c_prime[i] = upper[i] / denominator if i < n - 1 else 0.0
        d_prime[i] = (rhs[i] - lower[i] * d_prime[i - 1]) / denominator

    solution = np.empty(n, dtype=np.float64)
    solution[n - 1] = d_prime[n - 1]
    for i in range(n - 2, -1, -1):
        solution[i] = d_prime[i] - c_prime[i] * solution[i + 1]
    return solution


@njit(cache=True)
def _vert_diffusion_kernel(
    q,
    q_bottom,
    q_bottom_next,
    dt,
    theta,
    p_edge,
    temperature,
    mu,
    kzz,
    rho,
    z,
    dz,
    top_boundary_code,
    bottom_boundary_code,
    top_flux,
    top_flux_next,
    bottom_flux,
    bottom_flux_next,
    source,
    source_next,
    loss_rate,
):
    n_tracers, n_layers = q.shape
    inverse_dt = 1.0 / dt
    explicit = 1.0 - theta

    rho_edge = np.empty(n_layers + 1, dtype=np.float64)
    kzz_edge = np.empty(n_layers + 1, dtype=np.float64)
    conductance = np.zeros(n_layers + 1, dtype=np.float64)

    rho_edge[0] = p_edge[0] * mu[0] / (R_GAS * temperature[0])
    rho_edge[n_layers] = (
        p_edge[n_layers] * mu[n_layers - 1] / (R_GAS * temperature[n_layers - 1])
    )
    kzz_edge[0] = kzz[0]
    kzz_edge[n_layers] = kzz[n_layers - 1]

    for edge in range(1, n_layers):
        edge_temperature = 0.5 * (temperature[edge - 1] + temperature[edge])
        edge_mu = 0.5 * (mu[edge - 1] + mu[edge])
        rho_edge[edge] = p_edge[edge] * edge_mu / (R_GAS * edge_temperature)
        upper_distance = 0.5 * dz[edge - 1]
        lower_distance = 0.5 * dz[edge]
        resistance = (
            upper_distance / max(kzz[edge - 1], EPS)
            + lower_distance / max(kzz[edge], EPS)
        )
        if kzz[edge - 1] > EPS and kzz[edge] > EPS:
            kzz_edge[edge] = (
                upper_distance + lower_distance
            ) / resistance
        else:
            kzz_edge[edge] = 0.0
        conductance[edge] = (
            rho_edge[edge] * kzz_edge[edge] / (z[edge - 1] - z[edge])
        )

    bottom_conductance = (
        rho_edge[n_layers] * kzz_edge[n_layers] / (0.5 * dz[n_layers - 1])
    )

    result = np.empty_like(q)
    lower = np.zeros(n_layers, dtype=np.float64)
    diagonal = np.empty(n_layers, dtype=np.float64)
    upper = np.zeros(n_layers, dtype=np.float64)
    rhs = np.empty(n_layers, dtype=np.float64)

    for tracer in range(n_tracers):
        lower[:] = 0.0
        upper[:] = 0.0

        for layer in range(n_layers):
            weight = 1.0 / (rho[layer] * dz[layer])
            left_conductance = conductance[layer]
            right_conductance = conductance[layer + 1]

            if layer > 0:
                lower[layer] = -theta * weight * left_conductance
            if layer < n_layers - 1:
                upper[layer] = -theta * weight * right_conductance

            diagonal[layer] = inverse_dt - lower[layer] - upper[layer]
            if layer == n_layers - 1 and bottom_boundary_code == 0:
                diagonal[layer] += theta * weight * bottom_conductance
            diagonal[layer] += theta * loss_rate[tracer, layer]

            diffusion_old = 0.0
            if layer > 0:
                diffusion_old += left_conductance * (
                    q[tracer, layer] - q[tracer, layer - 1]
                )
            if layer < n_layers - 1:
                diffusion_old -= right_conductance * (
                    q[tracer, layer + 1] - q[tracer, layer]
                )
            elif bottom_boundary_code == 0:
                diffusion_old -= bottom_conductance * (
                    q_bottom[tracer] - q[tracer, layer]
                )

            rhs[layer] = inverse_dt * q[tracer, layer]
            rhs[layer] -= explicit * weight * diffusion_old
            rhs[layer] += (
                explicit * source[tracer, layer]
                + theta * source_next[tracer, layer]
            )
            rhs[layer] -= explicit * loss_rate[tracer, layer] * q[tracer, layer]

            if layer == 0 and top_boundary_code == 1:
                rhs[layer] += weight * (
                    explicit * top_flux[tracer] + theta * top_flux_next[tracer]
                )
            if layer == n_layers - 1:
                if bottom_boundary_code == 0:
                    rhs[layer] += (
                        theta * weight * bottom_conductance * q_bottom_next[tracer]
                    )
                elif bottom_boundary_code == 2:
                    rhs[layer] -= weight * (
                        explicit * bottom_flux[tracer]
                        + theta * bottom_flux_next[tracer]
                    )

        solution = _solve_tridiagonal(lower, diagonal, upper, rhs)
        for layer in range(n_layers):
            result[tracer, layer] = max(solution[layer], Q_MIN)

    return result


def vert_diffusion(
    q: np.ndarray,
    q_bottom: float | np.ndarray,
    dt: float,
    p_edge: np.ndarray,
    temperature: np.ndarray,
    mu: np.ndarray,
    kzz: np.ndarray,
    rho: np.ndarray,
    z: np.ndarray,
    dz: np.ndarray,
    *,
    bottom_boundary: str = "dirichlet",
    top_boundary: str = "zero_flux",
    q_bottom_next: float | np.ndarray | None = None,
    top_flux: float | np.ndarray = 0.0,
    top_flux_next: float | np.ndarray | None = None,
    bottom_flux: float | np.ndarray = 0.0,
    bottom_flux_next: float | np.ndarray | None = None,
    source: float | np.ndarray | None = None,
    source_next: float | np.ndarray | None = None,
    loss_rate: float | np.ndarray | None = None,
    theta: float = THETA,
) -> np.ndarray:
    """Advance tracer mixing ratios by one implicit diffusion step.

    Fluxes are positive downward. The default boundary conditions reproduce the
    original model: zero flux at the top and a prescribed mixing ratio at the
    bottom. ``source`` has units of mixing ratio per second and ``loss_rate``
    has units of inverse seconds. Time-varying boundary fluxes and sources can
    provide their end-of-step values through the corresponding ``*_next``
    arguments; otherwise they are held constant over the step. ``theta=0.5``
    gives Crank--Nicolson, while ``theta=1`` gives backward Euler. Values in
    between provide progressively stronger damping.
    """
    q_array = np.ascontiguousarray(q, dtype=np.float64)
    if q_array.ndim != 2:
        raise ValueError("q must have shape (n_tracers, n_layers)")
    if not np.all(np.isfinite(q_array)) or np.any(q_array < 0.0):
        raise ValueError("q must contain finite, non-negative values")
    n_tracers, n_layers = q_array.shape
    if n_layers < 2:
        raise ValueError("at least two atmospheric layers are required")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    if not np.isfinite(theta) or not 0.5 <= theta <= 1.0:
        raise ValueError("theta must be finite and lie in [0.5, 1.0]")

    p_edge_array = _as_positive_profile("p_edge", p_edge, n_layers + 1)
    temperature_array = _as_positive_profile("temperature", temperature, n_layers)
    mu_array = _as_positive_profile("mu", mu, n_layers)
    rho_array = _as_positive_profile("rho", rho, n_layers)
    dz_array = _as_positive_profile("dz", dz, n_layers)
    z_array = _as_positive_profile("z", z, n_layers)
    kzz_array = np.ascontiguousarray(kzz, dtype=np.float64)
    if kzz_array.shape != (n_layers,) or not np.all(np.isfinite(kzz_array)):
        raise ValueError(f"kzz must contain finite values with shape ({n_layers},)")
    if np.any(kzz_array < 0.0):
        raise ValueError("kzz must be non-negative")
    if not np.all(np.diff(p_edge_array) > 0.0):
        raise ValueError("p_edge must increase from the top to the bottom")
    if not np.all(np.diff(z_array) < 0.0):
        raise ValueError("z must decrease from the top to the bottom")

    top_codes = {"zero_flux": 0, "fixed_flux": 1}
    bottom_codes = {"dirichlet": 0, "zero_flux": 1, "fixed_flux": 2}
    if top_boundary not in top_codes:
        raise ValueError("top_boundary must be 'zero_flux' or 'fixed_flux'")
    if bottom_boundary not in bottom_codes:
        raise ValueError(
            "bottom_boundary must be 'dirichlet', 'zero_flux', or 'fixed_flux'"
        )

    q_bottom_array = _as_boundary_vector(
        "q_bottom", q_bottom, n_tracers, nonnegative=True
    )
    if q_bottom_next is None:
        q_bottom_next = q_bottom_array
    q_bottom_next_array = _as_boundary_vector(
        "q_bottom_next", q_bottom_next, n_tracers, nonnegative=True
    )

    top_flux_array = _as_boundary_vector("top_flux", top_flux, n_tracers)
    if top_flux_next is None:
        top_flux_next = top_flux_array
    top_flux_next_array = _as_boundary_vector(
        "top_flux_next", top_flux_next, n_tracers
    )
    bottom_flux_array = _as_boundary_vector("bottom_flux", bottom_flux, n_tracers)
    if bottom_flux_next is None:
        bottom_flux_next = bottom_flux_array
    bottom_flux_next_array = _as_boundary_vector(
        "bottom_flux_next", bottom_flux_next, n_tracers
    )

    source_array = _as_tracer_field("source", source, n_tracers, n_layers)
    if source_next is None:
        source_next = source_array
    source_next_array = _as_tracer_field(
        "source_next", source_next, n_tracers, n_layers
    )
    loss_rate_array = _as_tracer_field(
        "loss_rate", loss_rate, n_tracers, n_layers, nonnegative=True
    )

    return _vert_diffusion_kernel(
        q_array,
        q_bottom_array,
        q_bottom_next_array,
        float(dt),
        float(theta),
        p_edge_array,
        temperature_array,
        mu_array,
        kzz_array,
        rho_array,
        z_array,
        dz_array,
        top_codes[top_boundary],
        bottom_codes[bottom_boundary],
        top_flux_array,
        top_flux_next_array,
        bottom_flux_array,
        bottom_flux_next_array,
        source_array,
        source_next_array,
        loss_rate_array,
    )
