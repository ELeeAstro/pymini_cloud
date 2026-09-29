"""Tests for implicit downward settling advection."""

from __future__ import annotations

import unittest

import numpy as np

from pymini_cloud.vert_advection import vert_advection


class TestVerticalAdvection(unittest.TestCase):
  """Check steady transport, conservation, and independent moment speeds."""

  def test_constant_inflow_profile_is_steady_for_each_moment(self) -> None:
    n_layers = 12
    rho = np.full(n_layers, 1.0e-5)
    dz = np.full(n_layers, 1.0e5)
    q_top = np.array([2.0e-8, 7.0e-8])
    q = np.broadcast_to(q_top[:, None], (2, n_layers)).copy()
    velocity = np.vstack(
      (np.full(n_layers, 20.0), np.full(n_layers, 200.0))
    )

    for limiter in ("koren", "mc"):
      with self.subTest(limiter=limiter):
        result = vert_advection(
          q,
          velocity,
          500.0,
          rho,
          dz,
          q_top,
          limiter=limiter,
        )
        np.testing.assert_allclose(result, q, rtol=2.0e-15, atol=0.0)

  def test_closed_lower_boundary_conserves_column_mass(self) -> None:
    n_layers = 16
    rho = np.geomspace(1.0e-6, 1.0e-4, n_layers)
    dz = np.linspace(5.0e4, 1.5e5, n_layers)
    q = (1.0e-9 + 5.0e-8 * np.sin(np.linspace(0.0, np.pi, n_layers))**2)[
      None, :
    ]
    velocity = np.linspace(10.0, 100.0, n_layers)[None, :]
    initial_mass = np.sum(rho * dz * q[0])

    for method in ("muscl_ssprk2", "upwind_trbdf2"):
      with self.subTest(method=method):
        result = vert_advection(
          q,
          velocity,
          50.0,
          rho,
          dz,
          q_top=0.0,
          method=method,
          bottom_boundary="zero_flux",
        )
        final_mass = np.sum(rho * dz * result[0])
        self.assertAlmostEqual(final_mass, initial_mass, places=20)

  def test_zero_timestep_returns_an_independent_copy(self) -> None:
    q = np.full((1, 4), 2.0e-8)
    velocity = np.full_like(q, 100.0)
    result = vert_advection(
      q,
      velocity,
      0.0,
      np.full(4, 1.0e-5),
      np.full(4, 1.0e5),
    )
    np.testing.assert_array_equal(result, q)
    self.assertFalse(np.shares_memory(result, q))

  def test_large_outer_step_is_positive_and_conservative_when_closed(self) -> None:
    n_layers = 64
    rho = np.full(n_layers, 1.0e-5)
    dz = np.full(n_layers, 1.0e5)
    q = np.zeros((1, n_layers))
    q[0, 12] = 1.0e-7
    velocity = np.full_like(q, 100.0)
    initial_mass = np.sum(rho * dz * q[0])

    # Outer-step Courant number 20; the MUSCL method subcycles internally.
    result = vert_advection(
      q,
      velocity,
      2.0e4,
      rho,
      dz,
      bottom_boundary="zero_flux",
    )
    final_mass = np.sum(rho * dz * result[0])
    self.assertTrue(np.all(result >= 0.0))
    np.testing.assert_allclose(final_mass, initial_mass, rtol=2.0e-14, atol=0.0)

  def test_legacy_tr_bdf2_method_remains_available(self) -> None:
    q = np.full((1, 4), 2.0e-8)
    result = vert_advection(
      q,
      np.full_like(q, 100.0),
      100.0,
      np.full(4, 1.0e-5),
      np.full(4, 1.0e5),
      q_top=2.0e-8,
      method="upwind_trbdf2",
    )
    np.testing.assert_allclose(result, q, rtol=2.0e-15, atol=0.0)


if __name__ == "__main__":
  unittest.main()
