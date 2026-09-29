"""Tests for optional diffusion boundaries, sources, and loss terms."""

from __future__ import annotations

import unittest

import numpy as np

from pymini_cloud.vert_diffusion import R_gas, vert_diffusion


def simple_profiles(n_layers: int = 8) -> tuple[np.ndarray, ...]:
  """Build a constant-density test column."""

  column_height = 8.0e5
  dz = np.full(n_layers, column_height / n_layers)
  z = column_height - (np.arange(n_layers) + 0.5) * dz[0]
  p_edge = np.linspace(1.0e6, 2.0e6, n_layers + 1)
  mu = np.full(n_layers, 2.33)
  rho = np.full(n_layers, 1.0e-5)
  factor = mu[0] / (R_gas * rho[0])
  temperature = np.empty(n_layers)
  temperature[-1] = factor * p_edge[-1]
  for layer in range(n_layers - 1, 0, -1):
    temperature[layer - 1] = 2.0 * factor * p_edge[layer] - temperature[layer]
  return p_edge, temperature, mu, rho, z, dz


class TestVerticalDiffusionOptions(unittest.TestCase):
  """Exercise optional terms without coupling them to atmospheric setup."""

  def test_double_zero_flux_preserves_constant_and_mass(self) -> None:
    p_edge, temperature, mu, rho, z, dz = simple_profiles()
    q = np.full((2, rho.size), 3.0e-8)
    kzz = np.full(rho.size, 1.0e8)
    result = vert_diffusion(
      q,
      0.0,
      100.0,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
      bottom_boundary="zero_flux",
    )
    np.testing.assert_allclose(result, q, rtol=1.0e-13, atol=0.0)
    np.testing.assert_allclose(
      np.sum(rho * dz * result, axis=1),
      np.sum(rho * dz * q, axis=1),
      rtol=1.0e-13,
      atol=0.0,
    )

  def test_source_and_loss_use_crank_nicolson(self) -> None:
    p_edge, temperature, mu, rho, z, dz = simple_profiles()
    kzz = np.zeros(rho.size)
    dt = 20.0
    initial = np.full((1, rho.size), 2.0e-8)
    source = 4.0e-11
    loss_rate = 1.0e-2
    result = vert_diffusion(
      initial,
      0.0,
      dt,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
      bottom_boundary="zero_flux",
      source=source,
      loss_rate=loss_rate,
    )
    expected = (
      initial * (1.0 - 0.5 * dt * loss_rate) + dt * source
    ) / (1.0 + 0.5 * dt * loss_rate)
    np.testing.assert_allclose(result, expected, rtol=1.0e-14, atol=0.0)

  def test_backward_euler_source_and_loss(self) -> None:
    p_edge, temperature, mu, rho, z, dz = simple_profiles()
    kzz = np.zeros(rho.size)
    dt = 20.0
    initial = np.full((1, rho.size), 2.0e-8)
    source = 4.0e-11
    loss_rate = 1.0e-2
    result = vert_diffusion(
      initial,
      0.0,
      dt,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
      bottom_boundary="zero_flux",
      source=source,
      loss_rate=loss_rate,
      theta=1.0,
    )
    expected = (initial + dt * source) / (1.0 + dt * loss_rate)
    np.testing.assert_allclose(result, expected, rtol=1.0e-14, atol=0.0)

  def test_fixed_top_flux_matches_column_budget(self) -> None:
    p_edge, temperature, mu, rho, z, dz = simple_profiles()
    kzz = np.zeros(rho.size)
    initial = np.full((1, rho.size), 1.0e-8)
    top_flux = 3.0e-14
    dt = 40.0
    result = vert_diffusion(
      initial,
      0.0,
      dt,
      p_edge,
      temperature,
      mu,
      kzz,
      rho,
      z,
      dz,
      top_boundary="fixed_flux",
      top_flux=top_flux,
      bottom_boundary="zero_flux",
    )
    mass_change = np.sum(rho * dz * (result[0] - initial[0]))
    self.assertAlmostEqual(mass_change, top_flux * dt, places=22)


if __name__ == "__main__":
  unittest.main()
