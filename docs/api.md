# API reference

## Atmospheric state

```{eval-rst}
.. autoclass:: pymini_cloud.data_atmosphere.Atmosphere
   :members:
```

## Background atmosphere

```{eval-rst}
.. autofunction:: pymini_cloud.bg_altitude.bg_altitude

.. autofunction:: pymini_cloud.bg_nd_rho.bg_nd_rho

.. autofunction:: pymini_cloud.bg_viscosity.bg_viscosity

.. autofunction:: pymini_cloud.bg_mfp.bg_mfp

.. autofunction:: pymini_cloud.bg_conductivity.bg_conductivity
```

## Vertical transport

```{eval-rst}
.. autoclass:: pymini_cloud.data_cld_sp.CloudSpecies
   :members:

.. autoclass:: pymini_cloud.data_cld_sp.CloudState
   :members: initialise, n_species, pack_transport, update_from_transport, enforce_floor, transport_bottom_values

.. autoclass:: pymini_cloud.io_read.AdvectionOptions
   :members:

.. autofunction:: pymini_cloud.io_read.read_advection_options

.. autofunction:: pymini_cloud.main.advance_settling

.. autoclass:: pymini_cloud.io_read.DiffusionOptions
   :members:

.. autofunction:: pymini_cloud.io_read.read_diffusion_options

.. autofunction:: pymini_cloud.io_read.read_cloud_species

.. autoclass:: pymini_cloud.io_read.IntegratorOptions

.. autofunction:: pymini_cloud.io_read.read_integrator_options

.. autofunction:: pymini_cloud.main.advance_diffusion

.. autofunction:: pymini_cloud.main.advance_model_step

.. autofunction:: pymini_cloud.main.run

.. autofunction:: pymini_cloud.io_write.write_model_output

.. autofunction:: pymini_cloud.vert_advection.vert_advection

.. autofunction:: pymini_cloud.vert_diffusion.vert_diffusion
```

The API reference is generated from Python signatures and docstrings with
Sphinx autodoc. More modules will be added here as they become part of the
supported public interface.

Both NumPy-style and Google-style docstrings are enabled in the Sphinx
configuration.
