# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install (from source):**
```bash
pip install .
```

**Install with extras:**
```bash
pip install '.[test]'   # test dependencies
pip install '.[dev]'    # dev/lint dependencies
pip install '.[doc]'    # documentation dependencies
```

**Run all tests** (from the `test/` folder):
```bash
cd test && pytest
```

**Run a single test file:**
```bash
cd test && pytest unit/test_scene_utils.py
```

**Force CPU backend (LLVM instead of CUDA):**
```bash
cd test && pytest --cpu
```

**Lint:**
```bash
pylint src/
```

**Build documentation** (from `doc/`):
```bash
cd doc && make html
# Serve with: python -m http.server --dir build/html
```

## Architecture

Sionna RT is a differentiable ray tracer for radio propagation simulation, built on [Mitsuba 3](https://github.com/mitsuba-renderer/mitsuba3) and [DrJit](https://github.com/mitsuba-renderer/drjit). All heavy computation runs through DrJit's JIT-compiled kernels and supports GPU (CUDA) or CPU (LLVM) backends. It is interoperable with TensorFlow, PyTorch, and JAX tensors.

### Mitsuba variant

At import time (`src/sionna/rt/__init__.py`), the package sets the Mitsuba variant to `cuda_ad_mono_polarized` (GPU) with fallback to `llvm_ad_mono_polarized` (CPU). Two custom Mitsuba plugins — `SlicedPathIntegrator`/`SlicedDepthIntegrator` (`sliced_integrator.py`) and `TwosidedAreaEmitter` (`twosided_area.py`) — are registered as Mitsuba extensions and **reloaded via a variant-change callback** whenever `mi.set_variant()` is called.

### Core classes

| Class | File | Role |
|---|---|---|
| `Scene` | `scene.py` | Central object. Holds scene objects, transmitters, receivers, materials, cameras. Load via `load_scene()` or `load_scene_from_string()`. |
| `PathSolver` | `path_solvers/path_solver.py` | Computes propagation paths (LOS, specular/diffuse reflection, refraction) between all TX/RX antennas. Supports synthetic arrays. |
| `Paths` | `path_solvers/paths.py` | Output of `PathSolver`. Provides `.cir()` and `.cfr()` for channel impulse/frequency responses and Doppler. |
| `RadioMapSolver` | `radio_map_solvers/radio_map_solver.py` | Monte Carlo solver that generates coverage/path-gain maps over a measurement surface. |
| `PlanarRadioMap` / `MeshRadioMap` | `radio_map_solvers/` | Two measurement surface types: flat grid or arbitrary mesh. |
| `RadioMaterial` | `radio_materials/radio_material.py` | EM material defined by relative permittivity, conductivity, and thickness. Uses slab model for reflection/transmission Jones matrices. |
| `ITURadioMaterial` | `radio_materials/itu_material.py` | Frequency-dependent material using ITU-R P.2040 coefficients. |
| `Transmitter` / `Receiver` | `radio_devices/` | Radio devices placed in the scene, each carrying an `AntennaArray`. |
| `AntennaArray` / `PlanarArray` | `antenna_array.py` | Array geometry; supports synthetic array mode. |
| `AntennaPattern` | `antenna_pattern.py` | Antenna radiation pattern; extensible via `register_antenna_pattern`. |
| `SceneObject` | `scene_object.py` | Wrapper around a Mitsuba shape with an associated `RadioMaterial`. |

### Subdirectory map

```
src/sionna/rt/
├── scene.py                  # Scene class + load_scene()
├── scene_object.py           # SceneObject (geometry + material)
├── scene_utils.py            # XML processing, mesh helpers
├── path_solvers/             # PathSolver pipeline
│   ├── path_solver.py        # Orchestrator
│   ├── sb_candidate_generator.py  # Shooting & bouncing
│   ├── image_method.py       # Specular path refinement
│   ├── field_calculator.py   # E-field / Jones matrix computation
│   ├── paths.py              # Paths output object
│   └── paths_buffer.py       # Internal data buffers
├── radio_map_solvers/        # Coverage map computation
├── radio_materials/          # Material models + scattering patterns
├── radio_devices/            # Transmitter / Receiver
├── antenna_array.py          # Array geometry
├── antenna_pattern.py        # Radiation patterns + registries
├── registry/                 # Generic Registry base class
├── scenes/                   # Bundled example scenes (Munich, Florence, etc.)
├── utils/                    # Math utilities (Jones, geometry, hashing, …)
├── sliced_integrator.py      # Custom Mitsuba integrator plugin
├── twosided_area.py          # Custom Mitsuba emitter plugin
├── preview.py                # Interactive 3D preview (pythreejs)
└── renderer.py               # Mitsuba-based rendering
```

### Extension points

Custom components are registered via registry objects:
- **Antenna patterns**: `register_antenna_pattern`, `register_polarization`, `register_polarization_model`
- **Scattering patterns**: `register_scattering_pattern` (built-ins: `LambertianPattern`, `BackscatteringPattern`, `DirectivePattern`)
- **Custom radio materials**: subclass `RadioMaterialBase`

## Contributing

- All commits must be signed off with `git commit -s` (DCO requirement).
- New files must include the Apache-2.0 license header (see any existing source file).
- PRs branch from `main`; run linting and all tests before submitting.
