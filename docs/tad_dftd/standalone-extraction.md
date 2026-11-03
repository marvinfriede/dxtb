# `tad_dftd` Standalone Repository Extraction Specification (`H2.8` / `M1.4`–`M1.6`)

`src/tad_dftd` is architected as a self-contained top-level Python package inside `marvinfriede/dxtb` (`claude/tender-dijkstra-cda42s`) with zero imports from `dxtb`, `tad_dftd3`, or `tad_dftd4`. Extracting it into a standalone `tad-dftd` repository requires only file movement and packaging metadata configuration—no algorithmic, numerical, or public API changes.

---

## 1. Standalone Readiness & Isolation Verification (`M1.4` / `M1.5`)

- [x] **Zero `dxtb` / `tad_dftd3` / `tad_dftd4` imports in `src/tad_dftd/`**: Verified via AST inspection across all 56 Python modules in `src/tad_dftd/` (`test/test_tad_dftd/test_isolation.py`). External top-level imports are strictly limited to:
  `['__future__', 'abc', 'ase', 'collections', 'contextlib', 'dataclasses', 'enum', 'functools', 'importlib', 'math', 'numbers', 'numpy', 'tad_mctc', 'tad_multicharge', 'tomli', 'tomllib', 'torch', 'typing']`.
- [x] **Zero `dxtb` test-fixture imports in `test/test_tad_dftd/`**: Verified via AST inspection across all 14 test modules in `test/test_tad_dftd/`. No module imports `dxtb` or `dxtb`'s `test.conftest`, `test.molecules`, or `test.utils`. External top-level imports are strictly limited to:
  `['__future__', 'ase', 'ast', 'configparser', 'fnmatch', 'json', 'math', 'numpy', 'os', 'pathlib', 'pytest', 'shutil', 'subprocess', 'sys', 'tad_dftd', 'tad_mctc', 'tempfile', 'torch', 'typing']`.
- [x] **Strict optional-dependency isolation (`ase`)**: `src/tad_dftd/__init__.py` does not import `ase` or `tad_dftd.ase`. `src/tad_dftd/ase.py` guards `ase` imports behind `_require_ase()` so `import tad_dftd` and `import tad_dftd.ase` succeed when `ase` is absent, and instantiating `DFTD`, `DFTD3`, or `DFTD4` without `ase` raises a clear `ImportError` (`pip install 'tad-dftd[ase]'` or `pip install ase`) (`test/test_tad_dftd/test_ase_isolation.py`).
- [x] **Self-contained runtime resource loading**: Binary reference/data tables (`src/tad_dftd/data/*.pt`, `src/tad_dftd/reference/*.pt`) and TOML functional parameter tables (`src/tad_dftd/param/*.toml`) are loaded via `importlib.resources.files` relative to `tad_dftd` subpackages and are verified under an isolated built-wheel installation (`TAD_DFTD_VERIFY_WHEEL=1`).
- [x] **Self-contained version handling**: `src/tad_dftd/__version__.py` defines `__version__ = "0.1.0"` without referencing `dxtb.__version__`.
- [x] **Uniform `Apache-2.0` licensing**: Every Python module in `src/tad_dftd/`, `test/test_tad_dftd/`, and `tools/generate_tad_dftd_data.py` carries `SPDX-Identifier: Apache-2.0`, and `src/tad_dftd/LICENSE` (Apache-2.0), `src/tad_dftd/PROVENANCE.md`, and `src/tad_dftd/py.typed` are bundled inside `src/tad_dftd/` (with `LICENSE` and `PROVENANCE.md` also placed at the root of the standalone `tad-dftd` repository).

---

## 2. Complete 88-File Standalone Extraction Manifest (`M1.4`)

All **88 files** listed below are self-contained and move directly from `marvinfriede/dxtb` into the standalone `tad-dftd` repository:
- **68 files** in `src/tad_dftd/` (`56` `.py` modules, `6` `.pt` binary tensor tables, `3` `.toml` functional parameter files, `1` `LICENSE`, `1` `PROVENANCE.md`, `1` `py.typed`) -> `src/tad_dftd/` (plus top-level copies of `LICENSE` and `PROVENANCE.md` at the repository root).
- **15 files** in `test/test_tad_dftd/` -> `test/`
- **4 files** in `docs/tad_dftd/` -> `docs/`
- **1 file** in `tools/generate_tad_dftd_data.py` -> `tools/generate_tad_dftd_data.py`

| Source Path in `dxtb` | Target Path in `tad-dftd` | SPDX Identifier |
|---|---|---|
| `src/tad_dftd/LICENSE` | `src/tad_dftd/LICENSE` (and repo root `LICENSE`) | `N/A (Apache-2.0 license text)` |
| `src/tad_dftd/PROVENANCE.md` | `src/tad_dftd/PROVENANCE.md` (and repo root `PROVENANCE.md`) | `N/A (provenance documentation)` |
| `src/tad_dftd/__init__.py` | `src/tad_dftd/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/__version__.py` | `src/tad_dftd/__version__.py` | `Apache-2.0` |
| `src/tad_dftd/ase.py` | `src/tad_dftd/ase.py` | `Apache-2.0` |
| `src/tad_dftd/charge/__init__.py` | `src/tad_dftd/charge/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/charge/eeq.py` | `src/tad_dftd/charge/eeq.py` | `Apache-2.0` |
| `src/tad_dftd/cutoff.py` | `src/tad_dftd/cutoff.py` | `Apache-2.0` |
| `src/tad_dftd/damping/__init__.py` | `src/tad_dftd/damping/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/damping/atm.py` | `src/tad_dftd/damping/atm.py` | `Apache-2.0` |
| `src/tad_dftd/damping/base.py` | `src/tad_dftd/damping/base.py` | `Apache-2.0` |
| `src/tad_dftd/damping/cso.py` | `src/tad_dftd/damping/cso.py` | `Apache-2.0` |
| `src/tad_dftd/damping/fermi.py` | `src/tad_dftd/damping/fermi.py` | `Apache-2.0` |
| `src/tad_dftd/damping/koide.py` | `src/tad_dftd/damping/koide.py` | `Apache-2.0` |
| `src/tad_dftd/damping/mzero.py` | `src/tad_dftd/damping/mzero.py` | `Apache-2.0` |
| `src/tad_dftd/damping/optimizedpower.py` | `src/tad_dftd/damping/optimizedpower.py` | `Apache-2.0` |
| `src/tad_dftd/damping/param.py` | `src/tad_dftd/damping/param.py` | `Apache-2.0` |
| `src/tad_dftd/damping/rational.py` | `src/tad_dftd/damping/rational.py` | `Apache-2.0` |
| `src/tad_dftd/damping/screened.py` | `src/tad_dftd/damping/screened.py` | `Apache-2.0` |
| `src/tad_dftd/damping/select.py` | `src/tad_dftd/damping/select.py` | `Apache-2.0` |
| `src/tad_dftd/damping/z.py` | `src/tad_dftd/damping/z.py` | `Apache-2.0` |
| `src/tad_dftd/damping/zero.py` | `src/tad_dftd/damping/zero.py` | `Apache-2.0` |
| `src/tad_dftd/data/__init__.py` | `src/tad_dftd/data/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/data/d1.py` | `src/tad_dftd/data/d1.py` | `Apache-2.0` |
| `src/tad_dftd/data/d2.py` | `src/tad_dftd/data/d2.py` | `Apache-2.0` |
| `src/tad_dftd/data/en.py` | `src/tad_dftd/data/en.py` | `Apache-2.0` |
| `src/tad_dftd/data/hardness.py` | `src/tad_dftd/data/hardness.py` | `Apache-2.0` |
| `src/tad_dftd/data/r4r2.py` | `src/tad_dftd/data/r4r2.py` | `Apache-2.0` |
| `src/tad_dftd/data/radii.py` | `src/tad_dftd/data/radii.py` | `Apache-2.0` |
| `src/tad_dftd/data/table.py` | `src/tad_dftd/data/table.py` | `Apache-2.0` |
| `src/tad_dftd/data/tanh.py` | `src/tad_dftd/data/tanh.py` | `Apache-2.0` |
| `src/tad_dftd/data/wfpair-d3s.pt` | `src/tad_dftd/data/wfpair-d3s.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/data/wfpair-d4s.pt` | `src/tad_dftd/data/wfpair-d4s.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/data/wfpair.py` | `src/tad_dftd/data/wfpair.py` | `Apache-2.0` |
| `src/tad_dftd/data/zeff.py` | `src/tad_dftd/data/zeff.py` | `Apache-2.0` |
| `src/tad_dftd/defaults.py` | `src/tad_dftd/defaults.py` | `Apache-2.0` |
| `src/tad_dftd/disp.py` | `src/tad_dftd/disp.py` | `Apache-2.0` |
| `src/tad_dftd/exception.py` | `src/tad_dftd/exception.py` | `Apache-2.0` |
| `src/tad_dftd/integrator/__init__.py` | `src/tad_dftd/integrator/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/integrator/gauss_legendre.py` | `src/tad_dftd/integrator/gauss_legendre.py` | `Apache-2.0` |
| `src/tad_dftd/integrator/trapezoid.py` | `src/tad_dftd/integrator/trapezoid.py` | `Apache-2.0` |
| `src/tad_dftd/model/__init__.py` | `src/tad_dftd/model/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/model/base.py` | `src/tad_dftd/model/base.py` | `Apache-2.0` |
| `src/tad_dftd/model/c6.py` | `src/tad_dftd/model/c6.py` | `Apache-2.0` |
| `src/tad_dftd/model/catalogue.py` | `src/tad_dftd/model/catalogue.py` | `Apache-2.0` |
| `src/tad_dftd/model/d1.py` | `src/tad_dftd/model/d1.py` | `Apache-2.0` |
| `src/tad_dftd/model/d2.py` | `src/tad_dftd/model/d2.py` | `Apache-2.0` |
| `src/tad_dftd/model/d3.py` | `src/tad_dftd/model/d3.py` | `Apache-2.0` |
| `src/tad_dftd/model/d4.py` | `src/tad_dftd/model/d4.py` | `Apache-2.0` |
| `src/tad_dftd/model/weights.py` | `src/tad_dftd/model/weights.py` | `Apache-2.0` |
| `src/tad_dftd/ncoord/__init__.py` | `src/tad_dftd/ncoord/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/pairs.py` | `src/tad_dftd/pairs.py` | `Apache-2.0` |
| `src/tad_dftd/pairvalues.py` | `src/tad_dftd/pairvalues.py` | `Apache-2.0` |
| `src/tad_dftd/param/__init__.py` | `src/tad_dftd/param/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/param/aliases.toml` | `src/tad_dftd/param/aliases.toml` | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/d3_parameters.toml` | `src/tad_dftd/param/d3_parameters.toml` | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/d4_parameters.toml` | `src/tad_dftd/param/d4_parameters.toml` | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/functional.py` | `src/tad_dftd/param/functional.py` | `Apache-2.0` |
| `src/tad_dftd/py.typed` | `src/tad_dftd/py.typed` | `N/A (PEP 561 marker)` |
| `src/tad_dftd/reference/__init__.py` | `src/tad_dftd/reference/__init__.py` | `Apache-2.0` |
| `src/tad_dftd/reference/d3-c6.pt` | `src/tad_dftd/reference/d3-c6.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d3.py` | `src/tad_dftd/reference/d3.py` | `Apache-2.0` |
| `src/tad_dftd/reference/d3s-legacy-c6.pt` | `src/tad_dftd/reference/d3s-legacy-c6.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d4-ref.pt` | `src/tad_dftd/reference/d4-ref.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d4.py` | `src/tad_dftd/reference/d4.py` | `Apache-2.0` |
| `src/tad_dftd/reference/d4srev-ref.pt` | `src/tad_dftd/reference/d4srev-ref.pt` | `N/A (binary tensor asset)` |
| `src/tad_dftd/threebody.py` | `src/tad_dftd/threebody.py` | `Apache-2.0` |
| `src/tad_dftd/twobody.py` | `src/tad_dftd/twobody.py` | `Apache-2.0` |
| `src/tad_dftd/typing.py` | `src/tad_dftd/typing.py` | `Apache-2.0` |
| `test/test_tad_dftd/__init__.py` | `test/__init__.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_ase.py` | `test/test_ase.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_ase_isolation.py` | `test/test_ase_isolation.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_catalogue_and_param.py` | `test/test_catalogue_and_param.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_cutoff_and_pairs.py` | `test/test_cutoff_and_pairs.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_d1_and_d2.py` | `test/test_d1_and_d2.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_d3_and_d3s.py` | `test/test_d3_and_d3s.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_d4.py` | `test/test_d4.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_damping.py` | `test/test_damping.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_data.py` | `test/test_data.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_integrator.py` | `test/test_integrator.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_isolation.py` | `test/test_isolation.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_ncoord.py` | `test/test_ncoord.py` | `Apache-2.0` |
| `test/test_tad_dftd/test_periodic_and_transforms.py` | `test/test_periodic_and_transforms.py` | `Apache-2.0` |
| `docs/tad_dftd/architecture-and-mapping.md` | `docs/architecture-and-mapping.md` | `N/A (markdown documentation)` |
| `docs/tad_dftd/ase.md` | `docs/ase.md` | `N/A (markdown documentation)` |
| `docs/tad_dftd/standalone-extraction.md` | `docs/standalone-extraction.md` | `N/A (markdown documentation)` |
| `docs/tad_dftd/upstream_multicharge_periodic_eeq_issue.md` | `docs/upstream_multicharge_periodic_eeq_issue.md` | `N/A (markdown documentation)` |
| `tools/generate_tad_dftd_data.py` | `tools/generate_tad_dftd_data.py` | `Apache-2.0` |

### SPDX Header Audit Summary (72 `.py` files)

- **`Apache-2.0`**: 72 / 72 `.py` files (`56` in `src/tad_dftd/`, `15` in `test/test_tad_dftd/`, `1` in `tools/`).
- **Missing SPDX headers**: `0` files.
- **Package license**: `Apache-2.0`.

---

## 3. Verified Standalone Packaging Metadata & Dependency Lower Bounds (`M1.5`)

### 3.1 Dependency Lower-Bound Verification

| Dependency | Version Specifier | Required APIs Used by `tad_dftd` |
|---|---|---|
| `python` | `>=3.10, <3.15` | PEP 604 union types, structural typing, `tomllib` (3.11+) / `tomli` (3.10), validated across Python 3.10, 3.11, 3.12, 3.14 |
| `torch` | `>=2.7.0,<3` | `torch.func.{vmap,jacrev,jacfwd,grad}`, `torch.compile`, `torch.utils.checkpoint.checkpoint`, `torch.load(..., weights_only=True)` |
| `tad-mctc` | `>=0.11.3` | `tad_mctc.io.structure.Structure`, `tad_mctc.neighbor.{NeighborList,PeriodicShifts,TripleList,build_neighborlists,build_periodic_shifts,split_lattice,pair_distance_squared,gather_index}`, `tad_mctc.ncoord.{CNModel,cn_d3,cn_d4,cn_eeq,cn_eeq_en,cn_gfn2,erf_count,exp_count,gfn2_count,d4_en_weight,en_difference,cut_coordination_number}`, `tad_mctc.tree.{Node,child,context}`, `tad_mctc.autograd.{is_functorch_tensor,is_vmapped}`, `tad_mctc.tools.is_compiling`, `tad_mctc.batch.{real_pairs,real_triples}` |
| `tad-multicharge` | `>=0.8.0` | `tad_multicharge.model.eeq.EEQModel.param2019` accepting `tad_mctc.io.structure.Structure` |
| `tomli` | `python_version < "3.11"` | TOML parser fallback in `tad_dftd.param.functional` when `tomllib` is unavailable |
| `typing-extensions` | unpinned | `TypeAlias`, `Self`, `Literal`, `overload` compatibility |
| `ase` *(optional `[ase]`)* | `>=3.22.0` | `ase.calculators.calculator.{Calculator,all_changes,PropertyNotImplementedError}`, `ase.constraints.full_cell_filter`, `ase.Atoms` |
| `numpy` *(optional `[ase]`)* | `>=1.23` | NumPy array conversion at the `tad_dftd.ase` boundary |

### 3.2 Complete `setup.cfg` & `pyproject.toml` Specification for `tad-dftd`

`pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"
```

`setup.cfg`:

```ini
[metadata]
name = tad-dftd
version = attr: tad_dftd.__version__.__version__
description = Unified Differentiable DFT-D Dispersion Corrections in PyTorch
long_description = file: README.md
long_description_content_type = text/markdown
author = Marvin Friede
license = Apache-2.0
license_files =
    LICENSE
    PROVENANCE.md
platforms = any
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Science/Research
    License :: OSI Approved :: Apache Software License
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    Programming Language :: Python :: 3.12
    Programming Language :: Python :: 3.13
    Programming Language :: Python :: 3.14
    Topic :: Scientific/Engineering :: Chemistry
    Topic :: Scientific/Engineering :: Physics

[options]
packages = find:
package_dir =
    =src
python_requires = >=3.10, <3.15
install_requires =
    tad-mctc>=0.11.3
    tad-multicharge>=0.8.0
    tomli; python_version < "3.11"
    torch>=2.7.0,<3
    typing-extensions

[options.packages.find]
where = src

[options.package_data]
tad_dftd =
    py.typed
    LICENSE
    PROVENANCE.md
    data/*.pt
    param/*.toml
    reference/*.pt

[options.extras_require]
ase =
    ase>=3.22.0
    numpy>=1.23
dev =
    ase>=3.22.0
    black
    covdefaults
    coverage
    isort
    mypy
    pytest
    pytest-cov
```

---

## 4. Standalone Repository Creation & Release Sequence (`M1.5`)

1. **Initialize the standalone `tad-dftd` repository**:
   - Copy the 88 files from the manifest in Section 2 into their target paths.
   - Copy `src/tad_dftd/{LICENSE,PROVENANCE.md}` to the repository root as well so both wheel distributions (`src/tad_dftd/`) and sdist/GitHub repository inspection expose the Apache-2.0 license and provenance documentation.
   - Add `pyproject.toml`, `setup.cfg` (Section 3.2), `README.md`, and CI workflows testing Python `3.10`–`3.14` and PyTorch `2.7`–`2.14` (both with and without the `[ase]` extra).
2. **Validate the standalone package prior to tagging**:
   - Build sdist and wheel (`python -m build` or `pip wheel --no-deps --no-build-isolation .`).
   - Verify all 6 `.pt` files (`data/wfpair-d3s.pt`, `data/wfpair-d4s.pt`, `reference/d3-c6.pt`, `reference/d3s-legacy-c6.pt`, `reference/d4-ref.pt`, `reference/d4srev-ref.pt`), all 3 `.toml` files (`param/aliases.toml`, `param/d3_parameters.toml`, `param/d4_parameters.toml`), `py.typed`, `LICENSE`, and `PROVENANCE.md` are present inside the wheel archive.
   - Run `TAD_DFTD_VERIFY_WHEEL=1 pytest test/` across the supported Python/PyTorch matrix.
3. **Tag and publish `tad-dftd` `v0.1.0`**:
   - Tag `v0.1.0` matching `src/tad_dftd/__version__.py` (`__version__ = "0.1.0"`).

---

## 5. Downstream `dxtb` Dependency Cutover Plan (`M1.6`)

Once `tad-dftd>=0.1.0` is published and installable in `dxtb`'s target environments, execute the downstream cutover in `marvinfriede/dxtb` as a **single atomic branch/PR** (only after explicit maintainer authorization):

1. **Zero import-path changes in `src/dxtb/` or `test/`**:
   - All `dxtb` runtime modules (`src/dxtb/_src/components/classicals/dispersion/{base,d3,d4,d4sc,factory}.py`, `src/dxtb/_src/param/dispersion.py`) and dispersion tests (`test/test_classical/test_dispersion/*`) already import `tad_dftd` exclusively via top-level package imports (`import tad_dftd`, `from tad_dftd import ...`).
   - Do **not** reintroduce `tad_dftd3`, `tad_dftd4`, or `tad_dftd.interface` compatibility shims.
2. **Update `dxtb/setup.cfg`**:
   - Add `tad-dftd>=0.1.0,<0.2.0` (or `tad-dftd>=0.1.0`) to `install_requires` under `[options]`.
   - Remove the `tad_dftd = ...` block (`lines 102–108`) from `[options.package_data]`.
3. **Remove the in-tree `tad_dftd` files from `dxtb`**:
   - Delete `src/tad_dftd/` (`68` files).
   - Delete `test/test_tad_dftd/` (`15` files).
   - Delete `tools/generate_tad_dftd_data.py` (`1` file).
   - Delete `docs/tad_dftd/` (`4` files).
4. **Verify cutover gate in `dxtb`**:
   - Install `tad-dftd>=0.1.0` into the 5 supported environments (`py3.10-torch2.7.0` through `py3.14-torch2.14.1`).
   - Verify `python -c "import tad_dftd, dxtb; assert 'src/tad_dftd' not in tad_dftd.__file__"` confirms `tad_dftd` resolves from `site-packages`.
   - Run `pytest test/test_classical/test_dispersion test/test_interaction test/test_calculator` and static analysis (`mypy`, `isort --check`, `black --check`).
