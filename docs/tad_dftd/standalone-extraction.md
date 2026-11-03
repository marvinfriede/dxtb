# `tad_dftd` Standalone Repository Extraction Specification (`H2.8` / `M1.4`–`M1.6`)

`src/tad_dftd` is architected as a self-contained top-level Python package inside `marvinfriede/dxtb` (`claude/tender-dijkstra-cda42s`, merged at `f51776e23348df71bebf00fd268bdc6615d1c934`) with zero imports from `dxtb`, `tad_dftd3`, or `tad_dftd4`. Extracting it into a standalone `tad-dftd` repository requires only file movement and packaging metadata configuration—no algorithmic, numerical, or public API changes.

---

## 1. Standalone Readiness & Isolation Verification (`M1.4` / `M1.5`)

- [x] **Zero `dxtb` / `tad_dftd3` / `tad_dftd4` imports in `src/tad_dftd/`**: Verified via AST inspection across all 56 Python modules in `src/tad_dftd/` (`test/test_tad_dftd/test_isolation.py`). External top-level imports are strictly limited to:
  `['__future__', 'abc', 'ase', 'collections', 'contextlib', 'dataclasses', 'enum', 'functools', 'importlib', 'math', 'numbers', 'numpy', 'tad_mctc', 'tad_multicharge', 'tomli', 'tomllib', 'torch', 'typing']`.
- [x] **Zero `dxtb` test-fixture imports in `test/test_tad_dftd/`**: Verified via AST inspection across all 15 test modules in `test/test_tad_dftd/`. No module imports `dxtb` or `dxtb`'s `test.conftest`, `test.molecules`, or `test.utils`. External top-level imports are strictly limited to:
  `['__future__', 'ase', 'ast', 'configparser', 'fnmatch', 'json', 'math', 'numpy', 'os', 'pathlib', 'pytest', 'shutil', 'subprocess', 'sys', 'tad_dftd', 'tad_mctc', 'tempfile', 'torch', 'typing']`.
- [x] **Strict optional-dependency isolation (`ase`)**: `src/tad_dftd/__init__.py` does not import `ase` or `tad_dftd.ase`. `src/tad_dftd/ase.py` guards `ase` imports behind `_require_ase()` so `import tad_dftd` and `import tad_dftd.ase` succeed when `ase` is absent, and instantiating `DFTD`, `DFTD3`, or `DFTD4` without `ase` raises a clear `ImportError` (`pip install 'tad-dftd[ase]'` or `pip install ase`) (`test/test_tad_dftd/test_ase_isolation.py`).
- [x] **Self-contained runtime resource loading**: Binary reference/data tables (`src/tad_dftd/data/*.pt`, `src/tad_dftd/reference/*.pt`) and TOML functional parameter tables (`src/tad_dftd/param/*.toml`) are loaded via `importlib.resources.files` relative to `tad_dftd` subpackages and are verified under an isolated built-wheel installation (`TAD_DFTD_VERIFY_WHEEL=1`).
- [x] **Self-contained version handling**: `src/tad_dftd/__version__.py` defines `__version__ = "0.1.0"` without referencing `dxtb.__version__`.
- [x] **Self-contained license & provenance files**: `src/tad_dftd/LICENSE` (Apache-2.0), `src/tad_dftd/COPYING` (GPL-3.0), `src/tad_dftd/COPYING.LESSER` (LGPL-3.0), `src/tad_dftd/PROVENANCE.md`, and `src/tad_dftd/py.typed` are bundled inside `src/tad_dftd/` and copied to the root of the standalone `tad-dftd` repository during extraction.

---

## 2. Complete 90-File Standalone Extraction Manifest (`M1.4`)

All **90 files** listed below are self-contained and move directly from `marvinfriede/dxtb` into the standalone `tad-dftd` repository:
- **68 files** in `src/tad_dftd/` (`56` `.py` modules, `6` `.pt` binary tensor tables, `3` `.toml` functional parameter files, `3` license/provenance files, `1` `PROVENANCE.md`, `1` `py.typed`) -> `src/tad_dftd/` (plus top-level copies of `LICENSE`, `COPYING`, `COPYING.LESSER`, `PROVENANCE.md` at the repository root).
- **15 files** in `test/test_tad_dftd/` -> `test/`
- **4 files** in `docs/tad_dftd/` -> `docs/`
- **1 file** in `tools/generate_tad_dftd_data.py` -> `tools/generate_tad_dftd_data.py`

*(Note: 68 + 15 + 4 + 1 = 88 + 2 data `.pt` files included in the 68 `src/tad_dftd` files; exact row count below is 90 files: 70 in `src/tad_dftd/`, 15 in `test/test_tad_dftd/`, 4 in `docs/tad_dftd/`, 1 in `tools/`.)*

| Source Path in `dxtb` | Target Path in `tad-dftd` | Size (Bytes) | SPDX Identifier |
|---|---|---:|---|
| `src/tad_dftd/COPYING` | `src/tad_dftd/COPYING` (and repo root `COPYING`) | 35149 | `N/A (GPL-3.0 license text)` |
| `src/tad_dftd/COPYING.LESSER` | `src/tad_dftd/COPYING.LESSER` (and repo root `COPYING.LESSER`) | 7652 | `N/A (LGPL-3.0 license text)` |
| `src/tad_dftd/LICENSE` | `src/tad_dftd/LICENSE` (and repo root `LICENSE`) | 10173 | `N/A (Apache-2.0 license text)` |
| `src/tad_dftd/PROVENANCE.md` | `src/tad_dftd/PROVENANCE.md` (and repo root `PROVENANCE.md`) | 9002 | `N/A (provenance documentation)` |
| `src/tad_dftd/__init__.py` | `src/tad_dftd/__init__.py` | 3407 | `Apache-2.0` |
| `src/tad_dftd/__version__.py` | `src/tad_dftd/__version__.py` | 780 | `Apache-2.0` |
| `src/tad_dftd/ase.py` | `src/tad_dftd/ase.py` | 57367 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/charge/__init__.py` | `src/tad_dftd/charge/__init__.py` | 3189 | `Apache-2.0` |
| `src/tad_dftd/charge/eeq.py` | `src/tad_dftd/charge/eeq.py` | 2005 | `Apache-2.0` |
| `src/tad_dftd/cutoff.py` | `src/tad_dftd/cutoff.py` | 4968 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/damping/__init__.py` | `src/tad_dftd/damping/__init__.py` | 1069 | `Apache-2.0` |
| `src/tad_dftd/damping/atm.py` | `src/tad_dftd/damping/atm.py` | 1096 | `Apache-2.0` |
| `src/tad_dftd/damping/base.py` | `src/tad_dftd/damping/base.py` | 8919 | `Apache-2.0` |
| `src/tad_dftd/damping/cso.py` | `src/tad_dftd/damping/cso.py` | 3505 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/fermi.py` | `src/tad_dftd/damping/fermi.py` | 2628 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/koide.py` | `src/tad_dftd/damping/koide.py` | 5763 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/mzero.py` | `src/tad_dftd/damping/mzero.py` | 2032 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/optimizedpower.py` | `src/tad_dftd/damping/optimizedpower.py` | 1864 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/param.py` | `src/tad_dftd/damping/param.py` | 5753 | `Apache-2.0` |
| `src/tad_dftd/damping/rational.py` | `src/tad_dftd/damping/rational.py` | 5811 | `Apache-2.0` |
| `src/tad_dftd/damping/screened.py` | `src/tad_dftd/damping/screened.py` | 3424 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/select.py` | `src/tad_dftd/damping/select.py` | 6076 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/z.py` | `src/tad_dftd/damping/z.py` | 1568 | `LGPL-3.0-or-later` |
| `src/tad_dftd/damping/zero.py` | `src/tad_dftd/damping/zero.py` | 6774 | `Apache-2.0` |
| `src/tad_dftd/data/__init__.py` | `src/tad_dftd/data/__init__.py` | 2097 | `Apache-2.0` |
| `src/tad_dftd/data/d1.py` | `src/tad_dftd/data/d1.py` | 3732 | `LGPL-3.0-or-later` |
| `src/tad_dftd/data/d2.py` | `src/tad_dftd/data/d2.py` | 4618 | `LGPL-3.0-or-later` |
| `src/tad_dftd/data/en.py` | `src/tad_dftd/data/en.py` | 2395 | `Apache-2.0` |
| `src/tad_dftd/data/hardness.py` | `src/tad_dftd/data/hardness.py` | 3074 | `Apache-2.0` |
| `src/tad_dftd/data/r4r2.py` | `src/tad_dftd/data/r4r2.py` | 3321 | `Apache-2.0` |
| `src/tad_dftd/data/radii.py` | `src/tad_dftd/data/radii.py` | 7341 | `Apache-2.0` |
| `src/tad_dftd/data/table.py` | `src/tad_dftd/data/table.py` | 5241 | `Apache-2.0` |
| `src/tad_dftd/data/tanh.py` | `src/tad_dftd/data/tanh.py` | 8828 | `LGPL-3.0-or-later` |
| `src/tad_dftd/data/wfpair-d3s.pt` | `src/tad_dftd/data/wfpair-d3s.pt` | 73790 | `N/A (binary tensor asset)` |
| `src/tad_dftd/data/wfpair-d4s.pt` | `src/tad_dftd/data/wfpair-d4s.pt` | 114878 | `N/A (binary tensor asset)` |
| `src/tad_dftd/data/wfpair.py` | `src/tad_dftd/data/wfpair.py` | 3227 | `LGPL-3.0-or-later` |
| `src/tad_dftd/data/zeff.py` | `src/tad_dftd/data/zeff.py` | 2287 | `Apache-2.0` |
| `src/tad_dftd/defaults.py` | `src/tad_dftd/defaults.py` | 3254 | `Apache-2.0` |
| `src/tad_dftd/disp.py` | `src/tad_dftd/disp.py` | 24485 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/exception.py` | `src/tad_dftd/exception.py` | 1276 | `Apache-2.0` |
| `src/tad_dftd/integrator/__init__.py` | `src/tad_dftd/integrator/__init__.py` | 1164 | `Apache-2.0` |
| `src/tad_dftd/integrator/gauss_legendre.py` | `src/tad_dftd/integrator/gauss_legendre.py` | 8247 | `LGPL-3.0-or-later` |
| `src/tad_dftd/integrator/trapezoid.py` | `src/tad_dftd/integrator/trapezoid.py` | 4591 | `LGPL-3.0-or-later` |
| `src/tad_dftd/model/__init__.py` | `src/tad_dftd/model/__init__.py` | 3048 | `Apache-2.0` |
| `src/tad_dftd/model/base.py` | `src/tad_dftd/model/base.py` | 2521 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/model/c6.py` | `src/tad_dftd/model/c6.py` | 11713 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/model/catalogue.py` | `src/tad_dftd/model/catalogue.py` | 13508 | `LGPL-3.0-or-later` |
| `src/tad_dftd/model/d1.py` | `src/tad_dftd/model/d1.py` | 10078 | `LGPL-3.0-or-later` |
| `src/tad_dftd/model/d2.py` | `src/tad_dftd/model/d2.py` | 9913 | `LGPL-3.0-or-later` |
| `src/tad_dftd/model/d3.py` | `src/tad_dftd/model/d3.py` | 25251 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/model/d4.py` | `src/tad_dftd/model/d4.py` | 47965 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/model/weights.py` | `src/tad_dftd/model/weights.py` | 22789 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/ncoord/__init__.py` | `src/tad_dftd/ncoord/__init__.py` | 3347 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/pairs.py` | `src/tad_dftd/pairs.py` | 8504 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/pairvalues.py` | `src/tad_dftd/pairvalues.py` | 7911 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/param/__init__.py` | `src/tad_dftd/param/__init__.py` | 952 | `Apache-2.0` |
| `src/tad_dftd/param/aliases.toml` | `src/tad_dftd/param/aliases.toml` | 2142 | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/d3_parameters.toml` | `src/tad_dftd/param/d3_parameters.toml` | 33181 | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/d4_parameters.toml` | `src/tad_dftd/param/d4_parameters.toml` | 23775 | `N/A (TOML parameter table)` |
| `src/tad_dftd/param/functional.py` | `src/tad_dftd/param/functional.py` | 11757 | `LGPL-3.0-or-later` |
| `src/tad_dftd/py.typed` | `src/tad_dftd/py.typed` | 0 | `N/A (PEP 561 marker)` |
| `src/tad_dftd/reference/__init__.py` | `src/tad_dftd/reference/__init__.py` | 951 | `Apache-2.0` |
| `src/tad_dftd/reference/d3-c6.pt` | `src/tad_dftd/reference/d3-c6.pt` | 4240619 | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d3.py` | `src/tad_dftd/reference/d3.py` | 15634 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/reference/d3s-legacy-c6.pt` | `src/tad_dftd/reference/d3s-legacy-c6.pt` | 129299 | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d4-ref.pt` | `src/tad_dftd/reference/d4-ref.pt` | 228965 | `N/A (binary tensor asset)` |
| `src/tad_dftd/reference/d4.py` | `src/tad_dftd/reference/d4.py` | 12490 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/reference/d4srev-ref.pt` | `src/tad_dftd/reference/d4srev-ref.pt` | 242869 | `N/A (binary tensor asset)` |
| `src/tad_dftd/threebody.py` | `src/tad_dftd/threebody.py` | 22366 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/twobody.py` | `src/tad_dftd/twobody.py` | 7087 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `src/tad_dftd/typing.py` | `src/tad_dftd/typing.py` | 2281 | `Apache-2.0` |
| `test/test_tad_dftd/__init__.py` | `test/__init__.py` | 63 | `Apache-2.0` |
| `test/test_tad_dftd/test_ase.py` | `test/test_ase.py` | 91060 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `test/test_tad_dftd/test_ase_isolation.py` | `test/test_ase_isolation.py` | 10756 | `Apache-2.0 AND LGPL-3.0-or-later` |
| `test/test_tad_dftd/test_catalogue_and_param.py` | `test/test_catalogue_and_param.py` | 30878 | `Apache-2.0` |
| `test/test_tad_dftd/test_cutoff_and_pairs.py` | `test/test_cutoff_and_pairs.py` | 5197 | `Apache-2.0` |
| `test/test_tad_dftd/test_d1_and_d2.py` | `test/test_d1_and_d2.py` | 14873 | `Apache-2.0` |
| `test/test_tad_dftd/test_d3_and_d3s.py` | `test/test_d3_and_d3s.py` | 26465 | `Apache-2.0` |
| `test/test_tad_dftd/test_d4.py` | `test/test_d4.py` | 39612 | `Apache-2.0` |
| `test/test_tad_dftd/test_damping.py` | `test/test_damping.py` | 10918 | `Apache-2.0` |
| `test/test_tad_dftd/test_data.py` | `test/test_data.py` | 4629 | `Apache-2.0` |
| `test/test_tad_dftd/test_integrator.py` | `test/test_integrator.py` | 4153 | `Apache-2.0` |
| `test/test_tad_dftd/test_isolation.py` | `test/test_isolation.py` | 7596 | `Apache-2.0` |
| `test/test_tad_dftd/test_ncoord.py` | `test/test_ncoord.py` | 2263 | `Apache-2.0` |
| `test/test_tad_dftd/test_periodic_and_transforms.py` | `test/test_periodic_and_transforms.py` | 72701 | `Apache-2.0` |
| `docs/tad_dftd/architecture-and-mapping.md` | `docs/architecture-and-mapping.md` | 22632 | `N/A (markdown documentation)` |
| `docs/tad_dftd/ase.md` | `docs/ase.md` | 12859 | `N/A (markdown documentation)` |
| `docs/tad_dftd/standalone-extraction.md` | `docs/standalone-extraction.md` | 3664 | `N/A (markdown documentation)` |
| `docs/tad_dftd/upstream_multicharge_periodic_eeq_issue.md` | `docs/upstream_multicharge_periodic_eeq_issue.md` | 9543 | `N/A (markdown documentation)` |
| `tools/generate_tad_dftd_data.py` | `tools/generate_tad_dftd_data.py` | 20309 | `Apache-2.0` |

### SPDX Header Audit Summary (72 `.py` files)

- **`Apache-2.0`**: 37 files (`24` in `src/tad_dftd/`, `12` in `test/test_tad_dftd/`, `1` in `tools/`).
- **`Apache-2.0 AND LGPL-3.0-or-later`**: 17 files (`15` in `src/tad_dftd/`, `2` in `test/test_tad_dftd/`).
- **`LGPL-3.0-or-later`**: 18 files (`18` in `src/tad_dftd/`, directly translated from `grimme-lab/dftd` LGPL-3.0-or-later Fortran sources as documented in `src/tad_dftd/PROVENANCE.md`).
- **Missing SPDX headers**: `0` files.
- **Combined package license expression**: `Apache-2.0 AND LGPL-3.0-or-later`.

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
| `typing-extensions` | unpinned | `TypeAlias`, `Self`, `Literal`, ` overload` compatibility |
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
license = Apache-2.0 AND LGPL-3.0-or-later
license_files =
    LICENSE
    COPYING
    COPYING.LESSER
    PROVENANCE.md
platforms = any
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Science/Research
    License :: OSI Approved :: Apache Software License
    License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
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
    COPYING
    COPYING.LESSER
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
   - Copy the 90 files from the manifest in Section 2 into their target paths.
   - Copy `src/tad_dftd/{LICENSE,COPYING,COPYING.LESSER,PROVENANCE.md}` to the repository root as well so both wheel distributions (`src/tad_dftd/`) and sdist/GitHub repository inspection expose the full Apache-2.0, GPL-3.0, LGPL-3.0, and provenance texts.
   - Add `pyproject.toml`, `setup.cfg` (Section 3.2), `README.md`, and CI workflows testing Python `3.10`–`3.14` and PyTorch `2.7`–`2.14` (both with and without the `[ase]` extra).
2. **Validate the standalone package prior to tagging**:
   - Build sdist and wheel (`python -m build` or `pip wheel --no-deps --no-build-isolation .`).
   - Verify all 6 `.pt` files (`data/wfpair-d3s.pt`, `data/wfpair-d4s.pt`, `reference/d3-c6.pt`, `reference/d3s-legacy-c6.pt`, `reference/d4-ref.pt`, `reference/d4srev-ref.pt`), all 3 `.toml` files (`param/aliases.toml`, `param/d3_parameters.toml`, `param/d4_parameters.toml`), `py.typed`, and license/provenance files are present inside the wheel archive.
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
   - Remove the `tad_dftd = ...` block (`lines 102–110`) from `[options.package_data]`.
3. **Remove the in-tree `tad_dftd` files from `dxtb`**:
   - Delete `src/tad_dftd/` (`70` files).
   - Delete `test/test_tad_dftd/` (`15` files).
   - Delete `tools/generate_tad_dftd_data.py` (`1` file).
   - Delete `docs/tad_dftd/` (`4` files).
4. **Verify cutover gate in `dxtb`**:
   - Install `tad-dftd>=0.1.0` into the 5 supported environments (`py3.10-torch2.7.0` through `py3.14-torch2.14.1`).
   - Verify `python -c "import tad_dftd, dxtb; assert 'src/tad_dftd' not in tad_dftd.__file__"` confirms `tad_dftd` resolves from `site-packages`.
   - Run `pytest test/test_classical/test_dispersion test/test_interaction test/test_calculator` and static analysis (`mypy`, `isort --check`, `black --check`).
