# `tad_dftd` Standalone Repository Extraction Guide (`H2.8`)

`src/tad_dftd` is architected as a self-contained top-level Python package inside `marvinfriede/dxtb` with zero imports from `dxtb`, `tad_dftd3`, or `tad_dftd4`. Extracting it into a standalone `tad-dftd` repository requires only file movement and packaging metadata updates—no algorithmic or API changes.

## 1. Standalone Readiness Verification Checklist

- [x] **No `dxtb` imports in `src/tad_dftd/`**: Verified via AST inspection across all `.py` files in `src/tad_dftd/`.
- [x] **No `dxtb` imports in `test/test_tad_dftd/`**: All `tad_dftd` unit, parity, periodic, transform, and `torch.compile` tests import only `tad_dftd`, `tad_mctc`, `tad_multicharge`, `torch`, `pytest`, and standard library modules.
- [x] **Self-contained runtime resource loading**: Binary reference/data tables (`src/tad_dftd/reference/*.pt`) and TOML functional parameter tables (`src/tad_dftd/param/*.toml`) are loaded via `importlib.resources.files` relative to `tad_dftd` subpackages and work identically in editable installs and isolated wheel installations.
- [x] **Self-contained version handling**: `src/tad_dftd/__version__.py` defines `__version__ = "0.1.0"` without referencing `dxtb.__version__`.
- [x] **Self-contained license & provenance files**: `src/tad_dftd/LICENSE` (Apache-2.0), `src/tad_dftd/COPYING` (GPL-3.0), `src/tad_dftd/COPYING.LESSER` (LGPL-3.0), `src/tad_dftd/PROVENANCE.md`, and `src/tad_dftd/py.typed` are packaged inside `src/tad_dftd/`.

## 2. Files to Move to the New `tad-dftd` Repository

1. **Package source tree**:
   - Move `src/tad_dftd/` -> `src/tad_dftd/` in the new `tad-dftd` repository.
   - Includes packaged binary reference tables (`src/tad_dftd/reference/*.pt`), functional parameter tables (`src/tad_dftd/param/*.toml`), `py.typed`, `LICENSE`, `COPYING`, `COPYING.LESSER`, and `PROVENANCE.md`.

2. **Test suite**:
   - Move `test/test_tad_dftd/` -> `test/` in the new `tad-dftd` repository.

3. **Data generation & documentation**:
   - Move `tools/generate_tad_dftd_data.py` -> `tools/generate_tad_dftd_data.py`.
   - Move `docs/tad_dftd/` -> `docs/`.

## 3. Standalone Packaging Metadata (`pyproject.toml` / `setup.cfg`)

In the new `tad-dftd` repository, configure `setup.cfg` / `pyproject.toml` with:

```ini
[metadata]
name = tad-dftd
version = attr: tad_dftd.__version__.__version__
description = Unified Differentiable DFT-D Dispersion Corrections in PyTorch
license = Apache-2.0 AND LGPL-3.0-or-later

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
    param/*.toml
    reference/*.pt
```

## 4. Imports to Keep Unchanged

All imports in `src/tad_dftd/`, `test/test_tad_dftd/`, `src/dxtb/`, and `test/test_classical/test_dispersion/` already use top-level `tad_dftd` package paths (`import tad_dftd`, `from tad_dftd import ...`). No import paths need to be rewritten in either repository during extraction.

## 5. Changes Required in `dxtb` Upon Extraction

1. Remove `src/tad_dftd/`, `test/test_tad_dftd/`, `tools/generate_tad_dftd_data.py`, and `docs/tad_dftd/` from `dxtb`.
2. In `dxtb/setup.cfg`:
   - Add `tad-dftd>=0.1.0` to `install_requires`.
   - Remove the `tad_dftd = ...` entry from `[options.package_data]`.
3. Run `pytest test/test_classical/test_dispersion test/test_interaction` against the installed `tad-dftd` package.
