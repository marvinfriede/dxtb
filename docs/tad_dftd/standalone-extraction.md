# `tad_dftd` Standalone Repository Extraction Guide

`src/tad_dftd` is architected as a self-contained top-level Python package inside `marvinfriede/dxtb` with zero imports from `dxtb`, `tad_dftd3`, or `tad_dftd4`. Extracting it into a standalone `tad-dftd` repository requires only file movement and packaging metadata updates—no algorithmic or API changes.

## 1. Files to Move to the New `tad-dftd` Repository

1. **Package source tree**:
   - Move `src/tad_dftd/` -> `src/tad_dftd/` in the new `tad-dftd` repository.
   - Includes packaged binary atomic/reference tables (`src/tad_dftd/data/*.pt`, `src/tad_dftd/reference/*.pt`), functional parameter tables (`src/tad_dftd/param/*.toml`), `py.typed`, `LICENSE`, `COPYING`, `COPYING.LESSER`, and `PROVENANCE.md`.

2. **Test suite**:
   - Move `test/test_tad_dftd/` -> `test/` in the new `tad-dftd` repository.

3. **Data generation & documentation**:
   - Move `tools/generate_tad_dftd_data.py` -> `tools/generate_tad_dftd_data.py`.
   - Move `docs/tad_dftd/` -> `docs/`.

## 2. Standalone Packaging Metadata (`pyproject.toml` / `setup.cfg`)

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
python_requires = >=3.10
install_requires =
    tad-mctc>=0.11.3
    tad-multicharge>=0.8.0
    tomli; python_version < "3.11"
    torch>=2.6.0,<3
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
```

## 3. Changes Required in `dxtb` Upon Extraction

1. Remove `src/tad_dftd/`, `test/test_tad_dftd/`, and `tools/generate_tad_dftd_data.py` from `dxtb`.
2. In `dxtb/setup.cfg`:
   - Add `tad-dftd>=0.1.0` to `install_requires`.
   - Remove the `tad_dftd = ...` entry from `[options.package_data]`.
3. No import statements or numerical adapters in `src/dxtb/` or `test/` need to change, because `dxtb` already imports `tad_dftd` as a top-level package (`import tad_dftd`).
