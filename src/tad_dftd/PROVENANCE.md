# Provenance and Licensing of `tad_dftd`

## Overview

`tad_dftd` is a unified, differentiable PyTorch dispersion correction library implementing the Grimme-lab DFT-D family of dispersion models (`D1`, `D2`, `D3`, `D3S`, `D4`, `D4S`, `D4Srev`, `D5`) and associated charge models (`EEQ`, `EEQBC`), damping functions, Casimir-Polder integrators, and periodic/sparse neighbor-list evaluations.

Currently developed as an independent top-level Python package (`src/tad_dftd/`) inside `marvinfriede/dxtb` for subsequent standalone extraction to `tad-dftd`.

## Upstream Sources and Recorded Revisions

1. **Authoritative Fortran Reference (`grimme-lab/dftd`)**
   - Repository: `https://github.com/grimme-lab/dftd`
   - Local checkout: `/tmp1/friede/work/dftd-grimme-lab/dftd`
   - Commit SHA: `d2740777049b2a77fd5acc86c4e1abe75d5374ee`
   - License: `LGPL-3.0-or-later` (see `COPYING` and `COPYING.LESSER` in this directory)
   - Subprojects referenced:
     - `multicharge` (`subprojects/multicharge`): `LGPL-3.0-or-later` (for `eeqbc2025` model equations and parameters in `src/multicharge/model/eeqbc.f90`, `src/multicharge/param.f90`, and `src/multicharge/param/eeqbc2025.f90`)
     - `mctc-lib` (`subprojects/mctc-lib`): `Apache-2.0`

2. **Existing PyTorch Dispersion Libraries**
   - `tad-dftd3` (modified local checkout with `tad-mctc` neighbor-list and periodic support):
     - Local checkout: `/tmp1/friede/dxtb-agentic/tad-dftd3`
     - Commit SHA: `954c0f285e25939b7003e62fdcdc47a2a4be971b` (branch `neighborlist`)
     - License: `Apache-2.0` (see `LICENSE` in this directory)
   - `tad-dftd4`:
     - Version: `0.8.0` (`https://github.com/dftd4/tad-dftd4`)
     - License: `Apache-2.0`
   - `tad-mctc`:
     - Version: `0.11.3` (`https://github.com/tad-mctc/tad-mctc`)
     - License: `Apache-2.0`
   - `tad-multicharge`:
     - Version: `0.8.0` (`https://github.com/tad-mctc/tad-multicharge`)
     - License: `Apache-2.0`

## File-by-File Provenance Mapping

| `tad_dftd` Module / Asset | Upstream Source(s) | Upstream License | Notes |
|---|---|---|---|
| `tad_dftd/cutoff.py`, `tad_dftd/pairs.py` | `tad-dftd3/src/tad_dftd3/{cutoff,pairs}.py`, `dftd/src/dftd/cutoff.f90` | `Apache-2.0` / `LGPL-3.0-or-later` | Real-space cutoffs, quintic smooth switch, `DensePairs`, `SparsePairs` |
| `tad_dftd/data/*` | `dftd/src/dftd/data/{d1,d2,en,hardness,r4r2,tanh,zeff}.f90`, `dftd/src/dftd/data/wfpair/{d3s,d4s}.f90` | `LGPL-3.0-or-later` / `Apache-2.0` | Atomic and pairwise element constants ($Z=1..118$) |
| `tad_dftd/integrator/*` | `dftd/src/dftd/integrator/{gauss_legendre,trapezoid}.f90` | `LGPL-3.0-or-later` | 10-point Gauss-Legendre and 23-point composite trapezoid Casimir-Polder integrators |
| `tad_dftd/damping/*` | `tad-dftd3/src/tad_dftd3/damping/*.py`, `dftd/src/dftd/damping/*.f90` | `Apache-2.0` / `LGPL-3.0-or-later` | Unified 2B (`rational`, `screened`, `zero`, `mzero`, `optpower`/`optimizedpower`, `cso`, `koide`, `fermi`, `z`) and 3B (`zero_avg`/`zero_d3`/`zero_d4`, `zero`/`zero_product`, `rational`, `screened`, `koide`) dampings and damping radii |
| `tad_dftd/ncoord/*` | `tad-mctc/src/tad_mctc/ncoord/*`, `dftd/src/dftd/model/*` | `Apache-2.0` / `LGPL-3.0-or-later` | Coordination number configurations for D3, D4, D4Srev, D5, and EEQBC |
| `tad_dftd/charge/*` | `tad-multicharge`, `dftd/subprojects/multicharge/src/multicharge/{param.f90,model/eeqbc.f90,param/eeqbc2025.f90}` | `Apache-2.0` / `LGPL-3.0-or-later` | Differentiable `EEQ` (`eeq2019`) and `EEQBC` (`eeqbc2025`) charge solvers (0D and 3D periodic) |
| `tad_dftd/reference/*` | `dftd/src/dftd/model/{d1,d2,d3,d4,d5}/reference/*`, `tad-dftd3`, `tad-dftd4` | `LGPL-3.0-or-later` / `Apache-2.0` | Packaged reference tables for D1, D2, D3, D3S, D4, D4S, D4Srev, and D5 generated via `tools/generate_tad_dftd_data.py` |
| `tad_dftd/model/*` | `dftd/src/dftd/model/{catalogue,type,fixed,utils}.f90`, `dftd/src/dftd/model/{d1,d2,d3,d4,d5}/*`, `tad-dftd3`, `tad-dftd4` | `LGPL-3.0-or-later` / `Apache-2.0` | PyTorch dispersion models (`D1Model`, `D2Model`, `D3Model`, `D3SModel`, `D4Model`, `D4SModel`, `D4SrevModel`, `D5Model`) |
| `tad_dftd/param/*` | `dftd/src/dftd/param/{d1,d2,d3,d4,d5}.f90`, `dftd/assets/parameters.toml` | `LGPL-3.0-or-later` | Functional damping parameter lookup tables |
| `tad_dftd/{disp,twobody,threebody,pairvalues}.py` | `tad-dftd3/src/tad_dftd3/{disp,twobody,threebody,pairvalues}.py`, `dftd/src/dftd/disp.f90` | `Apache-2.0` / `LGPL-3.0-or-later` | High-level and term-level evaluation entrypoints |

## Licensing Notes

- Original PyTorch implementations derived from `tad-dftd3`, `tad-dftd4`, and `tad-mctc` are licensed under `Apache-2.0` (`LICENSE`).
- Numerical reference tables, parameter tables, and algorithms ported from `grimme-lab/dftd` and `grimme-lab/multicharge` are subject to `LGPL-3.0-or-later` (`COPYING.LESSER` and `COPYING`).
- Distributing `tad_dftd` as a separate package inside `src/tad_dftd/` preserves `dxtb`'s root `Apache-2.0` license while keeping `tad_dftd`'s licensing and attribution self-contained for standalone extraction.
