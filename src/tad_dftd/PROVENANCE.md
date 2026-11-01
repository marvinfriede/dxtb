# Provenance and Licensing of `tad_dftd`

## Overview

`tad_dftd` is a unified, differentiable PyTorch dispersion correction library implementing the Grimme-lab DFT-D family of dispersion models (`D1`, `D2`, `D3`, `D3S`, `D4`, `D4S`, `D4Srev`), the classical `EEQ` (`eeq2019`) charge model, damping functions, Casimir-Polder integrators, and periodic/sparse neighbor-list evaluations.

> **Scope Note (`D5` and `EEQBC`)**: Both the `D5` dispersion model (`model="d5"`) and the `EEQBC` charge model (`qmodel="eeqbc"` / `"eeqbc2025"`) are intentionally excluded from this release because their upstream reference models are not yet finalized. Requesting `model="d5"` or `qmodel="eeqbc"` / `"eeqbc2025"` raises `NotImplementedError`.

Currently developed as an independent top-level Python package (`src/tad_dftd/`) inside `marvinfriede/dxtb` for subsequent standalone extraction to `tad-dftd`.

## Upstream Sources and Recorded Revisions

1. **Authoritative Fortran Reference (`grimme-lab/dftd`)**
   - Repository: `https://github.com/grimme-lab/dftd`
   - Local checkout: `/tmp1/friede/work/dftd-grimme-lab/dftd`
   - Commit SHA: `d2740777049b2a77fd5acc86c4e1abe75d5374ee`
   - License: `LGPL-3.0-or-later` (see `COPYING` and `COPYING.LESSER` in this directory)
   - Subprojects referenced:
     - `multicharge` (`subprojects/multicharge`): `LGPL-3.0-or-later`
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

## File-by-File Provenance and License Classification

| `tad_dftd` Module / Asset | Upstream Source(s) | Effective License | Notes |
|---|---|---|---|
| `tad_dftd/cutoff.py`, `tad_dftd/pairs.py` | `tad-dftd3/src/tad_dftd3/{cutoff,pairs}.py`, `dftd/src/dftd/cutoff.f90` | `Apache-2.0 AND LGPL-3.0-or-later` | Real-space cutoffs, quintic smooth switch, `DensePairs`, `SparsePairs` |
| `tad_dftd/data/{en,hardness,r4r2,radii,zeff,table}.py` | `tad-dftd3`, `tad-dftd4`, `dftd/src/dftd/data/{en,hardness,r4r2,zeff}.f90` | `Apache-2.0` | Atomic and pairwise element constants ($Z=1..118$) |
| `tad_dftd/data/{d1,d2,tanh,wfpair}.py` | `dftd/src/dftd/data/{d1,d2,tanh}.f90`, `dftd/src/dftd/data/wfpair/{d3s,d4s}.f90` | `LGPL-3.0-or-later` | Direct translations of `grimme-lab/dftd` element/pair data tables |
| `tad_dftd/integrator/{gauss_legendre,trapezoid}.py` | `dftd/src/dftd/integrator/{gauss_legendre,trapezoid}.f90`, `tad-dftd4` | `LGPL-3.0-or-later` | 10-point Gauss-Legendre and 23-point composite trapezoid Casimir-Polder integrators |
| `tad_dftd/damping/{base,param,rational,zero,atm}.py` | `tad-dftd3/src/tad_dftd3/damping/*.py`, `tad-dftd4` | `Apache-2.0` | Core damping base classes, `DampingParam`, rational (BJ) and zero dampings |
| `tad_dftd/damping/{screened,mzero,optimizedpower,cso,koide,fermi,z,select}.py` | `dftd/src/dftd/damping/*.f90` | `LGPL-3.0-or-later` | Direct translations of `grimme-lab/dftd` 2B/3B damping families and damping radii |
| `tad_dftd/ncoord/__init__.py` | `tad-mctc/src/tad_mctc/ncoord/*`, `dftd/src/dftd/model/*` | `Apache-2.0 AND LGPL-3.0-or-later` | Coordination number configurations for D3 (`cn_d3`), D4 (`cn_d4`), and D4Srev (`cn_d4srev`) |
| `tad_dftd/charge/{__init__,eeq}.py` | `tad-multicharge`, `tad-dftd4` | `Apache-2.0` | Classical `EEQ` (`eeq2019`) charge model wrapper and charge model dispatch (`eeqbc` raises `NotImplementedError`) |
| `tad_dftd/reference/{d3,d4}.py`, `tad_dftd/reference/*.pt` | `dftd/src/dftd/model/{d3,d4}/reference/*`, `tad-dftd3`, `tad-dftd4` | `Apache-2.0 AND LGPL-3.0-or-later` | Packaged reference tables for D3 (`d3-c6.pt`), D3S (`d3s-legacy-c6.pt`), D4/D4S (`d4-ref.pt`), and D4Srev (`d4srev-ref.pt`) generated via `tools/generate_tad_dftd_data.py` |
| `tad_dftd/model/{catalogue,d1,d2}.py` | `dftd/src/dftd/model/{catalogue.f90,d1/orig.f90,d2/orig.f90}` | `LGPL-3.0-or-later` | Model catalogue registry and `D1Model` / `D2Model` implementations |
| `tad_dftd/model/{base,c6,weights,d3,d4}.py` | `tad-dftd3`, `tad-dftd4`, `dftd/src/dftd/model/{d3,d4}/*` | `Apache-2.0 AND LGPL-3.0-or-later` | PyTorch dispersion models (`D3Model`, `D3SModel`, `D4Model`, `D4SModel`, `D4SrevModel`) and `D4SC` matrix/energy/potential kernels |
| `tad_dftd/param/functional.py`, `tad_dftd/param/*.toml` | `tad-dftd3` / `s-dftd3` (`d3_parameters.toml`), `dftd/src/dftd/param/{d1,d2,d3,d4}.f90`, `dftd/assets/parameters.toml` | `Apache-2.0 AND LGPL-3.0-or-later` | Functional damping parameter lookup tables (`aliases.toml`, `d3_parameters.toml`, `d4_parameters.toml`) |
| `tad_dftd/{disp,twobody,threebody,pairvalues}.py` | `tad-dftd3/src/tad_dftd3/{disp,twobody,threebody,pairvalues}.py`, `dftd/src/dftd/disp.f90` | `Apache-2.0 AND LGPL-3.0-or-later` | High-level and term-level evaluation entrypoints |

## Licensing Policy

- Code derived purely from `tad-dftd3`, `tad-dftd4`, `tad-mctc`, and `tad-multicharge` is licensed under `Apache-2.0` (`LICENSE`).
- Numerical reference tables (`reference/*.pt`), functional parameter tables (`param/*.toml`), and algorithms/modules translated directly from `grimme-lab/dftd` are subject to `LGPL-3.0-or-later` (`COPYING.LESSER` and `COPYING`).
- Consequently, the combined `tad_dftd` package distribution carries the composite license expression `Apache-2.0 AND LGPL-3.0-or-later` (and under LGPLv3 terms, the combined work may be redistributed under `LGPL-3.0-or-later` while preserving `Apache-2.0` notices on the `tad-*`-derived portions).
- Distributing `tad_dftd` as a separate top-level package inside `src/tad_dftd/` keeps `tad_dftd`'s licensing and attribution self-contained for standalone extraction.

## Periodic Fortran Reference Fixtures (`_FORTRAN_PERIODIC_REFS`, `H3-F.4`)

The 21 periodic reference fixtures in `test/test_tad_dftd/test_periodic_and_transforms.py` (`1d`, `2d`, `3d` × `d1`, `d2`, `d3`, `d3s`, `d4`, `d4s`, `d4srev`) can be regenerated or audited directly against the pinned Fortran shared library (`/tmp1/friede/work/dftd-grimme-lab/dftd/_build/libdftd.so`, commit `d2740777049b2a77fd5acc86c4e1abe75d5374ee`) using:

```bash
PYTHONPATH=src /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/python tools/generate_tad_dftd_data.py --verify-periodic-fixtures
```

- **Units & Precision**: Atomic units (`Bohr`, `Eh`, `Eh/Bohr`), IEEE-754 `float64` (17 significant decimal digits).
- **Geometry & Cell**: `natoms = 2`, `numbers = [6, 6]`, `positions = [[0.15, 0.25, 0.35], [1.95, 2.15, 2.35]]` Bohr, row-vector lattice `L = [[6.2, 0.3, 0.1], [0.3, 6.4, 0.2], [0.1, 0.2, 6.6]]` Bohr, `periodic = [True, False, False]` (`1d`), `[True, True, False]` (`2d`), `[True, True, True]` (`3d`).
- **Coordinate Wrapping**: `dftd_new_structure` (`new_structure_api` in `src/dftd/api.f90`) wraps periodic coordinates into $[0, 1)^3$ via `wrap_to_central_cell`; fractional coordinates already lie in $(0.01, 0.35) \subset [0, 1)$.
- **Charge Model & Partial Charges**: `charge = 0.0`, `qmodel = 1` (EEQ, `ga=3.0, gc=2.0, wf=6.0`); by cell inversion symmetry between the two identical carbon atoms, $q = [0.0, 0.0]$ identically.
- **Damping Parameters, ATM, & Cutoffs (`dftd_set_model_realspace_cutoff`, `s9 = 1.0` for all 7 models)**:
  - `d1`: Fermi 2B (`damping_2b_id=8`) + `zero_avg` 3B (`damping_3b_id=4`), `s6=0.75, rs6=1.0, alp=23.0, s9=1.0, rs9=1.0, alp3=16.0, a1=0.4, a2=4.0, a13=0.4, a23=4.0`, `disp2=14.0, disp3=10.0, cn=40.0` Bohr.
  - `d2`: Fermi 2B (`damping_2b_id=8`) + `zero_avg` 3B (`damping_3b_id=4`), `s6=0.75, rs6=1.0, alp=20.0, s9=1.0, rs9=1.0, alp3=16.0, a1=0.4, a2=4.0, a13=0.4, a23=4.0`, `disp2=14.0, disp3=10.0, cn=40.0` Bohr.
  - `d3`, `d3s`: Rational (BJ) 2B (`damping_2b_id=1`) + `zero_avg` 3B (`damping_3b_id=4`), PBE parameters (`s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, rs9=4/3, alp=14.0, alp3=14.0, a13=1.0, a23=0.0`), `disp2=14.0, disp3=10.0, cn=40.0` Bohr.
  - `d4`, `d4s`, `d4srev`: Rational (BJ) 2B (`damping_2b_id=1`) + `zero_avg` 3B (`damping_3b_id=4`), PBE parameters (`s6=1.0, s8=0.95948085, a1=0.38574991, a2=4.80688534, s9=1.0, rs9=1.0, alp=16.0, alp3=16.0, a13=0.38574991, a23=4.80688534`), `disp2=14.0, disp3=10.0, cn=30.0` Bohr (`cn_eeq=25.0` Bohr).
- **Strain Tensor Definition**: $\sigma_{\alpha\beta} = \left.\partial E / \partial \varepsilon_{\alpha\beta}\right|_{\mathbf{s},\,\varepsilon=0} = \sum_i r_{i\alpha}\partial E/\partial r_{i\beta} + \sum_{k=1}^3 L_{k\alpha}\partial E/\partial L_{k\beta}$.

