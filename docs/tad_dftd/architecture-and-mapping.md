# `tad_dftd` Architecture, Model Catalogue, and Fortran-to-PyTorch Mapping

## 1. Recorded Source Authority & Git Revisions

- **Main integration repository**: `marvinfriede/dxtb` (`/tmp1/friede/dxtb-agentic/dxtb`)
- **Base branch**: `origin/claude/tender-dijkstra-cda42s`
- **Recorded base commit SHA**: `abdb4562a0607475e775072277b63f82501363a0` (merged onto target `fba253dd766566507cd6db2797b6926aa35c95c2` at `f51776e23348df71bebf00fd268bdc6615d1c934`)
- **Merged target commit SHA (`M1`)**: `f51776e23348df71bebf00fd268bdc6615d1c934` on `claude/tender-dijkstra-cda42s`
- **Authoritative Fortran reference (`grimme-lab/dftd`)**: `/tmp1/friede/work/dftd-grimme-lab/dftd` at commit SHA `d2740777049b2a77fd5acc86c4e1abe75d5374ee` (branch `split-c-api-tests`)
- **Modified PyTorch `tad-dftd3` reference**: `/tmp1/friede/dxtb-agentic/tad-dftd3` at commit SHA `954c0f2237c348063e0b8190cb2cb108a49cc3d9` (branch `neighborlist`)

## 2. Package Architecture (`src/tad_dftd/`)

`tad_dftd` is an independent top-level Python package inside `src/tad_dftd/` that depends only on `torch` (`>=2.7.0,<3`), `tad-mctc` (`>=0.11.3`), `tad-multicharge` (`>=0.8.0`), `tomli` (on Python < 3.11), `typing-extensions`, standard library modules, and optionally `ase` (`>=3.22.0`) + `numpy` when importing `tad_dftd.ase`. It never imports `dxtb`, `tad_dftd3`, or `tad_dftd4`.

```text
src/tad_dftd/
├── __init__.py
├── __version__.py
├── py.typed
├── LICENSE
├── COPYING
├── COPYING.LESSER
├── PROVENANCE.md
├── ase.py
├── typing.py
├── defaults.py
├── exception.py
├── cutoff.py
├── pairs.py
├── disp.py
├── twobody.py
├── threebody.py
├── pairvalues.py
├── data/
│   ├── __init__.py
│   ├── d1.py
│   ├── d2.py
│   ├── en.py
│   ├── hardness.py
│   ├── r4r2.py
│   ├── radii.py
│   ├── table.py
│   ├── tanh.py
│   ├── wfpair.py
│   ├── wfpair-d3s.pt
│   ├── wfpair-d4s.pt
│   └── zeff.py
├── integrator/
│   ├── __init__.py
│   ├── gauss_legendre.py
│   └── trapezoid.py
├── ncoord/
│   └── __init__.py
├── charge/
│   ├── __init__.py
│   └── eeq.py
├── damping/
│   ├── __init__.py
│   ├── atm.py
│   ├── base.py
│   ├── param.py
│   ├── rational.py
│   ├── screened.py
│   ├── zero.py
│   ├── mzero.py
│   ├── optimizedpower.py
│   ├── cso.py
│   ├── fermi.py
│   ├── koide.py
│   ├── z.py
│   └── select.py
├── reference/
│   ├── __init__.py
│   ├── d3.py
│   ├── d4.py
│   ├── d3-c6.pt
│   ├── d3s-legacy-c6.pt
│   ├── d4-ref.pt
│   └── d4srev-ref.pt
├── model/
│   ├── __init__.py
│   ├── base.py
│   ├── c6.py
│   ├── catalogue.py
│   ├── weights.py
│   ├── d1.py
│   ├── d2.py
│   ├── d3.py
│   └── d4.py
└── param/
    ├── __init__.py
    ├── functional.py
    ├── aliases.toml
    ├── d3_parameters.toml
    └── d4_parameters.toml
```

## 3. Model Catalogue & Element Coverage (`H2.2`)

### 3.1 Model Overview

| Model Key (`DispersionModelId`) | Implemented Status | Default CN & Cutoff (Bohr) | Charge Dependency (`qmodel`) | Default 2B / 3B Damping & Radii | Default 2B / 3B Cutoff (Bohr) |
|---|---|---|---|---|---|
| `d1` (`10`) | Implemented (`D1Model`) | None (fixed $C_6$) | None | 2B: `fermi` (`d1` radii); 3B: `zero_d4` (`r4r2` radii, expert option) | `disp2 = 90.0`, `disp3 = 40.0` |
| `d2` (`20`) | Implemented (`D2Model`) | None (fixed $C_6$) | None | 2B: `fermi` (`d2` radii); 3B: `zero_d4` (`r4r2` radii, expert option) | `disp2 = 90.0`, `disp3 = 40.0` |
| `d3` (`30`) | Implemented (`D3Model`) | `cn_d3` (`exp_count`, `kcn=16.0`, `cn = 40.0`) | None | 2B: `rational` (`r4r2`), `zero`, `mzero`, `optpower`, `cso`, `screened`, `koide`, `z`; 3B: `zero_d3` (`vdw` radii) | `disp2 = 60.0`, `disp3 = 40.0` |
| `d3s` (`31`) | Implemented (`D3SModel`) | `cn_d3` (`exp_count`, `kcn=16.0`, `cn = 40.0`) | None | Same as `d3` (excluding `z`) with pairwise CN Gaussian width $w_f(A,B)$ | `disp2 = 60.0`, `disp3 = 40.0` |
| `d4` (`40`) | Implemented (`D4Model`) | `cn_d4` (EN-weighted `erf_count`, `kcn=7.5`, `cn = 30.0`, `cn_eeq = 25.0`) | `eeq` (`eeq2019`, default) or external `q` (`gfn2`); `eeqbc` / `eeqbc2025` raises `NotImplementedError` | 2B: `rational` (`r4r2`), `screened`, `zero`, `mzero`, `optpower`, `koide`; 3B: `zero_d4` (`r4r2` radii, evaluated at $q = 0$) | `disp2 = 60.0`, `disp3 = 40.0` |
| `d4s` (`41`) | Implemented (`D4SModel`) | `cn_d4` (`cn = 30.0`, `cn_eeq = 25.0`) | Same as `d4` (`eeq`, external `q`; `eeqbc` raises `NotImplementedError`) | Same as `d4`, with pairwise CN Gaussian widths $w_f(A,B)$ instead of $1/\text{ngw}$ | `disp2 = 60.0`, `disp3 = 40.0` |
| `d4srev` (`42`) | Implemented (`D4SrevModel`) | `cn_d4srev` (unweighted `erf_count`, `kcn=7.5`, `cn = 30.0`, `cn_eeq = 25.0`) | `eeq` (`eeq2019`, default) or external `q` (`gfn2`, `gxtb`); `eeqbc` raises `NotImplementedError` | 2B: `rational`, `screened`, `zero`, `mzero`, `optpower`, `koide`; 3B: `zero_d4`, `zero_product`, `rational`, `screened`, `koide` (evaluated at $q = 0$) | `disp2 = 60.0`, `disp3 = 40.0` |
| `d5` (`50`) | **Excluded / Not Implemented** | Raises `NotImplementedError` | Raises `NotImplementedError` | Intentionally excluded because upstream `D5` and `EEQBC` are not finalized in `dftd` | Raises `NotImplementedError` |

### 3.2 Detailed Element Coverage by Table and Charge Mode

All models accept `Z == 0` as a padding/dummy atom (zero energy and zero nuclear force) and raise `ValueError` for negative atomic numbers ($Z < 0$) or unsupported positive atomic numbers in eager and `torch.func` (`vmap`, `jacrev`, `jacfwd`, `grad`) execution.

Under `torch.compile` (`eager`, `aot_eager`, and `inductor` with `fullgraph=True`), every model entrypoint (`__call__`, `environment`, `weights`, `coefficients`, `get_properties`, `get_polarizabilities`, `get_atomic_c6`, `d4sc_dispersion_matrix`, and `get_charges`) applies `guard_numbers(numbers, invalid, ...)`: invalid atomic numbers (including negative $Z < 0$, $Z$ above the model's maximum element, $Z \in \{2..5\}$ in `D1Model`, $Z \in \{104..111\}$ in `D4Model`/`D4SModel`, and $Z \in \{112..118\}$ in `D4Model`/`D4SModel` when `q=None`) are mapped to a graph-connected out-of-bounds table lookup without introducing graph breaks or host-device scalar syncs, deterministically raising an out-of-bounds exception (`IndexError` in `eager`/`aot_eager`, `RuntimeError: index out of bounds` in `inductor`) at runtime instead of silently returning meaningless finite energies. Callers who require a descriptive `ValueError` before entering a compiled region can call `tad_dftd.validate_numbers(numbers, model=..., q=..., qmodel=...)` at the structure boundary (`H3.6`).

| Model | (1) Reference Table Elements | (2) Damping / Atomic Parameter Elements | (3) CN Parameter Elements | (4) Charge-Model Parameter Elements (`eeq`) | (5) Supported with External Charges (`q`) | (6) Supported with Auto EEQ Charges (`q=None`) | (7) Numerically Validated Elements |
|---|---|---|---|---|---|---|---|
| `D1Model` (`d1`) | $Z \in \{1, 6, 7, 8, 9, 10\}$ (`D1_C6`; $Z \in \{2..5\}$ and $Z > 10$ are zero/unsupported) | $Z \in \{1, 6, 7, 8, 9, 10\}$ (`D1_RADII`, `D1_RADII_PAIRWISE`) | N/A (CN not used) | N/A (charges not used) | $Z \in \{1, 6, 7, 8, 9, 10\}$ ($q$ ignored) | $Z \in \{1, 6, 7, 8, 9, 10\}$ (no EEQ used) | $Z \in \{1, 6, 7, 8, 9, 10\}$ (e.g. dimers, MB16-43 subset, boundary $Z=1, 6, 10$, rejection of $Z \in \{2..5, 11\}$) |
| `D2Model` (`d2`) | $1 \le Z \le 86$ (`D2_C6`, H–Rn) | $1 \le Z \le 86$ (`D2_RADII`, `D2_RADII_PAIRWISE`) | N/A (CN not used) | N/A (charges not used) | $1 \le Z \le 86$ ($q$ ignored) | $1 \le Z \le 86$ (no EEQ used) | $1 \le Z \le 86$ (H–Rn, including main-group, transition-metal, lanthanide, and boundary $Z=86$ Rn; rejection of $Z=87$) |
| `D3Model` (`d3`) | $1 \le Z \le 103$ (`d3-c6.pt` shape `(104, 104, 7, 7)`, `_load_cn` shape `(104, 7)`, H–Lr) | $1 \le Z \le 103$ (`VDW_PAIRWISE` `(104, 104)`, `R4R2` `(119,)`) | $1 \le Z \le 118$ (`COV_D3` `(119,)`) | N/A (charges not used) | $1 \le Z \le 103$ | $1 \le Z \le 103$ (no EEQ used) | $1 \le Z \le 103$ (MB16-43, heavy actinides up to $Z=103$ Lr, periodic cells; rejection of $Z \ge 104$) |
| `D3SModel` (`d3s`) | $1 \le Z \le 94$ (`_C6_D3S` and `_load_cn_d3s` with legacy Fr–Pu references, H–Pu) | $1 \le Z \le 94$ (`WFPAIR_D3S` shape `(95, 95)`, `VDW_PAIRWISE`, `R4R2`) | $1 \le Z \le 118$ (`COV_D3` `(119,)`) | N/A (charges not used) | $1 \le Z \le 94$ | $1 \le Z \le 94$ (no EEQ used) | $1 \le Z \le 94$ (MB16-43, actinides up to $Z=94$ Pu, periodic cells; rejection of $Z \ge 95$) |
| `D4Model` (`d4`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (`d4-ref.pt` for `eeq` and `gfn2`; gap at $104 \le Z \le 111$ where `refn == 0`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (`R4R2`, `VDW_PAIRWISE_119`, `ZEFF`, `GAM` of shape `(119,)` / `(119, 119)`) | $1 \le Z \le 118$ (`COV_D3`, `PAULING_EN` `(119,)`) | $1 \le Z \le 103$ (`tad_multicharge` EEQ2019, `MAX_ELEM_EEQ = 103`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (rejects $104 \le Z \le 111$ and $Z > 118$) | $1 \le Z \le 103$ ($112 \le Z \le 118$ raises `ValueError` in `get_charges` because EEQ2019 is parameterized up to $Z=103$) | $1 \le Z \le 103$ (with auto EEQ and external `q`) and $112 \le Z \le 118$ (Cn–Og with external `q`); rejection of $104 \le Z \le 111$ and auto-EEQ on $112 \le Z \le 118$ |
| `D4SModel` (`d4s`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (`d4-ref.pt`; gap at $104 \le Z \le 111$) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (`WFPAIR_D4S` `(119, 119)`, `R4R2`, `VDW_PAIRWISE_119`, `ZEFF`, `GAM`) | $1 \le Z \le 118$ (`COV_D3`, `PAULING_EN`) | $1 \le Z \le 103$ (`MAX_ELEM_EEQ = 103`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ | $1 \le Z \le 103$ | $1 \le Z \le 103$ (auto EEQ & external `q`) and $112 \le Z \le 118$ (external `q`); rejection of $104 \le Z \le 111$ |
| `D4SrevModel` (`d4srev`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ in `d4srev-ref.pt`, restricted to $1 \le Z \le 103$ by `TANH_PARAMS` | $1 \le Z \le 103$ (`TANH_PARAMS` from `dftd/data/tanh.f90` is non-zero only for $1 \le Z \le 103$; `WFPAIR_D4S`, `R4R2`) | $1 \le Z \le 118$ (`COV_D3`) | $1 \le Z \le 103$ (`MAX_ELEM_EEQ = 103`) | $1 \le Z \le 103$ (rejects $Z \ge 104$) | $1 \le Z \le 103$ (rejects $Z \ge 104$) | $1 \le Z \le 103$ (MB16-43, actinides up to $Z=103$ Lr, periodic cells; rejection of $Z \ge 104$) |

## 4. Fortran-to-PyTorch Module & Algorithm Mapping

| Fortran Module (`grimme-lab/dftd`) | PyTorch Module (`tad_dftd`) | Mathematical & Algorithmic Summary |
|---|---|---|
| `dftd_cutoff` (`cutoff.f90`) | `tad_dftd.cutoff`, `tad_dftd.pairs` | Real-space cutoffs (`cn`, `disp2`, `disp3`, `cn_eeq`, `width2`, `width3`), quintic Hermite smooth cutoff $s(r) = x^3(10 - 15x + 6x^2)$, periodic shifts (`DensePairs`), and `tad-mctc` neighbor/triple lists (`SparsePairs`). |
| `dftd_data_*` (`data/*.f90`, `data/wfpair/*.f90`) | `tad_dftd.data.*` | Atomic $r_4/r_2$ (`r4r2.f90`), $Z_{\text{eff}}$ (`zeff.f90`), chemical hardness $\eta$ (`hardness.f90`), Pauling EN (`en.f90`), D1/D2 radii (`d1.f90`, `d2.f90`), $\tanh$ charge-scaling parameters (`tanh.f90`), and pairwise CN Gaussian weights $k_3^{AB}$ for D3S (`wfpair/d3s.f90`) and D4S/D4Srev (`wfpair/d4s.f90`). |
| `dftd_integrator_*` (`integrator/*.f90`) | `tad_dftd.integrator.*` | 23-point composite trapezoid integrator on fixed imaginary-frequency grid (D4/D4S/D4Srev) and 10-point Gauss-Legendre rational transformation $\omega(t) = \omega_0 \frac{1+t}{1-t}$ with $\omega_0 = 0.151295$. |
| `dftd_damping_*` (`damping/*.f90`, `param.f90`) | `tad_dftd.damping.*` | 2B dampings (`rational`, `screened`, `zero`, `mzero`, `optpower`/`optimizedpower`, `cso`, `koide`, `fermi`, `z`), 3B dampings (`zero_avg`/`zero_d3`/`zero_d4`, `zero`/`zero_product`, `rational`, `screened`, `koide`), and model/damping-specific base damping radii (`d1`, `d2`, `vdw`, `r4r2`). |
| `dftd_model_catalogue` (`model/catalogue.f90`) | `tad_dftd.model.catalogue` | Registry of models (`d1`=10, `d2`=20, `d3`=30, `d3s`=31, `d4`=40, `d4s`=41, `d4srev`=42; `d5`=50 reserved and raises `NotImplementedError`) with default 2B/3B dampings, default radii, and parameter table mappings. |
| `dftd_model_d1_orig` (`model/d1/orig.f90`) | `tad_dftd.model.d1` | Fixed atomic $C_6$ ($Z=1, 6..10$) with harmonic mean $C_6^{AB} = \frac{2 C_{6,A} C_{6,B}}{C_{6,A} + C_{6,B}}$, $C_8 = 0$, $C_9^{ABC} = -\sqrt{|C_6^{AB} C_6^{AC} C_6^{BC}|}$. |
| `dftd_model_d2_orig` (`model/d2/orig.f90`) | `tad_dftd.model.d2` | Fixed atomic $C_6$ ($Z=1..86$) with geometric mean $C_6^{AB} = \sqrt{C_{6,A} C_{6,B}}$, $C_8 = 0$. |
| `dftd_model_d3_{orig,smooth}` (`model/d3/*.f90`) | `tad_dftd.model.d3` | `cn_d3` (`exp_count`, `kcn=16.0`), log-space softmax Gaussian reference weighting ($w_f = 4.0$ for D3; element-pair $w_f(A,B)$ + legacy Fr–Pu references for D3S), factored/dense $C_6$, $C_8 = 3 C_6 r_{4r2,A} r_{4r2,B}$. |
| `dftd_model_d4_{orig,smooth,srev}` (`model/d4/*.f90`) | `tad_dftd.model.d4` | `cn_d4` (EN-weighted `erf_count`, `kcn=7.5`) for D4/D4S and unweighted `erf_count` (`kcn=7.5`) for D4Srev; $\zeta$ charge scaling for D4/D4S and $\zeta_{\tanh}$ for D4Srev; multi-Gaussian CN sum over `igw = 1..ngw(iref, Z)` (divided by `ngw` in D4, pairwise $w_f(A,B)$ in D4S/D4Srev); 23-point Casimir-Polder integration; charge-independent 3B ATM ($q=0$); supports `eeq`, `gfn2`, and `gxtb` (D4Srev) reference charges, plus `d4sc_dispersion_matrix`, `d4sc_atom_energy`, and `d4sc_atom_potential` for self-consistent `D4SC`. |
| `dftd_disp` (`disp.f90`) | `tad_dftd.disp` | Unified `dftd` and model-specific entrypoints (`dftd1`, `dftd2`, `dftd3`, `dftd3s`, `dftd4`, `dftd4s`, `dftd4srev`), property evaluation (`get_properties`), and C6-supplied `dispersion`, `dispersion2`, `dispersion3`. |

## 5. `dxtb` Integration Layer & D4 / D4SC Compatibility Boundary (`H2.5`)

- **D3 (`src/dxtb/_src/components/classicals/dispersion/d3.py`)**: Consolidated onto `tad_dftd` kernels (`tad_dftd.model.weight_references`, `tad_dftd.model.atomic_c6`, `tad_dftd.disp.dispersion2`, `tad_dftd.threebody.dispersion_atm`, `tad_dftd.damping.rational_damping`, `tad_dftd.ncoord.coordination_number`) while preserving `dxtb`'s historical default cutoffs (`cn = 25.0`, `disp = 50.0` Bohr) and `DispersionD3CoreSetup` contracts.
- **Classical D4 (`src/dxtb/_src/components/classicals/dispersion/d4.py` and `_d4_adapter.py`)**:
  - **System-owned canonical Node path (`DispersionD4CoreSetup`)**: Requires exact `DispersionD4`, built-in `d4.model.D4Model`, built-in `d4.damping.RationalDamping`, and `erf_count`. Passing custom `D4Model` subclasses or non-`RationalDamping` strategies to `core_dispersion_d4_setup` raises `TypeError`.
  - **Legacy direct setup path (`DispersionD4Setup` / `DispersionD4.get_cache`)**: Supports custom `D4Model` subclasses (`get_atomic_c6`, `weight_references`), custom `Damping` subclasses (`ZeroDamping`, custom `RationalDamping` subclasses) for two-body `dispersion2` and `get_atm_dispersion`, explicit static `q` overrides, trainable `s9 = 0.0` (which preserves a non-zero $\partial E / \partial s_9$ gradient equal to the unscaled ATM energy), and optional `s10` / `rs10` parameters.
- **Self-Consistent D4SC (`src/dxtb/_src/components/interactions/dispersion/d4sc.py` and `src/dxtb/_src/scf/functional/interactions.py`)**:
  - Consolidated onto `tad_dftd.model.d4.d4sc_dispersion_matrix`, `tad_dftd.model.d4.d4sc_atom_energy`, and `tad_dftd.model.d4.d4sc_atom_potential`.
  - **Custom `D4Model` subclasses**: Supported on both `D4SCSetup` and `D4SCCoreSetup` (and in both functional and legacy SCF routes) as long as the subclass accepts standard `D4Model` constructor arguments (`numbers, ga=..., gc=..., wf=..., ref_charges=..., rc6=..., device=..., dtype=...`) and stores only tensors and immutable scalar/container state in attributes or `__slots__`.
  - **Damping boundary**: Self-consistent D4SC in GFN2-xTB is defined exclusively for two-body Becke-Johnson (`RationalDamping`) damping (with `s6`, `s8`, `a1`, `a2`, and optional `s10`; `s9` is evaluated by the companion classical ATM component and does not enter `d4sc_dispersion_matrix`). Passing a non-`RationalDamping` instance to `setup_d4sc` (or `DispersionD4SC.get_cache` / `Model.setup`) explicitly raises `TypeError`; custom damping in D4SC is only supported via a `DispersionD4SC` subclass overriding `get_cache`.
- **Legacy Shims Removed**: `src/dxtb/_src/mctc_shim.py` and the `tad-dftd3` / `tad-dftd4` packaging dependencies in `setup.cfg` have been completely removed.

## 6. Execution Modes, Periodicity, `torch.compile`, and Environment Policy (`H2.1`, `H2.3`, `H2.4`, `H3.3`, `H3.4`, `H3.5`, `H3.6`)

- **Genuine 1D, 2D, and 3D Periodic Boundary Conditions & Coordinate Wrapping (`H3.3`, `H3.4`)**:
  - `DensePairs` (via `tad_mctc.neighbor.get_periodic_shifts`) and `SparsePairs` (via `tad_mctc.neighbor.build_neighborlist`) natively honor per-axis periodicity flags `periodic = [px, py, pz]`, supporting genuine 1D (`[True, False, False]`), 2D (`[True, True, False]`), and 3D (`[True, True, True]`) periodic boundary conditions across all 7 dispersion models (`d1`, `d2`, `d3`, `d3s`, `d4`, `d4s`, `d4srev`) in both 2-body and 3-body ATM terms, matching Fortran `libdftd.so` (`d2740777049b2a77fd5acc86c4e1abe75d5374ee`) to machine precision ($\le 1.4 \times 10^{-17}\text{ E}_{\text{h}}$) in energy, nuclear gradients, and cell strain/virial tensor $\sigma$.
  - **Charge-model periodic boundary**: Automatic EEQ (`q=None`, `tad_multicharge.EEQModel`) supports 0D molecular and 3D periodic (`[True, True, True]`) structures via 3D Ewald summation; for 1D or 2D periodic structures, `tad_multicharge` raises `NotImplementedError`, so charge-dependent models (`d4`, `d4s`, `d4srev`) in 1D or 2D require externally supplied partial charges `q=...`.
  - **Coordinate wrapping invariance**: Both `tad_mctc` (`get_periodic_shifts` / `build_neighborlist`) and the Fortran `dftd` C-API (`new_structure_api` in `src/dftd/api.f90`) wrap periodic atomic coordinates into the central unit cell $[0, 1)^3$ via `wrap_to_central_cell(positions, lattice, periodic)` before constructing lattice shifts. This guarantees exact machine-precision invariance under integer lattice translations $\mathbf{r}_i \to \mathbf{r}_i + n_1 \mathbf{a}_1 + n_2 \mathbf{a}_2 + n_3 \mathbf{a}_3$, uniform global translations, atom permutations, and supercell replication. (Note: Fortran's internal `test_periodic.f90` unit test bypasses `new_structure_api` and evaluates `mstore`'s unwrapped `acetic`, `adaman`, and `cyanamide` coordinates directly against a finite shift box without central-cell wrapping; when evaluated through Fortran's `dftd_new_structure` C-API or after `wrap_to_central_cell`, `tad_dftd` and Fortran agree to $< 3.4 \times 10^{-16}\text{ E}_{\text{h}}$ on all X23 crystals.)
- **Dense Periodic ATM 2D Tiling (`H2.1`)**: `dispersion_atm_periodic` bounds candidate-leg pairs in both row and column dimensions via `_periodic_tile_shape(n_leg, max_triples)` such that every numerical tile satisfies `tile_rows * tile_cols <= max_triples` (`1 <= tile_rows <= n_leg`, `1 <= tile_cols <= n_leg`), both in eager active-leg mode and in static-leg mode (`torch.func` / `torch.compile`).
- **`torch.compile` and `torch.func` Support (`H2.4`)**:
  - `torch.func` transforms (`grad`, `vjp`, `jvp`, `jacrev`, `jacfwd`, `vmap`) are supported across molecular and periodic dense calculations.
  - `torch.compile` is validated on CPU with both `backend="inductor"` (the default PyTorch Inductor backend, with `fullgraph=True`) and `backend="aot_eager"` across molecular D3, molecular D4, periodic dense D3 with ATM, periodic sparse D3 with ATM, periodic D4 with external charges, and differentiable damping parameters. First derivatives work directly via `torch.autograd.grad` on compiled outputs; second derivatives under Inductor are supported by compiling the `torch.func` derivative transform (`torch.compile(torch.func.jacfwd(torch.func.jacrev(fn)), backend="inductor", fullgraph=True)`), as PyTorch's `AOTAutograd` wraps compiled forward functions in a single-backward `CompiledFunction`. For periodic systems under `vmap`, `jacrev`/`jacfwd` wrt `lattice`, or `torch.compile(fullgraph=True)`, prebuild `DensePairs.build(...)` or `SparsePairs.build(...)` outside the compiled region.
- **Supported Python & PyTorch Environment Matrix (`H2.3`, `H3-F.3`)**:
  - Following commit `887f09b787c99a44d3d7a9fda683c3796edc8569` ("Update upstream dependency baseline for U1-U4"), `setup.cfg` requires `python_requires = >=3.10, <3.15`, `torch>=2.7.0,<3`, `tad-mctc==0.11.3`, and `tad-multicharge==0.8.0`.
  - Validated across the five CPU-only PyTorch environments (`torch.version.cuda = None`, `torch.cuda.is_available() = False`; all validation is CPU-only):
    - `/tmp1/friede/dxtb-agentic/envs/py3.10-torch2.7.0`: `sys.version = '3.10.22 (main, Oct  3 2026, 01:32:06) [Clang 22.1.3 ]'`, `torch.__version__ = '2.7.0+cpu'`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`, `tad-mctc = 0.11.3`, `tad-multicharge = 0.8.0`
    - `/tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0`: `sys.version = '3.11.15 (main, Apr 07 2026, 21:55:08) [GCC]'`, `torch.__version__ = '2.8.0+cpu'`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`, `tad-mctc = 0.11.3`, `tad-multicharge = 0.8.0`
    - `/tmp1/friede/dxtb-agentic/envs/py3.12-torch2.11.0`: `sys.version = '3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0]'`, `torch.__version__ = '2.11.0+cpu'`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`, `tad-mctc = 0.11.3`, `tad-multicharge = 0.8.0`
    - `/tmp1/friede/dxtb-agentic/envs/py3.12-torch2.13.0`: `sys.version = '3.12.10 | packaged by conda-forge | (main, Apr 10 2025, 22:21:13) [GCC 13.3.0]'`, `torch.__version__ = '2.13.0+cpu'`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`, `tad-mctc = 0.11.3`, `tad-multicharge = 0.8.0`
    - `/tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1`: `sys.version = '3.14.8 (main, Oct  3 2026, 01:04:31) [Clang 22.1.3 ]'`, `torch.__version__ = '2.14.1+cpu'`, `torch.version.cuda = None`, `torch.cuda.is_available() = False`, `tad-mctc = 0.11.3`, `tad-multicharge = 0.8.0`
  - Note: `/tmp1/friede/dxtb-agentic/envs/py3.10-torch2.6.0` remains on disk from the pre-U1–U4 baseline (`torch 2.6.0+cpu`, `tad-mctc 0.9.1`, `tad-multicharge 0.7.0`) and lacks `tad_mctc.Structure`, `tad_mctc.neighbor`, and `tad_mctc.pbc`.
