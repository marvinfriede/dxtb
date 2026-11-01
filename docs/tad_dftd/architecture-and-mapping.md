# `tad_dftd` Architecture, Model Catalogue, and Fortran-to-PyTorch Mapping

## 1. Recorded Source Authority & Git Revisions

- **Main integration repository**: `marvinfriede/dxtb` (`/tmp1/friede/dxtb-agentic/dxtb`)
- **Base branch**: `origin/claude/tender-dijkstra-cda42s`
- **Recorded base commit SHA**: `abdb4562a0607475e775072277b63f82501363a0`
- **Isolated worktree & branch**: `/tmp1/friede/dxtb-agentic/worktrees/unified-tad-dftd` on `feature/unified-tad-dftd`
- **Authoritative Fortran reference (`grimme-lab/dftd`)**: `/tmp1/friede/work/dftd-grimme-lab/dftd` at commit SHA `d2740777049b2a77fd5acc86c4e1abe75d5374ee` (branch `split-c-api-tests`)
- **Modified PyTorch `tad-dftd3` reference**: `/tmp1/friede/dxtb-agentic/tad-dftd3` at commit SHA `954c0f285e25939b7003e62fdcdc47a2a4be971b` (branch `neighborlist`)

## 2. Package Architecture (`src/tad_dftd/`)

`tad_dftd` is an independent top-level Python package inside `src/tad_dftd/` that depends only on `torch`, `tad-mctc`, `tad-multicharge`, `tomli` (on Python < 3.11), and standard library modules. It never imports `dxtb`, `tad_dftd3`, or `tad_dftd4`.

```text
src/tad_dftd/
├── __init__.py
├── __version__.py
├── py.typed
├── LICENSE
├── COPYING
├── COPYING.LESSER
├── PROVENANCE.md
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

## 3. Model Catalogue & Element Coverage

| Model Key (`DispersionModelId`) | Implemented Status | Supported Atomic Numbers ($Z$) | Default CN & Cutoff (Bohr) | Charge Dependency (`qmodel`) | Default 2B / 3B Damping & Radii |Default 2B / 3B Cutoff (Bohr) |
|---|---|---|---|---|---|---|
| `d1` (`10`) | Implemented (`D1Model`) | $Z \in \{1, 6, 7, 8, 9, 10\}$ (H, C–Ne; $Z=0$ padding) | None (fixed $C_6$) | None | 2B: `fermi` (`d1` radii); 3B: `zero_avg` (`r4r2` radii, expert option) | `disp2 = 36.8`, `disp3 = 40.0` |
| `d2` (`20`) | Implemented (`D2Model`) | $1 \le Z \le 86$ (H–Rn; $Z=0$ padding) | None (fixed $C_6$) | None | 2B: `fermi` (`d2` radii); 3B: `zero_avg` (`r4r2` radii, expert option) | `disp2 = 94.5`, `disp3 = 40.0` |
| `d3` (`30`) | Implemented (`D3Model`) | $1 \le Z \le 103$ (H–Lr; $Z=0$ padding) | `cn_d3` (`exp_count`, `kcn=16.0`, `cn = 40.0`) | None | 2B: `rational` (`r4r2`), `zero`, `mzero`, `optpower`, `cso`, `z`; 3B: `zero_d3` (`vdw` radii) | `disp2 = 94.86832980505137` ($\sqrt{9000}$), `disp3 = 40.0` |
| `d3s` (`31`) | Implemented (`D3SModel`) | $1 \le Z \le 94$ (H–Pu; $Z=0$ padding; raises `IndexError` for $Z \ge 95$) | `cn_d3` (`exp_count`, `kcn=16.0`, `cn = 40.0`) | None | Same as `d3` (`rational` / `zero` / `mzero` / `optpower`; 3B `zero_d3`) with pairwise $w_f(A,B)$ | `disp2 = 94.86832980505137`, `disp3 = 40.0` |
| `d4` (`40`) | Implemented (`D4Model`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (H–Lr, Cn–Og; $Z=0$ padding; note: internal `eeq` supports $1 \le Z \le 103$, while $112 \le Z \le 118$ requires external `q`) | `cn_d4` (EN-weighted `erf_count`, `kcn=7.5`, `cn = 30.0`, `cn_eeq = 25.0`) | `eeq` (`eeq2019`, default) or external `q` (`gfn2`, `gxtb`); `eeqbc` / `eeqbc2025` raises `NotImplementedError` | 2B: `rational` (`r4r2`), `zero`, `mzero`, `optpower`; 3B: `zero_d4` (`r4r2` radii, evaluated at $q = 0$) | `disp2 = 60.0`, `disp3 = 40.0` |
| `d4s` (`41`) | Implemented (`D4SModel`) | $1 \le Z \le 103$ and $112 \le Z \le 118$ (same as `d4`) | `cn_d4` (`cn = 30.0`, `cn_eeq = 25.0`) | Same as `d4` (`eeq`, external `q`; `eeqbc` raises `NotImplementedError`) | Same as `d4`, with pairwise CN Gaussian widths $w_f(A,B)$ instead of $1/\text{ngw}$ | `disp2 = 60.0`, `disp3 = 40.0` |
| `d4srev` (`42`) | Implemented (`D4SrevModel`) | $1 \le Z \le 103$ (H–Lr; $Z=0$ padding; `tanh.f90` defines charge-scaling parameters up to $Z = 103$) | `cn_d4srev` (unweighted `erf_count`, `kcn=7.5`, `cn = 30.0`, `cn_eeq = 25.0`) | `eeq` (`eeq2019`, default) or external `q` (`gfn2`, `gxtb`); `eeqbc` raises `NotImplementedError` | 2B: `rational`, `screened`, `zero`, `mzero`, `optpower`, `koide`; 3B: `zero_d4`, `zero_product`, `rational`, `screened`, `koide` (evaluated at $q = 0$) | `disp2 = 60.0`, `disp3 = 40.0` |
| `d5` (`50`) | **Excluded / Not Implemented** | Raises `NotImplementedError` | Raises `NotImplementedError` | Raises `NotImplementedError` | Intentionally excluded because upstream `D5` and `EEQBC` are not finalized in `dftd` | Raises `NotImplementedError` |

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

## 5. `dxtb` Integration Layer

- **D3 (`src/dxtb/_src/components/classicals/dispersion/d3.py`)**: Consolidated onto `tad_dftd` kernels (`tad_dftd.model.weight_references`, `tad_dftd.model.atomic_c6`, `tad_dftd.disp.dispersion2`, `tad_dftd.threebody.dispersion_atm`, `tad_dftd.damping.rational_damping`, `tad_dftd.ncoord.coordination_number`) while preserving `dxtb`'s historical default cutoffs (`cn = 25.0`, `disp = 50.0` Bohr) and `DispersionD3CoreSetup` contracts.
- **D4 (`src/dxtb/_src/components/classicals/dispersion/_d4_adapter.py`)**: Thin compatibility adapter translating `dxtb`'s `D4Model`, `Cutoff`, and pure `Damping` objects onto `tad_dftd` (`tad_dftd.reference.d4.load_d4`, `tad_dftd.model.weights.weight_references_d4`, `tad_dftd.model.c6.contract_rc6`, `tad_dftd.disp.dispersion2`, `tad_dftd.disp.dispersion3`, `tad_dftd.threebody.dispersion_atm`).
- **D4SC (`src/dxtb/_src/components/interactions/dispersion/d4sc.py` and `src/dxtb/_src/scf/functional/interactions.py`)**: Consolidated onto `tad_dftd.model.d4.d4sc_dispersion_matrix`, `tad_dftd.model.d4.d4sc_atom_energy`, and `tad_dftd.model.d4.d4sc_atom_potential`.
- **Legacy Shims Removed**: `src/dxtb/_src/mctc_shim.py` and the `tad-dftd3` / `tad-dftd4` packaging dependencies in `setup.cfg` have been completely removed.
