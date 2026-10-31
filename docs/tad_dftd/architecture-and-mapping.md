# `tad_dftd` Architecture and Fortran-to-PyTorch Mapping

## 1. Recorded Source Authority & Git Revisions

- **Main integration repository**: `marvinfriede/dxtb` (`/tmp1/friede/dxtb-agentic/dxtb`)
- **Base branch**: `origin/claude/tender-dijkstra-cda42s`
- **Recorded base commit SHA**: `abdb4562a0607475e775072277b63f82501363a0`
- **Isolated worktree & branch**: `/tmp1/friede/dxtb-agentic/worktrees/unified-tad-dftd` on `feature/unified-tad-dftd`
- **Authoritative Fortran reference (`grimme-lab/dftd`)**: `/tmp1/friede/work/dftd-grimme-lab/dftd` at commit SHA `d2740777049b2a77fd5acc86c4e1abe75d5374ee`
- **Modified PyTorch `tad-dftd3` reference**: `/tmp1/friede/dxtb-agentic/tad-dftd3` at commit SHA `954c0f285e25939b7003e62fdcdc47a2a4be971b` (branch `neighborlist`)

## 2. Package Architecture (`src/tad_dftd/`)

`tad_dftd` is an independent top-level Python package inside `src/tad_dftd/` that depends only on `torch`, `tad-mctc`, `tad-multicharge`, and standard library modules. It never imports `dxtb`, `tad_dftd3`, or `tad_dftd4`.

```text
src/tad_dftd/
├── __init__.py
├── __version__.py
├── py.typed
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
│   ├── eeq.py
│   └── eeqbc.py
├── damping/
│   ├── __init__.py
│   ├── base.py
│   ├── param.py
│   ├── rational.py
│   ├── screened.py
│   ├── zero.py
│   ├── mzero.py
│   ├── optpower.py
│   ├── cso.py
│   ├── fermi.py
│   ├── koide.py
│   ├── z.py
│   ├── radii.py
│   └── select.py
├── reference/
│   ├── __init__.py
│   ├── d1.py
│   ├── d2.py
│   ├── d3.py
│   ├── d4.py
│   ├── d5.py
│   └── *.pt
├── model/
│   ├── __init__.py
│   ├── base.py
│   ├── catalogue.py
│   ├── weights.py
│   ├── fixed.py
│   ├── d1.py
│   ├── d2.py
│   ├── d3.py
│   ├── d4.py
│   └── d5.py
└── param/
    ├── __init__.py
    └── ...
```

## 3. Fortran-to-PyTorch Module & Algorithm Mapping

| Fortran Module (`grimme-lab/dftd`) | PyTorch Module (`tad_dftd`) | Mathematical & Algorithmic Summary |
|---|---|---|
| `dftd_cutoff` (`cutoff.f90`) | `tad_dftd.cutoff`, `tad_dftd.pairs` | Real-space cutoffs (`cn`, `disp2`, `disp3`, `width2`, `width3`), quintic Hermite smooth cutoff $s(r) = x^3(10 - 15x + 6x^2)$, periodic shifts (`DensePairs`), and `tad-mctc` neighbor/triple lists (`SparsePairs`). |
| `dftd_data_*` (`data/*.f90`, `data/wfpair/*.f90`) | `tad_dftd.data.*` | Atomic $r_4/r_2$ (`r4r2.f90`), $Z_{\text{eff}}$ (`zeff.f90`), chemical hardness $\eta$ (`hardness.f90`), Pauling EN (`en.f90`), D1/D2 radii (`d1.f90`, `d2.f90`), $\tanh$ charge-scaling parameters (`tanh.f90`), and pairwise CN Gaussian weights $k_3^{AB}$ for D3S (`wfpair/d3s.f90`) and D4S/D4Srev (`wfpair/d4s.f90`). |
| `dftd_integrator_*` (`integrator/*.f90`) | `tad_dftd.integrator.*` | 23-point composite trapezoid integrator on fixed imaginary-frequency grid (D4/D4S/D4Srev) and 10-point Gauss-Legendre rational transformation $\omega(t) = \omega_0 \frac{1+t}{1-t}$ with $\omega_0 = 0.151295$ (D5). |
| `dftd_damping_*` (`damping/*.f90`, `param.f90`) | `tad_dftd.damping.*` | 2B dampings (`rational`, `screened`, `zero`, `mzero`, `optpower`/`optimizedpower`, `cso`, `koide`, `fermi`, `z`), 3B dampings (`zero_avg`/`zero_d3`/`zero_d4`, `zero`/`zero_product`, `rational`, `screened`, `koide`), and model/damping-specific base damping radii (`d1`, `d2`, `vdw`, `r4r2`). |
| `dftd_model_catalogue` (`model/catalogue.f90`) | `tad_dftd.model.catalogue` | Registry of models (`d1`=10, `d2`=20, `d3`=30, `d3s`=31, `d4`=40, `d4s`=41, `d4srev`=42, `d5`=50) with default 2B/3B dampings, default radii, and parameter table mappings. |
| `dftd_model_d1_orig` (`model/d1/orig.f90`) | `tad_dftd.model.d1` | Fixed atomic $C_6$ ($Z=1, 6..10$) with harmonic mean $C_6^{AB} = \frac{2 C_{6,A} C_{6,B}}{C_{6,A} + C_{6,B}}$, $C_8 = 0$, $C_9^{ABC} = -\sqrt{|C_6^{AB} C_6^{AC} C_6^{BC}|}$. |
| `dftd_model_d2_orig` (`model/d2/orig.f90`) | `tad_dftd.model.d2` | Fixed atomic $C_6$ ($Z=1..86$) with geometric mean $C_6^{AB} = \sqrt{C_{6,A} C_{6,B}}$, $C_8 = 0$. |
| `dftd_model_d3_{orig,smooth}` (`model/d3/*.f90`) | `tad_dftd.model.d3` | `cn_d3` (`exp_count`, `kcn=16.0`), log-space softmax Gaussian reference weighting ($w_f = 4.0$ for D3; element-pair $w_f(A,B)$ + legacy Fr–Pu references for D3S), factored/dense $C_6$, $C_8 = 3 C_6 r_{4r2,A} r_{4r2,B}$. |
| `dftd_model_d4_{orig,smooth,srev}` (`model/d4/*.f90`) | `tad_dftd.model.d4` | `cn_d4` (EN-weighted `erf_count`, `kcn=7.5`) for D4/D4S and unweighted `erf_count` (`kcn=7.5`) for D4Srev; $\zeta$ charge scaling for D4/D4S and $\zeta_{\tanh}$ for D4Srev; multi-Gaussian CN sum over `igw = 1..ngw(iref, Z)` (divided by `ngw` in D4, pairwise $w_f(A,B)$ in D4S/D4Srev); 23-point Casimir-Polder integration; charge-independent 3B ATM ($q=0$); supports `eeq`, `eeqbc`, `gfn2`, and `gxtb` (D4Srev) reference charges, plus `as_matrix=True` (`dispmat`) and analytical/autograd `dgwdq`/`dgwdcn` for `D4SC`. |
| `dftd_model_d5_orig` (`model/d5/*.f90`) | `tad_dftd.model.d5` | $Z=1..54$, `cn_d3` (`exp_count`, `kcn=16.0`), default `eeqbc` (`eeqbc2025`) or `gxtb` charges; 1D charge interpolation via 5th-order smoothstep ($n=2$) or quintic Hermite spline with limited nodal slopes (`limit_avg_deriv`, $n \ge 3$) clamped to $[q_{\min}, q_{\max}]$; 1D CN plateau-transition weighting (`cn_plateau_fraction=0.5`); 10-point Gauss-Legendre Casimir-Polder integration of $\alpha_{dd}(i\omega)$ and $\alpha_{qq}(i\omega)$ for $C_6^{ij}$, $C_8^{ij}$, and $C_9^{ijk}$. |
| `multicharge_model_eeqbc` (`subprojects/multicharge/...`) | `tad_dftd.charge.eeqbc` | Differentiable Bond-Capacity Electronegativity Equilibration (`eeqbc2025`) for 0D molecules and 3D periodic crystals (Ewald electrostatics + bond-capacity coordination matrix $A + B C$). |
| `dftd_disp` (`disp.f90`) | `tad_dftd.disp` | Unified `dftd` and model-specific entrypoints (`dftd1`, `dftd2`, `dftd3`, `dftd3s`, `dftd4`, `dftd4s`, `dftd4srev`, `dftd5`), property evaluation (`get_properties`), and pairwise energy matrices (`get_pairwise_dispersion`). |

## 4. Standalone Extraction Guide

To extract `tad_dftd` into a standalone `tad-dftd` repository:
1. Move `src/tad_dftd/` to the new repository's `src/tad_dftd/`.
2. Move `test/test_tad_dftd/` to the new repository's `test/`.
3. Move `tools/generate_tad_dftd_data.py` and `docs/tad_dftd/` to the new repository.
4. Add standalone `pyproject.toml` / `setup.cfg` declaring `torch`, `tad-mctc>=0.11.3`, and `tad-multicharge>=0.8.0` as runtime dependencies.
5. In `dxtb`, add `tad-dftd` to `install_requires` in `setup.cfg` once the standalone package is published. No code changes in `dxtb` or `tad_dftd` are required.
