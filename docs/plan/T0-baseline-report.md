# Track 0 baseline report (T0.11)

State of dxtb before the restructuring. Every later package is judged
against this record. Generated data and scripts live in `test/test_baseline`
and `benchmarks/baseline`; the tables in the appendix are rendered from the
recorded status with `python -m test.test_baseline.tables`.

## Summary

- **Energies and first derivatives are right.** Against tblite 0.7.0 (the
  reference implementation), energies agree to 2e-6 Eh (about 5e-8 Eh per
  atom, systematic), forces to 1e-8, charges and dipoles to 2e-8, for GFN1
  and GFN2 with both integral drivers. The two drivers agree with each other
  to 1e-11 (first derivatives).
- **Higher derivatives are wrong in three situations**, silently:
  1. *exactly degenerate orbitals* (benzene, Fe(CO)5): second and third
     derivatives through the eigensolver (e.g., GFN2 polarizability of benzene
     24 and -33 instead of 57 a.u.). Reproduced on a 5x5 matrix (T0.4).
  2. *implicit SCF modes* (`implicit`, `nonpure`): only first derivatives are
     right; Hessians off by 10-40%, the polarizability is missing entirely.
  3. *third order of the open-shell NO2 with GFN2* (hyperpolarizability and
     polarizability derivatives).

  In addition, the Lorentzian broadening of the eigensolver biases even first
  derivatives when the HOMO-LUMO gap is small (1.5% at a gap of 1e-3 Eh).
- **Bugs fixed in this track** (T0 rules: they contaminated the baseline or
  gave wrong results with a small fix):
  - stale autograd graphs from the component and integral-driver caches:
    `get_forces` after `get_energy` dropped terms (errors up to 0.1 Eh/bohr)
    (T0.5);
  - padding orbitals occupied in padded batches with the implicit SCF modes
    (anions; 8e-6 Eh for OH- with GFN2);
  - `device="cpu"` (string) and `device="cuda"` raised a `DeviceError`;
  - the D4 three-body exponent `alp` was not passed to tad-dftd4 (no effect,
    no gradient).

  Also removed: all TorchScript code (deprecated; compiling goes through
  `torch.compile`).
- **Memory limits the workloads, not time.** The unrolled SCF needs 8-9 GB
  for forces of 510 atoms or the Hessian of 33 atoms (GFN1); GFN2 runs out
  of 14 GB in the same workloads. Batched SCF with several threads hangs in
  batched LU factorizations (torch/MKL, reproduced without dxtb).
- **Forward mode and `vmap` do not work end to end.** `jacrev` of the energy
  works; `jacfwd`/`hessian` stop at custom autograd functions without `jvp`
  (eigensolver, tad-dftd3, tad-libcint, `CoulombMatrixAG`, `RepulsionAG`);
  `vmap` stops at the cache keys (`data_ptr`), batch-size heuristics and
  data-dependent control flow. `torch.compile` fails inside Dynamo.
- **`Node` (tad-mctc 0.9.0) covers what the plan needs**; two small helpers
  are worth adding in dxtb when containers migrate (section 10). dxtb cannot
  install tad-mctc 0.9 until the tad-* dependants are re-released.

## 1. Environment and commit

| Item | Value |
| --- | --- |
| dxtb | `main` at `a132ec8` (#270) plus this branch; reference data at `fc571e4` (physics identical to `a132ec8`) |
| Python | 3.11.15 |
| torch | 2.14.0 (PyPI wheel, build `2.14.0+cu130`; run on CPU, no GPU present) |
| tad-* | tad-mctc 0.7.0, tad-dftd3 0.6.0, tad-dftd4 0.8.0, tad-multicharge 0.5.0, tad-libcint 0.3.0 |
| numpy / scipy / pyscf | 1.26.4 / 1.17.1 / 2.14.0 |
| tblite (reference only) | 0.7.0 (PyPI) |
| Hardware | 4 vCPU Intel Xeon @ 2.8 GHz, 15 GB RAM, MKL 2024.2 |

`constraints/baseline.txt` pins these versions; the workflow
`.github/workflows/baseline.yaml` installs from it and runs the suite
(`large`/`slow` nightly). The plan's code references point to `46af7bc`;
`main` has moved on by #268 (CCA ordering), #269 (Fermi smearing) and #270
(PyTorch multipole integrals). `OverlapAG` and `EFunction` of the plan no
longer exist: the PyTorch integrals are plain torch code now, and GFN2 runs
with the PyTorch driver. T0.1 step 1 (`fix/type-conversion`) is on `main`
(#266).

## 2. Reference data and tolerances (T0.2)

**Molecule set** (`test/test_baseline/molecules.py`): H2O, CH4, benzene,
glycine (small organics); OH-, NH4+ (charged); NO2 doublet (open shell);
Fe(CO)5 (transition metal); PbH4-BiH3 (heavy main group); a padded batch of
H2O, NH4+, glycine and benzene (`batch_mode=1`); LYS_xao and seven displaced
copies (`batch_mode=2`, 33 atoms). Benzene, glycine, OH-, NH4+ and Fe(CO)5
were relaxed with GFN2.

**Quantities** for GFN1 and GFN2, libcint and PyTorch driver, float64:
total and per-term energies, atomic and shell charges, atomic dipoles and
quadrupoles (GFN2), forces, Hessian, dipole, polarizability, first
hyperpolarizability, dipole and polarizability derivatives; SCF iterations
(informational). Batches: energies, charges, forces. Fresh calculator per
quantity, result cache disabled, unrolled SCF, thresholds 1e-10.

**Storage**: `reference/<method>-<driver>/<system>.npz` with inputs and
metadata (commit, versions, driver, SCF settings); `refdata.py` compares
plain arrays. `test_reference.py` recomputes every file (energies, charges,
forces, dipoles on every run, about 2-3 min with four workers; the rest
nightly).

**Independent reference**: `reference/tblite/<method>/` from the tblite
Python API (energies, gradients, charges, dipoles analytically; higher
derivatives by finite differences of tblite gradients and dipoles). The
recomputed values are checked against both. Existing xtb comparisons of the
test suite (SCF energies in `test/test_scf/samples.py`) are recorded in the
metadata (deviations 1.5e-7 to 1.7e-6 Eh).

Agreement of dxtb with tblite (maximum over all systems, methods and
drivers, excluding the known wrong values below):

| quantity | max. abs. deviation | where | limited by |
| --- | --- | --- | --- |
| energy | 1.7e-6 Eh | LYS_xao (33 atoms) | systematic, ~5e-8 Eh per atom |
| forces | 8.7e-9 | Fe(CO)5 | SCF convergence |
| charges | 2.0e-8 | LYS_xao | SCF convergence |
| dipole | 2.3e-8 | OH- | SCF convergence |
| Hessian | 2.8e-6 | Fe(CO)5 | tblite finite differences |
| polarizability | 2.7e-4 (rel. 3e-6) | PbH4-BiH3 | tblite finite differences |
| dipole derivative | 5.4e-5 | Fe(CO)5 | tblite finite differences |
| hyperpolarizability | 2.2 (rel. 3e-3) | PbH4-BiH3 | tblite nested finite differences |
| polarizability derivative | 1.2e-2 | H2O (GFN2) | tblite nested finite differences |

Where tblite and dxtb differ beyond the finite difference error, dxtb's own
finite differences decided (`xcheck`): CH4/NH4+ dipole derivatives (dxtb
autograd = dxtb finite differences), PbH4-BiH3 hyperpolarizability (dxtb
right to 1e-4 rel.).

**Known wrong values** (`tolerances.KNOWN_WRONG`): benzene (Hessian and all
field/position second and third derivatives), Fe(CO)5 (hyperpolarizability,
polarizability derivative), NO2 with GFN2 (hyperpolarizability,
polarizability derivative). They are stored, but not enforced as regression
targets; the test fails as soon as they agree with tblite.

**Tolerances** (`tolerances.py`), from the observed spread:

| quantity | regression (atol, rtol) | spread observed | vs tblite (atol, rtol) |
| --- | --- | --- | --- |
| energy (and terms) | 1e-9 | 4e-14 (drivers) | 1e-6, 1e-7 |
| charges, multipoles | 1e-8 | 7e-13 | 1e-6 |
| forces | 1e-8 | 8e-12 | 1e-6 |
| dipole | 1e-8 | 5e-12 | 1e-6 |
| Hessian | 1e-7 | 5e-10 | 1e-5 |
| polarizability | 1e-6, 1e-8 | 8e-8 | 1e-4, 1e-5 |
| dipole derivative | 1e-7 | 7e-9 | 1e-4 |
| hyperpolarizability | 1e-3, 1e-6 | 4e-5 (NH4+, Td) | 0.1, 5e-3 |
| polarizability derivative | 1e-4, 1e-6 | 2e-5 (NH4+) | 5e-2, 5e-3 |

"Spread observed" is the libcint/PyTorch driver difference; reruns with
different thread counts stay below it. Finite-difference references of the
status matrix use central differences with steps of 1e-4 (positions, field)
and 1e-5 (parameters); see the step-size scan in the appendix.

## 3. Derivative status (T0.3, T0.4)

### 3.1 Matrix

1236 cells: 10 quantities x up to 5 paths x 2 methods x 2 drivers x 3 SCF
modes x 3 inputs (water; padded water/OH-; two water geometries), each
compared with finite differences of the quantity one order lower. The full
tables are in the appendix; the outcome per cell is in
`test/test_baseline/status/matrix.json` and drives `test_derivatives.py`
(non-passing cells are `xfail(strict=True)`).

Outcome over all 1236 cells: 430 pass, 129 wrong value, 676 exception,
1 not finite. By path and setting:

- **Unrolled SCF (`full`) + autograd, single system: correct to third
  order** for every quantity, both methods and both drivers (deviation from
  finite differences 1e-10 to 7e-7; 0.4 s for forces, 4-7 s for Hessians,
  7-22 s for third order, water). The only exception is the derivative of
  the forces with respect to parameters with libcint (`int1e_rripovlp` not
  available).
- **`functorch` paths** of the Calculator API match autograd for single
  systems with the unrolled SCF; for batches they return cross-system blocks
  (`(nb, nb, ...)`), and `pol_deriv`/`dipole_deriv` fail to reshape.
- **Implicit SCF modes** (`implicit`, `nonpure`): first derivatives right;
  every second and third derivative wrong (Hessian 10-40%, polarizability
  missing, contracted third order 10-30%), or an error in the xitorch
  `RootFinder` under `torch.func` (no vmap rule).
- **Forward mode** (`jacfwd`, `jvp`) works for first derivatives only:
  the dipole (both drivers), the GFN2 forces with the PyTorch driver (GFN1
  stops at the D3 custom function, libcint at its integrals) and the
  parameter derivatives of the energy with the PyTorch driver. Every second
  and third derivative stops at custom autograd functions without `jvp`
  or vmap rule (eigensolver, tad-dftd3, tad-libcint, `CoulombMatrixAG`,
  `RepulsionAG`), and through the plain `eigh` it gives NaN for degenerate
  orbitals (OH-, GFN1).
- **Analytical paths**: analytical forces exist only for GFN1 with libcint
  ("GFN2 not implemented", "no analytical overlap gradient" for the PyTorch
  driver), and so does `hessian_numerical`, which differentiates them; the
  analytical dipole and its derivatives are right with the unrolled SCF.
- **Numerical paths** are right except for the hyperpolarizability with
  GFN2 (nested finite differences with step 1e-5, error 4e-2).
- **Batches**: `Calculator.forces` raises ("grad can be implicitly created
  only for scalar outputs"); `hessian` is not supported in batch mode;
  `dipole_deriv`/`pol_deriv` (autograd and analytical) fail to reshape;
  dipole, polarizability and hyperpolarizability work with the unrolled
  SCF (autograd and `functorch`). The `nonpure` mode fails for batched
  dipoles ("`inputs` argument to `grad()` cannot be empty").

Step-size scan (water): central differences show clean `h^2` behaviour down
to `h = 1e-5` (forces 1.5e-10, Hessian 6e-11 at 1e-5); the references use
`1e-4` (truncation error about 1e-9, polarizability 1e-7). SCF convergence
scan: the unrolled third-order derivatives reach the finite difference
floor at thresholds of 1e-8 to 1e-10 and are within 4e-6 already at 1e-6,
so they are not limited by the convergence.

### 3.2 Component level

23 targets (custom autograd functions, integral kernels and builds, H0,
every classical and interaction energy, the eigensolver), 6 checks each
(`test/test_baseline/components.py`, status in `status/components.json`):

- The PyTorch integral kernels (overlap with McMurchie-Davidson and
  Obara-Saika, dipole, quadrupole) pass everything: third order, forward
  mode, `jvp` and `vmap`. So do ES2 and ES3 (charges and positions), D4SC,
  D4 (tad-dftd4) and the plain repulsion.
- **`CoulombMatrixAG`** and **`RepulsionAG`** pass third order by reverse
  mode but have no `jvp` and no `setup_context`/vmap support.
  `RepulsionAG` recurses without bound when differentiated with respect to
  its parameters (its backward differentiates its own output).
- **D3 dispersion** (tad-dftd3): custom function without `jvp`;
  `vmap` fails at a batch-size heuristic on `numbers`.
- **Eigensolver** (`storch.eighb`, Lorentzian broadening): exactly
  degenerate occupied pair - first order right, second and third order
  wrong; HOMO-LUMO gap 1e-3 Eh - already first order wrong (broadening
  bias); no `jvp`, no vmap rule.
- `vmap` additionally fails for H0 (`torch.allclose`), the halogen bond and
  the basis setup (data-dependent control flow) and AES2 (in-place
  broadcast).

Full table in the appendix.

## 4. Cache findings and actions (T0.5)

| Test | Result | Action |
| --- | --- | --- |
| 1. stale graph after a call without gradients | **confirmed**: forces wrong by up to 0.1 Eh/bohr (water; GFN1/GFN2, both drivers). A second variant, a new leaf with the same values, is also wrong. | Stopgap merged: the component caches (ES2, AES2, ALPB, D4SC, field, field gradient) and the integral driver record `requires_grad` and, for gradient-tracking tensors, a weak reference; a mismatch is a cache miss (`utils/tensors.grad_key`). |
| 2. numerical dipole with result cache | **confirmed**: `dipole_numerical` returns 0 | known issue (strict xfail) |
| 3. views of a batch with result cache | **confirmed, weaker**: `positions[0]` of a cached batch returns the cached batch result instead of raising the shape error | known issue (strict xfail) |
| extra: repeated `get_forces` on the same leaf | raises "backward through the graph a second time" | known issue (strict xfail) |

Tests: `test/test_calculator/test_cache/test_stale.py`.

## 5. Transform status (T0.7)

Water, GFN1/GFN2, both drivers, energy through the Calculator API
(`test/test_baseline/transforms.py`, `status/transforms.json`); vmap
fallback warnings are errors.

| check | GFN1 PyTorch | GFN1 libcint | GFN2 PyTorch | GFN2 libcint |
| --- | --- | --- | --- | --- |
| `jacrev(energy)` | works | works | works | works |
| `jacfwd(energy)` | error (D3 `jvp`) | error (D3 `jvp`) | **works** | error (libcint `jvp`) |
| `hessian(energy)` (`jacfwd(jacrev)`) | error | error | error (eigensolver vmap rule) | error |
| `vmap(energy)`, single | error | error | error | error |
| `vmap(forces)`, single | error | error | error | error |
| `vmap(energy)`, conformer batch | error | error | error | error |
| `torch.compile(energy)` incl. setup | error | error | error | error |
| `torch.compile(calc.energy)`, per call | error | error | error | error |

Nothing falls back silently: every `vmap` failure is an exception. The
first blocker is the same everywhere: `singlepoint` builds a cache key with
`tensor_id`, which calls `data_ptr()` (`utils/tensors.py`), even when the
cache is disabled. Behind it: the batch-size check of tad-dftd3 on
`numbers`, an in-place broadcast in the GFN2 energy path
(`calculators/types/energy.py:163`), and the custom functions without vmap
rule (T0.4). `torch.compile` fails with an internal Dynamo error in
tad-mctc's `einsum` wrapper (opt-einsum path) during `get_density`; with the
setup included, the first graph breaks are `.item()` in
`IndexHelper.from_numbers`.

## 6. Performance profile (T0.8)

CPU only (4 vCPU, 15 GB; no GPU in this environment, GPU runs remain to be
done). Script `benchmarks/baseline/workloads.py`; raw results in
`benchmarks/baseline/results/`. Unrolled SCF, thresholds 1e-8.

| workload | GFN1 (PyTorch driver) | GFN2 (PyTorch driver) | libcint |
| --- | --- | --- | --- |
| W1 water cluster, 510 atoms: energy + forces (4 threads) | 16.5 s + 25.2 s, peak 8.3 GB | **out of memory** (> 14 GB) | GFN1 12.6 s + 12.5 s, 7.3 GB; GFN2 out of memory |
| W2 conformers (tmpda, 69 atoms), energy + forces, 1 thread | 8 conformers: batch 42.7 s, loop 51.9 s, 7.3 GB (batch = loop to 1e-14) | 8 conformers: out of memory | - |
| W2 with 20 conformers | out of memory (> 14 GB) | - | - |
| W2 with 4 threads | hangs (see below) | - | - |
| W3 training step, padded batch, 1 thread | 12 molecules (<= 16 atoms): 2.2 s forward + 4.1 s parameter backward, 1.2 GB; 159 of 2261 leaves get a gradient | 10.2 s + 15.8 s, 2.7 GB; 215 of 1402 leaves | - |
| W3 with 32 molecules (incl. 69 atoms) | out of memory | - | - |
| W4 Hessian, LYS_xao (33 atoms), 4 threads | 58 s, 9.3 GB | **out of memory** after 30 min | - |
| W5 hyperpolarizability, glycine, 4 threads | 8.8 s, 1.3 GB | 17.3 s, 1.8 GB | - |

Stage breakdown (W1, GFN1, dxtb timer and `torch.profiler`): SCF 16.0 s,
integrals 0.75 s, classical terms 0.11 s, setup 0.2 s. In the SCF,
`linalg_eigh` is the largest operation (6.5 s self CPU), followed by matrix
products (3.2 s) and the inversion of the Cholesky factor of the overlap,
which `storch.eighb` recomputes in every iteration with a general LU solve
(1.2 s), although the overlap does not change during the SCF. The pair
loop of the old overlap (plan: `overlap.py:225`) no longer exists.

Findings:

1. **Memory is the limit, not time.** The unrolled SCF keeps every
   iteration on the graph: 8-9 GB for forces of 510 atoms (GFN1) or the
   Hessian of 33 atoms; GFN2 (multipole integrals and potentials) exceeds
   14 GB in the same workloads. Batched forces need roughly the memory of
   one big system of the same total size (8 x 69 atoms: 7.3 GB). This
   matters for E3 (implicit differentiation stores only the fixed point) and
   E6a.
2. **Batched SCF hangs with more than one thread.** Batched
   `torch.linalg.lu_factor` (behind `linalg.solve`/`inv`) does not finish
   with 4 torch threads for batches of 8 matrices of size 230 (0.12 s with
   one thread), and MKL reports "Parameter 6 was incorrect on entry to
   DLASWP" on the way; `cholesky`, `eigh` and `solve_triangular` are fine.
   Reproduction without dxtb:
   `torch.set_num_threads(4); torch.linalg.solve(L, I)` with `L` of shape
   `(8, 230, 230)` (torch 2.14.0 PyPI wheel, MKL 2024.2). `storch.eighb`
   uses this path for `L^-1`; a triangular solve, computed once per
   geometry, would avoid it (tad-mctc; D-track). Not reproduced on other
   hardware yet.
3. **Batching does not pay on CPU** for conformers: `batch_mode=2` is only
   18% faster than a loop at one thread (W2), and slower in memory.
4. Integral evaluation is negligible next to the SCF for large systems with
   the new pair builder; `eigh` dominates (relevant for D6).

## 7. Parameter-gradient coverage (T0.9)

`ParamModule` with gradients switched on for every floating point leaf;
energies and projected forces for H2O, CH4, NH4+, NO2, Fe(CO)5, PbH4-BiH3
and Br2-NH3 (halogen bond); for every leaf of the elements present and every
global leaf, the autograd directional derivative is compared with a central
finite difference (`test/test_baseline/params.py`,
`status/param_coverage.json`).

| method | correct gradient | no effect on the set | gradient path cut | not checked (element absent) |
| --- | --- | --- | --- | --- |
| GFN1 | 105 | 41 | 1 | 2114 |
| GFN2 | 141 | 13 | 0 | 1248 |

- **Bug, fixed:** `dispersion.d4.alp` (GFN2) was not passed to tad-dftd4,
  which used its built-in default (the same value): no effect, no gradient.
- **Bug, subproject:** `dispersion.d3.s9` (GFN1) has no gradient. GFN1 sets
  `s9 = 0`, and tad-dftd3 skips the three-body term with a value branch
  (`if param["s9"] != 0.0`), so `dE/ds9` (the three-body energy, 3e-5 Eh for
  Fe(CO)5) is lost. Relevant if `s9` is trained from zero (F2).
- *No effect, expected:* GFN1 does not use the multipole parameters
  (`dkernel`, `qkernel`, `mprad`, `mpvcn`) or `hamiltonian.xtb.wexp`; GFN2
  does not use `hamiltonian.xtb.kpol`; `xbond` only acts on halogen bonds
  in GFN1; `kpair.Fe-Fe` needs two Fe atoms; `mpvcn` of Fe, Pb, Bi and
  `dkernel` of Fe have no effect on this set (to be extended with more
  d-element and heavy-element systems).
- No `.item()`, `float()` or `tolist()` on parameter tensors was found on
  the parameter path to the energy; the remaining calls are on atomic
  numbers and shell structure (setup path, P3).

## 8. Test-suite inventory (T0.10)

Layer markers are assigned at collection (`LAYER_MARKERS` and
`KEYWORD_MARKERS` in `test/conftest.py`), so `pytest -m cache` selects
every test of a layer without editing the test files.

| marker | tests | rewritten or affected by |
| --- | --- | --- |
| `api_calculator` | 765 | B1-B8 (calculator API, property functions) |
| `cache` | 119 | B5, B6 (result object, no component caches) |
| `batch_mode` | 1858 | E4, E6 (masked SCF loop, `vmap` batching) |
| `efield` | 999 | B7 (fields as inputs) |
| `integrals` | 1346 | E2, E5 (fixed-shape kernels), C5 |
| `scf` | 2972 | E3, E4 (SCF differentiation, masked loop) |
| `param` | 226 | F1, C5 (parameter storage, nodes) |
| `physics_values` | 1740 | kept; migrate to the T0.2 utility where possible |
| `baseline` | 1583 | Track 0 itself (reference data, status matrix) |

Counts overlap (a test can have several layers); the total is 6359 tests
(1047 `large`, 1136 `slow`). `batch_mode` and `scf` include the derivative matrix cells.

**Runtime per file** (JUnit report of the CI marker set `not large and
not slow`, 4 workers; 4176 tests, 28153 s of test time; wall time
1 h 59 min on the baseline machine). The 30 most expensive files; the full
table is in the appendix:

| file | tests | time (s) | markers |
| --- | --- | --- | --- |
| `test/test_baseline/test_derivatives.py` | 206 | 3614.3 | baseline |
| `test/test_scf/test_scf.py` | 141 | 3506.8 | scf |
| `test/test_singlepoint/test_grad_pos_withfield.py` | 18 | 2031.1 | api_calculator, efield, physics_values |
| `test/test_properties/test_quadrupole.py` | 99 | 1990.6 | api_calculator, physics_values |
| `test/test_wavefunction/test_fermi_derivatives.py` | 180 | 1922.2 | scf |
| `test/test_scf/test_full_tracking.py` | 68 | 1750.0 | scf |
| `test/test_scf/test_scp.py` | 63 | 1240.9 | scf |
| `test/test_singlepoint/test_grad_gfn2.py` | 27 | 1229.3 | api_calculator, physics_values |
| `test/test_baseline/test_reference.py` | 117 | 812.2 | baseline |
| `test/test_singlepoint/test_grad_fieldgrad.py` | 24 | 810.8 | api_calculator, efield, physics_values |
| `test/test_scf/test_fermi_derivatives.py` | 19 | 773.5 | scf |
| `test/test_singlepoint/test_grad_field.py` | 40 | 695.2 | api_calculator, efield, physics_values |
| `test/test_properties/test_quadrupole_fieldgrad.py` | 14 | 680.6 | api_calculator, efield, physics_values |
| `test/test_baseline/test_components.py` | 124 | 510.0 | baseline |
| `test/test_singlepoint/test_grad_gfn1.py` | 45 | 465.6 | api_calculator, physics_values |
| `test/test_properties/test_pol_deriv.py` | 2 | 430.0 | api_calculator, physics_values |
| `test/test_libcint/test_coeff_grad.py` | 22 | 425.1 | integrals |
| `test/test_scf/test_hess.py` | 4 | 420.2 | scf |
| `test/test_properties/test_raman.py` | 1 | 363.9 | api_calculator, physics_values |
| `test/test_integrals/test_pytorch_forces_fd.py` | 8 | 348.3 | integrals |
| `test/test_external/test_field.py` | 54 | 303.6 | efield |
| `test/test_properties/test_forces.py` | 5 | 276.4 | api_calculator, physics_values |
| `test/test_singlepoint/test_energy.py` | 112 | 273.2 | api_calculator, physics_values |
| `test/test_properties/test_hyperpol.py` | 2 | 268.3 | api_calculator, physics_values |
| `test/test_a_memory_leak/test_scf.py` | 16 | 241.5 | api_calculator, cache |
| `test/test_properties/test_dipole.py` | 24 | 235.1 | api_calculator, physics_values |
| `test/test_scf/test_charged.py` | 34 | 207.0 | scf |
| `test/test_integrals/test_tblite_reference.py` | 24 | 186.9 | integrals |
| `test/test_calculator/test_cache/test_properties.py` | 30 | 159.6 | api_calculator, cache |
| `test/test_classical/test_dispersion/test_d4sc.py` | 40 | 133.0 | physics_values |

**Flaky tests:** none observed: the run had no failure (4021 passed,
29 skipped, 126 xfailed, all of them recorded known issues). The SCF
derivative tests (`test_scf`, `test_singlepoint/test_grad_*`,
`test_wavefunction/test_fermi_derivatives.py`) dominate the runtime besides
the baseline itself.


## 9. Open questions for B1 and E0

Findings that the decision notes have to answer. Each points to the
evidence above.

**E0 (SCF differentiation and integrals)**

1. *Implicit differentiation is first order only.* The xitorch-based
   implicit modes give wrong second and third derivatives (3.1). Either the
   implicit path gets higher-order rules (differentiate the fixed-point
   condition recursively, with forward-mode support), or the unrolled SCF is
   the only higher-order path. The cost columns of the matrix give the
   unrolled baseline to beat.
2. *Eigenvector-based backward at degeneracies and small gaps.* The
   broadened `1/(e_i - e_j)` backward of the eigensolver is wrong at exact
   degeneracies from second order on (benzene, Fe(CO)5, 5x5 reproduction in
   T0.4) and biased at small gaps even at first order (1.5% at 1e-3 Eh). A
   density-matrix formulation avoids both: with Fermi smearing, the response
   needs only the divided differences `(f_i - f_j)/(e_i - e_j)`, which stay
   finite and tend to `f'(e)` for degenerate pairs, and pairs with equal
   occupation drop out exactly. Candidate for E1.
3. *Third order of the open-shell GFN2 case (NO2).* Wrong although nothing is
   degenerate; GFN1 is right. To be located (alpha/beta channels, Fermi
   occupation derivatives) before E3.
4. *Forward mode.* The unrolled SCF switches the eigensolver broadening on
   only if the Hamiltonian `requires_grad`, which dual tensors do not set;
   forward mode then runs through the plain `torch.linalg.eigh` and gives
   NaN for exactly degenerate orbitals (GFN1 dipole of the linear OH-; the
   NaN cells of the matrix). Otherwise forward mode is blocked by custom
   autograd functions without `jvp`:
   `storch.eighb` (tad-mctc), the C6 interpolation of tad-dftd3, the libcint
   integrals (tad-libcint), `CoulombMatrixAG`, `RepulsionAG`. The PyTorch
   integral kernels and all other components pass forward mode and `jvp`.
5. *PyTorch multipole integrals* exist since #270 and pass every check to
   third order, forward mode and `vmap` (T0.4). Analytical forces, and hence
   `hessian_numerical`, still require libcint ("no analytical overlap
   gradient"). E0 can decide to drop libcint from the derivative paths.
6. *`RepulsionAG` parameter gradients recurse without bound* (T0.4); remove
   the custom function rather than fix it (E1).

**B1 (evaluation API and state)**

1. *Caches.* Beyond the fixed stale-graph bug, the result cache returns
   zeros for numerical field derivatives and stale results for views
   (T0.5), and a cached graph cannot be reused after a backward pass. The
   result object (B5) must replace both the result and the component caches.
2. *Batches.* `Calculator.forces` fails for batches; the `functorch` modes
   return cross-system blocks `(nb, nb, ...)`. Property functions need
   per-system semantics for batched input.
3. *`vmap` blockers in the per-call path* (T0.7): the cache key
   (`tensor_id` calls `data_ptr()` on every `singlepoint`, even with the cache
   disabled), batch-size heuristics on `numbers` (tad-dftd3), data-dependent
   control flow (halogen, basis setup), `torch.allclose` in the Hamiltonian,
   an in-place broadcast in AES2, and the eigensolver without a vmap rule.
4. *Fields as inputs.* `reset()` detaches user tensors (gradient to a field
   is lost); field updates are in-place component state (`update_efield`),
   which also breaks the numerical field derivatives with the cache. Passing
   fields to `singlepoint` (B7) removes both.
5. *Compile.* Dynamo fails inside tad-mctc's `einsum` wrapper (opt-einsum
   path) during `get_density`, also for the per-call path only; graph
   breaks start at `.item()` in `IndexHelper.from_numbers` (setup path).
   tad-mctc 0.9 may behave differently; recheck when the pins move.


## 9a. Starting points of the later tracks on current `main`

The track files describe the code at `46af7bc`. On `main` (`a132ec8` plus
this branch) some starting points have moved:

| Track item | At `46af7bc` (plan) | Now |
| --- | --- | --- |
| E5 PyTorch multipole integrals | missing; GFN2 needs libcint | **done** (#270): dipole and quadrupole integrals in the PyTorch driver; GFN2 runs without libcint and passes every T0.4 check (third order, forward mode, `vmap`) |
| E1 custom autograd functions | `RepulsionAG`, `CoulombMatrixAG`, `OverlapAG`, `EFunction` | `OverlapAG` and `EFunction` are gone (#268/#270). Remaining in dxtb: `RepulsionAG` (opt-in, unbounded recursion for parameters), `CoulombMatrixAG`. Also without `jvp`/vmap rule and on the default path: `storch.eighb` (tad-mctc), the C6 model of tad-dftd3, the libcint integrals (tad-libcint), the xitorch `RootFinder` (implicit modes) |
| E2 fixed-shape pair kernels | per-pair Python loop, geometry-dependent `unique_shell_pairs` | pair builder (`impls/pairs.py`) groups by ordered unique-shell-pair class, built from the index helper; Python loop over classes, not pairs; distance screening exists but is opt-in and refuses to run under `torch.compile`. Still to do for E2: setup in B3, scatter per class, the H0 build, the per-entry loop over batch members in `driver.py` |
| TB table, component caches | keyed on values | additionally check gradient-tracking state (T0.5 stopgap; B6 deletes it) |
| TE starting point, eigensolver | not covered | broadened `eighb` wrong at degeneracies (order >= 2) and biased at small gaps (T0.4); recomputes `L^-1` of the overlap by a general LU solve in every SCF iteration (T0.8) |
| B2 configuration | mutable | unchanged |
| T0.6 "cuda" vs "cuda:0" | known issue | fixed (device normalization), also for `device="cpu"` strings |
| TorchScript | present | removed (`set_jit_enabled`, test switch); compiling goes through `torch.compile` (TD) |

## 10. Review of `Node` (tad-mctc 0.9.0) for the dxtb tracks

`Node`, `ModuleNode` and the tree utilities (`partition`, `combine`,
`stack`, `leaf_paths`) of tad-mctc 0.9.0 were probed against the patterns
the plan needs (script in the report appendix; tad-mctc 0.9.0 installed
separately, dxtb itself still runs on 0.7.0).

| Pattern (plan reference) | Result |
| --- | --- |
| `vmap(f, in_dims=(None, 0))(system, positions)` (P2, conformers) | works |
| `stack` of padded systems with different compositions + `vmap(in_dims=(0, 0))` (P2, P9) | works |
| `jacfwd(jacrev(jacrev(f)))` through a node closure (P7, G1) | works |
| `vmap(grad(...))` over a node (per-sample gradients) | works |
| `replace()` inside `vmap`, `grad` and `torch.compile(fullgraph=True)` (P4) | works |
| node outputs from `vmap` and `jacrev` (frozen result object, B5) | works |
| `partition` float leaves → `grad` → `combine` (training, F1) | works |
| `.to(float32)` keeps integer tensors, converts nested nodes; gradients flow through `.to` | works |
| `ModuleNode` under `grad` and `vmap`; float32 module inside a float64 tree (C1b, F3) | works |
| pickle / deepcopy | works |
| mutation after construction (P4) | blocked (`FrozenInstanceError`) |

Against the C1 specification (`03-TC-node-migration.md`):

| C1 item | tad-mctc 0.9.0 |
| --- | --- |
| 1 field kinds `child`/`context` | as specified; children may also hold static hashable values (see 5) |
| 2 `__init_subclass__` applies the dataclass, registers every subclass | as specified |
| 3 layout check (untagged fields, non-field annotations, `__` names) | as specified, except that `InitVar` annotations are rejected rather than exempt (moot: `__post_init__` is forbidden) |
| 4 pytree registration with `flatten_with_keys_fn`, `serialized_type_name` | as specified |
| 5 construction checks | as specified, except "no Python scalars in child fields": hashable static values (floats, functions) are allowed in child fields and become part of the tree structure |
| 6 `dtype`/`device` | as specified |
| 7 conversion as a strict tree map | `.to`/`.type` convert per field; static child values pass unchanged instead of raising |
| 8 `replace` returns a checked copy, `Self` types | as specified |
| 9 `partition`/`combine` | as specified (plus `stack`, `leaf_paths`) |
| 10 `dataclass_transform`; field specifiers declare `default`, `default_factory`, `init`, `kw_only` | `dataclass_transform` present; `child()`/`context()` take `default`, `default_factory` (and `keep_dtype`), not `init`/`kw_only` |
| 11 CI grep for `object.__setattr__` | tad-mctc side; dxtb now has the same rule as a test (`test/test_utils/test_rules.py`, none found) |
| 12 Python >= 3.10 | as specified |
| release | 0.9.0 (the plan says 0.8) |

None of the differences affects dxtb before C5.

Limits found (none blocks a work package; all have a workaround):

1. **Per-field `in_dims`.** An `in_dims` tree that batches some fields of a
   node and not others cannot be built with the constructor or `replace()`:
   a field set to `None` becomes static, so the tree structure differs from
   the node's. It works when the `in_dims` tree is built from the node's own
   spec (`tree_unflatten(dims, tree_flatten(node)[1])`). A small helper
   (`in_dims_like(node, **fields)`) is worth having once dxtb batches
   per-molecule tables (E6).
2. **`torch.load` (weights-only default since torch 2.6) rejects nodes.**
   Use `torch.serialization.safe_globals([...node classes...])` or save the
   leaves by path (`leaf_paths`). Relevant for exporting trained models
   (F2e); parameter files (TOML) are not affected.
3. **`torch.func.grad` with respect to a whole node** fails if the node holds
   integer tensors (`numbers`, index tensors). By design of torch; the plan's
   `partition`/`combine` pattern is the way to do it.
4. By design and documented: keyword-only construction; a child field that
   is `None` in one object and a tensor in another, or a static child value
   that differs, gives different tree structures (no `stack`/`vmap` across
   them); containers in child fields may not hold `None`; `dtype` raises for
   a node without floating point tensors.

**Port to dxtb:** nothing is needed for Track 0, which does not use `Node`.
dxtb cannot install tad-mctc 0.9 yet: tad-dftd3 0.6.0/0.7.0,
tad-dftd4 0.8.0 and tad-multicharge 0.5.0/0.6.0 pin `tad-mctc==0.7.0` or
`==0.8.0`, so the dependants have to be released first (C2–C4, excluded
here). Items 1 and 2 are dxtb-side helpers to add with the first container
migrated to `Node` (C5), not changes to tad-mctc.


## Appendix: generated tables

<!-- generated by `python -m test.test_baseline.tables` -->

#### Derivative status matrix (T0.3)

##### Input: single

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E9 ✓3 | ✓ | E4 ✓8 | E11 ✓1 | ✓ |
| dipole | 1 | ✓ | E4 ✓8 | E4 ✓8 | E8 ✓4 | ✓ |
| dE_dparam | 1 | n/a | ✓ | n/a | E10 ✓2 | n/a |
| hessian | 2 | n/a | ✗8 ✓4 | E4 ✗4 ✓4 | E | E9 ✓3 |
| polarizability | 2 | E4 ✗4 ✓4 | E4 ✗4 ✓4 | E4 ✗4 ✓4 | E | ✓ |
| dipole_deriv | 2 | ✗8 ✓4 | ✗8 ✓4 | E4 ✗4 ✓4 | E | ✓ |
| dforces_dparam | 2 | n/a | E6 ✗4 ✓2 | n/a | E | n/a |
| third_order | 3 | n/a | ✗8 ✓4 | E | E | n/a |
| hyperpolarizability | 3 | n/a | E4 ✗4 ✓4 | E4 ✗4 ✓4 | E | ✗2 ✓10 |
| pol_deriv | 3 | n/a | E8 ✓4 | E4 ✗4 ✓4 | E | ✓ |

##### Input: padded

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E9 ✓3 | E | E8 ✗4 | E11 ✓1 | ✓ |
| dipole | 1 | ✓ | E4 ✓8 | E8 ✓4 | E10 ✗1 NaN1 | ✓ |
| hessian | 2 | n/a | E | E | E | E9 ✓3 |
| polarizability | 2 | E4 ✗4 ✓4 | E4 ✗4 ✓4 | E8 ✗2 ✓2 | E | ✓ |
| dipole_deriv | 2 | E | E | E8 ✗4 | E | ✓ |
| hyperpolarizability | 3 | n/a | E4 ✗4 ✓4 | E8 ✗2 ✓2 | E | ✗2 ✓10 |
| pol_deriv | 3 | n/a | E | E8 ✗4 | E | ✓ |

##### Input: conformer

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E9 ✓3 | E | E8 ✗4 | E11 ✓1 | ✓ |
| dipole | 1 | ✓ | E4 ✓8 | E8 ✓4 | E10 ✓2 | ✓ |
| hessian | 2 | n/a | E | E | E | E9 ✓3 |
| polarizability | 2 | E4 ✗4 ✓4 | E4 ✗4 ✓4 | E8 ✗2 ✓2 | E | ✓ |
| dipole_deriv | 2 | E | E | E8 ✗4 | E | ✓ |
| hyperpolarizability | 3 | n/a | E4 ✗4 ✓4 | E8 ✗2 ✓2 | E | ✗2 ✓10 |
| pol_deriv | 3 | n/a | E | E8 ✗4 | E | ✓ |


##### Single system by setting

| method | driver | SCF mode | ✓ | ✗ | E | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| gfn1 | libcint | full | 28 | 0 | 11 | 0 |
| gfn1 | libcint | implicit | 14 | 12 | 13 | 0 |
| gfn1 | libcint | nonpure | 11 | 4 | 24 | 0 |
| gfn1 | pytorch | full | 28 | 0 | 11 | 0 |
| gfn1 | pytorch | implicit | 12 | 13 | 14 | 0 |
| gfn1 | pytorch | nonpure | 9 | 5 | 25 | 0 |
| gfn2 | libcint | full | 25 | 1 | 13 | 0 |
| gfn2 | libcint | implicit | 12 | 12 | 15 | 0 |
| gfn2 | libcint | nonpure | 9 | 4 | 26 | 0 |
| gfn2 | pytorch | full | 28 | 1 | 10 | 0 |
| gfn2 | pytorch | implicit | 12 | 13 | 14 | 0 |
| gfn2 | pytorch | nonpure | 9 | 5 | 25 | 0 |

##### Cost of the cells

GFN2, PyTorch driver, unrolled SCF, water:

| quantity | path | status | max abs. dev. | wall time (s) | peak RSS (MB) |
| --- | --- | --- | --- | --- | --- |
| forces | analytical | E | – | 0.7 | 610 |
| forces | autograd | ✓ | 1.4e-09 | 0.4 | 576 |
| forces | functorch | ✓ | 1.4e-09 | 2.6 | 739 |
| forces | forward | ✓ | 1.4e-09 | 2.9 | 720 |
| forces | numerical | ✓ | 1.3e-09 | 4.5 | 566 |
| dipole | analytical | ✓ | 5.9e-09 | 0.5 | 570 |
| dipole | autograd | ✓ | 5.9e-09 | 0.6 | 576 |
| dipole | functorch | ✓ | 5.9e-09 | 3.1 | 727 |
| dipole | forward | ✓ | 5.9e-09 | 2.8 | 719 |
| dipole | numerical | ✓ | 5.9e-09 | 1.7 | 566 |
| dE_dparam | autograd | ✓ | 5.3e-10 | 8.6 | 580 |
| dE_dparam | forward | ✓ | 5.3e-10 | 54.0 | 726 |
| hessian | autograd | ✓ | 4.7e-09 | 7.2 | 953 |
| hessian | functorch | ✓ | 4.7e-09 | 4.2 | 806 |
| hessian | forward | E | – | 2.9 | 723 |
| hessian | numerical | E | – | 0.8 | 610 |
| polarizability | analytical | ✓ | 3.1e-06 | 0.6 | 585 |
| polarizability | autograd | ✓ | 2.4e-08 | 4.5 | 756 |
| polarizability | functorch | ✓ | 2.4e-08 | 4.8 | 746 |
| polarizability | forward | E | – | 3.2 | 718 |
| polarizability | numerical | ✓ | 1.3e-06 | 2.3 | 565 |
| dipole_deriv | analytical | ✓ | 1.4e-07 | 0.8 | 601 |
| dipole_deriv | autograd | ✓ | 1.4e-09 | 3.9 | 776 |
| dipole_deriv | functorch | ✓ | 1.4e-09 | 3.3 | 769 |
| dipole_deriv | forward | E | – | 2.6 | 719 |
| dipole_deriv | numerical | ✓ | 1.1e-07 | 6.1 | 565 |
| dforces_dparam | autograd | ✓ | 5.4e-11 | 14.2 | 738 |
| dforces_dparam | forward | E | – | 2.8 | 728 |
| third_order | autograd | ✓ | 5.1e-09 | 4.5 | 759 |
| third_order | functorch | E | – | 2.5 | 731 |
| third_order | forward | E | – | 2.9 | 729 |
| hyperpolarizability | autograd | ✓ | 2.0e-07 | 11.1 | 1354 |
| hyperpolarizability | functorch | ✓ | 2.0e-07 | 4.2 | 904 |
| hyperpolarizability | forward | E | – | 1.9 | 718 |
| hyperpolarizability | numerical | ✗ | 4.1e-02 | 16.9 | 565 |
| pol_deriv | autograd | ✓ | 8.3e-09 | 11.6 | 1466 |
| pol_deriv | functorch | ✓ | 8.3e-09 | 6.5 | 860 |
| pol_deriv | forward | E | – | 4.3 | 719 |
| pol_deriv | numerical | ✓ | 1.2e-03 | 22.3 | 566 |

##### Distinct outcomes of the cells that do not pass

| cells | status | message | example cell |
| --- | --- | --- | --- |
| 122 | error | NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward m | `dE_dparam-forward-gfn1-libcint-full-single` |
| 78 | error | ValueError: cannot reshape array of size 8 into shape (4,) | `dipole-forward-gfn2-libcint-full-conformer` |
| 73 | error | RuntimeError: You tried to vmap over RootFinder, but it does not have vmap support. Please override and implem | `dipole-forward-gfn1-libcint-implicit-single` |
| 54 | error | RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* | `dforces_dparam-forward-gfn1-libcint-nonpure-single` |
| 54 | error | ValueError: cannot reshape array of size 12 into shape (6,) | `dipole-forward-gfn1-libcint-full-conformer` |
| 48 | error | RuntimeError: `inputs` argument to `grad()` cannot be empty. | `dipole-autograd-gfn1-libcint-nonpure-conformer` |
| 48 | error | RuntimeError: shape '[2, 3, 3, 3]' is invalid for input of size 108 | `dipole_deriv-analytical-gfn1-libcint-full-conformer` |
| 48 | error | NotImplementedError: Hessian calculation is not supported in batch mode. Please use the single system mode. | `hessian-autograd-gfn1-libcint-full-conformer` |
| 36 | error | NotImplementedError: The PyTorch integral driver has no analytical overlap gradient. Use autograd on the overl | `forces-analytical-gfn1-pytorch-full-conformer` |
| 26 | error | RuntimeError: You tried to vmap over _SymEigBroad_V2, but it does not have vmap support. Please override and i | `dipole_deriv-forward-gfn2-pytorch-full-conformer` |
| 24 | error | RuntimeError: grad can be implicitly created only for scalar outputs | `forces-autograd-gfn1-libcint-full-conformer` |
| 24 | error | RuntimeError: The differentiated Tensor at index 0 appears to not have been used in the graph. Set allow_unuse | `pol_deriv-autograd-gfn1-libcint-implicit-conformer` |
| 18 | error | NotImplementedError: GFN2 not implemented yet. | `forces-analytical-gfn2-libcint-full-conformer` |
| 18 | fail | wrong value (max_abs=5.2e+00) | `pol_deriv-functorch-gfn2-libcint-implicit-single` |
| 16 | fail | wrong value (max_abs=3.8e+00) | `polarizability-analytical-gfn1-libcint-implicit-conformer` |
| 10 | fail | wrong value (max_abs=1.3e-01) | `dipole_deriv-analytical-gfn1-libcint-nonpure-single` |
| 8 | fail | shape (2, 3, 2, 3, 3) != (2, 3, 3, 3) | `dipole_deriv-functorch-gfn1-pytorch-full-conformer` |
| 8 | fail | shape (2, 2, 3, 3) != (2, 3, 3) | `forces-functorch-gfn1-pytorch-full-conformer` |
| 8 | error | RuntimeError: shape '[3, 3, 3, 3]' is invalid for input of size 324 | `pol_deriv-autograd-gfn1-libcint-full-conformer` |
| 8 | fail | shape (2, 3, 3, 2, 3, 3) != (2, 3, 3, 3, 3) | `pol_deriv-functorch-gfn1-pytorch-full-conformer` |
| 7 | fail | wrong value (max_abs=7.2e+00) | `hyperpolarizability-autograd-gfn1-libcint-implicit-conformer` |
| 7 | fail | wrong value (max_abs=8.1e+00) | `hyperpolarizability-autograd-gfn2-libcint-implicit-conformer` |
| 6 | error | AttributeError: The integral int1e_rripovlp is not available from libcint, please add it | `dforces_dparam-autograd-gfn1-libcint-full-single` |
| 6 | fail | wrong value (max_abs=1.5e-01) | `dipole_deriv-analytical-gfn1-libcint-implicit-single` |
| 4 | fail | wrong value (max_abs=2.2e-01) | `dipole_deriv-analytical-gfn2-libcint-nonpure-single` |
| 4 | fail | wrong value (max_abs=1.6e-02) | `hessian-autograd-gfn1-libcint-implicit-single` |
| 4 | fail | wrong value (max_abs=4.5e-03) | `hessian-autograd-gfn2-libcint-implicit-single` |
| 4 | fail | wrong value (max_abs=2.1e-02) | `hyperpolarizability-numerical-gfn2-libcint-full-conformer` |
| 3 | fail | wrong value (max_abs=9.6e+00) | `hyperpolarizability-autograd-gfn1-libcint-implicit-padded` |
| 3 | fail | wrong value (max_abs=9.9e+00) | `hyperpolarizability-autograd-gfn2-libcint-implicit-padded` |
| 3 | error | ValueError: treespec.unflatten(leaves): `leaves` has length 6 but the spec refers to a pytree that holds 7 ite | `pol_deriv-forward-gfn2-libcint-full-single` |
| 2 | fail | wrong value (max_abs=1.3e-02) | `hessian-autograd-gfn1-libcint-nonpure-single` |
| 2 | fail | wrong value (max_abs=2.0e-02) | `hessian-autograd-gfn2-libcint-nonpure-single` |
| 2 | fail | wrong value (max_abs=4.1e-02) | `hyperpolarizability-numerical-gfn2-libcint-full-single` |
| 2 | fail | wrong value (max_abs=4.0e+00) | `pol_deriv-functorch-gfn1-libcint-implicit-single` |
| 2 | fail | wrong value (max_abs=9.8e-03) | `third_order-autograd-gfn1-libcint-implicit-single` |
| 2 | fail | wrong value (max_abs=7.7e-03) | `third_order-autograd-gfn1-libcint-nonpure-single` |
| 2 | fail | wrong value (max_abs=4.0e-03) | `third_order-autograd-gfn2-libcint-implicit-single` |
| 2 | fail | wrong value (max_abs=1.2e-02) | `third_order-autograd-gfn2-libcint-nonpure-single` |
| 1 | fail | leaves: hamiltonian.xtb.kpol, hamiltonian.xtb.enscale, hamiltonian.xtb.shell.ss, hamiltonian.xtb.shell.pp, ham | `dforces_dparam-autograd-gfn1-pytorch-implicit-single` |

#### Component-level checks (T0.4)

| target | order1 | order2 | order3 | forward | jvp | vmap |
| --- | --- | --- | --- | --- | --- | --- |
| `repulsion_ag.positions` | ✓ | ✓ | ✓ | E | E | ✓ |
| `repulsion_ag.arep` | E | E | E | E | E | ✓ |
| `repulsion.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `coulomb_matrix_ag.positions` | ✓ | ✓ | ✓ | E | E | E |
| `coulomb_matrix_ag.hubbard` | ✓ | ✓ | ✓ | E | E | E |
| `overlap_md.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `overlap_os.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dipint_os.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `quadint_os.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `overlap.slater` | ✓ | ✓ | ✓ | ✓ | ✓ | E |
| `hcore_gfn1.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | E |
| `hcore_gfn2.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | E |
| `halogen.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | E |
| `dispersion_d3.positions` | ✓ | ✓ | ✓ | E | E | E |
| `dispersion_d4.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `repulsion_gfn2.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `es2.charges` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `es3.charges` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `aes2.dipoles` | ✓ | ✓ | ✓ | ✓ | ✓ | E |
| `d4sc.charges` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `es2.positions` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `eigensolver.degenerate_aufbau` | ✓ | ✗ | ✗ | E | E | E |
| `eigensolver.degenerate_fermi` | ✓ | ✗ | ✗ | E | E | E |
| `eigensolver.small_gap_aufbau` | ✗ | ✗ | ✗ | E | E | E |
| `eigensolver.small_gap_fermi` | ✗ | ✗ | ✗ | E | E | E |

Messages of the checks that do not pass:

- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD.: `coulomb_matrix_ag.hubbard:forward`, `coulomb_matrix_ag.positions:forward`, `dispersion_d3.positions:forward`, `dispersion_d3.positions:jvp`, `eigensolver.degenerate_aufbau:forward`, `eigensolver.degenerate_aufbau:jvp`, `eigensolver.degenerate_fermi:forward`, `eigensolver.degenerate_fermi:jvp`, `eigensolver.small_gap_aufbau:forward`, `eigensolver.small_gap_aufbau:jvp`, `eigensolver.small_gap_fermi:forward`, `eigensolver.small_gap_fermi:jvp`, `repulsion_ag.arep:forward`, `repulsion_ag.arep:jvp`, `repulsion_ag.positions:forward`, `repulsion_ag.positions:jvp`
- GradcheckError: Jacobian mismatch for output 0 with respect to input 0,: `eigensolver.degenerate_aufbau:order2`, `eigensolver.degenerate_aufbau:order3`, `eigensolver.degenerate_fermi:order2`, `eigensolver.degenerate_fermi:order3`, `eigensolver.small_gap_aufbau:order1`, `eigensolver.small_gap_aufbau:order2`, `eigensolver.small_gap_aufbau:order3`, `eigensolver.small_gap_fermi:order1`, `eigensolver.small_gap_fermi:order2`, `eigensolver.small_gap_fermi:order3`
- RuntimeError: In order to use an autograd.Function with functorch transforms (vmap, grad, jvp, jacrev, ...), it must override the setup_context static: `coulomb_matrix_ag.hubbard:jvp`, `coulomb_matrix_ag.hubbard:vmap`, `coulomb_matrix_ag.positions:jvp`, `coulomb_matrix_ag.positions:vmap`
- RuntimeError: You tried to vmap over _SymEigBroad_V2, but it does not have vmap support. Please override and implement the vmap staticmethod or set ge: `eigensolver.degenerate_aufbau:vmap`, `eigensolver.degenerate_fermi:vmap`, `eigensolver.small_gap_aufbau:vmap`, `eigensolver.small_gap_fermi:vmap`
- RuntimeError: Resource temporarily unavailable: `repulsion_ag.arep:order1`, `repulsion_ag.arep:order2`, `repulsion_ag.arep:order3`
- RuntimeError: vmap: It looks like you're attempting to use a Tensor in some data-dependent control flow. We don't support that yet, please file an iss: `halogen.positions:vmap`, `overlap.slater:vmap`
- RuntimeError: vmap over torch.allclose isn't supported yet. Please file an issue at https://github.com/pytorch/pytorch/issues if you need this feature: `hcore_gfn1.positions:vmap`, `hcore_gfn2.positions:vmap`
- RuntimeError: output with shape: `aes2.dipoles:vmap`
- ValueError: Batch size mismatch: expected 2, got 3 in `numbers`. The first dimension should be the batch dimension.: `dispersion_d3.positions:vmap`

#### Transform status (T0.7)

| check | gfn1-pytorch | gfn1-libcint | gfn2-pytorch | gfn2-libcint |
| --- | --- | --- | --- | --- |
| vmap(energy) single | error | error | error | error |
| vmap(forces) single | error | error | error | error |
| vmap(energy) conformer batch | error | error | error | error |
| jacrev(energy) | works | works | works | works |
| jacfwd(energy) | error | error | works | error |
| hessian(energy) | error | error | error | error |
| torch.compile(energy) | error | error | error | error |
| torch.compile(calc.energy) | error | error | error | error |

Messages:

- RuntimeError: Cannot access data pointer of Tensor that doesn't have storage [dxtb/_src/utils/tensors.py:71] (gfn1-pytorch: vmap(energy) single; gfn1-pytorch: vmap(energy) conformer batch; gfn1-libcint: vmap(energy) single; gfn1-libcint: vmap(energy) conformer batch; gfn2-pytorch: vmap(energy) single; gfn2-pytorch: vmap(energy) conformer batch; gfn2-libcint: vmap(energy) single; gfn2-libcint: vmap(energy) conformer batch)
- ValueError: Batch size mismatch: expected 2, got 3 in `numbers`. The first dimension should be the batch dimension. [tad_dftd3/model/c6.py:456] (gfn1-pytorch: vmap(forces) single; gfn1-libcint: vmap(forces) single)
- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD. [tad_dftd3/model/c6.py:95] (gfn1-pytorch: jacfwd(energy); gfn1-pytorch: hessian(energy); gfn1-libcint: jacfwd(energy); gfn1-libcint: hessian(energy))
- InternalTorchDynamoError: AsPythonConstantNotImplementedError: SymNodeVariable() is not a constant (gfn1-pytorch: torch.compile(energy); gfn1-pytorch: torch.compile(calc.energy); gfn1-libcint: torch.compile(energy); gfn1-libcint: torch.compile(calc.energy); gfn2-pytorch: torch.compile(energy); gfn2-pytorch: torch.compile(calc.energy); gfn2-libcint: torch.compile(energy); gfn2-libcint: torch.compile(calc.energy))
- RuntimeError: output with shape [3] doesn't match the broadcast shape [2, 3] [dxtb/_src/calculators/types/energy.py:163] (gfn2-pytorch: vmap(forces) single; gfn2-libcint: vmap(forces) single)
- RuntimeError: You tried to vmap over _SymEigBroad_V2, but it does not have vmap support. Please override and implement the vmap staticmethod or set generate_vmap_rule=True. For mor (gfn2-pytorch: hessian(energy))
- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD. [tad_libcint/interface/integrals/int_2c1e.py:353] (gfn2-libcint: jacfwd(energy); gfn2-libcint: hessian(energy))

#### Parameter-gradient coverage (T0.9)

**gfn1**: gradient path cut: 1, no effect on this set: 41, not checked (element not in set): 2114, ok: 105

| leaf | verdict | energy grad | forces grad |
| --- | --- | --- | --- |
| `dispersion.d3.s9` | gradient path cut | none | none |

No effect on the set (autograd and finite difference zero): `element.Bi.dkernel`, `element.Bi.mprad`, `element.Bi.mpvcn`, `element.Bi.qkernel`, `element.Bi.xbond`, `element.Br.dkernel`, `element.Br.mprad`, `element.Br.mpvcn`, `element.Br.qkernel`, `element.C.dkernel`, `element.C.mprad`, `element.C.mpvcn`, `element.C.qkernel`, `element.C.xbond`, `element.Fe.dkernel`, `element.Fe.mprad`, `element.Fe.mpvcn`, `element.Fe.qkernel`, `element.Fe.xbond`, `element.H.dkernel`, `element.H.mprad`, `element.H.mpvcn`, `element.H.qkernel`, `element.H.xbond`, `element.N.dkernel`, `element.N.mprad`, `element.N.mpvcn`, `element.N.qkernel`, `element.N.xbond`, `element.O.dkernel`, `element.O.mprad`, `element.O.mpvcn`, `element.O.qkernel`, `element.O.xbond`, `element.Pb.dkernel`, `element.Pb.mprad`, `element.Pb.mpvcn`, `element.Pb.qkernel`, `element.Pb.xbond`, `hamiltonian.xtb.kpair.Fe-Fe`, `hamiltonian.xtb.wexp`

**gfn2**: no effect on this set: 13, not checked (element not in set): 1248, ok: 141

No effect on the set (autograd and finite difference zero): `element.Bi.mpvcn`, `element.Bi.xbond`, `element.Br.xbond`, `element.C.xbond`, `element.Fe.dkernel`, `element.Fe.mpvcn`, `element.Fe.xbond`, `element.H.xbond`, `element.N.xbond`, `element.O.xbond`, `element.Pb.mpvcn`, `element.Pb.xbond`, `hamiltonian.xtb.kpol`

#### Test-suite inventory (T0.10), all files

| file | tests | time (s) | markers |
| --- | --- | --- | --- |
| `test/test_baseline/test_derivatives.py` | 206 | 3614.3 | baseline |
| `test/test_scf/test_scf.py` | 141 | 3506.8 | scf |
| `test/test_singlepoint/test_grad_pos_withfield.py` | 18 | 2031.1 | api_calculator, efield, physics_values |
| `test/test_properties/test_quadrupole.py` | 99 | 1990.6 | api_calculator, physics_values |
| `test/test_wavefunction/test_fermi_derivatives.py` | 180 | 1922.2 | scf |
| `test/test_scf/test_full_tracking.py` | 68 | 1750.0 | scf |
| `test/test_scf/test_scp.py` | 63 | 1240.9 | scf |
| `test/test_singlepoint/test_grad_gfn2.py` | 27 | 1229.3 | api_calculator, physics_values |
| `test/test_baseline/test_reference.py` | 117 | 812.2 | baseline |
| `test/test_singlepoint/test_grad_fieldgrad.py` | 24 | 810.8 | api_calculator, efield, physics_values |
| `test/test_scf/test_fermi_derivatives.py` | 19 | 773.5 | scf |
| `test/test_singlepoint/test_grad_field.py` | 40 | 695.2 | api_calculator, efield, physics_values |
| `test/test_properties/test_quadrupole_fieldgrad.py` | 14 | 680.6 | api_calculator, efield, physics_values |
| `test/test_baseline/test_components.py` | 124 | 510.0 | baseline |
| `test/test_singlepoint/test_grad_gfn1.py` | 45 | 465.6 | api_calculator, physics_values |
| `test/test_properties/test_pol_deriv.py` | 2 | 430.0 | api_calculator, physics_values |
| `test/test_libcint/test_coeff_grad.py` | 22 | 425.1 | integrals |
| `test/test_scf/test_hess.py` | 4 | 420.2 | scf |
| `test/test_properties/test_raman.py` | 1 | 363.9 | api_calculator, physics_values |
| `test/test_integrals/test_pytorch_forces_fd.py` | 8 | 348.3 | integrals |
| `test/test_external/test_field.py` | 54 | 303.6 | efield |
| `test/test_properties/test_forces.py` | 5 | 276.4 | api_calculator, physics_values |
| `test/test_singlepoint/test_energy.py` | 112 | 273.2 | api_calculator, physics_values |
| `test/test_properties/test_hyperpol.py` | 2 | 268.3 | api_calculator, physics_values |
| `test/test_a_memory_leak/test_scf.py` | 16 | 241.5 | api_calculator, cache |
| `test/test_properties/test_dipole.py` | 24 | 235.1 | api_calculator, physics_values |
| `test/test_scf/test_charged.py` | 34 | 207.0 | scf |
| `test/test_integrals/test_tblite_reference.py` | 24 | 186.9 | integrals |
| `test/test_calculator/test_cache/test_properties.py` | 30 | 159.6 | api_calculator, cache |
| `test/test_classical/test_dispersion/test_d4sc.py` | 40 | 133.0 | physics_values |
| `test/test_calculator/test_cache/test_stale.py` | 11 | 100.3 | api_calculator, cache |
| `test/test_wavefunction/test_filling.py` | 185 | 99.1 | scf |
| `test/test_properties/test_ir.py` | 1 | 99.0 | api_calculator, physics_values |
| `test/test_scf/test_elements_gfn2.py` | 14 | 89.8 | scf |
| `test/test_properties/test_dipole_deriv.py` | 2 | 88.1 | api_calculator, physics_values |
| `test/test_integrals/test_pytorch_gfn2.py` | 7 | 87.1 | integrals |
| `test/test_scf/test_padding_ghosts.py` | 6 | 85.4 | scf |
| `test/test_scf/test_elements_gfn1.py` | 14 | 79.2 | scf |
| `test/test_properties/test_hessian.py` | 1 | 66.8 | api_calculator, physics_values |
| `test/test_scf/test_gfn1_grad.py` | 19 | 64.5 | scf |
| `test/test_properties/test_vibration.py` | 1 | 63.9 | api_calculator, physics_values |
| `test/test_properties/test_pol.py` | 2 | 56.5 | api_calculator, physics_values |
| `test/test_solvation/test_alpb.py` | 10 | 49.5 | physics_values |
| `test/test_integrals/test_quadrupole_getter.py` | 6 | 47.9 | integrals |
| `test/test_integrals/test_screening.py` | 71 | 41.3 | integrals |
| `test/test_solvation/test_grad.py` | 2 | 32.4 | physics_values |
| `test/test_integrals/test_multipole_matrix.py` | 15 | 32.3 | integrals |
| `test/test_integrals/test_multipole_kernels.py` | 34 | 29.7 | integrals |
| `test/test_classical/test_repulsion/test_grad_param.py` | 20 | 27.9 | param, physics_values |
| `test/test_libcint/test_gradcheck.py` | 63 | 27.6 | integrals |
| `test/test_calculator/test_cache/test_invalid.py` | 6 | 25.1 | api_calculator, cache |
| `test/test_a_memory_leak/test_higher_deriv.py` | 16 | 24.8 | api_calculator, cache |
| `test/test_integrals/test_param_grad.py` | 3 | 23.6 | integrals, param |
| `test/test_integrals/test_pyscf_reference.py` | 26 | 20.6 | integrals |
| `test/test_libcint/test_multipole.py` | 128 | 16.3 | integrals |
| `test/test_classical/test_dispersion/test_hess.py` | 4 | 16.0 | physics_values |
| `test/test_properties/test_vibration_ref.py` | 1 | 15.2 | api_calculator, physics_values |
| `test/test_overlap/test_overlap_molecules.py` | 24 | 13.9 | integrals, physics_values |
| `test/test_overlap/test_uplo.py` | 12 | 13.6 | integrals, physics_values |
| `test/test_classical/test_dispersion/test_d4_2b.py` | 20 | 13.5 | physics_values |
| `test/test_hamiltonian/test_gfn1.py` | 23 | 13.4 | integrals, physics_values |
| `test/test_calculator/test_cache/test_integrals.py` | 6 | 13.2 | api_calculator, cache |
| `test/test_interaction/test_grad.py` | 20 | 12.2 | physics_values |
| `test/test_classical/test_dispersion/test_d4.py` | 4 | 12.1 | physics_values |
| `test/test_param/test_module.py` | 35 | 11.8 | param |
| `test/test_overlap/test_grad.py` | 21 | 11.7 | integrals, physics_values |
| `test/test_calculator/test_device_spec.py` | 3 | 11.6 | api_calculator |
| `test/test_scf/test_guess_grad.py` | 40 | 11.4 | scf |
| `test/test_hamiltonian/test_grad_pos.py` | 24 | 11.1 | integrals, physics_values |
| `test/test_overlap/test_overlap_pairs.py` | 24 | 10.9 | integrals, physics_values |
| `test/test_basis/test_export.py` | 176 | 10.9 | integrals |
| `test/test_classical/test_dispersion/test_d3.py` | 5 | 10.4 | physics_values |
| `test/test_calculator/test_cache/test_optional.py` | 4 | 9.1 | api_calculator, cache |
| `test/test_overlap/test_grad_pos.py` | 28 | 8.3 | integrals, physics_values |
| `test/test_classical/test_repulsion/test_grad_pos.py` | 48 | 8.2 | physics_values |
| `test/test_classical/test_dispersion/test_d4sc_grad.py` | 10 | 7.9 | physics_values |
| `test/test_hamiltonian/test_gfn2.py` | 16 | 7.7 | integrals, physics_values |
| `test/test_scf/test_nonselfconsistent.py` | 10 | 7.1 | scf |
| `test/test_integrals/test_wrappers.py` | 7 | 6.6 | integrals |
| `test/test_param/test_param.py` | 21 | 6.5 | param |
| `test/test_a_memory_leak/test_repulsion.py` | 4 | 6.1 | api_calculator, cache |
| `test/test_classical/test_repulsion/test_energy.py` | 84 | 6.0 | physics_values |
| `test/test_classical/test_dispersion/test_general.py` | 10 | 6.0 | physics_values |
| `test/test_cli/test_driver.py` | 9 | 5.8 | api_calculator |
| `test/test_scf/test_warnings_errors.py` | 4 | 5.8 | scf |
| `test/test_integrals/test_driver_precision.py` | 7 | 5.5 | integrals |
| `test/test_libcint/test_shape.py` | 56 | 5.3 | integrals |
| `test/test_calculator/test_dd.py` | 2 | 5.2 | api_calculator |
| `test/test_integrals/test_multipole_options.py` | 4 | 5.1 | integrals |
| `test/test_classical/test_dispersion/test_grad_param.py` | 8 | 5.1 | param, physics_values |
| `test/test_coulomb/test_grad_shell.py` | 16 | 4.7 | physics_values |
| `test/test_wavefunction/test_mulliken.py` | 96 | 4.7 | scf |
| `test/test_integrals/test_driver/test_manager.py` | 19 | 4.7 | integrals |
| `test/test_classical/test_dispersion/test_grad_pos.py` | 18 | 4.6 | physics_values |
| `test/test_overlap/test_overlap_atoms.py` | 10 | 4.4 | integrals, physics_values |
| `test/test_integrals/test_general.py` | 16 | 4.3 | integrals |
| `test/test_param/test_shared.py` | 7 | 4.2 | param |
| `test/test_coulomb/test_es2_general.py` | 14 | 4.2 | physics_values |
| `test/test_integrals/test_libcint.py` | 8 | 4.2 | integrals |
| `test/test_libcint/test_overlap.py` | 36 | 4.1 | integrals |
| `test/test_interaction/test_list.py` | 5 | 4.1 | physics_values |
| `test/test_coulomb/test_es3_general.py` | 11 | 4.1 | physics_values |
| `test/test_classical/test_halogen/test_grad_param.py` | 12 | 3.9 | param, physics_values |
| `test/test_coulomb/test_es3_shell.py` | 19 | 3.8 | physics_values |
| `test/test_classical/test_shortrangebond/test_general.py` | 10 | 3.5 | physics_values |
| `test/test_calculator/test_general.py` | 6 | 3.4 | api_calculator |
| `test/test_classical/test_halogen/test_grad_pos.py` | 22 | 3.1 | physics_values |
| `test/test_integrals/test_driver/test_pytorch.py` | 14 | 3.1 | integrals |
| `test/test_classical/test_ies/test_general.py` | 8 | 3.0 | physics_values |
| `test/test_coulomb/test_grad_atom.py` | 12 | 2.9 | physics_values |
| `test/test_singlepoint/test_hess.py` | 3 | 2.9 | api_calculator, physics_values |
| `test/test_coulomb/test_es2_shell.py` | 30 | 2.9 | physics_values |
| `test/test_hamiltonian/test_general.py` | 12 | 2.9 | integrals, physics_values |
| `test/test_cli/test_entrypoint.py` | 4 | 2.8 | api_calculator |
| `test/test_classical/test_repulsion/test_general.py` | 9 | 2.4 | physics_values |
| `test/test_libcint/test_overlap_grad.py` | 20 | 2.2 | integrals |
| `test/test_libcint/test_symmetry.py` | 24 | 2.2 | integrals |
| `test/test_classical/test_ies/test_energy.py` | 20 | 2.1 | physics_values |
| `test/test_integrals/test_factory.py` | 16 | 2.0 | integrals |
| `test/test_interaction/test_cache.py` | 6 | 2.0 | cache |
| `test/test_classical/test_halogen/test_general.py` | 9 | 2.0 | physics_values |
| `test/test_classical/test_shortrangebond/test_energy.py` | 24 | 1.9 | physics_values |
| `test/test_basis/test_general.py` | 13 | 1.9 | integrals |
| `test/test_coulomb/test_es2_atom.py` | 30 | 1.8 | physics_values |
| `test/test_classical/test_halogen/test_hess.py` | 6 | 1.7 | physics_values |
| `test/test_coulomb/test_es3_atom.py` | 27 | 1.7 | physics_values |
| `test/test_coulomb/test_grad_shell_pos.py` | 7 | 1.6 | physics_values |
| `test/test_integrals/test_pytorch.py` | 4 | 1.5 | integrals |
| `test/test_classical/test_dispersion/test_grad_general.py` | 4 | 1.3 | physics_values |
| `test/test_integrals/test_driver/test_factory.py` | 5 | 1.3 | integrals |
| `test/test_indexhelper/test_general.py` | 14 | 1.3 | - |
| `test/test_wavefunction/test_wiberg.py` | 24 | 1.2 | scf |
| `test/test_coulomb/test_aes2_general.py` | 1 | 1.2 | physics_values |
| `test/test_coulomb/test_grad_atom_pos.py` | 12 | 1.1 | physics_values |
| `test/test_basis/test_normalization.py` | 182 | 1.1 | integrals |
| `test/test_components/test_cache.py` | 4 | 1.0 | cache |
| `test/test_hamiltonian/test_base.py` | 2 | 1.0 | integrals, physics_values |
| `test/test_classical/test_halogen/test_energy.py` | 20 | 0.9 | physics_values |
| `test/test_classical/test_repulsion/test_hess.py` | 3 | 0.9 | physics_values |
| `test/test_classical/test_halogen/test_grad_general.py` | 1 | 0.9 | physics_values |
| `test/test_classical/test_dispersion/test_energy.py` | 2 | 0.9 | physics_values |
| `test/test_coulomb/test_grad_atom_param.py` | 12 | 0.8 | param, physics_values |
| `test/test_basis/test_setup.py` | 26 | 0.7 | integrals |
| `test/test_solvation/test_born.py` | 29 | 0.7 | physics_values |
| `test/test_coulomb/test_aes2.py` | 4 | 0.7 | physics_values |
| `test/test_coulomb/test_grad_shell_param.py` | 8 | 0.6 | param, physics_values |
| `test/test_classical/test_dispersion/test_d4_alp.py` | 1 | 0.5 | physics_values |
| `test/test_integrals/test_ao_ordering.py` | 16 | 0.5 | integrals |
| `test/test_scf/test_mixer.py` | 9 | 0.4 | scf |
| `test/test_utils/test_eigh.py` | 2 | 0.3 | - |
| `test/test_scf/test_general.py` | 4 | 0.3 | scf |
| `test/test_integrals/test_types.py` | 3 | 0.3 | integrals |
| `test/test_singlepoint/test_general.py` | 1 | 0.2 | api_calculator, physics_values |
| `test/test_classical/test_repulsion/test_grad_general.py` | 1 | 0.2 | physics_values |
| `test/test_cli/test_args.py` | 20 | 0.1 | api_calculator |
| `test/test_mol/test_external.py` | 14 | 0.1 | - |
| `test/test_indexhelper/test_spread_reduce.py` | 10 | 0.1 | - |
| `test/test_interaction/test_potential.py` | 12 | 0.1 | physics_values |
| `test/test_utils/test_rules.py` | 1 | 0.1 | - |
| `test/test_loader/test_lazy/test_param.py` | 5 | 0.1 | param |
| `test/test_scf/test_guess.py` | 5 | 0.0 | scf |
| `test/test_coulomb/test_average.py` | 6 | 0.0 | physics_values |
| `test/test_config/test_main.py` | 11 | 0.0 | api_calculator |
| `test/test_utils/test_timer.py` | 7 | 0.0 | - |
| `test/test_param/test_util.py` | 5 | 0.0 | param |
| `test/test_config/test_integral.py` | 7 | 0.0 | api_calculator |
| `test/test_io/test_outputs.py` | 7 | 0.0 | api_calculator |
| `test/test_indexhelper/test_extra.py` | 7 | 0.0 | - |
| `test/test_utils/test_misc.py` | 8 | 0.0 | - |
| `test/test_external/test_general.py` | 8 | 0.0 | efield |
| `test/test_config/test_exlibs_available.py` | 3 | 0.0 | api_calculator |
| `test/test_components/test_list.py` | 2 | 0.0 | cache |
| `test/test_indexhelper/test_culling.py` | 2 | 0.0 | batch_mode |
| `test/test_param/test_tensor.py` | 3 | 0.0 | param |
| `test/test_loader/test_lazy/test_attach_var.py` | 4 | 0.0 | param |
| `test/test_classical/test_list.py` | 1 | 0.0 | physics_values |
| `test/test_io/test_logging.py` | 1 | 0.0 | api_calculator |
| `test/test_overlap/test_general.py` | 1 | 0.0 | integrals, physics_values |
| `test/test_basis/test_orthogonalize.py` | 2 | 0.0 | integrals |
| `test/test_utils/test_tensors.py` | 6 | 0.0 | - |
| `test/test_interaction/test_base.py` | 2 | 0.0 | physics_values |
| `test/test_loader/test_lazy/test_attach_module.py` | 2 | 0.0 | param |
| `test/test_config/test_export.py` | 1 | 0.0 | api_calculator |
