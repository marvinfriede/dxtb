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
- **Higher derivatives are wrong in two situations**, silently:
  1. *exactly degenerate orbitals* (benzene, Fe(CO)5): second and third
     derivatives through the eigensolver (e.g., GFN2 polarizability of benzene
     24 and -33 instead of 57 a.u.). Reproduced on a 5x5 matrix (T0.4).
  2. *third order of the open-shell NO2 with GFN2* (hyperpolarizability and
     polarizability derivatives).

  In addition, the Lorentzian broadening of the eigensolver biases even first
  derivatives when the HOMO-LUMO gap is small (1.5% at a gap of 1e-3 Eh).
- **The implicit SCF mode was fixed on `main` (#272).** At `a132ec8`, the
  implicit modes were right only to first order (Hessians off by 10-40%, the
  polarizability missing). With the new fixed-point solver, `scf_mode="implicit"`
  matches the unrolled SCF cell by cell with autograd (single systems and
  batches, up to third order), and the `nonpure` mode is gone. What remains:
  every `torch.func` and forward-mode cell errors with the implicit mode
  (`requires_grad_()` inside a transformed function), so it is autograd only.
- **The derivative with respect to `refocc` was wrong since #271; fixed
  (T0.12).** The reference occupation (`element.<X>.refocc`) also sets the
  fractional number of electrons, which keeps its graph, but autograd missed
  that contribution: for water (GFN1), `dE/d refocc(H)` was -0.0036 with
  autograd and -0.2259 with central differences, `O`: 0.076 against 0.234,
  for all six `refocc` leaves of both methods (section 7). Cause: the Fermi
  occupations drop the derivative with respect to the number of electrons in
  gaps too wide for the differentiable Newton steps (water: about 300 kT), so
  the electron-count path contributed exactly zero. Those gaps now get the
  exact linear response of the occupations to the electron count (a softmax
  over the tails, independent of the Fermi energy). Autograd and finite
  differences agree again (1e-5 relative) for GFN1 and GFN2, unrolled and
  implicit; `test/test_scf/test_refocc_grad.py` guards it.
- **Bugs fixed in this track** (T0 rules: they contaminated the baseline or
  gave wrong results with a small fix):
  - stale autograd graphs from the component and integral-driver caches:
    `get_forces` after `get_energy` dropped terms (errors up to 0.1 Eh/bohr)
    (T0.5);
  - padding orbitals occupied in padded batches with the implicit SCF modes
    (anions; 8e-6 Eh for OH- with GFN2);
  - `device="cpu"` (string) and `device="cuda"` raised a `DeviceError`;
  - the D4 three-body exponent `alp` was not passed to tad-dftd4 (no effect,
    no gradient);
  - the batched eigensolver stall (below): the unrolled SCF now computes the
    inverse Cholesky factor of the overlap once and passes it to
    `storch.eighb` (`l_inv`, tad-mctc 0.9.1).

  Also removed: all TorchScript code (deprecated; compiling goes through
  `torch.compile`).
- **Memory limits the workloads, not time.** The unrolled SCF needs 8-9 GB
  for forces of 510 atoms or the Hessian of 33 atoms (GFN1); GFN2 runs out
  of 14 GB in the same workloads (section 6).
- **Forward mode and `vmap` do not work end to end.** `jacrev` of the energy
  works; `jacfwd`/`hessian` work only for GFN2 with the PyTorch driver and
  stop at custom autograd functions without `jvp` elsewhere (tad-dftd3,
  tad-libcint, `CoulombMatrixAG`, `RepulsionAG`); `vmap` stops at the cache
  keys (`data_ptr`), batch-size heuristics and data-dependent control flow.
  `torch.compile` fails inside Dynamo.
- **`Node` (tad-mctc 0.9) covers what the plan needs**; two small helpers
  are worth adding in dxtb when containers migrate (section 10). dxtb now
  runs on tad-mctc 0.9.1 and tad-multicharge 0.7.0; tad-dftd3 and tad-dftd4
  are not released for it yet and are bridged by a temporary shim
  (`dxtb/_src/mctc_shim.py`, section 1).

## 1. Environment and commit

| Item | Value |
| --- | --- |
| dxtb | `main` at `3b6a90f` (#274) plus this branch; the status data and references were regenerated on this branch after the move to tad-mctc 0.9.1 |
| Python | 3.11.15 |
| torch | 2.14.0 (PyPI wheel, build `2.14.0+cu130`; run on CPU, no GPU present) |
| tad-* | tad-mctc 0.9.1, tad-multicharge 0.7.0, tad-dftd3 0.6.0 and tad-dftd4 0.8.0 (installed with `--no-deps`, bridged by `dxtb/_src/mctc_shim.py`), tad-libcint 0.3.0 |
| numpy / scipy / pyscf | 1.26.4 / 1.17.1 / 2.14.0 |
| tblite (reference only) | 0.7.0 (PyPI) |
| Hardware | 4 vCPU Intel Xeon @ 2.8 GHz, 15 GB RAM, MKL 2024.2 |

`constraints/baseline.txt` pins these versions; the workflow
`.github/workflows/baseline.yaml` installs from it and runs the suite
(`large`/`slow` nightly). The first baseline was recorded with tad-mctc
0.7.0 and tad-multicharge 0.5.0 on `a132ec8`; moving to tad-mctc 0.9.1
shifts the dxtb energies by up to 4e-7 Eh (the CODATA constants were
refreshed in tad-mctc 0.8: `EV2AU` changes by 8e-9 relative; forces and
charges by about 2e-9). The dxtb references were regenerated and still agree
with tblite within the recorded tolerances. Run the suite with one thread per
xdist worker (`OMP_NUM_THREADS=1`): four workers with four threads each
oversubscribe four cores, which made the suite about ten times slower here.
The plan's code references point to `46af7bc`; `main` has moved on by #268
(CCA ordering), #269 (Fermi smearing), #270 (PyTorch multipole integrals),
#271 (fractional electrons), #272 (implicit solver, `nonpure` removed) and
#274 (smaller test suite). `OverlapAG` and `EFunction` of the plan no
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

824 cells: 10 quantities x up to 5 paths x 2 methods x 2 drivers x 2 SCF
modes x 3 inputs (water; padded water/OH-; two water geometries), each
compared with finite differences of the quantity one order lower. The full
tables are in the appendix; the outcome per cell is in
`test/test_baseline/status/matrix.json` and drives `test_derivatives.py`
(non-passing cells are `xfail(strict=True)`).

Outcome over all 824 cells: 383 pass, 45 wrong value, 395 exception,
1 not finite (unrolled SCF: 221 / 39 / 151 / 1; implicit: 162 / 6 / 244 / 0).
By path and setting:

- **Unrolled SCF (`full`) + autograd, single system: correct to third
  order** for every quantity, both methods and both drivers (deviation from
  finite differences 1e-10 to 7e-7; 0.4 s for forces, 4-7 s for Hessians,
  7-22 s for third order, water). The only exception is the derivative of
  the forces with respect to parameters with libcint (`int1e_rripovlp` not
  available). The `refocc` derivative, wrong between #271 and T0.12, is right
  again (section 7).
- **`functorch` paths** of the Calculator API match autograd for single
  systems with the unrolled SCF; for batches they return cross-system blocks
  (`(nb, nb, ...)`), and `pol_deriv`/`dipole_deriv` fail to reshape.
- **Implicit SCF mode** (`implicit`; the `nonpure` mode was removed in
  #272): with autograd, the status of every cell equals the unrolled SCF
  (single systems: 32 pass, 6 wrong value, 2 exceptions, in both modes), so
  all positional and field derivatives are right up to third order. At
  `a132ec8` the Hessian, the polarizability and the third derivatives were
  wrong (10-40%) or missing. The `functorch` and forward-mode paths do not
  work at all with the implicit mode: all 28-40 cells per input error
  (`requires_grad_()` inside a transformed function, no `vmap` rule,
  `flat_bdims must not be None`).
- **Forward mode** (`jacfwd`, `jvp`) works for the dipole and the
  polarizability (all methods and drivers) and, with GFN2 and the PyTorch
  driver, also for forces, the Hessian and the dipole derivatives. The
  hyperpolarizability, the third derivative and the polarizability
  derivative give wrong values, and GFN1 and libcint stop at custom autograd
  functions without `jvp` (tad-dftd3, tad-libcint, `CoulombMatrixAG`,
  `RepulsionAG`); through the plain `eigh` it gives NaN for degenerate
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
  SCF (autograd and `functorch`).

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
  wrong (forward mode, `jvp` and `vmap` pass the check for the first
  derivative); HOMO-LUMO gap 1e-3 Eh - already first order wrong in reverse
  and forward mode (broadening bias); no vmap rule with Fermi smearing
  (`.item()`).
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
| `hessian(energy)` (`jacfwd(jacrev)`) | error (D3 `jvp`) | error (D3 `jvp`) | **works** | error (libcint `jvp`) |
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
`benchmarks/baseline/results/`: `T0-cpu-4vcpu.json` (first baseline, `a132ec8`
with tad-mctc 0.7.0) and `T0-cpu-4vcpu-mctc0.9.1.json` (this branch on `main`
at `3b6a90f` with tad-mctc 0.9.1; its last entry is a profiled run, whose
stage times are inflated by the profiler). Unrolled SCF, thresholds 1e-8,
PyTorch integral driver, one workload at a time with no other load.

| workload | before (tad-mctc 0.7.0) | now (tad-mctc 0.9.1) |
| --- | --- | --- |
| W1 GFN1, water cluster, 510 atoms: energy + forces (4 threads) | 16.5 s + 25.2 s, 8.3 GB | 14.2 s + 10.6 s, 6.5 GB |
| W2 GFN1, 8 conformers of tmpda (69 atoms), energy + forces, batch vs loop (1 thread) | batch 42.7 s, loop 51.9 s, 7.3 GB | batch 32.5 s, loop 40.7 s, 5.5 GB |
| W2, 4 threads | **hangs** (see below) | batch 17.3 s, loop 31.2 s, 5.5 GB (batch = loop to 0.0) |
| W3 GFN1, training step, 12 molecules (<= 16 atoms), 1 thread | 2.2 s forward + 4.1 s parameter backward, 1.2 GB | 2.5 s + 2.7 s, 1.1 GB; 159 of 2261 leaves get a gradient |
| W3 GFN2, 10 molecules, 1 thread | 10.2 s + 15.8 s, 2.7 GB; 215 of 1402 leaves | 5.5 s + 7.9 s, 1.9 GB; 215 of 1402 leaves |
| W4 GFN1, Hessian, LYS_xao (33 atoms), 4 threads | 58 s, 9.3 GB | 58.6 s, 8.1 GB |
| W5 hyperpolarizability, glycine, 4 threads, GFN1 / GFN2 | 8.8 s, 1.3 GB / 17.3 s, 1.8 GB | 8.4 s, 1.3 GB / 16.6 s, 1.8 GB |

Not repeated (the limits did not change in a way that would move them, and
each takes up to 30 minutes until it runs out of memory): GFN2 for W1 (out of
memory, more than 14 GB), W2 (8 conformers) and W4 (out of memory after 30
minutes); W2 with 20 conformers and W3 with 32 molecules including a 69-atom
system (out of memory, GFN1); the libcint driver (before: W1 12.6 s + 12.5 s,
7.3 GB; GFN2 out of memory). The script defaults (`--nconf 100`, the whole
training set) are far above these sizes; pass `--nconf 8 --nsys 12`.

Stage breakdown (W1, GFN1, `torch.profiler`): in the SCF, `linalg_eigh` is
the largest operation (6.5 s self CPU), followed by matrix products (3.2 s).
Before, the inversion of the Cholesky factor of the overlap, which
`storch.eighb` recomputed in every iteration with a general LU solve (1.2 s),
came after them; the unrolled SCF now computes it once (`l_inv`), and it no
longer appears among the operations. Measured on the new stack, passing
`l_inv` instead of the overlap saves about 20% of the time and 1.5 GB for W1
(energy 18.3 s -> 14.2 s, forces 13.5 s -> 10.6 s, 8.0 GB -> 6.5 GB). The
rest of the improvement over the first baseline (forces 25.2 s) comes from
the move to tad-mctc 0.9.1 and the changes on `main`; it was not attributed
further. The pair loop of the old overlap (plan: `overlap.py:225`) no longer
exists.

Findings:

1. **Memory is the limit, not time.** The unrolled SCF keeps every
   iteration on the graph: 6.5-8 GB for forces of 510 atoms (GFN1) or the
   Hessian of 33 atoms; GFN2 (multipole integrals and potentials) exceeds
   14 GB in the same workloads. Batched forces need roughly the memory of
   one big system of the same total size (8 x 69 atoms: 5.5 GB). This
   matters for E3 (implicit differentiation stores only the fixed point) and
   E6a.
2. **The batched stall with several threads is fixed for the unrolled SCF.**
   Batched `torch.linalg.lu_factor` (behind `linalg.solve`/`inv`) does not
   finish with 4 torch threads for batches of 8 matrices of size 230 (0.12 s
   with one thread), and MKL reports "Parameter 6 was incorrect on entry to
   DLASWP" on the way; `cholesky`, `eigh` and `solve_triangular` are fine.
   Reproduction without dxtb:
   `torch.set_num_threads(4); torch.linalg.solve(L, I)` with `L` of shape
   `(8, 230, 230)` (torch 2.14.0 PyPI wheel, MKL 2024.2; not reproduced on
   other hardware). `storch.eighb` of tad-mctc 0.9.1 still stalls in the same
   way when it gets the overlap (`b`), but takes 0.04 s (forward) and 0.02 s
   (backward) with the inverse Cholesky factor (`l_inv`) computed beforehand
   (which takes 0.01 s). dxtb passes `l_inv` in the unrolled SCF. The
   implicit SCF mode does not use `storch.eighb`; a batched run of 4 x 33
   atoms with 4 threads completes there as well.
3. **Batching pays little on CPU** for conformers: `batch_mode=2` is 20-45%
   faster than a loop (W2: 32.5 s against 40.7 s with one thread, 17.3 s
   against 31.2 s with four), with the same memory as one big system.
4. Integral evaluation is negligible next to the SCF for large systems with
   the new pair builder; `eigh` dominates (relevant for D6).

## 7. Parameter-gradient coverage (T0.9)

`ParamModule` with gradients switched on for every floating point leaf;
energies and projected forces for H2O, CH4, NH4+, NO2, Fe(CO)5, PbH4-BiH3
and Br2-NH3 (halogen bond); for every leaf of the elements present and every
global leaf, the autograd directional derivative is compared with a central
finite difference (`test/test_baseline/params.py`,
`status/param_coverage.json`).

| method | correct gradient | wrong gradient | no effect on the set | gradient path cut | not checked (element absent) |
| --- | --- | --- | --- | --- | --- |
| GFN1 | 99 | 6 | 41 | 1 | 2114 |
| GFN2 | 135 | 6 | 13 | 0 | 1248 |

- **Bug on `main` (since #271), fixed in T0.12:** the six `element.<X>.refocc`
  leaves (H, C, N, O, Br, Fe) of both methods had a wrong gradient (water,
  GFN1: H -0.0036 against -0.2259, O 0.0757 against 0.2335). The reference
  occupation also determines the fractional number of electrons
  (`get_refocc`, `nel`), whose derivative the Fermi occupations dropped in
  wide gaps (see the summary). The table below is the status after the fix.
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
| `api_calculator` | 650 | B1-B8 (calculator API, property functions) |
| `cache` | 119 | B5, B6 (result object, no component caches) |
| `batch_mode` | 1592 | E4, E6 (masked SCF loop, `vmap` batching) |
| `efield` | 769 | B7 (fields as inputs) |
| `integrals` | 1291 | E2, E5 (fixed-shape kernels), C5 |
| `scf` | 2754 | E3, E4 (SCF differentiation, masked loop) |
| `param` | 264 | F1, C5 (parameter storage, nodes) |
| `physics_values` | 1610 | kept; migrate to the T0.2 utility where possible |
| `baseline` | 1171 | Track 0 itself (reference data, status matrix) |

Counts overlap (a test can have several layers); the total is 5971 tests
(1131 `large`, 771 `slow`), 4069 of them in the CI marker set `not large and
not slow`. `batch_mode` and `scf` include the derivative matrix cells. Main
reduced the suite (#274) from 6359 tests since the first baseline.

**Runtime per file** (JUnit report of the CI marker set `not large and
not slow`, including the baseline tests, 4 xdist workers with one thread each;
4069 tests, 2152 s of test time; wall time 10 min on the baseline machine, down
from 4176 tests, 28153 s and 1 h 59 min before: the suite is smaller (#274)
and the workers no longer oversubscribe the cores). The 30 most expensive
files; the full table is in the appendix:

| file | tests | time (s) | markers |
| --- | --- | --- | --- |
| `test/test_baseline/test_derivatives.py` | 158 | 600.8 | baseline |
| `test/test_baseline/test_components.py` | 125 | 233.7 | baseline |
| `test/test_scf/test_implicit_matrix.py` | 94 | 85.5 | scf |
| `test/test_singlepoint/test_grad_pos_withfield.py` | 14 | 75.6 | api_calculator, efield, physics_values |
| `test/test_scf/test_scf.py` | 141 | 75.1 | scf |
| `test/test_properties/test_pol_deriv.py` | 2 | 64.8 | api_calculator, physics_values |
| `test/test_scf/test_full_tracking.py` | 68 | 47.7 | scf |
| `test/test_properties/test_raman.py` | 1 | 47.2 | api_calculator, physics_values |
| `test/test_properties/test_hyperpol.py` | 2 | 47.0 | api_calculator, physics_values |
| `test/test_baseline/test_reference.py` | 117 | 46.8 | baseline |
| `test/test_libcint/test_coeff_grad.py` | 22 | 46.3 | integrals |
| `test/test_singlepoint/test_grad_field.py` | 40 | 40.6 | api_calculator, efield, physics_values |
| `test/test_wavefunction/test_fermi_derivatives.py` | 127 | 37.5 | scf |
| `test/test_scf/test_fermi_derivatives.py` | 9 | 35.8 | scf |
| `test/test_properties/test_forces.py` | 5 | 34.0 | api_calculator, physics_values |
| `test/test_singlepoint/test_grad_fieldgrad.py` | 24 | 33.7 | api_calculator, efield, physics_values |
| `test/test_singlepoint/test_grad_gfn2.py` | 18 | 33.2 | api_calculator, physics_values |
| `test/test_scf/test_scp.py` | 63 | 31.4 | scf |
| `test/test_properties/test_quadrupole.py` | 47 | 27.4 | api_calculator, physics_values |
| `test/test_a_memory_leak/test_scf.py` | 16 | 26.2 | api_calculator, cache |
| `test/test_a_memory_leak/test_higher_deriv.py` | 16 | 23.2 | api_calculator, cache |
| `test/test_scf/test_hess.py` | 4 | 21.8 | scf |
| `test/test_integrals/test_pytorch_forces_fd.py` | 4 | 21.1 | integrals |
| `test/test_properties/test_quadrupole_fieldgrad.py` | 12 | 20.6 | api_calculator, efield, physics_values |
| `test/test_calculator/test_cache/test_properties.py` | 30 | 16.3 | api_calculator, cache |
| `test/test_integrals/test_screening.py` | 47 | 14.5 | integrals |
| `test/test_properties/test_dipole.py` | 24 | 14.3 | api_calculator, physics_values |
| `test/test_scf/test_charge_derivative.py` | 15 | 13.3 | scf |
| `test/test_properties/test_ir.py` | 1 | 12.6 | api_calculator, physics_values |
| `test/test_scf/test_stateless_map.py` | 22 | 11.4 | scf |

**Flaky tests:** none observed in the final run: no failure (3939 passed,
29 skipped, 100 xfailed, 1 xpassed; the xfailed tests are recorded known
issues). Earlier runs showed two harness effects, not test flakiness: with
four threads per xdist worker the runs were about ten times slower, and a
test that aborts its worker (here, an invalid warning filter) makes xdist stop
with an internal error and hide the remaining results. The SCF derivative
tests (`test_scf`, `test_singlepoint/test_grad_*`,
`test_wavefunction/test_fermi_derivatives.py`, `test_scf/test_implicit_matrix.py`)
dominate the runtime besides the baseline itself.


## 9. Open questions for B1 and E0

Findings that the decision notes have to answer. Each points to the
evidence above.

**E0 (SCF differentiation and integrals)**

1. *Implicit differentiation works with autograd, not with transforms.* The
   fixed-point solver of #272 gives correct derivatives to third order
   (3.1), but no `torch.func` or forward-mode transform runs through it
   (`requires_grad_()` inside the transformed function). Either the implicit
   path becomes transform-compatible (differentiate the fixed-point
   condition with `torch.func` rules), or the unrolled SCF is the only path
   for transforms. The cost columns of the matrix give the unrolled baseline
   to beat.
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

The track files describe the code at `46af7bc`. On `main` (`3b6a90f` plus
this branch) some starting points have moved:

| Track item | At `46af7bc` (plan) | Now |
| --- | --- | --- |
| E5 PyTorch multipole integrals | missing; GFN2 needs libcint | **done** (#270): dipole and quadrupole integrals in the PyTorch driver; GFN2 runs without libcint and passes every T0.4 check (third order, forward mode, `vmap`) |
| E1 custom autograd functions | `RepulsionAG`, `CoulombMatrixAG`, `OverlapAG`, `EFunction` | `OverlapAG` and `EFunction` are gone (#268/#270). Remaining in dxtb: `RepulsionAG` (opt-in, unbounded recursion for parameters), `CoulombMatrixAG`. Also without `jvp`/vmap rule and on the default path: `storch.eighb` (tad-mctc), the C6 model of tad-dftd3, the libcint integrals (tad-libcint). The implicit mode no longer uses the xitorch `RootFinder` (#272, `scf/implicit/fixed_point.py`), but no transform runs through it either |
| E2 fixed-shape pair kernels | per-pair Python loop, geometry-dependent `unique_shell_pairs` | pair builder (`impls/pairs.py`) groups by ordered unique-shell-pair class, built from the index helper; Python loop over classes, not pairs; distance screening exists but is opt-in and refuses to run under `torch.compile`. Still to do for E2: setup in B3, scatter per class, the H0 build, the per-entry loop over batch members in `driver.py` |
| TB table, component caches | keyed on values | additionally check gradient-tracking state (T0.5 stopgap; B6 deletes it) |
| TE starting point, eigensolver | not covered | broadened `eighb` wrong at degeneracies (order >= 2) and biased at small gaps (T0.4); the unrolled SCF now computes `L^-1` of the overlap once and passes it to `eighb` (`l_inv`, tad-mctc 0.9.1); the implicit SCF modes still use the xitorch `lsymeig` |
| B2 configuration | mutable | unchanged |
| E0 SCF modes | `full`, `implicit`, `nonpure` | `nonpure` removed (#272); `implicit` has a new fixed-point solver with correct higher derivatives under autograd (3.1) |
| E0 occupations | rounded electron count | fractional electron count that keeps its graph (#271); the derivative with respect to `refocc` was wrong, fixed in T0.12 (section 7) |
| C2-C4 tad-mctc 0.9 | not released | tad-mctc 0.9.1 and tad-multicharge 0.7.0 are in use; tad-dftd3 and tad-dftd4 are bridged by `dxtb/_src/mctc_shim.py` and `dxtb/_src/ncoord/legacy.py` (functional coordination numbers of tad-mctc 0.7), to be removed with their releases |
| T0.6 "cuda" vs "cuda:0" | known issue | fixed (device normalization), also for `device="cpu"` strings |
| TorchScript | present | removed (`set_jit_enabled`, test switch); compiling goes through `torch.compile` (TD) |

## 10. Review of `Node` (tad-mctc 0.9) for the dxtb tracks

`Node`, `ModuleNode` and the tree utilities (`partition`, `combine`,
`stack`, `leaf_paths`) of tad-mctc 0.9.0 were probed against the patterns
the plan needs (probed with 0.9.0, before dxtb moved to 0.9.1; the probes
were not repeated, but `tree/node.py` of 0.9.1 differs from the current
tad-mctc source only in a comment).

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
dxtb now runs on tad-mctc 0.9.1 and tad-multicharge 0.7.0 (it uses
`Structure`, `CNModel` through the EEQ guess, and `EEQModel`). tad-dftd3
0.6.0 and tad-dftd4 0.8.0 pin older tad-mctc releases and are bridged by a
temporary shim until they are released (C2–C4, excluded here). Items 1 and 2
are dxtb-side helpers to add with the first container migrated to `Node`
(C5), not changes to tad-mctc.


## Appendix: generated tables

<!-- generated by `python -m test.test_baseline.tables` -->

#### Derivative status matrix (T0.3)

##### Input: single

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E6 ✓2 | ✓ | E4 ✓4 | E7 ✓1 | ✓ |
| dipole | 1 | ✓ | ✓ | E4 ✓4 | E4 ✓4 | ✓ |
| dE_dparam | 1 | n/a | ✗ | n/a | E6 ✗2 | n/a |
| hessian | 2 | n/a | ✓ | E4 ✓4 | E7 ✓1 | E6 ✓2 |
| polarizability | 2 | ✓ | ✓ | E4 ✓4 | E4 ✓4 | ✓ |
| dipole_deriv | 2 | ✓ | ✓ | E4 ✓4 | E7 ✓1 | ✓ |
| dforces_dparam | 2 | n/a | E4 ✗4 | n/a | E7 ✗1 | n/a |
| third_order | 3 | n/a | ✓ | E7 ✓1 | E7 ✗1 | n/a |
| hyperpolarizability | 3 | n/a | ✓ | E4 ✓4 | E4 ✗4 | ✗2 ✓6 |
| pol_deriv | 3 | n/a | ✓ | E4 ✓4 | E7 ✗1 | ✓ |

##### Input: padded

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E6 ✓2 | E | E6 ✗2 | E7 ✓1 | ✓ |
| dipole | 1 | ✓ | ✓ | E6 ✓2 | E6 NaN1 ✓1 | ✓ |
| hessian | 2 | n/a | E | E | E7 ✓1 | E6 ✓2 |
| polarizability | 2 | ✓ | ✓ | E6 ✓2 | E6 ✓2 | ✓ |
| dipole_deriv | 2 | E | E | E6 ✗2 | E7 ✓1 | ✓ |
| hyperpolarizability | 3 | n/a | ✓ | E6 ✓2 | E6 ✗2 | ✗2 ✓6 |
| pol_deriv | 3 | n/a | E | E6 ✗2 | E7 ✗1 | ✓ |

##### Input: conformer

Each entry aggregates method × driver × SCF mode (✓ pass, ✗ wrong value, E exception, NaN not finite; counts if mixed).

| quantity | order | analytical | autograd | functorch | forward | numerical |
| --- | --- | --- | --- | --- | --- | --- |
| forces | 1 | E6 ✓2 | E | E6 ✗2 | E7 ✓1 | ✓ |
| dipole | 1 | ✓ | ✓ | E6 ✓2 | E6 ✓2 | ✓ |
| hessian | 2 | n/a | E | E | E7 ✓1 | E6 ✓2 |
| polarizability | 2 | ✓ | ✓ | E6 ✓2 | E6 ✓2 | ✓ |
| dipole_deriv | 2 | E | E | E6 ✗2 | E7 ✓1 | ✓ |
| hyperpolarizability | 3 | n/a | ✓ | E6 ✓2 | E6 ✗2 | ✗2 ✓6 |
| pol_deriv | 3 | n/a | E | E6 ✗2 | E7 ✗1 | ✓ |


##### Single system by setting

| method | driver | SCF mode | ✓ | ✗ | E | NaN |
| --- | --- | --- | --- | --- | --- | --- |
| gfn1 | libcint | full | 28 | 2 | 9 | 0 |
| gfn1 | libcint | implicit | 19 | 1 | 19 | 0 |
| gfn1 | pytorch | full | 26 | 5 | 8 | 0 |
| gfn1 | pytorch | implicit | 17 | 2 | 20 | 0 |
| gfn2 | libcint | full | 25 | 3 | 11 | 0 |
| gfn2 | libcint | implicit | 17 | 1 | 21 | 0 |
| gfn2 | pytorch | full | 29 | 7 | 3 | 0 |
| gfn2 | pytorch | implicit | 17 | 2 | 20 | 0 |

##### Cost of the cells

GFN2, PyTorch driver, unrolled SCF, water:

| quantity | path | status | max abs. dev. | wall time (s) | peak RSS (MB) |
| --- | --- | --- | --- | --- | --- |
| forces | analytical | E | – | 0.3 | 699 |
| forces | autograd | ✓ | 1.4e-09 | 0.3 | 700 |
| forces | functorch | ✓ | 1.4e-09 | 0.6 | 718 |
| forces | forward | ✓ | 1.4e-09 | 1.3 | 707 |
| forces | numerical | ✓ | 1.4e-09 | 4.8 | 690 |
| dipole | analytical | ✓ | 5.9e-09 | 0.4 | 694 |
| dipole | autograd | ✓ | 5.9e-09 | 0.3 | 700 |
| dipole | functorch | ✓ | 5.9e-09 | 0.5 | 706 |
| dipole | forward | ✓ | 5.9e-09 | 1.1 | 707 |
| dipole | numerical | ✓ | 5.9e-09 | 1.7 | 690 |
| dE_dparam | autograd | ✗ | 3.6e-01 | 6.6 | 705 |
| dE_dparam | forward | ✗ | 3.6e-01 | 42.6 | 704 |
| hessian | autograd | ✓ | 4.7e-09 | 3.5 | 922 |
| hessian | functorch | ✓ | 4.7e-09 | 1.5 | 781 |
| hessian | forward | ✓ | 4.7e-09 | 3.0 | 718 |
| hessian | numerical | E | – | 0.3 | 699 |
| polarizability | analytical | ✓ | 3.1e-06 | 0.4 | 708 |
| polarizability | autograd | ✓ | 2.4e-08 | 0.8 | 734 |
| polarizability | functorch | ✓ | 2.4e-08 | 0.9 | 725 |
| polarizability | forward | ✓ | 2.4e-08 | 1.7 | 708 |
| polarizability | numerical | ✓ | 1.3e-06 | 1.6 | 690 |
| dipole_deriv | analytical | ✓ | 1.4e-07 | 0.9 | 724 |
| dipole_deriv | autograd | ✓ | 1.4e-09 | 1.1 | 753 |
| dipole_deriv | functorch | ✓ | 1.4e-09 | 0.9 | 746 |
| dipole_deriv | forward | ✓ | 1.4e-09 | 2.3 | 710 |
| dipole_deriv | numerical | ✓ | 1.1e-07 | 5.0 | 690 |
| dforces_dparam | autograd | ✗ | 3.4e-01 | 7.1 | 717 |
| dforces_dparam | forward | E | – | 3.1 | 728 |
| third_order | autograd | ✓ | 5.1e-09 | 1.6 | 740 |
| third_order | functorch | ✓ | 5.1e-09 | 4.9 | 779 |
| third_order | forward | ✗ | 3.4e-01 | 6.8 | 732 |
| hyperpolarizability | autograd | ✓ | 2.0e-07 | 11.9 | 1330 |
| hyperpolarizability | functorch | ✓ | 2.0e-07 | 3.1 | 879 |
| hyperpolarizability | forward | ✗ | 1.1e+02 | 4.0 | 712 |
| hyperpolarizability | numerical | ✗ | 4.1e-02 | 9.6 | 689 |
| pol_deriv | autograd | ✓ | 8.3e-09 | 12.8 | 1438 |
| pol_deriv | functorch | ✓ | 8.3e-09 | 2.0 | 836 |
| pol_deriv | forward | ✗ | 8.9e+00 | 5.6 | 718 |
| pol_deriv | numerical | ✓ | 1.2e-03 | 33.2 | 689 |

##### Distinct outcomes of the cells that do not pass

| cells | status | message | example cell |
| --- | --- | --- | --- |
| 102 | error | NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward m | `dE_dparam-forward-gfn1-libcint-full-single` |
| 54 | error | RuntimeError: You are attempting to call Tensor.requires_grad_() (or perhaps using torch.autograd.functional.* | `dforces_dparam-forward-gfn1-libcint-implicit-single` |
| 52 | error | ValueError: cannot reshape array of size 8 into shape (4,) | `dipole-forward-gfn2-libcint-full-conformer` |
| 36 | error | ValueError: cannot reshape array of size 12 into shape (6,) | `dipole-forward-gfn1-libcint-full-conformer` |
| 32 | error | RuntimeError: shape '[2, 3, 3, 3]' is invalid for input of size 108 | `dipole_deriv-analytical-gfn1-libcint-full-conformer` |
| 32 | error | NotImplementedError: Hessian calculation is not supported in batch mode. Please use the single system mode. | `hessian-autograd-gfn1-libcint-full-conformer` |
| 24 | error | NotImplementedError: The PyTorch integral driver has no analytical overlap gradient. Use autograd on the overl | `forces-analytical-gfn1-pytorch-full-conformer` |
| 16 | error | RuntimeError: grad can be implicitly created only for scalar outputs | `forces-autograd-gfn1-libcint-full-conformer` |
| 16 | error | RuntimeError: shape '[3, 3, 3, 3]' is invalid for input of size 324 | `pol_deriv-autograd-gfn1-libcint-full-conformer` |
| 15 | fail | leaves: element.H.refocc, element.O.refocc | `dE_dparam-autograd-gfn1-libcint-full-single` |
| 12 | error | NotImplementedError: GFN2 not implemented yet. | `forces-analytical-gfn2-libcint-full-conformer` |
| 11 | error | AssertionError: flat_bdims must not be None | `hyperpolarizability-forward-gfn1-libcint-implicit-single` |
| 4 | error | AttributeError: The integral int1e_rripovlp is not available from libcint, please add it | `dforces_dparam-autograd-gfn1-libcint-full-single` |
| 4 | fail | shape (2, 3, 2, 3, 3) != (2, 3, 3, 3) | `dipole_deriv-functorch-gfn1-pytorch-full-conformer` |
| 4 | fail | shape (2, 2, 3, 3) != (2, 3, 3) | `forces-functorch-gfn1-pytorch-full-conformer` |
| 4 | fail | wrong value (max_abs=1.0e+02) | `hyperpolarizability-forward-gfn1-libcint-full-single` |
| 4 | fail | wrong value (max_abs=1.1e+02) | `hyperpolarizability-forward-gfn2-libcint-full-single` |
| 4 | fail | wrong value (max_abs=2.1e-02) | `hyperpolarizability-numerical-gfn2-libcint-full-conformer` |
| 4 | fail | shape (2, 3, 3, 2, 3, 3) != (2, 3, 3, 3, 3) | `pol_deriv-functorch-gfn1-pytorch-full-conformer` |
| 2 | error | RuntimeError: level.has_value() && level <= current_level INTERNAL ASSERT FAILED at "/__w/pytorch/pytorch/aten | `dforces_dparam-forward-gfn2-libcint-full-single` |
| 2 | fail | wrong value (max_abs=4.1e-02) | `hyperpolarizability-numerical-gfn2-libcint-full-single` |
| 2 | error | ValueError: treespec.unflatten(leaves): `leaves` has length 6 but the spec refers to a pytree that holds 7 ite | `pol_deriv-forward-gfn2-libcint-full-single` |
| 2 | fail | wrong value (max_abs=8.9e+00) | `pol_deriv-forward-gfn2-pytorch-full-padded` |
| 1 | nonfinite | non-finite values | `dipole-forward-gfn1-pytorch-full-padded` |
| 1 | fail | wrong value (max_abs=9.4e+00) | `pol_deriv-forward-gfn2-pytorch-full-conformer` |
| 1 | fail | wrong value (max_abs=3.4e-01) | `third_order-forward-gfn2-pytorch-full-single` |

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
| `eigensolver.degenerate_aufbau` | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ |
| `eigensolver.degenerate_fermi` | ✓ | ✗ | ✗ | ✓ | ✓ | E |
| `eigensolver.small_gap_aufbau` | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| `eigensolver.small_gap_fermi` | ✗ | ✗ | ✗ | ✗ | ✗ | E |

Messages of the checks that do not pass:

- GradcheckError: Jacobian mismatch for output 0 with respect to input 0,: `eigensolver.degenerate_aufbau:order2`, `eigensolver.degenerate_aufbau:order3`, `eigensolver.degenerate_fermi:order2`, `eigensolver.degenerate_fermi:order3`, `eigensolver.small_gap_aufbau:order1`, `eigensolver.small_gap_aufbau:order2`, `eigensolver.small_gap_aufbau:order3`, `eigensolver.small_gap_fermi:order1`, `eigensolver.small_gap_fermi:order2`, `eigensolver.small_gap_fermi:order3`
- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD.: `coulomb_matrix_ag.hubbard:forward`, `coulomb_matrix_ag.positions:forward`, `dispersion_d3.positions:forward`, `dispersion_d3.positions:jvp`, `repulsion_ag.arep:forward`, `repulsion_ag.arep:jvp`, `repulsion_ag.positions:forward`, `repulsion_ag.positions:jvp`
- RuntimeError: In order to use an autograd.Function with functorch transforms (vmap, grad, jvp, jacrev, ...), it must override the setup_context static: `coulomb_matrix_ag.hubbard:jvp`, `coulomb_matrix_ag.hubbard:vmap`, `coulomb_matrix_ag.positions:jvp`, `coulomb_matrix_ag.positions:vmap`
- RuntimeError: vmap: It looks like you're calling .item() on a Tensor. We don't support vmap over calling .item() on a Tensor, please try to rewrite wh: `eigensolver.degenerate_fermi:vmap`, `eigensolver.small_gap_fermi:vmap`
- GradcheckError: Jacobian computed with forward mode mismatch for output 0 with respect to input 0,: `eigensolver.small_gap_aufbau:forward`, `eigensolver.small_gap_fermi:forward`
- RuntimeError: vmap: It looks like you're attempting to use a Tensor in some data-dependent control flow. We don't support that yet, please file an iss: `halogen.positions:vmap`, `overlap.slater:vmap`
- RuntimeError: vmap over torch.allclose isn't supported yet. Please file an issue at https://github.com/pytorch/pytorch/issues if you need this feature: `hcore_gfn1.positions:vmap`, `hcore_gfn2.positions:vmap`
- RuntimeError: std::bad_alloc: `repulsion_ag.arep:order1`, `repulsion_ag.arep:order2`
- RuntimeError: output with shape: `aes2.dipoles:vmap`
- ValueError: Batch size mismatch: expected 2, got 3 in `numbers`. The first dimension should be the batch dimension.: `dispersion_d3.positions:vmap`
- AssertionError: jvp differs from finite differences by 1.85e+01: `eigensolver.small_gap_aufbau:jvp`
- AssertionError: jvp differs from finite differences by 4.75e+00: `eigensolver.small_gap_fermi:jvp`
- RuntimeError: Resource temporarily unavailable: `repulsion_ag.arep:order3`

#### Transform status (T0.7)

| check | gfn1-pytorch | gfn1-libcint | gfn2-pytorch | gfn2-libcint |
| --- | --- | --- | --- | --- |
| vmap(energy) single | error | error | error | error |
| vmap(forces) single | error | error | error | error |
| vmap(energy) conformer batch | error | error | error | error |
| jacrev(energy) | works | works | works | works |
| jacfwd(energy) | error | error | works | error |
| hessian(energy) | error | error | works | error |
| torch.compile(energy) | error | error | error | error |
| torch.compile(calc.energy) | error | error | error | error |

Messages:

- RuntimeError: Cannot access data pointer of Tensor that doesn't have storage [dxtb/_src/utils/tensors.py:92] (gfn1-pytorch: vmap(energy) single; gfn1-pytorch: vmap(energy) conformer batch; gfn1-libcint: vmap(energy) single; gfn1-libcint: vmap(energy) conformer batch; gfn2-pytorch: vmap(energy) single; gfn2-pytorch: vmap(energy) conformer batch; gfn2-libcint: vmap(energy) single; gfn2-libcint: vmap(energy) conformer batch)
- ValueError: Batch size mismatch: expected 2, got 3 in `numbers`. The first dimension should be the batch dimension. [tad_dftd3/model/c6.py:456] (gfn1-pytorch: vmap(forces) single; gfn1-libcint: vmap(forces) single)
- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD. [tad_dftd3/model/c6.py:95] (gfn1-pytorch: jacfwd(energy); gfn1-pytorch: hessian(energy); gfn1-libcint: jacfwd(energy); gfn1-libcint: hessian(energy))
- RuntimeError: No wrappers present! (gfn1-pytorch: torch.compile(energy); gfn1-pytorch: torch.compile(calc.energy); gfn1-libcint: torch.compile(energy); gfn1-libcint: torch.compile(calc.energy); gfn2-pytorch: torch.compile(energy); gfn2-pytorch: torch.compile(calc.energy); gfn2-libcint: torch.compile(energy); gfn2-libcint: torch.compile(calc.energy))
- RuntimeError: output with shape [3] doesn't match the broadcast shape [2, 3] [dxtb/_src/calculators/types/energy.py:165] (gfn2-pytorch: vmap(forces) single; gfn2-libcint: vmap(forces) single)
- NotImplementedError: You must implement the jvp function for custom autograd.Function to use it with forward mode AD. [tad_libcint/interface/integrals/int_2c1e.py:353] (gfn2-libcint: jacfwd(energy); gfn2-libcint: hessian(energy))

#### Parameter-gradient coverage (T0.9)

**gfn1**: gradient path cut: 1, no effect on this set: 41, not checked (element not in set): 2114, ok: 99, wrong gradient: 6

| leaf | verdict | energy grad | forces grad |
| --- | --- | --- | --- |
| `dispersion.d3.s9` | gradient path cut | none | none |
| `element.Br.refocc` | wrong gradient | nonzero | nonzero |
| `element.C.refocc` | wrong gradient | nonzero | nonzero |
| `element.Fe.refocc` | wrong gradient | nonzero | nonzero |
| `element.H.refocc` | wrong gradient | nonzero | nonzero |
| `element.N.refocc` | wrong gradient | nonzero | nonzero |
| `element.O.refocc` | wrong gradient | nonzero | nonzero |

No effect on the set (autograd and finite difference zero): `element.Bi.dkernel`, `element.Bi.mprad`, `element.Bi.mpvcn`, `element.Bi.qkernel`, `element.Bi.xbond`, `element.Br.dkernel`, `element.Br.mprad`, `element.Br.mpvcn`, `element.Br.qkernel`, `element.C.dkernel`, `element.C.mprad`, `element.C.mpvcn`, `element.C.qkernel`, `element.C.xbond`, `element.Fe.dkernel`, `element.Fe.mprad`, `element.Fe.mpvcn`, `element.Fe.qkernel`, `element.Fe.xbond`, `element.H.dkernel`, `element.H.mprad`, `element.H.mpvcn`, `element.H.qkernel`, `element.H.xbond`, `element.N.dkernel`, `element.N.mprad`, `element.N.mpvcn`, `element.N.qkernel`, `element.N.xbond`, `element.O.dkernel`, `element.O.mprad`, `element.O.mpvcn`, `element.O.qkernel`, `element.O.xbond`, `element.Pb.dkernel`, `element.Pb.mprad`, `element.Pb.mpvcn`, `element.Pb.qkernel`, `element.Pb.xbond`, `hamiltonian.xtb.kpair.Fe-Fe`, `hamiltonian.xtb.wexp`

**gfn2**: no effect on this set: 13, not checked (element not in set): 1248, ok: 135, wrong gradient: 6

| leaf | verdict | energy grad | forces grad |
| --- | --- | --- | --- |
| `element.Br.refocc` | wrong gradient | nonzero | nonzero |
| `element.C.refocc` | wrong gradient | nonzero | nonzero |
| `element.Fe.refocc` | wrong gradient | nonzero | nonzero |
| `element.H.refocc` | wrong gradient | nonzero | nonzero |
| `element.N.refocc` | wrong gradient | nonzero | nonzero |
| `element.O.refocc` | wrong gradient | nonzero | nonzero |

No effect on the set (autograd and finite difference zero): `element.Bi.mpvcn`, `element.Bi.xbond`, `element.Br.xbond`, `element.C.xbond`, `element.Fe.dkernel`, `element.Fe.mpvcn`, `element.Fe.xbond`, `element.H.xbond`, `element.N.xbond`, `element.O.xbond`, `element.Pb.mpvcn`, `element.Pb.xbond`, `hamiltonian.xtb.kpol`


#### Test-suite inventory (T0.10), all files

| file | tests | time (s) | markers |
| --- | --- | --- | --- |
| `test/test_baseline/test_derivatives.py` | 158 | 600.8 | baseline |
| `test/test_baseline/test_components.py` | 125 | 233.7 | baseline |
| `test/test_scf/test_implicit_matrix.py` | 94 | 85.5 | scf |
| `test/test_singlepoint/test_grad_pos_withfield.py` | 14 | 75.6 | api_calculator, efield, physics_values |
| `test/test_scf/test_scf.py` | 141 | 75.1 | scf |
| `test/test_properties/test_pol_deriv.py` | 2 | 64.8 | api_calculator, physics_values |
| `test/test_scf/test_full_tracking.py` | 68 | 47.7 | scf |
| `test/test_properties/test_raman.py` | 1 | 47.2 | api_calculator, physics_values |
| `test/test_properties/test_hyperpol.py` | 2 | 47.0 | api_calculator, physics_values |
| `test/test_baseline/test_reference.py` | 117 | 46.8 | baseline |
| `test/test_libcint/test_coeff_grad.py` | 22 | 46.3 | integrals |
| `test/test_singlepoint/test_grad_field.py` | 40 | 40.6 | api_calculator, efield, physics_values |
| `test/test_wavefunction/test_fermi_derivatives.py` | 127 | 37.5 | scf |
| `test/test_scf/test_fermi_derivatives.py` | 9 | 35.8 | scf |
| `test/test_properties/test_forces.py` | 5 | 34.0 | api_calculator, physics_values |
| `test/test_singlepoint/test_grad_fieldgrad.py` | 24 | 33.7 | api_calculator, efield, physics_values |
| `test/test_singlepoint/test_grad_gfn2.py` | 18 | 33.2 | api_calculator, physics_values |
| `test/test_scf/test_scp.py` | 63 | 31.4 | scf |
| `test/test_properties/test_quadrupole.py` | 47 | 27.4 | api_calculator, physics_values |
| `test/test_a_memory_leak/test_scf.py` | 16 | 26.2 | api_calculator, cache |
| `test/test_a_memory_leak/test_higher_deriv.py` | 16 | 23.2 | api_calculator, cache |
| `test/test_scf/test_hess.py` | 4 | 21.8 | scf |
| `test/test_integrals/test_pytorch_forces_fd.py` | 4 | 21.1 | integrals |
| `test/test_properties/test_quadrupole_fieldgrad.py` | 12 | 20.6 | api_calculator, efield, physics_values |
| `test/test_calculator/test_cache/test_properties.py` | 30 | 16.3 | api_calculator, cache |
| `test/test_integrals/test_screening.py` | 47 | 14.5 | integrals |
| `test/test_properties/test_dipole.py` | 24 | 14.3 | api_calculator, physics_values |
| `test/test_scf/test_charge_derivative.py` | 15 | 13.3 | scf |
| `test/test_properties/test_ir.py` | 1 | 12.6 | api_calculator, physics_values |
| `test/test_scf/test_stateless_map.py` | 22 | 11.4 | scf |
| `test/test_integrals/test_multipole_matrix.py` | 15 | 11.0 | integrals |
| `test/test_properties/test_dipole_deriv.py` | 2 | 10.7 | api_calculator, physics_values |
| `test/test_singlepoint/test_grad_gfn1.py` | 30 | 10.5 | api_calculator, physics_values |
| `test/test_calculator/test_cache/test_stale.py` | 11 | 9.8 | api_calculator, cache |
| `test/test_external/test_field.py` | 54 | 9.7 | efield |
| `test/test_singlepoint/test_energy.py` | 84 | 9.6 | api_calculator, physics_values |
| `test/test_wavefunction/test_filling.py` | 183 | 9.2 | scf |
| `test/test_hamiltonian/test_gfn1.py` | 23 | 7.9 | integrals, physics_values |
| `test/test_properties/test_pol.py` | 2 | 7.3 | api_calculator, physics_values |
| `test/test_integrals/test_pytorch_gfn2.py` | 7 | 7.2 | integrals |
| `test/test_properties/test_hessian.py` | 1 | 6.9 | api_calculator, physics_values |
| `test/test_overlap/test_uplo.py` | 12 | 6.8 | integrals, physics_values |
| `test/test_properties/test_vibration.py` | 1 | 6.1 | api_calculator, physics_values |
| `test/test_integrals/test_tblite_reference.py` | 12 | 6.1 | integrals |
| `test/test_param/test_module.py` | 35 | 5.5 | param |
| `test/test_scf/test_charged.py` | 34 | 5.4 | scf |
| `test/test_scf/test_elements_gfn2.py` | 14 | 5.2 | scf |
| `test/test_scf/test_fixed_point.py` | 26 | 5.0 | scf |
| `test/test_integrals/test_multipole_kernels.py` | 34 | 5.0 | integrals |
| `test/test_overlap/test_overlap_molecules.py` | 24 | 4.8 | integrals, physics_values |
| `test/test_a_memory_leak/test_repulsion.py` | 4 | 4.7 | api_calculator, cache |
| `test/test_overlap/test_overlap_pairs.py` | 24 | 4.6 | integrals, physics_values |
| `test/test_integrals/test_pyscf_reference.py` | 11 | 4.4 | integrals |
| `test/test_scf/test_padding_ghosts.py` | 4 | 4.4 | scf |
| `test/test_integrals/test_quadrupole_getter.py` | 6 | 4.2 | integrals |
| `test/test_classical/test_dispersion/test_d4sc.py` | 40 | 4.2 | physics_values |
| `test/test_hamiltonian/test_grad_pos.py` | 24 | 3.9 | integrals, physics_values |
| `test/test_classical/test_dispersion/test_d4_2b.py` | 20 | 3.7 | physics_values |
| `test/test_scf/test_elements_gfn1.py` | 14 | 3.7 | scf |
| `test/test_basis/test_export.py` | 176 | 3.7 | integrals |
| `test/test_scf/test_nonselfconsistent.py` | 10 | 3.6 | scf |
| `test/test_integrals/test_driver/test_manager.py` | 19 | 3.6 | integrals |
| `test/test_param/test_param.py` | 21 | 3.5 | param |
| `test/test_hamiltonian/test_gfn2.py` | 16 | 3.4 | integrals, physics_values |
| `test/test_overlap/test_grad_pos.py` | 28 | 3.4 | integrals, physics_values |
| `test/test_libcint/test_gradcheck.py` | 63 | 3.3 | integrals |
| `test/test_config/test_scf_mode_removed.py` | 15 | 3.3 | api_calculator |
| `test/test_libcint/test_multipole.py` | 128 | 3.0 | integrals |
| `test/test_overlap/test_grad.py` | 21 | 2.8 | integrals, physics_values |
| `test/test_integrals/test_general.py` | 16 | 2.4 | integrals |
| `test/test_coulomb/test_es2_general.py` | 14 | 2.4 | physics_values |
| `test/test_integrals/test_multipole_options.py` | 4 | 2.4 | integrals |
| `test/test_scf/test_gfn1_grad.py` | 15 | 2.3 | scf |
| `test/test_coulomb/test_es3_shell.py` | 19 | 2.2 | physics_values |
| `test/test_hamiltonian/test_general.py` | 12 | 2.2 | integrals, physics_values |
| `test/test_classical/test_repulsion/test_energy.py` | 84 | 2.2 | physics_values |
| `test/test_integrals/test_driver/test_pytorch.py` | 14 | 1.9 | integrals |
| `test/test_classical/test_dispersion/test_hess.py` | 4 | 1.9 | physics_values |
| `test/test_classical/test_dispersion/test_general.py` | 10 | 1.8 | physics_values |
| `test/test_classical/test_dispersion/test_d4sc_grad.py` | 10 | 1.8 | physics_values |
| `test/test_calculator/test_cache/test_invalid.py` | 6 | 1.7 | api_calculator, cache |
| `test/test_coulomb/test_es3_general.py` | 11 | 1.7 | physics_values |
| `test/test_classical/test_repulsion/test_grad_param.py` | 20 | 1.7 | param, physics_values |
| `test/test_integrals/test_driver_precision.py` | 7 | 1.7 | integrals |
| `test/test_solvation/test_alpb.py` | 10 | 1.5 | physics_values |
| `test/test_param/test_shared.py` | 7 | 1.5 | param |
| `test/test_integrals/test_wrappers.py` | 7 | 1.5 | integrals |
| `test/test_classical/test_repulsion/test_grad_pos.py` | 48 | 1.5 | physics_values |
| `test/test_classical/test_shortrangebond/test_general.py` | 10 | 1.5 | physics_values |
| `test/test_overlap/test_overlap_atoms.py` | 10 | 1.5 | integrals, physics_values |
| `test/test_wavefunction/test_mulliken.py` | 96 | 1.5 | scf |
| `test/test_classical/test_repulsion/test_general.py` | 9 | 1.5 | physics_values |
| `test/test_basis/test_general.py` | 13 | 1.5 | integrals |
| `test/test_classical/test_halogen/test_grad_param.py` | 12 | 1.4 | param, physics_values |
| `test/test_integrals/test_factory.py` | 16 | 1.4 | integrals |
| `test/test_classical/test_dispersion/test_d4.py` | 4 | 1.4 | physics_values |
| `test/test_libcint/test_overlap.py` | 36 | 1.3 | integrals |
| `test/test_calculator/test_cache/test_integrals.py` | 6 | 1.3 | api_calculator, cache |
| `test/test_solvation/test_grad.py` | 2 | 1.3 | physics_values |
| `test/test_classical/test_ies/test_general.py` | 8 | 1.3 | physics_values |
| `test/test_integrals/test_param_grad.py` | 3 | 1.3 | integrals, param |
| `test/test_classical/test_halogen/test_general.py` | 9 | 1.2 | physics_values |
| `test/test_coulomb/test_es2_atom.py` | 30 | 1.2 | physics_values |
| `test/test_classical/test_shortrangebond/test_energy.py` | 24 | 1.1 | physics_values |
| `test/test_classical/test_halogen/test_grad_pos.py` | 22 | 1.1 | physics_values |
| `test/test_libcint/test_shape.py` | 56 | 1.1 | integrals |
| `test/test_interaction/test_grad.py` | 20 | 1.1 | physics_values |
| `test/test_calculator/test_cache/test_optional.py` | 4 | 1.1 | api_calculator, cache |
| `test/test_calculator/test_device_spec.py` | 3 | 1.1 | api_calculator |
| `test/test_interaction/test_list.py` | 5 | 1.1 | physics_values |
| `test/test_cli/test_driver.py` | 9 | 1.1 | api_calculator |
| `test/test_properties/test_vibration_ref.py` | 1 | 1.0 | api_calculator, physics_values |
| `test/test_integrals/test_libcint.py` | 8 | 1.0 | integrals |
| `test/test_coulomb/test_grad_atom.py` | 12 | 0.9 | physics_values |
| `test/test_classical/test_dispersion/test_d3.py` | 5 | 0.9 | physics_values |
| `test/test_scf/test_warnings_errors.py` | 4 | 0.9 | scf |
| `test/test_libcint/test_symmetry.py` | 24 | 0.9 | integrals |
| `test/test_classical/test_halogen/test_hess.py` | 6 | 0.9 | physics_values |
| `test/test_scf/test_implicit_batch.py` | 2 | 0.9 | batch_mode, scf |
| `test/test_coulomb/test_grad_shell.py` | 16 | 0.9 | physics_values |
| `test/test_indexhelper/test_general.py` | 14 | 0.8 | - |
| `test/test_components/test_cache.py` | 4 | 0.8 | cache |
| `test/test_classical/test_dispersion/test_grad_pos.py` | 18 | 0.8 | physics_values |
| `test/test_calculator/test_dd.py` | 2 | 0.7 | api_calculator |
| `test/test_classical/test_dispersion/test_grad_general.py` | 4 | 0.7 | physics_values |
| `test/test_classical/test_dispersion/test_grad_param.py` | 8 | 0.7 | param, physics_values |
| `test/test_calculator/test_general.py` | 6 | 0.7 | api_calculator |
| `test/test_basis/test_setup.py` | 26 | 0.7 | integrals |
| `test/test_singlepoint/test_hess.py` | 3 | 0.7 | api_calculator, physics_values |
| `test/test_integrals/test_driver/test_factory.py` | 5 | 0.7 | integrals |
| `test/test_coulomb/test_aes2.py` | 4 | 0.6 | physics_values |
| `test/test_classical/test_repulsion/test_hess.py` | 3 | 0.6 | physics_values |
| `test/test_coulomb/test_es2_shell.py` | 30 | 0.6 | physics_values |
| `test/test_scf/test_guess_grad.py` | 40 | 0.6 | scf |
| `test/test_basis/test_normalization.py` | 182 | 0.5 | integrals |
| `test/test_libcint/test_overlap_grad.py` | 20 | 0.5 | integrals |
| `test/test_classical/test_halogen/test_energy.py` | 20 | 0.5 | physics_values |
| `test/test_classical/test_repulsion/test_grad_general.py` | 1 | 0.5 | physics_values |
| `test/test_classical/test_ies/test_energy.py` | 20 | 0.5 | physics_values |
| `test/test_coulomb/test_es3_atom.py` | 27 | 0.4 | physics_values |
| `test/test_interaction/test_cache.py` | 6 | 0.4 | cache |
| `test/test_scf/test_inv_cholesky.py` | 2 | 0.4 | scf |
| `test/test_integrals/test_pytorch.py` | 4 | 0.3 | integrals |
| `test/test_cli/test_entrypoint.py` | 4 | 0.3 | api_calculator |
| `test/test_coulomb/test_grad_atom_param.py` | 12 | 0.3 | param, physics_values |
| `test/test_coulomb/test_grad_shell_param.py` | 8 | 0.2 | param, physics_values |
| `test/test_coulomb/test_grad_atom_pos.py` | 12 | 0.2 | physics_values |
| `test/test_classical/test_dispersion/test_d4_alp.py` | 1 | 0.2 | physics_values |
| `test/test_wavefunction/test_wiberg.py` | 24 | 0.2 | scf |
| `test/test_classical/test_dispersion/test_energy.py` | 2 | 0.1 | physics_values |
| `test/test_scf/test_general.py` | 4 | 0.1 | scf |
| `test/test_hamiltonian/test_base.py` | 2 | 0.1 | integrals, physics_values |
| `test/test_utils/test_eigh.py` | 2 | 0.1 | - |
| `test/test_coulomb/test_aes2_general.py` | 1 | 0.1 | physics_values |
| `test/test_mol/test_external.py` | 14 | 0.1 | - |
| `test/test_solvation/test_born.py` | 29 | 0.1 | physics_values |
| `test/test_singlepoint/test_general.py` | 1 | 0.1 | api_calculator, physics_values |
| `test/test_coulomb/test_grad_shell_pos.py` | 7 | 0.1 | physics_values |
| `test/test_classical/test_halogen/test_grad_general.py` | 1 | 0.1 | physics_values |
| `test/test_loader/test_lazy/test_param.py` | 5 | 0.1 | param |
| `test/test_cli/test_args.py` | 20 | 0.1 | api_calculator |
| `test/test_integrals/test_ao_ordering.py` | 16 | 0.1 | integrals |
| `test/test_integrals/test_types.py` | 3 | 0.1 | integrals |
| `test/test_utils/test_rules.py` | 1 | 0.0 | - |
| `test/test_indexhelper/test_spread_reduce.py` | 10 | 0.0 | - |
| `test/test_indexhelper/test_extra.py` | 7 | 0.0 | - |
| `test/test_scf/test_mixer.py` | 9 | 0.0 | scf |
| `test/test_scf/test_guess.py` | 5 | 0.0 | scf |
| `test/test_interaction/test_potential.py` | 12 | 0.0 | physics_values |
| `test/test_param/test_util.py` | 5 | 0.0 | param |
| `test/test_config/test_main.py` | 11 | 0.0 | api_calculator |
| `test/test_coulomb/test_average.py` | 6 | 0.0 | physics_values |
| `test/test_io/test_outputs.py` | 7 | 0.0 | api_calculator |
| `test/test_utils/test_misc.py` | 8 | 0.0 | - |
| `test/test_utils/test_timer.py` | 7 | 0.0 | - |
| `test/test_config/test_integral.py` | 7 | 0.0 | api_calculator |
| `test/test_indexhelper/test_culling.py` | 2 | 0.0 | batch_mode |
| `test/test_loader/test_lazy/test_attach_var.py` | 4 | 0.0 | param |
| `test/test_external/test_general.py` | 8 | 0.0 | efield |
| `test/test_basis/test_orthogonalize.py` | 2 | 0.0 | integrals |
| `test/test_utils/test_tensors.py` | 6 | 0.0 | - |
| `test/test_config/test_exlibs_available.py` | 3 | 0.0 | api_calculator |
| `test/test_param/test_tensor.py` | 3 | 0.0 | param |
| `test/test_interaction/test_base.py` | 2 | 0.0 | physics_values |
| `test/test_components/test_list.py` | 2 | 0.0 | cache |
| `test/test_loader/test_lazy/test_attach_module.py` | 2 | 0.0 | param |
| `test/test_classical/test_list.py` | 1 | 0.0 | physics_values |
| `test/test_io/test_logging.py` | 1 | 0.0 | api_calculator |
| `test/test_overlap/test_general.py` | 1 | 0.0 | integrals, physics_values |
| `test/test_config/test_export.py` | 1 | 0.0 | api_calculator |
