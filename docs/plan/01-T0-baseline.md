# Track 0: Baseline before any refactor

**Purpose.** Before any structural change, record what dxtb computes today, how its derivatives behave, and where its time goes. Fix only what would corrupt that record. Every later package is judged against this baseline.

**Principle for fixes.** A bug is fixed in this track only if:

1. it would contaminate the reference data or the status matrix, or
2. it gives current users wrong results *and* the fix is a few lines.

Everything else is documented as a known issue and left for the refactor that deletes the affected code.

**Exit criteria for the track.**

- Reference data committed and reproducible (T0.2).
- Derivative status matrix complete (T0.3, T0.4).
- Performance profile recorded (T0.8).
- Parameter-gradient coverage recorded (T0.9).
- Known issues documented (T0.5, T0.6).
- Baseline report written and agreed (T0.11).

---

## T0.1 Secure existing work and the environment

**Goal.** Nothing exists only on a local disk; the test environment can be rebuilt exactly.

**Steps.**

1. Push `fix/type-conversion` (`0bb073e`) and open the PR. Merge after CI passes.
2. Commit a constraints file pinning the versions the suite was run with: `tad-mctc==0.7.0`, `tad-dftd3==0.6.0`, `tad-dftd4==0.8.0`, `tad-multicharge==0.5.0`, `tad-libcint` (record the exact version), `torch` (record version and CUDA build).
3. Add a CI job that installs from the constraints file and runs the full suite.
4. Move the planning documents (this folder and the design review) into the repository, for example under `docs/plan/`.

**Done when.** The `.type()` fix is merged; a clean environment built from the constraints file passes the suite in CI.

**Needs.** None. **Size.** S.

---

## T0.2 Reference data ("golden values")

**Goal.** A frozen set of numbers that every refactor must reproduce, stored independently of the Python API.

**Molecule set** (keep small enough for CI, roughly 2–5 minutes for the whole set):

| Group | Purpose | Examples |
| --- | --- | --- |
| Small organics | basic correctness | H₂O, CH₄, benzene, glycine |
| Charged and open-shell | charge and spin handling | OH⁻, NH₄⁺, a doublet radical |
| Transition metal | d shells | a small Fe or Cu complex |
| Heavy main-group | elements up to 86 | a Pb or Bi compound |
| Padded batch | `batch_mode = 1` | 4 molecules of different sizes |
| Conformer batch | `batch_mode = 2` | 8 conformers of one molecule (~30 atoms) |

**Quantities** for GFN1 and GFN2, float64:

- total energy and per-term energies;
- atomic charges (and atomic dipoles and quadrupoles for GFN2);
- forces;
- Hessian;
- dipole moment, polarizability, first hyperpolarizability;
- dipole derivatives and polarizability derivatives (IR and Raman inputs);
- SCF iteration count (informational, not compared).

**Generation rules.**

- A fresh calculator for every quantity, with the calculator cache disabled. This keeps cache bugs (T0.5) out of the reference.
- Record the integral driver used; generate both PyTorch and libcint values where both are available (GFN1).
- Where `tests/` already compare against the standalone xtb program, record that comparison too.

**Storage.**

- One file per molecule and method (`.pt` or `.npz`) with metadata: dxtb commit, torch version, driver, SCF mode, convergence settings.
- A small comparison utility that takes plain arrays and tolerances, so it survives every API change.

**Tolerances.** Fixed in T0.3 from the observed spread between derivative paths and repeated runs, not guessed.

**Done when.** The reference set is committed, regenerating it on a clean environment reproduces it within tolerance, and the comparison utility is used by a CI test.

**Needs.** T0.1. **Size.** M.

---

## T0.3 Derivative status matrix (end to end)

**Goal.** Know exactly which derivative paths work today, at which order, how accurately and at what cost. This is the baseline for G1 and the input for decision E0.

**Dimensions of the matrix.**

| Dimension | Values |
| --- | --- |
| Method | GFN1, GFN2 |
| Integral driver | PyTorch, libcint |
| SCF mode | full (unrolled), implicit, implicit non-pure |
| Input | single system, padded batch, conformer batch |
| Derivative path | analytical, autograd (reverse), functorch (`jacrev`), forward mode (`jacfwd`/`jvp`), numerical (finite differences) |

**Quantities and comparisons.**

| Order | Quantity | Compare |
| --- | --- | --- |
| 1 | forces | analytical vs autograd vs finite differences of energies |
| 1 | dipole | autograd vs finite differences in the field |
| 1 | dE/dparameter | autograd vs finite differences, per parameter leaf |
| 2 | Hessian | autograd and `jacrev` vs finite differences of forces |
| 2 | polarizability | autograd vs finite differences of dipoles |
| 2 | dipole derivative (IR) | autograd vs finite differences |
| 2 | dForces/dparameter (force matching) | autograd vs finite differences of forces |
| 3 | third-order force constants | nested autograd and forward-over-reverse vs finite differences of Hessians |
| 3 | first hyperpolarizability | autograd vs finite differences of polarizabilities |
| 3 | polarizability derivative (Raman) | autograd vs finite differences |

**Recorded per cell.**

- result: pass, fail (wrong value), error (exception), not finite (NaN/inf);
- maximum absolute and relative deviation;
- wall time and peak memory;
- for finite differences: the step size used and a step-size scan for one molecule, to separate truncation error from real disagreement.

**Additional checks.**

- Padded batches: all derivatives finite up to the highest order that works for single systems.
- SCF convergence threshold: repeat the order-3 cells with a tighter threshold to see whether the unrolled mode's derivatives are limited by convergence.

**Implementation.**

- A parametrised test module, for example `test/test_baseline/test_derivatives.py`, generated from the dimensions above.
- Cells that fail today are marked `xfail(strict=True)` with the observed failure, so any change in status shows up in CI.
- Slow cells (order 3, larger molecules) marked `slow` and run nightly.

**Done when.** Every cell has a recorded status; the matrix is in the baseline report; tolerances for T0.2 are fixed from it.

**Needs.** T0.2. **Size.** L.

---

## T0.4 Component-level derivative checks

**Goal.** Locate failures from T0.3 at the level of individual operations, so TE packages know where to start.

**Targets.**

| Target | Location (at `46af7bc`) |
| --- | --- |
| `RepulsionAG` | `components/classicals/repulsion/rep.py:145` |
| `CoulombMatrixAG` | `components/interactions/coulomb/secondorder.py:930` |
| `OverlapAG` | `integral/driver/pytorch/impls/overlap.py:40` |
| `EFunction` (McMurchie–Davidson) | `integral/driver/pytorch/impls/md/recursion.py:227` |
| every component's energy | with respect to positions, parameters and (where relevant) charges |
| H0 and overlap builds | with respect to positions and basis parameters |
| eigensolver path | small systems with near-degenerate orbitals, with and without smearing |

**Checks per target.**

1. `gradcheck` (first order).
2. `gradgradcheck` (second order). About 24 test files already use it for overlap and Hamiltonian; reuse and extend.
3. Third order: `gradcheck` applied to the function returning second derivatives.
4. Forward mode: `torch.autograd.forward_ad` and `torch.func.jvp`. None of the four custom functions defines `jvp`, so these are expected to fail; record the error.
5. `vmap` over positions: record errors and fallback warnings.

**Done when.** A per-target table (order 1/2/3, forward mode, `vmap`) is in the baseline report, with tests marked as in T0.3.

**Needs.** T0.1. **Size.** M.

---

## T0.5 Cache findings: confirm, then fix only if needed

**Goal.** Confirm or drop the cache bugs found by code review, and protect current users from the confirmed ones.

**Tests (one-off; not maintained after TB removes the caches).**

1. **Stale graph after a no-gradient call.** `get_energy(pos)` without gradients, then `pos.requires_grad_(True)` and `get_forces(pos)`. Compare with a fresh calculator. Suspected cause: the ES2, multipole, ALPB and D4SC caches and the integral driver key on position *values* (`components/base.py`, `cache_is_latest`; `integral/driver/base.py:116`, `is_latest`).
2. **Numerical dipole with the result cache enabled.** `opts={"cache_enabled": True}`, then `dipole_numerical`. Suspected result: zero, because `update_efield` doesn't invalidate the result cache and the energy key covers only the method arguments (`calculators/types/numerical.py`, around line 297; `calculators/types/decorators.py`, `cache`).
3. **Views of a batch.** With the cache enabled: batched energy, then the energy of `positions[0]`. Suspected result: the batched value is returned, because `tensor_id` keys on `data_ptr()` (`utils/tensors.py:51`).

**Actions.**

- If test 1 fails: apply the two-line stopgap. Store `positions.requires_grad` alongside each cached copy and treat a mismatch as a cache miss. It covers the likely trigger at almost no cost; B6 deletes it with the caches.
- If tests 2 or 3 fail: no code fix (caching is opt-in, and B5 deletes the result cache). Add known-issue notes (T0.6).
- In all cases, generate T0.2 data with fresh calculators so the reference is clean.

**Done when.** Each test is recorded as confirmed or dropped; the stopgap is merged if needed; known issues are documented.

**Needs.** T0.1. **Size.** S.

---

## T0.6 Known-issue notes

**Goal.** Tell current users about problems that will be fixed only by the refactor.

Add to the docstrings and a "Known issues" page:

- `Calculator.to(device)` doesn't move all state. Construct the calculator on the target device (`device=`) instead.
- `reset()` and `reset_all()` replace component tensors with detached clones, which cuts gradients to user-supplied tensors (for example, a field tensor with `requires_grad=True`).
- With `cache_enabled=True`, numerical field derivatives and repeated calls on views of a batch can return stale results (if confirmed in T0.5).
- Comparing devices given as `"cuda"` and `"cuda:0"` can raise a device error.

**Done when.** Notes merged and visible in the documentation.

**Needs.** T0.5. **Size.** S.

---

## T0.7 Transform status

**Goal.** Record how far `torch.func` and `torch.compile` get on today's code. Informational only; nothing is fixed here.

**Checks.**

- `vmap` of energy and forces over positions, single system and conformer batch.
- `jacrev`, `jacfwd` and `hessian` of the energy.
- `torch.compile` of an energy call (`fullgraph=False`): number of graph breaks and their causes.
- All runs with `vmap` fallback warnings turned into errors, so silent per-system loops are visible.

**Done when.** A table of what works, what errors and what falls back is in the baseline report.

**Needs.** T0.1. **Size.** S.

---

## T0.8 Performance and memory baseline

**Goal.** Know where time and memory go before deciding on any optimisation. This replaces the earlier D0.

**Workloads.**

| ID | Workload | Why |
| --- | --- | --- |
| W1 | one large molecule (500–1000 atoms): energy and forces | shows `eigh` and integral scaling |
| W2 | conformer ensemble: 100–1000 conformers of 50–100 atoms; `batch_mode = 2` and a plain loop | the G2 case |
| W3 | training-like step: padded batch of 32 mixed molecules; energy, forces, gradient with respect to parameters | the G3 case |
| W4 | Hessian of a medium molecule (~50 atoms) | second-order cost |
| W5 | hyperpolarizability of a small molecule | third-order cost |

**Breakdown per workload.**

- Stages: setup (index helper, basis, driver), integrals, H0, classical terms, SCF (Fock build, `eigh`, mixing), derivative passes.
- Inside the overlap: time spent in the per-pair Python loop at `integral/driver/pytorch/impls/overlap.py:225`, compared with the vectorised `overlap_gto` calls.
- Per-entry Python loops in batched driver setup (`integral/driver/pytorch/driver.py`).

**Settings.** CPU and GPU; PyTorch and libcint drivers (GFN1); GFN2 with libcint (only option today). Tools: `torch.profiler`, dxtb's timers, peak memory via `torch.cuda.max_memory_allocated` and `tracemalloc`.

**Deliverables.** Scripts in `benchmarks/` with fixed inputs, plus a profile summary in the baseline report.

**Done when.** All five workloads profiled on CPU and GPU, with the stage breakdown recorded.

**Needs.** T0.1. **Size.** M.

---

## T0.9 Parameter-gradient coverage

**Goal.** Know which parameters actually influence results today. The baseline for G3.

**Steps.**

1. Build a `ParamModule` for GFN1 and GFN2 and set `requires_grad=True` on every numeric leaf. The default is `False` (`param/module/types.py:97`).
2. Compute energies and forces for a small set covering s, p and d elements.
3. List every leaf whose gradient is `None` or exactly zero.
4. Classify each one: unused by the method (expected), unused because no element in the set uses it (extend the set), or a bug (the parameter reaches the energy as a Python number, or the gradient path is cut).
5. Search the parameter path for `.item()`, `float()` and `tolist()` on parameter tensors.

**Done when.** A coverage table is in the baseline report, and every bug-class entry has an issue.

**Needs.** T0.1. **Size.** S.

---

## T0.10 Test-suite inventory

**Goal.** Know which tests each later package will have to rewrite.

**Steps.**

1. Tag tests with pytest markers by the layer they exercise: `api_calculator`, `cache`, `batch_mode`, `efield`, `integrals`, `scf`, `param`, `physics_values`.
2. Separate tests that assert API behaviour (rewritten by the refactor) from tests that assert physics values (kept, and migrated to the T0.2 utility where possible).
3. Record runtime per test file and known flaky tests.

**Done when.** Markers merged; a table mapping marker to affected work packages is in the baseline report.

**Needs.** T0.1. **Size.** S.

---

## T0.11 Baseline report

**Goal.** One document that records the state of dxtb before the refactor and feeds the decision notes B1 and E0.

**Contents.**

1. Environment and commit.
2. Reference data summary and tolerances (T0.2).
3. Derivative status matrix and component table (T0.3, T0.4).
4. Cache findings and actions taken (T0.5).
5. Transform status (T0.7).
6. Performance profile (T0.8).
7. Parameter-gradient coverage (T0.9).
8. Test inventory (T0.10).
9. Open questions for B1 and E0.

**Done when.** The report is merged and agreed as the reference for all later packages.

**Needs.** T0.2–T0.10. **Size.** S.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| T0.1 | Secure work and environment | none | S |
| T0.2 | Reference data | T0.1 | M |
| T0.3 | Derivative status matrix | T0.2 | L |
| T0.4 | Component-level derivative checks | T0.1 | M |
| T0.5 | Cache findings and stopgap | T0.1 | S |
| T0.6 | Known-issue notes | T0.5 | S |
| T0.7 | Transform status | T0.1 | S |
| T0.8 | Performance baseline | T0.1 | M |
| T0.9 | Parameter-gradient coverage | T0.1 | S |
| T0.10 | Test-suite inventory | T0.1 | S |
| T0.11 | Baseline report | T0.2–T0.10 | S |

Sizes: S up to a few days, M up to two weeks, L more than two weeks.
