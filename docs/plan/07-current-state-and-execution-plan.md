# Current state and execution plan

This document reconciles the original restructuring plan with the actual state of branch `claude/tender-dijkstra-cda42s`.

Review point:

- branch head at the start of the B5.5/E1.1 package: `3d0fb3cc8fbd12573f0fd056dd0609f018016d5f`
- reviewed B5 closure and E1.1 implementation commit: `24cd1d021c15db2e705ebb91b691012d5428c168`
- final status update is recorded in the following plan-only commit

E1.2 review point:

- branch head at the start of E1.2 / PR 7: `62b1e4fdda95ba3237ec64781055d109d2effc25`
- Luna/high-reviewed E1.2 implementation commit: `47c07e02bd906ebc9a92610feaa630908073ee1c`
- E1.2 implementation status was recorded, but closure was subsequently found incomplete; this package resolves the ownership and selection gaps.

This document supersedes package status and ordering in the earlier track files where they conflict. The architectural goals and principles P1-P10 of `00-overview.md` remain authoritative.

---

# 1. Executive status

The single-system B4 integral/H0 core and B5 immutable Result/core singlepoint are complete. Remaining work is scoped to setup separation, component-state removal, transforms, batching, and the downstream tracks listed below.

Current state:

| Package / area | State                           | Notes                                                                                              |
| -------------- | ------------------------------- | -------------------------------------------------------------------------------------------------- |
| T0.1-T0.7      | done                            | Baseline, reference data, derivative and transform characterization exist.                         |
| T0.8 CPU       | done                            | Workloads and memory profile recorded.                                                             |
| T0.8 GPU       | deferred                        | CPU workload and memory profile are done; GPU completion remains explicitly deferred.              |
| T0.9           | done                            | Parameter gradient coverage recorded.                                                              |
| T0.10-T0.12    | done                            | Inventory, report and `refocc` fix exist.                                                          |
| B1             | done                            | Model/System/Result split is implemented; System does not retain the complete Model.                |
| B2             | done                            | Frozen configuration and input validation are implemented.                                          |
| B3             | partial                         | B3a is done, transitional; B3b ES2, ES3, AES2, repulsion, SRB, classical D4, IES, halogen, and D4SC setup separation are done, while ALPB and other required setup/per-call work remain. |
| B4             | done, single-system core        | Pure integral/H0 evaluation is complete; legacy Calculator batching remains until E6.                |
| B5             | done                            | B5.1/B5.2, B5.4, and B5.5 public Result/property boundary are complete.                              |
| B6             | partial                         | B6a ES2/ES3, AES2, repulsion, SRB, classical D4, IES, halogen, and D4SC families are done; D3 remains blocked upstream and ALPB remains open. |
| B7             | not done                        | Explicit field API remains open.                                                                    |
| B8             | not done                        | Remaining later-track work remains open.                                                            |
| C1/C1b         | available upstream              | `Node`, `ModuleNode`, tree utilities already exist in tad-mctc 0.9.1.                              |
| C5-C9          | not done                        | dxtb still heavily uses `TensorLike` and mutable objects.                                          |
| E0             | decision required now           | Baseline gives enough evidence; this must no longer remain an open placeholder.                    |
| E1             | partial                         | E1.1 RepulsionAG and E1.2 CoulombMatrixAG/ES2 setup ownership closure are done. E1.3 D3 forward AD remains blocked upstream. |
| E2             | partial                         | Existing pair builder groups/scatters pairs and supports higher derivatives.                         |
| E3/E4/E6/E7/E8 | not done                        | These form the transform/batching critical path.                                                   |
| E5             | done                            | PyTorch multipole integrals landed upstream and pass the baseline transform checks.                |
| F/D            | not started                     | Correctly downstream.                                                                              |

Do not mark B3 complete until its original P3 requirement is actually true.

---

# 2. Non-negotiable invariants for all further work

Every implementation PR must satisfy these rules. If a convenient implementation violates one of them, do not merge it as temporary architecture unless the temporary code is isolated behind the legacy Calculator adapter and has an explicit deletion package.

## 2.1 Core state

The new core has exactly three persistent levels:

1. `Model`: method parameters, method-level configuration and term definitions.
2. `System`: tensors and structural data derived from `numbers` and the Model's parameter tables.
3. `Result`: values produced by one evaluation.

There is no fourth persistent cache or driver-state layer.

The whole `Model` must not be stored as a child of `System`.

`setup()` gathers tensors from Model parameters. Those gathered tensors already retain their autograd graph back to the Model leaves. Requiring `System.model` for gradient propagation is unnecessary.

This matters for mixed-system batching: stacking Systems must not stack complete copies of all model parameter tables.

## 2.2 Single-system contract

The final core physics functions operate on one padded or unpadded system.

The final `System` must not contain `batch_mode`.

Do not infer batching from `numbers.ndim`, `positions.ndim`, tensor shapes or lengths inside the new core.

Legacy `Calculator` support may continue to use existing batch modes until E6, but keep this behavior in the compatibility layer and do not design new APIs around it.

## 2.3 Device and dtype

The final `System` must not store a `dd` dictionary.

Device and dtype come from its tensors.

No configuration field stores device or dtype.

## 2.4 No hidden mutable values

Do not add:

- caches keyed by values, identity, `data_ptr()`, Python `id`, storage pointers or `requires_grad`;
- mutation-based `reset`, `invalidate`, `update`, `clear` or `is_latest` mechanisms;
- matrices stored on an integral builder after a call;
- mutable external-field components;
- mutable SCF objects that receive the results of an iteration.

A local variable inside one evaluation is fine. A frozen explicit data value passed between functions is fine.

## 2.5 Setup/per-call boundary

Anything that changes tensor shape or constructs structural indices from values belongs in `Model.setup(numbers)`.

Examples that must not occur in the final transformed per-call path:

- `unique`;
- `nonzero`;
- variable-length boolean indexing;
- `.tolist()` or `.item()` used to make structural decisions;
- dynamic pair-list construction;
- shape-based batch guessing.

A fixed Python loop whose iteration count is determined by immutable System structure is acceptable if it works under the required transforms and profiling shows it is not a bottleneck.

## 2.6 Geometry screening

Geometry-dependent screening must not change tensor shape in the transform-critical core.

The current PyTorch pair builder's optional geometry-dependent `screening_threshold`/`select_pairs` mechanism is therefore not part of the strict G1/G2 core path.

It may remain as an explicit eager optimization if clearly documented, but:

- the default transform path computes a fixed structural pair set;
- geometric cutoffs inside that path are masks;
- `vmap`, forward mode and third-order tests use the fixed-shape path.

## 2.7 Autograd functions

Do not introduce a new `torch.autograd.Function` merely for first-order speed.

A custom function is permitted only after all of the following are demonstrated:

- ordinary reverse gradient;
- reverse-over-reverse to third order where applicable;
- `jvp`;
- `jacfwd`;
- `vmap`;
- parameter derivatives;
- padding safety.

Plain PyTorch is the default.

## 2.8 Baseline references

Never regenerate golden reference values merely because a refactor changed them.

A reference update requires an explicit documented physics change and independent justification.

For normal restructuring PRs, a physics-value deviation is a regression.

---

# 3. Immediate stabilization package: R0

Do this before continuing feature refactoring.

This is not another architecture project. It is a short reconciliation and validation package.

## R0.1 Correct plan status

Update the plan so that:

- B3 becomes `partial`, split conceptually into B3a and B3b;
- E5 is marked done;
- C1/C1b are marked available in tad-mctc 0.9.1 instead of future work;
- E2 records the existing pair-builder improvements;
- T0.8 records CPU done / GPU outstanding;
- the obsolete tad-mctc 0.8 -> 0.9 release sequence is not a future dxtb dependency;
- the current tad compatibility shim is listed as a release blocker.

No code change belongs in this subpackage.

## R0.2 Make a clean development environment reproducible

The restructuring branch currently uses:

- tad-mctc 0.9.1;
- tad-multicharge 0.7.0;
- tad-dftd3 0.6.0 and tad-dftd4 0.8.0 installed with `--no-deps`;
- `dxtb._src.mctc_shim`;
- `dxtb._src.ncoord.legacy`.

The baseline workflow knows this, but the ordinary package dependency declaration does not install D3/D4.

Add one documented development-environment mechanism used by tox/CI.

Do not pretend the temporary combination is a releasable dependency set.

A suitable short-term arrangement is:

1. keep D3/D4 out of `install_requires` while their dependency pins conflict;
2. make tox explicitly install them with `pip install --no-deps` before running tests;
3. add a clean-environment import smoke test;
4. add a release check that fails while `_src/mctc_shim.py` exists.

When compatible upstream releases exist:

1. restore D3 and D4 as normal runtime dependencies;
2. remove the no-deps workaround;
3. remove `mctc_shim.py`;
4. remove `ncoord/legacy.py`;
5. run the complete baseline before and after the switch.

Do not publish a restructuring release while normal installation relies on an undocumented `--no-deps` step.

## R0.3 Validate the branch before calling B2/B3 stable

Run at least:

```sh
OMP_NUM_THREADS=1 python -m pytest -vv \
  -m "not large and not slow" \
  -n logical --dist worksteal \
  --random-order-bucket=global test
```

Run the baseline reference tests separately in the pinned environment:

```sh
OMP_NUM_THREADS=1 python -m pytest -vv \
  test/test_baseline/test_reference.py
```

Run the component transform baseline:

```sh
OMP_NUM_THREADS=1 python -m pytest -vv \
  test/test_baseline/test_components.py
```

Run the parameter coverage generator and require no verdict regression:

```sh
python -m test.test_baseline.params
git diff -- test/test_baseline/status/param_coverage.json
```

For structural packages that can affect derivatives, also run the derivative baseline subset appropriate to the changed code.

A package is not "done" merely because adapted tests pass locally in one focused file.

## R0.4 Add architectural guard tests early

Add a small `test/test_rules/` or extend the current rules test with checks that progressively become strict as packages land.

Eventually enforce:

```text
no object.__setattr__ in dxtb
no tensor_id/data_ptr cache key in evaluation code
no CalculatorCache
no Component._cache
no cache_is_latest
no IntDriver._positions
no update_efield
no batch_mode in final System
no TensorLike after C9
```

Do not enable a grep before the corresponding removal package is complete. Add each rule in the PR that removes the old mechanism.

---

# 4. Amend B1 before more Model/System work

The B1 design is mostly sound. Make one architectural amendment.

Replace:

> System keeps a reference to the model's parameters, not a copy, so a gradient through setup reaches the parameter leaves.

with:

> System stores only the gathered/derived tensors and structural data required for evaluation. It does not store the complete Model. Tensors produced by setup retain their autograd relationship to the Model parameter leaves. Training rebuilds System from the current Model for each differentiated loss evaluation.

Consequences:

- remove `System.model` before C8;
- stacking Systems does not duplicate complete parameter tables;
- setup remains the single gateway from model parameters to per-system data;
- after an optimizer update, call `model.setup(numbers)` again rather than expecting a previously created System to reflect new parameter values.

Also replace the phrase "per-call interaction cache" throughout B1 with "per-call interaction data" or "per-call data". It is an explicit value, not a cache.

---

# 5. Reclassify B3

Treat existing B3 as two subpackages.

## B3a Model/System structural split — done, transitional

Existing accomplishments:

- `Model` owns parameters/configuration/extra terms;
- `Model.setup(numbers)` exists;
- `System` exists;
- the old Calculator delegates construction through Model/System;
- classical numbers-only data is created once;
- parameter-gradient coverage is preserved.

Do not keep expanding the legacy fields of the present System.

The following current fields are transitional:

- `model`;
- `batch_mode`;
- `dd`;
- mutable `integrals`;
- mutable component/list objects required only by the old Calculator.

Document them as scheduled for removal.

## B3b Complete setup/per-call separation — open

B3 is finally complete only after:

1. ES2 element/shell hardness data is gathered in setup;
2. ES3 Hubbard derivative data is gathered in setup;
3. repulsion pair parameters and masks are gathered in setup;
4. integral structural data such as the PyTorch pair plan is created in setup;
5. all other numbers-only masks/index mappings needed by per-call kernels are created in setup;
6. no parameter-to-element gather remains in a transform-critical per-call kernel unless it is a fixed-shape tensor gather intentionally retained and tested;
7. System no longer owns a live mutable integral container.

B3b can be completed incrementally through B4/E1/B6. Do not block B5 on every B3b detail; mark B3 complete only when all are finished.

---

# 6. B4: pure integral and H0 construction

This is the next large implementation package.

Do it before Result/cache removal because B5 otherwise has to expose the existing live integral objects.

Split B4 into reviewable PRs.

## B4.1 Introduce immutable integral setup/data types

Create explicit concepts with names that distinguish structure from per-call matrices.

Suggested shape:

```python
@dataclass(frozen=True)
class IntegralSetup:
    # numbers-only structural information
    ...

@dataclass(frozen=True)
class IntegralMatrices:
    overlap: Tensor
    hcore: Tensor
    dipole: Tensor | None
    quadrupole: Tensor | None
```

The exact types may later become `Node`s in C7. Do not wait for C7 to make them immutable.

`IntegralMatrices` must contain tensors only. It must not contain live builder objects.

Do not store a mutable driver in `IntegralMatrices`.

## B4.2 PyTorch driver

Refactor `src/dxtb/_src/integral/driver/pytorch/driver.py`.

Current problematic state includes:

- `_positions`;
- `_positions_single`;
- `_positions_batch`;
- `_basis_batch`;
- `_ihelp_batch`;
- `_plan_batch`;
- setup mutating the driver;
- per-entry loops for current batch modes.

Target:

```python
build_overlap(integral_setup, positions) -> Tensor
build_dipole(integral_setup, positions) -> Tensor
build_quadrupole(integral_setup, positions) -> Tensor
```

The existing `PairPlan` work is useful. Move ownership of the structural plan out of the mutable driver and into setup/System data.

For the single-system core:

- `PairPlan` is made once during `model.setup(numbers)`;
- positions are direct function arguments;
- basis exponent/coefficient tensors are explicit setup data;
- no positions are retained after a call.

Do not solve mixed-composition batching here. E6 handles it by stacking/vmap.

## B4.3 libcint driver

The libcint path may need a local wrapper built from positions.

Its setup becomes:

```python
libcint_data = setup_libcint(system.integral_setup, positions)
overlap = build_overlap_libcint(libcint_data)
```

`libcint_data` exists only for the current call.

Delete or stop using:

- `IntDriver.is_latest`;
- `_positions`;
- `_grad_key`;
- `invalidate`;
- manager-level "skip setup because same positions" logic.

Do not cache a libcint wrapper across differentiated calls.

If libcint cannot support a required transform, retain it as an eager/reference/analytical backend rather than distorting the transform-critical PyTorch path.

## B4.4 Pure matrix transforms

Convert mutation-style operations such as normalization, traceless conversion and origin shifting to functions returning new tensors.

Example shape:

```python
overlap = normalize_overlap(raw_overlap, ...)
quadrupole = traceless(quadrupole)
dipole = shift_origin(dipole, ...)
```

No integral object receives `.matrix = ...`.

## B4.5 H0

Refactor the core Hamiltonian build the same way.

Inputs must be explicit:

- System numbers-only H0 data;
- positions;
- overlap;
- charge if physically required.

Return H0 and reference occupations as values.

Do not hide `refocc` in a mutable H0 integral instance.

## B4.6 Remove I/O from the transform-critical core

Arguments such as `write_overlap`, `write_dipole`, `write_quadrupole` and `write_hcore` do not belong inside the pure evaluation core.

Users can save matrices from `Result`.

Move optional writing to an adapter/CLI layer.

## B4 acceptance

Before merging each B4 sub-PR:

- integral unit tests pass for both drivers where supported;
- T0 component tests for the changed integrals do not regress;
- T0.2 energy/force/reference values do not change;
- reverse derivatives through the PyTorch integrals remain third-order correct;
- `jvp` and `vmap` of the PyTorch kernels remain passing;
- no changed class stores a matrix produced by a call.

B4 is done only when `singlepoint` can obtain all required integral matrices without mutating an object retained by System.

---

# 7. B5: Result and the new singlepoint core

Do this immediately after B4.

## B5.1 Replace the current mutable Result

The current `calculators/result.py` extends `TensorLike`, initializes zeros and is filled field-by-field.

Replace it with an immutable value.

Required fields from B1:

```text
energy
scf
classical
fenergy

charges
multipoles
density
coefficients
emo
occupation
potential
hamiltonian

overlap
hcore
dipole_integrals
quadrupole_integrals

iterations
converged
residual
```

Naming can be adjusted once, here.

Use `energy`, not a mixture of `total`, `get_energy` and cache keys, as the canonical total-energy field.

`iterations`, `converged` and `residual` must be tensors where they are per-system values.

Do not make Result own live `IntegralMatrices` builders.

An immutable nested tensor value is acceptable until C8 converts Result to `Node`.

## B5.2 Create a core evaluation function

Move the physics out of `EnergyCalculator.singlepoint`.

Target shape:

```python
def singlepoint(
    system: System,
    positions: Tensor,
    chrg: Tensor | float = 0,
    spin: Tensor | float | None = None,
    field: Tensor | None = None,
    field_grad: Tensor | None = None,
) -> Result:
    ...
```

`System.singlepoint` is a thin delegate.

No output is stored on System or Model.

## B5.3 Keep the legacy Calculator as an adapter only

During migration, the old Calculator may call the new core.

It must not be the implementation owner.

Avoid a two-way dependency where the new System calls old Calculator methods.

The dependency direction is:

```text
legacy Calculator -> new System/core
```

never:

```text
new System -> legacy Calculator
```

## B5.4 Delete calculator result caching

Delete:

- `CalculatorCache`;
- cache keys;
- `tensor_id` from the evaluation path;
- `@cdec.cache`;
- `ConfigCache`;
- `ConfigCacheStore`;
- `store_*` arguments;
- result reuse based on call argument identities.

A user who wants reuse keeps a `Result`.

## B5.5 Convert old property consumers

Analytical forces and bond orders that genuinely operate from SCF outputs receive a Result explicitly.

Do not read:

```python
calc.cache["overlap"]
calc.cache["charges"]
calc.cache["coefficients"]
...
```

## B5 acceptance

Add a history-independence test:

```python
r1 = system.singlepoint(pos1)
r2 = system.singlepoint(pos2)
r3 = system.singlepoint(pos1)

assert_same(r1, r3)
```

Repeat with:

- same tensor;
- equal-valued new tensor;
- tensor requiring grad;
- prior backward call;
- field/non-field calls.

T0.5's stale-cache failures should become structurally impossible.

---

# 8. E1: eliminate remaining transform-hostile custom derivative shortcuts

E1 can proceed in parallel with B4/B5.

Do not wait until the whole Track B is complete.

## E1.1 `RepulsionAG`

`RepulsionAG` is an opt-in analytical-gradient shortcut and has already been shown to recurse for parameter differentiation.

Remove the custom path unless a current benchmark proves it is required.

Prefer `repulsion_energy` written in ordinary torch operations.

After replacement:

- position first/second/third derivatives;
- parameter derivatives;
- `jvp`;
- `vmap`

must all use the same implementation.

Delete `with_analytical_gradient` if it has no remaining distinct implementation.

## E1.2 `CoulombMatrixAG`

Status: done. `CoulombMatrixAG` is removed; the plain-PyTorch `ES2Setup`/builder is authoritative for exact migrated ES2 terms. B6a removed the ES2/ES3 mutable-object/setup dual authority, resolved custom/additional exact-term selection, and rejects ambiguous duplicate labels. E1 remains partial because the D3 forward-mode work is blocked upstream.

Do not repair the current Function by adding progressively more custom rules.

The shell-resolved path already has a plain-torch implementation.

Make both atom and shell paths use ordinary torch.

While doing this, split numbers-only setup from geometry:

setup data should contain the gathered Hubbard values needed at atom/shell resolution.

The per-call function should conceptually be:

```python
coulomb = build_es2_coulomb(es2_setup, positions)
```

not:

```python
interaction.get_cache(numbers, positions, ihelp)
```

## E1.3 D3 external blocker

GFN1 forward mode currently stops inside tad-dftd3.

Treat this as a required upstream/subproject fix.

Do not hide the problem by excluding dispersion from forward-mode acceptance.

The required tad-dftd3 component tests are:

- reverse third order;
- `jvp`;
- `jacfwd`;
- `vmap`;
- parameter gradient for all active D3 parameters;
- gradient from `s9=0` when the three-body term is made trainable.

Until the dependency satisfies these tests, G1 is not done for GFN1.

## E1 acceptance

Update T0 component status.

No formerly passing component may regress.

The dxtb-local custom functions that exist only for first-order speed should be gone.

---

# 9. B6a: remove persistent component state

Do this after B5, with E1 landing alongside it.

B6 must be split because the old batched SCF still mutates/culls per-call structures.

B6a removes persistence across calls. E4 then removes culling. B6b finishes immutability.

## B6a.1 Classical components

Convert each classical contribution to:

```python
setup_term(model_parameters, numbers, structural_data) -> TermSetup
energy(term_setup, positions, call_inputs...) -> Tensor
```

No term caches its most recent call.

Families and current status:

- repulsion — done;
- D3 — blocked upstream for forward AD;
- D4 — classical D4 done; D4SC remains under interactions;
- halogen — done; fixed-shape setup/kernel removes the data-dependent-control-flow vmap error;
- IES — done;
- short-range bond — done.

Numbers-only tensors belong in setup.

Geometry-dependent quantities are locals.

## B6a.2 Interactions

Do the same for:

- ES2 — done;
- ES3 — done;
- AES2/multipole — done; explicit setup and per-call geometry data pass position/dipole forward AD, JVP, reverse orders 1–3, jacfwd, and vmap;
- D4SC — open;
- ALPB/solvation — open;
- field terms — remain in B7.

For D4SC, for example:

- D4 model/table data from elements belongs in System;
- coordination number and dispersion matrices belong to the call;
- no D4 model object has its `numbers` mutated by SCF culling.

## B6a.3 Delete persistent cache APIs

Delete persistent uses of:

```text
Component._cache
_cachevars
_cachegrad
cache_is_latest
cache_invalidate
cache_is_setup
cache_enable
cache_disable
Component.update
Component.reset
ComponentList.reset_all
Calculator.reset
```

A temporary per-call SCF data object may still exist until E4, but it must not survive the call or be reused by the next evaluation.

## B6a acceptance

Add a test which runs a sequence of calls in random order and compares every output with a fresh System evaluation.

Call history must not affect results or gradients.

---

# 10. B7: fields and field gradients are call inputs

This can proceed after B5 and largely in parallel with B6a.

Add `field` and `field_grad` directly to the core singlepoint signature.

Do not model these as mutable components that are "updated".

Delete:

```text
update_efield
update_efield_grad
requires_efield*
requires_efg*
```

The energy term is a function of the explicit perturbation tensor.

Rewrite:

- dipole;
- polarizability;
- hyperpolarizability;
- quadrupole field response;
- numerical field derivatives

around explicit arguments.

Tests must cover:

```python
field = torch.zeros(3, dtype=torch.float64, requires_grad=True)
energy(system, positions, field=field)
```

and nested differentiation of that call.

`None` may remain the user-facing "term absent" value. Within one `vmap`, every mapped entry must use the same argument tree structure; batched fields are tensors, not a mixture of `None` and tensors.

---

# 11. E0: write the SCF/eigensolver decision now

Do not let a general implementation agent improvise this package.

T0 has uncovered correctness problems which require an explicit mathematical choice.

The decision note must resolve three separate questions.

## E0.1 Canonical differentiation path

Decide which SCF path is authoritative for G1.

Current evidence:

- unrolled SCF is correct through third order for the ordinary single-system reverse-mode baseline;
- implicit SCF now agrees under ordinary autograd;
- current implicit implementation does not compose with `torch.func`/forward mode;
- unrolled SCF has severe memory cost.

Recommended policy:

1. use the functional unrolled implementation as the correctness/reference path;
2. implement the pure fixed-point map in E4;
3. build implicit differentiation only against that pure map;
4. do not keep two independently evolving physics implementations.

The implicit solver is an optimization/differentiation strategy around the same step function, not another SCF physics implementation.

## E0.2 Degenerate eigenspaces

The Lorentzian-broadened eigenvector backward is not acceptable as the final G1 solution.

It is demonstrably:

- wrong from second order onward at exact degeneracy;
- biased even at first order for sufficiently small gaps.

Do not "fix" this by adjusting the broadening epsilon.

The decision note should specify an invariant formulation for quantities required by the SCF, preferably in terms of the density/occupied subspace and finite divided differences of occupations where possible.

Before implementation, preserve minimal reproductions for:

- exact occupied degeneracy;
- small HOMO-LUMO gap;
- benzene;
- Fe(CO)5.

The acceptance criterion is physics correctness, not merely finite derivatives.

## E0.3 Open-shell GFN2 NO2

The unexplained wrong third-order GFN2 NO2 result is a hard G1 blocker.

Before E3 is called done:

1. isolate whether the failure starts in occupation/Fermi response, alpha/beta handling, eigensolver response or another GFN2-only term;
2. add the smallest reproducible test;
3. fix it;
4. remove the corresponding `KNOWN_WRONG` entry only after independent finite-difference agreement.

No implementation agent should weaken tolerances or remove the molecule to obtain a green matrix.

---

# 12. E2: finish the fixed-shape integral path without rewriting working code unnecessarily

The original E2 assumptions are stale.

The current PyTorch pair builder already:

- constructs a structural `PairPlan`;
- groups shell pairs;
- evaluates groups in tensor operations;
- performs out-of-place scatter;
- supports high-order reverse/forward differentiation in the baseline;
- avoids pair-by-pair Python evaluation.

Therefore re-scope E2.

## E2.1 Move structural pair planning into System setup

`PairPlan` is numbers/basis-structure dependent.

It belongs in `model.setup(numbers)`, not lazily in a driver after positions arrive.

Make its tensor-containing classes compatible with the later Node migration.

## E2.2 Remove batched driver construction

Delete the per-entry loop that creates:

- per-entry `IndexHelper`;
- per-entry Basis;
- per-entry PairPlan

inside `IntDriverPytorch.setup`.

The core driver handles one System.

E6 performs batching externally.

## E2.3 Fixed transform path

For the transform path:

- no dynamic screening;
- no pair count based on positions;
- no data-dependent chunk list;
- fixed structural classes.

A static chunk size is acceptable if needed for memory.

## E2.4 Only optimize grouping further if measured

The current builder groups by ordered unique-shell-pair class rather than only angular momentum.

Do not immediately rewrite it to an angular-momentum-only packed primitive representation.

First benchmark:

- integral wall time;
- number of pair classes;
- kernel-launch/dispatch overhead;
- memory.

T0 currently says integral evaluation is negligible relative to the SCF in the large workloads.

If the existing fixed class loop works under `vmap`/compile and remains negligible, preserve it.

This follows P10.

---

# 13. E4: replace the current mutable/culling SCF with a functional fixed-shape step

This is the most important structural package after state removal.

The existing `scf/pure` naming is misleading: `_Data` and iteration functions still mutate shared state.

Do not incrementally patch this object.

Define explicit immutable/static inputs and iteration state.

Suggested concepts:

```python
@dataclass(frozen=True)
class SCFInputs:
    # hcore, overlap, occupations/reference data,
    # interaction setup/per-call data, masks, settings
    ...

@dataclass(frozen=True)
class SCFState:
    charges: ...
    density: ...
    potential: ...
    hamiltonian: ...
    emo: ...
    coefficients: ...
    mixer_state: ...
    residual: ...
    converged: ...
```

Then:

```python
def scf_step(
    state: SCFState,
    inputs: SCFInputs,
) -> SCFState:
    ...
```

No mutation of `inputs`.

No mutation of the incoming state.

## E4.1 Functional mixers

Anderson/Broyden/simple mixing histories are state values.

Refactor mixer classes so that:

```python
new_value, new_mixer_state = mix(value, old_value, mixer_state, settings)
```

No mixer object owns changing history.

## E4.2 No culling

Delete the optimization where converged batch entries are sliced out and all dependent objects are culled/restored.

For the single-system core there is no reason to cull a batch.

For mapped execution, use an active/converged mask and preserve the previous state after convergence.

Do not use tensor-dependent Python `break` in the transform path.

Conceptually:

```python
for _ in range(maxiter):
    candidate = scf_step(state, inputs)
    state = tree_where(active, candidate, state)
    active = active & ~candidate.converged
```

The exact tree masking helper can be optimized later.

The loop trip count is static.

An eager-only outer wrapper may eventually stop early if measurement justifies it, but the transform-critical function remains fixed-shape.

## E4.3 Delete cull/restore after the new loop works

Delete:

- `IndexHelper.cull/restore`;
- cache `cull/restore`;
- `_Data.cull/restore`;
- mixer culling;
- shape recomputation after systems converge.

This completes the prerequisite for C5.

## E4 acceptance

For the reference set:

- same energies and charges;
- same convergence outcomes;
- same or documented iteration counts;
- single-system derivatives unchanged;
- `vmap` can enter the SCF loop without encountering cull/data-dependent shape logic;
- no state object is mutated.

Measure runtime before/after. Fixed-iteration masking may be slower for highly heterogeneous convergence counts; record it rather than reintroducing culling.

---

# 14. B6b: final state purge

After E4 lands, finish B6.

Remove every remaining `cull`, `restore`, mutable per-call container and legacy reset path.

At this point enforce the strong P4 rule:

> no attribute of Model, System, Result, components, integral setup, SCF inputs or SCF state is modified after construction.

The legacy Calculator adapter may maintain its own adapter-local state only if absolutely required for temporary API compatibility. Core evaluation must not observe it.

B6 is only now marked done.

---

# 15. E3: higher-order SCF differentiation

E3 starts only after:

- E0 decision approved;
- E1 transform blockers addressed;
- E4 pure fixed-point map available.

Do not build higher-order implicit differentiation around today's mutable `_Data` closure.

Required implementation structure:

1. one pure SCF fixed-point/step definition;
2. unrolled differentiation as the reference implementation;
3. optional implicit derivative mechanism wrapping that exact definition;
4. explicit tests comparing implicit and unrolled results.

Required test ladder:

1. scalar toy fixed point;
2. small matrix fixed point;
3. water GFN1;
4. water GFN2;
5. charged system;
6. open-shell system;
7. exact-degeneracy regression systems;
8. parameter differentiation;
9. field differentiation;
10. mixed geometry/field/parameter derivatives.

For each relevant case test:

- `grad`;
- `jacrev`;
- `jvp`;
- `jacfwd`;
- second order;
- third order;
- composition with `vmap`.

Do not declare E3 done while the implicit mode silently misses derivatives through non-leaf intermediates.

If a fully transform-compatible implicit method proves too complex, it is acceptable to ship the unrolled transform path first, provided:

- correctness requirements are met;
- the memory limitation is explicitly documented;
- implicit mode is not advertised as satisfying G1;
- work continues against a separately approved E3b package.

---

# 16. C track: migrate only after mutation has been removed

Do not implement another Node base in dxtb.

Use tad-mctc's existing `Node`, `ModuleNode`, `partition`, `combine`, `stack` and tree helpers.

Reframe Track C as dxtb migration plus external dependency cleanup.

## C5 IndexHelper/Basis

Start after E4.

Convert structural tensors to child fields.

No `batch_mode`.

No cull/restore.

No Python per-molecule scalar in context.

All per-molecule counts that must vary between stacked systems are tensors/children.

Add the `in_dims_like(node, **fields)` helper described by T0 when per-field batching is first needed.

## C6 components

Start after B6b/C5.

Most components should now be small immutable term/setup nodes.

Migrate family by family.

Do not migrate obsolete cache classes; delete them.

## C7 integrals/H0

Start after B4/B6b/C5.

Migrate the immutable setup descriptions created in B4.

Do not turn old mutable driver classes into Nodes merely to preserve them.

## C8 Model/System/Result

Before conversion, ensure the dataclass layouts already match the target architecture.

Specifically remove from final System:

```text
model
batch_mode
dd
live mutable integrals
legacy Calculator cache/list state
```

Then convert Model/System/Result to Node.

Required stacking test:

1. build two Systems with different compositions;
2. pad them to one size bucket;
3. stack;
4. confirm tree structures match;
5. confirm model-wide parameter tables were not duplicated into each System;
6. `vmap` one simple energy kernel over them.

Required training test:

```python
params, rest = partition(model, selector)
loss = ...
grads = torch.func.grad(loss)(params)
```

## C9 TensorLike deletion

Only migrate objects that survive the restructure.

Then delete TensorLike/ModuleLike scaffolding.

Do not spend time porting a class to Node if B4/B5/B6 is scheduled to delete the class.

---

# 17. E6a/E6: batching

Begin only when the single-system core is structurally clean.

Prerequisites:

- B6b;
- E2 structural integral plan;
- E4 fixed-shape SCF;
- C8 stackable System;
- transform-critical custom operations solved.

## E6a conformers

The first batching target is the simple case:

```python
system = model.setup(numbers)

energies = torch.func.vmap(
    system.energy,
    in_dims=0,
)(positions_batch)
```

Then:

```python
forces = torch.func.vmap(
    lambda p: dxtb.forces(system, p)
)(positions_batch)
```

Tests:

- compare every mapped result with a Python loop;
- no `(nbatch, nbatch, ...)` cross blocks;
- gradients have exactly one leading sample dimension;
- map fields/charges when they are batched;
- shared scalar charge works when explicitly broadcast before mapping.

## E6b mixed composition

Build Systems independently in setup.

Pad each System to a defined size bucket.

Stack them with `Node.stack`.

Map over `(system_batch, positions_batch)`.

No physics function receives a `batch_mode`.

No integral driver constructs one System per batch entry internally.

## E6c performance gate

Compare with the T0.8 legacy `batch_mode=2` workload.

Acceptance is throughput, not just correctness.

Record:

- wall time;
- peak memory;
- CPU threads;
- GPU memory and throughput once T0.8 GPU is available.

If `vmap` is slower, profile before introducing special batching rules.

---

# 18. E7: padding-safety audit

Do not leave this entirely to the end.

Start adding the audit tests while B4/B6/E4 are being rewritten.

For every singular or potentially singular operation on padded tensors:

1. mask dangerous inputs before division/root/power/eigendecomposition;
2. compute;
3. mask outputs again.

Test up to third order.

Important areas:

- distances;
- reciprocal distances;
- Coulomb matrices;
- coordination numbers;
- overlap/basis operations;
- occupations;
- padded orbitals;
- SCF convergence residuals;
- ML terms later.

The padding test should deliberately use zeros and coincident padded coordinates that would produce NaN without correct masking.

---

# 19. B8: split property API from final derivative validation

Do not make B8 one enormous PR.

## B8a property API

After B7 and the new core exist, implement pure public functions:

```text
energy
forces
hessian
dipole
quadrupole
polarizability
bond_orders
numerical variants
```

Use the best currently valid transform composition.

Delete corresponding Calculator mixin implementations as each function becomes authoritative.

## B8b third-order/mixed functions

After E3:

```text
third_order
hyperpolarizability
dipole_deriv
pol_deriv
IR
Raman
vibration
parameter-force mixed derivatives
```

The implementation is composition of the pure energy function, not a separate mutable pathway.

E8 is the acceptance package proving all baseline cells required by G1.

---

# 20. T0.8 GPU completion

This is not a blocker for B4-B8.

It is a blocker for final E6/D performance conclusions.

Run the same named workloads and record:

- hardware;
- CUDA/PyTorch version;
- driver;
- wall time;
- `torch.cuda.max_memory_allocated`;
- stage timings;
- batch-vs-loop;
- GFN1 and GFN2 where memory permits.

Preserve the CPU baseline unchanged.

Do not retroactively mix GPU numbers into the original CPU files without identifying hardware.

---

# 21. Early training invariant: F0

The full training track can remain downstream, but the architecture needs a tiny training smoke test much earlier.

Add F0 after B5/B6a.

Purpose: catch accidental graph cuts introduced by Model/System separation.

Use a tiny deterministic case, for example water and two or three selected parameters.

Test:

1. partition/select the parameters;
2. construct System inside the loss evaluation from the current parameter tree;
3. evaluate energy against an artificial nearby target;
4. compute `torch.func.grad`;
5. perform several tiny optimizer/functional update steps;
6. verify loss decreases;
7. verify an unselected parameter is bit-identical;
8. compare the initial gradient against T0.9/finite differences.

This is not the final training workflow and does not pull F2 forward.

It is an architecture regression test.

---

# 22. F2 actinides

Before designing parameter defaults, run a machine-readable capability inventory for every Z=89..103.

Do not begin by editing TOML.

Inventory every table or external model whose maximum supported element can limit evaluation:

- xTB element parameters;
- shell structure;
- reference occupations;
- STO-nG/basis data;
- D3 reference data;
- D4 model data;
- covalent/VDW radii;
- electronegativities;
- EEQ/tad-multicharge;
- coordination-number data;
- halogen/other special terms;
- optional libcint/PySCF basis data;
- any hardcoded `MAX_ELEMENT`;
- tests using arrays of length 86/87.

Produce a table:

```text
quantity | current max Z | upstream source | fallback possible? | trainable? | required for GFN1/GFN2
```

F2a decides the structural shell layout only after this audit.

Elements 1-86 get immutable regression checks before any table is extended.

When serializing the trained model, follow the Node serialization restrictions recorded in T0; do not assume default `torch.load(weights_only=True)` accepts arbitrary Nodes.

---

# 23. F3/F4 ML terms

Start only after the physical core has a stable pure interaction interface.

An ML contribution must be energy-defined.

The model provides something equivalent to:

```python
E_ml = term.energy(system_data, positions, charges, ...)
V_ml = dE_ml / d(charges)
```

The Fock/potential contribution comes from the energy derivative.

Do not allow a user module to inject an unconstrained Fock matrix directly as the primary interface.

Use `ModuleNode` plus `torch.func.functional_call`.

Parameters/buffers are pytree children; module architecture is static context.

Require smooth activations. Do not use ReLU for the acceptance example.

Acceptance tests must include:

- finite-difference force agreement;
- parameter gradients;
- `vmap`;
- forward mode;
- third-order total-energy differentiation for the final G1-compatible interface;
- SCF convergence;
- energy/potential consistency.

---

# 24. D performance track

Do not start compile-oriented restructuring before E4/E6.

The order is:

1. establish pure fixed-shape functions;
2. run eager profiles;
3. compile only functions that profile as worthwhile;
4. compare numerical output and derivatives;
5. retain eager as the correctness reference.

Likely compile targets:

- one SCF step;
- the fixed-count SCF loop if supported;
- outer mapped energy/force functions;
- fixed-shape integral class loops if they appear in the profile.

Do not compile inside an autograd callback or custom backward.

Do not add size buckets until mixed-batch E6 is correct and their benefit is measured.

---

# 25. Revised dependency order

The recommended execution order is:

```text
R0
 |
 +--> E0 decision -------------------------------+
 |                                               |
 +--> E1 remaining custom-function cleanup ------+
 |                                               |
 +--> B4 pure integrals --> B5 Result/core ------+--> B6a
 |                           |                   |
 |                           +--> B7 fields      |
 |                                               v
 +--> E2 finish structural integrals            E4 functional SCF
                                                  |
                                                  v
                                                B6b
                                                  |
                         +------------------------+----------------------+
                         |                        |                      |
                         v                        v                      v
                       C5-C8                    E3                     B8a
                         |                        |                      |
                         +----------+-------------+                      |
                                    v                                    |
                                   E6 <----------------------------------+
                                    |
                                    +--> D
                                    |
                                    +--> mixed-batch/training work

E3 + B8a/B7 --> B8b/E8 --> G1 gate
B5/B6a --> F0 training invariant
B6b + C1b + E3 --> F3/F4
C8 + stable training core --> F1/F2
T0.8 GPU --> final E6/D performance gates
```

Parallelism is encouraged only where packages touch distinct concerns.

Do not merge partially overlapping rewrites of the same core files from several agents at once.

---

# 26. PR sequencing for a less capable implementation agent

Use small PRs with explicit mechanical acceptance.

Recommended sequence:

### PR 0: plan and development environment

- plan status corrections;
- tox compatibility installation;
- clean import smoke test;
- no physics changes.

### PR 1: pure overlap construction

- no other integral types;
- remove overlap matrix mutation;
- reference tests.

### PR 2: pure dipole/quadrupole construction

- reuse existing pair builder;
- no batching rewrite.

### PR 3: pure H0 construction and immutable IntegralMatrices

- singlepoint still legacy but consumes returned values.

### PR 4: immutable Result and core singlepoint

- keep old Calculator adapter;
- no property rewrite yet.

### PR 5: delete Calculator result cache

- no component-cache work in the same PR.

### PR 6: remove `RepulsionAG`

- component tests only plus reference suite.

### PR 7: remove `CoulombMatrixAG`; introduce ES2 setup data

- parameter-gradient regression required.

### PR 8+: remove persistent component caches one family at a time

- ES2/ES3;
- classicals;
- D4SC;
- ALPB;
- fields separately in B7.

### PR E4.1: functional mixer state

- preserve old SCF driver initially.

### PR E4.2: functional SCF state/step

- old loop may call it.

### PR E4.3: fixed-shape loop, delete culling

- then delete cull/restore.

### PR C5+: Node migration from leaves upward

- never mix Node migration with physics changes.

### PR E6: `vmap` only after the single-system path passes standalone transform tests.

Each PR description must state:

```text
Plan package:
Old behavior removed:
New invariant:
Reference tests run:
Derivative tests run:
Known failures remaining:
No-reference-change confirmation:
```

---

# 27. Mandatory "do not do" rules for implementation agents

Do not:

1. change tolerances to make a refactor pass;
2. regenerate golden data without an approved physics change;
3. call `.detach()` on a user input or trainable parameter to make a transform work;
4. convert a differentiable tensor to Python with `.item()`, `float()` or `tolist()` inside the per-call physics path;
5. add a cache to recover performance lost by making the code pure;
6. add `object.__setattr__` to mutate a frozen object;
7. add shape-based batch detection to the new core;
8. add another custom autograd Function before demonstrating the required transform rules;
9. keep an old class alive by converting it to Node when its deletion is already planned;
10. refactor physics and migrate containers in the same PR;
11. make libcint limitations constrain the PyTorch transform-critical architecture;
12. optimize pair kernels before profiling the current improved pair builder;
13. hide a known G1 failure behind `xfail` unless it was already in the T0 baseline;
14. remove the benzene, Fe(CO)5 or NO2 regression cases;
15. claim a package done while its stated done criterion is knowingly deferred.

---

# 28. Merge gates

## Structural API gate

Before calling Track B complete:

- B2-B8a complete;
- no calculator/core caches;
- no mutable external fields;
- Result immutable;
- core evaluation is System + explicit inputs -> Result;
- T0.2 reproduced.

## Transform gate

Before claiming G1/G2 architecture complete:

- E0 decisions implemented;
- degenerate eigensolver cases correct;
- NO2 GFN2 third order correct;
- D3 forward-mode blocker fixed;
- E4 functional SCF;
- E3 required derivative modes pass;
- E6 conformer `vmap` passes;
- no cross-system derivative blocks.

## Node gate

Before C9:

- all surviving dxtb persistent objects are Nodes or plain immutable values;
- no cull/restore;
- no TensorLike subclass remains;
- mixed padded Systems stack with identical tree structure.

## Release gate

Do not make a public restructuring release while any of these are true:

- runtime setup requires manual `pip install --no-deps`;
- `mctc_shim.py` is required;
- the new public Model/System layout still contains documented transitional fields;
- branch CI has not passed;
- a package is marked done while acceptance is deferred.

---

# 29. Definition of success against G1-G5

## G1

Success means correctness, not merely that autograd returns a tensor.

All known-wrong baseline cases are resolved, third-order reverse and forward compositions agree with independent checks, and parameter/field/geometry mixed derivatives work.

## G2

Success means the single-system implementation is the only physics implementation.

Batching is composition with `vmap`/stacking, not a parallel batch-mode codebase.

Legacy batch modes are deleted.

## G3

Success means setup is differentiable from parameter tables to energy/forces, training uses functional trees, and extending the element axis does not perturb existing rows.

## G4

Success means the ML term is an energy contribution with its potential derived from that energy and passes the same transform contract as physical terms.

## G5

Success means measured speed/memory improves without creating a second semantics or weakening G1-G4.

---

# 30. Immediate next action

B4's single-system pure integral/H0 core and B5's immutable Result/core singlepoint are complete. Legacy Calculator batching remains scheduled for E6, and B3b remains incomplete until all required structural setup is explicit.

E1.1 (`RepulsionAG`) and E1.2 (`CoulombMatrixAG`/ES2 setup and ownership closure) are done. B6a is partial: ES2/ES3, AES2, repulsion, SRB, classical D4, IES, halogen, and D4SC are complete; D3 remains blocked upstream and ALPB remains open. B3b is partial: ES2, ES3, AES2, repulsion, SRB, classical D4, IES, halogen, and D4SC setup separation are complete, while ALPB and other required setup/per-call separation remain. E1 remains partial because D3 forward-mode support is blocked upstream.

Package record: the IES family and classical D4 compatibility hardening were validated from task base `12d0182d6b0dff7214bddfcfe8f34eb22f8e77e2` and implemented in `635a6bbfbd1dbe07eabce68d46a42e3d03f35de1`. The independent Luna/high review found no blocker or major issue. The classical halogen family was then validated from task base `c71e4a082ddfd76a2683b005fe3f8420553a16fd`; its fixed-shape kernel changed `halogen.positions:vmap` from ERROR to PASS without changing reference values. Two broad-suite seeds each reported four implicit-SCF memory-leak failures; all seven distinct failing nodes reproduced against the task base with the same one-tensor leak signature. The independent Luna/high review found no blocker or major issue. D4SC, D3, and other B6 families were not migrated in the halogen package.

The AES2/multipole package was validated from task base `f999f9fa3986e4bd73da6ad60d572bbb3cfedcde`. Halogen setup masks now honor the configured species lists. AES2 dipole vmap changed from ERROR to PASS, and the new position target passes forward AD, JVP, jacfwd, reverse orders 1–3, and vmap. Reference and parameter verdicts are unchanged. Broad seeds `20261010` and `20261011` showed nondeterministic implicit-SCF memory-leak failures under work stealing; every distinct failing node was reproduced against the task-base snapshot. The only current broad XPASS is the established implicit-matrix case; the base additionally XPASSes the old strict AES2-vmap xfail, as expected. Luna/high review found no blocker or major issue. D4SC, ALPB, fields/B7, E4, and other interaction families were not migrated.

The D4SC package was validated from task base `79d6f1115f19548b47b1d4ed55607d61dc7d172c`. On the task base, a fresh parameter-coverage generation produced `zero/zero` gradients for the five GFN1 xbond entries (`Bi`, `C`, `Fe`, `O`, `Pb`) while the checked-in base status still said `none/none`; the committed `zero` status is stale-baseline catch-up, with the `no effect on this set` verdict unchanged. D4SC now uses frozen `D4SCSetup` and fresh call-local model/CN/dispersion-matrix data; culling no longer mutates the persistent component model. The existing charge transform target remains green, and `d4sc.positions` passes forward AD, JVP, reverse orders 1–3, and vmap. Potential VJP matches its jacfwd result, and same-System neutral/charged/geometry/neutral SCF calls match a fresh neutral evaluation. D4SC references, classical-D4 regressions, component/reference baselines, and parameter verdicts are unchanged. Broad seeds `20261010` and `20261011` each had four implicit-SCF memory-leak failures; the union of eight exact nodes reproduced on the task base with the same one-tensor leak signature. One established implicit-matrix XPASS remained. Luna/high review found no blocker or major issue; it noted that custom external `D4Model` subclasses must accept the standard constructor and have extension state limited to tensors, scalar values, and basic containers, with unsupported state rejected explicitly. ALPB, D3, B7, E4, E6, and other families were not migrated.

The next local package should be selected from the current dependency plan after review; do not begin it automatically. E1.3 remains an upstream dependency task and should resume when the D3 transform blocker is available. The separate E0 eigensolver/implicit-differentiation design still requires technical review before implementation.

