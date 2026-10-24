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

**Current status authority.** Section 1 records current package state, section 25
is the authoritative dependency/order DAG, and section 30 names immediate
actionable work. Track files define package contracts and acceptance criteria.
Historical baseline reports, completed-package implementation plans, and
package-validation records remain evidence of their recorded point in time;
they do not override the current status or ordering above.

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
| B3             | B3a transitional; B3b final complete | B3b-Final.0 inventories System layout and transform-critical gathers; B3b-Final.1 reuses captured backend `CoreIndexData` on eligible functional SCF calls while direct callers retain isolated copies; B3b-Final.2 owns effective dtype in frozen `ExactSystemCore` while packed modes retain their real mode/cache; B3b-Final.2a restores legacy keyword forwarding and validates core/facade setup coherence and floating device/dtype; B3b-Final.3 adds ordered immutable built-in term descriptors and accepts the staged exact/packed compatibility boundary. The Anderson gradient discrepancy matches base and candidate and remains unresolved. B3b is complete at this boundary, not a registered/stackable System Node. |
| B4             | done, single-system core        | Pure integral/H0 evaluation is complete; legacy Calculator batch construction remains until E6broute production migration. |
| B5             | done                            | B5.1/B5.2, B5.4, and B5.5 public Result/property boundary are complete.                              |
| B5.6           | done                            | Public tensor-valued `SCFStatus` schema is complete and verified across the five approved Torch environments; C8a Result pytree migration is unblocked. |
| B6a-SCF-core   | done                            | Bounded state-removal implementation required by accepted E4core; named existing tests provide the evidence listed below. |
| B6a-SCF-validation | done | Seven parameterized family rows plus a separate all-seven integration test pass in all five environments; reused-System state is checked before and after fresh-System setup/evaluation. The intermittent PySCF loader failure family was reproduced on pre-package base `3f41f7f280a3c6df364cf7639360ca602b945118`. Broader B6a remains partial for legacy component/cache API retirement; D3 forward AD is accepted under E1.3. |
| B6             | partial                         | B6a-SCF-core, B6a-SCF-validation, B6b-core, B6a-D3.1, B6a-D3.1a, and B6a-Cache.1 are complete. Exact single-System classical families now own their captured setup values directly; normal setup does not call `ClassicalList.get_cache()`, and no persistent generic built-in cache is stored. `System.classical_cache` is an on-demand compatibility view; packed systems and explicit custom-extension data retain legacy ownership. Single-System setup rejects ambiguous duplicate classical labels without changing direct or packed `ClassicalList` policy. Broader B6a remains open for retirement of legacy Component/Cache lifecycle APIs and the separately owned compatibility purge. C5 is complete and unblocks C6/C7. |
| B6a-Cache.1   | done                            | Exact single-System Repulsion, SRB, IES, Halogen, classical D4, and D3 setups are captured directly through their family builders. The stored generic cache was replaced by explicit custom-extension data and packed-only legacy storage. The `System.classical_cache` compatibility property assembles an ordered view over family Nodes and stored extension data on demand. The System boundary rejects duplicate active labels and multiple exact instances of the singular-setup families; D3 supports multiple identity-associated setups. |
| B7             | done                            | Fields are explicit single-System call inputs; exact field components remain Calculator-only compatibility adapters, and response derivatives are mutation-free. The full original E6a GFN1/GFN2 PyTorch method/component/derivative boundary is now accepted; see [`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md). |
| B8             | not done                        | Remaining later-track work remains open.                                                            |
| C1/C1b         | available upstream              | `Node`, `ModuleNode`, tree utilities already exist in tad-mctc 0.9.1.                              |
| C5-C9          | C5/C6/C7/C8a/C8b complete | C6 is accepted at the exact-core/legacy boundary; public component/list/cache APIs remain compatibility owners for direct and packed routes. C8b.1 registered the production exact core and selected backend, C8b.2 registered the Model parameter tree with functional setup substitution, and C8b.3a/.3b accepted H2/H3 and canonical GFN1/PyTorch H2/H2O complete-core padding/stacking plus mapped Repulsion value/gradient parity. C8b remains complete at its documented structural/P9 pure-kernel boundary. Padded H0/integral numerical evaluation is now separately owned by E6bimpl.0; no mixed-System SCF or production route is accepted yet. Historical numerical failures remain separately tracked, including the ten inherited GFN1 density-parity failures and five GFN2 force/Hessian failures matched against the C8b.3b base. C7.0-C7.4 complete scoped single-System setup Node acceptance. `tad-libcint` 0.3.0 map and CPU-conversion policy passed its prior five-environment C7.4 checks; C8b focused structural tests passed in five CPU configurations. |
| E0             | done — decision fixed           | Physical map `G` is separate from the solver/mixer transition; canonical full-mode convergence tests both raw-residual x norms and excludes `f_atol`. Functional unrolling is the reference path and future implicit differentiation applies to `G(x)-x`. `maxiter`/iteration-count compatibility is characterized in [`E0-decision-note.md`](E0-decision-note.md). |
| E1             | complete                        | E1.1 RepulsionAG and E1.2 CoulombMatrixAG/ES2 setup ownership closure remain accepted. E1.3's official tad-dftd3 0.7.0 integration passes the original numerical/derivative acceptance, including trainable s9=0 and D3-enabled public GFN1 transforms. The 4,990-node non-slow broad selection has 4,886 passed, 64 inherited failures, 30 skipped, 9 xfailed, and 1 documented non-strict XPASS; focused D3/E6a tests pass in five CPU environments. The known wheel metadata mismatch is accepted under the documented official-release `--no-deps` development/test policy; ordinary resolver compatibility is not claimed. See [`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md) for the final audit and decisions. U1/U2 reports are historical only. |
| E2             | partial — E2-core done           | E2-core is the completed single-System fixed-shape `PairPlan` path and is the prerequisite for core consumers. E2-legacy packed-driver per-entry setup cleanup remains owned by E6broute. |
| E3.0            | complete                        | Both GFN2/NO2 endpoints reproduce for PyTorch and libcint; the diagnosis localizes the historical discrepancy to complete finite-tolerance fixed-point response / finite-root trajectory accuracy, without assigning a defective Anderson sub-operation. The inherited frozen-spectral-density third-order diagnostic also had an undefined local `calculator` in both driver rows; it now derives the equivalent effective mixer configuration from the prepared System and selected interactions and verifies matching Fermi settings. |
| E3.1            | complete                        | E3.1a reference gate, dxtb-local invariant spectral density/free energy, and transform-compatible implicit physical-root differentiation are complete. The matrix-free linear-solve relative tolerance floors at ten dtype epsilons while preserving the float64 threshold. |
| E3             | acceptance open                 | E3.1b is complete and E1.3 is now accepted. The E3-specific end-to-end reference/derivative matrix remains to be rerun under its own package gate; this E1/E6a closure does not close E3. E2-core and E4core are complete. |
| E4             | umbrella partial; E4core accepted | E4core (E4.0–E4.5b) is the canonical functional single-System SCF path. E4.6a is blocked on E6broute batch-route retirement; E4.6b is blocked on remaining legacy FULL fallback retirement. |
| E6/E7/E8       | E6a complete; downstream E6/E7/E8 open | E6a.0–E6a.4 satisfy the documented GFN1/GFN2 PyTorch public conformer, Result/status, built-in component, and derivative boundary. The final E1.3/E6a audit and five-environment official-D3 matrix are in [`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md). GFN0 IES/SRB vmap, whole-System libcint transforms, mixed-System batching, and arbitrary custom callback transforms are outside the original E6a contract. E6bcompat-policy is complete in [`E6bcompat-policy.md`](E6bcompat-policy.md). E6bimpl.0 now provides a private mapped GFN1/PyTorch padded H0/integral foundation; full mixed-System SCF remains open. See [`E6bimpl0-padded-numerical-foundation.md`](E6bimpl0-padded-numerical-foundation.md). E6bcompat validates replacements or explicit migrations. E6broute additionally requires batch-critical E7 and E6c1 W2/W3 clearance. Libcint eager/reference compatibility remains separate. |
| E6bimpl        | complete at private implementation boundary | E6bimpl.0/1a/1b-F, 2a/2b-F and 2c accept the private mixed GFN1/GFN2 PyTorch numerical paths. E6bimpl.3 adds ordered retained legacy composition for libcint and custom extensions without changing public Calculator routing. Its 24 focused tests pass in all five CPU profiles; actual CPU libcint calculations, fields, custom contributions/caches, low-level options, error parity, ordering and gradients are covered. The six matched functional SCF failures and interrupted 601-node run remain recorded baseline evidence; no new failure was found in completed selections. E6bcompat is the next differential acceptance package. E6broute remains blocked on E6bcompat, E7 batch safety and E6c1 W2/W3 clearance. See [`E6bimpl3-retained-legacy-composition.md`](E6bimpl3-retained-legacy-composition.md). |
| E5             | done                            | PyTorch multipole integrals landed upstream and pass the baseline transform checks.                |
| F/D            | F0 done; broader track not started | F0 verifies functional parameter updates through fresh Model/System construction. Full training remains downstream. |

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

These purity rules govern the new Model/System/Result core and functional
single-System evaluation path. Legacy adapter objects may temporarily retain
mutable batch/cull/reset machinery only when an explicitly legacy-owned route
still requires it. Such compatibility state must remain outside the core and
must not be observed by core evaluation. This qualification preserves, rather
than weakens, the core invariant.

## 2.2 Single-system contract

The final core physics functions operate on one padded or unpadded system.

The final `System` must not contain `batch_mode`.

Do not infer batching from `numbers.ndim`, `positions.ndim`, tensor shapes or lengths inside the new core.

Legacy `Calculator` support may continue to use existing batch modes until
E6broute retires the production batch route, but keep this behavior in the
compatibility layer and do not design new APIs around it.

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
- E2-core records the completed single-System pair-builder path; E2-legacy
  tracks the remaining packed-driver loop;
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

The baseline workflow knows this, but the ordinary package dependency declaration does not install tad-dftd3/tad-dftd4.

Add one documented development-environment mechanism used by tox/CI.

Do not pretend the temporary combination is a releasable dependency set.

A suitable short-term arrangement is:

1. keep tad-dftd3/tad-dftd4 out of `install_requires` while their dependency pins conflict;
2. make tox and the pinned Track-0 baseline CI explicitly install them with
   `pip install --no-deps` before running tests;
3. add a clean-environment import smoke test;
4. add a release check that fails while `_src/mctc_shim.py` exists.

When compatible upstream releases exist:

1. restore tad-dftd3 and tad-dftd4 as normal runtime dependencies;
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

R0 validation on Python 3.14/Torch 2.14.1 reported 4,715 passed, 31 skipped,
70 XFAIL, 1 XPASS, and two reproducible legacy xitorch memory-leak failures:
`test/test_a_memory_leak/test_scf.py::test_xitorch[True-False-dtype0]` and
`test/test_a_memory_leak/test_scf.py::test_xitorch_pure[False-False-dtype0]`.
The broad run is not fully green and is not evidence of new B2/B3 stability;
their existing statuses and B3 closure criteria are unchanged. The
parameter-coverage generator found no verdict changes, and its generated JSON
was restored to its pre-run workspace contents. The canonical Torch 2.14.0
baseline environment has not been run in this local validation; the 2.14.1
results are secondary evidence only.

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

- `System.model` has been removed; retain this as a completed B3b item;
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

**Completed setup work:**

- [x] ES2 element/shell hardness data;
- [x] ES3 Hubbard derivative data;
- [x] repulsion pair parameters and masks;
- [x] setup separation for AES2, SRB, classical D4, IES, halogen, D4SC,
  and ALPB;
- [x] E2-core setup-owned PyTorch `PairPlan` used by overlap, dipole, and
  quadrupole builders.
- [x] remove the complete `Model` (`System.model`) from the final core System;
- [x] remove the live mutable integral container and establish explicit
  `integral_setup`/`h0_setup` values.

**Remaining B3b contract:**

- [ ] move remaining numbers-only masks/index mappings required by transformed
  kernels into setup;
- [ ] move remaining transform-critical parameter-to-element gathers into
  setup where they are structural setup work; any retained fixed-shape tensor
  gather must be intentionally documented and tested;
- [ ] remove transitional `System.batch_mode` and `System.dd` from the final
  layout before C8b freezes it;
- [x] B6a-D3.1 migrates exact built-in DFT-D3 to explicit setup data and pure
  single-System evaluation while preserving direct and packed legacy routes.
- [x] B6a-Cache.1 constructs exact built-in classical setup values directly
  for eligible single Systems without calling `ClassicalList.get_cache()`;
  core `System.singlepoint` reads family-owned setup Nodes. No persistent
  generic built-in classical cache is retained. The compatibility property
  assembles an ordered view on demand, while extension-only caches are stored
  separately and packed Systems retain their full legacy cache.
- [ ] decide and retire remaining public `System.classical_cache` and
  component/cache lifecycle compatibility only after separately scoped API
  review; do not conflate that retirement with removal of persistent built-in
  cache ownership.

Use existing explicit builders already present for migrated families,
including repulsion, SRB, D3, D4, IES, and halogen. B6a-Cache.1 closes the
generic stored built-in cache mechanism for the single-System core; packed
batches and custom extensions remain explicit compatibility routes. Final
B3b closure still depends on the remaining masks, mappings, gather, and layout
items above. B6b-core and E3.0 remain independent of that closure.
B3a, not full B3, was the prerequisite for B5. Mark B3 complete only when
every remaining item is closed.

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

Do not solve mixed-composition batching here. E6b handles it by stacking and vmap.

## B4.3 libcint driver

Follow the canonical G1 transform-driver contract in
[`04-TE-differentiable-core.md`](04-TE-differentiable-core.md#g1-transform-driver-contract):
PyTorch owns higher-order transform acceptance; libcint remains an
eager/reference/analytical backend within its documented capabilities. This
does not relax libcint value and reference regressions.

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
classical  # tuple[(label, Tensor), ...]; labels are static context
fenergy

charges    # nested ChargeResult pytree
density
coefficients
emo
occupation
potential  # nested PotentialResult pytree
hamiltonian

overlap
hcore
dipole_integrals
quadrupole_integrals
overlap_norm

iterations
converged
residual
status     # SCFStatus | None; B5.6 closes this schema before C8a
```

`multipoles` is a derived property over `charges`, not a separate Result field
or duplicated pytree leaf. `hcore` is the canonical Result field; `h0` is not
used here. `SCFStatus.trajectory_failure` and
`SCFStatus.final_failure` are dynamic integer Tensor leaves whose values vary
by mapped System/conformer. Only the mapping from values to categories is
static public API metadata, defined by `SCFFailureCode` in B5.6. Existing
functional normalization drops this internal status today; B5.6 must carry it
through the common SCF-output representation and `Result.snapshot()` without
exposing `SCFSolveResult` itself.

The B1 names and field/derived-property distinctions above are closed before
C8a; do not change the pytree schema as part of E6a.

Use `energy`, not a mixture of `total`, `get_energy` and cache keys, as the canonical total-energy field.

`iterations`, `converged` and `residual` must be tensors where they are per-system values.

### B5.6 Public solver-status schema

Before C8a freezes Result's pytree layout, add immutable `SCFStatus | None`
to the Result value and `Result.snapshot()`. Functional output carries dynamic
integer-Tensor `trajectory_failure` and `final_failure` leaves; outer
convergence remains the separate `converged` field. `SCFFailureCode`, defined
once in the B5.6 API specification in Track B, is the canonical public
`IntEnum` mapping. The integer tensors are pytree leaves; only their
interpretation is static API metadata. Projection from internal categories
to those public constants must agree exactly. This preserves the eager policy order:
first failure in physical raw-map trajectory, then final-public-spectrum
failure, then outer nonconvergence. Legacy routes may use `status=None`.
Future low-level checks cover zero/finite-temperature success and failure,
each category including invalid-temperature projection, trajectory versus
final failure, valid subsolves with outer nonconvergence, and `maxiter=0`.
`_normalize_scf_output()` must retain the projected public status rather than
discarding it or exposing internal `SCFSolveResult`.

Do not make Result own live `IntegralMatrices` builders.

An immutable nested tensor value is acceptable until C8a gives Result its
transform-compatible pytree representation.

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

Status: done. `CoulombMatrixAG` is removed; the plain-PyTorch `ES2Setup`/builder is authoritative for exact migrated ES2 terms. B6a-SCF-core removed the ES2/ES3 mutable-object/setup dual authority, resolved custom/additional exact-term selection, and rejects ambiguous duplicate labels. E1 is now complete; the DFT-D3 forward-mode dependency is resolved under E1.3.

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

## E1.3 DFT-D3/tad-dftd3 integration

**Status: complete** under the project's documented controlled official-wheel
installation policy. The official PyPI `tad-dftd3==0.7.0` wheel (SHA-256
`44bb0ddec5fdce42f4c50f0f8270a52a996390a25db827ef2e370110bd1bde7e`) passes
the original numerical and derivative acceptance with dxtb's required
`tad-mctc==0.9.1`. The adapter delegates C6, two-body, and ATM physics to the
official release, preserving the legacy shared 50-Bohr dispersion cutoff
and trainable `s9=0` derivative. Exact-System and legacy D3 paths are covered.

The wheel metadata pins mctc 0.8.0, which is incompatible with dxtb's 0.9.1
pin under ordinary resolution. The accepted development/test policy installs
the official wheel with `--no-deps` after dxtb dependencies; the local shim
restores the required import alias. `pip check` is expected to report this
metadata conflict, and ordinary resolver compatibility is not claimed. This
is an accepted distribution-policy limitation, not a numerical blocker. The
retired candidate and wheel are not active dependencies; U1/U2 reports remain
historical.

The official-release D3/E6a.0–E6a.4 acceptance passed 146 tests in each of
the five approved CPU environments. The selected non-slow broad regression
set completed with 4,886 passed, 64 inherited failures, 30 skipped, 9
xfail, and one documented non-strict XPASS. Exact nodes, classifications,
environment, and wheel identity are in
[`E1.3-L2a-hardening.md`](E1.3-L2a-hardening.md),
[`E1.3-L2-official-d3.md`](E1.3-L2-official-d3.md), and the final audit
[`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md).

## E1 acceptance

**Status: complete.** E1.1 RepulsionAG and E1.2 CoulombMatrixAG/ES2 setup
ownership remain accepted; E1.3's official D3 values, parameter gradients,
transforms, and higher-derivative requirements pass. No E1-required
dxtb-local custom derivative workaround was introduced.

---

# 9. B6a: remove persistent component state

Do this after B5, with E1 landing alongside it.

B6 must be split because the old batched SCF still mutates/culls per-call
structures. B6a removes persistent component state. E4core provides an immutable
fixed-shape single-System solver, but does not delete culling from the legacy
batch route. B6b-core closes purity for the new core; repository-wide legacy
compatibility cleanup is a separate, later milestone after fallback migration.

## B6a-SCF-core implementation scope and DFT-D3 source audit

E4core's formal B6 prerequisite is the completed `B6a-SCF-core` subset: state
removal for self-consistent interaction/setup data used by `SCFInputs` and the
functional raw map. It excludes classical terms that are added outside the
SCF result. This is the state-removal contract E4core actually consumes; it
does not imply that broader B6a or B6 is complete.

Source audit of DFT-D3: `System` stores a `DispersionD3` component (with
`numbers`, parameters and charge configuration) plus a numbers-only
`DispersionD3Cache`. `System.singlepoint()` supplies that stored cache to
`DispersionD3.get_energy()`. The method reads the component/cache and computes
coordination numbers, reference weights, C6 values and energy into locals; it
does not write per-call values back to the component or cache. Those mutable,
legacy-shaped setup/cache objects still violate the broader B6a setup/value
target and remain there for later migration. The forward-mode failure is a
separate E1.3 tad-dftd3 custom-operation blocker. Classical DFT-D3 energy is
added outside `SCFInputs`/the self-consistent map, so it is not an E4core
SCF-state prerequisite. Thus the current outer `System.singlepoint()` path
operationally reads the stored DFT-D3 component/cache, but the SCF inputs and
fixed-point map do not own or consume it. E3.0 excludes DFT-D3 rather than
treating its unrelated outer-energy transform as a prerequisite.

### B6a-SCF-core mapped executable evidence

`B6a-SCF-core` is the bounded state-removal implementation consumed by the
already accepted E4core. The focused validation test below closes the
retrospective seven-interaction matrix; the family-specific tests provide
additional direct and physical-path coverage.

| Interaction/data | Existing executable tests | What they establish | Focused matrix coverage |
| --- | --- | --- | --- |
| ES2 | `test/test_calculator/test_cache/test_stale.py::test_es2_es3_reused_system_random_call_history`; `test/test_baseline/test_components.py::test_component` for `es2.positions`/`es2.charges`; focused matrix below | Reused Calculator outputs/forces are history independent; derivative targets are exercised | Fresh `_interaction_data`, repeated-call identity/value, A/B/A outputs and gradients, fresh-System parity, and setup/component snapshots pass |
| ES3 | `test/test_calculator/test_cache/test_stale.py::test_es2_es3_reused_system_random_call_history`; `test/test_baseline/test_components.py::test_component` for `es3.charges`; focused matrix below | Reused System outputs/forces and component charge derivatives are exercised | Setup-derived call data is geometry-independent (explicit N/A); call-local freshness, stable ES3 values, A/B/A output/gradient parity, and snapshots pass |
| AES2 | `test/test_baseline/test_components.py::test_component` for `aes2.positions`/`aes2.dipoles`; `test/test_scf/test_functional_map.py::test_map_is_history_independent_and_repeated_backward_is_fresh`; focused matrix below | Component derivative targets and generic functional-map history/fresh-backward behavior | Fresh call data, geometry-sensitive data, A/B/A outputs and gradients, fresh-System parity, and snapshots pass |
| D4SC | `test/test_baseline/test_components.py::test_component` for `d4sc.positions`/`d4sc.charges`; `test/test_classical/test_dispersion/test_d4sc_setup.py::test_direct_get_cache_is_fresh_and_nonpersistent`, `test_setup_is_frozen_and_contains_only_static_data`, `test_d4sc_charge_and_geometry_history_is_independent`, `test_system_scf_charge_history_is_independent`, and `test_stale_component_mutation_does_not_change_existing_setup`; focused matrix below | Component derivatives, direct fresh-cache/setup immutability, A/B/A history, System SCF history, and stale-component isolation | The focused matrix adds shared call-data identity/value checks, geometry A/B/A output/gradient parity, fresh-System parity, and snapshots |
| GeneralizedBorn | `test/test_solvation/test_setup.py::test_setup_is_frozen_and_call_data_is_fresh`, `test_position_transforms_rebuild_call_data`, `test_geometry_charge_and_backward_history_is_independent`, `test_core_repeated_force_matches_fresh_system`; `test/test_solvation/test_grad.py::test_gb_scf_grad`; `test/test_solvation/test_alpb.py::test_gb_scf`; focused matrix below | Frozen setup/fresh builder data, geometry/charge history, fresh-System force parity, and physical ALPB energy/force regressions | The focused matrix adds combined System A/B/A output/gradient parity with setup/component snapshots. No `test_baseline/test_reference.py` case is an ALPB reference |
| ElectricField | `test/test_calculator/test_field_inputs.py::test_system_field_inputs_are_fresh_and_history_independent`; `test/test_scf/test_functional_map.py::test_explicit_field_raw_map_matches_full_engine`; focused matrix below | Field call-data freshness/history and raw-map parity | Field and geometry A/B/A outputs/gradients, fresh-System parity, and setup/component snapshots pass |
| ElectricFieldGrad | `test/test_calculator/test_field_inputs.py::test_system_field_inputs_are_fresh_and_history_independent`, `test_field_call_data_vmap_matches_loop`, `test_system_field_jvp_and_jacfwd_match_reverse`, `test_field_gradient_geometry_mixed_derivative`; focused matrix below | Field-gradient call-data freshness, mapped builder parity, and derivative connectivity | Field-gradient and geometry A/B/A outputs/gradients, fresh-System parity, and setup/component snapshots pass |
| Shared functional inputs | `test/test_scf/test_functional_map.py::test_map_inputs_are_not_mutated_and_values_are_frozen`; `test_functional_eligibility_rejects_custom_mutable_interactions`; `test_functional_map_has_no_legacy_stateful_hooks`; `test/test_scf/test_functional_eager_compatibility.py::test_custom_interaction_routes_to_legacy_and_keeps_extension_behavior`; focused matrix below | Functional-input immutability, custom-interaction exclusion and legacy routing, and absence of legacy stateful hooks | Exact seven-family whitelist, call-local data, unchanged persistent setup/component state, and absence of outer DFT-D3 in the SCF interaction inputs pass |

`test/test_scf/test_functional_map.py::test_b6a_functional_interaction_boundary_is_history_independent`
remains the all-seven integration stress test. The actual matrix is
`test/test_scf/test_functional_map.py::test_b6a_functional_interaction_family_matrix`,
parameterized independently for ES2, ES3, AES2, D4SC, GeneralizedBorn,
ElectricField, and ElectricFieldGrad. Each row isolates its interaction in
`setup_scf_inputs`/`scf_map`, isolates and tests its input-dependency behavior,
and compares raw SCFMapResult tensor leaves (not only a scalar norm) across
A, B, A-again, and fresh-System A. A deterministic nonuniform projection is
used only for gradient checks. ElectricField and ElectricFieldGrad each have
separate field-only and geometry-only perturbations; ES3 geometry-dependent
data is explicitly N/A and its setup-derived tensor is stable. Call-data/cache
objects are pairwise fresh across A, B, A-again, and fresh-System A, except
that ES3 may share its documented setup-derived tensor leaf. Persistent
System/setup/component objects and tensor leaves are checked for identity,
value, and `requires_grad` preservation. The test also imports the canonical
internal whitelist and requires exactly the seven intended exact classes;
it checks custom-extension rejection and verifies that an outer DFT-D3
component is absent from functional SCF inputs. The final focused matrix
passed all seven isolated rows and the integration test on Python 3.10/Torch
2.6, Python 3.11/Torch 2.8, Python 3.12/Torch 2.11, Python 3.12/Torch 2.13,
and Python 3.14/Torch 2.14.1.

The acceptance command set is recorded explicitly: `test/test_scf/test_functional_map.py`,
`test/test_scf/test_functional_eager_compatibility.py`,
`test/test_calculator/test_singlepoint_core.py`,
`test/test_calculator/test_field_inputs.py`,
`test/test_coulomb/test_es2_pure.py` plus the ES2 atom/general/pure/shell
coverage, `test/test_coulomb/test_es3_setup.py`,
`test/test_coulomb/test_aes2_setup.py`,
`test/test_classical/test_dispersion/test_d4sc_setup.py` and the D4SC SCF/grad
tests, `test/test_solvation/test_setup.py`, `test/test_solvation/test_grad.py`,
`test/test_solvation/test_alpb.py`, `test/test_interaction/test_list.py`, the
custom-extension cases in `test/test_scf/test_functional_eager_compatibility.py`,
the component baseline, and the NO2 reference subset. The SCF matrix exercises
the active GFN2 ES2/ES3 setup instances. Separate setup suites cover atom- and
shell-resolved builder branches and parameter connectivity: `test_es2_pure.py`
includes `shell_resolved=False/True`, while `test_es3_setup.py` covers
atom/shell resolution and GFN2 shell-scale gradients. There is no dedicated
ES2 setup file. The seven-row parameterized family matrix passed, and the
separate all-seven integration stress test passed, on each of the five
supported environments (8 passes total per focused selection). The focused
command template used in each supported environment was:

```bash
python -m pytest -vv -n logical --dist worksteal -rxX \
  --junitxml=/tmp/b6a-presnapshot-<environment>.xml \
  test/test_scf/test_functional_map.py \
  -k 'test_b6a_functional_interaction_family_matrix or test_b6a_functional_interaction_boundary_is_history_independent'
```

The primary-environment supporting rerun used the worktree based on candidate
SHA `28e85ee8a8cb6ad944151393aa743181c851216c` plus the pre-fresh assertion
change in this follow-up. It ran on Python 3.14/Torch 2.14.1 with this exact
command:

```bash
set -o pipefail
export OMP_NUM_THREADS=1
/tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/python -m pytest -vv \
  -n logical --dist worksteal -rxX --durations=30 \
  --junitxml=/tmp/b6a-final-supporting.xml \
  test/test_scf/test_functional_map.py \
  test/test_scf/test_functional_eager_compatibility.py \
  test/test_calculator/test_singlepoint_core.py \
  test/test_calculator/test_field_inputs.py \
  test/test_coulomb/test_es2_pure.py \
  test/test_coulomb/test_es2_atom.py \
  test/test_coulomb/test_es2_general.py \
  test/test_coulomb/test_es2_shell.py \
  test/test_coulomb/test_es3_setup.py \
  test/test_coulomb/test_aes2_setup.py \
  test/test_classical/test_dispersion/test_d4sc_setup.py \
  test/test_classical/test_dispersion/test_d4sc.py \
  test/test_classical/test_dispersion/test_d4sc_grad.py \
  test/test_solvation \
  test/test_interaction/test_list.py \
  test/test_baseline/test_components.py \
  2>&1 | tee /tmp/b6a-final-supporting.log
rc=${PIPESTATUS[0]}
echo "pytest exit code: $rc"
```

This supporting invocation reported 623 passed, 2 skipped, 20 expected XFAILs,
and 1 PySCF native-loader error (exit code 1). The error occurred while
loading PySCF `libcvhf.so` with undefined symbol
`int2e_spsp1spsp2_spinor` in
`test/test_solvation/test_setup.py::test_exact_mutation_rejected_and_duplicates_rejected`.
A prior candidate run also reported this loader family in
`test/test_classical/test_dispersion/test_d4sc.py::test_batch[MB16_43_01-LiH-dtype1]`.
The supporting command used Python 3.14/Torch 2.14.1 and 128 logical xdist
workers. The exact solvation node passed in a focused rerun. The same loader
failure family was reproduced on pre-package comparison SHA
`3f41f7f280a3c6df364cf7639360ca602b945118` in
`test/test_classical/test_dispersion/test_d4sc_setup.py::test_sparse_d4_parameters_are_preserved[s9]`
(616 passed, 2 skipped, 20 XFAILs, 1 loader failure); another parent rerun was
clean (617 passed, 2 skipped, 20 XFAILs). A serial diagnostic also hit the
same loader error, after which tests in that process cascaded on the failed
PySCF import; this was not treated as an acceptance run. The GFN2-only NO2
reference selection was run with `python -m pytest -vv -n logical --dist
worksteal -rxX --junitxml=/tmp/b6a-no2-reference.xml
test/test_baseline/test_reference.py -k 'NO2 and gfn2'`; it passed 8 cases
and retained 2 expected third-derivative XFAILs. The
cross-revision reproduction and isolated candidate passes
attribute these intermittent native-loader errors to the shared environment,
not the interaction-matrix change. No reference, tolerance, or xfail marker
changed. This retrospective validation does not alter E4core's accepted
implementation dependency or close broader B6a's legacy-shaped DFT-D3
setup/cache migration.

Neither milestone replaces B6b-core's stronger immutable-container/purity
work, nor closes broader B6a's legacy-shaped DFT-D3 setup/cache migration.

## B6a.1 Classical components

Convert each classical contribution to:

```python
setup_term(model_parameters, numbers, structural_data) -> TermSetup
energy(term_setup, positions, call_inputs...) -> Tensor
```

No term caches its most recent call.

Families and current status:

- repulsion — done;
- DFT-D3/tad-dftd3 — official-release forward AD is accepted under E1.3;
  legacy-shaped component/cache setup remains in broader B6a;
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
- D4SC — done; setup owns static model/parameter data and each evaluation creates fresh call-local model, CN, and dispersion-matrix data;
- ALPB/solvation — done; setup owns static solvent configuration and each evaluation creates fresh Born radii and solvation-matrix data;
- field terms — remain in B7.

For D4SC, for example:

- D4 model/table data from elements belongs in System;
- coordination number and dispersion matrices belong to the call;
- no D4 model object has its `numbers` mutated by legacy SCF culling.

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

A temporary per-call SCF data object may still exist until E4core, but it must not survive the call or be reused by the next evaluation.

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

B7 closure record (task base `d1c63d3a418e2cf3b4aa665fd15c6a72075b4a78`):
field kernels and full-System reverse/JVP/jacfwd are transform-compatible.
Field-free and explicit-field whole-System `vmap` diagnostics both reach the
generic occupation/Fermi-search host-scalar extraction in
`wavefunction/filling.py::_read_flags` (`Tensor.item`). A diagnostic-only
bypass of the earlier charge-spreading conservation check exposed the same
Fermi-search blocker on both trees; no SCF source was changed. Whole-System
`vmap` is therefore deferred to transform-compatible SCF work in E4core and
conformer mapping in E6a, under the E0 design constraints. This does not start
E4core or E6a.

---

# 11. E0: SCF/eigensolver decision

The E0 decision is recorded in [`E0-decision-note.md`](E0-decision-note.md).
The raw physical map `G(x)` is distinct from the pure solver/mixer transition.
Functional unrolling evaluates `G`, applies the functional solver, and
differentiates that finite trajectory as the G1 reference path. Future
implicit differentiation applies to the mixer-independent physical equation
`G(x)-x=0`, only at a converged fixed point. Full-mode `maxiter` and
`Result.iterations` compatibility, including the supported `maxiter=0`
one-shot behavior, is characterized.

Spectral derivatives target gauge-invariant density/occupation quantities;
Lorentzian eigenvector broadening is not the final derivative definition.
Finite-temperature degeneracies use the analytic spectral divided-difference
limit, and a zero-temperature occupied/unoccupied crossing is not assigned an
artificial derivative. E4core owns both the pure physical map and the
functional fixed-shape solver around it, including tensor-masked SCF/Fermi
control flow and per-call spectral inputs. Diagnosis-only E3.0 reproduces and
localizes both NO2 endpoints; E3.1 owns mathematical fixes and invariant
spectral/SCF differentiation using the raw E4core map. No such fix is part of
E3.0.
E6a owns
conformer mapping over a fixed System; E6bimpl owns mixed-System construction,
composition, and correctness, while E6broute owns production activation and
legacy-route retirement. E0
changes no SCF implementation and does not mark E3, the E4 umbrella, E6a, or
E6b done.

---

# 12. E2: fixed-shape integral path and legacy batch boundary

The original E2 assumptions are stale. The single-System E2 core prerequisite
is implemented; E6bimpl validates the replacement path, and E6broute retires
the remaining packed-driver cleanup only when the new route becomes
authoritative.

The current PyTorch pair builder already:

- constructs a structural `PairPlan`;
- groups shell pairs;
- evaluates groups in tensor operations;
- performs out-of-place scatter;
- supports high-order reverse/forward differentiation in the baseline;
- avoids pair-by-pair Python evaluation.

Therefore re-scope E2.

## E2-core: setup-time structural planning — done

`PytorchIntegralSetup.pair_plan` is the structural `PairPlan`, owned at
`System.integral_setup.pytorch.pair_plan`. `Model.setup(numbers)` creates the
integral setup through `setup_integral_evaluation()` and `setup_integrals()`.
The plan depends on the structural `IndexHelper`/basis layout and device; it
contains no positions. Pure `build_overlap`, `build_dipole`, and
`build_quadrupole` pass this stored plan to the pair assembler. They do not use
the low-level `_resolve_plan()` compatibility fallback. The basis exponents
and contraction coefficients remain parameter-connected setup tensors; the
plan itself is discrete structural data. Setup conversion may rebuild
device-local plan indices alongside the converted `IndexHelper`.

Core evaluation passes no `screening_threshold`, geometry-selected pair list,
or dynamic chunk list. The stored ordered shell-pair classes determine the
fixed iteration shape, and positions change only integral values. The pure
integral tests lock higher reverse orders, JVP, `jacfwd`, `vmap`, setup-time
plan ownership, repeated-geometry independence, and parameter gradients.

This is the single-System structural pair-planning result; do not add a second
`System.pair_plan` field. Node migration is later work. E2-core acceptance
includes setup-time PairPlan ownership, no position-dependent pair count or
dynamic screening on the transform path, pure overlap/dipole/quadrupole
builders, higher reverse derivatives, JVP/jacfwd, position vmap,
repeated-geometry structural stability, and connected parameter gradients.

## E2-legacy-batch cleanup: deferred to E6broute

`IntDriverPytorch.setup()` still constructs, for the legacy packed Calculator
path, per-entry:

- per-entry `IndexHelper`;
- per-entry Basis;
- per-entry PairPlan

inside its Python loop. Keep this compatibility behavior until E6broute replaces
packed Calculator evaluation with externally mapped/stacked single-System
evaluation. E6bimpl validates the replacement route; E6broute retires this
packed per-entry path when the PyTorch mixed-System route becomes authoritative.
Removing it before that point would break legacy batching or begin E6bimpl
prematurely. This cleanup is the only remaining E2 umbrella item and does not
block E4core or any single-System transform consumer.

## E2-core fixed transform path — done

The transform-critical single-System path has:

- no dynamic screening;
- no pair count based on positions;
- no data-dependent chunk list;
- fixed structural classes.

The optional low-level screening API remains available to eager compatibility
callers, but the pure core does not pass it. No production batching rewrite or
change to legacy Calculator batch semantics is part of E2 closure.

## E2.4 Preserve grouping unless measurements justify change

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

## Canonical E4core milestone and umbrella boundary

`E4core` means completion and acceptance of the canonical single-System
functional SCF path: a pure physical map and solver, fixed-trip control,
tensor-valued status, and no core `cull`/`restore`. It comprises E4.0–E4.5b
and explicitly excludes E4.6a/E4.6b legacy compatibility cleanup. Every
upstream dependency on the functional SCF means `E4core`, never completion of
the E4 umbrella.

The E4 umbrella remains partial: E4core is accepted; E4.6a is blocked on E6broute
production batch-route retirement with W2/W3 and batch-critical E7 clearance;
E4.6b is blocked until every remaining legacy FULL fallback is migrated,
globally rejected, or explicitly retired.

## E4.0 — pure raw charge map extracted

The initial `scf/functional` module provides frozen `SCFInputs`, fixed
`ChargeLayout`, immutable internal charge/potential values, `SCFMapResult`,
and a zero-temperature charge-mode `scf_map(x, inputs)`. At E4.0, charge mode
and zero-temperature scope were the supported input scope. Single-System
shape and exact migrated built-in interaction eligibility are checked when
inputs are prepared; `ConfigSCF` and solver settings are not retained by the
map. The map computes call-local
inverse Cholesky at input construction, carries a structural orbital mask,
and does not use legacy restart data or invoke interaction state-management
operations. Its result names the potential `driving_potential` because it is
`V(x)`, while `mapped_x` and `charges` are `G(x)`. The legacy full solver's
finite-tolerance finalization remains a separate compatibility rule: final
Hamiltonian/density/orbital outputs come from the last raw map evaluation,
while its returned potential and interaction energy are recomputed from the
returned mapped charges. E4.1 must preserve this before any behavior change.

The raw map matches an independently prepared current full-mode map
evaluation for GFN1 H2O (initial and perturbed charge iterate), GFN2 H2O
multipoles, a charged/spin case, a monopole-only layout, and an explicit
electric field. Its reverse position derivatives through second order,
parameter graph, repeated-map behavior, and vmap over charge iterates are
covered by focused tests. Functional purity is guaranteed only for exact
migrated built-in interactions. Custom Interaction subclasses remain on the
legacy solver path until a separate pure extension protocol is designed.

At the E4.0 milestone, the existing full/implicit solver engines remained the
only production route. E4.0 did not add mixers, solver state, fixed-trip
control flow, Fermi handling, potential/Fock modes, or free-energy
partitioning. The transitional
full-mode eigensolver is isolated in one helper and preserves its current
`requires_grad`-selected Lorentzian behavior. On the characterized H2O map,
JVP/jacfwd are finite but their numerical derivatives differ from reverse
mode by about `1.205e-6` for an isolated Hamiltonian-to-density spectral
projection under that legacy mode selection. The discrepancy reproduces
inside the transitional eigensolver without map or interaction plumbing; it
is an existing spectral-contract characterization, not by itself proof of the
NO2 diagnosis. E3.0 records this and the exact two NO2 endpoints without
implementation changes. E3.1 owns any fix under E0's already-decided
contract; a new custom primitive requires a separately reviewed upstream
design package.

The legacy full solver is no longer the ordinary eligible single-System
production route, but remains an intentional compatibility/fallback engine.
E4core is accepted; the E4 umbrella remains partial while those fallback
owners still require `SelfConsistentFieldFull`. E4.6 cleanup is sequenced
after the remaining batch and compatibility migrations below.

## E4.1 — functional simple solver (done; production-routed by E4.5b)

The internal `scf/functional/solver.py` implements frozen
`SCFSolverSettings`, pure `simple_mix`, raw-residual status values, a
fixed-trip charge solver, and separate finalization. It accepts only simple
linear mixing, no dynamic damping, one System, charge mode, and the E4.0
zero-temperature inputs. Anderson and Broyden are unsupported by E4.1 rather
than approximated. Effective `x_atol` and `x_atol_max` include the existing
Simple-mixer dtype-resolution downgrade; `f_atol` is absent.

The solver evaluates the initial `G(guess)` once. With `mix_guess=True`, it
mixes that value but deliberately does not test convergence until the next raw
map call. Positive budgets then perform up to `maxiter` additional fixed-trip
calls. Convergence uses both L2 and max-absolute norms of the unmixed
`G(x)-x`; the first converged raw map result is selected with tensor masking.
`maxiter=0` returns the one-shot raw map, one reported evaluation, false
convergence, and its residual. Nonconvergence returns the last raw map. The
reported iteration count is the legacy-compatible active map-evaluation count,
including the initial evaluation.

Finalization is separate: `driving_potential` remains `V(x)` from the selected
map evaluation, while final potential and interaction energy are recomputed
from its mapped output charges `G(x)`. Hamiltonian, density, orbital energies,
coefficients, and orbital energy contribution remain from that same selected
spectral evaluation. E4.1 parity tests compare independent legacy and
functional preparations over the six `(maxiter=0/1/2, mix_guess=False/True)`
cases, converged/loose/nonconverged cases, finalization, and geometry/field
derivatives. A smooth affine toy map verifies reverse/JVP/jacfwd/vmap solver
composition; a real fixed-System charge-guess vmap is also covered.

At the E4.1 milestone this implementation was an internal parallel path;
`Calculator`, `System.singlepoint`, `scf.solve`, `SelfConsistentFieldFull`,
and implicit mode were unchanged. Production FULL routing was integrated later
by E4.5b. E4core is accepted; the E4 umbrella remains partial while E4.6
cleanup is blocked by remaining legacy fallback ownership.
E3.0 is diagnosis-only: it must first reproduce both T0.2 GFN2/NO2
`hyperpolarizability` and `pol_deriv` known-wrong endpoints with the historical
Anderson setup before tracing either failure through lower layers. The
isolated H2O mode-selection discrepancy remains supporting characterization
and cannot substitute for those endpoint reproductions.

This is the most important structural package after state removal.

The existing `scf/pure` naming is misleading: `_Data` and iteration functions still mutate shared state.

Do not incrementally patch this object.

Define explicit immutable/static inputs, raw physical-map results, and solver
state as separate values.

Suggested concepts:

```python
@dataclass(frozen=True)
class SCFInputs:
    # hcore, overlap, occupations/reference data,
    # interaction setup/per-call data, masks, settings
    ...

@dataclass(frozen=True)
class SCFMapResult:
    mapped_x: ...
    density: ...
    potential: ...
    hamiltonian: ...
    emo: ...
    coefficients: ...
    occupations: ...
    free_energy: ...

@dataclass(frozen=True)
class SCFSolverState:
    x: ...
    mixer_state: ...
    active: ...
    iterations: ...
    residual: ...
```

Then:

```python
def scf_map(
    x: Tensor,
    inputs: SCFInputs,
) -> SCFMapResult:
    ...

def solver_step(
    state: SCFSolverState,
    result: SCFMapResult,
    settings: ...,
) -> SCFSolverState:
    ...
```

The physical equation is `G(x) - x = 0`; mixer history belongs only to
`SCFSolverState`. No mutation of inputs, map results, or incoming solver state.

## E4.2 Functional Anderson mixer state (done; production-routed by E4.5b)

The functional full-solver mixer scope is Simple and Anderson. Legacy full
mode resolves configured linear mixing to Simple and Anderson to Anderson;
configured Broyden warns and falls back to Anderson unless strict mode is on,
where it raises `NotImplementedError`. Functional numerical settings receive
the already-resolved effective mixer. A future eager compatibility adapter
owns that warning/error behavior.

Functional Broyden is not a full-solver prerequisite. The legacy implicit
root solver uses xitorch `broyden1`; replacement of that root-solving route and
its Broyden behavior belongs to future E3/implicit work around the common
`G(x)-x=0` map.

Anderson history is a frozen state value, and the transition reproduces the
legacy warm-up, history ordering, diagonal rescaling, and linear solve without
in-place writes. E4.1 Simple solver behavior remains unchanged. Dynamic damping
is still unsupported by the functional solver and must be resolved before
production routing. `generations=0` is rejected as invalid: the legacy mixer
fails on its first transition when it reads the absent second history row.

The canonical functional solver tests the raw convergence residual before its
physical mixer transition. A raw map that satisfies convergence becomes the
selected physical result and does not advance functional mixer state. Fixed-
`transition_active` is false and no solver/history state advances.

This intentionally removes one legacy side effect. Current
`SelfConsistentFieldFull` calls `mixer.iter(q_new, q)` and only then reads
`mixer.converged`; thus it advances the mixer on the convergence step. The
canonical functional solver does not perform that dead transition. The
parity is not exact: legacy Anderson may fail while solving the mixer system
for a residual that already satisfies convergence, while canonical E4core accepts
the raw physical result without that transition. E4.5b accepted this deliberate
solver-bug fix for eligible production FULL routing.

## E4.3 Transform-compatible finite-temperature occupation (done; production-routed by E4.5b)

At the E4.3 milestone, the functional charge map and Simple/Anderson solvers
support finite-temperature occupation through a fixed-trip tensor-status
Fermi core. The eager public API
retains host-side error reporting, while both paths share the safeguarded
Newton/bisection search step. Fermi status, the separate `1e-4` electron-count
guard, and equal/atomic entropy partitions are preserved in the functional
results. Production routing was activated later in E4.5b for eligible
single-System FULL calculations; whole-System public `vmap` remains deferred to
E6a because the eager policy performs host inspection.

The temperature/free-energy value exists independently of whether smearing is
active: exactly zero temperature has zero free energy; positive temperatures
below `3e-7` Hartree retain the legacy Aufbau occupation and clamped low-T
entropy; the cutoff itself and higher temperatures use fixed-trip Fermi
occupation. `SCFSolveResult.converged` describes only the outer raw SCF
residual. `trajectory_status` separately accumulates Fermi convergence,
electron-count validity, invalid-temperature status, and too-many-electrons
status over every physically executed raw-map spectrum. The final public
representation-conversion spectrum has its own `final_evaluation.fermi_status`.
Future eager acceptance must require both aggregate subsolve validity and the
applicable outer-convergence policy; neither status may erase the other.
Below the smearing cutoff no Fermi search runs and both trajectory and final
Fermi status are structurally `None`.

The legacy `hamiltonian_to_density()` path calls eager
`get_fermi_occupation()`, which raises on search/input failure, and then
independently raises if `abs(occupation.sum(-1) - nel) > 1e-4`. Therefore every
physical spectrum in the SCF trajectory must contribute to eventual eager
validity; a later successful spectrum cannot erase an earlier failure. The
functional numerical solver retains these statuses without raising; the
E4.5a eager policy interprets them before accepting production results.

`maxiter=0` is a valid one-shot compatibility mode: it evaluates one raw map,
returns `converged=False`, and the legacy full solver returns before its
nonconvergence warning/`force_convergence` error path. The E4.5a eager policy
first rejects invalid functional subsolves; if they are valid and
`maxiter=0`, it accepts the one-shot result without an outer-convergence
warning/error. For positive budgets it accepts a converged result and applies
the existing `force_convergence` warning/error policy only to an unconverged
result. E4.5a implements this policy and E4.5b activates it for eligible
production FULL calls.

## E4.4a Functional potential representation (done; production-routed by E4.5b)

Executable legacy characterizations lock the three representation phases.
Charge mode maps `q_k -> V(q_k) -> H -> density ->
q_(k+1)` and returns `q_(k+1)` without another eigensolve. Potential mode maps
`p_k -> H(p_k) -> density -> q_(k+1) -> p_(k+1)`; after selecting the mapped
potential, legacy final conversion evaluates `p_(k+1) -> H -> density ->
q_(k+2)`, then reports `V(q_(k+2))`. Fock mode maps `H_k -> density ->
q_(k+1) -> p_(k+1) -> H_(k+1)`; final conversion evaluates `H_(k+1) ->
density -> q_(k+2)` and reports `V(q_(k+2))`.

Consequently, potential/Fock public charges, potential, density, occupation,
and free energy can come from a final spectral evaluation beyond the selected
raw map. Their returned Hamiltonian/energy fields must be characterized
individually; charge-mode finalization cannot simply be reused. The focused
characterization counts raw map evaluations, mixer transitions, and final
conversion eigensolves for `maxiter=0/1/2` and both `mix_guess` values, and
also covers 300 K final conversion plus GFN2 potential multipoles.

The characterized phase ownership is: charge mode returns charges from the
selected `G(q_k)` output, while Hamiltonian, density, `emo`, coefficients,
occupation, orbital energy, and `fenergy` remain from the selected spectral
evaluation driven by `V(q_k)`; public potential and interaction energy use
`V(q_(k+1))` and `q_(k+1)`. Potential mode returns `q_(k+2)` and
`V(q_(k+2))`; its Hamiltonian, density, `emo`, coefficients, occupation,
orbital energy, and `fenergy` come from the extra final spectrum driven by
selected `p_(k+1)`. Fock mode returns `q_(k+2)` and `V(q_(k+2))`; its public
Hamiltonian is selected raw `H_(k+1)`, while density, `emo`, coefficients,
occupation, orbital energy, and `fenergy` come from the final spectrum of
that same `H_(k+1)`. The public electronic energy combines the orbital energy
from the final phase applicable to each representation with interaction
energy computed from the returned charges. The zero-budget case has one raw
map in all modes and an additional conversion spectrum only in potential and
Fock modes; this conversion does not increment the SCF iteration count.

The legacy representation characterization above is now enforced by a
functional potential implementation. `SCFInputs` has no `scp_mode`; that
representation is selected at the solver API boundary. `PotentialLayout` is
fixed structural metadata, and pure pack/unpack preserves legacy single-System
mono/dipole/quadrupole rows and `PADNZ` tails. A call-local potential guess is
formed from initial charges without a spectrum. Charge and potential maps
share the immutable `SCFSpectralResult` through `evaluate_hamiltonian` and
`evaluate_potential`.

Potential Simple and Anderson solvers use the common fixed-trip mixer control.
The selected `map_result` remains the raw potential map result. A separate
`final_evaluation` performs exactly one additional spectrum from the selected
mapped potential; it does not add an outer iteration. Public charges come
from that final evaluation, public potential is recomputed from those charges,
and energy/free energy use the final spectral phase. `trajectory_status`
summarizes all physical raw potential-map spectra, while
`final_evaluation.fermi_status` reports only the final public conversion
spectrum; future success checks require both. Tests cover raw GFN1/GFN2 map
parity, each GFN2 potential row, the six Simple budgets, accelerated and
converged Anderson, maxiter zero, finite-temperature equal/atomic entropy,
fixed-System guess vmap, and geometry/parameter/field/fractional-charge
derivatives. At the E4.4a milestone, production routing remained unchanged.

Potential mixer transitions re-pack the fixed structural components after
affine mixing. This is necessary because Anderson's floating-point affine
combination can perturb constant `PADNZ` row tails even when every physical
map preserves those tails exactly. Projection restores the structural
sentinel before the next residual is evaluated; residual norms need no
representation-specific padding mask.

The functional result preserves three distinct facts: outer SCF residual
convergence, validity accumulated across every physically executed raw-map
spectrum, and validity of the final public conversion spectrum. Fixed-trip
dummy maps after convergence do not contribute to trajectory validity. The
E4.5a eager policy requires subsolve validity first; valid `maxiter=0`
remains an accepted one-shot result despite `converged=False`, matching the
legacy early return before its warning/forced-convergence branch.

## E4.4b Functional Fock representation (done; production-routed by E4.5b)

The functional single-System Fock map consumes an exact `(nao, nao)` matrix
without value-based `PADNZ` cleanup or packed-batch semantics. It shares
`evaluate_hamiltonian` with charge/potential maps and the common fixed-trip
Simple/Anderson solver controls. Initial guess formation is the pure
charge-to-potential-to-Hamiltonian conversion and performs no spectrum.

Fock finalization deliberately distinguishes map and public phases:

| Value | Phase |
| --- | --- |
| `map_result.mapped_x` | selected mapped `H_(k+1)` |
| `map_result.spectral.hamiltonian` | input `H_k` that generated that raw map |
| `final_evaluation.hamiltonian` and public Hamiltonian | one extra spectrum of selected `H_(k+1)` |
| public charges | `q_(k+2)` from that extra spectrum |
| public potential | `V(q_(k+2))` |
| energy and free energy | extra final spectral phase and its charges/occupation |

That extra spectrum does not increment SCF iterations. All physically active
raw-map Fermi statuses accumulate in `trajectory_status`; the final spectrum
has its own status in `final_evaluation`. Fock remains single-System only;
matrix padding and packed batching implementation belongs to E6bimpl, with
batch-critical safety in E7. E4.5b routes eligible
single-System FULL calculations through this implementation; legacy full SCF
remains responsible for batch calls and other explicitly unsupported cases.

## E4.5a Dynamic damping and eager compatibility (done; consumed by E4.5b)

The functional Simple and Anderson solvers carry immutable dynamic-damping
settings and tensor-valued runtime damping. The compatibility threshold remains
`0.1`; a nonconverged candidate is mixed with the previous damping first, then
the damping for the next transition is replaced by `damp_dynamic_factor` when
the raw L2 residual is below the threshold. Initial `mix_guess` transitions do
not switch damping. Anderson warm-up continues to use `damp_init`; only its
accelerated branch uses the runtime ordinary damping. Newly converged and
inactive fixed-trip lanes freeze mixer and damping state.

`functional/eager.py` provides a static route resolver and inactive eager
compatibility adapter. Full, iterative, single-System calls with exact migrated
interaction types are eligible. Configured Broyden resolves to effective
Anderson with the legacy warning in non-strict mode and raises in strict mode.
Custom interaction subclasses, implicit/experimental modes, non-SCF methods,
and batches remain legacy-owned. The adapter interprets Fermi trajectory and
final-spectrum status before outer convergence policy. It accepts valid
`maxiter=0` one-shot results even with `force_convergence=True`, and applies
legacy warning/error behavior for positive-budget nonconvergence. The
transform-critical solve retains tensor-valued `iterations`, `converged`, and
`residual`; only the eager adapter reads status on the host.

E4.5a hardening preserves chronological failure policy. Each physically
executed raw-map spectrum records its first failing Fermi category in tensor
state; fixed-trip dummy maps after convergence do not participate. Eager
interpretation checks that trajectory failure first, then the separate final
public-spectrum status, and only then applies outer convergence policy.
Exception types and stable message families are compatibility requirements;
full legacy diagnostic suffixes are not retained by tensor status.

The route resolver is static and side-effect free. Eager execution preserves
the legacy preparation order: `get_refocc`, the non-SCF early return, then
`get_guess`, and only then route resolution. Thus invalid spin/reference
occupation errors precede strict-Broyden and functional-only configuration
errors, while the configured-Broyden fallback warning is emitted only after
both common preparation steps succeed. Direct resolver calls may report a
route error without preparation; user-facing eager execution does not.
`run_prepared_functional_full` is the functional-only seam for E4.5b. It
accepts the original configuration, validates representation consistency,
normalizes the configured mixer from `route.effective_mixer` in one place,
and has no legacy/implicit/experimental dispatch. The separate
`apply_functional_full_policy` owns eager Fermi and outer-convergence
interpretation so E4.5b does not duplicate it. Production `scf.solve` remains
the owner of common preparation, non-SCF early return, existing mode dispatch,
and legacy `*args`/`**kwargs` forwarding.

Negative Fermi `maxiter` is not rejected by `ConfigFermi`; the functional
resolver therefore marks that configuration legacy-owned until a separate
configuration policy changes it. Negative outer SCF `maxiter` is the E0
canonical configuration error only after common preparation and only for an
otherwise functional-eligible single-System full-SCF route. Batch,
custom-interaction, non-full, non-SCF, and negative-Fermi-maxiter
legacy-owned configurations remain legacy-owned even when their SCF budget is
negative. In particular, negative Fermi plus negative SCF `maxiter` stays on
legacy. Eager convergence warnings use configured Python damping values,
while runtime damping and its switched state remain tensors.

`Result.snapshot()` accepts structural charge/potential values, so immutable
functional charge/potential tensors pass through without conversion to legacy
mutable containers. Functional `iterations`, `converged`, and `residual`
tensors are passed through by identity; legacy Python iteration integers are
normalized at the singlepoint boundary. B5.6 adds the public `SCFStatus` to
the common output and snapshot boundary, projecting trajectory and final
failure-code tensors without copying them or exposing `SCFSolveResult`. The
inactive eager adapter follows the same ordering and is not a replacement for
the legacy dispatch function.

The eager adapter is retained for side-by-side compatibility tests. Production
`scf.solve()` calls only its prepared functional runner and policy helper after
common preparation has completed.

## E4.5b Production routing (done)

`scf.iterator.solve()` remains the common preparation and dispatch owner:

```text
get_refocc()
if non-SCF: existing early return
get_guess()
if full mode: resolve route
    if functional: emit route warning, run_prepared_functional_full,
                   apply_functional_full_policy, return tensor result
    else: existing SelfConsistentFieldFull path with *args/**kwargs
elif implicit: existing implicit path
elif experimental: existing experimental path
```

The production FULL branch calls the reviewed E4.5a prepared runner for
eligible single-System calculations. The ordinary defaults are
`MIXER_BROYDEN`, `SCP_MODE_POTENTIAL`, and `FERMI_ETEMP=300 K`; configured
Broyden resolves once to functional Anderson and emits the compatibility
warning once. `scf.solve()` returns the legacy `SCFResult` mapping on legacy
routes and `FunctionalCoreResult` on the functional route. Core callers branch
explicitly at their result boundary; functional tensors and solver status pass
through without conversion to legacy mutable containers or Python scalars.

Legacy FULL ownership remains explicit for batches, unsupported interaction
extensions, negative electronic temperature, negative Fermi `maxiter`,
missing H0/overlap matrices, and supplied legacy engine overrides
(`*args`, `fwd_options`, `bck_options`, or `eigen_options`). Those options are
not dropped or reinterpreted by the functional implementation. Negative outer
SCF `maxiter` is rejected only when the call is otherwise a functional-route
candidate. Implicit, experimental, and non-SCF dispatch remain unchanged.
No catch-all retry occurs after functional numerical execution begins.

The accepted convergence-step behavior remains: canonical functional control
deliberately skips a transition after accepting a converged raw map, while the
legacy loop calls `mixer.iter` before checking convergence. This is the
intentional E0 solver correction; the production functional route retains it.

The eager policy preserves ordering: first raw-trajectory subsolve failure,
then final-public-spectrum failure, then positive-budget outer convergence
policy. `maxiter=0` remains an accepted one-shot if both subsolve phases are
valid, including with `force_convergence=True`. Negative Fermi `maxiter` stays
on legacy until its global configuration policy is deliberately changed;
negative SCF `maxiter` is intentionally rejected by E0. The production
`Result.snapshot()` boundary passes functional `iterations`, `converged`,
and `residual` tensors directly, while legacy iteration integers are
normalized eagerly. The singlepoint iteration normalization helper returns
an existing Tensor unchanged and only creates a Tensor for legacy integer
values. Functional values are not forced through legacy
`SCFResult(iterations: int)`.

Single-System dynamic damping is characterized in E4.5a. Legacy batch-only
GFN2 auto-Fock behavior remains with E6bimpl/E7; cull/restore retirement is
owned by E4.6a after E6broute. Neither belongs in this resolver.

## E4.6 Legacy cleanup (blocked/deferred)

The legacy full solver remains required by current production fallbacks. Do
not delete `SelfConsistentFieldFull` or its loop while any of these consumers
remain:

| Current owner | Prerequisite before removal |
| --- | --- |
| Every legacy FULL batch-facing behavior | E6bcompat-policy decides retain/retire before E6bimpl scope is frozen; its inventory is above. E6bimpl provides the selected replacements, and E6bcompat validates retained values/API or explicit retirement/migration/error behavior. This covers standard PyTorch and libcint batches, custom/unsupported interactions, `unexpected_option`, `solver_options`, any other nonempty/unknown `**kwargs`, positional engine extras, `fwd_options`, `bck_options`, `eigen_options`, negative SCF `maxiter`, negative temperature, negative Fermi `maxiter`, and missing/malformed `hcore` or overlap. For each retained legacy single-System call, forward `*args`/`**kwargs` unchanged: do not filter, normalize, or silently discard them. Preserve characterized `TypeError` behavior for unsupported options and constructor semantics for supported options. E6bcompat tests batch+`unexpected_option`, batch+`solver_options`, positional extra, and each named option. E6broute requires E6bcompat plus batch-critical E7 and E6c1 W2/W3; E6a mapping alone is insufficient. |
| Custom `Interaction` subclasses | Define a pure extension protocol or an explicit compatibility policy. |
| Negative electronic temperature | Make a global configuration-policy decision. |
| Negative Fermi `maxiter` | Make a global configuration-policy decision. |
| Legacy `fwd_options`, `bck_options`, `eigen_options`, `*args`, and `**kwargs` | Migrate their semantics or explicitly retire the accepted API. |
| Missing/malformed integral inputs | Decide validation ownership and preserve the selected error contract. |

Split the work by dependency:

```text
E4.6a — remove batch cull/restore only after E6broute retires the production
         batch route, with batch-critical E7 and E6c1 W2/W3 gates passed
E4.6b — delete the remaining legacy full loop only after every fallback above
         is migrated, globally rejected, or explicitly retired
```

The cull/restore machinery is still used by the legacy batch route; do not
remove it while batch FULL calls remain legacy-owned.

### Source ownership audit (current `src/dxtb`)

- New single-System functional map/solver: no `cull`/`restore` calls. Its
  `SCFInputs` still carries the existing `IndexHelper` class for structural
  transforms, and that class itself retains legacy cull/restore methods.
- Legacy FULL batch fallback: `SelfConsistentFieldFull` culls its `_Data` and
  mixer; `_Data` culls integral/state tensors, `IndexHelper`, and interaction
  caches; the interaction cache/component implementations expose cull/restore
  hooks. This is the active production owner of batch culling.
- Implicit solver: no direct cull/restore call in its solver implementation;
  it still shares the legacy `BaseSCF`/`IndexHelper` compatibility types.
- Calculator/integral adapter: no direct cull/restore call sites were found in
  those source areas. Calculator batch execution reaches the legacy FULL owner
  above.
- Other SCF helper: `scf/pure/data.py` retains a cull-capable `_Data` helper
  used by the non-self-consistent path, but the current source has no call to
  that helper's cull method.

In particular, the `IndexHelper` in functional `SCFInputs` and the legacy
batch path is currently the same concrete batch-aware class; the legacy setup
may pass that same per-call object. It has `batch_mode` and mutating
`cull/restore` methods, even though the functional core only reads its
structural operations. C5 must split the immutable core structural value from
the legacy adapter before removing those methods.

## Functional solver loop contract

The new single-System functional core has no cull/restore operations. The
legacy batch solver may continue slicing converged entries and culling its
batch-dependent compatibility objects until E6broute retires that production
route after the batch-critical E7 gate, E6c1 W2/W3 performance clearances,
and retained-backend compatibility correctness gate pass. E6bimpl owns
replacement construction/composition/correctness; E6broute owns route
migration; E7 owns safety validation. Do not delete the
legacy optimization while batch FULL calls still use it.

For mapped execution, use an active/converged mask and preserve the previous state after convergence.

Do not use tensor-dependent Python `break` in the transform path.

Conceptually, preserve the existing full-mode budget contract: evaluate the
initial map once, use its mapped value directly when `mix_guess=False`, or
apply the functional initial mix when `mix_guess=True`, then perform up to
`maxiter` additional map evaluations. In particular, `maxiter=0` still
returns the one-shot initial map result and reports one map evaluation.

```python
result = scf_map(state.x, inputs)
final_result = result
iterations = ones_like_systems(dtype=int64)
residual = residual_values(result.mapped_x - state.x)
converged = zeros_like(active)
if maxiter > 0:
    state = (
        initial_mix(state, result)
        if mix_guess
        else with_x(state, result.mapped_x)
    )

for _ in range(maxiter):
    result = scf_map(state.x, inputs)
    residual = result.mapped_x - state.x
    newly_converged = active & convergence_test(residual)
    final_result = tree_where(active, result, final_result)
    transition_active = active & ~newly_converged
    candidate = solver_step(state, result, settings, transition_active)
    state = tree_where(transition_active, candidate, state)
    iterations = where(active, iterations + 1, iterations)
    active = transition_active

if maxiter > 0:
    converged = ~active
```

For the zero-budget one-shot, report `converged=False` because no convergence
test ran, and retain `G(guess)-guess` as a diagnostic residual. For positive
budgets, `iterations` counts physical-map evaluations, including the initial
one, to preserve the current full-mode `Result.iterations` contract.

The exact tree masking helper can be optimized later.

The loop trip count is static.

An eager-only outer wrapper may eventually stop early if measurement justifies it, but the transform-critical function remains fixed-shape.

### Eventual legacy cleanup target

The following are eventual compatibility-cleanup targets, not immediate work
that follows from E4core single-System routing:

- legacy `IndexHelper.cull/restore` and cache `cull/restore`;
- legacy `_Data.cull/restore`;
- legacy mixer culling;
- shape recomputation after legacy batch systems converge.

Remove these only through E4.6a after E6broute retires the production batch route
and the batch-critical E7 gate plus E6c1 W2/W3 clearances pass. C5 migrates
the surviving immutable core structural representation and does not wait for
deletion of a separate legacy-only adapter.

## E4core acceptance

For the reference set:

- same energies and charges;
- same convergence outcomes;
- same or documented iteration counts;
- single-system derivatives unchanged;
- the fixed-trip functional SCF controller composes under `vmap` without
  cull/data-dependent shape logic. E4core supplies the numerical controller;
  E6a plus C8a makes the public `System.singlepoint` result transform-safe
  for the exact B1 conformer call. The eager `Calculator.singlepoint` adapter
  separately owns host-side warning/error policy and early stopping;
- no state object is mutated.

Measure runtime before/after. Fixed-iteration masking may be slower for highly heterogeneous convergence counts; record it rather than reintroducing culling.

---

# 14. B6b-core and legacy compatibility purge

Separate the new-core state-purge milestone from deletion of every legacy
compatibility object in the repository. They have different prerequisites.

## B6b-core: new single-System core purity

This milestone follows completed `B6a-SCF-core` and accepted `E4core`; it does not
wait for broader B6a, whose remaining DFT-D3 outer-classical setup/cache work
is outside the functional SCF inputs.

B6b-core requires:

- `Model` has no per-call mutable state;
- `System` has no mutable per-call physics state, hidden caches, or reset/update/
  invalidate protocol used by core evaluation;
- `Result` is an immutable call result;
- functional SCF inputs and solver state are immutable explicit values;
- migrated components have no persistent per-call caches.

B6b-core also establishes the structural and interaction adapter boundaries
needed by those inputs. The current non-Tensor children reachable from
`SCFInputs` are:

| Child | Current representation | B6b-core classification/target |
| --- | --- | --- |
| `ihelp` | Shared mutable, batch-aware `IndexHelper` with `cull/restore` | Legacy-only adapter; core receives a frozen structural value derived from it. |
| `integrals` | Frozen `IntegralMatrices` containing tensor leaves | Immutable core value; no replacement or in-place field update during evaluation. |
| `interactions` | Mutable `InteractionList` containing component objects | Core receives immutable/read-only descriptors or setup values for eligible built-ins; mutable legacy list and unsupported extensions stay outside the core. |
| `interaction_data` | Mutable `InteractionListCache` mapping with cache objects | Core receives a call-local data value with immutable structure and tensor leaves; no cache mutation API is called or reachable from core evaluation. |
| layouts/settings | Frozen `ChargeLayout`, `PotentialLayout`, `FermiSettings` values | Immutable structural/configuration descriptors. |
| reachable components | Mutable legacy `Interaction` objects in the current list | Core uses immutable/read-only setup descriptors or explicit per-call values; no component object is updated, reset, invalidated, culled, or restored. |

This is behavioral immutability as well as frozen outer dataclasses: no core
evaluation may call `update`, `reset`, `cache_invalidate`, `cull`, `restore`,
or otherwise change any attribute reachable from its inputs after SCF input
construction. Tensor leaves remain explicit differentiable values. Legacy
FULL continues to receive the existing batch-aware `IndexHelper`, mutable
`InteractionList`, and cache adapters. C5 later Node-migrates the frozen
structural value established here; do not pass the mutable compatibility
objects themselves into core SCF inputs once B6b-core is closed.

B6b-core regressions snapshot structural fields, identities, list membership,
cache keys/entries, component attributes, and integral container state before
and after repeated functional raw-map and full solver evaluations. They must
prove the input/container state is unchanged, including on finite-temperature
and potential/Fock paths.

For the new core, enforce the strong P4 rule:

> no attribute of Model, System, Result, components, integral setup, SCF inputs or SCF state is modified after construction.

This rule does not require deleting a separately isolated legacy adapter
object that an explicit fallback route still needs. Core evaluation must not
observe that adapter state. B6b-core enables C5 core migration, but does not
mark B6 complete while DFT-D3 or other declared B6 scope remains open.

## Legacy compatibility purge

Repository-wide removal of cull/restore, mutable per-call containers, and
legacy reset paths is later cleanup. Batch-owned machinery belongs to E4.6a
after E6broute retires the production batch route and the batch-critical E7
padding-safety gate plus E6c1 W2/W3 clearances pass. Remaining legacy FULL
compatibility objects and the loop belong to E4.6b after all documented
fallbacks are migrated, globally rejected, or explicitly retired.

---

# 15. E3.0 diagnosis, E3.1 implementation, and full E3 acceptance

E3.0 is complete as a diagnosis-only package, with evidence recorded in
[`E3.0-NO2-diagnosis.md`](../E3.0-NO2-diagnosis.md). It reproduces both
GFN2/NO2 `KNOWN_WRONG` endpoints for PyTorch and libcint under exact T0.2
reference settings. The first failing layer is the complete finite-tolerance
fixed-point response / finite-root trajectory: tightening the root tolerance
and increasing the effective trajectory move both responses toward the stable
physical finite-difference and tblite values, while no isolated Fermi,
frozen spectral, sampled raw-map, or Anderson sub-operation is identified as
the onset. No mathematical or spectral fix was made. E3.0 remains independent
of E1.3, E6a/E6b, E4.6, legacy batch cleanup, and Node migration.

E3.1a's physical-root strategy/reference gate and E3.1s's dxtb-local invariant
density/free-energy contract are complete. E3.1b now implements SCF root
differentiation using the SCF-relevant E1.1/E1.2 subset on
the canonical PyTorch transform driver. Outer classical DFT-D3 does not block
that algorithm. Full E3 end-to-end GFN1/GFN2 transform acceptance uses
PyTorch and additionally requires full E1 (including E1.3
DFT-D3/tad-dftd3 forward-mode support) and E2-core. Preserve PyTorch and
libcint eager/reference regression coverage wherever supported; libcint
higher-order transform parity is not required. The DFT-D3 blocker is separate
from E3.0/E3.1's SCF-map work but remains a full end-to-end acceptance blocker
where that outer term is active.

Do not build higher-order implicit differentiation around today's mutable `_Data` closure.

Required implementation structure:

1. one pure raw physical map `scf_map(x, inputs) -> SCFMapResult`;
2. a pure functional solver/mixer around that map;
3. unrolled differentiation of the actual solver trajectory as the reference;
4. optional implicit differentiation of `G(x)-x=0`, independent of mixer;
5. explicit tests comparing implicit and unrolled results.

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

## C5 core structural data: IndexHelper/Basis Node migration

Start after B6b-core, E2-core's structural integral plan, and E4core's fixed-shape
single-System inputs. Migrate the immutable core structural value established
at B6b-core into child fields: no `batch_mode`, no cull/restore, and no Python
per-molecule scalar in context. Per-molecule counts that vary between stacked
Systems are tensors/children.

Today `SCFInputs` and the legacy batch solver use the same `IndexHelper`
implementation (and may receive the same call-local object). That class still
owns `batch_mode` plus mutable `cull/restore` storage. B6b-core first separates
the frozen core structural value from this legacy object; C5 Node-migrates the
core value. The legacy batch adapter may temporarily keep the current
batch-aware `IndexHelper` and cull behavior until E6broute retires that production
route after the batch-critical E7 gate and E6c1 W2/W3 performance clearances
pass. Do not
remove cull/restore from the shared class while that legacy route uses it, and
do not Node-migrate an object solely because a legacy fallback still uses it.

Add the `in_dims_like(node, **fields)` helper described by T0 when per-field batching is first needed.

## C6 components

Final status: C6 is complete at the core/legacy compatibility boundary, as
accepted in C6-Final.3 and summarized in the current status above. The family
paragraphs below retain their package-time sequence and evidence; earlier
statements that broader C6 remained partial describe those historical review
points, not current open work. Start-of-track prerequisite: B6b-core/C5 core
structural migration.

C6.1 is complete: `System` stores ES2/ES3 core setup Nodes for the eligible
single-System path. C6.2 is complete: the same boundary stores an
`AES2CoreSetup` Node. C6.3 is complete: the eligible GeneralizedBorn setup is
stored as a `GeneralizedBornCoreSetup` Node. C6.3a is complete: core
conversion moves only CPU zero-dimensional `born_scale` and `born_offset`
tensors to the setup device; this is not general mixed-device support. C6.4
is complete: eligible single-System D4SC setup uses `D4SCCoreSetup`, with
ordered parameter labels and recursive model-extra structure static and all
tensor-valued settings dynamic. C6.4a is complete: functional FULL and
implicit preparation use the same D4SC setup snapshot as the call-local
matrix builder, while direct compatibility calls retain component-based
setup capture. Legacy setup dataclasses, components, caches, and batch routes
remain intact for these families. At this package-time review point, broader
C6 family migrations remained open. C7 scoped setup Node acceptance is
complete. Later C6-Final.0-.3 packages closed the broader core descriptor and
compatibility acceptance at its documented boundary.

C6.4b is complete: `_interaction_data()` associates the System-owned D4SC
setup only with the identical component in the resolved interaction list.
An alternate exact D4SC component follows the existing compatibility setup
path, and selections without the System-owned component carry no setup
authority. C7 scoped setup Node acceptance is complete. Later C6-Final.0-.3
closed the broader core descriptor and compatibility acceptance.

Most components should now be small immutable term/setup nodes.

Migrate family by family.

Do not migrate obsolete cache classes; delete them.

## C7 integrals/H0

**C7 setup Node acceptance complete; legacy driver retirement remains
separate.** See [`C7 integral Node schema`](C7.0-integral-node-schema.md)
for the schema, ownership, transform, validation, and matched-base evidence.
The exact single-System PyTorch path owns a `PytorchIntegralSetupNode` with an
ordered `PairPlanNode` and `CoreIndexData`. Its authoritative H0 setup is an
`H0SetupNode` containing dynamic parameter/occupation tensors and nested GFN0
EEQ parameters, plus a narrow five-map `H0IndexNode`; it retains no live
IndexHelper. Built-in CN strategy descriptors are validated during conversion,
while direct custom-callable H0 dataclass setups retain their legacy path.
Direct and packed Calculator drivers remain compatible; the Calculator-owned
legacy H0 adapter is a persistent isolated setup with graph-preserving tensor
clones. The exact single-System libcint path owns `LibcintIntegralSetupNode`
and ordered `LibcintBasisNode` values. It constructs `LibcintWrapper` without
an IndexHelper, letting tad-libcint derive shell/AO maps from the captured
ordered shells; call-local dxtb data carries `CoreIndexData` for multipole
origin shifts. The full `System.ihelp` remains outside the Node for SCF and
legacy duties. Packed and direct libcint routes retain their IndexHelper-based
compatibility behavior. C7.4 confirmed that current transforms do not require
the outer `IntegralSetup` as an input pytree or `IntegralMatrices` as a direct
output pytree; both remain validated frozen dataclasses. Legacy Calculator
drivers, integral containers, and packed routes remain in their separately
assigned compatibility-retirement tracks.

Start after B4/B6b-core/C5 core structural migration.

Migrate the immutable setup descriptions created in B4.

Do not turn old mutable driver classes into Nodes merely to preserve them.

## C8a Result pytree representation

C8a is a separate output-container prerequisite for E6a and can proceed from
B5/B5.6 and the available `Node`/pytree support without waiting for
C5/C6/C7. Acceptance uses a transform-safe probe that constructs the complete
Result tree from tensor inputs with `Result.snapshot()`, not
`System.singlepoint` or the SCF controller. Require flatten/unflatten
round-trip, default `out_dims=0`, reconstructed `Result`/nested types, one new
leading dimension on every tensor leaf, optional static `None` preservation,
static classical labels with mapped energy leaves, and autograd connectivity.
Both `SCFStatus` failure-code tensors must be dynamic pytree leaves, acquire
the mapped leading dimension, reconstruct inside `SCFStatus`, retain integer
dtype/device, and remain outside static pytree context.
The exact public B1 `vmap(system.singlepoint, in_dims=(0,))(positions_batch)`
integration acceptance belongs only to E6a after its fixed-trip controller
exists; do not use an SCF-excluded or `maxiter=0` shortcut in C8a.

## C8b Model/System stackability

C6 is accepted at the exact-core/legacy compatibility boundary. C8b.0
completed the source inventory and executable schema preflight. C8b.1
registered production `ExactSystemCore` and its selected-backend wrapper as
tad-mctc Nodes. C8b.2 adds a Model-owned registered parameter tree and
functional setup integration; public `Model` and legacy `System` remain
unregistered compatibility façades.

The production exact-core tree stores dynamic children: atomic numbers with
integer dtype preserved; optional H0 setup; one selected PyTorch or libcint
backend Node; and ordered classical and interaction descriptor tuples. Its
context contains effective dtype and frozen `Config`. The backend wrapper
contains driver, integral level, and the selected backend as a dynamic child.
`index` is derived from `integral.backend.index`, so the selected backend owns
the `CoreIndexData` mapping once. A logical System-device context on the
wrapper preserves the supported libcint force-CPU arrangement under Node's
device validation; CUDA validation remains outstanding because CUDA is not
available in the current environment. No full `IndexHelper`, component list,
compatibility cache, driver object, or per-call positions/charge/spin/field
enters the core tree. The legacy `System.integral_setup` facade remains the
evaluation adapter after backend identity has been validated.

The configuration audit found frozen dataclass configuration composed of
immutable scalar/tuple/path values and no tensor fields; it is suitable as
static context. The partition signature must combine configuration values,
effective dtype, selected backend and force-CPU policy, ordered descriptor
metadata and optional-child presence, plus dynamic leaf shape/dtype/device.
It must not use `repr()` or object identities. Dtype conversion updates the
effective-dtype metadata and preserves integer/Boolean structural leaves.

The production `Model.parameter_tree` is a registered
`ModelParameterNode` with one `ModuleNode.from_module(par)` child. Its `params`
dictionary dynamically exposes named parameters and buffers and supports
tad-mctc `partition`/`combine`; no Model parameter table is copied into each
System. The ModuleNode's module architecture remains static context.
`Model.setup_functional()` applies a supplied tree through
`torch.func.functional_call` around the existing `Model.setup()` method. The
Repulsion H2 and ES2 H2O controls prove that substituted values reach captured
family setup Nodes and alter full singlepoint energies, parameter gradients,
and supported position/mixed derivatives while restoring the original
ParamModule state. The adapter is call-local; legacy extension objects do not
enter the registered Model tree.

The C8b.2 independent hardening review reproduced a race in overlapping
`functional_call` substitutions through shared ParamModule owners.
Model-mediated parameter-tree snapshots and ordinary/functional setup now
share an owner-scoped reentrant lock. Metadata validation also rejects malformed ModuleNode state
keys, shapes, incompatible dtype categories, mixed floating dtypes, and mixed
devices before substitution while allowing homogeneous real floating dtype
conversion. Validation and substitution consume one shallow state mapping
snapshot. New tests cover converted-tree
float32 setup/energy/gradients, the GFN1 H0 `kpol` path, restoration after
setup exceptions, concurrent calls across Model façades sharing one owner,
mode-1 packed setup, and a custom classical extension. Concurrency tests
instrument protected implementation stages and each fails under a
test-only lock bypass. This is a setup-only
concurrency guarantee; direct ParamModule access and later System/extension
evaluation are not synchronized.

The C8b.1 selected-backend wrapper validates driver/backend concrete type, and
the active-core boundary verifies the logical System-device invariant. This
does not reject force-CPU libcint leaves: wrapper logical device remains
separate from physical backend tensor devices. C8b.2 is complete at the
parameter-tree and functional setup boundary. C8b.3a and C8b.3b complete the
different-size P9 structural/pure-kernel boundary, so overall C8b is complete
at the original documented acceptance scope.

Executable evidence is in [`C8b schema preflight`](C8b-schema-preflight.md).
Independent GFN1/C2 and GFN1/N2 cores have compatible registered tree specs
and stack; the existing Repulsion descriptor kernel maps to serial values and
position gradients. Independently constructed GFN2/H2 trees provide a
positive same-composition stack control. GFN2/H2 versus GFN2/LiH has
incompatible nested pair-plan and shell/AO dimensions even at equal atom
count. Current exact setup has no padding adapter, so the required P9 case of
different-size compositions padded to a shared atom/shell/AO bucket remains
implementation work. Do not satisfy it with dummy physical components.

The C8b.3 mixed-size audit also constructs independent GFN1/H2, GFN1/H3, and
GFN1/H2O cores. H2/H3 differ in atom, shell, AO, pair-entry, index-map,
Repulsion, D3 and ES2 setup shapes; their PairPlan class contexts match, but
their pair-entry counts differ. H2/H2O additionally differs in unique-shell
types, alpha/coefficient tuple arity, pair-plan class contexts, AO stride, H0
unique-species data, and shell-resolved setup. Direct whole-core stacking
rejects both pairs. At that audit stage, no accepted padding implementation
existed; current index Nodes had no validity masks, and PairClassNode flattened
AO indices had to be rebuilt for a common AO stride. The audit is recorded in
the C8b schema preflight and a dedicated structural test.

The following C8b.3a implementation adds `PaddedExactSystemCore`, a separate
registered wrapper around a padded production core and dynamic atom, shell,
orbital, unique-type, and pair-entry masks. Its value-based signature excludes
capacities; deterministic first-seen buckets report max atom/shell/AO/pair
capacity and original input indices. The validated bucket is independent
GFN1/H2 + GFN1/H3: pair-plan flattened AO indices are reconstructed at the
common AO stride, invalid pair entries target only padded AO rows/columns, and
complete wrapped cores stack. The real overlap AO block matches its unpadded
reference. Existing Repulsion energy and position-gradient kernels map over
the complete wrapper with serial parity, zero padded contributions and
gradients, and preserved parameter gradient connectivity. Same-size GFN1/C2
and GFN1/N2 also bucket together. This wrapper is not SCF-ready: H0 and
non-Repulsion family/integral consumers do not yet consume validity masks.
H2/H2O requires canonical mixed-element/shell and setup remapping. C8b.3a is
complete at its scoped GFN1/H2-H3 structural and Repulsion boundary; C8b.3
remains open.

The C8b.3a hardening pass split signature-equal layouts when no input core
realizes the prefix mapping (including C2/N3 in either order), records an
actual template index and its exact capacities, keys `charge_dependent`,
restricts callable contexts to known stateless built-ins, pads shell-resolved
ES3 by shell count, and rejects non-prefix position masks. Padded trees remain
structural wrappers; ordinary H0/integral/SCF dispatch is not enabled.

The implementation split is C8b.1 (registered exact-core/backend schema and
conversion; complete at its scoped boundary), C8b.2 (registered Model
parameter tree and functional setup integration; complete at its scoped
boundary), and C8b.3 (valid mixed-composition padding/partitioning and P9
stack/map acceptance).
The original “stacked padded systems run under `vmap`” wording is split by
owner: C8b validates compatible structural stacking and a pure mapped energy
kernel; E6a owns public whole-System `vmap(system.singlepoint)`; E6b owns
production mixed-System SCF batching. The original “remove Calculator.type”
step is reassigned to E4.6/E6b compatibility policy; C8b preserves the
existing Calculator/System façade conversion contracts and known limitations.

**Final C8b status.** The documented completion criteria are met: production
registered `ExactSystemCore` and selected-backend Nodes; a single Model-level
`ModuleNode` parameter owner with partition/combine and functional replacement
gradients; complete-core structural padding/stacking for independent GFN1 H2/H3
and H2/H2O (including both H2O atom orderings); and mapped Repulsion energy,
real-position gradient, padded-coordinate zero-gradient, and parameter-gradient
parity. C2/N2 remains a compatible equal-size different-composition control,
while C2/N3 is separated when the existing representative cannot realize the
prefix layout. The evidence and node IDs are listed in
[`C8b schema preflight`](C8b-schema-preflight.md). C8b.3 is complete at this
P9 structural/pure-kernel boundary; overall C8b is closed.

The scope remains deliberately bounded. Padded H0 numerical evaluation,
complete padded integral matrices, general padded D3/ES2/ES3, full padded SCF,
public mapped `System.singlepoint`, and production mixed-System batching are
not accepted. The demonstrated mixed-element path is GFN1/PyTorch only; no
claim is made for GFN2, arbitrary custom families, mixed-element libcint, or
CUDA numerical execution. E6a owns public whole-System transforms, E6b owns
production mixed-System batching, E7 owns the broader padding-safety audit,
and E4.6 owns legacy compatibility retirement. These packages remain open.
The ten GFN1 parity failures and five matched-base GFN2 force/Hessian failures
remain inherited numerical issues and were not changed by C8b.

The final C8b.3a bucket check found and fixed one remaining realizability
false positive: a malformed GFN1 D3 `reference_cn` leaf could vary in shape
without affecting the signature, although the adapter retained that tensor
unchanged and `tree.stack` rejected the padded outputs. The adapter now checks
the coupled D3 reference-table shape invariant and keys all dynamic shapes
that padding does not explicitly normalize. Supported padded dimensions stay
capacity-independent. A complete shell-resolved ES3 H2/H3 stack is covered as
structural evidence only. H2/H3 Repulsion values, gradients, and real AO
overlap evidence remain unchanged; padded H0/integral/ES2/ES3/D3/SCF consumers,
H2/H2O canonical remapping, E6a, E6b, E7, and E4.6 remain outside this slice.

The C8b.3b implementation now provides a separate canonical GFN1/PyTorch
adapter for independently constructed H2 and H2O exact cores. The canonical
element order is atomic-number order; shell identity is `(Z, local ordinal,
angular momentum)`. It maps atoms, shells, AOs, unique elements, unique shell
types, H0 tables, basis tuples, ordered pair-class unions, flattened AO
destinations, and the supported classical/interaction setup leaves. Pair
destinations are checked for unique occupancy, and per-input validity masks
identify absent pair entries. Complete registered cores stack, and the
existing Repulsion kernel matches serial energy and real-position gradients;
parameter-gradient continuity and the real AO overlap submatrix diagnostic
also pass. The dedicated C8b.3b tests and the C8b.3a/schema preflights pass in
all five available approved CPU environments. The broad primary selection
reported 674 passes, 15 known numerical failures (10 GFN1 legacy parity,
four LiH/SiH4 force discrepancies, and the LiH Hessian eigensolver issue).
Padded H0/integral/D3/ES2/ES3/SCF evaluation remains outside this adapter's
accepted numerical boundary; C8b.3, E6a, E6b, E7, and E4.6 remain open under
their existing scope.

C8b.3b correctness hardening preserves production ES2 optional-child
structure, includes retained dynamic leaf shapes in compatibility checks,
and revalidates source PairPlan destinations when applying a bucket. H2O
real AO overlap-submatrix parity is verified for both atom orderings. The
updated broad non-slow selection reported 715 passed, 15 failed, and 1
deselected; all five GFN2 PyTorch force/Hessian failures were reproduced at
base `2ac3b81` with the same outcomes and displayed numerical values. The
focused C8b.3b/C8b.3a/schema selection passed 38 tests in each of the five
available approved CPU environments. The padded numerical boundary remains
Repulsion values/gradients and real AO overlap-block diagnostics; C8b.3 overall
remains open.

## C9 final TensorLike/ModuleLike scaffold purge

C9 is the final repository-wide scaffold deletion, not a prerequisite for E6b.
It requires C2-C4, C8a/C8b, E4.6a, and E4.6b: every legacy batch and FULL
fallback owner must first be migrated, globally rejected, or explicitly
retired. Do not mark C9 done while any surviving legacy compatibility object
still inherits from or requires `TensorLike`/`ModuleLike`.

Only migrate objects that survive the restructure. Then remove the scaffolding
from tad-mctc and dxtb and complete coordinated releases. Its acceptance
checks include the dxtb search:

```bash
rg -n "TensorLike|ModuleLike" src/dxtb
```

The command must find no remaining runtime references. Run the corresponding
source-tree search in every participating external package before its release.
C9 cannot be marked complete while any runtime reference remains in dxtb or
those packages; every legacy owner must first be retired or migrated under
E4.6a/E4.6b.

Do not spend time porting a class to Node if B4/B5/B6 is scheduled to delete
the class.

---

# 17. E6a/E6b: batching

Separate conformer mapping over one fixed System from heterogeneous mixed-
System batching. The former does not require `Node.stack`; the latter does.

The E6bcompat-policy gate is complete; its binding, test-backed batch and
legacy FULL contract is in [`E6bcompat-policy.md`](E6bcompat-policy.md).
E6bimpl owns replacements, E6bcompat owns replacement validation, and
E6broute owns activation/packed-route retirement after its independent gates.
E6bimpl.0 adds the first internal numerical consumer: registered canonical
padding feeds the existing PyTorch overlap/multipole and H0 kernels through a
mapped evaluator. Its H2/H3 and H2/H2O parity, derivative evidence, GFN2
readiness boundary, and remaining SCF work are recorded in
[`E6bimpl0-padded-numerical-foundation.md`](E6bimpl0-padded-numerical-foundation.md).
This staged foundation does not change Calculator dispatch or accept
mixed-System SCF.

E6a.0 and its E6a.0a legacy-guess fallback hardening are recorded in
[`E6a0-system-singlepoint.md`](E6a0-system-singlepoint.md). E6a.1's selected
GFN1/GFN2 public conformer, Result/status, charge/spin, and reverse-derivative
evidence is recorded in [`E6a1-conformer-acceptance.md`](E6a1-conformer-acceptance.md).
E6a.2 adds production public status cases in
[`E6a2-public-status.md`](E6a2-public-status.md). E6a.3 accepts public mapping
of the default GFN1/GFN2 built-in aggregate families, active Halogen and
explicit ALPB, position and parameter gradients, conformer independence, and
a selected reverse-over-reverse Hessian; its method/family matrix and
limitations are recorded in
[`E6a3-component-derivatives.md`](E6a3-component-derivatives.md). E6a.4's
official-release forward and derivative evidence is recorded in
[`E6a4-transform-acceptance.md`](E6a4-transform-acceptance.md). The final
audit finds the original GFN1/GFN2 method/component/derivative boundary
satisfied; E6a is **complete** at that boundary. GFN0 IES/SRB whole-System
mapping remains separately scoped. No libcint whole-System mapping,
mixed-System execution, or arbitrary custom callback transform is accepted
by E6a. See [`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md)
for closure evidence.

## Transform-safe single-System entrypoint (E6a prerequisite)

The canonical public conformer call remains the B1 contract:

```python
torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
```

E6a changes the implementation under that existing public call, not the B1
public API. C8a must first make `Result`, `ChargeResult`, `PotentialResult`,
`Multipoles`, and nested outputs pytree-compatible for default `out_dims=0`.
Optional `None` values and classical labels are static pytree context; all
tensor-valued result fields, including classical energies, are leaves. The
exact call must return a `Result` with one new leading conformer dimension on
every tensor field, preserve absent optionals, reconstruct nested values, and
match a Python conformer loop field-by-field on GFN1 and GFN2 with differing
optional integral/multipole fields. No private tuple return or custom
`out_dims` is acceptable.

`System.singlepoint` exposes the transform-safe numerical result and delegates
internally to a private fixed-trip boundary. That boundary must:

- use the fixed-trip SCF controller;
- keep eager warning/error host inspection outside the transformed function;
- contain no tensor-to-Python convergence branch;
- share the same physical map and finalization as eager production;
- return tensor-valued convergence and subsolve diagnostics.

The private fixed-trip boundary owns transformed execution and tensor-valued
convergence/subsolve status; it contains no host warning/error policy. The
ordinary eager `Calculator.singlepoint` adapter owns eager early stop and
warning/error interpretation outside transformed execution. It no longer gets
its single-System result merely by delegating to `System.singlepoint`. Both
paths share the same raw physical map, finalization, spectral/Fermi numerical
primitives, solver-step implementation, and mixer formulas; only controller
and eager-policy ownership differ.
Direct eager `System.singlepoint` intentionally has the same behavior as its
transformed use: it executes the configured fixed trip count even after
convergence, returns invalid Fermi/subsolve conditions as tensor status, and
does not emit eager convergence/Fermi warnings or errors. This is a deliberate
public behavior change that requires migration documentation and direct-System
regressions; it must not happen accidentally as part of a transform refactor.
The private boundary is an implementation detail, not a new public conformer
API. No functorch detection chooses the controller; each API path selects it
statically.

Do not auto-detect functorch state or tensor wrapper types inside physics to
choose a controller. The public method and eager adapter select their
execution boundary statically by API path.

## E6a conformers

The B1 public call is backend-independent in shape, but the E6a transform
guarantee is scoped to Systems configured with PyTorch, the canonical G2
transform/batching driver. Keep libcint eager/reference/analytical regressions
where supported; E6a does not promise whole-System libcint `torch.func`
composition.

Prerequisites:

- full E1 term/component transform support for the tested method;
- B6b-core purity;
- B5.6 public solver-status schema;
- C8a Result pytree representation, accepted with its tensor-built Result probe;
- E2-core structural integral plan;
- E4core fixed-shape SCF;
- the transform-safe single-System entrypoint above;
- transform-critical custom operations solved for the tested method.

E6a does not require C5/C8b mixed-System stackability or E4.6a legacy batch
cull/restore deletion. Its first target maps one fixed PyTorch-driver System
over geometries using the B1 call shown above. No additional public function
is introduced.

Tests:

- run the exact `torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`
  call with default `out_dims=0` and verify a reconstructed `Result`;
- compare every Result field with a Python conformer loop on GFN1 and GFN2,
  including configurations with differing optional integral/multipole fields;
- require tensor-valued fields to gain exactly one leading conformer dimension,
  preserve optional `None` fields, keep classical labels static, batch
  classical energy tensors, and reconstruct nested charge/potential values;
- no `(nbatch, nbatch, ...)` cross blocks;
- gradients have exactly one leading sample dimension;
- map fields/charges when they are batched;
- shared scalar charge works when explicitly broadcast before mapping.
- test direct eager `System.singlepoint` separately from eager
  `Calculator.singlepoint`: converged positive-budget, nonconvergence warning,
  `force_convergence=True`, `maxiter=0`, finite-T/Fermi subsolve failure,
  dynamic damping, iteration count, and selected physical-result parity;
- direct `System.singlepoint` executes all fixed trips, returns tensor status,
  and does not inspect status to warn or raise; `Calculator.singlepoint`
  retains eager early stop and compatibility warning/error behavior.

The public `Result.status` is the stable B5.6 `SCFStatus`, not internal
`SCFSolveResult`: it preserves trajectory and final-spectrum failure codes
separately while `converged` remains outer convergence only. On the currently
supported public configuration domain, test valid zero-T and finite-T solves,
too many electrons, Fermi nonconvergence, electron-count failure,
trajectory-versus-final failure, valid subsolves with outer nonconvergence,
and `maxiter=0`. Tensor-detected spectral status is returned without eager
warning/error interpretation. B5.6 separately tests projection of a
tensor-valued invalid-temperature status; E6a does not promise that a
negative configured temperature returns this status. Negative configured
temperature and negative Fermi `maxiter` remain legacy-owned pending the
global E4.6b configuration-policy decision; E6a does not migrate either
configuration or change Calculator behavior.

E6a.2 has added public status integration for valid GFN1/GFN2 solves,
positive-budget outer nonconvergence, `maxiter=0`, finite-temperature Fermi
nonconvergence, and a lane-local too-many-electrons failure. A zero-budget
fixed-trip call has no trajectory status; its final spectrum status remains
separate. The eager Calculator controller retains its prior failure ordering.
`ELECTRON_COUNT`, trajectory-only failures, and invalid-temperature status
remain covered at their existing lower-level projection boundary where no
stable supported public System fixture was established. See
[`E6a.2 public status integration`](E6a2-public-status.md) for exact fixtures,
classifications, and validation. E6a.2 closed at its scoped status boundary;
the final E6a milestone decision is recorded in
[`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md).

## E6b mixed composition

For the authoritative PyTorch production route, mixed-System batching is
stack plus `vmap` over single-System physics. The no-`batch_mode` and no
driver-internal per-entry physics-loop requirements apply to that route. A
retained libcint eager/reference batch compatibility API may use an outer
Python loop over independent single-System calls; it is not the authoritative
G2 route and is outside G2 no-loop and throughput acceptance.

Prerequisites:

- full E1 term/component transform support for the tested method;
- E6a's accepted transform-safe single-System/public conformer boundary;
- B6b-core and C5 core structural migration;
- C5, C6, C7, and C8b stackable core structural/Model/System values (C8a
  separately covers Result);
- defined padding structure and E7-relevant padding safety;
- E2-core structural integral plan and E4core fixed-shape SCF;
- E6bcompat-policy completed before implementation scope is frozen;
- transform-critical custom operations solved for the tested method.

Build Systems independently in setup.

Pad each System to a defined size bucket.

Stack them with `Node.stack`.

Map over `(system_batch, positions_batch)`.

No physics function receives a `batch_mode`.

No integral driver constructs one System per batch entry internally on the
authoritative PyTorch path. Delete driver-internal batch construction/physics
loops in both PyTorch and libcint drivers; this does not prohibit a separate
outer libcint compatibility API loop over independent single-System calls.

The completed [`E6bcompat-policy.md`](E6bcompat-policy.md) consumes the
accepted B1 public batch API and E4.5b low-level FULL dispatch contracts. It
records the public `Calculator.singlepoint(**kwargs)` behavior separately from
direct `scf.iterator.solve(*args, **kwargs)`: public packed Calculator options
are selected Calculator controls and do not forward engine overrides, while
direct low-level nonempty extras force legacy FULL and reach its constructor.
In particular, public packed unknown kwargs are currently ignored; this is
not advertised as engine-option support, and the new route must reject unknown
public options rather than silently imply they were applied. Low-level unknown
keywords retain their constructor `TypeError`; named legacy option dictionaries
retain their `BaseSCF` constructor semantics. The policy inventories PyTorch
and libcint batches, custom interactions/classicals, positional extras,
temperature and iteration edge cases, incomplete integrals, charge/spin,
fields, and non-SCF/classical-only routes, with a future owner and test for
each. E6bimpl scope is frozen from that policy; E6bcompat validates the
selected replacements or explicit migrations. Retained compatibility may use
an outer composition of independent single-System legacy calls. That does not
make E6broute responsible for final single-System fallback semantics.

E6bimpl owns PyTorch mixed-System construction, stacking/vmap composition,
and correctness, plus any replacement composition required by
E6bcompat-policy. E6bcompat validates each retained replacement or explicit
retirement/migration/error contract. E6broute owns authoritative production
activation and retirement of the packed legacy batch FULL route. E7 owns
padding-safety validation/hardening. E6bimpl may proceed while W2/W3
performance clearances and the broader E7 audit continue. E6broute proceeds
only after E6bimpl correctness, E6bcompat validation, the batch-critical E7
gate, and both E6c1 performance clearances pass. E6c1 requires W2 conformer
clearance from E6a and W3 mixed-System clearance from E6bimpl. Each requires
acceptable measured performance or an explicit architecture-review decision
accepting a regression; missing measurements do not count. The minimum E7
gate inventories padded operations used by the route, verifies safe masking
of singular inputs, and passes E6bimpl value/derivative tests for supported
mapped orders. The broader E7 audit may continue afterward.

This is a correctness/API gate, not part of E6c/G2 performance or no-loop
acceptance. E6a conformer support alone does not retire the legacy batch FULL
route.

## E6c1 pre-retirement performance gates

E6c1 is the required pre-retirement clearance for both branches on the
authoritative PyTorch route against their corresponding T0.8 workloads:

- W2 for E6a conformers mapped over one fixed System;
- W3 for E6b mixed/training-like padded Systems, compared with the recorded
  legacy mixed-batch baseline.

W2 compares conformer throughput with the legacy `batch_mode=2` target. W3
compares mixed-System forward time, force evaluation where applicable, peak
memory, padding overhead, and bucket count/size distribution with the recorded
legacy mixed-batch baseline. Record CPU threads. Parameter-gradient timing is
added once the C8b training tree is available. Each gate needs acceptable
measured performance or an explicit architecture-review decision accepting a
regression; silence or a missing measurement is not clearance. The E7
batch-critical safety gate remains separate. Broadly unacceptable performance
blocks retirement pending explicit review; it cannot silently authorize a
second permanent batched physics core.

## E6c2 final performance acceptance

Record complete W2/W3 results, parameter-gradient timing when available, and
PyTorch-route GPU throughput/memory against the deferred T0.8 GPU baseline.
E6c2 is final G2
performance sign-off. The GPU baseline may remain outstanding after CPU
E6broute retirement because E6c1 already cleared both CPU workloads. An explicit
E6c1 architecture-review waiver may permit route retirement despite a CPU
regression, but it does not satisfy G2's throughput target; E6c2/G2 remains
open until documented throughput targets are met or the acceptance criterion
is explicitly amended. Missing measurements are never clearance.

---

# 18. E7: padding-safety audit

Do not leave this entirely to the end.

Start adding the audit tests while B4/B6/E4core are being rewritten.

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

### E7a batch-critical acceptance — in progress

E7a audits the mapped private GFN1/GFN2 execution boundary, without changing
public Calculator routing. The focused first-order safety selection passed
120 tests in each of the five supported CPU environments. It includes
coincident/displaced inactive coordinates, structural masks, GFN1/GFN2
integral and spectral behavior, GFN2 interactions, scatter reductions,
position/HVP/parameter controls, and the retained padding-ghost regression.

The audit found and fixed an AES2 invalid-pair defect: padded dipole-dipole
entries could be nonzero because only the displacement was masked while the
isotropic reciprocal-distance term remained active. Invalid pair geometry is
now made nonsingular before the norm and reciprocal powers, and all invalid
AES2 pair outputs are cleared. The padded fixed-trip boundary also sanitizes
inactive coordinates before interaction, guess, and classical preparation.

E7a remains open. A new float64 full-energy third-directional derivative
comparison disagrees between mapped and independent serial GFN1/GFN2 results
on PyTorch 2.6 and 2.14; the first divergent SCF operation has not been
isolated. Details and exact test nodes are in
[`E7a-batch-critical-padding-safety.md`](E7a-batch-critical-padding-safety.md)
and [`E7a-acceptance-manifest.txt`](E7a-acceptance-manifest.txt). E7a does not
implement the public packed Result adapter and does not activate a public
route. Broader E7 remains open.

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

E8 is the acceptance package proving G1 third-order transforms on the
canonical PyTorch driver. Existing libcint eager/reference regressions remain
enforced but do not expand E8's transform-driver scope.

---

# 20. T0.8 GPU completion

This is not a blocker for B4-B8.

It blocks E6c2's final GPU/G2 performance acceptance and any D-track GPU
performance conclusion that depends on these measurements. CPU E6broute
retirement is gated earlier by E6c1's W2/W3 clearances and the batch-critical
E7 checks; it does not wait for the GPU baseline.

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

F0 can run after B5 without waiting for broad B6a. It is an early, bounded
state-history regression whose acceptance itself verifies fresh setup for the
selected GFN1 H2O training fixture; it does not claim broader B6a completion.

Status: done. F0 uses `torch.func.functional_call` and `torch.func.grad` with three active GFN1 H2O parameters, constructs a fresh System for every evaluation, checks finite-difference agreement, applies functional updates that reduce an artificial energy loss, verifies the original ParamModule is unchanged, and retains a mixed force/parameter derivative check. This does not start the broader training track.

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
- DFT-D3 reference data;
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

Do not start compile-oriented restructuring before E4core and its relevant
implementation (E6a for conformers, E6bimpl for mixed Systems).

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

Do not add size buckets until E6bimpl mixed-batch execution is correct and
their benefit is measured.

---

# 25. Revised dependency order

## Dependency contradiction in the previous ordering

The previous requirements could not be executed literally: B6b required
repository-wide deletion of every cull/restore path; C5 started after B6b and
required a cull-free `IndexHelper`; C8b depended on C5; and E6bimpl required both
B6b-core and C8b. Yet E4.6a required E6broute and E7 to migrate batch execution
before batch cull/restore could be deleted, while the legacy batch FULL route
still needed that machinery. Thus B6b-core/C5/C8b/E6bimpl could not all complete
before the migration that was itself required to finish cleanup. The split between B6b-core and
legacy compatibility purge removes this cycle. C8 is now split as well:
C8a Result pytree support gates E6a, while C8b Model/System stackability gates
E6bimpl. The E6a public method also uses fixed-trip status-only semantics directly;
Calculator retains eager policy through its separate adapter.

## Acyclic dependency DAG

```text
T0.3/T0.4 + E0 + E4core + E1.1/E1.2 ──> E3.0 diagnosis-only preflight
E3.0 ──> E3.1a physical-root strategy/reference gate (complete)
E3.1a ──> E3.1s-a dxtb-local invariant SCF density core
E3.1s-a ──> E3.1s-b invariant entropy/free-energy spectral matrix
E3.1s (E3.1s-a + E3.1s-b, complete) ──> E3.1b transform-compatible implicit-root implementation
E3.1b + full E1/E1.3 + E2-core + E4core ──> full E3 acceptance

B3a ──> B5; B4 ──> B5
B3a + B4 ──> E2-core fixed-shape single-System PairPlan/integral path
B5 + E4core ──> B5.6 ──> C8a Result pytree
B4 + B5 + B6a-SCF-core ──> E4core
B6a-SCF-core + E4core + E4.5b ──> B6b-core ──> C5 ──> C6/C7
final B3b masks/index mappings and transform-critical gathers ──> final B3b closure
legacy classical cache/lifecycle retirement ──> separately scoped compatibility cleanup
final B3b + C5/C6/C7 + B2 ──> C8b
B7 ──> B8a; B8a + full E3 ──> B8b; B8b + full E3 ──> E8
B5 ──> F0 bounded training/state-history invariant

B6b-core + full E1 + E2-core + E4core + private fixed-trip boundary
  + B5.6 + C8a ──> E6a conformer vmap (complete at the documented GFN1/GFN2 PyTorch boundary)
  ──> E6c1-W2
B1 batch API + E4.5b ──> E6bcompat-policy (complete; see policy document)
E6a + final B3b + C5/C6/C7/C8b + E2-core + E4core
  + E6bcompat-policy + defined padding
  ──> E6bimpl.0 mapped padded integral/H0 foundation (GFN1/PyTorch)
  ──> E6bimpl.1a padded GFN1 SCF inputs and masked spectral evaluation
  ──> E6bimpl.1b mixed-System fixed-trip SCF and component consumers
E6bimpl ──> E6bcompat validation
E6bimpl + batch-critical E7 ──> E7 batch-safety gate
E6bimpl + T0.8 W3 ──> E6c1-W3
E6c1-W2 + E6c1-W3 + E6bcompat + E7 batch gate + E6bimpl
  ──> E6broute authoritative mixed-System route; retire packed legacy batch FULL
E6broute ──> E2-legacy packed-driver loop retired
E6broute ──> E4.6a ──> E4.6b ──> C9 (with C2-C4 and C8a/C8b)
remaining single-System fallback migrations/retirements ──> E4.6b
E6broute + T0.8 GPU + E6c1 ──> E6c2 final W2/W3/GPU G2 acceptance

full E3 + B8b ──> D2
E4core + E6a + T0.8 GPU ──> D1
B8b + C9 + E8 + D1 + D2 + F0 + T0.12 ──> F1
F1 ──> F2a ──> F2b ──> F2c ──> F2d ──> F2e
F2d + E6broute ──> Track D3; E6broute + D3 ──> D5 (conditional)
D1 + D2 + D3 ──> D4 compile policy/tooling
B6b-core + C1b + full E3 + E4core + E6a ──> F3 ──> F4 (conditional)
```

Recommended early ordering remains: R0, E0, E1, B4/B5/B6a-SCF-core, B7,
E2-core, and E4core according to their prerequisites. After E4.5b,
diagnosis-only E3.0 and B6b-core may proceed independently; E3.1 may implement
SCF differentiation after E3.0 without E1.3. E1 is now complete, so the E3
end-to-end acceptance gate is unblocked but still requires its own validation.
B5.6 has closed the public solver-status fields, unblocking C8a;
C8a's tensor-built Result probe is independent of SCF control. C8a plus the
private transform boundary unblock E6a, where the exact B1 public call is
finally tested. E6a + final B3b/C5/C6/C7/C8b unblock E6bimpl. E6a does not
wait for B3b, C5/C8b, or E4.6a.

B5.6 is complete and verified; C8a is complete. B6b-core and C5 are
complete. C6 is complete at its exact-core/legacy boundary, and C7 is
independently accepted. C8b.0 preflight, C8b.1 production core/backend
registration, C8b.2 Model parameter-tree integration, and C8b.3 structural/P9
padding acceptance are complete at their documented boundaries. E1 and E6a
are complete. E3.1a, E3.1s and E3.1b are complete; full E3 end-to-end
acceptance remains open for its own GFN1/GFN2 reference/derivative matrix.
B6b-core removes mutable legacy index and interaction containers from the
functional SCF core while preserving their legacy consumers; it does not
include classical DFT-D3 setup/cache migration or repository-wide compatibility
cleanup. C5 migrated `CoreIndexData` to tad-mctc `Node`; B5.6 completed the
public tensor-valued SCF solver-status schema. C6.1 migrated the ES2/ES3
single-System setup values; remaining C6 families and C7 are separate
structural packages.

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

- at that milestone, singlepoint still used legacy SCF while consuming the
  returned integral values.

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

### PR E4.1: functional simple solver (historical package plan)

- at that stage, raw map and production driver stay unchanged.
- characterize forward parity and transform behavior before routing.

### PR E4.2: functional Anderson state

- keep Anderson history explicit; Broyden remains future implicit/E3 work.
- do not alter the production driver.

### PR E4.3-E4.6: staged solver completion

- E4.3 finite-temperature occupation/Fermi;
- E4.4a potential representation (done); E4.4b Fock representation (done);
- E4.5a dynamic damping and eager compatibility (done); E4.5b production routing
  (done);
- E4.6a batch cull/restore cleanup (deferred until E6broute retires the legacy
  production batch route after batch-critical E7 and E6c1 W2/W3 clearance);
- E4.6b legacy full-loop deletion (blocked on compatibility fallback migration
  or explicit retirement).

### PR C5+: Node migration from leaves upward

- never mix Node migration with physics changes.

### PR C8a then E6a/E6b: `vmap` after Result pytree and single-System transform tests.

E6a preserves the exact B1 conformer `vmap(system.singlepoint, ...)` call with
default `out_dims`, after C8a makes Result pytree-compatible. E6a explicitly
changes direct-System calls to fixed-trip/status-only semantics; Calculator
keeps eager early-stop/warning/error behavior in its own adapter. E6bimpl adds
mixed-System stacking/composition; E6broute owns production batch-route
activation and legacy-route retirement. The route is authoritative only after the batch-critical E7 gate and both
E6c1 performance clearances pass: W2 for conformers and W3 for mixed Systems.

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

## New-core structural gate (not repository-wide legacy purge)

Before calling the new structural core complete:

- B2, final B3b core layout (masks/index mappings and transform-critical
  gathers), B4, B5,
  B6a-SCF-core, B6a-SCF-validation,
  B6b-core, B7, and B8a complete;
- no calculator/core caches;
- no mutable external fields;
- Result immutable;
- core evaluation is System + explicit inputs -> Result;
- T0.2 reproduced.

This milestone does not imply completion of E4.6a/E4.6b or C9 repository-wide
compatibility cleanup.

## Transform gate

Before claiming G1/G2 architecture complete:

- E0 decisions implemented;
- degenerate eigensolver cases correct;
- NO2 GFN2 third order correct;
- DFT-D3/tad-dftd3 forward-mode blocker fixed;
- E4core functional SCF;
- E3 required derivative modes pass;
- E6a conformer `vmap` passes;
- no cross-system derivative blocks.

## Node gate

Before C9:

- all surviving core persistent objects are Nodes or plain immutable values;
- no cull/restore remains in those surviving core objects;
- no TensorLike subclass remains in the surviving core;
- mixed padded Systems stack with identical tree structure.

This gate does not require separately isolated legacy fallback adapters to
become Nodes or lose their cull/restore APIs. Repository-wide compatibility
purge is owned by E4.6a for batch machinery and E4.6b for remaining legacy FULL
objects, after their fallback routes are retired or migrated.

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

Success means the single-System implementation is the only physics
implementation for the authoritative PyTorch production route. Batching is
composition with `vmap`/stacking, not a parallel batch-mode codebase. G2's
no-per-entry-loop and throughput criteria apply to that PyTorch route. A
retained libcint eager/reference compatibility API may compose independent
single-System calls with an outer loop; that composition is outside G2
production/performance acceptance and must not become a second physics
implementation. Legacy PyTorch production batch modes are deleted after route
migration; libcint remains supported within its documented
eager/reference/analytical capabilities.

## G3

Success means setup is differentiable from parameter tables to energy/forces, training uses functional trees, and extending the element axis does not perturb existing rows.

## G4

Success means the ML term is an energy contribution with its potential derived from that energy and passes the same transform contract as physical terms.

## G5

Success means measured speed/memory improves without creating a second semantics or weakening G1-G4.

---

# 30. Roadmap handoff after C8b

**Current status and ownership.** C8b is complete at its documented P9
structural and pure-kernel boundary. No C8b blocker remains. E6a owns the
public whole-System transform contract; E6b owns production mixed-System
batching and numerical route integration; E7 owns the broader padding-safety
and derivative audit; E4.6 owns legacy compatibility retirement. These remain
separate open packages and must validate their own consumers before using
padded cores numerically. The dependency DAG in section 25 remains the source
for selecting and ordering downstream work. E6bcompat-policy may proceed
early as specified there; E6a and E6b implementation remain subject to their
listed prerequisites. Do not treat the historical C8b status below as a
current blocker or prerequisite.

E6bimpl.0 has now established a private, mapped GFN1/PyTorch padded
integral/H0 evaluator using the existing physical kernels. It passes
source-to-canonical physical AO block parity for H2/H3 and H2/H2O, optional
multipole levels, and selected position/parameter derivatives in five CPU
environments. This begins numerical consumption downstream of C8b; it does
not change C8b's acceptance boundary, add mixed-System SCF, or activate
Calculator batching. See
[`E6bimpl0-padded-numerical-foundation.md`](E6bimpl0-padded-numerical-foundation.md).

E6bimpl.1a adds a private GFN1/PyTorch padded SCF-input and masked spectral
boundary over the registered canonical core. The existing spectral/Fermi,
density, charge, and inverse-Cholesky primitives are reused. A reproduced
negative-index sum-scatter contamination was fixed by excluding invalid
sources before reduction; optional static output extents keep padded mapping
free of tensor-to-Python size inference. The new boundary passes source/mapped
spectral parity and selected value/derivative/status tests for H2/H3+,
H2/H2O, reversed input order, and a noncontiguous canonical AO layout. See
[`E6bimpl1a-padded-spectral.md`](E6bimpl1a-padded-spectral.md).

E6bimpl.1b adds a private mapped fixed-trip GFN1 evaluator that reuses the
existing charge/potential/Fock representation solvers and linear/Anderson
mixers, prepares captured ES2/ES3 descriptors, evaluates the exact-System
classical descriptors, and returns a registered padded `Result` with public
`SCFStatus`. Decoding uses canonical AO maps separately from the
energy-ordered molecular-orbital columns. Its 35-test padded SCF/spectral
module passes in all five approved CPU environments. It covers mixed
H2/H3+, open-shell H2/H3, H2/H2O, reversed H2O/H2, H2O/CH4, dipole and
quadrupole integral results, all supported GFN1 SCP/mixer pairings, finite
temperature, one-shot evaluation, lane-local Fermi failure, and position,
parameter, JVP/jacfwd/jacrev/vmap-grad checks. The primary-environment
combined E6bimpl.0/1a, C8b, E6a, solver/status, padding-safe-scatter, and
E6bcompat-policy selection passes 248 tests. Explicit intermediate-trajectory
instrumentation and remaining component/negative-case coverage are still
open; E6bimpl and E6bimpl.1b remain open. Public Calculator batching is
unchanged. See
[`E6bimpl1b-mixed-gfn1-scf.md`](E6bimpl1b-mixed-gfn1-scf.md).

E6bimpl.2b extends the private fixed-trip path to default mixed GFN2/PyTorch
Systems. Its original implementation and the E6bimpl.2b-F final acceptance
record together establish independent ES2/ES3/AES2/D4SC references, Repulsion
and classical D4 parity, the shared static EEQ cutoff rule, all six supported
component-gradient controls, charged/open-shell lanes, lane-local
too-many-electrons status, finite-temperature mask/free-energy parity, and
mixed position/parameter response. The 41-case package suite passes in all
five CPU environments. Six unrelated functional SCF failures in the broader
primary selection reproduce on the exact package base. No public Calculator
batch route is activated. E6bimpl.2c owns the remaining cross-method,
component-combination, derivative-accuracy, and compatibility-readiness
matrix. See [`E6bimpl2b-final-acceptance.md`](E6bimpl2b-final-acceptance.md)
and the historical implementation record
[`E6bimpl2b-mixed-gfn2-scf.md`](E6bimpl2b-mixed-gfn2-scf.md).

E6bimpl.2c completes the private numerical acceptance review for the
supported mixed GFN1/GFN2 PyTorch paths. It records the 16 frozen numerical
requirements against owning code/tests, cross-method Results and status,
padding/order invariance, component coverage, derivative isolation, and a
GFN2 full-energy Hessian-vector comparison. The GFN1 51-case and GFN2
41-case mixed execution suites pass in all five CPU profiles; the added GFN2
HVP control also passes in all five. The completed primary focused selection
contains 431 unique nodes with 425 passing and six previously matched
functional SCF baseline failures. A stale C8b test that expected canonical
padding to reject GFN2 was corrected to assert method-specific bucket
separation. No production code or public Calculator route changed.

The **private E6bimpl numerical implementation and its retained composition
obligation are complete**. E6bimpl.3 uses independent existing single-System
setup/evaluation, explicitly selects the legacy SCF engine for
libcint/extensions/engine overrides, returns heterogeneous physical Results
in source order, and rejects mapped-eligible systems instead of acting as a
fallback. Its 24 focused tests pass in all five approved CPU profiles. They
cover real CPU libcint H2/H2O, custom interaction and classical contributions,
cache refresh, fields, option identity/merge and malformed-option behavior,
heterogeneous classical labels, serial error parity, ordering and gradients.
The broader primary completed selections pass; the six functional-SCF
failures match their previously recorded baseline, and the interrupted
601-node run is not counted. **E6bimpl.3 and overall E6bimpl are COMPLETE.**
E6bcompat now completes the independent differential review of mapped and
retained execution. Its public GFN1 mode-1/mode-2 and GFN2 Fock controls compare
the unchanged packed Calculator with independent serial Systems and the
private mapped evaluator; retained libcint/custom-composition, option/error,
charge/spin and shared-field contracts are separately exercised. The frozen
policy matrix and exact acceptance nodes are recorded in
[`E6bcompat-final-acceptance.md`](E6bcompat-final-acceptance.md) and
[`E6bcompat-acceptance-manifest.txt`](E6bcompat-acceptance-manifest.txt).
Public Calculator dispatch remains unchanged. E6broute is still blocked on
its public packed Result adapter, E7 batch-critical safety, and E6c1 W2/W3
performance gates. See
[`E6bimpl3-retained-legacy-composition.md`](E6bimpl3-retained-legacy-composition.md),
[`E6bcompat-policy.md`](E6bcompat-policy.md),
[`E6bimpl2c-final-numerical-acceptance.md`](E6bimpl2c-final-numerical-acceptance.md)
and the exact E6bimpl.2c acceptance node inventory in
[`E6bimpl2c-acceptance-manifest.txt`](E6bimpl2c-acceptance-manifest.txt).

### Historical implementation/status record

B3b-Final.3 accepted the staged exact/packed layout and compatibility
boundary. The C6-Final.0 audit found that persistent classical/interaction
descriptors hid their nested setup Nodes from standard pytree traversal.
C6-Final.1 closes that descriptor gap; `ExactSystemCore` is
a frozen non-Node value for exact Systems that aliases existing setup Nodes
and owns effective dtype; packed modes 1/2 retain their actual mode and legacy
cache. C6-Final.1a removes temporary descriptor Nodes from repeated exact-core
coherence validation while preserving the same identity and metadata checks.
This does not remove the compatibility facade or claim stackability.
The Anderson position-gradient discrepancy failed identically on the
B3b-Final.1 base and candidate and remains unresolved. B3b-Final.2a restores
public forwarding for arbitrary `core_index_owner` keywords while retaining
the trusted internal map route, and validates setup coherence and floating
device/dtype. The evidence and inherited GFN1 parity failures are recorded in
[`B3b final layout inventory`](B3b-Final-layout-inventory.md). Final B3b is
complete. C6-Final.2 cross-family numerical/transform
acceptance passed the available family Node, functional-map, implicit-root,
E0, and pure integral controls. C6-Final.3 exercised current-source
descriptor/model, packed/cache, interaction-family, and selected PySCF-backed
libcint controls in Python 3.12/Torch 2.14.1. Focused current-source tests cover all family core Nodes, Calculator/result
compatibility, packed mode energy/gradient parity, and selected libcint/System
paths. Only this one of the five approved environments was available. C6 is
complete at its exact-core/legacy boundary. C8b.0 completed schema and
stacking preflight; C8b.1 registered the production exact core/backend. Model
parameter migration and padded P9 stacking remain open.

The C6.1 ES2/ES3, C6.2 AES2, C6.3 GeneralizedBorn, C6.3a mixed-device
scalar hardening, C6.4 D4SC, C6.4a setup-authority hardening, C6.4b
interaction-identity hardening, C6.5 deterministic component-list ordering,
C6.6 Repulsion, C6.7 ShortRangeBond, C6.8 IES, C6.9 Halogen, and C6.10
classical D4 setup Node migrations are complete. C6.10 retains direct
classical D4 setup/cache APIs and legacy custom model/damping routes, while
its System classical-cache entry aliases the captured D4 Node. The classical
D4 suite passed 109 tests, and the focused D4 Node suite passed 9 tests in
each of the five approved CPU environments. The C6.10 E0 selection first had
the tracked `eigensolver.small_gap_aufbau:order2` failure (23/24); a repeat
passed 24/24. The exact C6.10 base passed 24/24, and isolated reruns passed on
both base and candidate, so the intermittent observation remains unresolved.
The ten
historical NO2/GFN2 reference rows were
validated across low-order, second-order, and third-order selections: 8 passed
and 2 expected XFAIL; the third-order pair required a separate 495-second
two-worker run. C6.4c's matched NO2 worker-mode diagnosis did not
reproduce the reported third-derivative NaN; its cause remains unresolved and
the instability is not fixed. The C6.5 insertion-order correction is a
separately established structural requirement and is not established as the
NaN's cause. C6.6 retains legacy Repulsion setup/cache behavior for direct and
packed-batch use. C6.9 retains direct and packed Halogen compatibility caches,
and aliases the exact System-owned core setup through the classical cache. A
read-only review of installed tad-multicharge 0.7.0 confirmed nested EEQModel
parameter and coordination-model tensor leaves; no C6.8 defect was found. The
C6.7 broad-suite failures remain incompletely classified, and the E0
intermittent gradcheck, ten GFN1 density-parity failures, and original NO2
worker issue remain unresolved. See the
[diagnostic note](C6.4c-NO2-worker-investigation.md). C6 is complete at the
accepted exact-core/legacy compatibility boundary; the unclassified C6.7
broad-suite node manifest remains unavailable as recorded historical evidence.
C8a — complete
public Result pytree representation — is complete and its E6a output-container
prerequisite is satisfied. E6a still requires its other listed prerequisites.
E3.1b transform-compatible implicit physical-root implementation is complete.
Full E3 is acceptance-only and remains open for its own end-to-end matrix.
E1.3 now enables the DFT-D3-enabled GFN1 row; do not approximate that row by
omitting DFT-D3. B5.6 —
public tensor-valued SCF solver-status schema — is complete and verified.
C7 setup acceptance and final B3b exact-core/compatibility acceptance are
complete. B3b-Final.3 established ordered classical and interaction descriptors
for exact built-ins, with legacy identity associations, extensions, and packed
Calculator state kept at the compatibility boundary. C8b.0 completed the
schema/alias/partition/stacking preflight; C8b.1 registered the production
exact core/backend, and C8b.2 added the Model parameter tree and functional
setup boundary. C8b.3 remains implementation work. B2, C5, C6, C7, and B3b
are accepted prerequisites. C9 is deferred until
legacy-owner retirement through E4.6a/E4.6b.

E3.0 diagnosis-only spectral/NO2 preflight is complete. Its report identifies
the historical failure at the complete finite-tolerance fixed-point response /
finite-root trajectory boundary; it does not assign the cause to an individual
Anderson operation. E3.1a's corrected physical-root reference gate and
E3.1s's dxtb-local invariant density/free-energy contract are complete.
E3.1b is complete: it attaches derivatives to the convergence-tested
functional `root_x`, uses only the configured raw map in the root equation,
and keeps representation-specific finalization outside the implicit
primitive. Its linear-solve convergence threshold is dtype-aware; float64
retains the existing criterion and float32 uses a ten-epsilon relative floor.
The measured NO2 third-order implicit path is slower and uses more peak memory
than tight unrolling, so performance remains a separately measured debt.
Whole-System transforms remain E6a scope.

Already complete for this path: E0; E1.1/E1.2; E2-core; E4core through
E4.5b; B4/B5; B6a-SCF-core and B6a-SCF-validation; B7.

Still separately open: C8b.3; E1.3; E6a/E6b; and broader B6a/E4.6 legacy
component/cache lifecycle retirement and compatibility-owner purge. C6 is
complete at its accepted core/legacy boundary. B6b-core and C5 are complete;
C9 remains deferred.

## Historical package records

Package record: the IES family and classical DFT-D4 compatibility hardening were validated from task base `12d0182d6b0dff7214bddfcfe8f34eb22f8e77e2` and implemented in `635a6bbfbd1dbe07eabce68d46a42e3d03f35de1`. The independent Luna/high review found no blocker or major issue. The classical halogen family was then validated from task base `c71e4a082ddfd76a2683b005fe3f8420553a16fd`; its fixed-shape kernel changed `halogen.positions:vmap` from ERROR to PASS without changing reference values. Two broad-suite seeds each reported four implicit-SCF memory-leak failures; all seven distinct failing nodes reproduced against the task base with the same one-tensor leak signature. The independent Luna/high review found no blocker or major issue. D4SC, DFT-D3, and other B6 families were not migrated in the halogen package.

The AES2/multipole package was validated from task base `f999f9fa3986e4bd73da6ad60d572bbb3cfedcde`. Halogen setup masks now honor the configured species lists. AES2 dipole vmap changed from ERROR to PASS, and the new position target passes forward AD, JVP, jacfwd, reverse orders 1–3, and vmap. Reference and parameter verdicts are unchanged. Broad seeds `20261010` and `20261011` showed nondeterministic implicit-SCF memory-leak failures under work stealing; every distinct failing node was reproduced against the task-base snapshot. The only current broad XPASS is the established implicit-matrix case; the base additionally XPASSes the old strict AES2-vmap xfail, as expected. Luna/high review found no blocker or major issue. D4SC, ALPB, fields/B7, E4, and other interaction families were not migrated.

The D4SC package was validated from task base `79d6f1115f19548b47b1d4ed55607d61dc7d172c`. On the task base, a fresh parameter-coverage generation produced `zero/zero` gradients for the five GFN1 xbond entries (`Bi`, `C`, `Fe`, `O`, `Pb`) while the checked-in base status still said `none/none`; the committed `zero` status is stale-baseline catch-up, with the `no effect on this set` verdict unchanged. D4SC now uses frozen `D4SCSetup` and fresh call-local model/CN/dispersion-matrix data; culling no longer mutates the persistent component model. The existing charge transform target remains green, and `d4sc.positions` passes forward AD, JVP, reverse orders 1–3, and vmap. Potential VJP matches its jacfwd result, and same-System neutral/charged/geometry/neutral SCF calls match a fresh neutral evaluation. D4SC references, classical-D4 regressions, component/reference baselines, and parameter verdicts are unchanged. Broad seeds `20261010` and `20261011` each had four implicit-SCF memory-leak failures; the union of eight exact nodes reproduced on the task base with the same one-tensor leak signature. One established implicit-matrix XPASS remained. Luna/high review found no blocker or major issue; it noted that custom external `D4Model` subclasses must accept the standard constructor and have extension state limited to tensors, scalar values, and basic containers, with unsupported state rejected explicitly. ALPB, DFT-D3, B7, E4, E6, Node, GPU, and other families were not migrated.

The D4SC closure and ALPB package was validated from task base `170babe22d825a9d4680bbe967d61ce79ac16012`. Direct and System D4SC paths preserve sparse parameter mappings with `s9` or `s10` omitted; the stale-component test now reassigns its tensor without mutating the source ParamModule leaf. ALPB now uses frozen `GeneralizedBornSetup` and fresh per-call Born radii/matrix data, with exact System evaluation bypassing persistent `get_cache()`. P16 and Still references are unchanged. Preflight geometry and charge transforms passed; `alpb.positions` passes forward AD, JVP, reverse orders 1–3, and vmap; potential and Born-radii transform checks pass. Charge derivatives match the monopole potential, and dielectric and mixed position/dielectric gradients are finite and nonzero. The five-environment D4SC/ALPB focused matrix passed (75 tests per environment), and the later mixed-derivative/Calculator-reset additions passed in all five environments (2 tests per environment). Component and reference baselines passed; parameter-coverage verdicts were unchanged and generated floating-detail noise was restored. The ALPB-enabled packed legacy batch remains unsupported as on the task base; the non-ALPB packed case passes. Final broad seed `20261010` reported five memory-leak failures and seed `20261011` reported four, all in `test_a_memory_leak/test_scf.py`; repeated identical seeded work-stealing runs changed the exact parameterized nodes. The task base reproduced the same failure family under both seeds, and every exact node from the final current runs failed when rerun alone against the task base. One existing implicit-matrix XPASS remained. The independent Luna/high review found no remaining findings. No DFT-D3, B7, E4/E6, Node, GPU, or unrelated component-family work was done.

The ALPB fresh-setup and F0 package was validated from task base `cf0c98266d8c12c1fa3d572c7acc8d65fec528ae`. Fresh GeneralizedBorn construction and setup inside a transformed function pass dielectric reverse, JVP, jacfwd, and central finite-difference agreement; a tensor-valued Born-scale configuration also passes fresh-setup JVP. The `bool(alpbet > 0)` setup branch remains unchanged because scalar dielectric differentiation passes; heterogeneous solvent-configuration setup vmap is outside the single-System contract. F0 exercises full GFN1 H2O with DFT-D3 enabled, `functional_call`, and fresh System construction on each evaluation for `element.O.arep`, `element.O.gam`, and `repulsion.effective.kexp`. Initial gradients agree with central finite differences, four fixed functional updates monotonically reduce the loss, original selected and unselected ParamModule leaves remain unchanged, and the mixed force/parameter derivative is finite and nonzero. Torch 2.6 exposed `Tensor.tolist()` on a functorch GradTrackingTensor during H0 setup; replacing it with a structural integer-list conversion made F0 pass in all five supported environments. Component and reference baselines pass, parameter-coverage verdicts are unchanged, and the generated finite-difference detail noise was restored. Broad seeds `20261010` and `20261011` had three and one implicit-SCF memory-leak failures respectively; the four distinct exact nodes reproduced when isolated on the task base. Each run also had the established implicit-matrix XPASS. Requested Luna/high review found no blocker or major issue; its minor scalar-state recommendation is covered by an explicit assertion. No references, tolerances, xfails, DFT-D3 behavior, B7, E4/E6, Node, GPU, or other component families were changed.

B7 compatibility hardening: exact Calculator field defaults now reject any retained same-label interaction during construction, rather than deferring the ambiguity to evaluation. Custom same-label field subclasses remain ordinary legacy interactions for energy and forces; generic standard-field response methods report that they cannot infer a differentiation variable from an arbitrary extension. On the pre-B7 base, same-label `ElectricField` subclasses supported energy, dipole, and polarizability, while same-label `ElectricFieldGrad` subclasses supported energy, quadrupole, and numerical quadrupole when their perturbation tensors required gradients. The explicit-field architecture intentionally narrows those response APIs to exact/default/explicit standard perturbations; no interaction mutation is restored. At that historical point E6a.0 exercised the exact public whole-System `vmap` call for the initial GFN1/PyTorch slice; full method and derivative acceptance remained open then. The final E6a boundary is now closed as recorded in [`E1.3-E6a-final-acceptance.md`](E1.3-E6a-final-acceptance.md). Eligible `System.singlepoint` uses fixed-trip tensor status, while Calculator retains host warning/error inspection outside transforms. Static legacy-owned or unsupported custom routes are outside this initial transform guarantee.

E0's convergence-tolerance contract is now explicit: canonical full/unrolled
convergence requires both raw-residual x norms to meet `x_atol` and
`x_atol_max`; `f_atol` remains transitional legacy solver configuration and
does not add a full-mode convergence condition. The focused H2O regression
demonstrates that changing `f_atol` alone leaves full-mode outputs and
stopping unchanged, while changing the x tolerances changes the stopping
point.

E2-core's single-System prerequisite is complete and is no longer a blocker
for E4. The setup-owned `PairPlan` is reused by overlap, dipole, and
quadrupole evaluation across geometries. The per-entry structural setup loop
in the legacy packed `IntDriverPytorch` remains until E6broute replaces that batch
path; no batch behavior was changed here. E0 decisions are accepted and
remain binding design constraints. The package records above describe their
own validation points and do not replace the current status and next action
at the start of this section.

