# Track B: Evaluation API, results and state

**Purpose.** Replace the calculator's hidden state (five caching layers, mutable components, mutable configuration) with the model/system/result structure from principle P1. Afterwards nothing is keyed or invalidated: what is reused is an explicit value.

**Why this track comes first among the refactors.** It removes most of the mutation that the `Node` migration (TC) would otherwise have to handle, and it creates the pure functions that TE needs (`setup`, the SCF step, property functions).

**Today's state (at `46af7bc`).**

| Layer | Where | Problem |
| --- | --- | --- |
| Result store | `CalculatorCache` (`calculators/types/base.py:67`), `@cdec.cache` (`calculators/types/decorators.py`) | Always written, even with `CACHE_ENABLED = False`; used to pass SCF outputs between methods; key ignores calculator state and uses data pointers |
| Component caches | `Component._cache` (`components/base.py`) | Keyed on values of `numbers`/`positions`; ignore tensor identity and `requires_grad` |
| Per-call caches | `InteractionListCache`, `ClassicalListCache` | Already explicit values; culled in place in the SCF |
| Driver setup | `IntDriver._positions`, `is_latest` (`integral/driver/base.py:116`) | Keyed on position values; libcint wrapper captures positions |
| Stored integrals | `Integrals` container, `BaseIntegral.matrix` | Rebuilt in place; results alias live objects |

---

## B1 Decision note: evaluation API and state

**Goal.** Fix the API shape before code changes. One to two pages.

**Questions to answer.**

1. **Levels.** Confirm model, system and result (P1). What does each hold, and which methods does each expose?
2. **Result contents.** Which SCF outputs are always included: energy and per-term energies, charges (and multipoles), density, coefficients, orbital energies, occupations, potential, Fock matrix, overlap and multipole integrals, iteration count, convergence flags. Default proposal: all of them; memory is the only cost.
3. **Property functions.** Signatures of `forces`, `hessian`, `third_order`, `dipole`, `polarizability`, `hyperpolarizability`, `dipole_derivative`, `polarizability_derivative`, `ir`, `raman`, `vibration`. Do they take `(system, positions, ...)`, a result, or both?
4. **Fields.** Electric field and field gradient become `singlepoint` arguments (B7). Confirm.
5. **Batching API.** Single-system functions plus `vmap`; explicit batch entry points (for example `dxtb.batch.singlepoint`) instead of guessing from shapes. `batch_mode` removed.
6. **Parameter storage.** Model holds per-element tables. Trainable subsets (needed by F2c): fixed base plus trainable block, scattered in during `setup`.
7. **Dtype and device.** Read from the tensors, not stored in configuration. Float64 policy.
8. **Numerical derivatives.** Separate functions taking the same inputs; no mutation.
9. **ASE.** Out of core; optional adapter (B9).
10. **Removal list.** Every public name removed, with its replacement, as the basis for the migration guide.

**Done when.** The note is agreed and linked from `00-overview.md`.

**E6a migration note.** The public conformer call remains
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`. After E6a,
direct eager `System.singlepoint` intentionally uses fixed-trip execution,
returns tensor-valued convergence/subsolve status, and does not emit eager
SCF/Fermi warnings or errors. `Calculator.singlepoint` becomes the eager
compatibility boundary and no longer gets its result merely by delegating to
`System.singlepoint`; it retains early stop and warning/error policy. Both
share the same physical map, spectral/Fermi primitives, solver step and mixer
formulas, and finalization. This behavioral migration must be described in
the public migration guide; callers needing eager compatibility policy should
use Calculator.

**Needs.** T0.11. **Size.** S.

---

## B2 Frozen configuration

**Goal.** Configuration becomes an immutable value, so the model can hold it as pytree context.

**Today.** `Config` (`calculators/config/main.py:41`) is a plain mutable class with identity equality. It carries `device` and `dtype`. It is changed after construction: the calculator's `__init__` sets `opts.ints.level` and `opts.batch_mode`, `Calculator.type` writes `opts.dtype`, and the CLI sets `calc.opts.scf.*` on an existing calculator.

**Steps.**

1. Convert `Config` and all sub-configs to frozen dataclasses with value equality and hashing.
2. Remove `device` and `dtype` from the configuration.
3. Remove `batch_mode` from the new core configuration (superseded by B1 item 5). The old Calculator may continue deriving it for its legacy adapter; that adapter's batch state is retired only when E6broute makes mixed-System production routing authoritative. Conformer throughput uses the B1 `vmap(system.singlepoint, ...)` contract in E6a and does not require a public `batch_mode` option.
4. Replace post-construction adjustments (integral level from the method) with construction-time derivation.
5. Change the CLI to build a new configuration with `replace()`.

**Done when.** No assignment to a configuration attribute anywhere outside construction; equal settings compare equal.

**Status: done.** Frozen dataclasses (`Config`, `ConfigSCF`, `ConfigFermi`, `ConfigIntegrals`, `ConfigCache`, `ConfigCacheStore`); user input (strings, lists) is converted by the `create` class methods, the constructors take final values only. Changes use `dataclasses.replace`. Device, dtype and batch mode are not fields; tolerances are checked against the tensor dtype in the SCF; the eigensolver options live on the SCF data. Calculator-level state that changes after construction (`opts` rebinding for the integral level and the cache flags of the analytical gradient) disappears with B5/B6.

**Needs.** B1. **Size.** M.

---

## B3 Model and setup

**Goal.** Split today's calculator into a molecule-independent model and `model.setup(numbers)`, which returns a system holding all data that depends only on `numbers`.

**Steps.**

1. Introduce the model: parameters (`ParamModule` or plain tensors, per B1), configuration (B2), per-element tables.
2. Implement `setup(numbers)` as a pure function of parameters and numbers:
   - index helper and basis;
   - per-atom and per-shell parameters gathered from the element tables;
   - classical-term setup data that today sits in caches keyed only on `numbers` (repulsion, DFT-D3, DFT-D4, halogen bond, short-range bond, IES, third-order Coulomb).
3. Move all data-dependent operations (`torch.unique`, masks that decide shapes) into `setup` (P3).
4. Make `setup` differentiable with respect to the parameters (needed by G3); check with the T0.9 coverage test.
5. Keep the old calculator constructor as a thin wrapper until B5 lands.

**Done when.**

- `setup` contains every numbers-only computation for the final exact built-in
  core; those built-in classical families do not use `get_cache` keyed on
  `numbers`. Explicitly legacy/custom extensions remain outside this contract.
- Gradients flow from energies to every parameter leaf the method uses and match finite differences, `refocc` included (T0.9 table, T0.12).
- T0.2 reference reproduced.

**Needs.** B1, B2, T0.12. **Unblocks.** B3a unblocks B5, the C5 structural starting point, and E2-core. B3b unblocks C8b and therefore E6bimpl mixed-System stackability. **Size.** L.

**Status: partial. B3a is done; B3b remains open.**

**B3a — structural split: done.** Model/System exist, the legacy Calculator
constructs them, classical numbers-only setup is built once and parameter
coverage is preserved.

**B3b — setup/per-call separation: open.** Setup separation is complete for
ES2, ES3, AES2, repulsion, SRB, classical D4, IES, halogen, D4SC, and ALPB.
The setup-owned PyTorch `PairPlan` is complete under E2-core. Remaining work is
limited to numbers-only masks/index mappings needed by transformed kernels,
transform-critical parameter-to-element gathers that belong in setup, and the
transitional `System.batch_mode` and `System.dd` fields before C8b freezes the
layout. The complete Model and live mutable integral container are already
removed from System; `integral_setup` and `h0_setup` are established.

Final B3b closure also requires broader B6a's built-in DFT-D3 setup/cache
migration, which retires the last generic built-in classical-cache boundary:

1. Give exact built-in DFT-D3 explicit setup data.
2. Construct exact built-in setup values directly through their setup
   builders/values, without first building a generic `ClassicalListCache`.
3. Remove `System.classical_cache` from the final built-in System layout.
4. Ensure core `System.singlepoint` never reads `System.classical_cache` for an
   exact built-in classical family.
5. Retain generic `ClassicalListCache`/`get_cache` only on explicitly
   legacy/custom-extension paths outside the final stackable core.

Reuse existing explicit builders for migrated families, including the
repulsion, SRB, D4, IES, and halogen setup paths already present. This does not
require arbitrary custom classical extensions to become stackable; those
remain under the explicit legacy compatibility policy. The migration belongs
to broader B6a, not a new package. B6b-core and E3.0 do not wait for it. B3a,
not full B3, was the prerequisite for B5.

The complete Model must not be retained by the final System. Gathered setup
tensors retain the gradient path to Model parameters.

B3a record: `dxtb.calculators.Model` (parameters, configuration, additional components) and `System`; `Model.setup(numbers)` builds the index helper, the interactions and classicals with their element parameters gathered, the integral container and the data of the classical terms that depends on the numbers only. The classical components' `get_cache` is a pure function. `setup` is differentiable with respect to the parameters (test); T0.9 parameter coverage and parameter-derivative cells are unchanged.

---

## B4 Pure integral builders

**Status: done for the single-System core.** The legacy Calculator batch
construction path remains separately owned by E6broute.

**Goal.** Integral and Hamiltonian builders return matrices; no integral object changes after construction.

**Today.** `Integrals.build_overlap` creates the overlap object once and rebuilds its matrix in place on every call (`integral/container.py`, around lines 197–227). `BaseIntegral` stores `matrix`, `gradient` and `norm`, and `normalize` overwrites `matrix`. The calculator's result store keeps a reference to the live overlap object.

**Steps.**

1. Builders become functions: `(system data, positions) -> matrix` (and gradient, where analytical gradients are kept).
2. `normalize`, `traceless`, `shift_r0_rj` and similar transforms become pure functions returning new tensors.
3. Remove stored `matrix`, `gradient`, `norm` from integral objects; integral objects keep only static description (driver type, basis references).
4. The `Integrals` container disappears or becomes a plain grouping of builder functions.

**Additional requirements.**

- PairPlan/other structural PyTorch-integral metadata is constructed in setup.
- PyTorch builders take positions directly; no driver owns current positions.
- libcint setup is a per-call value.
- integral I/O (`write_overlap`, etc.) is outside the core.
- geometry-dependent screening is not used to change shapes in the
  transform-critical path.

**Done when.** No integral or Hamiltonian object is mutated after construction; T0.2 reference reproduced.

**Needs.** B1. **Unblocks.** B5, C7, E2-core. **Size.** L (can be split per integral type: overlap, dipole, quadrupole, H0).

---

## B5 Result object and `singlepoint`

**Status: done for the immutable single-System Result/core singlepoint
boundary.** B5.6 separately closes the public solver-status schema.

**Goal.** `system.singlepoint(...)` returns a frozen result; nothing is stored on the calculator.

**Steps.**

1. Define the result type with the fields fixed in B1.
2. Change `singlepoint` to return it. Inside, all intermediate values are local variables or fields of the result.
3. Change `energy`, the analytical forces and `bond_orders` to read from a result instead of `calc.cache` (today they read `cache["energy"]`, `cache["charges"]`, `cache["overlap"]`, `cache["coefficients"]`, `cache["mo_energies"]`).
4. Delete `CalculatorCache`, `@cdec.cache`, `opts.cache` (`ConfigCache`, `ConfigCacheStore`) and all `store_*` keyword arguments.
5. Delete the `calculate()` dispatcher's cache handling.

`Result` is immutable from its first construction; do not construct an empty
Result and fill its fields.

`iterations`, `converged` and `residual` are explicit diagnostics.

The old Calculator is an adapter to the new core, never the implementation
called by the new System.

**Done when.** No code reads or writes a calculator-level cache; T0.2 reference reproduced; T0.5 tests 2 and 3 no longer apply and are removed.

**Needs.** B3a, B4. **Unblocks.** B5.6, B6a-SCF-core, B7, and E4core. C8b additionally needs B3b and C5-C7; it is not unblocked by B5 alone. **Size.** M.

### B5.6 Close the public solver-status Result schema

**Status: complete and verified.** `Result.status` exposes the immutable
public `SCFStatus` representation, and C8a is unblocked.

Before C8a registers the Result pytree, add the agreed immutable
`SCFStatus | None` field to Result and `Result.snapshot()`. Its functional
value contains dynamic integer-tensor leaves `trajectory_failure` and
`final_failure`; outer convergence remains the separate `converged` field.
Define the single public mapping as `SCFFailureCode(IntEnum)`:

```python
class SCFFailureCode(IntEnum):
    NONE = 0
    INVALID_TEMPERATURE = 1
    TOO_MANY_ELECTRONS = 2
    FERMI_NONCONVERGENCE = 3
    ELECTRON_COUNT = 4
```

Define it once in `src/dxtb/_src/scf/status.py` and re-export it from
`src/dxtb/__init__.py` as `dxtb.SCFFailureCode`. This is the canonical user
import path; the implementation must not create a second solver-local mapping. The
enum provides static names for integer tensor values; enum objects do not
appear as pytree leaves. These numeric values are stable public API values,
including when stored with a Result; changing them requires an explicit
migration note. Acceptance includes a top-level import/API test and exact
agreement between internal category projection and the public constants.
The trajectory code is the first failure among physical raw-map evaluations;
the final code describes only the final public spectrum. Eager interpretation
therefore preserves chronological trajectory-before-final ordering without
exposing the internal solver result. Legacy routes may leave `status=None`.

This schema closure also resolves the canonical field spellings and
representations in B1 (`hcore`, tuple-of-pairs `classical`, and
`multipoles` derived from `charges`) before any pytree layout is frozen.
Regression requirements cover zero-T and finite-T success, each failure
category, trajectory-versus-final priority, valid subsolves with outer
nonconvergence, and `maxiter=0`. Low-level status projection must map every
internal category to the named public constant exactly and pytree/vmap the
integer tensor leaves without putting enum objects in the tree. Invalid
configured input may still be rejected during static preparation; the
negative-temperature policy remains deferred and is not decided by E6a.

**Needs.** B5 and E4core finite-temperature/trajectory-status outputs.
**Unblocks.** C8a and E6a. **Size.** S.

---

## B6 Per-call geometry data; remove all caches and mutation methods

**Current status.** B6a-SCF-core and B6a-SCF-validation are complete. Broader
B6a remains partial for built-in DFT-D3 explicit setup and removal of the final
generic stored built-in `ClassicalListCache` boundary. B6b-core is open; legacy
compatibility purge is separately gated by E4.6a/E4.6b.

**Split.**

**B6a:** remove all state persistent across calls. Temporary per-call SCF data
may still be mutable because the old batched SCF culls it.

**B6a-SCF-core (complete implementation prerequisite):** the bounded state-removal subset required by the
canonical functional SCF inputs and raw map. It covers self-consistent
interaction/setup data used by `SCFInputs`; it excludes classical terms added
outside the SCF result. E4core depends on this completed subset, not on the
broader B6a package. The source audit found that DFT-D3 is an outer classical
term: `DispersionD3.get_energy()` reads the System's stored component and
numbers-only `DispersionD3Cache`, computes coordination numbers, weights, C6,
and energy locally, and does not mutate the component or cache. Those legacy
setup/cache objects still need B6a migration; their separate forward-mode
blocker belongs to E1.3.

**B6a-SCF-validation (complete):** the seven-row parameterized isolated-family
matrix constructs its fresh-System oracle after reused-System history and
checks reused-System state both before and after fresh-System setup/evaluation.
The separate all-seven integration stress test also passes. The focused B6a
selection reports 8 passes per environment (40 passes across the five
supported environments).
Strong references are retained for persistent identity and value checks. The
large supporting invocation reported an intermittent PySCF native-loader
error from `libcvhf.so`; the same undefined-symbol failure family was
reproduced on pre-package base
`3f41f7f280a3c6df364cf7639360ca602b945118`, and the exact candidate node passes
in isolation. See `07-current-state-and-execution-plan.md` for the run counts
and node identities. This retrospective
validation does not change E4core's accepted implementation dependency or
close broader B6a's legacy-shaped DFT-D3 setup/cache migration.

**B6b-core:** after the E4core functional single-System path lands, close core
state purity and establish immutable structural/interaction input values.
This does not delete cull/restore from the separate legacy batch adapter.

**Legacy compatibility purge:** E4.6a removes batch cull/restore only after
E6broute owns production batch routing and the batch-critical E7 padding gate
passes. E4.6b removes the remaining legacy FULL loop only after every fallback
is migrated, globally rejected, or retired.

**Dependency split.** B6a-SCF-core is the E4core prerequisite. B6b-core owns
the immutable core structural/component boundary and is a prerequisite for
C5/C6/C7 and F3. The legacy compatibility purge belongs to E4.6a/E4.6b after
its separate migration gates. Generic B6 does not unblock E4core or F3; those
consumers require B6a-SCF-core and B6b-core, respectively.

**Goal.** Geometry-dependent intermediates are computed inside each call; no component or driver keeps state between calls.

**Steps.**

1. Coulomb matrices (ES2, multipole), ALPB and D4SC data, and electric-field data are computed per call inside `singlepoint` and carried in the per-call interaction cache (already an explicit value).
2. Driver setup returns a value (libcint wrapper, basis batches) used for that call only. Delete `IntDriver._positions`, `is_latest`, `invalidate`.
3. Delete `Component._cache`, `_cachevars`, `cache_is_latest`, `cache_invalidate`, `cache_is_setup`, `cache_enable`, `cache_disable`.
4. Delete `Component.update`, `Component.reset`, `ComponentList.reset_all`, `Integrals.reset_all`, `Calculator.reset`, and the `update_*`/`reset_*` helpers on `InteractionList`. Changes go through `replace()` or construction.
5. Delete the manual resets in `ir()` and `raman()` (`calculators/types/autograd.py`, around lines 902 and 960) and the T0.5 stopgap.

**Done when.**

- No attribute of a component, list or driver is written outside construction.
- The T0.5 test 1 scenario is impossible by construction (call history no longer matters).
- T0.2 reference reproduced, including Hessians and IR/Raman without manual resets.

**Needs.** B5, B4. **Unblocks.** The B6a-SCF-core subset is the E4core
prerequisite; B6b-core is the prerequisite for C5/C6/C7 and F3. The broader
B6 umbrella does not itself unblock E4core or F3. **Size.** L (can be split:
interactions, classicals, driver).

---

## B7 External fields as inputs

**Status: done.** Fields are explicit inputs on the core path; legacy field
components remain isolated in the Calculator compatibility adapter.

**Goal.** Electric field and field gradient are arguments of `singlepoint`, not mutable components (P5). Required for hyperpolarizabilities (G1).

**Today.** The field lives in the `ElectricField` component; `interactions.update_efield(field=...)` replaces it; decorators `requires_efield`, `requires_efield_grad`, `requires_efg`, `requires_efg_grad` check component state; numerical field derivatives shift the field in place (`calculators/types/numerical.py`).

**Steps.**

1. Add `field` and `field_grad` arguments to `singlepoint`; the field interactions read them from the call inputs.
2. Delete `update_efield`, `update_efield_grad`, the `requires_efield*` and `requires_efg*` decorators.
3. Rewrite dipole, polarizability and hyperpolarizability as derivatives with respect to the `field` input (autograd and numerical variants).
4. Same for field-gradient properties (quadrupole response).

**Done when.** No mutable field state; dipole, polarizability and hyperpolarizability reproduce T0.2 by both autograd and finite differences.

**Needs.** B5. **Unblocks.** B8a property functions and, transitively, B8b/E8. **Size.** M.

---

## B8 Property functions

**Status.** B8a and B8b remain open packages; B8a may complete before E3, while
B8b requires B8a and full E3. E8 is separate final acceptance.

**Split.**

B8a: pure first/second-order property API.
B8b: third-order and mixed-response properties after E3.

**Goal.** All properties are functions of the system and the call inputs, implemented with `torch.func`. They replace the calculator mixins (`AnalyticalCalculator`, `AutogradCalculator`, `NumericalCalculator`).

**B8a functions.** `energy`, `forces`, `hessian`, `dipole`, `quadrupole`,
`polarizability`, `bond_orders`, and numerical variants.

**B8b functions.** `third_order` (new), `hyperpolarizability`,
`dipole_derivative`, `polarizability_derivative`, `vibration`, `ir`, `raman`,
and mixed parameter/force responses.

**Steps.**

1. Implement each property as a composition of `torch.func` transforms over `energy(system, positions, field, ...)`, choosing reverse or forward mode per dimension (field: reverse is fine, 3 dimensions; geometry at third order: forward-over-reverse).
2. Analytical forces stay as an option where they are faster; they take a result as input.
3. Numerical variants call the same functions with perturbed inputs; no state.
4. Delete the calculator mixins and the `numerical` decorator's state handling.

**B8a done when.** Its listed first/second-order properties and numerical
variants are pure functions of the System and explicit inputs and reproduce
the applicable T0.2 references.

**B8b done when.** Its third-order and mixed-response functions compose the
B8a functional API, pass the applicable T0.3 transform contract after full
E3, and reproduce T0.2 references. E8 is the final G1 acceptance package; it
does not implement B8b.

**Needs.** B8a needs B7. B8b needs B8a and full E3 (including full E1 where
DFT-D3 is active). **Size.** L.

---

## B9 Optional ASE adapter

**Goal.** Keep ASE-style use (`get_potential_energy`, `get_forces`) outside the core.

**Steps.** Implement an ASE `Calculator` subclass using ASE's own state comparison for reuse; it holds the last result and calls the property functions.

**Done when.** ASE workflows run; no caching logic in dxtb's core.

**Needs.** B8. **Size.** S. **Optional.**

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| B1 | Decision note | T0.11 | S |
| B2 | Frozen configuration | B1 | M |
| B3a | Model/System structural split (done, transitional) | B1, B2 | done |
| B3b | Remaining setup/per-call structural separation, including broader B6a built-in DFT-D3 setup and final generic stored built-in ClassicalListCache removal | B3a, B4/E1 and broader B6a DFT-D3 work as applicable | open |
| B4 | Pure integral builders | B1 | L |
| B5 | Result object and `singlepoint` | B3a, B4 | M |
| B5.6 | Public tensor-valued solver-status Result schema (complete) | B5, E4core | done |
| B6a-SCF-core | SCF-only state-removal prerequisite for E4core | B5, B4 | done |
| B6a-SCF-validation | Retrospective seven-family validation | B6a-SCF-core | done |
| B6a (broader) | Built-in DFT-D3 explicit setup and final generic stored built-in ClassicalListCache removal | B4/B6a-SCF-core | partial |
| B6b-core | Immutable core structural/component boundary | B6a-SCF-core, E4core, E4.5b | open |
| Legacy compatibility purge | Remove fallback-owned cull/reset machinery | E4.6a/E4.6b | later |
| B7 | Fields as inputs | B5 | M |
| B8a | Energy/forces/Hessian/dipole/quadrupole/polarizability/bond_orders and numerical variants | B7 | L |
| B8b | Third-order, mixed-response properties, IR/Raman/vibration | B8a, full E3 | L |
| B9 | ASE adapter (optional) | B8a/B8b as needed | S |

**Release gate.** No intermediate public release is planned; the structural-core milestone defined in `00-overview.md` section 8 feeds the single public restructuring release, with a migration guide built from B1's removal list.
