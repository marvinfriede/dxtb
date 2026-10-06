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

**Needs.** T0.11. **Size.** S.

---

## B2 Frozen configuration

**Goal.** Configuration becomes an immutable value, so the model can hold it as pytree context.

**Today.** `Config` (`calculators/config/main.py:41`) is a plain mutable class with identity equality. It carries `device` and `dtype`. It is changed after construction: the calculator's `__init__` sets `opts.ints.level` and `opts.batch_mode`, `Calculator.type` writes `opts.dtype`, and the CLI sets `calc.opts.scf.*` on an existing calculator.

**Steps.**

1. Convert `Config` and all sub-configs to frozen dataclasses with value equality and hashing.
2. Remove `device` and `dtype` from the configuration.
3. Remove `batch_mode` from the configuration (superseded by B1 item 5). The calculator derives it from the shape of `numbers` (the `batch_mode` option, mode 2 for conformers, stays a constructor option) and the index helper carries it; the SCF reads it from there. The mode itself is removed with the batching rewrite (E6), because conformer throughput (G2) still depends on mode 2.
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
   - classical-term setup data that today sits in caches keyed only on `numbers` (repulsion, D3, D4, halogen bond, short-range bond, IES, third-order Coulomb).
3. Move all data-dependent operations (`torch.unique`, masks that decide shapes) into `setup` (P3).
4. Make `setup` differentiable with respect to the parameters (needed by G3); check with the T0.9 coverage test.
5. Keep the old calculator constructor as a thin wrapper until B5 lands.

**Done when.**

- `setup` contains every numbers-only computation; classical components have no `get_cache` keyed on `numbers`.
- Gradients flow from energies to every parameter leaf the method uses and match finite differences, `refocc` included (T0.9 table, T0.12).
- T0.2 reference reproduced.

**Needs.** B1, B2, T0.12. **Unblocks.** B5, C5, E2 (index lists in setup). **Size.** L.

---

## B4 Pure integral builders

**Goal.** Integral and Hamiltonian builders return matrices; no integral object changes after construction.

**Today.** `Integrals.build_overlap` creates the overlap object once and rebuilds its matrix in place on every call (`integral/container.py`, around lines 197–227). `BaseIntegral` stores `matrix`, `gradient` and `norm`, and `normalize` overwrites `matrix`. The calculator's result store keeps a reference to the live overlap object.

**Steps.**

1. Builders become functions: `(system data, positions) -> matrix` (and gradient, where analytical gradients are kept).
2. `normalize`, `traceless`, `shift_r0_rj` and similar transforms become pure functions returning new tensors.
3. Remove stored `matrix`, `gradient`, `norm` from integral objects; integral objects keep only static description (driver type, basis references).
4. The `Integrals` container disappears or becomes a plain grouping of builder functions.

**Done when.** No integral or Hamiltonian object is mutated after construction; T0.2 reference reproduced.

**Needs.** B1. **Unblocks.** B5, C7, E2, E5. **Size.** L (can be split per integral type: overlap, dipole, quadrupole, H0).

---

## B5 Result object and `singlepoint`

**Goal.** `system.singlepoint(...)` returns a frozen result; nothing is stored on the calculator.

**Steps.**

1. Define the result type with the fields fixed in B1.
2. Change `singlepoint` to return it. Inside, all intermediate values are local variables or fields of the result.
3. Change `energy`, the analytical forces and `bond_orders` to read from a result instead of `calc.cache` (today they read `cache["energy"]`, `cache["charges"]`, `cache["overlap"]`, `cache["coefficients"]`, `cache["mo_energies"]`).
4. Delete `CalculatorCache`, `@cdec.cache`, `opts.cache` (`ConfigCache`, `ConfigCacheStore`) and all `store_*` keyword arguments.
5. Delete the `calculate()` dispatcher's cache handling.

**Done when.** No code reads or writes a calculator-level cache; T0.2 reference reproduced; T0.5 tests 2 and 3 no longer apply and are removed.

**Needs.** B3, B4. **Unblocks.** B6, B7, C8, E4. **Size.** M.

---

## B6 Per-call geometry data; remove all caches and mutation methods

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

**Needs.** B5, B4. **Unblocks.** C6, C7, E4, F3. **Size.** L (can be split: interactions, classicals, driver).

---

## B7 External fields as inputs

**Goal.** Electric field and field gradient are arguments of `singlepoint`, not mutable components (P5). Required for hyperpolarizabilities (G1).

**Today.** The field lives in the `ElectricField` component; `interactions.update_efield(field=...)` replaces it; decorators `requires_efield`, `requires_efield_grad`, `requires_efg`, `requires_efg_grad` check component state; numerical field derivatives shift the field in place (`calculators/types/numerical.py`).

**Steps.**

1. Add `field` and `field_grad` arguments to `singlepoint`; the field interactions read them from the call inputs.
2. Delete `update_efield`, `update_efield_grad`, the `requires_efield*` and `requires_efg*` decorators.
3. Rewrite dipole, polarizability and hyperpolarizability as derivatives with respect to the `field` input (autograd and numerical variants).
4. Same for field-gradient properties (quadrupole response).

**Done when.** No mutable field state; dipole, polarizability and hyperpolarizability reproduce T0.2 by both autograd and finite differences.

**Needs.** B5. **Unblocks.** B8, E8. **Size.** M.

---

## B8 Property functions

**Goal.** All properties are functions of the system and the call inputs, implemented with `torch.func`. They replace the calculator mixins (`AnalyticalCalculator`, `AutogradCalculator`, `NumericalCalculator`).

**Functions.** `energy`, `forces`, `hessian`, `third_order` (new), `dipole`, `quadrupole`, `polarizability`, `hyperpolarizability`, `dipole_derivative`, `polarizability_derivative`, `vibration`, `ir`, `raman`, `bond_orders`, plus `*_numerical` variants.

**Steps.**

1. Implement each property as a composition of `torch.func` transforms over `energy(system, positions, field, ...)`, choosing reverse or forward mode per dimension (field: reverse is fine, 3 dimensions; geometry at third order: forward-over-reverse).
2. Analytical forces stay as an option where they are faster; they take a result as input.
3. Numerical variants call the same functions with perturbed inputs; no state.
4. Delete the calculator mixins and the `numerical` decorator's state handling.

**Done when.** Every property in T0.2 reproduced; third-order force constants available (validated in E8).

**Needs.** B7; full third-order support needs TE (E1, E3). **Size.** L.

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
| B3 | Model and setup | B1, B2 | L |
| B4 | Pure integral builders | B1 | L |
| B5 | Result object and `singlepoint` | B3, B4 | M |
| B6 | Per-call geometry data; remove caches and mutation | B5, B4 | L |
| B7 | Fields as inputs | B5 | M |
| B8 | Property functions | B7 (TE for order 3) | L |
| B9 | ASE adapter (optional) | B8 | S |

**Release gate.** dxtb release 1 ships after B2–B8 with a migration guide built from B1's removal list.
