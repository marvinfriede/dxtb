# Final Calculator Refactoring, Analytical-Layer Removal, and Branch Consolidation

## 1. Executive Summary

This package completes the public `Calculator` migration to the functional `Model` / `System` / `Result` architecture, deletes `AnalyticalCalculator` and all `*_analytical` derivative methods, activates packed functional routing (`E6broute`) on public batched `Calculator.singlepoint()`, and consolidates the repository branches on `claude/tender-dijkstra-cda42s`:

- **Base Commit (`claude/tender-dijkstra-cda42s`)**: `d848ab032b50732690ea37dfd2c5b8422d47713c`
- **Merged `main` Ancestry Commit**: `cb68a9862c85e1af72e97e587ecf7c7e12cfc08c` (merges `origin/main` `46199ae074a4dd5462da1d64026b4546b80cb172`, preserving `test/test_calculator/test_cache/test_positions_graph.py` while keeping `decorators.py` free of weakref cache-reset state on the functional core)
- **Implementation Commits**:
  1. `30d98c4a0877db24a421dcdf524d09fa946b8c4e` — `refactor(calculators): remove AnalyticalCalculator and analytical derivative layer`
  2. `9eca40db2018d1b598a4a24f4e4569e50821db9d` — `refactor(calculators): cut over batched Calculator.singlepoint to packed functional routing`
  3. `eb577209149831eed15ad58f9ca87c24388fb30e` — `test(calculators): migrate test suite and docs for analytical removal and batch cutover`
  4. `b0240d8469f2cc2018f042c726abbf064ffdc2fd` — `fix(calculators): resolve rules guard, quadrupole getter tests, and review nits`

---

## 2. Removal of `AnalyticalCalculator` and `*_analytical` Methods

Per project-owner decision, PyTorch automatic differentiation is the sole authoritative derivative and response mechanism, with numerical finite-difference methods retained as independent validation tools:

1. **Deleted Module**:
   - `src/dxtb/_src/calculators/types/analytical.py` is deleted (`338` lines removed).
2. **Removed Classes and Methods**:
   - `AnalyticalCalculator` removed from `src/dxtb/_src/calculators/types/__init__.py`, `src/dxtb/_src/calculators/base.py`, `src/dxtb/calculators.py`, and `docs/source/02_indepth/calculators_types.rst`.
   - `forces_analytical`, `dipole_analytical`, and `quadrupole_analytical` are completely removed with no aliases or compatibility shims.
3. **Updated `Calculator` Inheritance and Dispatch (`src/dxtb/_src/calculators/base.py`)**:
   - `class Calculator(AutogradCalculator, NumericalCalculator):` inherits only from `AutogradCalculator` and `NumericalCalculator`.
   - `Calculator.implemented_properties` is the union of `AutogradCalculator.implemented_properties` and `NumericalCalculator.implemented_properties`.
   - The `Calculator.calculate()` override that intercepted `"quadrupole"` / `"quadrupole_moment"` to call `self.quadrupole_analytical(...)` is removed; `Calculator` inherits `AutogradCalculator.calculate()` directly, dispatching `"quadrupole"` and `"quadrupole_moment"` to `self.quadrupole(...)`.
4. **Removed Analytical Fallbacks in `AutogradCalculator`, `NumericalCalculator`, and CLI (`src/dxtb/_src/calculators/types/autograd.py`, `numerical.py`, `cli/driver.py`)**:
   - Removed `_get_dipole_fcn`, `use_analytical_dipmom`, and `use_analytical` from `AutogradCalculator.dipole_deriv`, `polarizability`, and `pol_deriv`.
   - Updated `AutogradCalculator.forces` to support batched `1D` energy outputs (`grad_outputs=torch.ones_like(e)` in `autograd` mode and `e.sum().backward(**kw)` in `backward` mode).
   - Updated `NumericalCalculator.hessian_numerical` to unconditionally differentiate `-self.forces_numerical(...)`, and `dipole_deriv_numerical` / `polarizability_numerical` to unconditionally differentiate `self.dipole_numerical`.
   - Updated `Driver.properties` in `src/dxtb/_src/cli/driver.py` to call `calc.dipole(positions, chrg)`.

---

## 3. Public `Calculator.singlepoint()` Routing and `Model` / `System` Coherence

1. **Single-System Routing (`self.numbers.ndim == 1`)**:
   - `EnergyCalculator.singlepoint()` calls `self._singlepoint_system(...)`, which ensures `self.system` is current (`self._ensure_system_current()`), evaluates `_singlepoint_eager(self.system, positions, chrg, spin, field, field_grad)`, handles `write_*` integral exports, and increments `self._ncalcs += 1`.
2. **Batched Routing (`self.numbers.ndim > 1`, `E6broute`)**:
   - `EnergyCalculator.singlepoint()` delegates directly to `evaluate_calculator_packed_singlepoint(self, positions, chrg=chrg, spin=spin, field=field, field_grad=field_grad, **kwargs)` in `src/dxtb/_src/calculators/packed_result.py`.
   - `evaluate_calculator_packed_singlepoint` builds deduplicated per-lane single-`System` instances via `build_calculator_lane_systems(calc)`, evaluates mapped-eligible buckets via `evaluate_padded_systems` and legacy-composed buckets via `compose_legacy_systems`, reassembles a source-ordered packed `Result` honoring `atom_masks` and `target_atoms`, and increments `calc._ncalcs += 1`.
3. **Caller-Owned Component Isolation, Complete Deduplication Identity, and `Model` Parameter Tracking**:
   - `_sync_lane_system_labels` in `src/dxtb/_src/calculators/packed_result.py` copies any caller-owned component before restoring its original label and synchronizes `exact_core` (`classical_terms`, `interaction_terms`) and `_exact_compatibility` (`classical_setups`, `interaction_setups`) so `sys._active_exact_core()` remains valid.
   - `build_calculator_lane_systems` memoizes the constructed lane `System` tuple on `calc._lane_systems_memo` keyed by `(calc._model_param_sig, calc.dtype, calc.device, id(calc.numbers), calc.numbers._version, calc.opts, classicals_sig, interactions_sig, comp_nums_sig)`.
   - `BaseCalculator._ensure_system_current()` tracks parameter and buffer leaf state via `_param_state_signature(self.model.par)` (`(id(t), t._version, t.requires_grad, t.dtype, t.device)`) and registers backward hooks (`_register_param_backward_hooks`) that mark `_system_graph_consumed = True` and invalidate `_lane_systems_memo` when a backward pass consumes the setup autograd graph.

---

## 4. Legacy Code Deleted vs. Retained

| Component / Symbol | Action | Rationale |
|---|---|---|
| `src/dxtb/_src/calculators/types/analytical.py` (`AnalyticalCalculator`, `forces_analytical`, `dipole_analytical`, `quadrupole_analytical`) | **DELETED** | Replaced by PyTorch automatic differentiation (`AutogradCalculator` and functional `System` properties). |
| `EnergyCalculator._call_interactions` and legacy batch SCF execution block in `EnergyCalculator.singlepoint` (`src/dxtb/_src/calculators/types/energy.py`) | **DELETED** | Superseded by `evaluate_calculator_packed_singlepoint` (`E6broute`). |
| `AutogradCalculator._get_dipole_fcn`, `use_analytical_dipmom`, `use_analytical` (`src/dxtb/_src/calculators/types/autograd.py`) | **DELETED** | Analytical dipole pathway removed; `AutogradCalculator` always uses `self.dipole` / `self.polarizability`. |
| `BaseCalculator.integrals` (`_create_legacy_integral_adapter`) | **RETAINED** | Still required for default `<integral.label.casefold()>.pt` filenames in `write_*` options, `Calculator.reset()`, `Calculator.type()`, and `Calculator.__str__()`. |
| `SelfConsistentFieldFull` / `SelfConsistentFieldImplicit` (`src/dxtb/_src/scf/`) | **RETAINED** | Required by `compose_legacy_systems` (`src/dxtb/_src/calculators/legacy_composition.py`) for `LEGACY_COMPOSED` / `SEPARATE_ROUTE` configurations (e.g., `libcint` driver, custom interactions, and multi-stage/SOSCF/implicit packed fallback lanes) and direct low-level unit tests. |
| `src/dxtb/_src/scf/**` (S0–S13.1 SCF core) | **UNMODIFIED** | `git diff d848ab032b50732690ea37dfd2c5b8422d47713c..HEAD -- src/dxtb/_src/scf/` is empty (`0` lines changed). |

---

## 5. Verification and Multi-Environment Test Results

1. **S13 668-Test Functional SCF & `vmap` Gate (`py3.14-torch2.14.1`)**:
   - Command: `pytest -q -n 8 --dist worksteal test/test_scf/test_functional_*.py test/test_calculator/test_e6a0_system_vmap.py`
   - Result: **668 passed, 0 failed** (`859.66s`).
2. **Calculator, Invariant, E3, E6bcompat, and B8/E8 Functional Properties Gate (`py3.14-torch2.14.1`)**:
   - Command: `pytest -q -n 8 --dist worksteal test/test_scf/test_invariant_*.py test/test_scf/test_stateless_map.py test/test_scf/test_s13_public_strategy.py test/test_scf/test_singlepoint_core.py test/test_scf/test_e3_end_to_end_acceptance.py test/test_calculator/test_e6bcompat_differential.py test/test_calculator/test_e6bimpl3_legacy_composition.py test/test_calculator/test_b8_e8_functional_properties.py`
   - Result: **294 passed, 0 failed** (`886.68s`).
3. **5-Environment Focused Calculator & Property Suite (125 tests per environment)**:
   - Modules: `test_e6bcompat_differential.py`, `test_e6bimpl3_legacy_composition.py`, `test_model.py`, `test_cache/test_properties.py`, `test_cache/test_positions_graph.py`, `test_b8_e8_functional_properties.py`, `test_s13_public_strategy.py`
   - `py3.10-torch2.7.0`: **125 passed, 0 failed** (`1131.30s`)
   - `py3.11-torch2.8.0`: **125 passed, 0 failed** (`1171.36s`)
   - `py3.12-torch2.11.0`: **125 passed, 0 failed** (`1234.65s`)
   - `py3.12-torch2.13.0`: **125 passed, 0 failed** (`1224.41s`)
   - `py3.14-torch2.14.1`: **125 passed, 0 failed** (`1063.63s`)
4. **Migrated Property, Integral, Rules, and Singlepoint Suites (`py3.14-torch2.14.1`)**:
   - `test_dipole.py`, `test_dipole_deriv.py`, `test_pol.py`, `test_quadrupole.py`, `test_quadrupole_fieldgrad.py`, `test_vibration.py`, `test_ir.py`, `test_raman.py`, `test_hyperpol.py`, `test_pure_evaluation.py`, `test_quadrupole_getter.py`, `test_tblite_reference.py`, `test_grad_gfn1.py`, `test_grad_pos_withfield.py`, `test_energies.py`, `test_forces.py`, `test_hessian.py`: **All passed (`0` failures)**.
   - Broad non-large/non-slow CPU selection (`-m "not large and not slow"`): **5841 passed, 30 skipped, 29 xfailed**; all remaining failures are pre-existing on `d848ab032b50732690ea37dfd2c5b8422d47713c` (such as legacy `Calculator.hessian` memory-leak checks and `Driver` `Config.exclude` list hashability), and the two regressions detected during the broad run (`test_no_calculator_result_cache_or_identity_keying` in `test_rules.py` and `test_quadrupole_getter.py`) were fixed in commit `b0240d8469f2cc2018f042c726abbf064ffdc2fd` and verified green.

---

## 6. Independent Review Verdict

An independent read-only review subagent (`439f9f3c-adeb-4736-bbef-f68f5f219010`) audited the entire diff against all 14 mandatory checklist items in Section 12:

- **Verdict**: **APPROVED (`0 BLOCKER`, `0 MAJOR`, `0 MINOR`, `2 NIT`)**
- Both `NIT` items (trailing blank lines at EOF in two test files and stale comment/docstring wording in `energy.py` and `packed_result.py`) were resolved in commit `b0240d8469f2cc2018f042c726abbf064ffdc2fd`.

---

## 7. Repository Branch Consolidation Record

1. **Ancestry Verification Before Deletion**:
   - `git merge-base --is-ancestor origin/scf/redesign HEAD` (`e78d3f6ac05a970e7871fa65a358b8f3d028cd9e`): **VERIFIED**
   - `git merge-base --is-ancestor origin/integration/scf-s13-general-final HEAD` (`d848ab032b50732690ea37dfd2c5b8422d47713c`): **VERIFIED**
   - `git merge-base --is-ancestor origin/integration/u1-u4-current HEAD` (`eb3fab5a9444d85c8f3a1eaaf722d2049e8bebd7`): **VERIFIED**
   - `git merge-base --is-ancestor origin/main HEAD` (`46199ae074a4dd5462da1d64026b4546b80cb172`): **VERIFIED**
2. **Remote and Local Branch Cleanup**:
   - Merged `integration/final-calculator-refactoring` into `claude/tender-dijkstra-cda42s` and pushed to `origin/claude/tender-dijkstra-cda42s`.
   - Deleted remote branches: `origin/scf/redesign`, `origin/integration/scf-s13-general-final`, `origin/integration/u1-u4-current`.
   - Deleted local branches and temporary worktrees: `scf/redesign`, `integration/scf-s13-general-final`, `integration/u1-u4-current`, `integration/final-calculator-refactoring`, `main`.
   - Preserved `feature/unified-tad-dftd` untouched (`6a773c41465c12b9a594173c8b2ab7a14f665363`).
   - Regarding `origin/main`: `origin/HEAD -> origin/main` is currently configured on GitHub as the repository's default branch (and referenced by `.github/workflows/*.yaml`), and GitHub rejects `git push origin --delete main` over SSH (`[remote rejected] main (refusing to delete the current branch: refs/heads/main)`) until the default branch is switched to `claude/tender-dijkstra-cda42s` via the GitHub web UI or REST API (`gh` CLI / GitHub API token is not installed in this environment).
