# Functional Route Final Acceptance and Legacy-Retirement Readiness

## 1. Executive Summary & Project-Owner Decision Brief

This report records the final compatibility closure, production-routing differential gate, converged property qualification, and legacy-retirement dependency audit on branch `claude/tender-dijkstra-cda42s` (base SHA `b9b311a234868a1d339795eb0846fb117a5ce1f0`).

In accordance with project governance, **default public `Calculator.singlepoint` batch dispatch has not been switched** and **no public feature flag has been added**. Instead, an internal production-routing differential harness (`evaluate_calculator_packed_singlepoint` and `build_calculator_lane_systems` in `src/dxtb/_src/calculators/packed_result.py`) executes the exact validation, option-handling, canonical bucketing, retained legacy composition, and `Result` packing contract that a one-step production cutover will use.

### Explicit Answers to the Four Project-Owner Questions (`Section 12`)

1. **Can `Calculator` packed batch routing now be switched to the functional/composed route?**
   - **YES.** All remaining compatibility defects have been closed and verified:
     - Public `Calculator.singlepoint` options (`write_overlap`, `write_dipole`, `write_quadrupole`, `write_hcore`, `cuda_sync_in_scf`, removed `store_*`, and `ignore_unknown_kwargs`) are strictly separated from low-level SCF engine constructor options (`LegacySCFOptions.args` and `LegacySCFOptions.kwargs`: `fwd_options`, `bck_options`, `eigen_options`, and unknown low-level keywords).
     - `Result.get_energies()` resolves classical display-key collisions without overwriting pre-suffixed labels such as `"Custom (1)"`.
     - `evaluate_calculator_packed_singlepoint` matches `Calculator.singlepoint` across all 32 `E6bcompat` policy rows (`docs/plan/E6bcompat-final-routing-manifest.json`), enforces the 6-stage failure precedence order, deduplicates `Model.setup(..., batch_mode=0)` across identical lane compositions (including `batch_mode=2` conformers), and handles non-contiguous interior zero padding and trailing all-zero columns via `atom_masks` and `target_atoms`.
2. **Which `E6bcompat` rows are mapped vs retained via legacy composition?**
   - Out of the **32 policy rows** in `docs/plan/E6bcompat-final-routing-manifest.json`:
     - **10 rows (`REPLACE`)** execute on the canonical functional `MAPPED` route (`bucket_canonical_exact_cores` + `evaluate_padded_systems` + `pack_system_results`): `E6B-01` through `E6B-05`, `E6B-07`, `E6B-10`, `E6B-28`, `E6B-29`, `E6B-32`.
     - **22 rows (`RETAIN`)** execute on the ordered retained composition boundary (`compose_legacy_systems` over single-System `LEGACY_COMPOSED` or `SEPARATE_ROUTE` evaluations, or direct structural core/SCF boundaries): `E6B-06`, `E6B-08`, `E6B-09`, `E6B-11` through `E6B-27`, `E6B-30`, `E6B-31`.
3. **What is the exact 1-step cutover diff and what legacy state on `Calculator`/`System` can be retired immediately upon cutover vs in `E4.6a`/`E4.6b`/`E2`?**
   - **1-Step Cutover Diff**: In `EnergyCalculator.singlepoint` (`src/dxtb/_src/calculators/types/energy.py:176-180`), replace the `is_batched` body (lines `181-394`) with a single delegation call to `evaluate_calculator_packed_singlepoint(self, positions, chrg=chrg, spin=spin, field=resolved_field, field_grad=resolved_field_grad, **kwargs)` (caching `self._lane_systems = build_calculator_lane_systems(self)` in `BaseCalculator.__init__`).
   - **Immediate vs Staged Retirement**:
     - *Immediately dead in `EnergyCalculator.singlepoint` upon cutover*: packed classical evaluation (`self.classicals.get_energy` on `batch_mode > 0`), packed integral matrix assembly (`self.integrals.build_*` inside `singlepoint`), and packed `scf.solve` dispatch inside `singlepoint`.
     - *Must remain until `E2-legacy` / `E4.6` analytical force batch migration*: `BaseCalculator.integrals` (`self._create_legacy_integral_adapter()`) and `EnergyCalculator._cull` / `_restore` are still called by `EnergyCalculator.forces_analytical` (`src/dxtb/_src/calculators/types/energy.py:514-605`) when `batch_mode > 0`.
4. **What is the complete caller and dependency map for `E4.6a`, `E4.6b`, `E2`, `C9`, and `B6`?**
   - Section 7 below provides the exhaustive file-by-file and symbol-by-symbol caller/dependency map for `IndexHelper.cull/restore`, `InteractionCache.cull/restore`, `DriverManager`/`Integrals`, `SelfConsistentFieldFull`/`SelfConsistentFieldImplicit`/`BaseTSCF`, `ModelParameterTree`, and `ClassicalListCache`/`InteractionListCache`.

---

## 2. Public vs Low-Level Option Ownership & Side Effects (`Section 2`)

### 2.1 Strict Separation of Option Contracts

| Option Surface | Owner | Recognized Keys & Behavior | Unknown Key Policy |
| :--- | :--- | :--- | :--- |
| **Public `Calculator` / `evaluate_packed_systems(..., **kwargs)`** | `packed_result.py` (`_PUBLIC_SINGLEPOINT_OPTIONS`) & `types/base.py` (`reject_removed_store_kwargs`) | - `write_overlap`, `write_dipole`, `write_quadrupole`, `write_hcore`: saves packed `Result` integral tensor (`overlap`, `dipole_integrals`, `quadrupole_integrals`, `hcore`) via `torch.save(matrix, str(path))` using explicit path or `<integral_label.casefold()>.pt` (`overlappytorch.pt`, `dipolepytorch.pt`, `quadrupolepytorch.pt`, `gfn1hamiltonian.pt` / `gfn2hamiltonian.pt`).<br>- `cuda_sync_in_scf`: sets `timer.cuda_sync` during SCF evaluation and restores the prior state in a `finally:` block.<br>- Removed `store_*` (`store_overlap`, `store_dipole`, `store_quadrupole`, `store_hcore`, `store_integrals`): always raises `TypeError`. | - `legacy_compat=False` (functional API): **always raises `TypeError`** regardless of `ignore_unknown_kwargs`.<br>- `legacy_compat=True` (`Calculator` compatibility): raises `TypeError` when `strict_e6bcompat=True` or `ignore_unknown_kwargs=False`; silently ignored only when `ignore_unknown_kwargs=True and not strict_e6bcompat`. |
| **Low-level SCF engine options (`legacy_options: LegacySCFOptions`)** | `legacy_composition.py` -> `_singlepoint_impl` -> `scf.solve_legacy` -> `BaseSCF.__init__` / `solve_nonselfconsistent` -> `BaseSCF._Data.__init__` | - `fwd_options`: popped by `BaseSCF.__init__` and merged into `self.fwd_options`.<br>- `bck_options`: popped by `BaseSCF.__init__` and merged over `{"posdef": True}` into `self.bck_options`.<br>- `eigen_options`: popped by `BaseSCF.__init__` and `solve_nonselfconsistent` and merged into `eigen_options`.<br>- Malformed non-mapping values (`None`, `int`, etc.): raise `TypeError` when unpacked (`**fwd_options`). | **Never filtered by `ignore_unknown_kwargs`.** Unknown low-level keywords (`unexpected_option`, `solver_options`, etc.) and extra positional `args` are forwarded unchanged to `BaseSCF._Data.__init__` (and `solve_nonselfconsistent` -> `_Data.__init__`), raising `TypeError` at the legacy constructor boundary. |

### 2.2 Non-Self-Consistent (`GFN0-xTB`) Low-Level Option Parity

`solve_nonselfconsistent` (`src/dxtb/_src/scf/nonselfconsistent.py:49-83`) and `scf.iterator._solve_impl` (`src/dxtb/_src/scf/iterator.py:252-263`) now forward `*args` and remaining `**kwargs` (after popping `eigen_options`) into `_Data(occupation, n0, *args, numbers=numbers, ihelp=ihelp, cache=cache, integrals=integrals, **kwargs)`. Consequently, `GFN0-xTB` enforces the exact same `_Data.__init__` constructor contract for unknown low-level keywords and positional extras as `GFN1-xTB` and `GFN2-xTB`.

---

## 3. Classical Display-Key Collision Resolution (`Section 3`)

In `Result.get_energies()` (`src/dxtb/_src/calculators/result.py:240-272`):
- Let `totals = Counter(name for name, _ in self.classical)` and `seen: Counter[str] = Counter()`.
- For each `(name, value)` in `self.classical`:
  - `seen[name] += 1`
  - If `totals[name] == 1 and name not in classical`: `display_name = name`
  - Otherwise: start at `suffix = seen[name]` and increment `suffix` while `f"{name} ({suffix})" in classical`, then set `display_name = f"{name} ({suffix})"`.
- **Verified invariants (`test_classical_display_key_collision_resolution` in `test/test_calculator/test_e6bcompat_differential.py`)**:
  - `(("Custom", e1), ("Custom", e2), ("Custom (1)", e3), ("Custom (2)", e4))` produces distinct keys `["Custom (1)", "Custom (2)", "Custom (1) (1)", "Custom (2) (1)"]`.
  - `(("Custom (1)", e3), ("Custom", e1), ("Custom", e2))` produces distinct keys `["Custom (1)", "Custom (2)", "Custom (3)"]`.
  - `len(energies["Classical"]["sub"]) == len(res.classical)` and `sum(v["value"] for v in sub.values()) == energies["Classical"]["value"]` always hold, while `res.classical` retains the original raw labels.

---

## 4. Complete Packed Compatibility Matrix, 6-Stage Failure Precedence & Machine-Readable Manifest (`Sections 4 & 6`)

### 4.1 Six-Stage Deterministic Failure Precedence

Verified in `test_failure_precedence_six_stages` (`test/test_calculator/test_e6bcompat_differential.py`):

| Precedence Stage | Check | Exception Raised |
| :---: | :--- | :--- |
| **Stage 1** | Removed `store_*` keyword arguments (`reject_removed_store_kwargs`) and unknown public `**kwargs` (when `not legacy_compat`, `strict_e6bcompat=True`, or `not ignore_unknown_kwargs`) | `TypeError` |
| **Stage 2** | Sequence/batch length mismatches and `positions` / `chrg` / `spin` / `field` / `field_grad` shape, dtype, or device mismatches (`DtypeError`, `DeviceError`, `ValueError`, `TypeError`) | `DtypeError` / `DeviceError` / `ValueError` / `TypeError` |
| **Stage 3** | Perturbation vs setup checks: `field` with `ints.level < INTLEVEL_DIPOLE` (`RuntimeError`), `field_grad` with `ints.level < INTLEVEL_QUADRUPOLE` (`RuntimeError`), or explicit `field`/`field_grad` conflicting with custom `efield`/`efieldgrad` interaction label (`ValueError`) | `RuntimeError` / `ValueError` |
| **Stage 4** | Route classification & setup checks: GFN2 non-Fock `scp_mode` in strict mode (`NotImplementedError`), `batch_mode != 0` on lane `System` (`ValueError`), functional `maxiter < 0` (`ValueError`), missing `h0_setup` or `ints.level < INTLEVEL_HCORE` (`NotImplementedError`) | `NotImplementedError` / `ValueError` |
| **Stage 5** | Low-level SCF constructor / Fermi capacity checks: unknown `legacy_options.kwargs` or `legacy_options.args` (`TypeError`), malformed `fwd_options`/`bck_options`/`eigen_options` (`TypeError`), or electron count exceeding orbital capacity in `legacy_compat=True` (`ValueError`) | `TypeError` / `ValueError` |
| **Stage 6** | Post-SCF convergence policy in `legacy_compat=True`: `SCFConvergenceError` (with `.source_id`) when `force_convergence=True`, or `SCFConvergenceWarning` when `force_convergence=False` and `maxiter > 0` | `SCFConvergenceError` / `SCFConvergenceWarning` |

### 4.2 Complete 32-Row `E6bcompat` Routing Manifest (`docs/plan/E6bcompat-final-routing-manifest.json`)

The machine-readable manifest `docs/plan/E6bcompat-final-routing-manifest.json` is validated by `test_e6bcompat_final_routing_manifest` (`test/test_calculator/test_e6bcompat_differential.py`) to ensure every row has valid fields, existing test files, and `verdict == "PASS"`:

| ID | Policy Case | Decision | Candidate Route | Primary Test Evidence | Verdict |
| :--- | :--- | :---: | :--- | :--- | :---: |
| `E6B-01` | GFN1, mode 1, equal-size PyTorch systems | `REPLACE` | `MAPPED` (`evaluate_padded_systems`) | `test_public_gfn1_mode1_matches_serial_and_private_mapped[h2-h2]` | `PASS` |
| `E6B-02` | GFN1, mode 1, different-size padded systems (incl. non-contiguous & trailing zeros) | `REPLACE` | `MAPPED` + mask-aware `pack_system_results` | `test_public_gfn1_mode1_matches_serial_and_private_mapped`, `test_production_routing_differential_gate_matrix` | `PASS` |
| `E6B-03` | GFN1, mode 2, equal-composition conformers (deduplicated `Model.setup`) | `REPLACE` | `MAPPED` (`build_calculator_lane_systems`) | `test_public_gfn1_mode2_conformers_match_serial_and_private_mapped`, `test_production_routing_differential_gate_matrix` | `PASS` |
| `E6B-04` | GFN1, batch_mode=1 and 2 with PyTorch integrals & classicals | `REPLACE` | `MAPPED` | `test_public_and_private_position_gradients_match_serial` | `PASS` |
| `E6B-05` | GFN2, packed batch (Fock mode and non-Fock strict/non-strict SCP policy) | `REPLACE` | `MAPPED` (`_normalize_system_for_e6bcompat`) | `test_public_gfn2_fock_mode_matches_serial_and_private_mapped`, `test_production_routing_differential_gate_matrix` | `PASS` |
| `E6B-06` | C8b H2/H3 and H2/H2O padded cores | `RETAIN` | Direct `PaddedExactSystemCore` | `test_c8b3b_canonical_padding.py`, `test_e6bimpl2b_gfn2_scf.py` | `PASS` |
| `E6B-07` | PyTorch batch integrals (`overlap`, `dipole`, `quadrupole`, `hcore`) | `REPLACE` | `MAPPED` (`evaluate_padded_integrals`) | `test_e6bimpl0_padded_integrals.py`, `test_complete_packed_result_adapter_all_fields_and_model_param_gradients` | `PASS` |
| `E6B-08` | Standard libcint batch setup and eager evaluation | `RETAIN` | `LEGACY_COMPOSED` (`compose_legacy_systems`) | `test_libcint_public_and_composed_match_serial` | `PASS` |
| `E6B-09` | Libcint mixed logical/physical device (`force_cpu_for_libcint`) | `RETAIN` | `LEGACY_COMPOSED` | `test_libcint_composition_matches_independent_serial_systems` | `PASS` |
| `E6B-10` | Built-in supported interactions (`ES2`, `ES3`, `AES2`, `D4SC`, `GeneralizedBorn`/ALPB) | `REPLACE` | `MAPPED` (`PaddedInteractionEvaluation`) | `test_alpb_solvation_packed_matches_serial` | `PASS` |
| `E6B-11` | Custom `Interaction` subclass | `RETAIN` | `LEGACY_COMPOSED` | `test_custom_interaction_is_retained_and_order_independent` | `PASS` |
| `E6B-12` | Custom `Classical` component & multiple `DispersionD3` instances | `RETAIN` | `LEGACY_COMPOSED` / `MAPPED` | `test_calculator_packed_duplicate_classical_labels_select_legacy_composition`, `test_production_routing_differential_gate_matrix` | `PASS` |
| `E6B-13` | Legacy-cached interaction and classical extensions (per-call cache refresh) | `RETAIN` | `LEGACY_COMPOSED` (`refresh_extension_caches=True`) | `test_custom_classical_cache_is_refreshed_per_composition_call` | `PASS` |
| `E6B-14` | Low-level `unexpected_option` in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF._Data.__init__` (`TypeError`) | `test_low_level_unknown_keywords_reach_legacy_constructor[unexpected_option]`, `test_option_ownership_and_side_effects_contract` | `PASS` |
| `E6B-15` | Low-level `solver_options` in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF._Data.__init__` (`TypeError`) | `test_low_level_unknown_keywords_reach_legacy_constructor[solver_options]`, `test_option_ownership_and_side_effects_contract` | `PASS` |
| `E6B-16` | Arbitrary unknown low-level keyword in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF._Data.__init__` (`TypeError`) | `test_low_level_unknown_keywords_reach_legacy_constructor[arbitrary_unknown_scf_option]` | `PASS` |
| `E6B-17` | Low-level positional engine extra in `LegacySCFOptions.args` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF._Data.__init__` (`TypeError`) | `test_low_level_positional_extra_reaches_legacy_constructor`, `test_option_ownership_and_side_effects_contract` | `PASS` |
| `E6B-18` | Low-level `fwd_options` in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF.__init__` | `test_legacy_composition_preserves_option_values_and_constructor_merge` | `PASS` |
| `E6B-19` | Low-level `bck_options` in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF.__init__` | `test_legacy_composition_preserves_option_values_and_constructor_merge` | `PASS` |
| `E6B-20` | Low-level `eigen_options` in `LegacySCFOptions.kwargs` | `RETAIN` | `LEGACY_COMPOSED` / `SEPARATE_ROUTE` | `test_legacy_composition_preserves_option_values_and_constructor_merge` | `PASS` |
| `E6B-21` | Malformed named option dict (`fwd_options=None`, etc.) | `RETAIN` | `LEGACY_COMPOSED` -> `BaseSCF.__init__` (`TypeError`) | `test_legacy_composition_preserves_malformed_option_errors` | `PASS` |
| `E6B-22` | Negative electronic temperature (`fermi_etemp < 0`) | `RETAIN` | `LEGACY_COMPOSED` | `test_negative_fermi_and_outer_maxiter_routing_parity` | `PASS` |
| `E6B-23` | Negative Fermi maxiter (`fermi_maxiter < 0`) | `RETAIN` | `LEGACY_COMPOSED` | `test_negative_fermi_and_outer_maxiter_routing_parity` | `PASS` |
| `E6B-24` | Negative outer SCF maxiter (`maxiter < 0`) on eligible functional FULL | `RETAIN` | `classify_legacy_system` (`ValueError`) | `test_negative_outer_maxiter_keeps_functional_error_policy` | `PASS` |
| `E6B-25` | Negative outer maxiter on legacy-owned FULL (`etemp < 0`, custom ext.) | `RETAIN` | `LEGACY_COMPOSED` | `test_negative_outer_maxiter_on_legacy_owned_routes` | `PASS` |
| `E6B-26` | Missing `h0_setup` / `overlap` or `int_level < INTLEVEL_HCORE` | `RETAIN` | `LEGACY_COMPOSED` (`NotImplementedError` at singlepoint; `ValueError` at direct SCF) | `test_production_routing_differential_gate_matrix`, `test_failure_precedence_six_stages` | `PASS` |
| `E6B-27` | Malformed integral shapes / nonfinite matrices at direct SCF boundary | `RETAIN` | Direct `scf.solve` / `evaluate_padded_integrals` | `test_direct_scf_malformed_integral_shapes_and_nonfinite_matrices` | `PASS` |
| `E6B-28` | Shared charge/spin scalar and per-System charge/spin tensors | `REPLACE` | `_normalize_packed_charges` / `_normalize_packed_spins` | `test_packed_options_charge_spin_and_fermi_contracts` | `PASS` |
| `E6B-29` | Electric field and field gradient (`(3,)` / `(3, 3)` shared vs per-lane) | `REPLACE` | `_normalize_packed_fields` + Stage 3 checks | `test_public_shared_fields_match_serial_and_private_mode2` | `PASS` |
| `E6B-30` | SCF excluded / classical-only (`exclude='scf'` or `'all'`) | `RETAIN` | `SEPARATE_ROUTE` (`_singlepoint_impl` early return + `pack_system_results`) | `test_classical_only_calculation_retains_separate_route`, `test_classical_only_and_non_scf_gfn0_packed_parity` | `PASS` |
| `E6B-31` | Non-self-consistent method (`GFN0-xTB`, `requires_iterations=False`) | `RETAIN` | `SEPARATE_ROUTE` (`solve_nonselfconsistent` + `pack_system_results`) | `test_heterogeneous_batch_matches_separate_solves`, `test_classical_only_and_non_scf_gfn0_packed_parity` | `PASS` |
| `E6B-32` | Public `Calculator.singlepoint` option ownership (`write_*`, `cuda_sync_in_scf`, `store_*`, `ignore_unknown_kwargs`) | `REPLACE` | `_PUBLIC_SINGLEPOINT_OPTIONS` + `_write_packed_integrals` + `timer.cuda_sync` | `test_option_ownership_and_side_effects_contract`, `test_failure_precedence_six_stages` | `PASS` |

---

## 5. Converged Property Acceptance Qualifications (`Section 5`)

In `test/test_calculator/test_b8_e8_functional_properties.py` (`test_section5_primal_vs_derivative_fixed_point_convergence`), we explicitly distinguish three mathematical concepts that must not be conflated:

1. **Primal SCF Root Convergence**:
   - The state residual $\|G(x_K) - x_K\|$ after $K$ fixed-trip iterations satisfies `x_atol=1e-12` (`res.converged == True`, `res.residual.abs().max() < 1e-12`).
   - By variational stationarity of the Mermin free energy ($\partial \mathcal{E}(x_*, \theta)/\partial x_* = 0$), **first-order derivatives** (`forces`, `dipole`, `quadrupole`) have derivative error proportional to the primal root error $O(\|x_K - x_*\|)$ and are already converged to $< 10^{-12}$ once the primal residual reaches $10^{-12}$.
2. **Convergence of Finite-Trip Unrolled Derivatives Toward the Physical Fixed-Point Response (`dx*/dtheta`)**:
   - Higher-order derivatives (`hessian`, `polarizability`, `dipole_deriv`, `pol_deriv`, `hyperpolarizability`, `third_order_directional`) depend on the first- and second-order state responses $\mathrm{d}x_K/\mathrm{d}\theta$ and $\mathrm{d}^2 x_K/\mathrm{d}\theta^2$.
   - Even if the primal state stops updating early or runs a fixed number of unrolled steps $K$, differentiating an unrolled trajectory of length $K$ accumulates $(I + J + \dots + J^{K-1})\,\partial G/\partial \theta$, which converges to $(I - J)^{-1}\,\partial G/\partial \theta$ at rate $O(\rho(J)^K)$.
   - `test_section5_primal_vs_derivative_fixed_point_convergence` verifies on $\text{H}_2\text{O}$ and $\text{H}_2$ (`GFN1-xTB` and `GFN2-xTB`) that:
     - At the just-converged root (`k_just=12` for `GFN1-xTB`, `15` for `GFN2-xTB` at `x_atol=1e-12`), primal energies already agree with deep trajectories (`k_deep=14/17` at `x_atol=1e-14`) to $< 10^{-13}\text{ E}_\text{h}$.
     - Between `k_deep=14/17` and `k_ref=16/19` (`x_atol=1e-14`, Anderson mixing), 1st-order forces agree to $< 10^{-11}$, 2nd-order Hessian agrees to $< 10^{-9}$, and 3rd-order `third_order_directional` (`k=12` vs `14` on $\text{H}_2$) agrees to $< 10^{-8}$.
     - Unrolled `k_ref` forces agree with the independent `SCF_MODE_IMPLICIT` fixed-point response (`(I - \partial G/\partial x_*)^{-1}`, `maxiter=40`, `x_atol=1e-14`) to $< 10^{-9}$.
3. **Internal Automatic-Differentiation Consistency vs Finite Differences**:
   - Identities such as `jacfwd(grad(E)) == jacrev(grad(E))` and agreement with central finite differences of the same $K$-step function verify that PyTorch's forward/reverse AD rules through `System.singlepoint` are algebraically consistent for a given $K$, whereas comparison across increasing $K$ (`k_deep` vs `k_ref`) and `SCF_MODE_IMPLICIT` certifies convergence to the physical fixed-point derivative.

---

## 6. Production Cutover Mechanics (`Section 7`)

### 6.1 Exact 1-Step Cutover Diff (When Authorized by Project Owner)

1. In `BaseCalculator.__init__` (`src/dxtb/_src/calculators/types/base.py:350-358`), cache the deduplicated per-lane `System` tuple for batched calculators:
   ```python
   self._lane_systems = (
       build_calculator_lane_systems(self)
       if self.numbers.ndim == 2
       else (self.system,)
   )
   ```
2. In `EnergyCalculator.singlepoint` (`src/dxtb/_src/calculators/types/energy.py:176-185`), delegate `is_batched` directly to `evaluate_calculator_packed_singlepoint`:
   ```python
   if not is_batched:
       return self._singlepoint_system(
           positions, _chrg, _spin, field, field_grad, kwargs
       )
   result = evaluate_calculator_packed_singlepoint(
       self,
       positions,
       chrg=chrg,
       spin=spin,
       field=field,
       field_grad=field_grad,
       **kwargs,
   )
   self._ncalcs += 1
   return result
   ```

### 6.2 Setup Deduplication in `build_calculator_lane_systems`

- `build_calculator_lane_systems(calc)` (`src/dxtb/_src/calculators/packed_result.py:1351-1444`) keys lane setup by `(unpadded_row_numbers, per_component_2d_numbers, classical_labels, interaction_labels)`.
- For `batch_mode == 2` (conformer batches of size $B$) and repeated compositions in `batch_mode == 1`, `Model.setup(..., batch_mode=0)` is invoked **once per unique composition**, and all identical conformer lanes share the same immutable `System` instance (`lane_sys_m2[0] is lane_sys_m2[1] is lane_sys_m2[2]`).

---

## 7. Complete Legacy Retirement Caller & Dependency Map (`Section 8`)

### 7.1 `IndexHelper.cull` / `IndexHelper.restore` & `InteractionCache.cull` / `InteractionCache.restore` (`E4.6a` / `E4.6b`)

| Symbol / Method | Definition | Remaining Callers in `src/dxtb/` | Retirement Gate |
| :--- | :--- | :--- | :--- |
| `IndexHelper.cull` / `IndexHelper.restore` | `src/dxtb/_src/basis/indexhelper.py:725-777` | Called only by `EnergyCalculator._cull` (`src/dxtb/_src/calculators/types/energy.py:424`) and `EnergyCalculator._restore` (`energy.py:446`). | Retire together with `SelfConsistentFieldFull` (`E4.6a`) once `EnergyCalculator.singlepoint` uses `evaluate_calculator_packed_singlepoint` and `EnergyCalculator.forces_analytical` batched mode is migrated or retired. |
| `InteractionListCache.cull` / `InteractionListCache.restore` & component `Cache.cull` / `Cache.restore` | `src/dxtb/_src/components/interactions/list.py:130-169`, `coulomb/secondorder.py:153-163`, `coulomb/thirdorder.py:124-134`, `coulomb/multipole.py:163-189`, `dispersion/d4sc.py:140-150`, `solvation/alpb.py:216-232`, `field/efield.py:106-116`, `field/efieldgrad.py:109-119` | Called only by `EnergyCalculator._cull` (`energy.py:425`) and `EnergyCalculator._restore` (`energy.py:447`), which in turn are called only by `SelfConsistentFieldFull.cull` / `restore` (`src/dxtb/_src/scf/unrolling/default.py:158-193`) via the `callback` hook passed in `EnergyCalculator.singlepoint` (`energy.py:302`) and `EnergyCalculator.forces_analytical` (`energy.py:604`). | Zero callers in `dxtb._src.scf.functional` or single-System paths. Deletable in `E4.6a` once `SelfConsistentFieldFull` and legacy batched `EnergyCalculator` are retired. |

### 7.2 `SelfConsistentFieldFull`, `SelfConsistentFieldImplicit`, `BaseTSCF`, and Legacy Mixers (`E4.6a` / `E4.6b`)

| Symbol / Module | Definition | Remaining Callers in `src/dxtb/` | Retirement Gate |
| :--- | :--- | :--- | :--- |
| `SelfConsistentFieldFull`, `SelfConsistentFieldFullMixer`, `SelfConsistentFieldSingleShot` | `src/dxtb/_src/scf/unrolling/default.py`, `mixer.py`, `singleshot.py` | Instantiated only in `scf.iterator._solve_impl` (`src/dxtb/_src/scf/iterator.py:355-358`) when `force_legacy=True` (`scf.solve_legacy`), `batch_mode > 0`, `INTDRIVER_LIBCINT`, `fermi_etemp < 0`, `fermi_maxiter < 0`, custom `Interaction`, or `SCF_MODE_EXPERIMENTAL`. | `E4.6a` (after `E6broute` cutover and migration/retirement of remaining legacy-only FULL edge cases). |
| `SelfConsistentFieldImplicit` | `src/dxtb/_src/scf/implicit.py` | Instantiated only in `scf.iterator._solve_impl` (`src/dxtb/_src/scf/iterator.py:311`) when `_use_functional_implicit_route` is `False` or `force_legacy=True` (`MIXER_BROYDEN`, `batch_mode > 0`, `INTDRIVER_LIBCINT`, or custom `Interaction`). | `E4.6b` (once `SCF_MODE_IMPLICIT` routes all supported configurations to `run_prepared_functional_implicit` and Broyden/xitorch implicit fallback is retired). |
| `BaseTSCF` / `BaseSCF` & `src/dxtb/_src/scf/mixer/` (`Simple`, `Anderson`, `Broyden`) | `src/dxtb/_src/scf/base.py`, `src/dxtb/_src/scf/mixer/` | Inherited by `SelfConsistentFieldFull*`, `SelfConsistentFieldImplicit`, and `_Data` used in `solve_nonselfconsistent`. | `E4.6b` (after `E4.6a` and `SelfConsistentFieldImplicit` retirement; replace `_Data` in `solve_nonselfconsistent` with a lightweight non-SCF helper). |

### 7.3 `DriverManager`, `Integrals`, and Packed Integral Drivers (`E2-legacy`)

| Symbol / Module | Definition | Remaining Callers in `src/dxtb/` | Retirement Gate |
| :--- | :--- | :--- | :--- |
| `BaseCalculator._create_legacy_integral_adapter` (`self.integrals: Integrals`) | `src/dxtb/_src/calculators/types/base.py:361-400` | Used in `EnergyCalculator.singlepoint` (`energy.py:234-250`, batched path only) and `EnergyCalculator.forces_analytical` (`energy.py:521-600`, both 1D and batched analytical gradient paths). Single-System `singlepoint` uses `system.integral_setup` / `ExactSystemCore.integral`. | Upon `E6broute` cutover in `EnergyCalculator.singlepoint`, `self.integrals` is no longer used by `singlepoint`; full deletion from `BaseCalculator` is gated on migrating `EnergyCalculator.forces_analytical` to `System.forces` (`E2-legacy`). |
| `DriverPyTorch._positions_batch` / `_basis_batch` / `_ihelp_batch` / `_plan_batch` (`batch_mode > 0` loop) | `src/dxtb/_src/integral/driver/pytorch/driver.py:102-158, 253-263` | Invoked only when `Integrals.build_*` or `Integrals.grad_*` is called on a `Calculator` with `batch_mode > 0`. | Deletable in `E2-legacy` once batched `EnergyCalculator.singlepoint` (`E6broute`) and batched `EnergyCalculator.forces_analytical` route through per-lane `System` evaluation. |

### 7.4 `ClassicalListCache`, `InteractionListCache`, `TensorLike`, and `ModelParameterTree` (`B6` / `C9`)

| Symbol / Module | Definition | Remaining Callers in `src/dxtb/` | Retirement Gate |
| :--- | :--- | :--- | :--- |
| `System._packed_classical_cache` & `ClassicalList.get_cache` (`batch_mode > 0`) | `src/dxtb/_src/calculators/model.py:495, 1711` | Used only by `EnergyCalculator.singlepoint` (`energy.py:192`) and `EnergyCalculator.forces_analytical` (`energy.py:501`) when `batch_mode > 0`. Single-System `Model.setup` (`batch_mode == 0`) never calls `ClassicalList.get_cache()`. | Deletable in `B6` once `Model.setup` drops `batch_mode > 0` after `E6broute` and `forces_analytical` migration. |
| `Classical.cache_is_latest` / `Interaction.cache_is_latest` | `src/dxtb/_src/components/classicals/base.py`, `interactions/base.py` | Only called inside legacy `ClassicalList.get_cache` / `InteractionList.get_cache` when `batch_mode > 0` or by direct legacy component tests. Single-System `_singlepoint_impl` uses immutable `CoreSetup` descriptors on `ExactSystemCore` (`_active_exact_core`). | Deletable in `B6` when legacy stateful component cache lifecycle is retired. |
| `ModelParameterTree` / `ModelParameterNode` & `TensorLike` (`C9`) | `src/dxtb/_src/calculators/model.py`, `src/dxtb/_src/typing/tensorlike.py` | `ExactSystemCore`, `H0SetupNode`, `IntegralBackendNode`, `ModelParameterNode`, `Result`, and all built-in `CoreSetup` classes are registered `Node`s. `TensorLike` remains on legacy `Calculator`, `IndexHelper`, `Basis`, `Component`, `Integrals`, and `BaseSCF`. | `C9` retires `TensorLike` inheritance from remaining legacy classes after `E4.6a`/`E4.6b`/`E2-legacy`/`B6` remove those classes from the active tree. |

---

## 8. Multi-Environment CPU Validation (`Section 10`)

The complete functional, compatibility, and differential test gate was executed across all 5 supported CPU environments (`OMP_NUM_THREADS=1`):

- `test/test_calculator/test_e6bcompat_differential.py`
- `test/test_calculator/test_e6bimpl3_legacy_composition.py`
- `test/test_calculator/test_b8_e8_functional_properties.py`
- `test/test_calculator/test_e7a_padding_safety.py`
- `test/test_calculator/test_e3_end_to_end_acceptance.py`
- `test/test_scf/test_e6bcompat_policy.py`
- `test/test_scf/test_nonselfconsistent.py`
- `test/test_scf/test_functional_anderson.py`
- `test/test_scf/test_functional_temperature_cutoff.py`

| Environment | Python | PyTorch | Result |
| :--- | :---: | :---: | :--- |
| `py3.10-torch2.7.0` | 3.10 | 2.7.0+cpu | **179 passed, 0 failed** |
| `py3.11-torch2.8.0` | 3.11 | 2.8.0+cpu | **179 passed, 0 failed** |
| `py3.12-torch2.11.0` | 3.12 | 2.11.0+cpu | **179 passed, 0 failed** |
| `py3.12-torch2.13.0` | 3.12 | 2.13.0+cpu | **179 passed, 0 failed** |
| `py3.14-torch2.14.1` | 3.14 | 2.14.1+cpu | **179 passed, 0 failed** |

---

## 9. Independent `Luna / high` Review Record (`Section 11`)

- **Reviewer**: Independent Scientific Code Reviewer (`439f9f3c-adeb-4736-bbef-f68f5f219010`; `Luna / high` requested, runtime model identity not exposed).
- **Scope**: All changes in `src/dxtb/_src/calculators/packed_result.py`, `src/dxtb/_src/calculators/legacy_composition.py`, `src/dxtb/_src/calculators/result.py`, `src/dxtb/_src/calculators/types/energy.py`, `src/dxtb/_src/scf/nonselfconsistent.py`, `src/dxtb/_src/scf/iterator.py`, `test/test_calculator/test_e6bcompat_differential.py`, `test/test_calculator/test_b8_e8_functional_properties.py`, `docs/plan/E6bcompat-final-routing-manifest.json`, and `docs/plan/functional-route-final-acceptance.md`.
- **Verdict**: **`APPROVED (NO FINDING)` (`0 BLOCKER`, `0 MAJOR`, `0 MINOR`)**.
  - Verified public vs low-level option separation, `write_*` file output, `cuda_sync_in_scf` `try`/`finally` scoping, `GFN0-xTB` `_Data.__init__` option parity, classical display-key collision resolution, conformer `Model.setup` deduplication, non-contiguous/trailing zero-padding masks, 6-stage failure precedence, Section 5 fixed-point response convergence qualification, and the 32-row `E6bcompat` routing manifest.
  - Applied reviewer informational note synchronizing the `E6B-30`/`E6B-31` row summary (`10 REPLACE` / `22 RETAIN`) between `docs/plan/functional-route-final-acceptance.md` and `docs/plan/E6bcompat-final-routing-manifest.json`.
