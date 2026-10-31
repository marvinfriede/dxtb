# Functional Refactoring Closure Report

## 1. Executive Summary & Three-Level Architecture

This report documents the completion of the functional, differentiable `Model` / `System` / `Result` refactoring across Tracks TB, TC, and TE prior to any further conformer-batching performance optimization.

### Non-Negotiable Three-Level Persistent Architecture

1. **`Model` (`src/dxtb/_src/calculators/model.py`)**
   - Owns method parameters (`ParamModule`), parameter trees (`Model.parameter_tree` / `ModelParameterNode`), configuration (`Config`), and method-level component definitions.
   - Provides `Model.setup(numbers)` and `Model.setup_functional(numbers, parameter_tree)` to construct immutable `System` instances with full parameter differentiability.
2. **`System` (`src/dxtb/_src/calculators/model.py`)**
   - Owns structural, immutable setup derived solely from atomic numbers and `Model` setup (`ExactSystemCore`, `H0SetupNode`, `IntegralBackendNode`, and immutable `CoreSetup` descriptors for Repulsion, SRB, D3, D4, IES, Halogen, ES2, ES3, AES2, D4SC, and GeneralizedBorn/ALPB).
   - Contains no per-call geometry, charge, spin, field, or trajectory state.
   - Exposes pure functional evaluation (`System.singlepoint`) and 1st-, 2nd-, and 3rd-order response/property evaluators (`System.energy`, `System.forces`, `System.hessian`, `System.dipole`, `System.quadrupole`, `System.polarizability`, `System.dipole_deriv`, `System.pol_deriv`, `System.hyperpolarizability`, `System.third_order_directional`, `System.bond_orders`, `System.vibration`, `System.ir`, `System.raman`).
3. **`Result` (`src/dxtb/_src/calculators/result.py`)**
   - Owns all tensor values produced by one calculation (`energy`, `scf`, `classical`, `fenergy`, `iterations`, `charges`, `density`, `coefficients`, `emo`, `occupation`, `potential`, `hamiltonian`, `overlap`, `hcore`, `dipole_integrals`, `quadrupole_integrals`, `overlap_norm`, `converged`, `residual`, `status`).
   - Registered as a PyTorch PyTree node; no persistent fourth mutable cache or driver-state layer exists in the functional path.

---

## 2. Milestone A — Runtime Parallelism Policy & W2c9–W2c11 Component Audit

### 2.1 Removal of Application-Managed Multiprocessing

All application-managed multiprocessing and thread-pool orchestration introduced during experimental `W2c11` benchmarking has been removed from the codebase:

- **`src/dxtb/_src/calculators/batch_implicit.py`**:
  - Removed `import multiprocessing as mp`, `BatchPreparationDiagnostics`, `_snapshot_object_tree`, `_restore_object_tree`, `_read_process_memory_mb`, `_init_primal_worker`, `_worker_solve_lane_task`, `preparation_diagnostics`, and the `num_workers` parameter on `prepare_converged_root_batch` and `evaluate_converged_root_batch`.
  - Removed all internal `OMP_NUM_THREADS` and `torch.set_num_threads` overrides.
- **`src/dxtb/_src/calculators/batch_streaming.py`**:
  - Removed the `num_workers` parameter from `stream_batch_value_and_grad`.
- **`test/test_scf/test_e6c1_w2c9_batch_implicit.py`**:
  - Removed `concurrent.futures.ThreadPoolExecutor` and replaced multiprocessing tests with `test_w2c11_direct_system_spectral_prep_parity_and_failure_semantics`.
- **Policy**: Parallel CPU execution inside `dxtb` is delegated exclusively to PyTorch and user/environment thread settings (`OMP_NUM_THREADS`). `pytest-xdist` is retained for parallel test execution. Historical `W2c9`–`W2c11` benchmark records are preserved with explicit provenance notes that `W2c11` worker-pool measurements used an experimental architecture that has been retired.

### 2.2 Classification of W2c9–W2c11 Components

| Component / Symbol | Location | Classification | Rationale |
| :--- | :--- | :--- | :--- |
| Cancellation-safe Fermi Newton slope (`sech²` via `4e⁻ˣ/(1+e⁻ˣ)²` and `1 - tanh²`) | `src/dxtb/_src/wavefunction/filling.py` | **(1) Necessary for final functional SCF core** | Eliminates catastrophic cancellation in Fermi chemical-potential Newton steps and implicit derivatives. |
| Derivative-aware electron-count response guard (`_electron_count_ok` active under transforms) | `src/dxtb/_src/wavefunction/filling.py` | **(1) Necessary for final functional SCF core** | Ensures `FermiStatus.electron_count_ok` is tracked accurately during both primal and `torch.func` evaluations. |
| Direct-RHS `tanh` doubling in matrix-Fermi projector (`_fermi_matrix_ newton` / `_solve_positive_definite`) | `src/dxtb/_src/scf/functional/spectral.py` | **(1) Necessary for final functional SCF core** | Provides gauge-invariant, eigenvector-free spectral density and free-energy matrices across degeneracies. |
| Low-temperature equal-partition analytical entropy trace (`free_energy_from_spectral`) | `src/dxtb/_src/scf/functional/solver.py` | **(1) Necessary for final functional SCF core** | Avoids $1\text{ ULP}$ ($2.65\times 10^{-23}\text{ E}_\text{h}$) matrix-product roundoff at sub-cutoff $k_B T$ while preserving exact fractional-charge derivatives. |
| Detached-primal matrix-Fermi mode (`_eager_primal_fermi_mode("spectral")`) | `src/dxtb/_src/scf/functional/spectral.py` | **(2) Useful as an isolated numerical helper** | Accelerates detached primal SCF root preparation without altering the differentiable evaluation path. |
| `PreparedConvergedRootBatch`, `prepare_converged_root_batch`, `evaluate_converged_root_batch` | `src/dxtb/_src/calculators/batch_implicit.py` | **(3) Useful only for experimental verification** | Internal converged-root IFT batch prototype; retained as private experimental verification infrastructure (not a public API). |
| `stream_batch_value_and_grad`, `StreamingBatchResult` | `src/dxtb/_src/calculators/batch_streaming.py` | **(3) Useful only for experimental verification** | Internal constant-memory streaming differentiation prototype; retained for private verification only. |
| Application-managed `mp.get_context("spawn")`, `_worker_solve_lane_task`, `_snapshot_object_tree` | Formerly in `batch_implicit.py` | **(4) Redundant / unnecessarily complicated (Deleted)** | Violated runtime parallelism policy and duplicated PyTorch's native CPU thread parallelism; deleted in Milestone A. |
| Legacy `BaseTSCF` / `SelfConsistentFieldFull` / `SelfConsistentFieldImplicit` unrolled/implicit classes | `src/dxtb/_src/scf/base.py`, `unrolling.py`, `implicit.py` | **(5) Intended for eventual deletion** | Superseded for all standard PyTorch `SCF_MODE_FULL` calculations by `dxtb._src.scf.functional`; retained only for compatibility gates (`E4.6a`/`E4.6b`/`C9`/`B6`). |

---

## 3. Milestone B — E3 Numerical & Derivative Acceptance

### 3.1 Root Causes and Resolution of InheritedSCF Test Discrepancies

1. **No-Legacy-Hooks Gate (`test_functional_map_has_no_legacy_stateful_hooks`)**:
   - *Cause*: `int(scale_power.max().item())` in `src/dxtb/_src/scf/functional/spectral.py:342` triggered the AST ban on `.item()` inside `dxtb._src.scf.functional`.
   - *Fix*: Replaced with `int(scale_power.max())` (under `torch.no_grad()` on a detached exponent bound).
2. **`maxiter=0` Raw vs Final Spectrum Status (`test_fock_maxiter_zero_*`, `test_potential_maxiter_zero_*`)**:
   - *Cause*: In Potential (`SCP_MODE_POTENTIAL`) and Fock (`SCP_MODE_FOCK`) representations, `finalize_scf_potential` and `finalize_scf_fock` evaluate a second, distinct final spectrum after the initial raw map even when `maxiter == 0`. Passing `record_initial_status=maxiter > 0` omitted the initial raw map's Fermi status at `maxiter=0`.
   - *Fix*: Added explicit `record_initial_status: bool | None = None` to `_solve_map`, `_solve_map_control`, and `_solve_map_anderson`, passing `record_initial_status=True` in `solve_scf_potential_simple`, `solve_scf_fock_simple`, `solve_scf_potential_anderson`, and `solve_scf_fock_anderson`.
3. **Sub-Cutoff Positive Temperature Free Energy (`test_positive_subcutoff_temperature_preserves_legacy_free_energy`)**:
   - *Cause*: Computing `torch.diagonal(inputs.integrals.overlap @ aufbau_free_energy_matrix(...)).sum(-1)` at $T = 10^{-3}\text{ K}$ ($k_B T \approx 3.16\times 10^{-9}\text{ E}_\text{h} < 3\times 10^{-7}\text{ E}_\text{h}$) introduced a $1\text{ ULP}$ ($2.65\times 10^{-23}\text{ E}_\text{h}$) floating-point roundoff relative to $-k_B T \sum_i s(n_i)$.
   - *Fix*: Added an exact low-temperature equal-partition trace branch in `free_energy_from_spectral` (`src/dxtb/_src/scf/functional/solver.py`) using `filling.get_fermi_entropy_low_temperature(spectral.occupation, settings.kt)` masked by `inputs.orbital_mask`.
4. **Anderson & Sub-Cutoff Gradient Parity vs Legacy Eigenvector Broadening (`test_anderson_solver_real_position_parameter_and_field_gradients`, `test_subcutoff_fractional_charge_gradient_matches_legacy`)**:
   - *Mathematical Cause*: Legacy `BaseTSCF.diagonalize` calls `storch.eighb(..., broadening_method="lorn", factor=eps**0.5)` which replaces exact eigenvalue differences $\varepsilon_i - \varepsilon_j$ in the eigenvector VJP with the Lorentzian approximation $(\varepsilon_i - \varepsilon_j) / ((\varepsilon_i - \varepsilon_j)^2 + b_f^2)$ where $b_f = \sqrt{\epsilon_{\text{mach}}} \approx 1.49\times 10^{-8}$. Even for non-degenerate orbitals with gap $\Delta \varepsilon \approx 0.14\text{ E}_\text{h}$, this injects a systematic relative derivative bias of $O(b_f^2 / \Delta\varepsilon^2) \approx 1.1\times 10^{-14}$ into legacy eigenvector gradients. By contrast, the functional SCF core (`aufbau_density_matrix` in `spectral.py`) uses the exact gauge-invariant matrix-sign projector without artificial Lorentzian gap broadening.
   - *Verification*: Both tests now verify (i) agreement with central finite differences (`_assert_directional_fd`) and (ii) exact agreement (`atol=2e-13`, actual max diff $3.88\times 10^{-16}$) against unbroadened legacy (`broadening_method=None`).

### 3.2 Finite-Trajectory Unrolled vs Converged Physical-Root Implicit Differentiation

- **Finite-Trajectory Unrolled Differentiation (`System.singlepoint`)**: Differentiates the exact $K$-step fixed-trip map composition $x_K(\theta) = (\mathcal{M}_\theta \circ \cdots \circ \mathcal{M}_\theta)(x_0(\theta))$. At small unconverged `maxiter` (e.g. `maxiter=1`), $\partial E(x_K(\theta), \theta)/\partial \theta$ is the exact derivative of the truncated $K$-step approximation and differs from the converged root derivative by $O(\|J_{\mathcal{M}}^K\|)$.
- **Converged Physical-Root Implicit Differentiation (`run_prepared_functional_implicit` / `implicit_root`)**: Solves $(I - \partial \mathcal{F}/\partial x_*)\,\mathrm{d}x_* = (\partial \mathcal{F}/\partial \theta)\,\mathrm{d}\theta$ at a converged fixed point $x_* = \mathcal{F}(x_*, \theta)$. When the unrolled trajectory is run to tight convergence (`x_atol=1e-12`), unrolled 1st and 2nd derivatives agree with implicit-root 1st and 2nd derivatives to $< 10^{-9}$ and $< 10^{-7}$, respectively (`test_unrolled_trajectory_vs_converged_implicit_root_derivatives` in `test/test_calculator/test_e3_end_to_end_acceptance.py`).

---

## 4. Milestone C — Functional Mixed-System Integration (`E6b`) & Public Packed `Result` Adapter (`E6broute`)

Implemented in `src/dxtb/_src/calculators/packed_result.py`:

- **`pack_system_results(results, *, legacy_compat=False) -> Result`**:
  - Converts any sequence of decoded single-System physical `Result` objects into the public batched `Result` layout (`(B, max_atoms)` and `(B, max_orbitals, max_orbitals)` with right-zero padding), preserving autograd across `energy`, `scf`, `classical`, `fenergy`, `charges` (`mono`, `dipole`, `quadrupole`), `multipoles`, `potential` (`mono`, `dipole`, `quadrupole`), `density`, `coefficients`, `emo`, `occupation`, `hamiltonian`, `overlap`, `hcore`, `dipole_integrals`, `quadrupole_integrals`, `overlap_norm`, `iterations`, `converged`, `residual`, and `status`.
- **`evaluate_packed_systems(systems, positions, chrg=0, spin=None, field=None, field_grad=None, *, legacy_options=None, legacy_compat=False, strict_e6bcompat=False) -> Result`**:
  - Candidate `E6broute` mixed-System batch evaluator that classifies each lane via `classify_legacy_system`, buckets `MAPPED` lanes by canonical exact-core signature via `bucket_canonical_exact_cores` + `evaluate_padded_systems`, routes `LEGACY_COMPOSED` and `SEPARATE_ROUTE` lanes via `compose_legacy_systems`, enforces eager `E6bcompat` warnings/errors (including GFN2 non-Fock `scp_mode` and unknown kwargs), and packs the resulting physical `Result`s in original source order.
  - Per governance rules, default `Calculator.singlepoint` batch dispatch remains on the existing route until `E6c1` `W2`/`W3` performance gates are either cleared or explicitly waived by the project owner.

---

## 5. Milestone C/D — Broader `E7` Padding Safety Audit

Verified in `test/test_calculator/test_e7a_padding_safety.py` (`test_broader_e7_padding_safety_audit_matrix`, `test_mixed_valid_and_invalid_padded_lanes_isolation`, and existing `E7a` tests) across GFN1 and GFN2:

| Numerical Stage / Primitive | Padding Protection Mechanism | Verified Under Zero / Coincident / Noncontiguous Padding & 1st–3rd Derivatives |
| :--- | :--- | :--- |
| Pair distances & reciprocal powers | Masked off-diagonal pair masks (`atom_pair_mask`, `shell_pair_mask`, `mask`) and zeroed inactive coordinates at entry (`positions = torch.where(atom_mask, positions, 0.0)`) | Passed |
| Coordination numbers (`d3`, `d4`, `d4sc`) | Masked pair weights (`rcov`/`r4r2` zeroed on inactive slots) | Passed |
| Second- & third-order Coulomb (`es2`, `es3`) | `ES2CoreSetup` pair masks zero inactive rows/cols; `PaddedCoreIndexData` masks shell/atom reductions | Passed |
| Anisotropic multipoles (`aes2`) | `AES2CoreSetup.pair_mask` zeros `amat_sd`, `amat_dd`, `amat_sq` on invalid atom pairs before contraction | Passed |
| Generalized overlap & $\mathbf{S}^{-1/2}$ (`inverse_cholesky`) | `safe_overlap` zeros off-block rows/cols and sets $1.0$ on inactive diagonal before `_symmetric_inverse_sqrt` | Passed |
| Core Hamiltonian (`padded_hcore`) & orbital masks | Inactive diagonal shifted to `PADDED_ORBITAL_ENERGY = 1.0e4`; off-block entries zeroed by `matrix_mask` | Passed |
| Occupations & Fermi chemical potential | `orbital_mask` excludes dummy orbitals from electron count, Aufbau filling, Fermi Newton iterations, and entropy traces | Passed |
| Spectral transformations (`evaluate_padded_spectrum`) | Inactive block decoupled by block-diagonal $\mathbf{S}$ and $\mathbf{H}$; inactive density and Mulliken charges are identically zero | Passed |
| Density & potential assembly | `PaddedCoreIndexData.spread_*` and `reduce_*` mask inactive shells/orbitals at every transfer | Passed |
| Convergence residuals & mixed valid/invalid lanes | Inactive state slots projected to zero; invalid Fermi/SCF lanes record status codes in `SCFStatus` without `NaN`/`Inf` or cross-lane gradient leakage | Passed |

---

## 6. Milestone D & E — Downstream Functional Property API (`B8a`/`B8b`/`E8`) & Legacy Retirement Inventory

### 6.1 Functional Property Composition (`B8a` / `B8b` / `E8`)

Implemented in `src/dxtb/_src/calculators/properties/functional.py` and bound directly on `System` (`src/dxtb/_src/calculators/model.py`):
- **1st & 2nd order (`B8a`)**: `energy`, `forces`, `hessian`, `dipole`, `quadrupole`, `polarizability`, `dipole_deriv`, `bond_orders`, `vibration`, `ir`, `system_parameter_force_jacobian`.
- **3rd order (`B8b` / `E8`)**: `pol_deriv`, `hyperpolarizability`, `third_order_directional`, `raman`.
- **Legacy Cache-Reset Cleanup**: `AutogradCalculator.ir` and `AutogradCalculator.raman` (`src/dxtb/_src/calculators/types/autograd.py`) no longer invoke `reset_all()` when evaluating single-System exact cores (`self.ihelp.batch_mode == 0 and self.system._active_exact_core() is not None`).

### 6.2 Legacy Retirement Readiness Inventory (`E6broute` / `E4.6a` / `E4.6b` / `C9` / `B6`)

| Package / Legacy Component | Current Status Category | Details & Remaining Gate |
| :--- | :--- | :--- |
| **`E6bimpl` + `E6bcompat` + `E6broute` candidate (`packed_result.py`)** | **(a) Implemented and ready to merge** | `pack_system_results` and `evaluate_packed_systems` are implemented and verified against `Calculator(batch_mode=1\|2)` and `System.singlepoint`. |
| **`B8a` / `B8b` / `E8` functional property & 3rd-order suite** | **(a) Implemented and ready to merge** | Pure `torch.func` property evaluators on `System` and `dxtb._src.calculators.properties.functional` verified for GFN1 and GFN2. |
| **Default `Calculator.singlepoint` batch routing (`E6broute` activation)** | **(b) Blocked only by routing approval / `E6c1` waiver** | Candidate route `evaluate_packed_systems` is ready; switching `EnergyCalculator.singlepoint` batch dispatch awaits either `E6c1` `W2`/`W3` performance clearance or an explicit project-owner waiver deferring `W2`/`W3` to the post-refactoring optimization phase. |
| **Legacy `SelfConsistentFieldFull` (`src/dxtb/_src/scf/unrolling.py`) & `BaseTSCF` (`src/dxtb/_src/scf/base.py`) (`E4.6a` / `E4.6b`)** | **(c) Retained for explicit legacy compatibility scopes** | Single-System PyTorch `SCF_MODE_FULL` already routes to `dxtb._src.scf.functional`. Legacy `unrolling.py` and `base.py` remain required only for: (1) `Calculator(batch_mode=1\|2)` until `E6broute` is activated, (2) `INTDRIVER_LIBCINT` batched/unrolled paths, (3) `SCF_MODE_UNROLL_MIXER` / `SCF_MODE_EXPERIMENTAL`, (4) negative `fermi_etemp`, and (5) custom non-Node `Interaction` subclasses. |
| **Legacy `SelfConsistentFieldImplicit` (`src/dxtb/_src/scf/implicit.py`)** | **(c) Retained for `SCF_MODE_IMPLICIT` on `Calculator`** | Functional implicit root (`run_prepared_functional_implicit` / `implicit_root` in `scf/functional/implicit.py`) is complete; legacy `SelfConsistentFieldImplicit` is retained for `Calculator` `scf_mode="implicit"` compatibility until `E4.6b` cutover. |
| **Legacy mutable component caches (`ClassicalList.get_cache`, `InteractionList.get_cache`) (`C9` / `B6`)** | **(c) Retained for packed `Calculator` & custom extension subclasses** | Single-System built-in terms use immutable `CoreSetup` Nodes on `ExactSystemCore` exclusively (`_active_exact_core`). Legacy `get_cache` is only invoked for `batch_mode > 0` or user-supplied custom `Classical`/`Interaction` subclasses. |

---

## 7. Independent `Luna / high` Review & Multi-Environment Verification

### 7.1 Independent Code Review Findings & Resolutions
- **Reviewer**: Independent Scientific Code Reviewer (`Luna / high` requested; actual model identity not exposed by runtime).
- **Pass 1 Findings Resolved**:
  1. **`MAJOR 1` — Electronic Free Energy (`fenergy`) Double-Counting**: Because `Result.energy` is already defined as $E_{\text{classical}} + E_{\text{scf}} + E_{\text{fenergy}}$, `system_energy` (`src/dxtb/_src/calculators/properties/functional.py`) and `System.energy` (`src/dxtb/_src/calculators/model.py`) now default to `include_free_energy: bool = True` (returning `res.energy.sum(-1)`) and subtract `res.fenergy` only when `include_free_energy=False`. `test_e3_end_to_end_acceptance.py` (`openshell_energy`, `free_energy`) now evaluates `out.energy.sum()`, and `test_b8_e8_functional_properties.py` explicitly verifies finite-temperature Mermin stationarity (`system.forces` vs. `Calculator.forces_analytical` on open-shell $\text{OH}^\bullet$ at $5000\text{ K}$).
  2. **`MAJOR 2` — Default `spin=None` Handling for Open-Shell Systems in `evaluate_packed_systems`**: `evaluate_packed_systems` (`src/dxtb/_src/calculators/packed_result.py`) now preserves `None` when `spin` is omitted, forwarding `None` to `compose_legacy_systems` and deriving the default ground-state spin parity `(nab[0] - nab[1]).reshape(())` via `validate_padded_charge_spin(core, charge, None)` for `MAPPED` lanes. Covered in `test_e6bcompat_differential.py`.
  3. **`MINOR 1` — `pack_system_results` Occupation Fallback Shape**: In mixed SCF / non-SCF batches (`r.occupation is None`), `pack_system_results` now constructs zero-padded fallback tensors using the leading spin-channel prefix `(*occ_prefix, max_mos)` (`(2, max_mos)`). Covered in `test_e6bcompat_differential.py`.

### 7.2 Multi-Environment CPU Validation
The 10-module acceptance gate (`test_e6c1_w2c9_batch_implicit.py`, `test_functional_anderson.py`, `test_functional_temperature_cutoff.py`, `test_functional_map.py`, `test_functional_fock.py`, `test_functional_potential.py`, `test_e3_end_to_end_acceptance.py`, `test_b8_e8_functional_properties.py`, `test_e7a_padding_safety.py`, `test_e6bcompat_differential.py`) passed across all 5 supported CPU environments (`py3.10-torch2.7.0`, `py3.11-torch2.8.0`, `py3.12-torch2.11.0`, `py3.12-torch2.13.0`, `py3.14-torch2.14.1`): **241 passed, 0 failed** in every environment.

