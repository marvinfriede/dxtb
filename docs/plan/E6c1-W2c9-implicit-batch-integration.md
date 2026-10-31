# E6c1-W2c9: Converged-Root Conformer Batch Architecture, Spectral Optimization, and Staged `tmpda` Certification

## 1. Executive Summary

Package `E6c1-W2c9` transitions the functional SCF converged-root architecture from the `W2c8` design decision (`7d71e930e5bdbd52a252bf23d21803b94d2695f4`) to a reusable internal orchestration layer, a targeted spectral/map performance optimization, a comprehensive acceptance test suite, and a staged `tmpda` conformer batch benchmark progression (published in commits `38745b0f1f13344602c022a8d1a8779351b9f062`, `b53bf6d2270b6e85708214edd01887f9cde1aa63`, and `7b946a21fcd27807127d54b4262b5ec5177d607d`):

1. **Converged-Root Batch Orchestration (`src/dxtb/_src/calculators/batch_implicit.py`)**:
   - Implemented [`prepare_converged_root_batch`](../../src/dxtb/_src/calculators/batch_implicit.py) and [`evaluate_converged_root_batch`](../../src/dxtb/_src/calculators/batch_implicit.py) along with [`PreparedConvergedRootBatch`](../../src/dxtb/_src/calculators/batch_implicit.py), [`ConvergedRootLaneRecord`](../../src/dxtb/_src/calculators/batch_implicit.py), and [`BatchRootFailureCode`](../../src/dxtb/_src/calculators/batch_implicit.py).
   - Supports conformer batches on a single [`System`](../../src/dxtb/_src/calculators/model.py) as well as noncontiguous canonical padded [`PaddedExactSystemCore`](../../src/dxtb/_src/calculators/padding.py) owners without cloning `Model` or retaining the primal SCF trajectory graph.
   - Preserves public `System.singlepoint()` finite-trip derivative semantics and public `Calculator` routing unchanged.

2. **Targeted Spectral & Map Optimization (`src/dxtb/_src/scf/functional/spectral.py`)**:
   - Guarded `_gap_electron_response` with `_tracks_derivatives(nel) or _tracks_derivatives(count_delta)` (including outer functorch `Jvp` interpreter levels) so the extra matrix-Fermi solve for the chemical-potential electron-count sensitivity $\partial P / \partial N_{\mathrm{el}}$ is skipped whenever the target electron count does not carry active forward- or reverse-mode derivatives (which holds on every primal SCF step and every position/Hamiltonian/overlap derivative evaluation).
   - Unified `fermi_density_matrix` and `_fermi_free_energy_setup`, replacing the redundant `N x N` matrix multiplication `(density - density @ density).diagonal().sum(-1)` in the Newton slope denominator with the symmetric-density identity $\mathrm{Tr}(D - D^2) = N_{\mathrm{occ}} - \sum_{ij} D_{ij}^2$ while keeping `spectral.py` 100% pure functional and stateless.
   - The historical `kernel_ab.json` recorded faster timings, but its “baseline” forcibly enabled electron-count response and was not the complete pre-W2c9 implementation. Those figures are retained as a forced-response microcomparison only; they are not accepted as source-revision attribution. The corrected four-way pre-W2c9/guard-only/slope-only/full comparison is recorded in [`E6c1-W2c10-batch-hardening-and-streaming.md`](E6c1-W2c10-batch-hardening-and-streaming.md).

3. **End-to-End `tmpda` Conformer Batch Performance & Numerical Parity (`benchmarks/results/e6c1-w2c9/`)**:
   - **1 conformer (`positions-only`)**: Completed in **97.53 s** with **1,422.21 MiB (1.39 GiB)** peak RSS (down from W2c7's **143.94 s** and **2,567.79 MiB / 2.51 GiB** — **32.24% faster** and **44.61% lower peak RSS**).
   - **1 conformer (`positions-and-model`)**: Completed in **102.42 s** with **1,667.18 MiB (1.63 GiB)** peak RSS across all 2,261 floating `Model` parameters (`73` connected non-zero, `4` connected zero, `2,184` disconnected).
   - **2 conformers (`positions-only`)**: Completed in **239.21 s** with **2,366.15 MiB (2.31 GiB)** peak RSS (down from W2c7's **429.36 s** and **3,707.16 MiB / 3.62 GiB** — **44.29% faster / 1.80x speedup** and **36.17% lower peak RSS**).
   - **2 conformers (`positions-only`, reversed order `[1, 0]`)**: Completed in **241.46 s** with **2,399.49 MiB (2.34 GiB)** peak RSS, matching forward-order energies within `2.54e-13 Eh`, free energies within `3.32e-16 Eh`, and position gradient norms within `2.84e-14 Eh/Bohr`.
   - **2 conformers (`positions-and-model` + serial/cross-lane parity)**: Completed in **251.91 s** with **2,929.82 MiB (2.86 GiB)** peak RSS (down from W2c7's **470.02 s** and **4,418.49 MiB / 4.31 GiB** — **46.40% faster / 1.87x speedup** and **33.69% lower peak RSS**). Exact parity certification (`tmpda_n2_model_parity.json`) confirmed `cross_lane_pos_grad_max_abs = 0.0`, `energy_max_abs_diff = 2.54e-13 Eh`, `fenergy_max_abs_diff = 3.32e-13 Eh`, `pos_grad_max_abs_diff = 6.35e-14 Eh/Bohr`, and `model_param_grad_max_abs_diff = 1.56e-13` across all 2,261 leaves.
   - **4 conformers (`positions-only`)**: Completed in **499.24 s** with **5,732.04 MiB (5.60 GiB)** peak RSS. All 4 conformers converged (`66, 96, 112, 72` SCF iterations; `5, 7, 6, 7` BiCGSTAB linear-response iterations), and lanes 0 and 1 matched the 1- and 2-conformer energies, free energies, and position gradients to `0.0`.
   - **8-conformer viability assessment**: Based on the 4-conformer memory profile (`5.60 GiB` peak RSS, with `~1.1–1.2 GiB` per simultaneous mapped lane retained in the batched `vmap(implicit_root)` forward/linearize autograd graph), an unchunked 8-conformer simultaneous mapped evaluation is projected to require **~11.5–12.2 GiB peak RSS** (exceeding the historical `8.16 GiB` W2 ceiling) and **~1,000 s** single-threaded wall time. In accordance with the Workstream E gate, an unchunked 8-conformer run was not launched; `W2` and `W3` remain **OPEN** and `E6broute` remains **BLOCKED**.

---

## 2. Architectural Diagnosis and Design

### 2.1 Shared vs. Lane-Local Data Ownership

The converged-root conformer batch architecture in [`src/dxtb/_src/calculators/batch_implicit.py`](../../src/dxtb/_src/calculators/batch_implicit.py) enforces a strict separation between geometry-independent shared data and per-conformer call-local data:

- **Shared across the batch (not cloned per conformer)**:
  - `System` (or `PaddedExactSystemCore`) structural setup, atomic `numbers`, basis/shell indexing (`ihelp`), classical and interaction setup `Node`s, and shared `Model` parameter leaves.
- **Lane-local per conformer (`ConvergedRootLaneRecord`)**:
  - Conformer `positions` (`(nat, 3)`), scalar `charge`, `spin`, optional external `field` / `field_grad`, geometry-dependent `IntegralMatrices`, `InteractionListCache`, `SCFInputs`, detached physical `root_x`, independent residual norms, Fermi/solver status, and `_SCFMapTensorTree` adapter.

### 2.2 Two-Stage Orchestration (`prepare_converged_root_batch` and `evaluate_converged_root_batch`)

1. **Stage 1 — Detached Primal Root Preparation (`prepare_converged_root_batch`)**:
   - Unwraps any active functorch/dual/autograd tracking from `positions`, `charge`, `spin`, `field`, `field_grad`, and `System` parameter leaves via `_unwrap_primal_tensor` and `_detach_scf_inputs`.
   - Runs the existing functional SCF primal solver (`_solve_map` for `Linear` mixing or `_solve_map_anderson` for `Anderson`/`Broyden` mixing) with `eager=True` under `torch.no_grad()` independently for each lane.
   - For canonical padded systems (`PaddedExactSystemCore`), projects the initial guess onto the active subspace (`initial_state = state_projection(initial_state)`) prior to eager solving so inactive padded slots remain identically zero.
   - Independently validates each lane's candidate root by evaluating one fresh raw `scf_map` / `potential_map` / `fock_map` call and checking all physical root invariants without retaining the SCF trajectory graph.

2. **Stage 2 — Differentiable Implicit Attachment and Finalization (`evaluate_converged_root_batch`)**:
   - Rebuilds live, graph-bearing `IntegralMatrices`, `SCFInputs`, and classical energies for each valid lane while preserving the level-0 `_SCFMapTensorTree` template (`lane.adapter`) so `torch.func` transforms (`jvp`, `jacfwd`, `jacrev`, `grad`, `vmap`, higher-order AD) never encounter cross-level functorch wrapper mismatches inside `_BatchedNumericalSolve.vmap`.
   - Maps `implicit_root` + representation-specific finalization (`finalize_scf`, `finalize_scf_potential`, or `finalize_scf_fock`) across all valid lanes simultaneously via `torch.func.vmap(_evaluate_valid_lane, in_dims=...)`.
   - Decorates the `vmap(implicit_root)` dispatch boundary (`_run_mapped_lanes`) with `@torch._dynamo.disable` so `torch.compile` compiles setup, integrals, classicals, and result assembly while cleanly graph-breaking at the custom functorch/autograd primitive boundary.
   - Assembles a batched [`Result`](../../src/dxtb/_src/calculators/result.py) with leading batch dimension `B` across all energy terms, charges, dipoles, potentials, densities, orbital energies/occupations, and public tensor-valued [`SCFStatus`](../../src/dxtb/_src/scf/status.py).

---

## 3. Root Preparation and Validation Contract

Every conformer lane is independently validated against an explicit eight-point physical-root checklist before implicit differentiation is attached:

1. **Fermi configuration validity**: Rejects invalid electronic temperature (`FERMI_FAILURE_INVALID_TEMPERATURE`) and unphysical electron counts exceeding basis capacity (`FERMI_FAILURE_TOO_MANY_ELECTRONS`).
2. **Fermi solver convergence**: Requires all spin channels of `fermi_status.converged` to be `True` (`BatchRootFailureCode.FERMI_NONCONVERGENCE`).
3. **Electron-count conservation**: Requires $\max_s |\mathrm{Tr}(P_s S) - N_{\mathrm{el},s}| \le 10^{-6}$ (`BatchRootFailureCode.ELECTRON_COUNT`).
4. **Finite state and density**: Requires `root_x`, `mapped_x`, and `spectral.density` to be finite (`BatchRootFailureCode.NONFINITE_STATE`).
5. **Canonical padding zero-invariance**: For padded systems, requires inactive atom/shell/orbital entries of `root_x`, `mapped_x`, and `spectral.density` to be identically zero (`BatchRootFailureCode.INVALID_PADDING`).
6. **Independent raw-map residual check**: Recomputes $r(x^\star) = G(x^\star) - x^\star$ from a fresh map evaluation and requires both $\|r(x^\star)\|_2 \le \max(\text{x\_atol}, 10\epsilon)$ and $\|r(x^\star)\|_\infty \le \max(\text{x\_atol\_max}, 10\epsilon)$.
7. **Iteration budget vs. residual classification**: Distinguishes `BatchRootFailureCode.MAXITER_EXCEEDED` (when iterations reach `maxiter` without meeting tolerances) from `BatchRootFailureCode.RESIDUAL_TOLERANCE`.
8. **Transform-time consistency**: Outside `torch.func` transforms, `evaluate_converged_root_batch` re-verifies that the live parameters and positions still satisfy the root residual tolerance at `lane.root_x`, preventing silent attachment of a stale root to perturbed geometries or parameters.

---

## 4. Failure Policy and Status Contract

`BatchRootFailureCode` defines ten explicit failure categories:

| Code | Name | Description |
|---:|---|---|
| `0` | `NONE` | Validated physical root and converged linear response |
| `1` | `INVALID_TEMPERATURE` | Negative or nonfinite Fermi electronic temperature |
| `2` | `TOO_MANY_ELECTRONS` | Target electron count exceeds orbital capacity |
| `3` | `FERMI_NONCONVERGENCE` | Scalar or matrix Fermi chemical-potential Newton solver did not converge |
| `4` | `ELECTRON_COUNT` | Converged density violates electron count conservation ($> 10^{-6}$) |
| `5` | `MAXITER_EXCEEDED` | Primal SCF solver exhausted `maxiter` without meeting `x_atol` / `x_atol_max` |
| `6` | `RESIDUAL_TOLERANCE` | Independent raw-map residual check failed `x_atol` or `x_atol_max` |
| `7` | `NONFINITE_STATE` | Nonfinite values in `root_x`, `mapped_x`, or `spectral.density` |
| `8` | `INVALID_PADDING` | Nonzero values in inactive padded atom/shell/orbital slots |
| `9` | `RESPONSE_FAILURE` | Implicit response failure category during differentiation; surfaced as lane-local nonfinite (`NaN`) derivatives when selected and `0.0` when masked, never written retroactively into immutable primal `Result.status` |

- **Strictness precedence**:
  - `config.scf.force_convergence=True` is mandatory and cannot be weakened by `strict=False`.
  - Otherwise only `strict=True` requests a batch-level primal `SCFConvergenceError`; `strict=False` and `strict=None` use lane-local primal failure results.
- **Non-strict invalid lanes**:
  - Floating primal observables (`energy`, `scf`, `fenergy`, `charges`, `potential`, `density`, `emo`, `occ`) are NaN for invalid lanes; valid lanes remain independent.
  - Selecting an invalid lane for differentiation produces invalid (NaN) derivatives. A zero-weight/masked invalid lane contributes zero to a masked valid-lane objective.
  - `Result.status` reports primal facts known at construction (`converged`, `iterations`, `residual`, `trajectory_failure`, and `final_failure`). Implicit-response failures occur after primal evaluation during backward or forward AD and produce lane-local nonfinite (`NaN`) derivatives when selected (or `0.0` when masked) without mutating the already returned immutable `Result.status`.

---

## 5. Spectral/Map Optimizations and Rejected Ideas

### 5.1 Implemented Optimizations (`src/dxtb/_src/scf/functional/spectral.py`)

1. **Guarding `_gap_electron_response` with `_tracks_derivatives(nel) or _tracks_derivatives(count_delta)` (`spectral.py:390-427, 875-880, 905-911, 932-939`)**:
   - Previously, `fermi_density_matrix`, `fermi_free_energy_matrix`, and `fermi_free_energy_trace` unconditionally executed `_gap_electron_response`, which invokes a second complete matrix-Fermi solve (`_fermi_occupations_matrix` -> 29 dense `255 x 255` linear solves and matrix squarings) to attach the custom derivative $\partial P / \partial N_{\mathrm{el}}$ via `count_delta = nel - nel.detach()`.
   - Because `count_delta` is identically zero in value and only carries a derivative when `nel` (the target electron count, e.g., when differentiating with respect to `refocc` or `charge`) tracks forward- or reverse-mode AD, `_gap_electron_response` was performing a full redundant matrix-Fermi solve on every primal SCF step and every position/Hamiltonian/overlap derivative evaluation.
   - Adding `_tracks_derivatives(nel) or _tracks_derivatives(count_delta)` (checking `requires_grad`, functorch `Grad` and `Jvp` interpreter levels, and `forward_ad.unpack_dual(x).tangent is not None`) skips `_gap_electron_response` whenever `nel` does not track derivatives, while preserving exact first-, second-, and third-order `refocc` / `n0` derivatives and nested outer-`Jvp` `nel` tangents when `nel` is differentiated.

2. **Unifying `fermi_density_matrix` and `_fermi_free_energy_setup` & Eliminating `density @ density` (`spectral.py:725-755, 839-950`)**:
   - Replaced `(density - density @ density).diagonal(dim1=-2, dim2=-1).sum(-1)` in the chemical-potential Newton refinement loop with `count - (density * density).sum(dim=(-2, -1))` (valid because `density` is explicitly symmetrized via `_symmetrize` on the preceding line), eliminating two `O(N^3)` matrix multiplications per matrix-Fermi call.
   - Unified `fermi_density_matrix` to delegate to `_fermi_free_energy_setup` while keeping `spectral.py` 100% pure functional and stateless (no mutable tensor attributes or `is` identity checks).

### 5.2 Historical Kernel A/B (Provenance Correction)

The historical `kernel_ab.json` values below are preserved, but their baseline
label is not accepted as a complete pre-W2c9 source comparison. That run forced
`_tracks_derivatives` to behave as though the electron-count response were
always required; this isolates response-work overhead but does not reproduce
the complete spectral implementation at the pre-W2c9 revision. The W2c10
four-way fresh-process experiment in
[`E6c1-W2c10-batch-hardening-and-streaming.md`](E6c1-W2c10-batch-hardening-and-streaming.md)
is the authoritative attribution. These legacy numbers must not be used to
claim that the trace-squared slope caused the complete reported speedup.

The old benchmark recorded these values on `tmpda` conformer 0 (`69` atoms,
`255` orbitals, `float64`, `OMP_NUM_THREADS=1`, Python 3.11.15 / PyTorch
2.8.0+cpu) for `scf_map(root_x, inputs) + free_energy_from_spectral(...)`:

| Metric | Pre-W2c9 Baseline | W2c9 Optimized | Improvement | Speedup |
|---|---:|---:|---:|---:|
| `no_grad` median (`s`) | `2.5875` | `2.0982` | **-18.91%** | **1.23x** |
| Forward-grad median (`s`) | `2.6134` | `2.3090` | **-11.65%** | **1.13x** |
| Backward-grad median (`s`) | `5.0257` | `2.7782` | **-44.72%** | **1.81x** |
| Max abs gradient diff vs forced-response run | — | `0.0` | Bitwise parity | — |

This is a controlled forced-response comparison, not the complete pre-W2c9
baseline A/B requested for source-level attribution.

### 5.3 Evaluated and Rejected Ideas

- **Replacing `_SCFMapTensorTree` per call inside `_refresh_prepared_batch_inputs`**:
  - *Diagnosis*: Rebuilding `_SCFMapTensorTree` inside an active `jacfwd`/`jvp` transform captures `GradTrackingTensor` masks inside `adapter.template`. When `_BatchedNumericalSolve.vmap` subsequently unwraps one functorch level, those captured masks retain a higher functorch level than `current_level`, triggering a C++ assertion (`ADInterpreters.cpp:46`).
  - *Resolution*: Rejected rebuilding `adapter` during transform-time input refresh; preserved the level-0 `lane.adapter` template and updated only `parameters`.
- **Lowering the static 29-step tanh-doubling loop bound globally**:
  - *Diagnosis*: Forbidden by the `S0.1b` range contract (which requires power-30 / 29 doublings for wide-spectrum/low-temperature matrices) and `vmap` compatibility (which forbids data-dependent Python branching on tensor scaling powers).

---

## 6. Acceptance Test Matrix and Results

The acceptance test suite [`test/test_scf/test_e6c1_w2c9_batch_implicit.py`](../../test/test_scf/test_e6c1_w2c9_batch_implicit.py) covers 14 comprehensive test functions (22 parameterized test items) across all required Workstream B categories and independent code review checks:

| Test Function | Coverage | Status |
|---|---|---|
| `test_conformer_batch_matches_serial_and_permutes_cleanly` | GFN1 & GFN2 `H2O` conformers across Charge, Potential, and Fock SCP and Linear & Anderson mixers; energy/free-energy/force parity, zero cross-lane blocks, and permutation invariance | PASS |
| `test_open_shell_finite_temperature_partially_occupied_no2` | Finite-temperature (`1000 K`) open-shell `NO2` across equal and atomic partitions with fractional occupations and nontrivial electronic entropy vs finite differences | PASS |
| `test_exact_and_near_spectral_degeneracy_h3_plus` | Equilateral $D_{3h}$ `H3+` (exact degeneracy) and $10^{-7}$ Bohr distorted `H3+` (near degeneracy) batch implicit derivatives | PASS |
| `test_noncontiguous_padding_preserves_values_and_gradients` | Noncontiguous canonical padding (`LiH` padded into `(LiH, H2O)` via `PaddedExactSystemCore`) with coincident and displaced inactive coordinates | PASS |
| `test_shared_model_parameter_gradients_and_leaf_classification` | Shared `Model` parameter gradients vs summed serial gradients; connected non-zero, connected zero, and disconnected leaves | PASS |
| `test_failure_policy_strict_and_non_strict_mixed_lanes` | `maxiter=0` rejection, `maxiter=1` strict error vs non-strict warning, mixed valid/invalid lane `NaN` isolation, and stale-root geometry rejection | PASS |
| `test_fermi_and_linear_response_failure_isolation` | Unphysical electron count (`TOO_MANY_ELECTRONS`) and singular linear-response (`implicit_root` under `vmap`) `NaN` isolation | PASS |
| `test_transforms_grad_jvp_jacfwd_jacrev_hvp_vmap_and_higher_order` | `grad`, `jvp`, `jacfwd`, `jacrev`, HVP, nested `vmap`, 2nd/3rd-order scalar directional derivatives vs finite differences, and `torch.compile` | PASS |
| `test_open_shell_and_charged_conformer_batches` | Open-shell UHF (`Li`, `NH2` radical) and charged (`OH-`, `H3O+`) conformer batches; prepared `chrg`/`spin` preservation and mismatch rejection | PASS |
| `test_electric_field_and_field_grad_perturbations` | External `field` and `field_grad` perturbations on `H2O` conformers; prepared perturbation preservation and finite-difference gradient parity | PASS |
| `test_multi_orbital_noncontiguous_padding_and_potential_field` | Multi-orbital noncontiguous `HF` in `(HF, H2O)` canonical bucket at `300 K` and `5000 K` across Charge SCP and Potential SCP with Anderson mixing + electric field | PASS |
| `test_nested_jvp_and_outer_nel_tangent` | Nested `jvp(jvp(...))` through `_unwrap_primal_tensor` and outer-`jvp` `nel` tangent propagation inside inner Hamiltonian `jvp` in `spectral.py` | PASS |
| `test_conformer_batch_size_sweep_1_2_4` | Conformer batch size sweep ($N = 1, 2, 4$) verifying exact per-lane energy and gradient parity | PASS |
| `test_public_singlepoint_contract_preserved` | Verifies public `System.singlepoint()` finite-trip contract is untouched and no premature batch method is exposed on `System` | PASS |

### Multi-Environment Focused Gate & Primary Regression Suites

1. **Five-Environment Focused SCF/Implicit/Spectral Gate**:
   Executed across all five supported `tad-mctc 0.11.3` CPU environments (`py3.10-torch2.7.0`, `py3.11-torch2.8.0`, `py3.12-torch2.11.0`, `py3.12-torch2.13.0`, `py3.14-torch2.14.1`) over 8 SCF/implicit/spectral test modules:
   - `test/test_scf/test_e6c1_w2c9_batch_implicit.py`
   - `test/test_scf/test_invariant_spectral_density.py`
   - `test/test_scf/test_invariant_free_energy.py`
   - `test/test_scf/test_e6c1_w2c5b_shift_contract.py`
   - `test/test_scf/test_e6c1_w2c6_tanh_solve.py`
   - `test/test_scf/test_functional_implicit_root.py`
   - `test/test_scf/test_e6c1_w2c1_mapped_implicit.py`
   - `test/test_scf/test_functional_root_derivatives.py`

   Result: **216/216 passed** (full selection including `@pytest.mark.slow` third-order `NO2` reference tests on `py3.10-torch2.7.0`, `py3.11-torch2.8.0`, `py3.12-torch2.11.0`, `py3.14-torch2.14.1`; **212/212 passed** non-slow selection on `py3.12-torch2.13.0` and `py3.14-torch2.14.1`).

2. **E7a, E6bcompat, E6a, and E6bimpl Primary Regression Gates**:
   Executed on both `py3.11-torch2.8.0` and `py3.14-torch2.14.1` across 15 calculator/padding/transform/compatibility suites (`test_e7a_padding_safety.py`, `test_e6bcompat_differential.py`, `test_e6bcompat_policy.py`, `test_e6a0_system_vmap.py`, `test_e6a1_conformer.py`, `test_e6a1a_gfn0_spin_validation.py`, `test_e6a2_public_status.py`, `test_e6a3_component_vmap.py`, `test_e6a4_transform_acceptance.py`, `test_e6bimpl0_padded_integrals.py`, `test_e6bimpl1a_padded_spectral.py`, `test_e6bimpl2a_gfn2_canonical.py`, `test_e6bimpl2b_gfn2_scf.py`, `test_e6bimpl3_legacy_composition.py`, `test_e6c1_w2c2a_configuration.py`):
   - `py3.11-torch2.8.0`: **261 passed** (`0 failed`).
   - `py3.14-torch2.14.1`: **261 passed** (`0 failed`).

---

## 7. Staged `tmpda` Benchmark Progression

All benchmarks were executed in fresh, isolated single-threaded processes (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`) on `AMD EPYC 7763 64-Core Processor` (`m199`) using Python `3.11.15` and PyTorch `2.8.0+cpu` on frozen `tmpda` conformers (`69` atoms, `255` orbitals, `float64`, GFN1, charge SCP, Anderson mixing, `T = 300 K`, `maxiter = 300`, `x_atol = 1e-8`, `x_atol_max = 1e-5`).

### 7.1 Phase Timing and Memory Summary (`benchmarks/results/e6c1-w2c9/`)

| Run Artifact | $N$ | Order | Mode | Setup (`s`) | Root Prep (`s`) | Attach+Finalize (`s`) | Backward (`s`) | Linearize (`s` / calls) | Total (`s`) | Peak RSS (`MiB` / `GiB`) |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `tmpda_n1_pos.json` | 1 | `[0]` | `positions-only` | `0.22` | `70.88` | `1.68` | `24.75` | `18.71` (13) | **97.53** | **1422.21** (`1.39 GiB`) |
| `tmpda_n1_model.json` | 1 | `[0]` | `positions-and-model` | `0.22` | `71.21` | `2.44` | `28.54` | `18.74` (13) | **102.42** | **1667.18** (`1.63 GiB`) |
| `tmpda_n2_pos.json` | 2 | `[0, 1]` | `positions-only` | `0.20` | `176.51` | `2.76` | `59.73` | `48.29` (17) | **239.21** | **2366.15** (`2.31 GiB`) |
| `tmpda_n2_pos_rev.json` | 2 | `[1, 0]` | `positions-only` | `0.28` | `178.31` | `2.76` | `60.09` | `48.34` (17) | **241.46** | **2399.49** (`2.34 GiB`) |
| `tmpda_n2_model.json` | 2 | `[0, 1]` | `positions-and-model` | `0.24` | `177.22` | `4.11` | `70.34` | `49.41` (17) | **251.91** | **2929.82** (`2.86 GiB`) |
| `tmpda_n4_pos.json` | 4 | `[0, 1, 2, 3]` | `positions-only` | `0.21` | `374.14` | `5.35` | `119.52` | `96.53` (17) | **499.24** | **5732.04** (`5.60 GiB`) |

### 7.2 Comparison Against W2c7 and W2c8 Baselines

The following is a historical architecture comparison, not a controlled
same-process source A/B. The runs have different routes and instrumentation;
in particular, a mapped-only run must not be compared to a mapped run that
also includes serial parity work as though the difference were an optimization
speedup. Treat these values as the recorded end-to-end observations only. The
controlled spectral attribution is the separate W2c10 four-way experiment.

| Benchmark Configuration | W2c7 (`7d71e93`) | W2c9 (Current) | Wall-Time Reduction | Peak RSS Reduction |
|---|---:|---:|---:|---:|
| 1-conformer `positions-only` | `143.94 s` / `2.51 GiB` | **`97.53 s` / `1.39 GiB`** | **-32.24% (1.48x)** | **-44.61%** |
| 2-conformer `positions-only` | `429.36 s` / `3.62 GiB` | **`239.21 s` / `2.31 GiB`** | **-44.29% (1.80x)** | **-36.17%** |
| 2-conformer `positions-and-model` | `470.02 s` / `4.31 GiB` | **`251.91 s` / `2.86 GiB`** | **-46.40% (1.87x)** | **-33.69%** |

Compared with the public fixed-trip unrolled `System.singlepoint()` route profiled in `W2c8` (where a single conformer's peak RSS grew from `3.33 GiB` at 1 trip to `4.25 GiB` at 2 trips and `6.07 GiB` at 4 trips), the W2c9 converged-root route completes the full **66-iteration converged root + implicit gradient** for 1 conformer in **1.39 GiB** (`58%` less memory than a *single* unrolled trip) and for 2 conformers (66 and 96 iterations) in **2.31 GiB**.

### 7.3 Per-Lane Convergence and Numerical Parity Across Batch Sizes

| Conformer Index | Primal SCF Iterations | Raw Residual $\|r\|_2$ | Raw Residual $\|r\|_\infty$ | BiCGSTAB Response Iterations | Total Energy (`Eh`) | Free Energy (`Eh`) | Position Gradient Norm (`Eh/Bohr`) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0` (`N=1, 2, 4`) | `66` | `6.0120e-09` | `4.3560e-09` | `5` | `-29.758437462176317` | `-0.001391311654408853` | `41.29473633572573` |
| `1` (`N=2, 4`) | `96` | `6.4161e-09` | `2.7394e-09` | `7` | `-11.312642703835380` | `-0.001886992801543350` | `66.29496021458891` |
| `2` (`N=4`) | `112` | `8.9675e-09` | `4.9969e-09` | `6` | `-21.692981421967666` | `-0.002603711695789798` | `56.45315879169315` |
| `3` (`N=4`) | `72` | `6.2742e-09` | `3.4172e-09` | `7` | `-33.762321030729446` | `-0.002096552316983033` | `43.28854912977592` |

From `tmpda_n2_model_parity.json` (`--verify-parity` on 2 conformers with all 2,261 `Model` parameters enabled):
- Maximum energy discrepancy vs. serial `System`: `2.5402e-13 Eh`
- Maximum free-energy discrepancy vs. serial `System`: `3.3180e-13 Eh`
- Maximum position-gradient discrepancy vs. serial `System`: `6.3505e-14 Eh/Bohr`
- Maximum cross-lane position-gradient block $\max_{i \neq j} \|\partial E_i / \partial R_j\|_\infty$: `0.0`
- Maximum shared `Model` parameter-gradient discrepancy vs. summed serial gradients across 2,261 leaves: `1.5632e-13` (`73` connected non-zero, `4` connected zero, `2,184` disconnected; metadata match `True`).

---

## 8. Public API Options and Recommendation

As established in `W2c8`, finite-trip trajectory differentiation (`System.singlepoint()`) and converged-root implicit differentiation represent distinct mathematical contracts:
- `System.singlepoint()` differentiates the fixed-trip solver map $x^{(K)}(\theta)$ and is transform-pure under literal `torch.func.vmap(System.singlepoint)`.
- [`evaluate_converged_root_batch`](../../src/dxtb/_src/calculators/batch_implicit.py) prepares detached primal roots $x_b^\star$ with adaptive per-lane stopping and differentiates the physical fixed-point equation $G(x_b^\star, \theta) - x_b^\star = 0$ via the implicit function theorem.

### Options Evaluated

1. **Option 1 (Recommended): Explicit method `System.singlepoint_batch(positions, ...)` backed by `prepare_converged_root_batch` + `evaluate_converged_root_batch`**
   - *Pros*: Keeps `System.singlepoint()` unambiguous as the single-System finite-trip entrypoint; makes the conformer-batch contract and `strict` failure policy explicit; naturally accepts `(B, nat, 3)` positions and shared `Model` parameters without `batch_mode` inference.
   - *Cons*: Introduces one additional method on `System` once approved by the project owner.
2. **Option 2: Separate functional module entrypoint `dxtb.scf.singlepoint_converged_batch(system, positions, ...)`**
   - *Pros*: Zero surface expansion on the `System` class.
   - *Cons*: Less discoverable for users expecting singlepoint operations on `System`.
3. **Option 3 (Rejected): Overloading `System.singlepoint()` with a `mode="converged_root"` or shape-based dispatch**
   - *Reason for rejection*: Violates Invariant 2.2 (no shape-based batch guessing in the core) and mixes two incompatible derivative contracts behind one method signature.

Pending project-owner approval of Option 1, the implementation remains isolated in the internal module [`src/dxtb/_src/calculators/batch_implicit.py`](../../src/dxtb/_src/calculators/batch_implicit.py).

---

## 9. Independent Code Review and Resolution

Requested reviewer: `Luna / high`
Actual model identity: `not exposed by runtime`

The independent read-only review inspected the full `E6c1-W2c9` diff and surrounding SCF/calculator modules and reported findings that were resolved as follows:
- **Multi-orbital noncontiguous padded electron-count check (`batch_implicit.py`)**: Replaced AO-mask indexing of the energy-sorted MO `occupation` vector with contiguous spectral-mask indexing (`0 .. active_count - 1`) combined with an explicit AO-basis Mulliken population check `((n0 - charges.mono)[active_orb].sum() - nel.sum()).abs().amax()`. Verified by `test_multi_orbital_noncontiguous_padding_and_potential_field` on `HF` in `(HF, H2O)` at `300 K` and `5000 K`.
- **Preservation of prepared `chrg`, `spin`, `field`, and `field_grad` (`batch_implicit.py`)**: Added `_UNSET` sentinel defaults on `evaluate_converged_root_batch` so refreshing `positions` on a `PreparedConvergedRootBatch` preserves each lane's prepared `charge`, `spin`, `field`, and `field_grad` unless explicitly overridden, and verifies `nel` compatibility against `lane.nel`. Verified by `test_open_shell_and_charged_conformer_batches` and `test_electric_field_and_field_grad_perturbations`.
- **Nested `jvp(jvp(...))` unwrapping (`batch_implicit.py`)**: Simplified `_unwrap_primal_tensor` to peel all functorch wrapper layers via `is_functorch_wrapped_tensor` / `get_unwrapped` before unpacking any `forward_ad` dual tensor under `_DisableFuncTorch()`. Verified by `test_nested_jvp_and_outer_nel_tangent`.
- **Outer `Jvp` `nel` tangent detection and pure functional `spectral.py`**: Extended `_tracks_derivatives` to detect outer `Jvp` interpreter levels (`level != top_jvp_level`), checked `_tracks_derivatives(nel) or _tracks_derivatives(count_delta)`, and removed the temporary `_local_free_energy_setup` attribute on `_SpectralDecomposition` so `spectral.py` remains 100% stateless. Verified by `test_nested_jvp_and_outer_nel_tangent`.
- **Padded `SCP_MODE_POTENTIAL` with electric field (`batch_implicit.py`)**: Applied `state_projection` to `potential_map(...).mapped_x` for padded lanes in `SCP_MODE_POTENTIAL` during both eager root solving and implicit attachment. Verified by `test_multi_orbital_noncontiguous_padding_and_potential_field`.
- **Detached adapter template and dtype-resolved tolerances (`batch_implicit.py`)**: Prepared `adapter` from detached `primal_inputs` so `adapter.template` never retains caller autograd graphs, and validated independent raw-map residuals against dtype-resolved `settings.x_atol` and `settings.x_atol_max`.

---

## 10. Remaining Bottlenecks and Next Engineering Steps

1. **Why 8 Conformers Still Exceeds the W2 Memory Budget (`5.60 GiB` at $N=4$)**:
   - Primal root preparation uses minimal memory (`468 MiB` at $N=1$, `584 MiB` at $N=2$, `767 MiB` at $N=4$ — roughly `100 MiB` per lane).
   - The memory jump occurs during `evaluate_converged_root_batch` + backward (`1.39 GiB` at $N=1$, `2.31 GiB` at $N=2$, `5.60 GiB` at $N=4$), because `vmap(_evaluate_valid_lane)` simultaneously materializes the forward autograd graph and BiCGSTAB `linearize` JVP state for all $B$ lanes at once across 29 matrix-Fermi tanh-doubling steps (`255 x 255` dense matrices per step).
   - **Next Step (Micro-batched / Chunked Lane Evaluation)**: Because conformer lanes in `evaluate_converged_root_batch` are mathematically independent (zero cross-lane blocks) and only accumulate into shared `Model` parameters during backward, adding an optional `chunk_size: int | None` to `evaluate_converged_root_batch` (or custom chunked VJP accumulation) will bound peak RSS at the 2-lane level (`~2.4 GiB`) regardless of whether $N = 4, 8,$ or $64$ conformers are evaluated.

2. **Remaining Wall-Clock Bottleneck (`root_prep_s` and `linearize`)**:
   - Even after the `scf_map` and backward speedups delivered in `W2c9`, each `scf_map` evaluation on 255 orbitals still executes $2 \times 2 \times 29 = 116$ dense `255 x 255` linear solves (`torch.linalg.solve`) inside `_fermi_occupations_matrix` (two Newton steps $\times$ two spin channels $\times$ 29 static tanh-doubling iterations), of which roughly half (`p > power`, where `power = 16` for `tmpda`) are masked out after execution.
   - **Next Step**: In `prepare_converged_root_batch`, primal root solving runs eagerly *outside* `vmap` on one lane at a time (`_unwrap_primal_tensor` + `torch.no_grad()`). In that eager single-lane primal context, data-dependent loop termination at `int(power.max().item())` (16 doublings instead of 29) is completely safe and does not affect `vmap` or AD transforms, which would reduce `root_prep_s` by an additional **~40%**.
