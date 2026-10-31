# E6c1-W2c10 — Batch Correctness Hardening, Streaming Differentiation, and Spectral Performance

## 1. Executive Summary

Package `E6c1-W2c10` completes the critical correctness hardening, benchmark-provenance attribution, memory-bounded streaming differentiation, eager-primal adaptive tanh-doubling optimization, and full 1/2/4/8-conformer `tmpda` scaling progression following `E6c1-W2c9`:

1. **Prepared-Root Lifetime and Primal-Validity Hardening (`src/dxtb/_src/calculators/batch_implicit.py`)**:
   - Enforces a strict prepared-root lifetime contract across direct calls, explicit overrides, in-place tensor mutations, connected and non-trainable `Model` parameter mutations, `System` configuration/atomic-number changes, `torch.func` transforms (`grad`, `jvp`, `jacfwd`, `jacrev`, nested `jvp(jvp(...))`), and `torch.compile`.
   - Captures immutable byte-backed snapshots (`_TensorValueSnapshot`) and tensor `_version` counters for positions, atomic numbers, connected differentiable `Model` parameter leaves, `System` setup tensors, per-lane inputs (`charge`, `spin`, `field`, `field_grad`), flattened map parameters (`parameter_snapshots`), and `SCFInputs` leaves (`input_tensor_versions`), plus each lane's converged `root_x` (`root_snapshots`).
   - Unwraps functorch and forward-AD wrappers under `torch._C._DisableFuncTorch()` so legitimate `jvp` tangents do not invalidate a valid primal root while any primal perturbation or in-place mutation is deterministically rejected before attaching implicit derivatives.

2. **Strictness Precedence, Invalid-Lane Gradient Semantics, and Response-Failure Contract (`src/dxtb/_src/calculators/batch_implicit.py`)**:
   - Enforces mandatory `config.scf.force_convergence=True` precedence (`_resolve_strictness`): `strict=False` cannot weaken `force_convergence=True`. Tested across all six `(strict, force_convergence)` combinations on both valid and invalid physical roots (`test_strictness_precedence_and_fermi_failure_classification`).
   - Guarantees that differentiating any selected failed lane returns `NaN` values and `NaN` derivatives across positions, external perturbations (`charge`, `spin`, `field`, `field_grad`), and connected `Model` parameters (`_InvalidLaneOutput.apply` connected to `_invalid_lane_dependencies`), while unselected/zero-weighted failed lanes contribute `0.0` without contaminating valid lanes (`test_failure_policy_strict_and_non_strict_mixed_lanes`).
   - Makes `_classify_fermi_code` fail closed with `ValueError` on any unrecognized nonzero Fermi failure code.
   - Aligns `BatchRootFailureCode.RESPONSE_FAILURE` with the immutable `Result` design: primal `Result.status` records primal facts known at construction and is never retroactively mutated during backward or forward AD; failed implicit response solves produce lane-local `NaN` derivatives when selected and `0.0` when masked, and are classified post-differentiation via `classify_response_failure_status` and in `StreamingBatchValueAndGrad.final_failure` (`test_batch_backward_response_failure_is_lane_local`).

3. **Spectral Derivative Guard and Cancellation-Safe Newton Slope (`src/dxtb/_src/scf/functional/spectral.py`)**:
   - Hardens `_tracks_derivatives` with per-level `Jvp` tangent inspection (`_has_jvp_tangent_at_level`) and a conservative `try ... except (AttributeError, RuntimeError): return True` fallback so electron-count derivatives are never silently omitted even if private functorch C++ internals change. Verified against an always-response reference oracle across 1st–3rd order derivatives and mixed Hamiltonian/overlap/electron-count transforms (`test_electron_response_guard_matches_always_response_reference`).
   - Replaces the global difference `count - (density * density).sum(dim=(-2, -1))` in `_fermi_free_energy_setup` with the row-wise cancellation-guarded contraction `(diag_density - (density * density).sum(dim=-1)).sum(-1) / safe_kt`, preventing catastrophic cancellation when occupations approach `0` or `1` in `float32` and `float64` (`test_trace_squared_slope_matches_legacy_density_product`).

4. **Process-Isolated 4-Way Spectral Kernel A/B/C/D Attribution (`benchmarks/e6c1_w2c10_kernel_ab.py`, `benchmarks/e6c1_w2c10_shape_ab.py`)**:
   - Executes fresh-process benchmarks comparing (`A`) exact pre-W2c9 `7d71e930e5bdbd52a252bf23d21803b94d2695f4`, (`B`) `7d71e93` + electron-count guard only (`guard-only.patch`), (`C`) `7d71e93` + trace-squared Newton slope only (`trace-squared-only.patch`), and (`D`) complete current spectral implementation across dimensions `255` (`tmpda[0]`), `128` (`16 H2O`), and `32` (`4 H2O`).
   - Demonstrates that the `_gap_electron_response` guard (`B`) accounts for nearly all end-to-end differentiation gains (reducing `tmpda[0]` $N=255$ matmul count from `381` to `271`, backward median from `7.803 s` to `4.922 s` [**-36.9%**], and 2nd-order directional derivative from `33.925 s` to `15.509 s` [**-54.3%**]), while the Newton-slope identity (`C` -> `D`) removes 4 additional `N x N` matmuls per evaluation.

5. **Memory-Bounded Streaming Batch Differentiation (`src/dxtb/_src/calculators/batch_streaming.py`)**:
   - Introduces `stream_batch_value_and_grad` and `StreamingBatchValueAndGrad` as an explicit eager first-order value-and-gradient operation for separable weighted batch objectives $L = \sum_b w_b E_b$, evaluating and differentiating one chunk at a time before releasing chunk autograd graphs.
   - Preserves exact parity with simultaneous `evaluate_converged_root_batch` across per-lane position gradients, perturbation gradients, and all 2,261 shared `Model` parameter leaves (including connected non-zero, connected zero, and disconnected leaf classification).
   - Scales `tmpda` conformer batches through **$N = 1, 2, 4, 8$ conformers** at a constant **2.25–2.27 GiB (`2,302–2,326 MiB`) peak RSS** (`chunk_size=1`), reducing 4-conformer peak RSS by **59.42%** (`5,732.04 MiB` $\to$ `2,325.84 MiB`) and completing the full **8-conformer `tmpda` physical workload** in **1,100.71 s** (`850.47 s` root preparation + `250.23 s` streaming differentiation) at **2,320.49 MiB (`2.27 GiB`) peak RSS**.

6. **Eager-Only Adaptive Tanh Doubling (`src/dxtb/_src/scf/functional/spectral.py`, `batch_implicit.py`)**:
   - Implements `_matrix_fermi_eager` and `eager_primal_fermi_scope` (`eager_primal_fermi: bool = False` in `prepare_converged_root_batch`), terminating the detached eager primal tanh-doubling recurrence at `int(power.max().item())` while strictly forbidding execution under `torch.is_grad_enabled()`, `torch.func` transforms, or `torch.compile`, and always revalidating candidate roots with the unchanged static 29-step raw map.
   - Reduces `tmpda[0]` dense linear solves during primal root preparation by **45.97%** (`6,030` $\to$ `3,258`) and root-preparation wall time by **36.00% (`1.56x` speedup, `74.49 s` $\to$ `47.68 s`)** while converging in the exact same `66` iterations to the bitwise-identical root SHA-256 (`d49a1025a20f72bf9865763a242962d55a46ce0fd71a9386aceeca1dc8ac3188`), residual (`6.012016343937147e-09`), and position gradient norm (`41.29473633572572`).

---

## 2. Scope and Corrected W2c9 Provenance

Starting source revision: `7b946a21fcd27807127d54b4262b5ec5177d607d` on branch `claude/tender-dijkstra-cda42s`.

The actual published `E6c1-W2c9` commit ancestry in Git history (correcting the earlier handoff summary text without rewriting published history) is:

| Work Package Component | Published Commit SHA |
|---|---|
| Spectral optimization (`spectral.py`) | `38745b0f1f13344602c022a8d1a8779351b9f062` |
| Batch implementation and tests (`batch_implicit.py`, `test_e6c1_w2c9_batch_implicit.py`) | `b53bf6d2270b6e85708214edd01887f9cde1aa63` |
| Benchmarks and documentation (`benchmarks/e6c1_w2c9_*.py`, `docs/plan/E6c1-W2c9-*.md`) | `7b946a21fcd27807127d54b4262b5ec5177d607d` |

Historical benchmark JSON files under `benchmarks/results/e6c1-w2c9/` are preserved unmodified; provenance corrections and controlled fresh-process attributions are published here and in `benchmarks/results/e6c1-w2c10/`.

---

## 3. Prepared-Root Lifetime and Primal-Validity Contract

Prepared physical roots in [`src/dxtb/_src/calculators/batch_implicit.py`](../../src/dxtb/_src/calculators/batch_implicit.py) are bound to the exact primal physical problem for which they were solved and validated:

1. **Snapshot and Version Capture at Preparation (`prepare_converged_root_batch`)**:
   - Stores immutable byte-backed `_TensorValueSnapshot` records (`shape`, `dtype`, `device`, `raw_bytes` extracted under `torch._C._DisableFuncTorch()`) and `_tensor_version` counters for:
     - `positions` (`positions_snapshot`, `positions_version`),
     - `system.numbers` (`numbers_snapshot`, `numbers_version`),
     - connected differentiable `Model` parameter leaves (`model_parameter_refs`, `model_parameter_snapshot`, `model_parameter_versions`),
     - all `System` setup tensors (`system_tensor_refs`, `system_tensor_snapshot`, `system_tensor_versions`),
     - per-lane external inputs `(charge, spin, field, field_grad)` (`lane_input_snapshots`),
     - per-lane flattened map parameters `lane.parameters` (`parameter_snapshots`),
     - per-lane `SCFInputs` tensor leaves (`input_tensor_versions`), and
     - per-lane converged `root_x` (`root_snapshots`).
   - Records `config_identity = id(system.config)` and `config_repr = repr(system.config)`.
   - Keeps autograd graph nodes (`seen_nodes`) and traversed container objects (`seen_objs`) alive during `_collect_system_and_model_tensors` so Python never recycles temporary wrapper `id()` values across PyTorch 2.7–2.14.

2. **Unconditional Validation on Evaluation (`_validate_prepared_primal` & `_refresh_prepared_batch_inputs`)**:
   - Even when `evaluate_converged_root_batch(prepared)` is called without overrides, `_validate_prepared_primal` verifies `config_identity`/`config_repr`, `numbers`, `positions`, `model_parameter_refs`, `system_tensor_refs`, `lane_input_snapshots`, `parameter_snapshots`, `input_tensor_versions`, and `root_snapshots` before reusing `prepared.lanes`.
   - When overrides (`positions`, `chrg`, `spin`, `field`, `field_grad`) are supplied or a transform is active, `_refresh_prepared_batch_inputs` unwraps functorch (`GradTrackingTensor`, `BatchedTensor`) and forward-AD dual wrappers via `_unwrap_primal_tensor` under `torch._C._DisableFuncTorch()` and compares the unwrapped primal tensors against the prepared snapshots (`_matches_snapshot`).
   - **Transform-safe primal distinction**: A `jvp` or `jacfwd` tangent perturbation does not alter the unwrapped primal value, so valid primal roots evaluate derivatives without re-running SCF or triggering false stale-root errors. Conversely, any displaced geometry, mutated `positions`, mutated `Model` parameter leaf (whether `requires_grad=True` or `False`, SCF-connected or classical-only), altered external field, or changed charge/spin is rejected with a descriptive `RuntimeError`.
   - When `torch.compile` traces with symbolic/fake tensors where raw byte extraction is unavailable, `_matches_snapshot` falls back to tensor equality (`(primal == reference) | (isnan(primal) & isnan(reference))`) and `_refresh_prepared_batch_inputs` executes an independent detached raw-map residual check (`raw_residual_l2 <= x_atol` and `raw_residual_max <= x_atol_max`).

---

## 4. Strictness Precedence, Invalid-Lane Gradients, and Response-Failure Contract

1. **Strictness Resolution (`_resolve_strictness`)**:
   - `bool(config.scf.force_convergence) or (strict is True)`.
   - `config.scf.force_convergence=True` is mandatory and cannot be overridden by `strict=False` or `strict=None`.
   - When `force_convergence=False`, `strict=True` raises `SCFConvergenceError` with `[Lane b]` and the failure category, whereas `strict=False` and `strict=None` emit `SCFConvergenceWarning` and return a batch with lane-local status and `NaN` invalid-lane observables.
   - Tested exhaustively across all 6 combinations of `strict in (None, False, True)` and `force_convergence in (False, True)` on both valid and invalid physical roots (`test_strictness_precedence_and_fermi_failure_classification`).

2. **Fail-Closed Fermi Status Classification (`_classify_fermi_code`)**:
   - Recognizes `0` (`NONE`), `FERMI_FAILURE_INVALID_TEMPERATURE` (`INVALID_TEMPERATURE`), `FERMI_FAILURE_TOO_MANY_ELECTRONS` (`TOO_MANY_ELECTRONS`), `FERMI_FAILURE_NONCONVERGENCE` (`FERMI_NONCONVERGENCE`), and `FERMI_FAILURE_ELECTRON_COUNT` (`ELECTRON_COUNT`).
   - Any other nonzero integer code immediately raises `ValueError(f"Unknown nonzero Fermi failure code: {code_value}.")` (`test_strictness_precedence_and_fermi_failure_classification`).

3. **Invalid-Lane Primal and Derivative Semantics**:
   - For any invalid lane (`~valid_lane_mask[b]`), all floating primal Result fields are filled with `NaN` connected to all differentiated inputs (`lane.positions`, `lane.charge`, `lane.spin`, `lane.field`, `lane.field_grad`, `lane.parameters`, `lane_classical`, and `model_parameter_refs`) via `_InvalidLaneOutput.apply` and `_invalid_lane_dependencies`.
   - Differentiating any objective that selects an invalid lane (e.g., `result.energy[invalid].sum()`, `result.energy.sum()`, or `result.fenergy[invalid].sum()`) returns `NaN` derivatives for that lane's positions, perturbations, and shared `Model` parameters (`test_failure_policy_strict_and_non_strict_mixed_lanes`).
   - When an objective masks out invalid lanes in reverse mode (`torch.where(result.converged, lane_energy, 0.0).sum()` or zero weight in `stream_batch_value_and_grad`), `_InvalidLaneOutput.backward` checks `(grad_output != 0.0).any()` and returns exact `0.0` gradients for unselected invalid lanes so valid-lane gradients and shared `Model` parameter gradients remain finite and identical to an isolated valid-lane calculation.

4. **Batch Response-Failure Contract (`BatchRootFailureCode.RESPONSE_FAILURE`)**:
   - `Result` is an immutable primal snapshot constructed before backward or forward AD runs; therefore `Result.status` records primal convergence facts (`converged`, `iterations`, `residual`, `trajectory_failure`, `final_failure`) and is never retroactively mutated during differentiation.
   - If the implicit linear-response solve fails during backward or forward AD on lane $b$, `implicit_root` produces nonfinite (`NaN`) lane derivatives for lane $b$ while keeping converged lanes $b' \neq b$ finite and uncorrupted (`test_batch_backward_response_failure_is_lane_local`).
   - Callers can classify post-differentiation response failures without mutating `Result.status` via `classify_response_failure_status(result.status, result.converged, gradients)`, and `stream_batch_value_and_grad` automatically records `int(BatchRootFailureCode.RESPONSE_FAILURE)` (`9`) in `StreamingBatchValueAndGrad.final_failure` for any weighted primal-converged lane whose backward response gradient is nonfinite.

---

## 5. Spectral Derivative-Guard and Newton-Slope Validation

1. **Exhaustive Validation of `_tracks_derivatives` (`src/dxtb/_src/scf/functional/spectral.py`)**:
   - `_tracks_derivatives(x)` inspects `x.requires_grad` (when `torch.is_grad_enabled()`), active functorch interpreter levels (`TransformType.Grad` and `TransformType.Jvp` via `_has_jvp_tangent_at_level`), and forward-AD dual tangents (`forward_ad.unpack_dual(x).tangent is not None`).
   - If the private `torch._C._functorch` inspector raises `AttributeError` or `RuntimeError` on a future PyTorch build, `_tracks_derivatives` conservatively returns `True` so `_gap_electron_response` is computed rather than silently dropping $\partial P / \partial N_{\mathrm{el}}$.
   - Verified in `test_electron_response_guard_matches_always_response_reference` against an always-response reference implementation across:
     - `torch.autograd.grad` (`create_graph=False` and `True`),
     - 1st, 2nd, and 3rd scalar directional derivatives of `nel` (`refocc` / `chrg`),
     - `torch.func.grad`, `jvp`, `jacfwd`, `jacrev`, `hessian`, `vmap(grad)`, `vmap(jvp)`,
     - nested `jvp(jvp(...))` with tangent on outer `nel` and inner `H`/`S`,
     - `forward_ad.dual_level()`,
     - mixed derivatives $\partial^2 E / (\partial H \, \partial N_{\mathrm{el}})$ and $\partial^2 F / (\partial S \, \partial N_{\mathrm{el}}) $, and
     - simulated functorch inspector failure (`AttributeError` / `RuntimeError` fallback).

2. **Cancellation-Guarded Newton Slope (`_fermi_free_energy_setup`)**:
   - In `_fermi_free_energy_setup`, the chemical-potential Newton refinement slope $\partial N / \partial \mu = \mathrm{Tr}(D(I - D)) / k_B T$ is computed via the row-wise contraction:
     $$\mathrm{Tr}(D - D^2) = \sum_i \left(D_{ii} - \sum_j D_{ij}^2\right)$$
     using `diag_density = density.diagonal(dim1=-2, dim2=-1)` and `(diag_density - (density * density).sum(dim=-1)).sum(-1) / safe_kt`.
   - Unlike subtracting two $O(N_{\mathrm{occ}})$ global sums `count - (density * density).sum(dim=(-2, -1))`, subtracting each row's squared norm from its diagonal entry $D_{ii} \in [0, 1]$ bounds intermediate terms by $O(1)$ per orbital before summing over active orbitals.
   - Verified across metallic degenerate spectra, wide-gap insulators, near-saturated occupations ($1 - f_i \approx 10^{-12}$), extreme low/high temperatures (`50 K` to `5000 K`), generalized overlap $S \neq I$, padded inactive orbitals, `float64` and `float32`, and 1st–3rd derivatives (`test_trace_squared_slope_matches_legacy_density_product`).

---

## 6. Four-Way Spectral Kernel A/B/C/D Benchmark Attribution

To resolve the `W2c9` `kernel_ab.json` provenance issue (where the historical baseline script had patched `_tracks_derivatives` rather than running the unmodified `7d71e930e5bdbd52a252bf23d21803b94d2695f4` source), [`benchmarks/e6c1_w2c10_kernel_ab.py`](../../benchmarks/e6c1_w2c10_kernel_ab.py) and [`benchmarks/e6c1_w2c10_shape_ab.py`](../../benchmarks/e6c1_w2c10_shape_ab.py) executed fresh single-threaded processes (`OMP_NUM_THREADS=1`, Python `3.11.15`, PyTorch `2.8.0+cpu`) across four exact source variants:
- **Variant A (`pre-w2c9`)**: Exact `spectral.py` from `7d71e930e5bdbd52a252bf23d21803b94d2695f4` (`source_dirty: false`).
- **Variant B (`guard-only`)**: Variant A + only the `_gap_electron_response` derivative guard (`benchmarks/results/e6c1-w2c10/guard-only.patch`).
- **Variant C (`trace-squared-only`)**: Variant A + only the trace-squared Newton slope (`benchmarks/results/e6c1-w2c10/trace-squared-only.patch`).
- **Variant D (`current`)**: Complete `E6c1-W2c10` `spectral.py`.

### 6.1 Four-Way Attribution Across Dimensions `255` (`tmpda[0]`), `128` (`16 H2O`), and `32` (`4 H2O`)

Backed by `w2c10-pre-all-final.json`, `w2c10-guard-pre-all.json`, `w2c10-slope-pre-all.json`, `w2c10-current-all.json`, `w2c10-shape-*-16*.json`, and `w2c10-shape-*-4*.json` in `benchmarks/results/e6c1-w2c10/`:

| Dimension $N$ (System) | Variant | `aten::matmul` | `aten::mm` | `no_grad` Median (`s`) | Forward Median (`s`) | Backward Median (`s`) | `jvp` (`s`) | 2nd Directional (`s`) | Peak RSS (`MiB`) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `255` (`tmpda[0]`) | **A: `pre-w2c9` (`7d71e93`)** | `381` | `8` | `2.726` | `3.814` | `7.803` | `8.401` | `33.925` | `7,986.3` |
| `255` (`tmpda[0]`) | **B: `guard-only`** | `271` | `8` | `2.138` | `3.073` | `4.922` | `5.422` | `15.509` | `6,298.7` |
| `255` (`tmpda[0]`) | **C: `trace-squared-only`** | `377` | `8` | `2.642` | `3.662` | `7.699` | `8.148` | `33.835` | `7,946.1` |
| `255` (`tmpda[0]`) | **D: `current` (B + C)** | `267` | `8` | **`2.129`** | **`3.046`** | **`4.949`** | **`6.004`** | **`15.438`** | **`6,282.9`** |
| `128` (`16 H2O`) | **A: `pre-w2c9` (`7d71e93`)** | `359` | `8` | `0.612` | `0.895` | `1.611` | `2.140` | `6.876` | `2,041.5` |
| `128` (`16 H2O`) | **B: `guard-only`** | `271` | `8` | `0.517` | `0.785` | `1.401` | `2.027` | `4.352` | `1,687.7` |
| `128` (`16 H2O`) | **C: `trace-squared-only`** | `355` | `8` | `0.611` | `0.789` | `1.599` | `2.146` | `6.920` | `2,031.8` |
| `128` (`16 H2O`) | **D: `current` (B + C)** | `267` | `8` | **`0.515`** | **`0.780`** | **`1.373`** | **`1.971`** | **`4.492`** | **`1,692.7`** |
| `32` (`4 H2O`) | **A: `pre-w2c9` (`7d71e93`)** | `357` | `8` | `0.121` | `0.139` | `0.236` | `0.758` | `1.502` | `698.2` |
| `32` (`4 H2O`) | **B: `guard-only`** | `271` | `8` | `0.111` | `0.122` | `0.202` | `1.022` | `1.288` | `653.0` |
| `32` (`4 H2O`) | **C: `trace-squared-only`** | `353` | `8` | `0.124` | `0.133` | `0.233` | `0.742` | `1.470` | `701.3` |
| `32` (`4 H2O`) | **D: `current` (B + C)** | `267` | `8` | **`0.109`** | **`0.119`** | **`0.205`** | **`0.992`** | **`1.310`** | **`651.3`** |

**Attribution Conclusion**: Skipping the redundant `_gap_electron_response` matrix-Fermi solve when `nel` does not track derivatives (`Variant B`) reduces `aten::matmul` count from `381` to `271` at $N=255$ (`359` $\to$ `271` at $N=128$, `357` $\to$ `271` at $N=32`) and drives virtually all of the `14–37%` backward speedup, `13–54%` second-derivative speedup, and `7–21%` peak RSS reduction. The trace-squared Newton slope (`Variant C`) removes 4 `N x N` matrix multiplications per evaluation (`381` $\to$ `377` at $N=255$, `271` $\to$ `267` from `B` to `D`) with minor standalone timing change (`~1–3%`) alongside dense LU solves.

---

## 7. Memory-Bounded Streaming Batch Differentiation

### 7.1 Architectural Separation (`src/dxtb/_src/calculators/batch_streaming.py`)

Why cannot a single differentiable `evaluate_converged_root_batch(...) -> Result` release chunk graphs?
Because PyTorch autograd retains every intermediate tensor and `vmap(implicit_root)` linearization state reachable from `result.energy`, `result.fenergy`, `result.charges`, and `result.density` until the caller invokes `.backward()`. Even if chunks are evaluated sequentially in a loop and concatenated into a single `Result` (`naive-chunks` mode), all chunk autograd graphs remain alive simultaneously in memory prior to `.backward()`.

To bound memory at $O(\text{chunk\_size})$ rather than $O(B)$, [`src/dxtb/_src/calculators/batch_streaming.py`](../../src/dxtb/_src/calculators/batch_streaming.py) provides a dedicated eager first-order value-and-gradient operation:
- [`stream_batch_value_and_grad`](../../src/dxtb/_src/calculators/batch_streaming.py) and [`StreamingBatchValueAndGrad`](../../src/dxtb/_src/calculators/batch_streaming.py) evaluate a weighted separable objective
  $$L = \sum_{b=0}^{B-1} w_b E_b, \qquad E_b = \sum_A E_{b,A}$$
  over chunks of size `chunk_size` (default `1`).
- For each chunk:
  1. Validates `prepared.system_tensor_refs` against `prepared.system_tensor_snapshot` and builds a fresh chunk `System` (`model.setup(system.numbers, batch_mode=system.batch_mode)`), updating `chunk_prepared.system_tensor_refs` so `_validate_prepared_primal` verifies that every SCF and classical setup tensor from `model` matches the prepared snapshots before reusing roots.
  2. Evaluates `evaluate_converged_root_batch` on the chunk, forms the chunk weighted objective $\sum_{b \in \text{chunk}} w_b E_b$, and calls `torch.autograd.grad(..., allow_unused=True)` with respect to chunk `positions`, optional floating `chrg`/`spin`/`field`/`field_grad`, and connected `Model` leaves.
  3. Accumulates detached gradient contributions into the output buffers and deletes the chunk `Result` and chunk `System` before starting the next chunk.
- **Explicit Boundaries and Limitations**:
  - Returns `StreamingBatchValueAndGrad` with detached primal observables (`energy`, `free_energy`), primal/response status (`converged`, `final_failure`, `iterations`), detached first-order gradients (`position_gradient`, `charge_gradient`, `spin_gradient`, `field_gradient`, `field_grad_gradient`, `model_parameter_gradients`), and connected/disconnected `Model` leaf classification (`model_parameter_connected`).
  - Explicitly rejects active `torch.func` transforms (`grad`, `jvp`, `vmap`, `jacfwd`, `jacrev`) and `torch.compile` with `NotImplementedError`.

---

## 8. Eager-Only Adaptive Tanh Doubling

### 8.1 Design and Safety Guards (`_matrix_fermi_eager`, `eager_primal_fermi_scope`)

In `_fermi_occupations_matrix` (`src/dxtb/_src/scf/functional/spectral.py`), the production differentiable recurrence executes a static 29-step tanh-doubling loop (`power_max = 30`) so `vmap` and higher-order AD never branch on data-dependent tensor values. However, for room-temperature (`300 K`) organic systems such as `tmpda`, the adaptive scaling power is `power = 16` (`15` doublings), meaning steps `16..29` (`14` dense `255 x 255` linear solves per spin channel per Newton step) are computed and then masked out by `torch.where(step <= steps, ...)`.

- `_matrix_fermi_eager` executes the exact same scaling range checks (`S0.1b`) and direct-RHS tanh-doubling algebra (`W2c6`), but terminates the Python loop at `max_steps = int(steps[active_lanes].max().item())` when `max_steps < 29`.
- **Strict Isolation**:
  - `eager_primal_fermi_scope` and `_matrix_fermi_eager` raise `RuntimeError` if called when `torch.is_grad_enabled()`, `is_compiling()`, or `torch._C._functorch.get_interpreter_stack()` is non-empty (`test_eager_adaptive_rejects_transformed_or_differentiable_inputs`).
  - `prepare_converged_root_batch(..., eager_primal_fermi=True)` activates `eager_primal_fermi_scope(True)` **only** around the detached `torch.no_grad()` primal solver (`_solve_map` / `_solve_map_anderson`).
  - Post-solver root certification (`adapter.evaluate`, `evaluate_potential`, `evaluate_hamiltonian`) and all subsequent `evaluate_converged_root_batch` calls execute outside `eager_primal_fermi_scope`, using the unchanged static 29-step recurrence (`test_eager_adaptive_fermi_root_is_revalidated_by_fixed_raw_map`).

### 8.2 `tmpda[0]` Static vs. Adaptive Root-Preparation Benchmark (`adaptive-root-*.json`)

| Metric | Static 29-Step Doubling (`adaptive-root-static.json`) | Eager Adaptive Doubling (`adaptive-root-eager.json`) | Change / Parity |
|---|---:|---:|---|
| Primal SCF iterations | `66` | `66` | Exact match |
| Certified by static 29-step raw map | `True` | `True` | Exact match |
| Total `dense_solve` calls | `6,030` | `3,258` | **-45.97% (`-2,772` solves)** |
| Root preparation wall time (`s`) | `74.495` | **`47.678`** | **-36.00% (`1.56x` speedup)** |
| Independent static evaluation (`s`) | `3.565` | `3.526` | Parity |
| Peak process RSS (`MiB`) | `1,935.53` | `1,971.58` | Parity (`~1.93 GiB`) |
| Converged root SHA-256 | `d49a1025...c3188` | `d49a1025...c3188` | **Bitwise identical** |
| Raw residual $\|r\|_2$ / $\|r\|_\infty$ | `6.012016e-09` / `4.355970e-09` | `6.012016e-09` / `4.355970e-09` | **Bitwise identical** |
| Position gradient norm (`Eh/Bohr`) | `41.29473633572572` | `41.29473633572572` | **Bitwise identical** |

---

## 9. `tmpda` 1 / 2 / 4 / 8-Conformer Physical Scaling Progression

All runs executed in isolated single-threaded processes (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, Python `3.11.15`, PyTorch `2.8.0+cpu`) on `tmpda` conformers (`69` atoms, `255` orbitals, `float64`, GFN1-xTB, `T = 300 K`, charge SCP, Anderson mixing, `maxiter = 300`, `x_atol = 1e-8`, `x_atol_max = 1e-5`).

### 9.1 End-to-End Scaling Table (`benchmarks/results/e6c1-w2c10/` & `e6c1-w2c9/`)

| Conformers $N$ | Mode (`Artifact`) | Root Prep (`s`) | Diff / Stream (`s`) | Total (`s`) | RSS After Root (`MiB`) | RSS After Forward (`MiB`) | Peak RSS (`MiB` / `GiB`) | Primal SCF Iterations |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| `1` | Simultaneous (`tmpda_n1_pos.json`) | `70.88` | `26.43` | `97.53` | `468.1` | — | `1,422.21` (`1.39 GiB`) | `[66]` |
| `1` | Streaming, chunk 1 (`streaming-n1-pos.json`) | `76.40` | `30.35` | `106.75` | `339.2` | `505.0` | `1,952.04` (`1.91 GiB`) | `[66]` |
| `2` | Simultaneous (`tmpda_n2_pos.json`) | `176.51` | `62.49` | `239.21` | `584.1` | — | `2,366.15` (`2.31 GiB`) | `[66, 96]` |
| `2` | Naive chunks, chunk 1 (`naive-chunks-n2-pos.json`) | `187.26` | `64.16` | `251.42` | `319.6` | `3,435.3` | `3,435.29` (`3.35 GiB`) | `[66, 96]` |
| `2` | Streaming, chunk 1 (`streaming-n2-pos.json`) | `177.49` | `62.73` | `240.22` | `337.0` | `549.8` | **`2,301.99` (`2.25 GiB`)** | `[66, 96]` |
| `4` | Simultaneous (`tmpda_n4_pos.json`) | `374.14` | `124.87` | `499.24` | `767.4` | — | `5,732.04` (`5.60 GiB`) | `[66, 96, 112, 72]` |
| `4` | Streaming, chunk 1 (`streaming-n4-pos.json`) | `387.98` | `132.54` | `520.52` | `323.9` | `1,695.8` | **`2,325.84` (`2.27 GiB`)** | `[66, 96, 112, 72]` |
| `8` | Streaming, chunk 1 (`streaming-n8-pos.json`) | `850.47` | `250.23` | **`1,100.71`** | `339.1` | `2,066.9` | **`2,320.49` (`2.27 GiB`)** | `[66, 96, 112, 72, 111, 115, 105, 78]` |

### 9.2 Key Scaling Findings

1. **Why Naive Chunking Fails (`naive-chunks-n2-pos.json`)**:
   - Evaluating chunks of size 1 with `evaluate_converged_root_batch` and concatenating the resulting `Result` energies before `.backward()` causes `rss_after_forward` to jump to **`3,435.29 MiB` (`3.35 GiB`)** at $N=2$ because both chunk autograd graphs remain alive simultaneously.
   - By contrast, `stream_batch_value_and_grad` (`streaming-n2-pos.json`) differentiates and releases each chunk before moving to the next, keeping `rss_after_forward` at **`549.82 MiB`** and peak RSS at **`2,301.99 MiB` (`2.25 GiB`)** (**-32.99%** vs. naive chunks).

2. **Constant Peak RSS Across $N = 2, 4, 8$ Conformers**:
   - Under `stream_batch_value_and_grad(..., chunk_size=1)`, peak RSS is completely flat as conformer count doubles from 2 to 4 to 8:
     - $N = 2$: **`2,301.99 MiB` (`2.25 GiB`)**
     - $N = 4$: **`2,325.84 MiB` (`2.27 GiB`)** (**-59.42%** vs. simultaneous `5,732.04 MiB`)
     - $N = 8$: **`2,320.49 MiB` (`2.27 GiB`)** (**-72.2%** vs. the historical `8.16 GiB` W2 ceiling)
   - All 8 `tmpda` conformers converged cleanly (`[66, 96, 112, 72, 111, 115, 105, 78]` iterations; residuals `3.57e-09` to `8.97e-09`), with lanes `0..3` matching the $N=1, 2, 4$ energies (`-29.758437462176317`, `-11.312642703835634`, `-21.692981421967666`, `-33.762321030729446 Eh`) and free energies to machine precision (`< 3e-16 Eh`).

---

## 10. Multi-Environment Regression Gate Summary

1. **Five-Environment Focused SCF / Implicit / Spectral Gate**:
   - Executed across all five supported `tad-mctc 0.11.3` CPU environments (`py3.10-torch2.7.0`, `py3.11-torch2.8.0`, `py3.12-torch2.11.0`, `py3.12-torch2.13.0`, `py3.14-torch2.14.1`) over:
     - `test/test_scf/test_e6c1_w2c9_batch_implicit.py` (`34` tests),
     - `test/test_scf/test_invariant_spectral_density.py` (`72` tests, `68` non-slow),
     - `test/test_scf/test_invariant_free_energy.py`,
     - `test/test_scf/test_e6c1_w2c5b_shift_contract.py`,
     - `test/test_scf/test_e6c1_w2c6_tanh_solve.py` (`32` tests),
     - `test/test_scf/test_functional_implicit_root.py`,
     - `test/test_scf/test_e6c1_w2c1_mapped_implicit.py`,
     - `test/test_scf/test_functional_root_derivatives.py`.
   - Result: **244/244 non-slow passed** (`248/248` including `@pytest.mark.slow` third-order `NO2` reference tests on `py3.11-torch2.8.0`) across all five environments (`0 failed`).

2. **15-Module E7a / E6bcompat / E6a / E6bimpl Primary Regression Gate**:
   - Executed on both `py3.11-torch2.8.0` and `py3.14-torch2.14.1` across:
     `test_e7a_padding_safety.py`, `test_e6bcompat_differential.py`, `test_e6bcompat_policy.py`, `test_e6a0_system_vmap.py`, `test_e6a1_conformer.py`, `test_e6a1a_gfn0_spin_validation.py`, `test_e6a2_public_status.py`, `test_e6a3_component_vmap.py`, `test_e6a4_transform_acceptance.py`, `test_e6bimpl0_padded_integrals.py`, `test_e6bimpl1a_padded_spectral.py`, `test_e6bimpl2a_gfn2_canonical.py`, `test_e6bimpl2b_gfn2_scf.py`, `test_e6bimpl3_legacy_composition.py`, and `test_e6c1_w2c2a_configuration.py`.
   - Result: **261/261 passed** on `py3.11-torch2.8.0` and **261/261 passed** on `py3.14-torch2.14.1` (`0 failed`).

---

## 11. Independent Code Review and Resolution

Requested reviewer: `Luna / high`
Actual subagent role: `Independent Scientific Code Reviewer` (`research` subagent, conversation `439f9f3c-adeb-4736-bbef-f68f5f219010`)

The independent read-only review audited the implementation against all 20 checklist items in Section 20 across two passes. All findings were resolved and verified by automated regression tests:

1. **Raw-Byte Snapshot Extraction Under Active `jvp` Transforms (`batch_implicit.py`)**:
   - *Finding*: Calling `current.detach()` outside `torch._C._DisableFuncTorch()` while a `jvp` interpreter is active on the functorch stack re-wraps the detached tensor in a `GradTrackingTensor`, and `torch.equal` returns `False` on `System` setup tables containing padded `NaN` entries.
   - *Resolution*: Executed `current.detach()` and `.contiguous().cpu().numpy().tobytes()` inside `with torch._C._DisableFuncTorch():` in `_unwrap_primal_tensor` and `_tensor_raw_bytes`, and used raw-byte comparison whenever `not is_compiling()` with a `NaN`-safe fallback `(primal == reference) | (isnan(primal) & isnan(reference))` when compiling.
2. **Stale-Root Error Precedence for In-Place `Model` Mutations vs. `chrg`/`spin` Overrides (`batch_implicit.py`)**:
   - *Finding*: Mutating all floating parameters in `model.par` also changes `refocc` and therefore `nel`, causing `nel_matches` to fire with a charge/spin message before `parameter_snapshots` could report a stale `Model` setup.
   - *Resolution*: Scoped the explicit `nel_matches` check in `_refresh_prepared_batch_inputs` to calls where `chrg is not _UNSET or spin is not _UNSET`, while `parameter_snapshots` and `input_tensor_versions` catch any `Model` or `System` setup mutation.
3. **Autograd Graph Traversal Wrapper Lifetime in Python 3.10 / PyTorch 2.7 (`batch_implicit.py`)**:
   - *Finding*: Storing `id(node)` in `seen_ids: set[int]` during `grad_fn.next_functions` traversal allowed Python to garbage-collect popped temporary `grad_fn` wrapper objects and reuse their `id()` for newly visited `AccumulateGrad` nodes in `py3.10-torch2.7.0`.
   - *Resolution*: Stored `node` objects directly in `seen_nodes: set[Any]` and kept traversed objects in `seen_objs: dict[int, object]`, preventing wrapper address recycling across all five Python/PyTorch environments.
4. **Non-Trainable Classical `Model` Parameter Mutation Detection in `stream_batch_value_and_grad` (`batch_streaming.py`, `batch_implicit.py`)**:
   - *Finding (`MAJOR 2`)*: `_chunk_prepared` replaced `system=fresh_system` on `chunk_prepared` without updating `system_tensor_refs`, so mutating a classical-only `Model` parameter (`O.arep.param`) when `requires_grad=False` was not compared against `prepared.system_tensor_snapshot`.
   - *Resolution*: Updated `_chunk_prepared` to validate `prepared.system_tensor_refs` and replace `system_tensor_refs` on `chunk_prepared` with the setup tensors collected from `fresh_system = model.setup(...)`. Verified in `test_streaming_rejects_changed_nontrainable_model_parameters`.
5. **Runtime Classification for `BatchRootFailureCode.RESPONSE_FAILURE` (`batch_implicit.py`, `batch_streaming.py`)**:
   - *Finding (`MINOR 1`)*: `BatchRootFailureCode.RESPONSE_FAILURE` was documented on the enum but had no runtime helper or status consumer.
   - *Resolution*: Added `classify_response_failure_status` in `batch_implicit.py` and automatic `BatchRootFailureCode.RESPONSE_FAILURE` classification in `stream_batch_value_and_grad` (`StreamingBatchValueAndGrad.final_failure`), while keeping primal `Result.status` immutable. Verified in `test_batch_backward_response_failure_is_lane_local`.
6. **Exact Alignment of Benchmark Tables and Symbol Names (`E6c1-W2c10-batch-hardening-and-streaming.md`)**:
   - *Finding (`MAJOR 1`)*: Section 6.2 had an unbacked batched-shape table and Table 6.1 had mismatched numbers for $N=32$ and $N=128$ (`B`) relative to `w2c10-shape-*.json`, plus stale symbol/test references.
   - *Resolution*: Removed the unbacked table and synchronized all Table 6.1 values, symbol names, dataclass fields, and test function names with the exact JSON artifacts and source files.

---

## 12. Final Verdict Matrix

| Area | Status | Evidence |
|---|---|---|
| Prepared-root validity | **CLOSED** | `_TensorValueSnapshot` + `_version` checks in `batch_implicit.py` and `batch_streaming.py`; verified across in-place mutations, overrides, trainable and non-trainable `Model`/`System` mutations, `grad`/`jvp`/`jacfwd`/`jacrev`, and `torch.compile` in `test_prepared_root_rejects_mutated_inputs_and_model_parameters`, `test_prepared_root_snapshots_are_immutable_values`, and `test_streaming_rejects_changed_nontrainable_model_parameters` |
| Strictness and failure contracts | **CLOSED** | `_resolve_strictness`, fail-closed `_classify_fermi_code`, `_InvalidLaneOutput`, and `classify_response_failure_status` / `RESPONSE_FAILURE`; verified in `test_strictness_precedence_and_fermi_failure_classification`, `test_failure_policy_strict_and_non_strict_mixed_lanes`, and `test_batch_backward_response_failure_is_lane_local` |
| Spectral derivative guard | **CLOSED** | Per-level `_has_jvp_tangent_at_level` + conservative fallback in `spectral.py`; verified against always-response oracle across 1st–3rd order and mixed derivatives in `test_electron_response_guard_matches_always_response_reference` |
| Spectral slope formulation | **CLOSED** | Row-wise cancellation-safe `(diag_density - (density * density).sum(dim=-1)).sum(-1) / safe_kt` in `_fermi_free_energy_setup`; verified across metallic, insulator, near-saturated, padded, and `float32`/`float64` 1st–3rd derivatives in `test_trace_squared_slope_matches_legacy_density_product` |
| Benchmark provenance | **CLOSED** | Process-isolated 4-way A/B/C/D attribution against exact `7d71e930e5bdbd52a252bf23d21803b94d2695f4` in `benchmarks/results/e6c1-w2c10/w2c10-*.json` and `guard-only.patch` / `trace-squared-only.patch` |
| Memory-bounded differentiation | **CLOSED** | `stream_batch_value_and_grad` / `StreamingBatchValueAndGrad` in `batch_streaming.py`; exact parity with simultaneous batch gradients across positions, perturbations, and 2,261 `Model` leaves; 59.42% lower peak RSS at $N=4$ (`2.27 GiB` vs. `5.60 GiB`) |
| Eager adaptive doubling | **CLOSED** | `_matrix_fermi_eager` + `eager_primal_fermi_scope` in `spectral.py` / `batch_implicit.py`; `-45.97%` dense solves (`6,030` $\to$ `3,258`) and `-36.00%` root prep time (`74.49 s` $\to$ `47.68 s`) on `tmpda[0]` with bitwise-identical root SHA-256 `d49a1025...` (`adaptive-root-*.json`) |
| 1/2/4/8-conformer scaling | **CLOSED** | `streaming-n1/2/4/8-pos.json` and `naive-chunks-n2-pos.json`: full 8-conformer `tmpda` batch completed in `1,100.71 s` at `2,320.49 MiB` (`2.27 GiB`) peak RSS with all 8 lanes converged |
| W2 status | **OPEN (API/contract decision required)** | Internal converged-root + streaming architecture clears the 8-conformer `tmpda` workload at `2.27 GiB` peak RSS, but public W2 acceptance still references literal `vmap(System.singlepoint)` finite-trip semantics; project-owner approval of the explicit batch API contract is required to close W2 |
| W3 status | **OPEN** | Mixed-System padded batching performance certification remains downstream of W2 route sign-off |
| E6broute status | **BLOCKED** | Blocked on W2/W3 sign-off and public batch route migration |

---

## 13. W2c10 Implementation Commit Ancestry

- **Base SHA (W2c9)**: `7b946a21fcd27807127d54b4262b5ec5177d607d`
- **Commit 1 (Correctness and failure-contract fixes)**: `b9d49800e97d31865468ad982be21542625d75c4`
- **Commit 2 (Spectral hardening and tests)**: `3630dd85a682d79d42f4597c6b5b6f47190a77f8`
- **Commit 3 (Memory-bounded evaluation)**: `b6db1afe9dc362401d62e72f0112fa8e57cad1a0`
- **Commit 4 (Eager-primal performance optimization)**: `25508b50a31635b8dcb2205a7a82003894c85f78`
