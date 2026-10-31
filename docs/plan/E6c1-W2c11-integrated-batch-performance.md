# E6c1-W2c11 — Integrated Converged-Root Performance, Streaming Certification, and Public Batch API Readiness

## 1. Starting State, Scope, and Architectural Invariants

- **Repository**: `marvinfriede/dxtb`
- **Branch**: `claude/tender-dijkstra-cda42s`
- **Verified Base SHA (W2c10)**: `8ed902f762804a9da7f4537e8d40bab80822ed1d`
- **Published W2c11 Commits**:
  - Spectral/root-preparation optimization and tests: `8e3722923d2565bd2fd654c8be9ca8e0bebcbf89`
  - Streaming failure-classification, parallel root preparation, and contract improvements: `da43ec0779d2b0b852003fcb78f2d17c72f5e13a`
  - Integrated, parallel, Model-gradient, and matched legacy benchmarks: `8be7e32542d9644b3aadd4a26dcc29179bd641fa`
- **Primary Benchmark Environment**: `py3.11-torch2.8.0` (Python `3.11.15`, PyTorch `2.8.0+cpu`, `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`)
- **Multi-Environment Regression Matrix**: `py3.10-torch2.7.0`, `py3.11-torch2.8.0`, `py3.12-torch2.11.0`, `py3.12-torch2.13.0`, `py3.14-torch2.14.1`

### 1.1 Retained Starting Components

All accepted W2c6–W2c10 numerical and safety mechanisms are retained without regression:

1. **W2c6 direct-RHS tanh doubling** (`_solve_positive_definite_direct` / `dense_solve` in `src/dxtb/_src/scf/functional/spectral.py`).
2. **W2c9/W2c10 derivative-aware electron-count response guard** (`_tracks_derivatives` and per-level `_has_jvp_tangent_at_level` in `src/dxtb/_src/scf/functional/spectral.py`).
3. **W2c10 cancellation-safe Newton slope** (`(diag_density - (density * density).sum(dim=-1)).sum(-1) / safe_kt` in `_fermi_free_energy_setup`).
4. **W2c10 immutable byte-backed prepared-root lifetime validation** (`_TensorValueSnapshot` and `_version` checks in `src/dxtb/_src/calculators/batch_implicit.py`).
5. **W2c10 memory-bounded streaming differentiation** (`stream_batch_value_and_grad` in `src/dxtb/_src/calculators/batch_streaming.py`).
6. **W2c10 eager-adaptive matrix-Fermi primal evaluation** (`_matrix_fermi_eager` and `eager_primal_fermi_scope` in `src/dxtb/_src/scf/functional/spectral.py`).

### 1.2 Ownership Boundaries

In accordance with `AGENTS.md` and the shared SCF coordination record:
- Functional solver files owned by `scf-redesign` (`map.py`, `eager.py`, `solver.py`, `anderson.py`, `implicit.py`) were not modified.
- All W2c11 production changes are confined to `src/dxtb/_src/scf/functional/spectral.py`, `src/dxtb/_src/calculators/batch_implicit.py`, and `src/dxtb/_src/calculators/batch_streaming.py`.
- Public `System.singlepoint()` semantics, `Calculator` routing, and the W2 `torch.func.vmap(System.singlepoint)` acceptance definition remain unchanged.

---

## 2. Part A — Streaming Response-Failure Classification and API Contracts

### 2.1 Five-Way Failure Classification Policy (`batch_implicit.py`, `batch_streaming.py`)

In W2c10, `stream_batch_value_and_grad` classified post-differentiation failures using only the finiteness of `position_gradient`, and labeled any post-root nonfinite position gradient on a valid primal lane as `BatchRootFailureCode.RESPONSE_FAILURE` (`9`). That conflated confirmed implicit linear-response solver failures with nonfinite derivatives originating in other input families (`chrg`, `spin`, `field`, `field_grad`, shared `Model` parameters) or in post-root energy/VJP assembly.

W2c11 completes the failure-classification contract:

1. **Failure Codes (`BatchRootFailureCode` in `src/dxtb/_src/calculators/batch_implicit.py`)**:
   - `NONE = 0` $\to$ category `"success"`
   - `INVALID_TEMPERATURE = 1`, `TOO_MANY_ELECTRONS = 2`, `FERMI_NONCONVERGENCE = 3`, `ELECTRON_COUNT = 4` $\to$ category `"fermi_failure"`
   - `MAXITER_EXCEEDED = 5`, `RESIDUAL_TOLERANCE = 6`, `NONFINITE_STATE = 7`, `INVALID_PADDING = 8` $\to$ category `"primal_scf_failure"`
   - `RESPONSE_FAILURE = 9` $\to$ category `"response_failure"` (reserved strictly for lanes where the implicit linear-response solver ran and reported `status != 0` or a nonfinite response vector/residual)
   - `NONFINITE_DERIVATIVE = 10` $\to$ category `"nonfinite_derivative"` (emitted when primal SCF and the linear-response solver succeeded or were bypassed, yet a requested derivative with respect to `positions`, `chrg`, `spin`, `field`, `field_grad`, or shared `Model` parameters is nonfinite)
   - `RESPONSE_NOT_RUN = -1` (sentinel used in `response_status` for unselected zero-weight lanes, primal-failed lanes, or calls that do not invoke the linear-response VJP)

2. **Call-Local Response Diagnostics (`ResponseDiagnosticCollector`, `BatchResponseDiagnostics`)**:
   - `evaluate_converged_root_batch` accepts an optional `response_diagnostics: ResponseDiagnosticCollector | None = None`.
   - When provided, `_ResponseParamsMonitor` and `_ResponseOutputMonitor` (`torch.autograd.Function` subclasses) wrap the mapped `implicit_root` parameter inputs and state outputs to record incoming/outgoing cotangent finiteness, while `response_diagnostics.solve` is passed as the linear solver callback to `implicit_fixed_point` (`bicgstab`) to record per-lane `(converged, status, iterations, residual, target)` into the call-local `ResponseDiagnosticCollector` (and mask any unconverged linear-response solution with `NaN`).
   - No module-global mutable state is used, and immutable primal `Result.status` is never mutated retrospectively.
   - `StreamingBatchValueAndGrad` returns explicit per-lane response diagnostic fields (`converged` remains the primal SCF convergence flag while `response_converged`, `final_failure`, and `failure_category` report post-differentiation status):
     - `response_converged: Tensor` (`(B,)`, `bool`)
     - `response_status: Tensor` (`(B,)`, `int64`; `0` on converged BiCGSTAB, `1..3` on BiCGSTAB failure, `-1` when response solver did not run)
     - `response_iterations: Tensor` (`(B,)`, `int64`)
     - `response_residual: Tensor` (`(B,)`, floating; `NaN` when response solver did not run)
     - `final_failure: Tensor` (`(B,)`, `int64` `BatchRootFailureCode`)
     - `failure_category: tuple[str, ...]` (`"success"`, `"primal_scf_failure"`, `"fermi_failure"`, `"response_failure"`, `"nonfinite_derivative"`)

3. **Per-Lane Attribution Across All Input Families and Chunk Sizes (`chunk_size = 1, 2, 4`)**:
   - Shared external inputs (`chrg`, `spin`, `field`, `field_grad` supplied as unbatched tensors or with leading dimension `1`) are expanded to per-lane chunk intermediate tensors (`(n_active, ...)`) inside each active chunk so that per-lane derivatives (including any caller `register_hook` transformations on shared leaves) are inspected for finiteness before reducing back to the caller's input shape.
   - For shared `Model` parameters in multi-lane chunks (`n_active > 1`), if chunk-level parameter gradients contain nonfinite values, `stream_batch_value_and_grad` isolates the offending lane(s) by re-evaluating single-lane subsets `_subset_prepared(prepared, model, [lane_idx])` on that anomalous chunk. Normal execution never materializes a `(B, num_params)` Jacobian.
   - Under `strict=False`, any lane classified as `RESPONSE_FAILURE` (`9`) has its requested gradients (`positions`, lane-specific or shared `chrg`/`spin`/`field`/`field_grad`, and connected `Model` parameters) masked with `NaN` and emits `SCFConvergenceWarning`, whereas a lane classified as `NONFINITE_DERIVATIVE` (`10`) preserves finite derivatives on unaffected input families while propagating `NaN` on the affected input family and emitting `SCFConvergenceWarning`. Under `strict=True` (or `force_convergence=True`), both `RESPONSE_FAILURE` (`9`) and `NONFINITE_DERIVATIVE` (`10`) raise `SCFConvergenceError` identifying the offending lane.

### 2.2 Strengthened Streaming API Contracts (`batch_streaming.py`)

- **Direct `System` or `PreparedConvergedRootBatch` Input**: `stream_batch_value_and_grad` accepts either a pre-computed `PreparedConvergedRootBatch` or a single-`System` owner (`System`), automatically preparing roots with `eager_primal_fermi` and `num_workers` when given a `System`.
- **Objective Weights Validation and Zero/Negative Weight Semantics**:
  - `weights` must be a 1D tensor of shape `(B,)` with all finite entries; `NaN` or `Inf` weights raise `ValueError("weights must contain only finite values.")`.
  - Zero-weight lanes (`weights[i] == 0.0`) are evaluated under `with torch.no_grad():` for primal energies/free energies, contribute exact `0.0` gradients across all input families (even if the lane's primal root failed under `strict=False`), and record `response_status = -1` (`RESPONSE_NOT_RUN`), `response_converged = False`, `response_iterations = 0`, `response_residual = NaN`.
  - Negative weights (`weights[i] < 0.0`) scale all lane gradients linearly without altering primal root validity or response convergence.
- **Parameter Aliasing, Connected-Zero vs. Disconnected Leaves, and Caller Hooks**:
  - `model.par.parameters()` is deduplicated by parameter identity while preserving leaf order.
  - Structural connectivity is determined via autograd graph reachability (`_collect_system_and_model_tensors`) so connected parameters whose numerical gradient happens to be zero receive an explicit zero `Tensor` (`model_parameter_connected[k] == True`), whereas structurally disconnected parameters receive `None` (`model_parameter_connected[k] == False`).
  - When callers pass inputs with `requires_grad=True` and registered `register_hook` callbacks, `stream_batch_value_and_grad` fires the caller's hooks on the final accumulated gradient via `_deliver_input_hook`.
- **Unsupported Transform and Compilation Contexts**:
  - `create_graph=True`, active functorch transforms (`vmap`, `jvp`, `jacfwd`, `jacrev`, `grad`), and `torch.compile` on `stream_batch_value_and_grad` are explicitly rejected with `NotImplementedError` directing callers to `evaluate_converged_root_batch` for higher-order/transform evaluation.

---

## 3. Part B — Combined Eager-Primal & Streaming Integration

### 3.1 Thread-Local `eager_primal_fermi` Scope and Unconditional Static Validation

- `src/dxtb/_src/scf/functional/spectral.py` implements `_normalize_primal_fermi_mode(mode: bool | str) -> str`, accepting:
  - `False` or `"static"`: fixed 29-step matrix-Fermi tanh doubling (`90` dense solves per SCF iteration).
  - `True` or `"adaptive"`: detached-primal adaptive matrix-Fermi tanh doubling (`_matrix_fermi_eager`), terminating the doubling recurrence at `max_power = int(power.max().item())`.
  - `"spectral"`: detached-primal spectral Fermi density evaluation (`_matrix_fermi_spectral`) combined with single-diagonalization 1D Newton chemical-potential updates in `_fermi_free_energy_setup`.
- `eager_primal_fermi_scope(enabled: bool | str = True)` stores the normalized mode in thread-local storage (`_EAGER_PRIMAL_STATE.mode`) and restores the previous mode in a `finally:` block.
- Every non-static primal mode is guarded by `_require_detached_eager_primal(caller, *tensors)`, which raises `RuntimeError` if `torch.is_grad_enabled()`, `_tracks_derivatives(*tensors)`, `_inside_functorch_interpreter()`, or `is_compiling()` is active.
- Inside `_solve_and_validate_lane_primal` (`src/dxtb/_src/calculators/batch_implicit.py`), the primal fixed-point loop runs inside `with eager_primal_fermi_scope(eager_primal_fermi):`, and the converged root is then unconditionally re-evaluated and certified inside `with eager_primal_fermi_scope(False):` using the unchanged static matrix-Fermi `scf_map`.

### 3.2 Four-Way Integrated Benchmark Matrix ($N = 1, 2, 4, 8$ `tmpda` Conformers)

All runs were executed in isolated single-threaded processes (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `py3.11-torch2.8.0`) on the frozen `tmpda` conformer workload (`69` atoms, `255` orbitals, GFN1-xTB, PyTorch integrals, `float64`, FULL SCF, Charge SCP, Anderson mixer, `etemp = 300 K`, `maxiter = 300`, `x_atol = f_atol = 1e-8`, `x_atol_max = 1e-5`, position gradients).

| Mode | Fermi Mode | Diff Mode | $N$ | Root Prep (`s`) | Diff / Stream (`s`) | Total (`s`) | RSS After Root (`MiB`) | RSS After Fwd (`MiB`) | Peak RSS (`MiB` / `GiB`) | Max $\|\Delta \nabla_{\mathbf{R}} E\|_\infty$ vs. Serial Ref (`Eh/Bohr`) | Artifact |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **A** | `static` | `simultaneous` | `1` | `74.467` | `30.832` | `105.548` | `338.6` | `1,989.6` | `2,030.5` (`1.98 GiB`) | `0.000e+00` | `integrated-static-simultaneous-n1.json` |
| **A** | `static` | `simultaneous` | `2` | `184.863` | `71.450` | `256.555` | `329.5` | `2,382.1` | `3,489.9` (`3.41 GiB`) | `6.772e-14` | `integrated-static-simultaneous-n2.json` |
| **A** | `static` | `simultaneous` | `4` | `395.210` | `144.963` | `540.494` | `335.6` | `6,812.8` | `8,273.3` (`8.08 GiB`) | `9.592e-14` | `integrated-static-simultaneous-n4.json` |
| **B** | `static` | `streaming (c=1)` | `1` | `73.600` | `30.832` | `104.689` | `339.1` | `1,156.5` | `2,010.7` (`1.96 GiB`) | `0.000e+00` | `integrated-static-streaming-n1.json` |
| **B** | `static` | `streaming (c=1)` | `2` | `177.481` | `66.072` | `243.798` | `322.4` | `1,099.7` | `1,997.0` (`1.95 GiB`) | `0.000e+00` | `integrated-static-streaming-n2.json` |
| **B** | `static` | `streaming (c=1)` | `4` | `389.601` | `136.197` | `526.020` | `346.2` | `1,123.3` | `2,020.4` (`1.97 GiB`) | `0.000e+00` | `integrated-static-streaming-n4.json` |
| **B** | `static` | `streaming (c=1)` | `8` | `850.474` | `250.233` | `1,100.707` | `339.1` | `2,066.9` | `2,320.5` (`2.27 GiB`) | `0.000e+00` | `e6c1-w2c10/streaming-n8-pos.json` |
| **C** | `adaptive` | `simultaneous` | `1` | `46.554` | `30.726` | `77.507` | `348.0` | `1,962.6` | `2,002.7` (`1.96 GiB`) | `0.000e+00` | `integrated-adaptive-simultaneous-n1.json` |
| **C** | `adaptive` | `simultaneous` | `2` | `113.681` | `75.480` | `189.409` | `343.4` | `2,415.0` | `3,485.6` (`3.40 GiB`) | `6.772e-14` | `integrated-adaptive-simultaneous-n2.json` |
| **C** | `adaptive` | `simultaneous` | `4` | `241.378` | `139.492` | `381.176` | `344.9` | `6,800.5` | `8,340.7` (`8.15 GiB`) | `9.592e-14` | `integrated-adaptive-simultaneous-n4.json` |
| **D** | `adaptive` | `streaming (c=1)` | `1` | `47.459` | `31.266` | `78.976` | `337.3` | `1,067.3` | `2,022.2` (`1.97 GiB`) | `0.000e+00` | `integrated-adaptive-streaming-n1.json` |
| **D** | `adaptive` | `streaming (c=1)` | `2` | `114.437` | `67.015` | `181.719` | `338.1` | `1,326.9` | `2,011.4` (`1.96 GiB`) | `0.000e+00` | `integrated-adaptive-streaming-n2.json` |
| **D** | `adaptive` | `streaming (c=1)` | `4` | `240.929` | `137.964` | `379.184` | `347.2` | `1,451.8` | `2,017.1` (`1.97 GiB`) | `0.000e+00` | `integrated-adaptive-streaming-n4.json` |
| **D** | `adaptive` | `streaming (c=1)` | `8` | `536.245` | `265.898` | `802.379` | `360.5` | `1,871.6` | **`2,033.3` (`1.99 GiB`)** | `0.000e+00` | `integrated-adaptive-streaming-n8.json` |
| **E** | `spectral` | `simultaneous` | `1` | `7.742` | `31.080` | `39.130` | `345.4` | `1,966.6` | `1,998.8` (`1.95 GiB`) | `1.075e-13` | `integrated-spectral-simultaneous-n1.json` |
| **E** | `spectral` | `simultaneous` | `2` | `17.657` | `70.323` | `88.304` | `342.0` | `2,397.0` | `3,481.3` (`3.40 GiB`) | `1.286e-13` | `integrated-spectral-simultaneous-n2.json` |
| **E** | `spectral` | `simultaneous` | `4` | `37.074` | `143.839` | `181.151` | `364.1` | `6,821.4` | `8,470.0` (`8.27 GiB`) | `1.920e-13` | `integrated-spectral-simultaneous-n4.json` |
| **F** | `spectral` | `streaming (c=1)` | `1` | `7.770` | `31.628` | `39.720` | `337.3` | `973.0` | `2,018.2` (`1.97 GiB`) | `1.075e-13` | `integrated-spectral-streaming-n1.json` |
| **F** | `spectral` | `streaming (c=1)` | `2` | `18.403` | `67.462` | `86.095` | `336.9` | `980.2` | `2,009.9` (`1.96 GiB`) | `1.075e-13` | `integrated-spectral-streaming-n2.json` |
| **F** | `spectral` | `streaming (c=1)` | `4` | `37.222` | `137.140` | `174.585` | `370.3` | `1,077.9` | `2,042.2` (`1.99 GiB`) | `1.920e-13` | `integrated-spectral-streaming-n4.json` |
| **F** | `spectral` | `streaming (c=1)` | `8` | **`82.377`** | `278.157` | **`360.767`** | `376.4` | `1,120.9` | **`2,053.8` (`2.01 GiB`)** | `1.920e-13` | `integrated-spectral-streaming-n8.json` |

---

## 4. Part C — Attacking the Dominant Primal SCF Root-Preparation Cost

### 4.1 Physical Root-Preparation Phase Breakdown (`tmpda` Conformer 0)

Single-conformer root preparation on `tmpda[0]` (`66` SCF iterations, `255` orbitals) was profiled across all three primal Fermi modes (`benchmarks/results/e6c1-w2c11/root-profile-{static,adaptive,spectral}.json`):

| Metric | Static (`False` / `"static"`) | Eager-Adaptive (`True` / `"adaptive"`) | Detached Spectral (`"spectral"`) | Spectral vs. Static | Spectral vs. Adaptive |
|---|---:|---:|---:|---:|---:|
| Converged SCF iterations | `66` | `66` | `66` | Identical | Identical |
| Total root-preparation wall time (`s`) | `73.691` | `46.140` | **`7.539`** | **9.77x faster (`-89.77%`)** | **6.12x faster (`-83.66%`)** |
| SCF map loop wall time (`s`) | `73.338` | `45.814` | **`7.192`** | **10.20x faster** | **6.37x faster** |
| `fermi_density_matrix` total (`s`, `67` calls) | `69.691` (`94.57%`) | `42.179` (`91.42%`) | **`3.535`** (`46.89%`) | **19.71x faster (`-94.93%`)** | **11.93x faster (`-91.62%`)** |
| `fermi_density_matrix` inside 66-iter loop (`s`) | `68.562` | `41.092` | **`2.436`** | **28.15x faster** | **16.87x faster** |
| Static validation single `scf_map` call (`s`) | `1.129` | `1.087` | `1.099` | `1.03x` | `0.99x` |
| Fock + atomic charges + potential (`s`) | `3.647` | `3.635` | `3.657` | `1.00x` | `0.99x` |
| Anderson mixer step (`s`, `65` calls) | `0.0372` | `0.0363` | `0.0375` | `0.99x` | `0.97x` |
| `dense_solve` (Cholesky) calls | `6,030` | `3,258` | **`90`** (validation only) | **`-98.51%`** | **`-97.24%`** |
| `eigh` calls | `67` | `67` | `133` | `1.99x` | `1.99x` |
| Certified static-map $\|r\|_2$ / $\|r\|_\infty$ | `6.012016e-09` / `4.355970e-09` | `6.012016e-09` / `4.355970e-09` | `6.011914e-09` / `4.355904e-09` | `< 1.1e-13` | `< 1.1e-13` |
| Max $\|\Delta x^\star\|_\infty$ vs. static root | `0.000e+00` | `0.000e+00` | **`2.343e-14`** | Machine precision | Machine precision |
| Peak RSS (`MiB`) | `387.4` | `383.1` | `385.4` | `-2.0 MiB` | `+2.3 MiB` |

### 4.2 Detached-Primal Spectral Matrix-Fermi Formulation (`eager_primal_fermi="spectral"`)

Under `static` and `adaptive` modes, each SCF iteration diagonalizes the orthogonalized Fock matrix in `filling._get_fermi_occupation_core_state` to bracket the chemical potential $\mu_0$, and then `_fermi_free_energy_setup` runs `diff_steps = 2` Newton refinement steps by invoking `_matrix_fermi` 3 times per SCF iteration (`3 * 30 = 90` dense $255 \times 255$ Cholesky solves in `static`, or ~`49` in `adaptive`).

Because the primal SCF root-finding loop executes entirely under `torch.no_grad()` and its fixed-point root $x^\star$ is theorem-backed by the subsequent static matrix-Fermi residual check $\|G_{\text{static}}(x^\star) - x^\star\|$, the spectral decomposition of the symmetric orthogonalized Hamiltonian $\tilde{H} = L^{-1} H L^{-\top} = V \operatorname{diag}(\varepsilon) V^\top$ is mathematically identical in value to the matrix-function series $\big(I + \exp((\tilde{H} - \mu I)/k_B T)\big)^{-1} = V \operatorname{diag}\!\big(f(\varepsilon_i)\big) V^\top$:

1. **Single Eigendecomposition per SCF Iteration (`_fermi_free_energy_setup`)**:
   - When `_current_primal_fermi_mode(eager_primal) == "spectral"`, `_fermi_free_energy_setup` diagonalizes `safe_spectral_base` once via `torch.linalg.eigh(safe_spectral_base)` (`evals, evecs`).
   - All `diff_steps` Newton chemical-potential updates are performed in 1D $O(N_{\text{orb}})$ vector arithmetic on `evals`:
     $$f_i = \frac{1}{2}\left(1 - \tanh\!\left(\frac{\varepsilon_i - \mu}{2 k_B T}\right)\right), \qquad \frac{\partial N_e}{\partial \mu} = \frac{1}{k_B T} \sum_i f_i (1 - f_i)$$
   - After the final Newton step, the Loewdin-transformed eigenvectors $U = L^{-\top} V$ (`l_inv.transpose(-1, -2) @ evecs`) are contracted once to form the final density matrix $D = U \operatorname{diag}(f_i) U^\top$.
2. **Direct Spectral Density Helper (`_matrix_fermi_spectral`)**:
   - Direct calls to `fermi_density_matrix(..., eager_primal="spectral")` outside `_fermi_free_energy_setup` also evaluate $D = U \operatorname{diag}(f(\varepsilon_i)) U^\top$ under `_require_detached_eager_primal`.
3. **Unchanged Differentiable & Validation Path**:
   - Post-solve root certification in `_solve_and_validate_lane_primal`, `evaluate_converged_root_batch`, and all first- and higher-order derivatives (`grad`, `jvp`, `jacfwd`, `jacrev`) continue to use the exact eigensolver-free static matrix-Fermi operator (`eager_primal_fermi=False`).
   - Verified across `26` targeted tests in `test/test_scf/test_invariant_spectral_density.py` (`test_detached_spectral_fermi_*`), including metallic/degenerate systems, multi-orbital noncontiguous padding, `float32`/`float64`, and strict rejection under `requires_grad=True`, `jvp`, `vmap`, and `torch.compile`.

### 4.3 Process-Isolated Parallel Physical-Root Preparation (`num_workers = 1, 2, 4`)

Because detached-primal root preparation is independent across conformers, `prepare_converged_root_batch`, `evaluate_converged_root_batch`, and `stream_batch_value_and_grad` now accept `num_workers: int = 1`:

- When `effective_workers = min(num_workers, batch_size) > 1`, detached lane inputs (`SCFInputs`, initial `SCFCharges`, `numbers`, solver settings) are serialized into immutable `_TensorValueSnapshot` trees and dispatched to a `multiprocessing` worker pool (using `"fork"` on Linux CPU when CUDA is not initialized for low-overhead copy-on-write page sharing, and `"spawn"` otherwise) initialized via `_init_primal_worker` (`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`, `torch.set_num_threads(1)`).
- Each worker solves and static-validates its assigned conformers (`_worker_solve_lane_task`) and reports its process PID, CPU time (`time.process_time()`), peak RSS (`VmHWM` / `ru_maxrss`), and proportional set size (`Pss` from `/proc/self/smaps_rollup`).
- `PreparedConvergedRootBatch.preparation_diagnostics` (`BatchPreparationDiagnostics`) records:
  - `wall_time_s`, `parent_cpu_time_s`, `worker_cpu_time_s`, `total_cpu_time_s`
  - `parent_peak_rss_mb`, `worker_peak_rss_sum_mb`, `process_tree_peak_rss_mb` (parent + sum of worker peaks), and `process_tree_pss_mb` (parent PSS + sum of worker PSS, accounting for shared read-only `libtorch` pages).

| Conformers $N$ | Workers | Root Prep Wall (`s`) | Prep Speedup vs. 1 Worker | Prep Speedup vs. Static 1 Worker | Total CPU Time (`s`) | Parent Peak RSS (`MiB`) | Worker Peak RSS Sum (`MiB`) | Prep Tree Peak RSS (`MiB`) | Prep Tree PSS (`MiB`) | End-to-End Total Wall (`s`) | Overall Peak Tree RSS (`MiB` / `GiB`) | Artifact |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `4` | `1` | `37.200` | `1.00x` | `10.47x` | `37.458` | `397.1` | `0.0` | `397.1` | `280.4` | `174.585` | `2,042.2` (`1.99 GiB`) | `integrated-spectral-streaming-n4.json` |
| `4` | `2` | `19.377` | **`1.92x`** | **`20.11x`** | `37.620` | `354.7` | `644.5` | `999.1` | `494.8` | `158.925` | `2,034.0` (`1.99 GiB`) | `parallel-spectral-streaming-n4-w2.json` |
| `4` | `4` | **`12.281`** | **`3.03x`** | **`31.72x`** | `37.889` | `349.1` | `1,285.5` | `1,634.6` | `571.7` | **`148.476`** | `2,028.7` (`1.98 GiB`) | `parallel-spectral-streaming-n4-w4.json` |
| `8` | `1` | `82.353` | `1.00x` | `10.33x` | `83.072` | `410.4` | `0.0` | `410.4` | `293.5` | `360.767` | `2,053.8` (`2.01 GiB`) | `integrated-spectral-streaming-n8.json` |
| `8` | `2` | `42.715` | **`1.93x`** | **`19.91x`** | `82.639` | `374.8` | `672.3` | `1,047.1` | `542.9` | `324.436` | `2,045.7` (`2.00 GiB`) | `parallel-spectral-streaming-n8-w2.json` |
| `8` | `4` | **`22.810`** | **`3.61x`** | **`37.28x`** | `82.406` | `371.5` | `1,340.4` | `1,711.9` | **`640.6`** | **`295.074`** | **`2,055.0` (`2.01 GiB`)** | `parallel-spectral-streaming-n8-w4.json` |

Because the worker pool terminates and releases worker memory (`1,711.9 MiB` sum RSS / `640.6 MiB` PSS at 4 workers) before streaming differentiation starts in the parent process (`2,055.0 MiB` peak RSS), **4-worker parallel root preparation does not increase overall process-tree peak RSS (`2.01 GiB`)**.

---

## 5. Part D — Complete Eight-Conformer Physical Batch Certification

### 5.1 Eight-Conformer `tmpda` Per-Lane Convergence and Serial Reference Parity (`0..7`)

Independent single-`System` serial references were computed in `benchmarks/results/e6c1-w2c11/serial-ref-n8-pos.json` (`adaptive` Fermi mode) and `benchmarks/results/e6c1-w2c11/serial-ref-n8-model.json` (`spectral` Fermi mode). Below is the per-conformer comparison across all 8 conformers (`0..7`, with conformers `4..7` newly certified against independent single-`System` serial evaluation):

| Conformer | SCF Iters | Certified Static $\|r\|_2$ | Certified Static $\|r\|_\infty$ | Energy $E$ (`Eh`) | Free Energy $F_{\text{el}}$ (`Eh`) | $\|\nabla_{\mathbf{R}} E\|_2$ (`Eh/Bohr`) | BiCGSTAB Iters / Residual | $\|\Delta x^\star\|_\infty$ (Spectral vs. Adaptive Serial) | $|\Delta E|$ (Spectral vs. Adaptive Serial) | $\|\Delta \nabla_{\mathbf{R}} E\|_\infty$ (Spectral vs. Adaptive Serial) | $\|\Delta \nabla_{\mathbf{R}} E\|_\infty$ (Adaptive Streaming vs. Serial) |
|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| `0` | `66` | `6.011914e-09` | `4.355904e-09` | `-29.758437462176` | `-0.001391311654` | `41.294736336` | `5` / `3.57e-13` | `2.343e-14` | `2.061e-13` | `1.075e-13` | `0.000e+00` |
| `1` | `96` | `6.416262e-09` | `2.739517e-09` | `-11.312642703836` | `-0.001886992802` | `66.294960215` | `7` / `9.31e-14` | `2.687e-14` | `5.329e-15` | `1.069e-13` | `0.000e+00` |
| `2` | `112` | `8.968167e-09` | `4.997459e-09` | `-21.692981421968` | `-0.002603711696` | `56.453158792` | `6` / `1.21e-13` | `3.508e-14` | `4.228e-13` | `1.920e-13` | `0.000e+00` |
| `3` | `72` | `6.274417e-09` | `3.417452e-09` | `-33.762321030730` | `-0.002096552317` | `43.288549130` | `7` / `1.86e-13` | `3.597e-14` | `2.203e-13` | `6.573e-14` | `0.000e+00` |
| **`4`** | `111` | `3.571365e-09` | `2.220020e-09` | `-32.870420886648` | `-0.002620547153` | `43.522869038` | `6` / `2.14e-14` | **`2.354e-14`** | **`2.842e-14`** | **`8.971e-14`** | **`0.000e+00`** |
| **`5`** | `115` | `3.960040e-09` | `1.538648e-09` | `-18.886485525779` | `-0.003211266273` | `56.659026782` | `6` / `7.68e-13` | **`3.897e-14`** | **`2.416e-13`** | **`1.315e-13`** | **`0.000e+00`** |
| **`6`** | `105` | `5.346041e-09` | `2.635201e-09` | `-16.611094756777` | `-0.003193014192` | `68.519360541` | `6` / `4.30e-13` | **`2.687e-14`** | **`3.482e-13`** | **`9.881e-14`** | **`0.000e+00`** |
| **`7`** | `78` | `6.422384e-09` | `3.205265e-09` | `-16.441186225726` | `-0.002717394442` | `61.541001103` | `6` / `7.67e-14` | **`4.152e-14`** | **`7.816e-14`** | **`5.862e-14`** | **`0.000e+00`** |

### 5.2 Shared `Model` Parameter Gradient Certification (`2,261` Floating Leaves)

Shared `Model` parameter gradients were certified across `N = 4` and `N = 8` `tmpda` conformers for both unit weights (`w = [1, ..., 1]`) and mixed-sign/zero weights (`w = [1.25, -0.75, 0.0, 0.5]`, verifying that lane 2 with `w_2 = 0.0` skips differentiation and contributes `0.0` while lane 1 with `w_1 = -0.75` scales its gradient negatively):

| Run / Artifact | $N$ | Chunk Size | Workers | Weights Mode | Total Leaves / Connected / Nonzero / Zero / Disconnected | Root Prep (`s`) | Stream Diff (`s`) | Total (`s`) | Peak Tree RSS (`MiB` / `GiB`) | Max $\|\Delta \nabla_{\theta} L\|_\infty$ vs. Serial Ref | Rel $L_2$ $\Delta \nabla_{\theta} L$ vs. Serial Ref | Max $\|\Delta \nabla_{\mathbf{R}} L\|_\infty$ vs. Serial Ref |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `model-grad-mixed-spectral-streaming-n4-c1.json` | `4` | `1` | `1` | `[1.25, -0.75, 0.0, 0.5]` | `2261 / 77 / 73 / 4 / 2184` | `38.136` | `112.957` | `151.334` | `2,293.7` (`2.24 GiB`) | **`0.000e+00`** | **`0.000e+00`** | **`0.000e+00`** |
| `model-grad-mixed-spectral-streaming-n4-c2.json` | `4` | `2` | `1` | `[1.25, -0.75, 0.0, 0.5]` | `2261 / 77 / 73 / 4 / 2184` | `37.130` | `116.294` | `153.650` | `3,884.8` (`3.79 GiB`) | **`1.013e-13`** | **`9.717e-16`** | **`4.163e-14`** |
| `model-grad-spectral-streaming-n4-c1.json` | `4` | `1` | `1` | `unit` | `2261 / 77 / 73 / 4 / 2184` | `37.685` | `150.361` | `188.282` | `2,295.3` (`2.24 GiB`) | **`0.000e+00`** | **`0.000e+00`** | **`0.000e+00`** |
| `model-grad-spectral-streaming-n4-c2.json` | `4` | `2` | `1` | `unit` | `2261 / 77 / 73 / 4 / 2184` | `36.354` | `155.359` | `191.933` | `3,887.5` (`3.80 GiB`) | **`1.990e-13`** | **`3.248e-16`** | **`4.974e-14`** |
| `model-grad-spectral-streaming-n8-c1-w1.json` | `8` | `1` | `1` | `unit` | `2261 / 77 / 73 / 4 / 2184` | `79.451` | `291.671` | `371.372` | **`2,316.4` (`2.26 GiB`)** | **`0.000e+00`** | **`0.000e+00`** | **`0.000e+00`** |
| `model-grad-spectral-streaming-n8-c1-w4.json` | `8` | `1` | `4` | `unit` | `2261 / 77 / 73 / 4 / 2184` | **`23.266`** | `299.326` | **`322.848`** | **`2,307.9` (`2.25 GiB`)** | **`0.000e+00`** | **`0.000e+00`** | **`0.000e+00`** |

---

## 6. Part E — Matched Legacy `Calculator` Audit and Benchmark

### 6.1 Eight-Point Architectural Audit: Legacy `Calculator(batch_mode=2)` vs. Functional Converged-Root Route

1. **Exact Code Path**:
   - *Legacy Packed*: `Calculator.singlepoint` (`src/dxtb/_src/calculators/singlepoint.py`) dispatches `batch_mode=2` (`conformer` mode) to `solve_legacy` (`src/dxtb/_src/scf/solver.py`) $\to$ `SelfConsistentFieldFull` (`src/dxtb/_src/scf/scf.py`).
   - *Functional Converged-Root*: `prepare_converged_root_batch` solves each conformer's fixed-point root via `_solve_map_anderson` on `scf_map`, certifies the root with the static matrix-Fermi map, and differentiates at the converged root via `evaluate_converged_root_batch` or `stream_batch_value_and_grad` using `apply_implicit_root`.
2. **Diagonalization & Fermi Occupation**:
   - *Legacy Packed*: Calls `storch.eighb` on the symmetrized generalised eigenproblem and computes 1D eigenvalue Fermi occupations in `fermi.get_occ` (`src/dxtb/_src/wavefunction/filling.py`).
   - *Functional Converged-Root*: Primal SCF under `eager_primal_fermi="spectral"` now also uses `torch.linalg.eigh` + 1D Newton chemical-potential refinement per SCF step (`7.54 s` for `tmpda[0]`), while post-solve certification and implicit differentiation use the eigensolver-free 90-solve static matrix-Fermi operator (`_matrix_fermi_static`) so that derivatives never differentiate through `eigh` or degenerate eigenvalue gaps.
3. **Mixer Algorithm**:
   - *Legacy Packed*: Uses the stateful `Anderson` mixer (`src/dxtb/_src/scf/mixer/anderson.py`) which mutates history buffers in place and culls converged batch entries.
   - *Functional Converged-Root*: Uses the pure functional `_solve_map_anderson` (`src/dxtb/_src/scf/functional/anderson.py`) on immutable `AndersonState` tensors.
4. **Convergence Criteria & SCP Target (Critical Historical Mismatch Identified)**:
   - *Legacy Historical W2 Baseline*: `benchmarks/baseline/workloads.py` (`OPTS`) omitted `scp_mode`, causing legacy `Calculator` to default to `defaults.SCP_MODE = labels.SCP_MODE_POTENTIAL` (`"potential"`), whereas functional SCF defaults to `scp_mode = "charge"`. Under `scp_mode = "potential"`, legacy `Calculator` tests convergence on potential residuals $\Delta V$ and takes `[77, 82, 124, 205]` max iterations for $N = 1, 2, 4, 8$.
   - *Matched Legacy (`scp_mode = "charge"`)*: When `scp_mode = "charge"` is explicitly matched, legacy `SelfConsistentFieldFull` converges in `[66, 96, 112, 115]` max iterations for $N = 1, 2, 4, 8$ — **matching the exact per-conformer iteration counts (`[66, 96, 112, 72, 111, 115, 105, 78]`) of the functional solver**.
5. **Batch Culling**:
   - *Legacy Packed*: `SelfConsistentFieldFull.iterate` calls `self._data.cull(conv)` after each iteration, slicing active conformers out of the batch tensor so already-converged conformers stop iterating.
   - *Functional Converged-Root*: Solves each conformer independently in `prepare_converged_root_batch`, naturally stopping each conformer at its own iteration count (`[66, 96, 112, 72, 111, 115, 105, 78]`, average `94.4` iterations vs. `115` max).
6. **Integral & Interaction Caching**:
   - *Legacy Packed*: Stores mutable `_positions_batch`, `_cache`, and packed `ClassicalCache` / `InteractionCache` on `Calculator`.
   - *Functional Converged-Root*: Uses immutable `ExactSystemCore` setup `Node`s and per-lane `SCFInputs` without persistent mutable caches.
7. **Differentiation Mechanism (Primary Source of Remaining Runtime Difference)**:
   - *Legacy Packed (`full` mode)*: Records the entire forward SCF iteration loop (`66..115` iterations of `storch.eighb`, Fock builds, and Anderson mixing) on the PyTorch autograd tape and backpropagates through the unrolled trajectory. Because every step's intermediates are retained on the autograd tape, `rss_after_forward` grows linearly with $N \times \text{iterations}$, reaching **`4,529.5 MiB` (`4.42 GiB`)** at $N=8$ (`scp_mode="charge"`) and **`5,142.4 MiB` (`5.02 GiB`)** (`scp_mode="potential"`).
   - *Functional Converged-Root*: Runs primal SCF completely detached (`torch.no_grad()`), retaining only the converged root $x^\star$ (`rss_after_root = 376.4 MiB` at $N=8$), and computes exact implicit fixed-point derivatives via matrix-free BiCGSTAB (`5..7` iterations per conformer). Each BiCGSTAB iteration evaluates the VJP of `scf_map`, which traverses the 90-solve static matrix-Fermi operator (`~4.9 s` per VJP evaluation, `~34 s` per conformer).
8. **Transform Compatibility**:
   - *Legacy Packed*: Incompatible with `torch.func.vmap`, `jvp`, `jacfwd`, and degenerate/3rd-order spectral differentiation due to in-place mutations, batch culling, and `storch.eighb` backward singularities.
   - *Functional Converged-Root*: `evaluate_converged_root_batch` is compatible with `grad`, `jvp`, `jacfwd`, `jacrev`, `vmap`, higher-order derivatives, and `torch.compile(backend="eager")`; `stream_batch_value_and_grad` provides $O(1)$-memory first-order differentiation across conformers and shared `Model` parameters.

### 6.2 Matched Legacy vs. Functional Benchmark Table ($N = 1, 2, 4, 8$ `tmpda` Conformers)

| Route | SCP Mode | $N$ | Max / Per-Lane SCF Iters | Forward / Root Prep (`s`) | Backward / Diff (`s`) | Total Wall (`s`) | RSS After Fwd / Root (`MiB`) | Peak Tree RSS (`MiB` / `GiB`) | Max $|\Delta E|$ vs. Functional Ref (`Eh`) | $\|\nabla_{\mathbf{R}} E\|_2$ Diff vs. Functional Ref (`Eh/Bohr`) | Artifact |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| Legacy Packed (`batch_mode=2`) | `potential` (unmatched default) | `1` | `77` | `2.085` | `1.752` | `4.086` | `820.8` | `826.8` (`0.81 GiB`) | `3.553e-14` | `2.493e-08` | `legacy-packed-potential-n1.json` |
| Legacy Packed (`batch_mode=2`) | `potential` (unmatched default) | `2` | `82` | `3.787` | `2.900` | `6.881` | `1,396.1` | `1,405.1` (`1.37 GiB`) | `1.652e-13` | `2.055e-08` | `legacy-packed-potential-n2.json` |
| Legacy Packed (`batch_mode=2`) | `potential` (unmatched default) | `4` | `124` | `7.485` | `6.383` | `14.079` | `2,282.4` | `2,298.4` (`2.24 GiB`) | `1.652e-13` | `6.018e-09` | `legacy-packed-potential-n4.json` |
| Legacy Packed (`batch_mode=2`) | `potential` (unmatched default) | `8` | `205` | `17.693` | `17.346` | `35.269` | `5,142.4` | `5,185.3` (`5.06 GiB`) | `2.522e-13` | `1.220e-07` | `legacy-packed-potential-n8.json` |
| **Matched Legacy Packed (`batch_mode=2`)** | **`charge` (matched)** | `1` | `66` | `1.831` | `1.307` | `3.357` | `758.1` | `765.1` (`0.75 GiB`) | `2.487e-14` | `4.015e-12` | `legacy-packed-charge-n1.json` |
| **Matched Legacy Packed (`batch_mode=2`)** | **`charge` (matched)** | `2` | `96` | `3.903` | `3.311` | `7.408` | `1,304.9` | `1,309.9` (`1.28 GiB`) | `1.794e-13` | `8.924e-12` | `legacy-packed-charge-n2.json` |
| **Matched Legacy Packed (`batch_mode=2`)** | **`charge` (matched)** | `4` | `112` | `7.397` | `7.672` | `15.307` | `2,248.5` | `2,257.5` (`2.20 GiB`) | `1.794e-13` | `4.501e-10` | `legacy-packed-charge-n4.json` |
| **Matched Legacy Packed (`batch_mode=2`)** | **`charge` (matched)** | `8` | `115` | `14.736` | `14.028` | `29.005` | `4,529.5` | **`4,541.5` (`4.43 GiB`)** | `2.061e-13` | `3.709e-10` | `legacy-packed-charge-n8.json` |
| Functional Serial Single-`System` (`adaptive`) | `charge` | `8` | `[66..115]` | `511.753` | `267.648` | `779.409` | — | `2,317.8` (`2.26 GiB`) | `0.000e+00` | `0.000e+00` | `serial-ref-n8-pos.json` |
| Functional Serial Single-`System` (`spectral` + Model) | `charge` | `8` | `[66..115]` | `76.669` | `290.148` | `366.827` | — | `2,653.8` (`2.59 GiB`) | `4.228e-13` | `3.695e-13` | `serial-ref-n8-model.json` |
| **Functional Spectral + Streaming (`1` worker)** | `charge` | `8` | `[66..115]` | **`82.377`** | `278.157` | **`360.767`** | `376.4` | **`2,053.8` (`2.01 GiB`)** | `4.228e-13` | `3.695e-13` | `integrated-spectral-streaming-n8.json` |
| **Functional Spectral + Streaming (`4` workers)** | `charge` | `8` | `[66..115]` | **`22.841`** | `272.010` | **`295.074`** | `383.7` | **`2,055.0` (`2.01 GiB`)** | `4.228e-13` | `3.695e-13` | `parallel-spectral-streaming-n8-w4.json` |

---

## 7. Part F — Architectural Assessment and Public Batch API Readiness

### 7.1 Performance Outcome Classification (Section 14)

Under the classification criteria of Section 14, the integrated W2c11 architecture is classified as:

> **Outcome 2 — Memory-Qualified and Substantially Improved Functional Batch Architecture, with Explicit Differentiation Runtime Trade-Off Relative to Unrolled Legacy `eighb`**

**Quantitative Justification**:
1. **Memory Qualification (Exceeds Legacy Packed `Calculator`)**:
   - Across $N = 1, 2, 4, 8$ `tmpda` conformers, `stream_batch_value_and_grad(..., chunk_size=1)` holds peak RSS completely flat at **`2.01 GiB` (`2,053.8 MiB`)** for position gradients and **`2.25 GiB` (`2,307.9 MiB`)** for simultaneous position + `2,261`-leaf `Model` parameter gradients.
   - At $N = 8$, this is **54.8% lower memory (`2.21x` reduction)** than matched legacy packed `Calculator(batch_mode=2, scp_mode="charge")` (`4,541.5 MiB` / `4.43 GiB`) and **60.4% lower memory (`2.52x` reduction)** than the historical `scp_mode="potential"` legacy packed run (`5,185.3 MiB` / `5.06 GiB`), whose memory grows linearly with conformer count and SCF iteration count due to unrolled trajectory storage.
2. **Primal Root-Preparation Speedup (Gap to Legacy Nearly Closed)**:
   - Combining `eager_primal_fermi="spectral"` with process-isolated parallel preparation reduces 8-conformer `tmpda` root preparation from `850.47 s` (W2c10 static) and `536.25 s` (W2c11 adaptive) to **`82.38 s` (1 worker, `10.32x` speedup)**, **`42.75 s` (2 workers, `19.89x` speedup)**, and **`22.84 s` (4 workers, `37.23x` speedup)** — within `1.55x` of legacy packed forward SCF (`14.74 s`).
3. **Remaining Differentiation Runtime Gap (`272.01 s` vs. `14.03 s` at $N=8$)**:
   - After reducing 8-conformer root preparation to `22.84 s`, **92.3%** of the `295.07 s` end-to-end runtime (`272.01 s`, or `~34.0 s` per conformer) is spent in the 8 implicit response solves (`5..7` BiCGSTAB iterations per conformer, each differentiating through the 90-solve static matrix-Fermi operator).
   - This cost is the direct mathematical price of **eigensolver-free, degeneracy-safe implicit differentiation** on the invariant matrix-Fermi fixed-point map versus legacy `Calculator`'s unrolled differentiation through `storch.eighb`.

### 7.2 Public Batch API Readiness Proposal (Section 15)

The internal converged-root and streaming architecture in `batch_implicit.py` and `batch_streaming.py` is functionally complete, failure-hardened, and ready for exposure as an **explicit public conformer-batch API** (distinct from `torch.func.vmap(System.singlepoint)`), subject to project-owner sign-off:

1. **Proposed Public Entry Points**:
   - `prepare_converged_root_batch(system, positions, *, chrg=_UNSET, spin=_UNSET, field=_UNSET, field_grad=_UNSET, strict=None, eager_primal_fermi="spectral", num_workers=1) -> PreparedConvergedRootBatch`
     - Solves detached primal SCF roots for a `(B, nat, 3)` conformer batch (or `PaddedExactSystemCore`) and certifies every root with the exact static matrix-Fermi map.
   - `evaluate_converged_root_batch(system_or_prepared, positions=_UNSET, *, model=None, chrg=_UNSET, spin=_UNSET, field=_UNSET, field_grad=_UNSET, strict=None, eager_primal_fermi="spectral", response_diagnostics=None) -> Result`
     - Evaluates differentiable batched `Result` outputs at validated physical roots; supports `grad`, `jvp`, `jacfwd`, `jacrev`, `vmap`, higher-order derivatives, and `torch.compile(backend="eager")`.
   - `stream_batch_value_and_grad(prepared_or_system, model, positions, *, chrg=None, spin=None, field=None, field_grad=None, weights=None, chunk_size=1, strict=None, eager_primal_fermi="spectral", num_workers=1) -> StreamingBatchValueAndGrad`
     - Evaluates and differentiates separable weighted batch objectives $L = \sum_{b=0}^{B-1} w_b E_b$ in bounded $O(\text{chunk\_size})$ memory across `positions`, `chrg`, `spin`, `field`, `field_grad`, and all shared `Model` parameters, returning explicit per-lane `SCFStatus`, response solver diagnostics (`response_converged`, `response_status`, `response_iterations`, `response_residual`), and 5-category `failure_category`.
2. **Relationship to `System.singlepoint()` and W2 Acceptance**:
   - `System.singlepoint()` remains the pure single-`System` core entry point.
   - Because `torch.func.vmap(System.singlepoint)` traces a single homogeneous fixed-trip computation graph simultaneously across all batch lanes, literal `vmap(System.singlepoint)` cannot perform per-conformer early stopping, detached spectral root preparation, or chunk-streamed autograd graph release.
   - Accordingly, **W2 remains OPEN** until the project owner approves replacing or supplementing the literal `vmap(System.singlepoint)` W2 batch benchmark criterion with the explicit converged-root / streaming batch API.

---

## 8. Multi-Environment Regression Gate and Independent Review

### 8.1 Multi-Environment Regression Gate (Part G, Section 17)

1. **Five-Environment Focused SCF / Implicit / Spectral Suite (`274/274` non-slow passed per environment; `277/277` with slow on `py3.11-torch2.8.0`)**:
   - Executed across `py3.10-torch2.7.0` (`274 passed in 253.04s`), `py3.11-torch2.8.0` (`274 passed in 245.56s` + `3 passed in 521.17s` slow in `test_functional_root_derivatives.py`), `py3.12-torch2.11.0` (`274 passed in 307.61s`), `py3.12-torch2.13.0` (`274 passed in 320.92s`), and `py3.14-torch2.14.1` (`274 passed in 236.37s`) over:
     - `test/test_scf/test_e6c1_w2c9_batch_implicit.py` (`38` tests, including the 4 new W2c11 comprehensive test suites),
     - `test/test_scf/test_invariant_spectral_density.py` (`94` tests, including the `26` new W2c11 detached-spectral Fermi tests),
     - `test/test_scf/test_invariant_free_energy.py`,
     - `test/test_scf/test_e6c1_w2c5b_shift_contract.py`,
     - `test/test_scf/test_e6c1_w2c6_tanh_solve.py`,
     - `test/test_scf/test_functional_implicit_root.py`,
     - `test/test_scf/test_e6c1_w2c1_mapped_implicit.py`,
     - `test/test_scf/test_functional_root_derivatives.py`.
2. **Two-Environment 15-Module Primary Regression Gate (`py3.11-torch2.8.0` and `py3.14-torch2.14.1`)**:
   - Executed across `test_e7a_padding_safety.py`, `test_e6bcompat_differential.py`, `test_e6bcompat_policy.py`, `test_e6a0_system_vmap.py`, `test_e6a1_conformer.py`, `test_e6a1a_gfn0_spin_validation.py`, `test_e6a2_public_status.py`, `test_e6a3_component_vmap.py`, `test_e6a4_transform_acceptance.py`, `test_e6bimpl0_padded_integrals.py`, `test_e6bimpl1a_padded_spectral.py`, `test_e6bimpl2a_gfn2_canonical.py`, `test_e6bimpl2b_gfn2_scf.py`, `test_e6bimpl3_legacy_composition.py`, and `test_e6c1_w2c2a_configuration.py` (`261/261` passed on both environments).

### 8.2 Independent Code Review (`Luna / high`)

- **Requested reviewer**: `Luna / high` (`research` subagent `439f9f3c-adeb-4736-bbef-f68f5f219010`)
- **Actual model identity**: `not exposed by runtime`
- **Pass 1 Findings (`0 BLOCKER`, `3 MAJOR`, `2 MINOR`) — All Resolved**:
  1. `MAJOR 1`: Added `num_workers: int = 1` and `eager_primal_fermi: bool | str = False` forwarding to `stream_batch_value_and_grad` and `evaluate_converged_root_batch` when invoked with a raw `System`, and added direct `System`-input parallel tests in `test_w2c11_parallel_root_preparation_parity_and_failure_semantics`.
  2. `MAJOR 2`: Enforced `effective_strict = _resolve_strictness(model.config.scf, strict)` on post-differentiation failures (`RESPONSE_FAILURE = 9` and `NONFINITE_DERIVATIVE = 10`) in `stream_batch_value_and_grad`, masked unconverged BiCGSTAB solutions (`status != STATUS_CONVERGED`) with `NaN` in `ResponseDiagnosticCollector.solve`, and added strict/non-strict post-differentiation tests in `test_w2c11_streaming_failure_classification_all_input_families_and_causes`.
  3. `MAJOR 3`: Generated `benchmarks/results/e6c1-w2c11/serial-ref-n4-model.json`, refreshed `model-grad-spectral-streaming-n4-c{1,2}.json` with full `parity_vs_reference` metrics (`model_param_grad_max_abs_diff = 0.0` and `1.9895e-13`), and aligned all enum names, helper names, and mixed-weights descriptions in the documentation.
  4. `MINOR 1 & 2`: Added `l_inv` to `_require_detached_eager_primal` in `_fermi_free_energy_setup`, expanded `test_detached_spectral_fermi_wide_spectra_saturation_s01b_and_nonfinite` to cover `jvp`, `vmap`, forward-AD on `l_inv`, and `float32`, and sampled `parent_peak_rss_mb, parent_pss_mb` inside the active worker pool context in `prepare_converged_root_batch`.
- **Pass 2 Final Verdict**: **NO FINDING (`0 BLOCKER`, `0 MAJOR`, `0 MINOR`)**; all 15 review checklist items **PASS**.

---

## 9. Final Verdict Matrix (Part J)

| Area | Status | Evidence |
|---|---|---|
| Streaming failure classification | **CLOSED** | 5-category `classify_batch_failure_category` (`"success"`, `"primal_scf_failure"`, `"fermi_failure"`, `"response_failure"`, `"nonfinite_derivative"`), `BatchRootFailureCode.NONFINITE_DERIVATIVE = 10`, `RESPONSE_NOT_RUN = -1`, call-local `ResponseDiagnosticCollector` / `BatchResponseDiagnostics`, per-lane shared-input expansion and multi-lane chunk parameter fallback isolation in `batch_implicit.py` and `batch_streaming.py`; verified in `test_w2c11_streaming_failure_classification_all_input_families_and_causes` |
| Streaming API contracts | **CLOSED** | `PreparedConvergedRootBatch | System` input support, nonfinite `weights` rejection (`ValueError`), zero/negative weight semantics, parameter deduplication, connected-zero vs. disconnected leaf classification, caller input hook preservation, mutation rejection, transform/compile guards, and bounded-repetition stability in `batch_streaming.py`; verified in `test_w2c11_streaming_api_contracts_aliasing_hooks_and_long_lived_execution` |
| Eager-adaptive + streaming integration | **CLOSED** | Thread-local `eager_primal_fermi_scope` with exception-safe restoration and unconditional static matrix-Fermi root validation; process-isolated 4-way A/B/C/D benchmarks across $N = 1, 2, 4, 8$ in `benchmarks/results/e6c1-w2c11/integrated-{static,adaptive}-{simultaneous,streaming}-n*.json` (`802.38 s`, `1.99 GiB` peak RSS at $N=8$ for `D` vs. `1,100.71 s` for `B`) |
| Root-preparation optimization | **CLOSED** | Phase profiling (`root-profile-{static,adaptive,spectral}.json`), detached-primal spectral Fermi mode (`eager_primal_fermi="spectral"`, **9.77x** single-conformer speedup `73.69 s` $\to$ `7.54 s` with `2.34e-14` max root difference), and process-isolated parallel worker pool (`num_workers = 1, 2, 4`, **3.61x** parallel speedup `82.35 s` $\to$ `22.81 s` on $N=8$, **37.28x** faster than W2c10 static `850.47 s`, `640.6 MiB` prep tree PSS) |
| 8-conformer position gradients | **CLOSED** | Certified across all 8 `tmpda` conformers (`0..7`, specifically including `4..7`) against independent single-`System` serial references (`serial-ref-n8-pos.json`, `serial-ref-n8-model.json`): `0.000e+00` max gradient diff for `adaptive` streaming (`802.38 s`, `2,033.3 MiB` RSS) and `<= 1.920e-13 Eh/Bohr` max gradient diff for `spectral` streaming (`360.77 s` 1-worker / `295.07 s` 4-worker, `2,055.0 MiB` overall peak tree RSS) |
| 8-conformer `Model` parameter gradients | **CLOSED** | Certified across all `2,261` floating `Model` leaves (`77` connected: `73` nonzero, `4` connected zero; `2,184` disconnected) for $N=4$ (`chunk_size = 1, 2`, unit and `[1.25, -0.75, 0.0, 0.5]` weights) and $N=8$ (`num_workers = 1, 4`, `371.37 s` / `322.85 s`, `2,316.4 MiB` / `2,307.9 MiB` peak tree RSS) with `0.000e+00` (`c=1`) and `<= 1.990e-13` (`c=2`) max abs leaf gradient difference vs. independent serial references |
| Matched legacy comparison | **CLOSED** | 8-point audit and benchmarks (`legacy-packed-{potential,charge}-n{1,2,4,8}.json`) identifying the historical `scp_mode="potential"` default mismatch (`[77, 82, 124, 205]` iters) vs. matched `scp_mode="charge"` (`[66, 96, 112, 115]` iters, `29.01 s`, `4,541.5 MiB` RSS at $N=8$); functional spectral+streaming uses **54.8% less memory (`2.01 GiB` vs. `4.43 GiB`)** at $N=8$ |
| Public batch API readiness | **READY FOR OWNER SIGN-OFF** | Explicit three-function batch API (`prepare_converged_root_batch`, `evaluate_converged_root_batch`, `stream_batch_value_and_grad`) specified with complete failure, memory, and transform contracts in Section 7.2 without altering `System.singlepoint()` |
| W2 status | **OPEN (API/contract decision required)** | Functional converged-root + streaming architecture is memory-qualified (`2.01 GiB` at $N=8$, `2.21x` below legacy packed) and `3.73x` faster than W2c10 (`295.07 s` vs. `1,100.71 s`), but literal `torch.func.vmap(System.singlepoint)` remains the formal W2 gate until project-owner sign-off |
| W3 status | **OPEN** | Mixed-`System` padded batching performance certification remains downstream of W2 sign-off |
| E6broute status | **BLOCKED** | Blocked on W2/W3 sign-off and production batch route migration |
