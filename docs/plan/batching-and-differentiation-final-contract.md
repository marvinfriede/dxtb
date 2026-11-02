# Batching, Differentiation, and Calculator Lifecycle Final Contract

## 1. Executive Summary

This document defines the final architectural contract for batching, differentiation, and `Calculator` lifecycle management in `dxtb` on `claude/tender-dijkstra-cda42s`, completing the post-S13.1 `Calculator` refactoring:

1. **Coexisting Batching Interfaces**:
   - Single-system functional evaluation (`Model.setup(numbers)` -> `System.singlepoint`, `System.energy`, `System.forces`, etc.).
   - Explicit batched `Calculator` evaluation (`numbers` shaped `(B, Nmax)`, `positions` shaped `(B, Nmax, 3)`, supporting `batch_mode=1` padded heterogeneous batches and `batch_mode=2` unpadded conformer batches) backed by `evaluate_calculator_packed_singlepoint` (`E6broute`).
   - Functional `torch.vmap` over `System` (`torch.vmap(system.singlepoint)`, `torch.vmap(system.energy)`, `torch.vmap(system.forces)`).
2. **Shared and Per-Lane Electric Perturbations**:
   - Batched `Calculator` and `evaluate_packed_systems` accept shared `(3,)` / `(3, 3)` tensors, per-lane `(B, 3)` / `(B, 3, 3)` tensors, and length-`B` sequences `Sequence[Tensor | None]` for `field` and `field_grad`.
   - Ambiguous shapes (`(3, 3)` for `field`, `(3, 3, 3)` for `field_grad`, or mismatched sequence lengths) raise a clear `ValueError`.
3. **Complete Removal of `grad_mode` and Single Authoritative AD Force Definition**:
   - The public `grad_mode` selector and all its dispatch branches (`"autograd"`, `"backward"`, `"functorch"`, `"row"`) are removed from `AutogradCalculator.forces`, `GetPropertiesMixin.get_forces`, and `AutogradCalculator.hessian`.
   - Passing `grad_mode` by keyword or as a 4th positional argument to `forces` / `get_forces` raises `TypeError`.
   - `AutogradCalculator.forces` uses one authoritative reverse-mode AD definition: `torch.autograd.grad(e if e.ndim == 0 else e.sum(), positions, create_graph=create_graph, retain_graph=create_graph if retain_graph is None else retain_graph)`, negated and masked to zero on inactive padded atom rows (`self.numbers == 0`) when `self.ihelp.batch_mode > 0`, without mutating `positions.grad`.
4. **Stateless Call-Local Setup Lifecycle**:
   - Persistent setup-graph bookkeeping and memoization (`_param_state_signature`, `_register_param_backward_hooks`, `_clear_param_backward_hooks`, `_system_graph_consumed`, `_model_param_sig`, `_ensure_system_current`, and `_lane_systems_memo`) are removed from `BaseCalculator` and `build_calculator_lane_systems`.
   - Call-local single-`System` (`_prepare_eval_system`) and per-lane (`build_calculator_lane_systems`, with intra-call composition deduplication) setups are constructed cleanly from `self.model.par` and `self.dtype`, supporting all 13 grad/no-grad/inference-mode, repeated differentiation, parameter mutation/replacement, `type(dtype)`, and `reset()` transitions.
5. **Component Ownership Isolation and Lazy `Integrals` Adapter**:
   - Caller-owned `Classical` and `Interaction` instances passed to `Calculator` are shallow-copied when building call-local or per-lane setups so user objects are never mutated in place (`numbers`, `label`) while preserving `exact_core` eligibility.
   - `BaseCalculator.integrals` is a lazy compatibility property (`_integrals: Integrals | None = None`) that does not instantiate `DriverManager` or `Integrals` unless `calc.integrals` or `write_integrals_to_store=True` is explicitly accessed.

---

## 2. Coexisting Batching Interfaces

`dxtb` supports three complementary evaluation interfaces that share the same underlying single-`System` functional physics core:

| Interface | Input Shapes | Execution Route | Primary Use Case |
|---|---|---|---|
| **Single `System`** | `numbers: (N,)`, `positions: (N, 3)` | `System.singlepoint` -> `_singlepoint_functional` (`ExactSystemCore`) or `_singlepoint_eager` | Composable functional workflows, `torch.func` transforms (`grad`, `jacrev`, `jvp`, `jacfwd`, `vmap`), and parameter training |
| **Batched `Calculator`** (`batch_mode=1` or `batch_mode=2`) | `numbers: (B, Nmax)`, `positions: (B, Nmax, 3)` | `Calculator.singlepoint` -> `evaluate_calculator_packed_singlepoint` (`E6broute`: `MAPPED` canonical padded buckets + `LEGACY_COMPOSED` fallback) | Heterogeneous padded molecule batches (`batch_mode=1`, including trailing and non-contiguous zero padding) and explicit conformer batches (`batch_mode=2`) |
| **Functional `torch.vmap`** | `system = model.setup(numbers)` (`(N,)`), `positions: (B, N, 3)` | `torch.vmap(system.singlepoint)`, `torch.vmap(system.energy)`, `torch.vmap(system.forces)` | Vectorized same-composition conformer evaluation and composed `torch.func` transforms |

Within a single `evaluate_calculator_packed_singlepoint` call, `build_calculator_lane_systems(calc)` deduplicates identical unpadded lane compositions in a call-local dictionary so conformer batches (`batch_mode=2`) and repeated compositions in `batch_mode=1` invoke `Model.setup(..., batch_mode=0)` only once per unique composition per call.

---

## 3. Shared and Per-Lane Electric Perturbations (`field` and `field_grad`)

Both batched `Calculator` methods (`singlepoint`, `energy`, `forces`, `dipole`, `quadrupole`, etc.) and `evaluate_packed_systems` normalize electric perturbations via `_normalize_packed_fields`:

- **`field`**:
  - `None`: no external electric field (or falls back to `calc._field_default` on `Calculator`).
  - Shared `Tensor` of shape `(3,)`: broadcast across all `B` lanes while preserving autograd graph connectivity to the shared tensor.
  - Per-lane `Tensor` of shape `(B, 3)` (when `B != 3`; for `B == 3`, a `(3, 3)` tensor is rejected as ambiguous with `field_grad`—pass `tuple(field)` of three `(3,)` tensors instead): sliced per lane `field[i]` while preserving autograd graph connectivity back to the `(B, 3)` tensor.
  - `Sequence[Tensor | None]` of length `B`: per-lane optional `(3,)` tensors.
- **`field_grad`**:
  - `None`: no external field gradient (or falls back to `calc._field_grad_default` on `Calculator`).
  - Shared `Tensor` of shape `(3, 3)`: broadcast across all `B` lanes.
  - Per-lane `Tensor` of shape `(B, 3, 3)` (when `B != 3`; for `B == 3`, a `(3, 3, 3)` tensor is rejected as ambiguous—pass `tuple(field_grad)` of three `(3, 3)` tensors instead): sliced per lane `field_grad[i]`.
  - `Sequence[Tensor | None]` of length `B`: per-lane optional `(3, 3)` tensors.

In `evaluate_packed_systems`, `MAPPED` bucket grouping includes `(lane_fields[i] is not None, lane_field_grads[i] is not None)` in the bucket key and passes either a shared tensor or a stacked `(B_bucket, ...)` tensor to `evaluate_padded_systems`.

---

## 4. Authoritative Differentiation Contract and Removal of `grad_mode`

### 4.1 Removal of `grad_mode`

- `grad_mode` is removed from `AutogradCalculator.forces`, `GetPropertiesMixin.get_forces`, and `AutogradCalculator.hessian`.
- `AutogradCalculator.forces(self, positions, chrg=None, spin=None, *, create_graph=False, retain_graph=None, **kwargs)` and `GetPropertiesMixin.get_forces(self, positions, chrg=None, spin=None, **kwargs)` enforce keyword-only arguments after `spin`, rejecting any 4th positional argument with `TypeError`.
- `reject_removed_store_kwargs` in `src/dxtb/_src/calculators/types/base.py` explicitly rejects `"grad_mode"` with a clear `TypeError` across all `Calculator` entry points (`forces`, `get_forces`, `hessian`, `calculate`, `get_property`, `singlepoint`, `energy`, etc.).

### 4.2 Single Authoritative AD Force Implementation

`AutogradCalculator.forces` computes nuclear forces via a single reverse-mode AD path:

```python
e = self.energy(positions, chrg, spin, **kwargs)
e_scalar = e if e.ndim == 0 else e.sum()
eff_retain = create_graph if retain_graph is None else retain_graph
(deriv,) = torch.autograd.grad(
    e_scalar,
    positions,
    create_graph=create_graph,
    retain_graph=eff_retain,
)
forces = -deriv
if self.ihelp.batch_mode > 0:
    active_mask = (self.numbers != 0).unsqueeze(-1)
    forces = torch.where(active_mask, forces, torch.zeros_like(forces))
return forces
```

Key invariants:
- Never writes to or mutates `positions.grad`.
- Sums independent batch energies (`e.sum()`) when `e.ndim > 0`, producing exact per-lane gradients with respect to `(B, Nmax, 3)` `positions`.
- Masks inactive padded atom rows (`self.numbers == 0`) to exact zeros using `torch.where` (preserving differentiability for higher-order derivatives).
- Defaults `retain_graph` to `create_graph` when `retain_graph is None`, so `calc.forces(pos, create_graph=True)` preserves the graph needed to differentiate `forces` again (e.g., for Hessian rows or force-matching losses).

### 4.3 Audit of Other Differentiation Switches

| Parameter / Option | Scope | Status | Rationale |
|---|---|---|---|
| `grad_mode` | `Calculator.forces`, `get_forces`, `hessian` | **REMOVED** | Redundant and error-prone (`"backward"` mutated `.grad` and failed on batches; `"row"` / `"functorch"` duplicated AD). |
| `scf_mode` (`UNROLLED` vs `IMPLICIT` / `"full"` vs `"implicit"`) | `ConfigSCF` / S13 SCF core | **RETAINED (Authoritative S13)** | Controls physical SCF fixed-point differentiation strategy (unrolled trajectory vs implicit response equation). |
| `use_functorch` | `AutogradCalculator` higher-order properties (`hessian`, `dipole_deriv`, `polarizability`, `pol_deriv`, `hyperpolarizability`) | **RETAINED (`AutogradCalculator` only)** | Selects between `tad_mctc.autograd.jacrev` (`torch.func`) and loop-based `torch.autograd.grad` Jacobian assembly on `Calculator` (needed when `libcint` or custom legacy extensions are incompatible with `vmap`-backed `jacrev`). Functional `System` property methods (`dxtb._src.calculators.properties.functional`) have a single pure `torch.func` implementation and do not expose `use_functorch`. |
| `derived_quantity` | `AutogradCalculator.hessian` (`"energy"` vs `"forces"`), `AutogradCalculator.polarizability` (`"energy"` vs `"dipole"`) | **RETAINED (`AutogradCalculator` only)** | Selects whether the 2nd derivative is assembled as `jacrev(jacrev(energy))` or `-jacrev(forces)` / `jacrev(dipole)` on `Calculator`. Functional `System` property methods use one canonical `torch.func` formulation. |

---

## 5. Setup Graph Lifetime and Calculator Lifecycle

### 5.1 Elimination of Persistent Graph-Consumption & Parameter-Version State

The following attributes and methods have been deleted from `BaseCalculator` (`src/dxtb/_src/calculators/types/base.py`) and `packed_result.py`:
- `_param_state_signature`
- `_register_param_backward_hooks`
- `_clear_param_backward_hooks`
- `_system_graph_consumed`
- `_model_param_sig`
- `_ensure_system_current`
- `_lane_systems_memo`

### 5.2 Call-Local Setup Construction (`_prepare_eval_system` and `build_calculator_lane_systems`)

1. **Construction under `torch.no_grad()` or `torch.inference_mode()`**:
   - `BaseCalculator.__init__` clones inference-mode `numbers` / field defaults and exits `torch.inference_mode(False)` during initial setup so `IndexHelper` and structural metadata never store inference tensors.
2. **1D Single-`System` Evaluation (`BaseCalculator._prepare_eval_system`)**:
   - When `self.system` uses only built-in exact components (`is_all_exact`) or when any `self.model.par` parameter/buffer requires gradients, is functorch-wrapped, is an inference tensor, or differs in `dtype` from `self.dtype`, `_prepare_eval_system()` constructs a fresh call-local `System` via `eval_model.setup(self.numbers, batch_mode=0, dd=setup_dd)` and converts its dtype via `_convert_eval_system_dtype(eval_sys, self.dtype)` when `self.dtype != self.model.par.dtype`.
3. **2D Batched Evaluation (`build_calculator_lane_systems`)**:
   - Constructs call-local per-lane `System` instances from `calc.model.par` (deduplicated by composition within the single call) and applies `_convert_eval_system_dtype(lane_sys, calc.dtype)` when `calc.dtype != calc.model.par.dtype`.
4. **Supported Lifecycle Transitions**:
   All 13 lifecycle transitions are supported and verified in `test/test_calculator/test_batch_differentiation_lifecycle.py`:
   1. Constructed under `torch.no_grad()`, then differentiated wrt `positions` and `Model.par`.
   2. Evaluated under `torch.no_grad()`, then differentiated wrt `positions` and `Model.par`.
   3. Constructed and evaluated under `torch.inference_mode()`, then differentiated wrt `positions` and `Model.par`.
   4. Repeated `forces()` calls on the same `Calculator`.
   5. Repeated `torch.autograd.grad(calc.energy(pos), param)` calls on the same `Calculator`.
   6. `calc.energy(pos).backward()` followed by subsequent `forces()` / `autograd.grad` calls.
   7. `torch.func.grad(lambda p: calc.energy(p))(pos)`.
   8. Nested `torch.func.jacrev(torch.func.jacrev(lambda p: calc.energy(p)))(pos)` matching `calc.hessian(pos)`.
   9. In-place parameter mutation (`param.add_(...)` under `torch.no_grad()`) immediately reflected in energy, forces, and parameter gradients.
   10. Replacement of `calc.model` / `Model.par` tensors immediately reflected in energy, forces, and parameter gradients.
   11. `calc.type(torch.float32)` -> evaluate/differentiate -> `calc.type(torch.float64)` on both 1D and 2D Calculators (supported by `Component.type(dtype)` preserving component `.label` and `Halogen.__slots__` including `halogens` and `bases`).
   12. Repeated evaluation across distinct geometries.
   13. `calc.reset()` followed by subsequent singlepoint and derivative evaluations.

---

## 6. Verification and Independent Review Results

1. **Multi-Environment Acceptance Suite Across All 5 Supported CPU Environments**:
   - Modules: `test_batch_differentiation_lifecycle.py`, `test_field_inputs.py`, `test_b3b_layout_contract.py`, `test_s13_public_strategy.py` (`89` tests per environment) plus `test_model.py`, `test_e6bcompat_differential.py`, `test_b8_e8_functional_properties.py` (`148` tests total on `py3.10-torch2.7.0` and `py3.14-torch2.14.1`):
     - `py3.10-torch2.7.0`: **148 passed, 0 failed** (including fix in `LibcintBasisNode.create_libcint`, `Basis.create_libcint`, `packed_result.py`, and `canonical_padding.py` to extract integer metadata via `[int(x.item()) for x in ...]` instead of `.detach().cpu().tolist()`, which fails on `GradTrackingTensor` inside `torch.func` transforms on PyTorch 2.7).
     - `py3.11-torch2.8.0`: **89 passed, 0 failed** (`306.32s`).
     - `py3.12-torch2.11.0`: **89 passed, 0 failed** (`316.51s`).
     - `py3.12-torch2.13.0`: **89 passed, 0 failed** (`323.72s`).
     - `py3.14-torch2.14.1`: **89 passed, 0 failed** (`265.50s`).
2. **S13 668-Test Functional SCF & `vmap` Gate (`py3.14-torch2.14.1`)**:
   - Command: `pytest -q -n 8 --dist worksteal test/test_scf/test_functional_*.py test/test_calculator/test_e6a0_system_vmap.py`
   - Result: **668 passed, 0 failed** (`839.26s`).
3. **Combined Calculator, Properties, and SCF Strategy/Invariant Gate (`py3.14-torch2.14.1`)**:
   - Command: `pytest -q -n 8 --dist worksteal -m "not large" test/test_calculator/ test/test_properties/ test/test_scf/test_invariant_*.py test/test_scf/test_stateless_map.py test/test_scf/test_s13_public_strategy.py test/test_scf/test_singlepoint_core.py test/test_scf/test_e3_end_to_end_acceptance.py`
   - Result: **880 passed, 0 failed** (including 3rd-order numerical finite-difference step-size forwarding in `NumericalCalculator.pol_deriv_numerical`, `hyperpolarizability_numerical`, and `raman_numerical`, and `ES3.__init__` `dd` conversion).
4. **Independent Read-Only Review (`5e84127d-fb75-4c13-9f06-f6dcd0e7a7ed`)**:
   - **Verdict**: **APPROVED (`0` remaining findings across Round 1 and Round 2)**.
   - S0–S13.1 SCF core untouched (`git diff 50bea4ac4870952948b6971cc56c8fe48ed1d834 -- src/dxtb/_src/scf/` is empty).
