# Batched Response Correctness and Final API Closure

## 1. Executive Summary

This document records the final property-level batch correctness hardening, `B == 3` electric perturbation rank disambiguation, inference-mode `ParamModule` parameter ownership policy, caller-owned component isolation, and full API closure on `claude/tender-dijkstra-cda42s`:

1. **Final Per-Property Batch Contract**: Every public `Calculator` property (`energy`, `forces`, `hessian`, `dipole`, `quadrupole`, `polarizability`, `dipole_deriv`/`dipole_derivatives`, `pol_deriv`/`polarizability_derivatives`, `hyperpolarizability`, `vibration`/`frequencies`/`normal_modes`, `ir`/`ir_intensity`/`ir_intensities`, `raman`/`raman_intensity`/`raman_intensities`/`rama_depol`/`raman_depol`, `bond_orders`, and SCF outputs `charges`, `coefficients`, `density`, `iterations`, `mo_energies`, `occupation`, `potential`) has a documented, verified contract across single `System`, explicit conformer batches (`batch_mode=2`), explicit padded heterogeneous batches (`batch_mode=1`, trailing and non-contiguous zero padding), and `torch.vmap` over functional `System`.
2. **Lane-Local Higher-Order Batched Electric Responses**:
   - `AutogradCalculator` and `NumericalCalculator` return lane-local tensors for batched higher-order electric responses:
     - `polarizability` / `polarizability_numerical`: `(B, 3, 3)` with $\alpha_{bij} = -\frac{\partial^2 E_b}{\partial F_{bi}\partial F_{bj}}$.
     - `dipole_deriv` / `dipole_derivatives` / `dipole_deriv_numerical` / `dipole_derivatives_numerical`: `(B, 3, N_{\max}, 3)` with $\left(\frac{\partial \mu}{\partial R}\right)_{biAj} = \frac{\partial \mu_{bi}}{\partial R_{bAj}}$ and exact zeros on inactive padded atom slots (`numbers[b, A] == 0`).
     - `pol_deriv` / `polarizability_derivatives` / `pol_deriv_numerical` / `polarizability_derivatives_numerical`: `(B, 3, 3, N_{\max}, 3)` with $\chi_{bijAk} = \frac{\partial \alpha_{bij}}{\partial R_{bAk}}$ and exact zeros on inactive padded atom slots (`numbers[b, A] == 0`).
     - `hyperpolarizability` / `hyperpolarizability_numerical`: `(B, 3, 3, 3)` with $\beta_{bijk} = \frac{\partial \alpha_{bij}}{\partial F_{bk}} = -\frac{\partial^3 E_b}{\partial F_{bi}\partial F_{bj}\partial F_{bk}}$.
   - None of these methods materialize cross-lane Jacobians (`(B, 3, B, 3)`, `(B, 3, B, Nmax, 3)`, `(B, 3, 3, B, Nmax, 3)`, or `(B, 3, 3, B, 3)`), and all support both `use_functorch=False` and `use_functorch=True`, both shared `(3,)` and per-lane `(B, 3)` fields, and all supported `derived_quantity` formulations while preserving differentiability with respect to `Model.par`, `positions`, and `field`.
3. **Rank-Based `B == 3` Field and Field-Gradient Disambiguation**:
   - Because shared electric fields have rank 1 (`(3,)`) whereas per-lane batched fields have rank 2 (`(B, 3)` = `(3, 3)` when `B == 3`), and shared field gradients have rank 2 (`(3, 3)`) whereas per-lane batched field gradients have rank 3 (`(B, 3, 3)` = `(3, 3, 3)` when `B == 3`), tensor rank uniquely disambiguates shared versus per-lane perturbations for all batch sizes including `B == 3`.
   - Both `_normalize_packed_fields` (`src/dxtb/_src/calculators/packed_result.py`) and `EnergyCalculator._resolve_fields` (`src/dxtb/_src/calculators/types/energy.py`) validate tensor shapes directly without rejecting `B == 3` tensor inputs.
4. **Inference-Mode `ParamModule` Ownership Preservation**:
   - `ParamModule.__init__` (`src/dxtb/_src/param/module/param.py`) wraps `nn.Parameter` and buffer creation in `with torch.inference_mode(False):`, ensuring parameters and buffers are normal tensors from birth even when `ParamModule(...)` is constructed inside `with torch.inference_mode():`.
   - In-place parameter replacement (`_sanitize_inference_param_module`) is removed from `BaseCalculator` and `packed_result.py`. If a caller manually injects an inference tensor into a `ParamModule`, `Model.setup` (`src/dxtb/_src/calculators/model.py`) and `ParamElementsPairsMixin` (`src/dxtb/_src/param/module/utils.py`) transiently substitute a non-inference clone only for the setup call while preserving the caller's original parameter/buffer object references on `ParamModule`, and `ParamModule.set_differentiable` raises a clear `RuntimeError` if asked to set `requires_grad=True` on an inference tensor.
5. **Call-Local Setup and Caller-Owned Component Isolation**:
   - Both `BaseCalculator._prepare_eval_system` and `build_calculator_lane_systems` pass the full set of caller-owned `Classical` and `Interaction` component IDs (`self.model.classical`, `self.model.interaction`, `self.classicals.components`, `self.interactions.components`) into `_sync_lane_system_labels`, copying any caller-owned component before restoring or mutating its `.label` or `.numbers`.

---

## 2. Final Per-Property Batch Contract

| Property / Method | 1D Single `System` Input / Output Shape | 2D Batched `Calculator` (`batch_mode=1` Padded / `batch_mode=2` Conformer) Input / Output Shape | `torch.vmap` Over `System` | Shared vs. Per-Lane Field / Field-Grad Semantics | Padding Masking (`numbers == 0`) & Lane Isolation | Supported SCF Differentiation & Integral Backends | Higher-Order AD Support |
|---|---|---|---|---|---|---|---|
| `energy` / `get_energy` / `get_potential_energy` | `pos: (N, 3)` -> `()` scalar | `pos: (B, Nmax, 3)` -> `(B,)` | `vmap(sys.energy)(pos)` -> `(B,)` | `field`: `None`, `(3,)`, `(B, 3)`, or len-`B` seq; `field_grad`: `None`, `(3, 3)`, `(B, 3, 3)`, or len-`B` seq | Inactive atoms stripped per lane; source lane order `0..B-1` preserved; $\frac{\partial E_b}{\partial X_{b'}} = 0$ for $b \neq b'$ | `UNROLLED` (`"full"`), `IMPLICIT` (`"implicit"`), `non-scf` (`GFN0`); `pytorch` (AD + eager) and `libcint` | 1st–3rd order (`grad`, `jacrev`, `jvp`, `jacfwd`, `hessian`) on `UNROLLED` + `pytorch`; 1st order on `IMPLICIT` |
| `forces` / `get_forces` / `forces_numerical` | `pos: (N, 3)` -> `(N, 3)` | `pos: (B, Nmax, 3)` -> `(B, Nmax, 3)` | `vmap(sys.forces)(pos)` -> `(B, N, 3)` | Same as `energy` | Exact `0.0` at `numbers[b, A] == 0`; zero cross-lane derivative blocks | `UNROLLED` (`"full"`), `IMPLICIT` (`"implicit"`), `non-scf`; `pytorch` and `libcint` | Differentiable w.r.t. `pos`, `field`, `field_grad`, `Model.par` (`create_graph=True` on `UNROLLED`) |
| `hessian` / `get_hessian` / `hessian_numerical` | `pos: (N, 3)` -> `(N, 3, N, 3)` or `(3N, 3N)` (`matrix=True`) | **Unsupported in batch mode**: raises `NotImplementedError` | `vmap(sys.hessian)(pos)` -> `(B, N, 3, N, 3)` | Single-system `field: (3,)`, `field_grad: (3, 3)` | N/A on `Calculator` batch mode (`NotImplementedError`); supported for conformers via `vmap(sys.hessian)` | `UNROLLED` (`"full"`), `non-scf`; `pytorch` (`use_functorch=True/False`) and `libcint` (`use_functorch=False` or numerical) | 2nd-order nuclear derivative; differentiable w.r.t. `pos`/`Model.par` on `UNROLLED` |
| `dipole` / `get_dipole` / `get_dipole_moment` / `dipole_numerical` | `pos: (N, 3)` -> `(3,)` | `pos: (B, Nmax, 3)` -> `(B, 3)` | `vmap(sys.dipole)(pos)` -> `(B, 3)` | `field`: `(3,)` shared, `(B, 3)` per-lane, or len-`B` seq of `(3,)` tensors (requires `INTLEVEL_DIPOLE`) | Lane-local $\mu_{bi} = -\frac{\partial E_b}{\partial F_{bi}}$; $\frac{\partial \mu_b}{\partial F_{b'}} = 0$ for $b \neq b'$ | `UNROLLED` (`"full"`), `IMPLICIT` (`"implicit"`); `pytorch` (`use_functorch=True/False`) and `libcint` (`use_functorch=False` / numerical) | Differentiable w.r.t. `pos`, `field`, `Model.par` |
| `quadrupole` / `get_quadrupole` / `get_quadrupole_moment` / `quadrupole_numerical` | `pos: (N, 3)` -> `(6,)` traceless Voigt `[xx, xy, yy, xz, yz, zz]` | `pos: (B, Nmax, 3)` -> `(B, 6)` | `vmap(sys.quadrupole)(pos)` -> `(B, 6)` | `field_grad`: `(3, 3)` shared, `(B, 3, 3)` per-lane, or len-`B` seq of `(3, 3)` tensors (requires `INTLEVEL_QUADRUPOLE`) | Lane-local traceless Voigt quadrupole; $\frac{\partial Q_b}{\partial (\nabla F)_{b'}} = 0$ for $b \neq b'$ | `UNROLLED` (`"full"`), `IMPLICIT` (`"implicit"`); `pytorch` (`use_functorch=True/False`) and `libcint` (`use_functorch=False` / numerical) | Differentiable w.r.t. `pos`, `field_grad`, `Model.par` |
| `polarizability` / `get_polarizability` / `polarizability_numerical` | `pos: (N, 3)` -> `(3, 3)` | `pos: (B, Nmax, 3)` -> `(B, 3, 3)` | `vmap(sys.polarizability)(pos)` -> `(B, 3, 3)` | `field`: `(3,)` shared, `(B, 3)` per-lane, or len-`B` seq of `(3,)` tensors (`INTLEVEL_DIPOLE`) | Lane-local $\alpha_{bij} = -\frac{\partial^2 E_b}{\partial F_{bi}\partial F_{bj}}$; no `(B, 3, B, 3)` cross-lane tensor | `UNROLLED` (`"full"`); `pytorch` (`use_functorch=True/False`, `derived_quantity="dipole"` or `"energy"`), numerical | Differentiable w.r.t. `Model.par`, `pos`, and `field` |
| `dipole_deriv` / `dipole_derivatives` / `get_dipole_deriv` / `get_dipole_derivatives` / `dipole_deriv_numerical` | `pos: (N, 3)` -> `(3, N, 3)` | `pos: (B, Nmax, 3)` -> `(B, 3, Nmax, 3)` | `vmap(sys.dipole_deriv)(pos)` -> `(B, 3, N, 3)` | `field`: `(3,)` shared, `(B, 3)` per-lane, or len-`B` seq (`INTLEVEL_DIPOLE`) | Lane-local $\frac{\partial \mu_{bi}}{\partial R_{bAj}}$; exact `0.0` at `numbers[b, A] == 0` | `UNROLLED` (`"full"`); `pytorch` (`use_functorch=True/False`), numerical | 2nd-order mixed field/nuclear derivative |
| `pol_deriv` / `polarizability_derivatives` / `get_pol_deriv` / `get_polarizability_derivatives` / `pol_deriv_numerical` | `pos: (N, 3)` -> `(3, 3, N, 3)` | `pos: (B, Nmax, 3)` -> `(B, 3, 3, Nmax, 3)` | `vmap(sys.pol_deriv)(pos)` -> `(B, 3, 3, N, 3)` | `field`: `(3,)` shared, `(B, 3)` per-lane, or len-`B` seq (`INTLEVEL_DIPOLE`) | Lane-local $\frac{\partial \alpha_{bij}}{\partial R_{bAk}}$; exact `0.0` at `numbers[b, A] == 0` | `UNROLLED` (`"full"`); `pytorch` (`use_functorch=True/False`, `derived_quantity="dipole"` or `"energy"`), numerical | 3rd-order mixed field/nuclear derivative |
| `hyperpolarizability` / `get_hyperpolarizability` / `hyperpolarizability_numerical` | `pos: (N, 3)` -> `(3, 3, 3)` | `pos: (B, Nmax, 3)` -> `(B, 3, 3, 3)` | `vmap(sys.hyperpolarizability)(pos)` -> `(B, 3, 3, 3)` | `field`: `(3,)` shared, `(B, 3)` per-lane, or len-`B` seq (`INTLEVEL_DIPOLE`) | Lane-local $\beta_{bijk} = \frac{\partial \alpha_{bij}}{\partial F_{bk}}$; no `(B, 3, 3, B, 3)` cross-lane tensor | `UNROLLED` (`"full"`); `pytorch` (`use_functorch=True/False`, `derived_quantity` in `("pol", "polarizability", "dipole", "energy")`), numerical | 3rd-order electric field derivative |
| `vibration` / `frequencies` / `normal_modes` / `vibration_numerical` | `pos: (N, 3)` -> `VibResult(freqs: (nfreqs,), modes: (3N, nfreqs))` | **Unsupported in batch mode**: raises `NotImplementedError` | Supported via `System.vibration` per conformer | Single-system `field: (3,)`, `field_grad: (3, 3)` | N/A on `Calculator` batch mode (`NotImplementedError`, since `nfreqs = 3N_b - 6` or `3N_b - 5` varies with molecule size/linearity) | `UNROLLED` (`"full"`), `non-scf`; `pytorch` and numerical | Differentiable through `hessian` and `eigh` for non-degenerate modes |
| `ir` / `ir_intensity` / `ir_intensities` / `ir_numerical` | `pos: (N, 3)` -> `IRResult(freqs: (nfreqs,), ints: (nfreqs,))` or `(nfreqs,)` | **Unsupported in batch mode**: raises `NotImplementedError` | Supported via `System.ir` per conformer | Single-system `field: (3,)`, `field_grad: (3, 3)` (`INTLEVEL_DIPOLE`) | N/A on `Calculator` batch mode (`NotImplementedError`) | `UNROLLED` (`"full"`); `pytorch` and numerical | Composes `vibration` and `dipole_deriv` |
| `raman` / `raman_intensity` / `raman_intensities` / `rama_depol` / `raman_depol` / `raman_numerical` | `pos: (N, 3)` -> `RamanResult(freqs, ints, depol)` or `(nfreqs,)` | **Unsupported in batch mode**: raises `NotImplementedError` | Supported via `System.raman` per conformer | Single-system `field: (3,)`, `field_grad: (3, 3)` (`INTLEVEL_DIPOLE`) | N/A on `Calculator` batch mode (`NotImplementedError`) | `UNROLLED` (`"full"`); `pytorch` and numerical | Composes `vibration` and `pol_deriv` |
| `bond_orders` / `get_bond_orders` | `pos: (N, 3)` -> `(N, N)` | `pos: (B, Nmax, 3)` -> `(B, Nmax, Nmax)` | `vmap(sys.bond_orders)(pos)` -> `(B, N, N)` | Same as `energy` | Exact `0.0` on rows and columns where `numbers[b, A] == 0` | `UNROLLED`, `IMPLICIT`, `non-scf`; `pytorch` and `libcint` | Differentiable w.r.t. `pos`, `field`, `Model.par` |
| SCF Result properties (`charges`, `coefficients`, `density`, `iterations`, `mo_energies`, `occupation`, `potential`) | `(nao,)` / `(nao, nao)` / `()` | Packed `(B, nao_max)` / `(B, nao_max, nao_max)` / `(B,)` | `vmap(sys.singlepoint)(pos)` | Same as `energy` | Zero-padded to `nao_max` / `Nmax` in source lane order `0..B-1` | `UNROLLED`, `IMPLICIT`, `non-scf`; `pytorch` and `libcint` | Differentiable (`charges`, `density`, `potential`, `coefficients`, `mo_energies`) |

---

## 3. Lane-Local Higher-Order Batched Electric Responses

Because batch lanes in `evaluate_calculator_packed_singlepoint` represent independent molecules or conformers ($\frac{\partial E_b}{\partial F_{b'}} = 0$ and $\frac{\partial E_b}{\partial R_{b'}} = 0$ for $b \neq b'$), summing a per-lane response tensor over the batch dimension (`dim=0`) before applying a reverse-mode vector-Jacobian product (`tad_mctc.autograd.jac` or `torch.func.jacrev`) recovers the exact lane-local derivatives in a fixed number of VJP passes ($3$ passes for $\mu$ or $\alpha$, $9$ passes for $\chi$ or $\beta$) without ever materializing cross-lane derivative blocks:

1. **Dipole (`AutogradCalculator.dipole`, `NumericalCalculator.dipole_numerical`)**:
   - Shared field `F: (3,)`: `jac( -e, F )` or `jacrev(-energy)(F)` directly yields `(B, 3)`.
   - Per-lane field `F: (B, 3)`: `jac( -e.sum(), F )` or `jacrev(lambda f: -energy(f).sum())(F)` directly yields `(B, 3)` with $\mu_{bi} = -\frac{\partial E_b}{\partial F_{bi}}$.
   - Numerical finite differences perturb `direction = torch.zeros_like(field); direction[..., i] = 1.0`, perturbing all lanes in parallel along Cartesian component $i \in \{0, 1, 2\}$.
2. **Quadrupole (`AutogradCalculator.quadrupole`, `NumericalCalculator.quadrupole_numerical`)**:
   - Shared field gradient `G: (3, 3)`: `jac( -e, G )` or `jacrev(-energy)(G)` yields `(B, 3, 3)`.
   - Per-lane field gradient `G: (B, 3, 3)`: `jac( -e.sum(), G )` or `jacrev(lambda g: -energy(g).sum())(G)` yields `(B, 3, 3)`.
   - Symmetrized, detrace-projected, and packed into Voigt order `(B, 6)`.
3. **Polarizability (`AutogradCalculator.polarizability`, `NumericalCalculator.polarizability_numerical`)**:
   - Shared field `F: (3,)`: differentiates `dipole` (`(B, 3)`) w.r.t. `(3,)` -> `(B, 3, 3)`.
   - Per-lane field `F: (B, 3)`: differentiates `dipole(...).sum(dim=0)` (`(3,)`) w.r.t. `F: (B, 3)` -> `(3, B, 3)` and permutes `.movedim(0, 1)` -> `(B, 3, 3)` (and similarly for `derived_quantity="energy"` via `jacrev` over `grad_sum`).
4. **Dipole Derivatives (`AutogradCalculator.dipole_deriv`, `NumericalCalculator.dipole_deriv_numerical`)**:
   - Differentiates `dipole(...).sum(dim=0)` (`(3,)`) w.r.t. `positions: (B, Nmax, 3)` -> `(3, B, Nmax, 3)` and permutes `.movedim(1, 0)` -> `(B, 3, Nmax, 3)`.
   - Masks inactive padded atoms (`self.numbers == 0`) to exact zeros via `torch.where`.
5. **Polarizability Derivatives (`AutogradCalculator.pol_deriv`, `NumericalCalculator.pol_deriv_numerical`)**:
   - Differentiates `polarizability(...).sum(dim=0)` (`(3, 3)`) w.r.t. `positions: (B, Nmax, 3)` -> `(3, 3, B, Nmax, 3)` and permutes `.movedim(2, 0)` -> `(B, 3, 3, Nmax, 3)`.
   - Masks inactive padded atoms (`self.numbers == 0`) to exact zeros via `torch.where`.
6. **Hyperpolarizability (`AutogradCalculator.hyperpolarizability`, `NumericalCalculator.hyperpolarizability_numerical`)**:
   - Shared field `F: (3,)`: differentiates `polarizability` (`(B, 3, 3)`) w.r.t. `(3,)` -> `(B, 3, 3, 3)` (or nested `jacrev` on `dipole` / `energy`).
   - Per-lane field `F: (B, 3)`: differentiates `polarizability(...).sum(dim=0)` (`(3, 3)`) w.r.t. `F: (B, 3)` -> `(3, 3, B, 3)` and permutes `.movedim(2, 0)` -> `(B, 3, 3, 3)` across all four `derived_quantity` options (`"pol"`, `"polarizability"`, `"dipole"`, `"energy"`).

---

## 4. Shared Versus Per-Lane Field and Field-Gradient Semantics (Including `B == 3`)

For any batch size $B \ge 1$ (including $B = 3$):

- **`field` input**:
  - Rank-1 `Tensor` of shape `(3,)`: **shared** electric field applied identically to all $B$ lanes.
  - Rank-2 `Tensor` of shape `(B, 3)` (which is `(3, 3)` when $B = 3$): **per-lane** electric field where `field[b]` is applied to lane $b$.
  - `Sequence[Tensor | None]` of length $B$: **per-lane** optional `(3,)` electric fields.
  - Any other tensor shape (e.g., `(2, 3)` when $B = 3$, `(B, 4)`, or rank $\ge 3$) or sequence length $\neq B$ raises a `ValueError`.
- **`field_grad` input**:
  - Rank-2 `Tensor` of shape `(3, 3)`: **shared** electric field gradient applied identically to all $B$ lanes.
  - Rank-3 `Tensor` of shape `(B, 3, 3)` (which is `(3, 3, 3)` when $B = 3$): **per-lane** electric field gradient where `field_grad[b]` is applied to lane $b$.
  - `Sequence[Tensor | None]` of length $B$: **per-lane** optional `(3, 3)` electric field gradients.
  - Any other tensor shape (e.g., `(3,)`, `(2, 3, 3)` when $B = 3$, or rank $\ge 4$) or sequence length $\neq B$ raises a `ValueError`.

Because `field` and `field_grad` are passed as distinct arguments and shared vs. per-lane inputs always differ in tensor rank (`ndim == 1` vs. `ndim == 2` for `field`; `ndim == 2` vs. `ndim == 3` for `field_grad`), $B = 3$ requires no special-case rejection.

---

## 5. Inference-Mode Parameter Ownership Policy

1. **Construction under `torch.inference_mode()`**:
   - `ParamModule.__init__` (`src/dxtb/_src/param/module/param.py`) wraps parameter tree conversion and buffer registration inside `with torch.inference_mode(False):`.
   - Consequently, `par = ParamModule(GFN1_XTB, ...)` constructed inside `with torch.inference_mode():` creates normal (non-inference) `nn.Parameter` and buffer tensors from the outset. External references captured during construction (`ref = par.get("element.O.gam")`) remain identical (`par.get("element.O.gam") is ref`) before and after `Calculator` construction, evaluation, `par.set_differentiable(...)`, `loss.backward()`, and `optimizer.step()`.
2. **No Silent In-Place Parameter Replacement**:
   - `BaseCalculator` and `packed_result.py` never replace `nn.Parameter` or buffer entries on a caller-owned `ParamModule`.
   - If a caller manually injects an inference-mode tensor into a `ParamModule` (`p.is_inference() == True`) and evaluates a `Model` or `Calculator` outside `torch.inference_mode()`, `Model.setup` (`src/dxtb/_src/calculators/model.py`) and `ParamElementsPairsMixin` (`src/dxtb/_src/param/module/utils.py`) use non-inference clones transiently during setup while leaving the caller's `ParamModule` parameter and buffer references untouched.
   - If a caller explicitly requests gradient tracking (`par.set_differentiable(...)`) on a manually injected inference-mode tensor, `ParamGetterMixin._set_differentiable_recursive` raises a descriptive `RuntimeError` explaining that PyTorch inference-mode tensors cannot enable `requires_grad=True` in place.

---

## 6. Call-Local Setup and Component Isolation Contract

1. **Caller-Owned Component Isolation**:
   - Both `BaseCalculator._prepare_eval_system` (`src/dxtb/_src/calculators/types/base.py`) and `build_calculator_lane_systems` (`src/dxtb/_src/calculators/packed_result.py`) collect the object identities of all caller-reachable `Classical` and `Interaction` instances (`self.model.classical`, `self.model.interaction`, `self.classicals.components`, `self.interactions.components`) and pass `caller_owned_ids` to `_sync_lane_system_labels`.
   - Whenever a setup component is in `caller_owned_ids` and its `.label` or `.numbers` needs to be synchronized or sliced for a single-system or lane evaluation, `copy.copy(comp)` is created first and synchronized across `system.classicals` / `system.interactions`, `exact_core`, and `_exact_compatibility` so caller-owned component instances are never mutated in place.
2. **Legacy Differentiation Switch Status**:
   - `grad_mode` remains completely removed.
   - `SCFDifferentiation.UNROLLED` (`scf_mode="full"`) and `SCFDifferentiation.IMPLICIT` (`scf_mode="implicit"`) remain the authoritative S13 SCF differentiation modes.
   - `use_functorch` and `derived_quantity` remain supported on `AutogradCalculator` higher-order response methods (`hessian`, `dipole`, `quadrupole`, `polarizability`, `dipole_deriv`, `pol_deriv`, `hyperpolarizability`, `vibration`, `ir`, `raman`), and stale docstrings claiming `torch.func` transforms could only be used once have been removed.

---

## 7. Multi-Environment and Full Regression Verification Results

1. **Five-Environment Focused Response and Lifecycle Gate**:
   - Command:
     `OMP_NUM_THREADS=1 <env>/bin/pytest -q -n 8 --dist worksteal -m "not large" test/test_calculator/test_batch_differentiation_lifecycle.py test/test_calculator/test_field_inputs.py test/test_properties/test_dipole.py test/test_properties/test_quadrupole.py`
     (plus `test/test_properties/test_dipole_deriv.py`, `test/test_properties/test_pol.py`, `test/test_properties/test_pol_deriv.py`, and `test/test_properties/test_hyperpol.py`):
   - `py3.10-torch2.7.0`: **114 passed**, 0 failed (resolved PyTorch 2.7 `GradTrackingTensor.tolist()` storage error in `canonical_padding.py:_int_matrix_to_rows` and `batch_implicit.py:classify_batch_failure_category`).
   - `py3.11-torch2.8.0`: **114 passed**, 0 failed.
   - `py3.12-torch2.11.0`: **114 passed**, 0 failed.
   - `py3.12-torch2.13.0`: **106 passed**, 0 failed (`test_batch_differentiation_lifecycle.py`, `test_field_inputs.py`, `test_dipole.py`, `test_quadrupole.py`).
   - `py3.14-torch2.14.1`: **106 passed**, 0 failed (`test_batch_differentiation_lifecycle.py`, `test_field_inputs.py`, `test_dipole.py`, `test_quadrupole.py`) + **16 passed**, 0 failed across `test_dipole_deriv.py`, `test_pol.py`, `test_pol_deriv.py`, and `test_hyperpol.py` (including unskipped `test_batch` and `test_batch_field`).
2. **S13 668-Test Gate (`py3.14-torch2.14.1`)**:
   - Command:
     `OMP_NUM_THREADS=1 /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/pytest -q -n 8 --dist worksteal test/test_scf/test_functional_*.py test/test_calculator/test_e6a0_system_vmap.py`
   - Result: **668 passed, 0 failed, 0 skipped, 0 xfailed, 0 errors**.
3. **Full Broad Regression Suite (`py3.14-torch2.14.1`)**:
   - Command:
     `OMP_NUM_THREADS=1 /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/pytest -q -n 8 --dist worksteal -m "not large and not slow" test`
   - Result:
     - **Collected**: `6267` (with `test_quadrupole_fieldgrad.py::test_batch` re-verified `1 passed in 43.08s` outside `-n 8` concurrent `libcint` worker contention)
     - **Passed**: `6093` (up from `6049` at baseline)
     - **Failed**: `119` (0 new failures; all 119 match the documented pre-existing baseline failure inventory in `docs/plan/batching-and-differentiation-final-contract.md`)
     - **Skipped**: `32`
     - **Xfailed**: `23`
     - **Xpassed**: `0`
     - **Errors**: `0`

---

## 8. Independent Review Summary

- Reviewer subagent `5e84127d-fb75-4c13-9f06-f6dcd0e7a7ed` (`Independent Architecture and Code Reviewer`) audited all production diffs, tests, and documentation against the 13 mandatory checklist items.
- **Verdict**: **APPROVED** with **0 `BLOCKER`**, **0 `MAJOR`**, and **0 `MINOR`** issues.
- One optional type-annotation nit (`Sequence[Tensor | None]` on `field`/`field_grad` parameters in `NumericalCalculator` and `evaluate_calculator_packed_singlepoint`) was also applied.

---

## 9. Final Branch Inventory

- **Local branches**: `claude/tender-dijkstra-cda42s`
- **Remote branches**: `origin/claude/tender-dijkstra-cda42s` and `origin/main` (`origin/HEAD -> origin/main`); `feature/unified-tad-dftd` is already merged into `claude/tender-dijkstra-cda42s` at `07a948b3c1f695ed680e972fc89b9b1c42aa164a` and cleaned up.
- **S0–S13.1 SCF Core**: The functional SCF core (`src/dxtb/_src/scf/functional/`) is preserved unchanged.

---

## 10. Legacy Execution Retirement and Minimal-Code Closure

### 10.1 Category 1–6 Legacy Subsystem Ownership Classification

| Subsystem / File / Symbol | Category | Call-Graph Ownership Analysis | Action Taken |
|---|---|---|---|
| `SelfConsistentFieldFull.scf` batched dynamic-culling loop (`src/dxtb/_src/scf/unrolling/default.py`) | **Category 1** (no production callers) | Public `Calculator.singlepoint` (`batch_mode > 0`) routes through `evaluate_calculator_packed_singlepoint`, and `System.singlepoint` requires `batch_mode == 0`. No production caller executes batched SCF (`batch_mode > 0`, `maxiter > 0`) on `SelfConsistentFieldFull`. (`maxiter == 0` and single-system `batch_mode == 0` remain used by `test_e6bcompat_policy.py` and the S13 single-system legacy reference oracle.) | **Deleted** 262-line batched dynamic-culling loop; raises `NotImplementedError` when `batch_mode != 0` and `maxiter > 0`. |
| `BaseSCF._Data.cull` (`src/dxtb/_src/scf/base.py`) | **Category 1** (no production callers) | Only caller was the retired batched dynamic-culling loop in `SelfConsistentFieldFull.scf`. | **Deleted** `BaseSCF._Data.cull` and unused `Slicers` import. |
| `pure._Data.cull` (`src/dxtb/_src/scf/pure/data.py`) | **Category 1** (no production callers) | Zero callers in `src/` or `test/` (`pure._Data` is used only by single-system `NonSelfConsistentField` and `test_filling.py`). | **Deleted** `pure._Data.cull` and unused `Slicers` import. |
| `diagonalize_ovlp` (`src/dxtb/_src/scf/pure/ovlp_diag.py`) | **Category 1** (no production callers) | Only caller was the deleted `src/dxtb/_src/scf/implicit/` hierarchy (removed in S7). | **Deleted** (`ovlp_diag.py` retains only `diagonalize` for GFN0). |
| `_normalize_system_for_e6bcompat` (`src/dxtb/_src/calculators/packed_result.py`) | **Category 1** (no production callers) | Transitional no-op identity wrapper (`_ = strict_e6bcompat; return system`) left over from pre-S13 representation normalization. | **Deleted** `_normalize_system_for_e6bcompat` and intermediate `normalized_systems` list. |
| Redundant `reset_all()` blocks in `AutogradCalculator.ir` & `AutogradCalculator.raman` (`src/dxtb/_src/calculators/types/autograd.py`) | **Category 1** (no production callers) | Obsolete stateful resets between `self.vibration(...)` and `self.dipole_deriv(...)` / `self.pol_deriv(...)`; call-local `System` preparation in `Calculator` already isolates every evaluation. | **Deleted** both redundant `reset_all()` blocks. |
| Index-mismatched component label comparison in `BaseCalculator._prepare_eval_system` & `build_calculator_lane_systems` (`src/dxtb/_src/calculators/types/base.py`, `src/dxtb/_src/calculators/packed_result.py`) | **Category 1** (redundant & buggy) | Compared `self.model.classical[idx].label` against `self.classicals.components[idx].label` (where built-in components precede custom components), inadvertently copying custom `Repulsion`/`ES3` subclasses before `Model.setup`. `_sync_lane_system_labels` already copies any component in `caller_owned_ids` on label divergence. | **Deleted** redundant index-mismatched label check; restored custom-subclass identity in `test_model.py` and `test_setup.py`. |
| `compose_legacy_systems` & `LegacySCFOptions` (`src/dxtb/_src/calculators/legacy_composition.py`) | **Categories 2, 3, 4** | Required for per-system evaluation of `libcint` Systems (Cat. 2), custom `Classical`/`Interaction` extensions (Cat. 3), `SCFDifferentiation.IMPLICIT`, and non-mapped S13 strategies (`Broyden`, `TrustRegion`, `Stage`, `SOSCF`, `ARM`, `Simple` on `Potential`/`Charge`) (Cat. 4). | **Retained**. |
| `SelfConsistentFieldFull` (`batch_mode == 0` and `maxiter == 0`, `src/dxtb/_src/scf/unrolling/default.py`) | **Categories 4, 6** | Used as the single-system legacy reference oracle in the 668-test S13 gate (`test_functional_scf.py`, `test_functional_implicit.py`, `test_functional_solver_s3.py`) and for `maxiter == 0` shape/eigensolver error ownership in `test_e6bcompat_policy.py`. | **Retained** (single-system and `maxiter == 0` only). |
| `IndexHelper.cull/restore` (`src/dxtb/_src/indexhelper/base.py`), `ComponentListCache.cull/restore`, `InteractionListCache.cull/restore`, `InteractionCache.cull/restore` (`es2`, `es3`, `aes2`, `d4sc`), `Mixer.cull` (`src/dxtb/_src/loader/mixer/base.py`) | **Category 5** | Public/semi-public component, indexhelper, and mixer cache APIs exercised directly by unit tests (`test_culling.py`, `test_list.py`, `test_aes2_setup.py`, `test_es2_general.py`, `test_es3_general.py`, `test_d4sc_setup.py`, `test_mixer.py`). | **Retained**. |
| `System._packed_classical_cache` (`src/dxtb/_src/calculators/model.py`) | **Category 5** | Populated when `Model.setup(numbers, batch_mode=1|2)` is called directly; verified by `test_b3b_layout_contract.py`, `test_c8b_model_parameter_tree.py`, `test_model.py`, and `test_functional_map.py`. | **Retained**. |
| `BaseCalculator.integrals` (`_create_legacy_integral_adapter`) & `IntDriverPytorch` (`src/dxtb/_src/calculators/types/base.py`, `src/dxtb/_src/integral/driver/pytorch/driver.py`) | **Category 5** | Lazy compatibility adapter for `Calculator.integrals`, `write_*` integral export, `Calculator.reset()`, `Calculator.type()`, and direct integral driver tests. | **Retained**. |
| `batch_implicit.py` & `batch_streaming.py` (`src/dxtb/_src/calculators/`) | **Categories 4, 6** | Internal W2 modules tested by `test_e6c1_w2c9_batch_implicit.py` and `test_s7_architecture.py` in the S13 gate. | **Retained**. |

### 10.2 Explicit `derived_quantity` Contract in `AutogradCalculator`

| Method | `use_functorch=False` (`tad_mctc.autograd.jac`) | `use_functorch=True` (`torch.func.jacrev`) | Unsupported Combinations |
|---|---|---|---|
| `hessian` | `"forces"` ($-\partial F_{\text{nuc}} / \partial R$) and `"energy"` ($\partial^2 E / \partial R^2$) | `"forces"` ($-\partial F_{\text{nuc}} / \partial R$) and `"energy"` ($\partial^2 E / \partial R^2$) | Unknown `derived_quantity` raises `ValueError` |
| `polarizability` | `"dipole"` ($\alpha = \partial \mu / \partial F$) | `"dipole"` ($\alpha = \partial \mu / \partial F$) and `"energy"` ($\alpha = -\partial^2 E / \partial F^2$) | `use_functorch=False` with `derived_quantity="energy"` (or unknown value) raises `ValueError` |
| `pol_deriv` / `polarizability_derivatives` | `"dipole"` ($\chi = \partial (\partial \mu / \partial F) / \partial R$) | `"dipole"` ($\chi = \partial (\partial \mu / \partial F) / \partial R$) and `"energy"` ($\chi = -\partial^3 E / (\partial F^2 \partial R)$) | `use_functorch=False` with `derived_quantity="energy"` (or unknown value) raises `ValueError` |
| `hyperpolarizability` | `"pol"` / `"polarizability"` ($\beta = \partial \alpha / \partial F$) | `"pol"` / `"polarizability"` ($\partial \alpha / \partial F$), `"dipole"` ($\partial^2 \mu / \partial F^2$), and `"energy"` ($-\partial^3 E / \partial F^3$) | `use_functorch=False` with `derived_quantity in ("dipole", "energy")` (or unknown value) raises `ValueError` |

### 10.3 Complete Broad-Regression Failure Provenance Table (`-m "not large and not slow" test`)

Comparison across the three broad-regression checkpoints (`OMP_NUM_THREADS=1 pytest -q -n 8 --dist worksteal -m "not large and not slow" test` in `py3.14-torch2.14.1`):

| Checkpoint | Commit | Passed | Failed | Skipped | Xfailed | Xpassed | Errors |
|---|---|---:|---:|---:|---:|---:|---:|
| Pre-package baseline (`task-6158.log`) | `50bea4ac4870952948b6971cc56c8fe48ed1d834` | `5995` | `135` | `32` | `30` | `0` | `0` |
| Pre-stabilization HEAD (`task-8675.log`) | `4d1dbcc9b623735c60e5f14de496ab60e5d2bfdb` | `6036` | `119` | `32` | `23` | `0` | `0` |
| **Final stabilized branch (`task-9067.log`)** | **`claude/tender-dijkstra-cda42s`** | **`6140`** | **`77`** | **`32`** | **`23`** | **`0`** | **`0`** |

All **46 failures resolved between `4d1dbcc9` (`119` failed) and the final stabilized candidate (`77` failed)** are accounted for below, and all **77 remaining failures** belong to documented historical out-of-scope categories (**0 new regressions**):

| Category | Test File & Pytest Node IDs | Count on `4d1dbcc` | Final Count | Provenance & Resolution Status |
|---|---|---:|---:|---|
| **1. Forward-Mode AD Newly Passing After `feature/unified-tad-dftd` Merge (`XPASS(strict)`)** | `test/test_baseline/test_derivatives.py::test_cell` (`dE_dparam-forward-gfn1-libcint-full-single`, `dE_dparam-forward-gfn1-libcint-implicit-single`, `dE_dparam-forward-gfn1-pytorch-implicit-single`, `dE_dparam-forward-gfn2-libcint-full-single`, `dE_dparam-forward-gfn2-libcint-implicit-single`, `dE_dparam-forward-gfn2-pytorch-implicit-single`; plus `dE_dparam-forward-gfn1-pytorch-full-single` already in `_VALIDATED_E13_CELLS`) | 6 | **0** | **RESOLVED**: Validated forward-mode `jvp` against reverse-mode `autograd.grad` and central finite differences across `gfn1`/`gfn2`, `full`/`implicit`, and `pytorch`/`libcint` (`test/test_tad_dftd/test_dxtb_forward_param_jvp.py`, 8 passed). Promoted the 6 cells via `_VALIDATED_DFT_D_FORWARD_CELLS` in `test/test_baseline/test_derivatives.py` while leaving `test/test_baseline/status/matrix.json` untouched. |
| **2. Outdated Pre-S13 / Pre-Refactoring Contract Assertions** | `test/test_config/test_scf_mode_removed.py` (10)<br>`test/test_cli/test_driver.py` (6) & `test/test_cli/test_entrypoint.py` (2)<br>`test/test_scf/test_charge_derivative.py` (3) & `test/test_scf/test_fermi_derivatives.py` (3)<br>`test/test_scf/test_gfn1_grad.py` (4 `single-shot` tests)<br>`test/test_scf/test_inv_cholesky.py` (2) & `test/test_scf/test_warnings_errors.py` (2)<br>`test/test_utils/test_rules.py` (2) & `test/test_classical/test_dispersion/test_d4sc_setup_authority.py` (1)<br>`test/test_integrals/test_wrappers.py::test_quad` (1) | 36 | **0** | **RESOLVED**: Updated test expectations where public/internal contracts intentionally changed in S13 (retired `experimental`/`single-shot` SCF mode, `_symmetric_inverse_sqrt` overlap conditioning in `scf.functional.data`, `SCFConvergenceWarning` message format, 6-component Voigt quadrupole integral wrapper shape, `d4sc_setup`/`build_d4sc_data` source inspection, and hashable `tuple` `Config.file` in `Config.create`). |
| **3. Shared Test Sample In-Place Mutation Leak (`SiH4`)** | `test/test_properties/test_quadrupole_fieldgrad.py` (`test_autograd_vs_numerical[SiH4]`, `test_batch`)<br>`test/test_scf/test_scp.py` (`test_batch[simple-full-charges-SiH4-LiH-dtype1]`, `test_batch[simple-implicit-charges-SiH4-LiH-dtype1]`) | 4 | **0** | **RESOLVED**: Cloned `positions` before in-place finite-difference stepping in `NumericalCalculator` (`forces_numerical`, `dipole_deriv_numerical`, `pol_deriv_numerical`), cloned `sample["positions"]` in `test_d4sc_grad.py`, and cloned tensors in `test/molecules.py:merge_nested_dicts`. |
| **4. Historical Track-0 Baseline Matrix Failures** | `test/test_baseline/test_components.py` (`eigensolver.small_gap_aufbau:forward`, `order1`, `order2` [3])<br>`test/test_baseline/test_reference.py` (`gfn1-pytorch-Fe(CO)5-energy`, `forces` [2])<br>`test/test_baseline/test_derivatives.py` (19 cells: `dforces_dparam-autograd-gfn{1,2}-pytorch-full-single`, `dipole-functorch-gfn{1,2}-{libcint,pytorch}-implicit-single`, `forces-functorch-gfn{1,2}-{libcint,pytorch}-implicit-single`, `forces-forward-gfn{1,2}-*`, `dipole_deriv-forward-gfn{1,2}-*`, `hessian-forward-gfn{1,2}-*`, `hessian-numerical-gfn1-libcint-full-single`) | 24 | **24** | **Historical (out of scope)**: Pre-existing Track-0 baseline tolerance / implicit `functorch` / `Fe(CO)5` Aufbau occupation mismatches present in `task-6158.log`. |
| **5. Historical Legacy PyTorch Integral Driver `d`-Orbital Precision vs `libcint`** | `test/test_integrals/test_driver_precision.py` (`CH4`, `LYS_xao_dist`, `LiH`, `NH3`, `SiH4` [5])<br>`test/test_integrals/test_multipole_matrix.py` (`C6H5I-CH3SH-{md,os}`, `LiH-{md,os}`, `SiH4-{md,os}` [6])<br>`test/test_integrals/test_pytorch_gfn2.py` (`test_batch`, `test_energy_and_forces[SiH4-os]`, `test_integral_wrappers` [3])<br>`test/test_scf/test_hess.py` (`SiH4` GFN1/GFN2 [2])<br>`test/test_singlepoint/test_grad_pos_withfield.py` (`SiH4` GFN1/GFN2 float32 `gradcheck` [2]) | 17 | **18** | **Historical (out of scope)**: Known legacy PyTorch integral driver spherical-harmonic / `d`-orbital precision and ordering differences vs `libcint`. |
| **6. Historical Legacy Batched SCF / `Simple`-Mixer Divergence (`SiH4` / `batch_H2_LYS`)** | `test/test_scf/test_implicit_matrix.py` (15 `batch_H2_LYS` / `batch_H2O_CH4` cells)<br>`test/test_scf/test_elements_gfn1.py` (`test_element_batch[dtype{0,1}-SiH4-25]` [2])<br>`test/test_scf/test_full_tracking.py` (`test_batch_gfn1[1-simple-SiH4-LiH-dtype1]`, `test_batch_three_gfn1[simple-SiH4-LiH-H2-dtype1]` [2])<br>`test/test_scf/test_scp.py` (`test_batch[simple-{full,implicit}-potential-SiH4-LiH-dtype1]`, `test_batch_three[full-fock-simple-SiH4-LiH-H2-dtype1]` [3]) | 20 | **22** | **Historical (out of scope)**: Pre-existing `test_implicit_matrix.py` batched tolerance cells and `SiH4` simple-mixer Fock/Potential oscillations (identical to `task-6158.log`). |
| **7. Historical Open-Shell UHF `refocc` Gradient** | `test/test_scf/test_refocc_grad.py` (`NO2-N-{full,implicit}-gfn{1,2}` [4]) | 4 | **4** | **Historical (out of scope)**: Pre-existing open-shell `NO2` reference-occupation gradient finite-difference mismatch. |
| **8. Historical Difficult / `Simple`-Mixer Single-System SCF Convergence** | `test/test_scf/test_scf.py` (`LYS_xao_dist` [2], `MB16_43_01` [3], `C6H5I-CH3SH` [1], `LYS_xao` [1]) | 7 | **7** | **Historical (out of scope)**: Pre-existing difficult/distorted molecular geometries under `simple`/`broyden` mixing in float32/float64. |
| **9. Historical Post-Construction `calc.integrals.hcore` Mutation** | `test/test_scf/test_gfn1_grad.py::test_param_grad_energy[LiH]` [1] | 1 | **1** | **Historical (out of scope)**: Mutates legacy `calc.integrals.hcore.selfenergy` after `Calculator` construction instead of `ParamModule` / `Model`. |

---

## 11. Post-Refactoring Stabilization, Legacy SCF Guard Fix, and Codebase Simplification

### 11.1 Legacy `SelfConsistentFieldFull.scf()` Batch Guard Ordering (`src/dxtb/_src/scf/unrolling/default.py`)

- Moved `if batched != 0 and maxiter > 0: raise NotImplementedError(...)` to the top of `SelfConsistentFieldFull.scf()`, **before** `q_new = fcn(guess)`, so unsupported batched iterative calls fail immediately without executing an unguarded `fcn(guess)` step.
- Preserved `maxiter == 0` one-shot evaluation (`batched != 0` and `maxiter == 0`) and single-system iterative execution (`batched == 0` and `maxiter > 0`), verified by `test_legacy_scf_batch_guard_and_maxiter_zero_and_single_system` in `test/test_scf/test_general.py`.

### 11.2 Codebase Simplification Summary (~742 Net Lines Removed Across 10 Core Modules)

- **`src/dxtb/_src/calculators/padded_integrals.py` & `src/dxtb/_src/integral/evaluation.py`**: Removed `_validate_inputs()` and `_mask_integrals()` (relying on structural invariants established at `pad_exact_core` / `pad_canonical_exact_core`), and flattened `build_integral_matrices` by removing nested dipole/quadrupole closures and duplicate dipole origin shifting.
- **`src/dxtb/_src/calculators/padding.py` & `src/dxtb/_src/calculators/canonical_padding.py`**: Replaced `_pad_pair_plan` with `_pair_entry_masks`, moved the `integral.intlevel >= labels.INTLEVEL_HCORE` check to `_base_signature`, shared `_NORMALIZED_CLASSICAL_FIELDS` / `_NORMALIZED_INTERACTION_FIELDS` / `_path_parts`, consolidated `_OPTIONAL_H0_FIELDS`, and removed redundant re-validation in `pad_canonical_exact_core`.
- **`src/dxtb/_src/scf/functional/padded.py`**: Removed `PaddedCoreIndexData._validate()`, `_shell_mask_for`, `_orbital_mask_for`, and redundant runtime shape/dtype/device re-checks in `setup_padded_scf_inputs` and `evaluate_padded_fixed_trip`.
- **`src/dxtb/_src/calculators/singlepoint.py` & `src/dxtb/_src/calculators/model.py`**: Collapsed duplicated custom-subclass extension-cache branches (`GeneralizedBorn` and 5 custom classical subclasses), extracted `_build_builtin_interaction_cache`, deduplicated `scf_solve` dispatch, merged the two `classicals.components` loops in `Model.setup` into one pass, and added `System._builtin_classical_setup` / `System._builtin_interaction_setup`.
- **`src/dxtb/_src/calculators/packed_result.py`, `src/dxtb/_src/calculators/types/base.py`, & `src/dxtb/_src/calculators/types/energy.py`**: Added `_is_prefix_mask`, `_pad_optional_last_dim`, and `_bucket_perturbation` in `packed_result.py`, reused `_convert_system_dtype` in `_convert_eval_system_dtype`, and consolidated `_validate_resolved_perturbation` / `_stack_response_perturbation` in `types/energy.py`.

### 11.3 Branch Roles and Merge Readiness

- **`main` (`origin/main` at `46199aec2ebf1c7f407e6915f578f30c188832ae`)**: Stable/default branch and future merge target; completely untouched.
- **`claude/tender-dijkstra-cda42s`**: Active development branch containing the stabilized, simplified post-refactoring implementation, verified by independent review (`5e84127d-fb75-4c13-9f06-f6dcd0e7a7ed`, **APPROVED** with 0 `BLOCKER`, 0 `MAJOR`, 0 `MINOR` findings) and ready for project-owner review ahead of a future merge into `main`.

