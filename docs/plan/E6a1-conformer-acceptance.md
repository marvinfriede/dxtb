# E6a.1 — Public conformer method, Result, and SCFStatus acceptance

Status: scoped GFN1/GFN2 public conformer slice implemented and validated;
overall E6a remains open.

## Baseline and change

This package starts at `4702e20823e77260580a414eabd16b7b02754737`.
The accepted API remains
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions_batch)` over one
production System built through `Model.setup()`, using the PyTorch integral
driver and default `out_dims=0`. No mixed-System stack, packed Calculator
route, private result surrogate, or alternate public API is involved.

The first mapped explicit charge/spin reproduction failed in
`filling.get_alpha_beta_occupation`: eager `.any()` checks branch on tensor
values to validate spin/electron consistency. Those checks are retained by
default. The fixed-trip path calls `get_refocc(..., validate_spin=False)` after
the ordinary shape validation, so valid conformer-varying charges and spins
can flow through tensor arithmetic without a data-dependent Python branch.
The eager Calculator and legacy SCF paths keep the default validation. As with
other value-dependent checks, callers of transformed System evaluation must
provide physically consistent charge/spin values; the fixed-trip transform
path does not raise eager Python exceptions for invalid tensor contents.

The broader tests exposed a D3 `AtomicC6` vmap alignment issue: atomic
numbers were constant while reference weights carried the conformer batch
dimension. The D3 adapter now adds a zero-valued integer tensor with the
geometry shape to the captured atomic numbers. This carries the same mapped
axis into AtomicC6 without changing atomic-number values or their gradients.
The full D3 contribution remains enabled in GFN1 and GFN2 tests. This does
not address the separate E1.3 forward-mode/JVP limitation.

## Supported public method/configuration evidence

All Systems use independently constructed production `ParamModule`,
`Config`, `Model`, and `Model.setup()` values with the PyTorch integral driver
and float64 CPU tensors. Fixed-trip budgets are positive except where the
existing E6a.0 zero-budget test is explicitly used.

| Method / composition | SCF configuration | Conformer count | Evidence |
| --- | --- | ---: | --- |
| GFN1 / H2 | FULL, linear mixer, charge SCP, zero K | 3 | Explicit per-conformer charge/spin; Result parity |
| GFN1 / H2O | FULL, linear mixer, charge SCP, zero K | 2 | Complete Result parity; batch size differs from atom count |
| GFN1 / H2O | FULL, linear mixer, charge SCP, zero K, `maxiter=1` | 2 | Outer nonconvergence and public status parity |
| GFN2 / H2 | FULL, Anderson, Fock SCP, zero K | 3 | Complete Result parity and three-conformer coverage |
| GFN2 / H2O | FULL, Anderson, Fock SCP, zero K | 2 | Complete Result parity |
| GFN1 / H2 | FULL, linear mixer, charge SCP, 300 K | 2 | Existing E6a.0 serial/vmap Result and status parity |
| GFN1 / H2 | FULL, linear mixer, charge SCP, `maxiter=0` | 2 | Existing E6a.0 one-shot status contract |

The new Result parity helper checks the registered pytree specification,
leaf order/count, tensor shape, dtype, device, and values against independent
serial public System calls. Every mapped tensor has exactly one leading
conformer axis. Classical labels/order, nested status and charge/potential
children, and optional fields are compared as part of the tree. Hcore-level
results include overlap and core Hamiltonian; E6a.0 tests cover dipole and
quadrupole integral levels, per-conformer fields and field gradients, and
their result fields. E6a.1 also covers captured shared fields and field
gradients, per-conformer and explicitly shared charge/spin values. Existing
integral-level validation remains in force.

## Derivatives and status

Reverse-mode position gradients are compared with independent serial
gradients for both GFN1 linear and GFN2 Anderson/Fock configurations. The
public energy-vector Jacobian has shape
`(nconformer, nconformer, natom, 3)`; diagonal blocks match serial gradients
and off-diagonal blocks are zero. A differentiable H Repulsion parameter is
owned by the original `ParamModule`; the gradient of the mapped total energy
matches the sum of serial gradients. No parameter table is copied into the
System.

The tests compare public `SCFStatus` tensor leaves and outer convergence,
iteration count and residual for converged/fixed-budget cases, and for a
deliberately insufficient positive budget. Existing B5.6 tests independently
exercise the public failure-code projection (invalid temperature, too many
electrons, Fermi nonconvergence and electron count) and existing trajectory
tests distinguish trajectory from final-spectrum failure. Full public
System-level integration fixtures for every failure category, especially
Fermi failure and too-many-electrons, remain outstanding; unit projection
coverage is not claimed as complete integration acceptance.

## E1.3 and remaining boundary

The existing full-method GFN1 `jacfwd` blocker test still reaches upstream
`tad-dftd3` `AtomicC6`, which lacks a JVP. The test remains an expected blocker
characterization, not a positive forward-mode result. When upstream JVP
support lands, replace it with positive `jacfwd`/JVP regressions on the full
enabled method.

The supported boundary here is fixed-System public conformer `vmap` for the
listed GFN1/GFN2 PyTorch configurations, with reverse-mode geometry and one
parameter-gradient check. This does not accept whole-System libcint mapping,
arbitrary custom interactions, every SCF status failure as a public
integration case, all derivative orders, or the complete E1 method/family
matrix. E6a remains open. E6bimpl/E6bcompat/E6broute, E7, and E4.6 retain
their separate ownership; C8b and E6bcompat-policy remain complete.

## Validation

The focused command was:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py
```

It passed 24 tests in each available CPU environment: Python 3.10 / Torch
2.6.0, Python 3.11 / Torch 2.8.0, Python 3.12 / Torch 2.11.0, Python 3.12
/ Torch 2.13.0, and Python 3.14 / Torch 2.14.1. The primary broader selection
including the under-budget status regression passed 339 tests and reproduced
11 established failures; the added GFN2 Anderson-gradient regression then
passed separately. The failures are the ten GFN1 production/legacy
parity cases listed in the E6a.0 report and
`test_anderson_solver_real_position_parameter_and_field_gradients`, whose
`2.5006982595776606e-11` discrepancy against `2e-13` was already matched to
the package base. No new failure appeared in that selection.

The broader command also included Result/status, trajectory, E0, functional
solver/Anderson/eager, E6bcompat, field, packed singlepoint, PyTorch setup and
multipole controls:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal --tb=short \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_result_pytree.py \
  test/test_scf/test_public_scf_status.py \
  test/test_scf/test_functional_trajectory_status.py \
  test/test_scf/test_e0_iteration_compat.py \
  test/test_scf/test_functional_solver.py \
  test/test_scf/test_functional_anderson.py \
  test/test_scf/test_functional_eager_compatibility.py \
  test/test_scf/test_e6bcompat_policy.py \
  test/test_calculator/test_field_inputs.py \
  test/test_calculator/test_singlepoint_core.py \
  test/test_integrals/test_pytorch_setup_node.py \
  test/test_integrals/test_multipole_options.py
```

The independent C8b/parameter-tree and D3 numerical regression group passed
141 tests:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_c8b_schema_preflight.py \
  test/test_calculator/test_c8b3a_mixed_size_padding.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b3_mixed_size_preflight.py \
  test/test_classical/test_dispersion/test_d3.py \
  test/test_classical/test_dispersion/test_d3_core_node.py \
  test/test_classical/test_dispersion/test_grad_pos.py \
  test/test_classical/test_dispersion/test_grad_param.py
```

Primary environment: Python 3.14.8, PyTorch 2.14.1+cpu, tad-mctc 0.9.1,
tad-libcint 0.3.0, PySCF 2.14.0. CUDA and a no-libcint profile were
unavailable. E1.3 and the status-integration limitations above remain open.
