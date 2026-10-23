# E6bimpl.1b-F — Mixed GFN1 SCF final acceptance

Status: **complete at the private GFN1/PyTorch mixed-System boundary**.
Overall E6bimpl remains **open** for GFN2 mixed execution and remaining
component, compatibility, and production-route acceptance.

Starting SHA: `1b2dafe3b65897ac048b99dc9bf59ae828672f26`.

This package did not activate `Calculator(batch_mode=1/2)`, change the packed
Calculator route, or add GFN2 mixed-System evaluation. It uses the official
`tad-dftd3==0.7.0` and `tad-mctc==0.9.1` environments.

## Implementation boundary

The mapped call still consumes independently created, canonically padded
`PaddedExactSystemCore` values. The physical evaluation runs through
`torch.func.vmap(evaluate_padded_fixed_trip)` and the shared production
`_solve_representation(..., eager=False)` controller. No mutable System,
IndexHelper, InteractionList, per-System parameter table, or per-System
physics loop was introduced inside mapped execution.

The fixed-trip controller now has an opt-in `SCFSolverTrace` emitted from its
existing initialization and transition operations. The trace contains map
inputs and raw outputs, mixed iterates, spectra, density, charges, residuals,
active flags, iteration counts, damping, Fermi state, and tensor-valued mixer
history. It is absent by default and does not change the public `Result`.

For padded execution only, the shared controller receives explicit state and
residual projections. Charge and potential states mask fictitious orbital and
atom multipoles; Fock states retain the existing inactive-orbital diagonal
convention while clearing padded couplings. Residual norms exclude inactive
entries and PADNZ row tails. The same projection is applied before the first
padded map and after each mixed transition. Ordinary single-System and eager
Calculator paths retain their existing projection and convergence policy.

The private evaluator prepares explicit shared electric-field and
field-gradient call data using the existing field kernels. It validates the
`(3,)` and `(3, 3)` shapes, dtype/device, and dipole/quadrupole integral
requirements. Field inputs are shared closures over the mapped lanes, matching
the packed-call contract. Existing custom same-label conflict checks remain
owned by the original System/Calculator input boundary; arbitrary callback
interactions are not represented by the private exact-core evaluator and
remain unsupported there.

Before mapping, `evaluate_padded_systems()` now rejects duplicate or missing
source IDs, duplicate maps, source-map/mask mismatches, malformed atom/shell/
orbital maps, and mismatched padded position shapes/dtypes/devices. Source IDs
are identifiers, not dense lane indices. The regression uses `(4, 19)` and
reversed `(19, 4)` while preserving input and decoded Result order.

## Supported GFN1 matrix

| Family/configuration | Classification | Evidence |
| --- | --- | --- |
| GFN1 / PyTorch / float64 / FULL | Supported | H2/H3+, H2/H3 open shell, H2/H2O, reversed H2O/H2, H2O/CH4 |
| Charge SCP, linear mixer | Supported | Full decoded Result and trajectory parity |
| Charge SCP, Anderson mixer | Supported | Result and trace parity; inactive-state adversarial test |
| Potential SCP, linear and Anderson | Supported | Serial Result parity and inactive-state projection |
| Fock SCP, linear and Anderson | Supported | Serial Result parity and inactive-state projection |
| Zero and finite electronic temperature | Supported | Status/Result and finite-temperature trajectory parity |
| Positive fixed-trip budgets and `maxiter=0` | Supported | Serial/mapped Result and status parity |
| Shared electric field | Supported | H2/H2O both orders; energy, density, charge, reverse and forward field response |
| Shared electric-field gradient | Supported | H2/H2O both orders; energy, density, quadrupoles, reverse and forward response |
| Field and field-gradient together | Supported | Combined serial/mapped parity |
| Default GFN1 Repulsion and D3 | Supported | Component labels/order/values and total Result parity; D3 `s6` gradient retained |
| Default GFN1 ES2 and ES3 | Supported | Active map effects in Result, charge/density/potential and ES2 parameter gradient |
| AES2, D4SC, ALPB/GeneralizedBorn descriptors | Unsupported by this private evaluator | Captured descriptors are rejected explicitly; no contribution is silently skipped |
| Arbitrary custom interaction/classical callbacks | Out of scope | They are not part of the exact registered numerical descriptor boundary; route preflight remains required before any future activation |
| Non-default GFN1 classical families | Not accepted by this package | Their private padded public-Result parity is not established here |
| GFN2 mixed Systems | Out of scope | Next E6bimpl stage |

Other registered classical kernels remain dispatched by the shared exact
descriptor evaluator, but this acceptance does not claim mixed-System parity
for non-default GFN1 component configurations without a production fixture.
The required default Repulsion and D3 contributions are retained exactly
once. Custom callbacks are never silently treated as supported by vmap.

## Trajectory parity

Independent serial production `System.singlepoint()` calls supplied the
reference. A test-only wrapper enabled tracing on the same single-System
fixed-trip solver; mapped traces came from the same controller under vmap.
The selected H2/H3+, H2/H2O, and Anderson cases compare, after explicit
canonical atom/AO selection:

- initial state, map inputs, raw maps, and mixed iterates;
- Hamiltonians, density matrices, and physical charges;
- residual L2/max-norm pairs, active flags, and iteration counts;
- damping values and Anderson history tensors;
- finite-temperature Fermi status and residual where present.

The asserted maximum absolute tolerances are `2e-9` for state/mixer tensors,
`2e-8` for Hamiltonian/density traces, and `2e-10` for residual/controller
values. The independent serial and mapped trajectories pass these bounds.
The H2/H2O fixture also has distinct lane activity at a shared iteration:
one lane becomes inactive while the other remains active. The fixed-trip
controller still executes the configured static trip budget, and each lane
retains its independently selected state and status.

An adversarial Anderson regression injects `999` into inactive entries of the
initial charge, potential, or Fock state. The mapped Result, residual,
convergence, mixer history, and selected trajectory match the clean run for
all three representations. This demonstrates that inactive state dimensions
do not enter physical residual norms or damping/convergence decisions.

## Components, fields, and Result decoding

Default GFN1 comparisons use independent production Systems. The mapped
default contribution ordering and atom-resolved values retain Repulsion and
official-release D3; ES2 and ES3 remain active in the SCF maps. The registered
ES2 `element.H.gam` and D3 `s6` Model-owned derivatives retain serial-sum
parity in the existing padded suite.

Explicit field tests compare both H2/H2O lane orders. They establish energy,
density and charge parity, nonzero field response, `jacfwd` response parity,
and equality of zero-field and no-field calculations. Field-gradient tests
compare energy, density, physical quadrupole charges and `jacfwd` response.
The combined field/gradient test compares against serial System calls. Missing
integral levels and invalid shapes fail with the established error category.

Decoded Results compare pytree specification, energy, SCF energy, free energy,
classical labels/order/values, charges, potential, density, orbital energies,
occupations, Hamiltonian, overlap, Hcore, overlap normalization, optional
multipole integrals, convergence, residual, iterations, and both public
SCFStatus fields. Coefficients retain correct physical shape and dtype; density
and spectral projectors are the gauge-invariant spectral comparisons. Missing
integrals remain `None`. Explicit source-to-canonical maps restore atom and AO
order, including noncontiguous canonical AO layouts.

## Validation

The full padded mixed-spectral module completed in all five CPU profiles. Each
run used official D3 0.7.0, mctc 0.9.1, `OMP_NUM_THREADS=1`, and
`pytest -q -n logical --dist worksteal`:

| Python | PyTorch | Result |
| --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 51 passed |
| 3.11.15 | 2.8.0+cpu | 51 passed |
| 3.12.10 | 2.11.0+cpu | 51 passed |
| 3.12.10 | 2.13.0+cpu | 51 passed |
| 3.14.8 | 2.14.1+cpu | 51 passed |

Primary broader selection: 360 passed, 1 failed, 23 warnings in 168.57 s.
The sole failure was
`test/test_scf/test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`.
It fails at the required starting SHA with the same dependencies: maximum
absolute difference `2.5006982595776606e-11` against `atol=2e-13` in the
functional-versus-legacy position gradient. It is therefore inherited, not a
new padded-path regression. The complete architecture guard
`test_functional_map_has_no_legacy_stateful_hooks` passed after replacing the
source-map validation's initial `.tolist()` check with tensor-mask equality.

The broader command was:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6bimpl0_padded_integrals.py \
  test/test_calculator/test_e6bimpl1a_padded_spectral.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e6a4_transform_acceptance.py \
  test/test_calculator/test_field_inputs.py \
  test/test_scf/test_functional_solver.py \
  test/test_scf/test_functional_anderson.py \
  test/test_scf/test_functional_map.py \
  test/test_scf/test_e6bcompat_policy.py
```

PyTorch emitted known transform batching performance warnings for
finite-temperature `matrix_exp`/`count_nonzero` operations and Python 3.14
`torch.jit.script` warnings; tests completed. CUDA and libcint whole-System
vmap are outside this CPU/PyTorch package.

Black, isort, compileall and `git diff --check` passed. Ruff and mypy were not
available in the primary environment.

## Decision and remaining work

E6bimpl.1b-F closes the original private GFN1/PyTorch mixed fixed-trip
boundary: trajectory parity, lane-local controller behavior, padded residual
safety, default physical contributions, shared fields, decoded Results, and
derivative controls are covered. The one broad-suite Anderson gradient failure
is matched on the starting SHA and remains owned by the prior functional
Anderson discrepancy work.

Overall E6bimpl remains open. Next work is GFN2 mixed-System execution and
remaining component-family compatibility. E6broute alone may activate public
Calculator batching after its independent prerequisites pass. This package
does not alter that routing.
