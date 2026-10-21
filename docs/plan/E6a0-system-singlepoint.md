# E6a.0 — Transform-safe System.singlepoint slice

Status: implementation slice complete; E6a remains open.

This note records the first executable fixed-trip boundary beneath the
existing public `System.singlepoint(positions, ...)` API. It does not close
the full E6a method/derivative matrix, satisfy full E1, or promise whole-System
libcint transforms.

## Baseline and reproduced failure

The implementation starts from `3a0b569e1f201914381851fb709131a3fd98526a`.
On Python 3.14.8 / PyTorch 2.14.1+cpu, an independent GFN1/H2 System built
through `Model.setup()` with the PyTorch integral driver reproduced the exact
public call failure:

```python
torch.func.vmap(system.singlepoint, in_dims=(0,))(positions_batch)
```

The call completed input validation, classical evaluation, PyTorch integral
construction, and interaction preparation. It failed before SCF iteration in
`scf.guess.spread_charges_atomic_to_orbital`, whose charge-conservation check
used `torch.any(...)` in Python control flow. The exception was PyTorch's
`RuntimeError` that vmap does not support a Tensor in data-dependent control
flow. The charge spreading arithmetic is retained; the numerical public
System path skips only this eager assertion because the fixed structural
spread is itself the operation under test. The default eager Calculator path
retains the assertion.

## Execution and ownership

`System.singlepoint` now selects `system_singlepoint`, which enters the shared
singlepoint setup and Result code with the fixed-trip controller selected
statically. Eligible FULL SCF uses `run_system_fixed_trip` and the existing
`run_prepared_functional_full(..., eager=False)` path. The transform-visible
path uses the existing SCF map, spectral/Fermi evaluation, finalization,
mixer formulas, and tensor-valued solver diagnostics. It does not inspect
convergence/status tensors to warn or raise. The configured trip count is
fixed for direct and mapped System calls.

The legacy internal `singlepoint` compatibility entry and Calculator's
`_singlepoint_eager` adapter retain the eager route. `Calculator.singlepoint`
therefore keeps early stopping, convergence warnings/errors, and existing
legacy fallback dispatch. For fixed-trip-eligible calculations, both paths
share integral preparation, interactions, physical maps, and finalization;
only the controller and eager policy differ. The public B1 API did not gain a
new conformer function.

Static legacy-owned configurations (including negative temperature,
negative Fermi `maxiter`, and unsupported custom interaction routes) retain
their existing legacy path. These configurations are outside the accepted
vmap slice. Route eligibility is resolved before constructing the
transform-only initial guess. A legacy-owned route delegates directly to the
established eager adapter, which constructs and validates its own guess; it
does not first compute and discard a `validate=False` guess. Non-SCF
calculations retain their existing non-iterative solver path. No libcint
device policy or packed Calculator route changed.

Direct `System.singlepoint` behavior intentionally changes for eligible FULL
configurations: it now executes the fixed configured trips, returns tensor
`converged`, `iterations`, `residual`, and public `SCFStatus`, and does not
apply eager `force_convergence`, warning, or Fermi-error policy. Callers that
need the prior eager convergence/error policy should use
`Calculator.singlepoint`. At `maxiter=0`, the initial physical map is returned
with `converged=False` as specified by E0, even when the System config sets
`force_convergence=True`.

## Executed acceptance slice

The exact public conformer call passes for independently constructed
GFN1/PyTorch H2 Systems with positive fixed budgets at zero and finite
electronic temperature, and at `maxiter=0`. It returns the registered public
`Result` and `SCFStatus` types. The tests flatten the mapped result and compare
each tensor leaf, dtype, device, shape, optional field, classical label, and
value against serial calls through the same fixed-trip System API. No second
conformer dimension is introduced.

Reverse-mode position gradients of mapped energies match independently
evaluated serial gradients. Explicit per-conformer electric fields and field
gradients map with matching integral levels and energy parity. A direct
Calculator control confirms that eager execution still applies
`force_convergence=True` errors while direct System fixed-trip execution
returns status without raising.

The tests are in `test/test_calculator/test_e6a0_system_vmap.py`. They do not
claim GFN2 acceptance, all interaction families, all status failure modes, or
complete E6a derivative coverage.

## E1.3 and remaining boundary

Reverse-mode position differentiation passes for the tested GFN1/PyTorch
H2 path. `torch.func.jacfwd` of the full GFN1 energy still fails in
`tad-dftd3`'s `AtomicC6` custom autograd Function because it has no `jvp`
implementation. This is the known E1.3 DFT-D3 forward-mode blocker. No D3
contribution was removed to manufacture an E6a pass, and no dependency code
was changed.

Still open for full E6a: GFN2 and required method/family rows, complete
SCFStatus failure-mode integration through public `System.singlepoint`,
higher-order/forward derivatives once E1.3 is available, broader mapped
charge/spin/field combinations, custom-operation transform support, and the
remaining optional integral/result configurations. Whole-System libcint
vmap is not claimed; CUDA was unavailable. E6b mixed-System batching, E7
padding safety, E6broute, E4.6, C8b, and E6bcompat-policy retain their
separate roadmap ownership and status.

## Validation record

The dedicated E6a.0 suite passed 9 tests in each of the five available
approved CPU environments:

| Python | PyTorch | Result |
| --- | --- | --- |
| 3.10 | 2.6.0 | 9 passed |
| 3.11 | 2.8.0 | 9 passed |
| 3.12 | 2.11.0 | 9 passed |
| 3.12 | 2.13.0 | 9 passed |
| 3.14 | 2.14.1 | 9 passed |

The broader primary-environment selection covering E6a.0, Result, fields,
C8b parameter trees, public/status trajectories, functional maps, eager
compatibility, E0, functional solvers, Anderson, E6bcompat, and singlepoint
controls reported 405 passed and one failure. The failure was
`test_scf/test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`.
It was rerun on the package base SHA with the same Python/PyTorch environment
and reproduced the same gradient discrepancy (maximum absolute difference
`2.5006982595776606e-11` against the existing `2e-13` tolerance), so it is
inherited rather than introduced here.

The focused singlepoint/eager/E0 selection reported 162 passed and the ten
known GFN1 production/legacy parity failures. The exact parameter IDs in
`test_singlepoint_core.py::test_production_singlepoint_matches_legacy_full`
were `zero-k-charge-anderson`, `zero-k-potential-anderson`,
`zero-k-fock-anderson`, `zero-k-fock-simple`, `300-k-fock-anderson`,
`zero-k-potential-simple`, `300-k-charge-anderson`, `zero-k-charge-simple`,
`300-k-charge-simple`, and `300-k-potential-anderson`; these are tracked
inherited failures. A diagnostic run interrupted during collection was not
counted as a pass. The primary environment was Python 3.14.8, PyTorch
2.14.1+cpu, tad-mctc 0.9.1, tad-libcint 0.3.0, and PySCF 2.14.0. CUDA and
a no-libcint environment were unavailable. A separate supported Python/Torch
matrix beyond the five listed environments was not available. Ruff and mypy
were unavailable; formatting and compilation checks are recorded in the
package delivery report.

## E6a.0a fallback hardening

At E6a.0's accepted base `76db242734eb94d57c9d4577b5333724d5c9efee`,
instrumenting `get_guess` on an independently built GFN1/PyTorch H2 System
with negative electronic temperature observed calls with `validate=False`
and then `validate=True`. The route was legacy-owned and the first guess was
unused; the eager FULL adapter then constructed the guess it needed. The
regression test executes the real eager fallback and records that behavior.

`run_system_fixed_trip` now resolves eligibility before any iterative guess
construction. Legacy routes immediately use `run_functional_full_eager` and
retain its eager charge-conservation validation and error behavior. Eligible
functional routes compute reference occupations and one initial guess with
`validate=False`, then use the same fixed-trip solver path as before. No SCF
map, formula, convergence criterion, or public API changed.

The new regressions verify one eager-validated guess for a real negative-
temperature fallback and one transform-safe guess for an eligible functional
System call. The complete E6a.0a suite, including exact public vmap, Result and
SCFStatus parity, position gradients, fields, Calculator eager policy,
`maxiter=0`, legacy-owned negative-temperature/Fermi/non-FULL dispatch,
strict-mixer validation order, and the explicit E1.3 blocker characterization,
passed 14 tests
in each of the five CPU environments listed above. On Python 3.14/Torch
2.14.1, E4.5a/eager compatibility, E0, and direct System/Calculator controls
passed 160 tests; Result pytree, public status/trajectory, and E6bcompat-policy
controls passed 56 tests. A broader combined selection was interrupted after
one failure appeared at 88%; it is not counted as a pass. The same isolated
Anderson node,
`test_scf/test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`,
reproduced the established maximum absolute discrepancy
`2.5006982595776606e-11` against the `2e-13` tolerance. It was previously
matched against the E6a.0 package base and is not caused by this route-order
change. E1.3 remains open; its blocker-characterization test must become a
positive `jacfwd` regression after upstream `AtomicC6` JVP support is
available.
