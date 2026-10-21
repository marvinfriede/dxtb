# E6a.2 — Public SCFStatus failure-mode integration

Status: accepted at the public status-integration boundary; overall E6a remains
open.

Starting SHA: `69999037cf4d8e4ee50005839bf1d9e41b5e5c69`.

## Contract

The tested public call remains
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions_batch)` on one
production System constructed independently through `Model.setup()` with the
PyTorch integral backend. It returns the registered public `Result`; the
public `status` is `SCFStatus(trajectory_failure, final_failure)`, with int64
code tensors. `iterations`, `converged`, and `residual` remain separate outer
solver fields. A valid Fermi subsolve is not outer convergence, and an outer
nonconvergence does not imply a Fermi failure.

## Public production matrix

All production rows use float64 CPU tensors, the default GFN method
components, independent `ParamModule`/`Model.setup()` construction, three
distinct geometries, and serial references through the same public fixed-trip
System API.

| Method / composition | Temperature and budget | Input | Public status | Outer result | Evidence |
| --- | --- | --- | --- | --- | --- |
| GFN1 / H2 | 0 K, maxiter 3 | neutral singlet | trajectory 0, final 0 | fixed-trip parity; strict zero outer tolerances need not converge | public serial/vmap |
| GFN1 / H2 | 300 K, maxiter 3, default Fermi search | neutral singlet | trajectory 0, final 0 | independent of valid spectral status | public serial/vmap |
| GFN2 / H2 | 0 K, Fock/Anderson, maxiter 2 | neutral singlet | trajectory 0, final 0 | compared with serial Result | public serial/vmap |
| GFN1 / H2O | 300 K, maxiter 1 | neutral singlet | trajectory 0, final 0 | `converged=False`; iteration tensor is 2 (initial map plus one fixed trip) | public serial/vmap |
| GFN1 / H2 | 0 K, maxiter 0 | neutral singlet | trajectory 0, final 0 | `converged=False`, iterations 1 | public serial/vmap |
| GFN1 / H2 | 300 K, maxiter 0, Fermi maxiter 0, threshold 0 | neutral singlet | trajectory 0, final 3 | `converged=False`, iterations 1 | public serial/vmap; final-spectrum-only failure |
| GFN1 / H2 | 300 K, maxiter 2, Fermi maxiter 0, threshold 0 | neutral singlet | trajectory 3, final 3 | fixed-trip result returned without eager status inspection | public serial/vmap |
| GFN1 / H2 | 300 K, maxiter 2 | charges `[0, 0, -7]`, spins `[0, 0, 1]` | trajectory `[0, 0, 2]`, final `[0, 0, 2]` | per-lane mapped energy/result agrees with serial; over-capacity lane remains explicitly failed | public 3-lane serial/vmap |

Code values are `NONE=0`, `INVALID_TEMPERATURE=1`,
`TOO_MANY_ELECTRONS=2`, `FERMI_NONCONVERGENCE=3`, and `ELECTRON_COUNT=4`.
The `-7` charge row requests nine electrons, exceeding the configured
occupation capacity. The Fermi evaluator flags `TOO_MANY_ELECTRONS`; the public
result is not classified as a successful solve because both failure codes are
nonzero and outer convergence is false for that lane. Neighboring valid lanes
retain code zero. No orbital capacity is fabricated or electron count clamped
in the adapter.

For every mapped row the test compares the actual Result type and the energy,
free energy, iteration, convergence, residual, and both status fields against
independent serial public calls. The status vectors have one conformer axis,
int64 dtype, and the expected device. Existing E6a.1 tests continue to cover
the full Result pytree, optional fields, charge/spin mapping, fields,
cross-conformer position Jacobians, and parameter-gradient connectivity.

## Failure ownership and coverage limits

Finite-temperature Fermi nonconvergence is reachable through a real
production System using the supported configurable Fermi iteration limit and
threshold. With positive outer budget, its first active raw map and final
spectrum both report code 3. With `maxiter=0`, no physical outer trajectory
exists, so the public System reports trajectory code 0 and final code 3.

This final-only case exposed a status bookkeeping defect at the starting SHA:
the fixed-trip controller initialized trajectory status from its one-shot
initial map even when zero outer trips were configured. The public result
therefore reported the same failure as both trajectory and final failure.
The fixed-trip controller now omits initial trajectory status only when its
static `maxiter` is zero. The eager Calculator controller retains its prior
initial-map status and chronological error precedence. Existing Calculator
failure-order tests and new zero-/positive-budget Calculator Fermi-error tests
pass. Physical maps, Fermi equations, convergence criteria, iteration
accounting, and Calculator error handling are unchanged.

`TOO_MANY_ELECTRONS` has a public production integration row as shown above.
`FERMI_NONCONVERGENCE` also has public positive- and zero-outer-budget rows.
The following categories remain projection-only or outside the public
transform domain:

| Category / case | Classification | Reason |
| --- | --- | --- |
| `ELECTRON_COUNT` | Low-level projection only | The Fermi core checks the post-smearing electron-count residual and the existing public-status test verifies code 4, but no stable physically consistent production input was found that violates this invariant. Public integration remains unproven. |
| Trajectory failure followed by a valid final spectrum | Controlled solver/status test only | Existing `test_functional_trajectory_status.py` verifies first-failure retention and final-status independence with solver-level fixtures; no deterministic physical System fixture was found. |
| Valid trajectory followed by final-spectrum failure | Public final-only zero-budget case plus low-level projection | Public maxiter-zero exercises trajectory 0/final 3. Existing status projection tests independently exercise final-only categories. |
| Both phases failing with different categories | Low-level projection only | Existing projection tests verify separate codes and precedence; no production System fixture is claimed. |
| `INVALID_TEMPERATURE` | Projection only; negative configured temperature is legacy-owned | B5.6 verifies the tensor projection. The negative-temperature route is not in the accepted fixed-trip public domain. |
| Negative Fermi maxiter and GFN0 | Legacy-owned | They remain routed through existing eager/non-SCF behavior. E6a.1a GFN0 spin validation remains intact; no GFN0 vmap is claimed. |

Failure precedence, trajectory first-failure retention, lane-wise projection,
and the five public enum values remain covered by the existing
`test_public_scf_status.py` and `test_functional_trajectory_status.py` tests.
They are not reimplemented in this package's public integration module.

## Calculator policy and remaining scope

`Calculator.singlepoint` remains the eager policy owner. New controls confirm
that Fermi nonconvergence still raises at both zero and positive iteration
budgets. Existing E6a.0 controls retain force-convergence and nonconvergence
behavior; existing E4.5 compatibility tests retain negative-temperature,
negative-Fermi-maxiter, and legacy fallback routing.

GFN1/H2O, GFN2/H2, and GFN1/H2 status integrations pass for the listed
PyTorch configurations. The E1.3 full-method forward-mode blocker remains in
upstream tad-dftd3 `AtomicC6` and is still characterized by the existing
expected blocker test. E6a remains open for its remaining method/component,
derivative, status-failure, and E1.3-dependent acceptance gates. This package
does not claim public GFN0 vmap, whole-System libcint vmap, or production
mixed-System execution.

## Validation

The five-environment focused command was:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_scf/test_public_scf_status.py \
  test/test_scf/test_functional_trajectory_status.py
```

It passed 62 tests in each available CPU environment: Python 3.10.22 / Torch
2.6.0+cpu, Python 3.11.15 / Torch 2.8.0+cpu, Python 3.12.10 / Torch
2.11.0+cpu, Python 3.12.10 / Torch 2.13.0+cpu, and Python 3.14.8 / Torch
2.14.1+cpu. Each environment had tad-mctc 0.9.1, tad-libcint 0.3.0, and
PySCF 2.14.0. CUDA and a no-libcint profile were unavailable.

The broader primary-environment selection was run as:

```bash
OMP_NUM_THREADS=1 /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/python \
  -m pytest -q -n logical --dist worksteal --tb=short \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_scf/test_nonselfconsistent.py \
  test/test_wavefunction/test_filling.py \
  test/test_wavefunction/test_functional_filling.py \
  test/test_wavefunction/test_fermi_derivatives.py \
  test/test_scf/test_fermi_derivatives.py \
  test/test_scf/test_e0_iteration_compat.py \
  test/test_scf/test_functional_eager_compatibility.py \
  test/test_scf/test_functional_trajectory_status.py \
  test/test_scf/test_e6bcompat_policy.py \
  test/test_scf/test_public_scf_status.py \
  test/test_calculator/test_result_pytree.py \
  test/test_calculator/test_singlepoint_core.py \
  test/test_classical/test_dispersion/test_d3.py \
  test/test_classical/test_dispersion/test_d3_core_node.py \
  test/test_classical/test_dispersion/test_grad_pos.py \
  test/test_classical/test_dispersion/test_grad_param.py
```

It reported 662 passed, 2 skipped, and these 10 inherited failures in
`test_production_singlepoint_matches_legacy_full`:

- `zero-k-charge-simple`
- `zero-k-charge-anderson`
- `zero-k-fock-simple`
- `zero-k-fock-anderson`
- `zero-k-potential-anderson`
- `zero-k-potential-simple`
- `300-k-charge-simple`
- `300-k-charge-anderson`
- `300-k-potential-anderson`
- `300-k-fock-anderson`

No new failure was observed. Black, isort, compileall, and `git diff --check`
passed. Ruff and mypy were unavailable.
