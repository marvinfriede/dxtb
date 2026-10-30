# S2b.1 certification contract hardening

## Scope and synchronization

S2b.1 starts from the reviewed S2b candidate `fc32d652c2cea79d77c2182d76531f6c0d7f0505` and synchronizes onto the published W2c6 boundary `d943e6dea42e552e9451c1e89bf38d1e564bc0ef`. The synchronization adds the reviewed direct-RHS matrix-Fermi update and its tests. This package does not change matrix-Fermi behavior or start S3.

The independent S2b review identified four contract gaps. S2b.1 preserves the authoritative physical-certification controller and fixes only the contracts below.

## Full and core finite validity

`certification_core_metrics` and `certify_scf_core_state` remain the controller-safe path. They do not compute energy or entropy. Their `finite_ok` continues to describe the paired physical state and core certification quantities.

The full `certification_metrics` API now includes finiteness of both `electronic_energy` and total `electronic_free_energy` in its own `finite_ok`. `certify_scf_state` reuses its already-computed core metrics, adds full observables once, and includes full finite validity in its diagnostic verdict. Thus optional observable NaNs fail the full diagnostic without adding energy or entropy work to the authoritative controller.

## Implicit preparation boundary

`_prepare_scf_root` now requires an explicit certification record. It checks that `outcome.converged` agrees with `certification.certified`, requires both to be true alongside final-spectrum validity, and then retains the independent exact-parameter raw-map residual check. Missing certification raises `SCFConvergenceError`; an inconsistent outcome/certification pair raises an internal `RuntimeError`. Trajectory history remains outside physical-root acceptance.

## Exact finalization root pairing

`finalize_scf`, `finalize_scf_potential`, and `finalize_scf_fock` now require keyword-only `root_x`. They no longer substitute `map_result.mapped_x`, which is `G(root_x)` and may differ from the input that generated the map result. Ordinary solver paths pass `outcome.root_x`. The attached implicit path passes `attached_root`, the tensor actually paired with its attached map result, so the stored result preserves implicit derivative semantics.

Direct helper/test callers were updated to pass the input used for their exact map evaluation. Tests cover charge, potential, and Fock selected-root pairing, plus the attached implicit root's map and parameter derivative.

## Explicit profile tolerance policy

`SCFInternalCertificationProfile.inherit_solver_root_tolerances` replaces singleton-identity detection. The standard built-in profile opts in; copied or modified standard profiles retain the policy. Arbitrary custom profiles default to explicit profile tolerances. The unused electron-count policy field was removed; the current policy continues to set the count threshold from the selected fixed-point L2 threshold.

## Fixed-trip selected-pair consistency

The fixed-trip controller still performs its statically required candidate certifications and final selected-pair diagnostic recomputation. S2b.1 adds checks that the final stored diagnostic agrees with `converged` for successful and exhausted runs, and that terminal failure agrees with the recomputed selected-pair diagnostic. The extra recomputation remains performance debt; controller-state complexity was not increased to remove it.

## Validation

Python 3.14 / PyTorch 2.14.1 results:

| Selection | Result |
| --- | ---: |
| Certification | 61 passed |
| S1/lifecycle/E6a.2 | 259 passed |
| Implicit/root (`not slow`) | 54 passed |
| Slow NO2 potential implicit response | 1 passed |
| Anderson | 32 passed, 1 inherited failure |
| Spectral/free-energy + W2c5b/W2c6 | 156 passed |
| Mapped implicit root-pair regressions | 3 passed |
| Phase-A Group 1 | 225 passed, 1 inherited failure |
| Phase-A Group 2 | 54 passed |
| Phase-A Group 3 | 131 passed |

The only failure is `test_anderson_solver_real_position_parameter_and_field_gradients`, with
maximum absolute difference `2.5007065862503453e-11` against `2e-13`; its
tolerance and Anderson implementation are unchanged. Python 3.11 / PyTorch
2.8.0 passed 13 focused checks for full/core finite semantics, profile copying,
implicit certification, attached root pairing, and transform compatibility.
The Python 3.14 compiler emitted only the existing TorchScript deprecation
warning.

The post-fix H2O/GFN2 performance smoke used the same 40-step budget and
single-thread CPU setup. One run measured eager at `0.147 s`, 19 iterations,
and 1 core-certification call; fixed-trip measured `0.436 s`, 19 iterations,
and 41 calls (40 loop candidates plus the selected-pair diagnostic). These
single-run timings are a call-count/separation check, not a new timing baseline.

Static checks passed: compileall, Black, isort for changed production and SCF
test files, and `git diff --check`. The edited baseline diagnostics file had
an unrelated pre-existing isort import-order diff; its imports were not
changed in S2b.1.

Public `SCFStatus`, convergence thresholds, controller policy, method mathematics, and trajectory-history reporting remain unchanged. S3 is not started.
