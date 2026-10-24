# E7a.1 — Third-order derivative localization and correction

## Scope and baseline

- Starting SHA: `6b95898d8efdd8ca0431ee75670f4b683f6097bc`.
- Final implementation SHA: `e433341` (the production/test acceptance commit;
  this report's SHA-only follow-up commit is recorded in delivery).
- Scope: private mapped GFN1/GFN2 third-order energy derivatives. Public
  Calculator routing, broader E7, performance work, and legacy retirement are
  unchanged.
- `AGENTS.md` was left untouched and unstaged.

The two inherited failures were run against the starting SHA before changing
production code. The retained integration tests use independent production
Systems, float64, neutral closed-shell fixtures, one fixed linear SCF trip,
zero electronic temperature, and identical serial/mapped energy functions.
The full scalar total energy is differentiated, including SCF and configured
classical contributions.

| PyTorch | Fixture and differentiated lane | Serial D3 | Mapped D3 | Absolute error | Relative error* |
|---|---|---:|---:|---:|---:|
| 2.6.0 | GFN1 H2/H2O, water lane | -1.3799981115381825 | -1.3629367193659156 | 1.7061392173e-2 | 1.2363e-2 |
| 2.6.0 | GFN2 H2/LiH, LiH lane | 4.6024227236975856e-2 | 4.5988713438981600e-2 | 3.5513797994e-5 | 7.7163e-4 |
| 2.14.1 | GFN1 H2/H2O, water lane | -1.3799981115381865 | -1.3629367193659208 | 1.7061392173e-2 | 1.2363e-2 |
| 2.14.1 | GFN2 H2/LiH, LiH lane | 4.602651788402370e-2 | 4.598862567781950e-2 | 3.7892206204e-5 | 8.2327e-4 |

\* Relative error is absolute error divided by the absolute serial derivative.
The acceptance tolerance remained `atol=3e-5, rtol=3e-5`.

The deterministic coordinate direction has entries `1..n` reshaped to the
physical coordinate tensor and divided by its Euclidean norm. The exact same
physical tangent is copied through the source-to-canonical atom map; inactive
coordinate slots receive zero. No padding-dependent renormalization is used.
The tests also use three distinct deterministic directions, cross-System
directions, inactive-coordinate directions, and reversed source order.

## Localization

The diagnostic ladder compared the same physical energy at four boundaries:

1. Original independently prepared serial System.
2. Canonically padded System evaluated without `vmap`.
3. The same padded System in a registered batch of size one.
4. The System in the original heterogeneous mixed bucket.

The first issue depended on the derivative composition. Nested all-forward
differentiation through `torch.linalg.solve` disagreed with reverse-over-forward
and an HVP finite difference. A minimal symmetric positive-definite solve gave
the following second directional derivatives on the five supported PyTorch
versions:

| Formulation | Nested forward | Reverse-over-forward | HVP finite difference |
|---|---:|---:|---:|
| `torch.linalg.solve(A, b)` | -0.07223222362712572 | -0.14446444725425140 | -0.1444638676296 |
| Cholesky factorization and `torch.cholesky_solve` | -0.14446444725425167 | -0.14446444725425167 | -0.1444638676294 |

The Fermi matrix function and the matrix-sign projector solve symmetric
positive-definite systems. They now use a local Cholesky solve helper for
those systems. The projector computes `x**-1` as the positive-definite solve
`(x @ x) y = x`, valid for the symmetric invertible `x` used by the sign
iteration. This is a dxtb-side formulation; no dependency was changed.

A second issue remained for GFN1 when comparing the corrected reverse-over-
forward derivative with the original triangular Cholesky orthogonalizer. The
serial and mapped paths use different canonical AO orderings. Their generalized
spectra and physical density agree, but `SCFInputs.inverse_cholesky` differs
under those AO permutations. That is the earliest divergent input to the
spectral boundary; the orthogonalized Hamiltonian used by
`spectral._prepare_matrix` and the matrix-sign density projector then retain a
third-derivative dependence on that representation. A smooth,
permutation-covariant symmetric inverse square root removes that
source-order dependence. It is computed by a fixed 24-step Newton–Schulz
matrix iteration after Frobenius scaling; it does not differentiate overlap
eigenvectors. The physical overlap is not modified.

The original GFN2 all-forward discrepancy was resolved by the SPD-solve
correction. Its derivative parity already held with reverse-over-forward and
the Cholesky orthogonalizer; the symmetric orthogonalizer preserves it. The
PyTorch 2.14.1 serial and mapped values above were recovered by rerunning the
unmodified starting-SHA test with the same environment.

## Corrected four-path results

With both corrections, the primary PyTorch 2.14.1 diagnostic produced:

| Method and lane | Original serial A | Serial padded B | Mapped size-one C | Mixed mapped D | Max absolute A–D error |
|---|---:|---:|---:|---:|---:|
| GFN1 water | -1.3729496380638748 | -1.3729496380638448 | -1.3729496380638450 | -1.3729496380638450 | 3.0e-14 |
| GFN2 LiH | 4.6044538003597396e-2 | 4.6044538003595410e-2 | 4.6044538003597035e-2 | 4.6044538003595410e-2 | 2.0e-15 |

For these same paths, primal energy maximum errors were below `9e-14`, physical
gradient errors below `1.8e-13`, and HVP errors below `5.8e-14`. Both
same-direction and distinct-direction third derivatives are evaluated by a
shared scalar-energy helper with reverse-over-forward; all-forward composition
is checked as a second implementation of the same derivative. Serial and
mapped helper values agree within the unchanged `3e-5` absolute/relative
tolerance. Reversed GFN1/GFN2 source orders pass the distinct-direction test.

Each lane is compared to its own independent `System.singlepoint` reference.
The full same-direction test differentiates both lanes, and for each lane
checks that third derivatives with respect to the other System's coordinates
and that lane's inactive coordinates are zero. The mixed test also checks
source-to-canonical tangent mapping directly.

A separate aggregate-energy test uses three distinct directions with nonzero
physical entries in both Systems. It compares the mapped derivative of the
sum of lane energies against the sum of two independent serial evaluations.
That combined-tangent test passes in all five supported CPU environments.

## Additional test correction

The symmetric orthogonalizer can select a different sign for an eigenvector
column than the legacy triangular factor. A broader potential-map selection
therefore failed eight assertions that compared coefficient matrices
elementwise even though energies, density, occupations, and occupied
projectors agreed. The test now compares the occupation-weighted projector
instead of a gauge-dependent coefficient matrix. It continues to compare
Hamiltonian, density, eigenvalues, occupations, orbital energy, and physical
charges with the existing strict tolerances. The eight corrected comparisons
pass in all five CPU environments.

## AES2 safety

The existing AES2 invalid-pair sanitization remains in place. Its test checks
zero invalid pair coupling before multipole contraction, coincident padded
coordinates, position gradients, JVP, and `jacfwd`. The full E7a module passed
in the primary environment after the third-order changes.

## Validation

The prescribed environments report:

| Python / PyTorch | `tad-mctc` | D3 | D4 | Third-order checks |
|---|---:|---:|---:|---|
| 3.10.22 / 2.6.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | lane-local, distinct, reversed, and combined directions pass |
| 3.11.15 / 2.8.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | lane-local, distinct, reversed, and combined directions pass |
| 3.12.10 / 2.11.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | lane-local, distinct, reversed, and combined directions pass |
| 3.12.10 / 2.13.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | lane-local, distinct, reversed, and combined directions pass |
| 3.14.8 / 2.14.1+cpu | 0.9.1 | 0.7.0 | 0.8.0 | complete E7a module: 12 passed; all derivative cases pass |

The updated first-order selection passed `121` tests in each of the five
environments. The potential-map gauge-invariant parity selection passed
`8/8` in each environment. The primary broad focused
selection was split into completed groups: the implementation regressions
passed `183`; the SCF/mixer/status and padding-ghost group reported `208
passed, 5 failed`. All five failures are the previously reproduced starting-
SHA cases listed below. A mistakenly requested nonexistent `test_functional_
filling.py` path produced “no tests ran”; it was removed from the completed
selection and is not counted as a test result.

Inherited failures, reproduced at the starting SHA, remain:

- `test_fock_valid_trajectory_with_invalid_final_spectrum`.
- `test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[0]`.
- `test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[1]`.
- `test_potential_maxiter_zero_tracks_raw_and_final_fermi_status`.
- `test_anderson_solver_real_position_parameter_and_field_gradients` (the
  gradient discrepancy is about `2.5e-11` against an existing `2e-13`
  tolerance).

Black, isort, compileall, and `git diff --check` pass. Ruff and mypy are not
installed in the primary environment. CUDA was not run under the CPU-only
policy.

## Completion status

The third-order blocker is resolved for the tested GFN1/GFN2 boundary. The
first-order AES2 and padded-system tests remain passing, and no production
route or public Calculator dispatch changed. E7a can close at its accepted
batch-critical boundary after this report is committed. Broader E7 remains
open. E6broute remains blocked by E6c1-W2/W3 and the public packed Result
adapter.
