# SCF S0.1 — matrix-Fermi large-spectrum hardening

## Synchronization and scope

The change is based on rebased S0 commit
`04d39932fdb02bd618a9242a275a8a37c3a5893c`, descended from the accepted
general-refactor head `5dc157bd12e854d2a8df9bee084d768eaa270a35`. The upstream
commits after `e127ea4` did not change the production spectral kernel or its
focused test. No S1 contracts or routing changes are included here.

## Inherited defect and diagnosis

Before the fix,
`test_fermi_matrix_handles_large_positive_core_spectrum` failed its forward
density assertion with maximum absolute error `5.2714674690836025` (required
`atol=5e-11`, `rtol=2e-10`). Its largest orbital energy is `70.1 Eh`, and
`kT=9.5e-4 Eh`, giving a dimensionless exponent magnitude about `73,789`.
The temperature-only normal scale was `2^11`; the initial exponent therefore
reached about `36` in spectral magnitude. The resulting matrix exponential
spanned a range that made the initial rational solve ill-conditioned for the
rotated matrix.

The test stopped at the forward assertion, so its later derivative assertions
were not reached. An isolated before-state diagnostic nevertheless evaluated
them: the analytic commuting-direction derivative was `-0.0358372695`, while
reverse/`grad` returned `0.5318386319` and `jacfwd`/JVP returned
`2.0231013245`. `jacrev` matched reverse/`grad` at the incorrect value.

## Scaling change

Previously `_matrix_fermi` used scale `2^11` with ten effective doublings at
ordinary temperature and `2^22` with 21 doublings below `5e-4 Eh`. It now
computes a detached maximum absolute active-row-sum bound on the dimensionless
exponent and chooses the smallest power of two that keeps that bound at most
four, subject to the previous temperature-dependent minimum scale. The scale
power is capped at 22, matching the existing fixed maximum of 21 doubling
updates. For the failing matrix the row-sum bound is about `166,096`, so the
selected scale is `2^16` with 15 effective doublings.

This preserves the matrix-function identity. For a selected scale `2^p`, the
initial solve evaluates `tanh(X/2^p)`; `p-1` exact doubling steps produce
`tanh(X/2)`, and the occupation remains
`(I - tanh(X/2))/2`. The accepted `dense_solve` primitive remains unchanged.
The magnitude bound is detached because it selects only an algebraically
equivalent scaling schedule. The scale and per-lane update masks are tensor
operations; the Python loop remains statically bounded, with no
tensor-dependent Python branch.

## Tests and runtime

The expanded test coverage compares the invariant matrix result to a test-only
spectral oracle over ordinary, moderate, large (`70.1 Eh`), and low-temperature
spectra. It checks finite symmetric output and electron trace. The existing
large-spectrum case retains reverse, `jacrev`, `jacfwd`, and JVP comparisons;
a noncommuting JVP is checked against the analytic divided-difference matrix.
A `vmap` test combines ordinary and large spectra and verifies lane-local
scaling against explicit calls.

On the isolated environment (Python 3.14.8, PyTorch 2.14.1+cpu), results were:

| Selection | Result | Runtime |
| --- | ---: | ---: |
| Large-spectrum case before | 1 failed | 4.72 s pytest time |
| Large-spectrum case after | 1 passed | 4.56 s pytest time |
| Full invariant spectral-density file | 53 passed | 21.05 s |
| Full invariant free-energy file | 37 passed | 87.18 s |
| Implicit-root + invariant free-energy control | 68 passed | 130.43 s |
| Phase-A Group 1 | 224 passed, 1 inherited failure | 97.53 s |
| Phase-A Group 2 | 54 passed | 47.05 s |
| Phase-A Group 3 | 104 passed | 96.08 s |

The full spectral and free-energy selections include exact-degenerate third
derivatives, small-gap forward/reverse/JVP checks, wide-gap response, and the
NO2 finite-temperature free-energy case. The new large-spectrum tests passed
in 5.47 s together. The before/after isolated test runtimes show no material
regression. PyTorch's existing Python 3.14 JIT `FutureWarning` appeared; it is
not a test failure.

## Compatibility checks and remaining baseline issue

The accepted D3/D4 development import order was verified: importing `dxtb`
first, then D3 0.7.0 and D4 0.8.0 succeeds. After `dxtb` import,
`tad_mctc.autograd.is_batched` and `tad_mctc.typing.Any` are present through
the documented compatibility shim. The libcint 0.4 `Basis`, `Shell`, `int1e`,
and `overlap` imports work, and the old libcint symbols do not occur in dxtb
production source.

The only remaining focused failure is the inherited
`test_anderson_solver_real_position_parameter_and_field_gradients` strict
parity check: maximum absolute delta `2.5007065862503453e-11` against `2e-13`.
It reproduces on the synchronized general-refactor source and is outside this
package. No tolerance, reference, dependency version, or physical occupation
was changed.

## S1 gate

The S1 gate is **UNBLOCKED**: the branch descends from the accepted refactor
head; libcint 0.4 and the dxtb-first D3/D4 shim checks pass; Group 2 and Group 3
are green; Group 1 has only its documented inherited parity mismatch; and
there are zero unexplained focused failures. S1 implementation has not started
and remains a separate package.
