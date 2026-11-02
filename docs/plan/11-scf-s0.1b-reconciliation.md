# SCF S0.1b/c — matrix-Fermi reconciliation and acceptance

## 1. Published synchronization and conflict resolution

The SCF branch is rebased on the published general-refactor head
`5365b16ca48ee2d7bac0326eeaad779993f244f6`. This contains the full overlapping
spectral history: `8368371` introduced adaptive row-sum scaling, `e7d0943`
added active-mask handling and diagnostics, `6a3df45` replaced the cancellation-
prone matrix exponential start with the bounded `expm1` polynomial, `80e2a95`
added scale-boundary derivative coverage, and `9c0be0f` established the
power-30 envelope and fail-closed behavior.

The upstream implementation is the sole `_matrix_fermi` implementation. The
SCF branch's duplicate S0.1b source algorithm was discarded during rebase.
SCF-only cutoff, public-density, and schedule-transition tests were retained;
this reconciliation adds direct-batch bad-lane isolation, vmapped bad-lane
JVP isolation, a 512-lane heterogeneous batch comparison, and a functional
SCF lifecycle check. No production source file differs from the synchronized
upstream kernel.

| Concern | Published implementation retained |
| --- | --- |
| Bound | Detached maximum active absolute row sum of `(H - mu I) / kT` |
| Target scaled bound | `4` |
| Base powers | 11 normally; 22 when `kT < 5e-4 Eh` |
| Maximum power / fixed doubling count | 30 / 29 |
| Over-limit finite lane | Sanitized to a benign input before linear algebra; NaN result |
| Non-finite input/bound | Replaced before scale conversion and linear algebra; NaN result |
| Batch behavior | Lane-local scale and validity masks |
| Inner rational solve | `tad_mctc.storch.linsolve.dense_solve` retained |

## 2. Upstream ceiling reproduction

I evaluated the published implementation with the seed-61 rotated 32-orbital
spectrum, and with the same spectrum extended to 80 and 100 Eh. The bound is
the maximum active row sum of the dimensionless exponent. Oracle errors compare
against the test-only eigenspectral Fermi function; no production
eigendecomposition was added.

| Maximum energy | `kT` (Eh) | Row-sum bound | Required/selected power | Max oracle error | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 70.1 Eh | 9.5e-4 | 1.66096e5 | 16 / 16 | 4.33e-13 | finite |
| 70.1 Eh | 1.0e-6 | 1.57791e8 | 26 / 26 | 1.49e-12 | finite |
| 70.1 Eh | 3.1e-7 | 5.09004e8 | 27 / 27 | 8.19e-13 | finite |
| 80 Eh | 3.1e-7 | 5.80550e8 | 28 / 28 | 3.06e-13 | finite |
| 100 Eh | 3.1e-7 | 7.25086e8 | 28 / 28 | 4.16e-13 | finite |

This confirms that the published implementation no longer clamps these lanes
to power 22. Its finite-temperature boundary cases pass the third-order
coverage in the spectral test file.

## 3. Numerical envelope and isolation

For a required power `p`, the initial exponent norm is bounded by four after
scaling by `2^p`; the paired `p - 1` tanh doublings recover `tanh(X/2)`. Power
30 therefore supports active row-sum bounds through `4 * 2^30 =
4,294,967,296`. The policy is a finite numerical envelope, not a promise for
arbitrary finite Hamiltonians. For symmetric `X` of dimension `n` and spectral
radius `R`, `||X||_infinity <= sqrt(n) R`; the usable spectral-radius envelope
therefore depends on dimension, basis/rotation, temperature, and chemical
potential shift. At 100 Eh and `kT=3.1e-7 Eh`, the seed-61 32-orbital example
uses power 28, leaving two powers of margin. Different rotations are covered
by independent cutoff stress cases.

Before the exponent reaches `matrix_exp`/`dense_solve`, invalid and unsupported
lanes are replaced by zero. The bounded inner `matrix_expm1` helper likewise
sanitizes non-finite or out-of-domain arguments before its polynomial and
solves. Only after safe evaluation does the lane-local mask produce NaNs for
unsupported results. Direct leading-batch tests combine a supported lane with
an over-limit, NaN, or `+Inf` lane; the supported result is bitwise identical
to its standalone result. A `vmap` test also checks the supported lane's JVP
against the standalone JVP. A 512-lane batch with several selected powers
matches explicit per-lane calculations.

The public `fermi_density_matrix` cutoff test checks finite density,
electron-count trace, spectral-oracle agreement, and JVP at `kT=3.1e-7 Eh`.
Existing free-energy tests ensure invalid active occupations remain invalid in
the free-energy matrix/trace paths. Empty/full/gap closures may analytically
replace a matrix-Fermi result only where that channel does not depend on the
failed evaluation; active unsupported channels retain the failure signal.

## 4. Functional SCF failure behavior

A synthetic functional eager probe forces the matrix-Fermi input beyond the
supported envelope while retaining normal public defaults. For `maxiter=0`,
the accepted one-shot policy returns `converged=False` with visibly non-finite
density and energy, even with `force_convergence=True`. With iteration enabled,
the NaN state reaches a later eigensolve and raises a PyTorch linear-algebra
runtime error before an apparently successful result can be returned. The
implicit route also fails before `_SCFMapTensorTree.prepare` and before root
attachment. These checks characterize current behavior; they do not introduce
a new status category.

There is no dedicated public failure code for matrix-function numerical
failure. The current behavior can surface as a lower-level linear-algebra
exception on iterative eager routes. S2's final certification/status work must
distinguish numerical spectral failure from ordinary SCF nonconvergence.

## 5. Transform and regression coverage

The final tests include:

- the original 70.1 Eh rotated spectrum and its reverse, `jacrev`, `jacfwd`,
  JVP, and higher-order derivative checks;
- a noncommuting divided-difference JVP oracle;
- cutoff spectra at 70.1, 80, and 100 Eh across independent rotations;
- public Fermi density at the smeared temperature cutoff;
- detached schedule transitions 11→12, 22→23, and 26→27, including value,
  reverse/JVP/`jacfwd`, second derivative, and third derivative;
- direct-batch, `vmap`, and 512-lane heterogeneous schedule checks;
- finite over-limit, NaN, and `+Inf` failure behavior;
- the existing exact-degeneracy, small-gap, wide-gap, mask, third-order, and
  finite-temperature free-energy controls.

Schedule selection remains detached and lane-local. The maximum Python loop
trip count is static. There is no tensor-dependent Python branch, `.item()`
algorithm decision, dynamic loop, `torch.cond`, or production eigendecomposition.
No tolerance, physical reference, or Fermi cutoff was changed.

## 6. Performance and realistic envelope audit

The fixed 29-step schedule performs the full loop body even for lanes that do
not need every update. I compared the pre-reconciliation `3798c6e` worktree and
the final synchronized tree in the same environment, using the repository
default SCF tolerances, Anderson, and 300 K. Map counts, iterations, convergence,
and energies match. The measured max required powers also show that these
ordinary systems are far inside the power-30 envelope.

| System / method | Orbitals | Required power | Before seconds / maps | Reconciled seconds / maps | Energy (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2 / GFN1, 300 K | 4 | 8 | 0.487 / 7 | 0.494 / 7 | -1.05859835672401 |
| H2O / GFN2, 300 K | 6 | 8 | 0.736 / 10 | 0.725 / 10 | -5.10415879940951 |
| NO2 / GFN2, 300 K, spin 1 | 12 | 10 | 0.831 / 11 | 0.838 / 11 | -10.9530758032646 |

At 1 K, the H2/GFN1 run converged in 7 maps with required power 16 (0.495 s
reconciled). This provides a lower-temperature still-smeared example; the
synthetic public density test separately covers `kT=3.1e-7 Eh` directly.
Matched reconciliation timings differ by about -1.6% to +1.3%, within normal
run-to-run noise. The earlier S0.1b 21-to-29-step microbenchmark remains useful
as an estimate of the fixed-loop cost; the final upstream kernel was not
replaced with a shorter schedule. This remains performance debt, and no
alternative rational algorithm was introduced.

## 7. Validation and S1 gate

On the synchronized tree, validation results were:

| Selection | Result |
| --- | ---: |
| Invariant spectral density | 80 passed |
| Invariant free energy | 37 passed |
| Implicit root + invariant free energy | 68 passed |
| Phase-A Group 1 | 225 passed, 1 documented inherited failure |
| Phase-A Group 2 | 54 passed |
| Phase-A Group 3 | 131 passed |
| Unsupported functional eager lifecycle | 1 passed |

The one Group-1 failure remains
`test_anderson_solver_real_position_parameter_and_field_gradients`: maximum
absolute delta `2.5007065862503453e-11` against `2e-13`. It is the already
documented inherited parity issue. Group 2 and Group 3 are green. The Python
3.14 JIT `FutureWarning` is separate from failures.

S0.1c resolves the synchronization and numerical-isolation gate. S1 is
unblocked for a later package, but was not started here. The functional eager
probe also characterizes that `maxiter=0` can return visibly non-finite values
with `converged=False`; iterative execution and implicit preparation fail
before they can report or attach an invalid root. A dedicated numerical
failure status remains deferred to S2.
