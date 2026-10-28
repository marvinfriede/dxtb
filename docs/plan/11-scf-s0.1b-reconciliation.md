# SCF S0.1b — matrix-Fermi scaling reconciliation

## 1. Review point and source reconciliation

The SCF branch was locally rebased from
`629980b974970b0c5dcf2aa318188154fa348379` onto the overlapping general-refactor commit
`8368371c4abc404d0b7218aa01205bde1921cd01`. That commit is not yet present on
`origin/claude/tender-dijkstra-cda42s`, which remains at
`5dc157bd12e854d2a8df9bee084d768eaa270a35`. The SCF branch must not be pushed
until the overlapping base is published.

The local general-refactor branch has since advanced to
`2de612571934f3891b4b6ac31ed23aa8941532fc`.
Its unpublished commit `e7d0943` also masks inactive spectral inputs before
the matrix exponential and adds threshold/mask transform tests. Those changes
were inspected; this local reconciliation includes the same active-mask
behavior and focused padding test. The SCF branch was not rebased onto that
unpublished successor, and its equivalent hunk/tests should be dropped if that
successor is later selected as the published synchronization point.

The pre-reconciliation S0.1a changes were saved as
`/tmp1/friede/dxtb-agentic/scf-s0.1a-before-reconcile.patch` and snapshots
outside Git. During the rebase, the duplicate SCF implementation was dropped;
the upstream-owned `_matrix_fermi` is the sole source implementation. This
document records the reconciliation without rewriting the historical S0.1
report in `10-scf-s0.1-matrix-fermi-hardening.md`.

| Concern | S0.1 pre-reconciliation | Saved S0.1a patch | Upstream `8368371` | Final S0.1b |
| --- | --- | --- | --- | --- |
| Bound | Detached active exponent row-sum bound | Same | Same | Same conservative bound |
| Detach policy | Schedule only | Schedule only | Schedule only | Schedule only |
| Base powers | 11 ordinary / 22 below 5e-4 Eh | Same | Same | Same |
| Maximum power | 22 | 27 | 22 | 30 |
| Maximum doublings | 21 | 26 | 21 | 29 |
| Non-finite estimate | Not sanitized before integer conversion | Replaced with safe finite estimate | Not sanitized before conversion | Sanitized before conversion and marked unsupported |
| Above-ceiling behavior | Power silently clamped; finite wrong result possible | Kernel returned NaNs | Power silently clamped; finite wrong result possible | Unsupported lane returns NaNs through density/free-energy APIs |
| `vmap` | Lane-local schedule | Lane-local schedule | Lane-local schedule | Lane-local schedule; explicit mixed-lane test retained |
| Schedule tests | Ordinary/large oracle and derivative coverage | Added cutoff and low transition tests | Ordinary/large oracle and derivative coverage | Adds 11→12, 22→23, 26→27 derivative-continuity tests |
| Cutoff test | No row crossing the old ceiling | Added 70.1 Eh near cutoff | No row crossing the old ceiling | 70.1/80/100 Eh across independent 32-orbital rotations |
| Derivatives | Commuting and noncommuting cases | Commuting higher-order and noncommuting JVP | Commuting higher-order coverage | Preserves third order and noncommuting divided-difference JVP |

## 2. Upstream ceiling reproduction

The seed-61 rotated 32-orbital spectrum was evaluated on the upstream kernel.
The row bound is the maximum absolute active row sum of the dimensionless
exponent. The required power uses target bound four; actual scale power retains
the low-temperature base of 22.

| `kT` (Eh) | Row-sum bound | Required power | Selected power | Max forward error | Output |
| ---: | ---: | ---: | ---: | ---: | --- |
| 9.5e-4 | 166,096 | 16 | 16 | 1.25e-12 | finite |
| 2.0e-4 | 788,957 | 18 | 22 | 4.42e-12 | finite |
| 1.0e-5 | 15,779,137 | 22 | 22 | 3.37e-12 | finite |
| 1.0e-6 | 157,791,373 | 26 | 22 | 1.330e-1 | finite, incorrect |
| 3.1e-7 | 509,004,430 | 27 | 22 | 7.805e-1 | finite, incorrect |

At `kT=1e-6`, a commuting derivative expected near `-2.58e-214` instead
returned zero in reverse, `jacfwd`, and JVP in the isolated probe; at
`kT=3.1e-7`
the expected and observed commuting derivatives were both zero. At the two
higher temperatures reverse/JVP matched the analytic derivatives within
`9e-11`. These derivative probes did not mask the forward defect: the kernel
still returned finite but incorrect densities at `1e-6` and `3.1e-7`.

## 3. Final support envelope and algorithm

The implementation targets an initial scaled exponent row-sum bound no larger
than four. It preserves base powers 11 and 22 and supports required powers up
to 30. Therefore the maximum supported unscaled row-sum bound is
`4 * 2^30 = 4,294,967,296`; a finite bound above that or any non-finite bound
returns an explicit NaN lane. No integer conversion is performed on a
non-finite estimate. Unsupported lanes use a zero safe input for internal
matrix operations before the explicit NaN result is selected.

Power 30 was selected to cover the intended stress envelope with margin, not
to fit one random rotation. At the public smeared cutoff `kT=3.1e-7 Eh`, a
32-orbital symmetric spectrum with absolute spectral radius at most 100 Eh has
the conservative row-sum bound `sqrt(32) * 100 / 3.1e-7`, about `1.83e9`; the
supported bound is about 2.35 times larger. The tests exercise 70.1, 80, and 100 Eh with three
independent orthogonal rotations. This policy does not promise arbitrary
finite Hamiltonians beyond that envelope.

Measured cutoff cases selected powers 27, 28, and 28 respectively. Their
active row-sum bounds were `5.09e8`, `5.53e8`, and `7.57e8`; maximum forward
errors against the test-only spectral oracle were `4.00e-12`, `6.37e-12`, and
`2.35e-13`.

For scale `2^p`, the initial tangent is `tanh(X/2^p)` and the active loop
performs exactly `p-1` double-angle updates, preserving `tanh(X/2)`. The loop
has a static 29 iterations. Inactive lanes feed a neutral zero tangent to the
unused solve, preventing saturated unused branches from destabilizing nested
derivatives. Scale selection remains detached and lane-local. No tensor-valued
Python branch, `.item()`, dynamic loop, `torch.cond`, eigendecomposition, or
replacement of `dense_solve` was introduced.

## 4. Fail-closed propagation

The unsupported mask is explicit. `_matrix_fermi` returns NaNs for an
unsupported lane. `fermi_density_matrix` reapplies that lane result after
saturation, gap, overlap, and padding selections. The free-energy setup carries
the occupation's finite-lane mask, and both `fermi_free_energy_matrix` and
`fermi_free_energy_trace` reapply NaNs after their validity closures.

Tests cover finite over-limit, `+inf`, and `nan` direct inputs, as well as
over-limit propagation through density, free-energy matrix, and free-energy
trace. A synthetic unsupported Hamiltonian was not passed through the public
Calculator, so public `FermiStatus` is not directly tested. The outer SCF
residual comparison is strict `< tolerance`; a NaN map residual cannot satisfy
it and therefore cannot report a converged outer SCF. This is not a new public
status code for numerical-envelope violations.

## 5. Tests and transforms

The added regression coverage retains the upstream ordinary/moderate/large
oracle rows and lane-local `vmap`. It adds the 70.1 Eh cutoff row and rotated
32-orbital spectra at 70.1, 80, and 100 Eh. The public
`fermi_density_matrix` path is exercised at `3.1e-7 Eh` for finite output,
electron-count trace, spectral-reference agreement, and JVP. Unsupported
finite, infinite, and NaN bounds are checked directly and through all three
Fermi spectral APIs.

Detached schedule transitions at 11→12, 22→23, and 26→27 compare values,
reverse derivatives, JVP, `jacfwd`, second derivatives, and third derivatives
on both sides. The seed-61 70.1 Eh stress case retains reverse, `jacrev`,
`jacfwd`, JVP, and nested third reverse checks. Its noncommuting perturbation
still matches the test-only spectral divided-difference oracle. No tolerance,
reference, or physical occupation was relaxed.

Across transition sides, the first-derivative discrepancy was at most
`1.87e-4` absolute (relative below `2.4e-10` for the large response); reverse,
JVP, and `jacfwd` agreed within the same scale. Value and second-derivative
differences were zero at printed precision. The largest third-derivative discrepancy was
`9.60e8` on a response of `4.12e18` (relative `2.33e-10`); all remained within
the transition test's `2e-8` relative comparison.

## 6. Performance

The fixed 29-step loop evaluates unused solves, so it has measurable cost. A
microbenchmark compared the previous 21-step ceiling to the 29-step policy:

| Workload | 21 steps | 29 steps | Change |
| --- | ---: | ---: | ---: |
| 8x8 ordinary 300 K matrix | 5.87 ms | 7.75 ms | +32% |
| 32x32, 100 Eh, cutoff matrix | 7.00 ms | 9.78 ms | +40% |

The isolated large-spectrum pytest node took 5.91 s total (0.89 s test call),
versus 4.56 s total reported for the S0.1 run.

End-to-end functional Anderson SCFs used identical settings and map counts:

| System | 21-step time / maps | 29-step time / maps | Energy |
| --- | ---: | ---: | ---: |
| H2 / GFN1, 300 K | 0.206 s / 2 | 0.230 s / 2 | -1.0585983567240072 Eh |
| H2O / GFN2, 300 K | 0.925 s / 13 | 1.019 s / 13 | -5.104158799409648 Eh |
| NO2 / GFN2, 300 K | 1.381 s / 20 | 1.539 s / 20 | -10.953075803272071 Eh |

The observed end-to-end penalty is 10–12%, with identical convergence, map
counts, and reported energy. Existing suite timing was 25.76 s for the full
spectral-density file (versus 21.05 s in S0.1) and 103.22 s for free energy
(versus 87.18 s). This is a correctness-first bounded loop. A shorter rational
scheme would be a separate numerical algorithm and is not adopted without the
same transform and third-order acceptance matrix.

## 7. Validation and remaining issue

| Selection | Result | Runtime |
| --- | ---: | ---: |
| Invariant spectral density | 63 passed | 25.76 s |
| Invariant free energy | 37 passed | 103.22 s |
| Implicit-root + invariant free-energy control | 68 passed | 160.30 s |
| Phase-A Group 1 | 224 passed, 1 inherited failure | 109.37 s |
| Phase-A Group 2 | 54 passed | 57.96 s |
| Phase-A Group 3 | 114 passed | 136.91 s |

The only Group-1 failure is the inherited
`test_anderson_solver_real_position_parameter_and_field_gradients` parity
check: maximum absolute delta `2.5007065862503453e-11`, tolerance `2e-13`.
It is unchanged from S0.1 and was not modified. Group 2 and Group 3 are green;
the combined implicit-root/free-energy control is green. PyTorch's existing
Python 3.14 JIT `FutureWarning` is separate from test failures.

## 8. Publication and S1 gate

This local reconciliation is based on upstream commit `8368371`, which is not
yet published on the selected general-refactor remote. The remote still points
to `5dc157b`, while the local general-refactor branch has reached `2de6125`.
The reconciled SCF branch must not be pushed until the overlapping
implementation or a reviewed successor is published and fetched. S1 remains blocked on that
publication and the resulting remote-based synchronization check. No S1
architecture or production routing work has started.
