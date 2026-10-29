# E6c1-W2a.2 — Fermi third-order correction

## Status

The p14→p15 third-order discrepancy is corrected at the tested supported
range. The original tolerance is unchanged. The implementation is local to
`_matrix_fermi()` and its stable matrix `expm1` helper. E6bcompat remains
accepted. W2 and W3 performance clearance remain open; this correction is not
a benchmark clearance.

## Source and environment

- Starting local SHA: `2de612571934f3891b4b6ac31ed23aa8941532fc`.
- Implementation/test SHA: `80e2a95f969562d9232c25db9bb89b895bf719ad`.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote baseline: `5dc157bd12e854d2a8df9bee084d768eaa270a35`; this change has
  not been published yet.
- Dependencies: tad-mctc 0.11.3, tad-multicharge 0.8.0,
  tad-libcint 0.4.0, tad-dftd3 0.7.0, tad-dftd4 0.8.0.
- CPU profiles: Python 3.11.15 / PyTorch 2.8.0+cpu and Python 3.14.8 /
  PyTorch 2.14.1+cpu. Pytest used `PYTHONPATH=src`, `OMP_NUM_THREADS=1`,
  `-n logical --dist worksteal`.
- `AGENTS.md` remains untouched and untracked.

## Original failure and derivative definition

The exact node was
`test/test_scf/test_invariant_spectral_density.py::test_fermi_matrix_scaling_thresholds_preserve_nested_derivatives`.
It differentiated one diagonal matrix-Fermi occupation with respect to a
single frontier-energy parameter. The chemical potential was fixed at zero;
the upper eigenvalue was held constant. The physical direction was the unit
frontier-energy direction. The dtype was float64 and `kt=9.5e-4` Hartree.

The frontier energy was `0.4 kt = 3.8e-4` Hartree. The p14→p15 boundary was
`2**16 * kt = 62.2592` Hartree. The failing lane used the upper eigenvalue
`62.25920000622592` Hartree, just above the boundary, selecting p15. The
scalar oracle was the independent fixed-chemical-potential Fermi-Dirac
function `sigmoid(-epsilon/kt)` with its analytical derivatives, including
`-f(1-f)(1-6f+6f²)/kt³` at third order.

The oracle values were:

| Quantity | Scalar oracle |
|---|---:|
| Occupation | 0.401312339887548 |
| First derivative | -252.90604814897804 |
| Second derivative | 52544.64446338898 |
| Third derivative | 123738865.63536976 |

Before correction, the direct single-lane automatic p15 result had reverse
third derivative `123738847.93445444` (absolute error `-17.70091532`, relative
error `-1.4305057e-7`), reverse-over-forward
`123738842.03988999` (absolute error `-23.59547977`, relative
`-1.9068770e-7`), and forward third derivative `123738865.6411908` (absolute
error `0.00582104`, relative `4.7042894e-11`). The existing relative
tolerance was `2e-8`; it was not changed. The full 15-lane pytest regression
had maximum reverse third-order absolute error `11.7889051288` and relative
error `9.5272452e-8`.

The two schedules were also frozen at the same just-above-boundary input.
Frozen p14 produced reverse third derivative `123738871.52886966`
(`+4.7628527e-8` relative); frozen p15 produced `123738853.84646463`
(`-9.5272452e-8`). Both failed reverse and reverse-over-forward third-order
acceptance while forward third order passed. Thus automatic schedule selection
alone was not the defect.

## Localization and correction

The first tangent values were already close to the scalar oracle, but reverse
third-order error grew during late tanh-doubling steps. At p14, the reverse
third-order relative error was approximately `-2.99e-12` after step 8,
`-1.1909e-8` after step 12, and `-1.9053e-7` after step 13. The p15 path
showed the same late-step growth. Replacing `matrix_exp(X) - I` with the
algebraically equivalent but cancellation-free scalar `expm1(X)` restored
third-order accuracy in a one-dimensional diagnostic. The first significant
error was therefore cancellation in the small spectral components formed by
the matrix-exponential subtraction and amplified by repeated doubling, not a
bad dense-solve derivative or lane interaction.

`_matrix_expm1()` now:

1. Masks inactive orbital pairs before it is called.
2. Chooses a detached, lane-local power-of-two scale so supported inputs have
   induced infinity norm at most `1/4`.
3. Evaluates a degree-12 Taylor polynomial with Horner-style matrix blocks;
   the truncation remainder on that norm interval is below float64 roundoff.
4. Restores the scale using `expm1(2X) = expm1(X) @ (expm1(X) + 2I)`.
5. Uses the existing `dense_solve` for the initial Cayley transform and the
   existing tanh-doubling recurrence.

No eigenvector derivative, physical formula, chemical-potential algorithm,
padding mask, solver controller, or public route changed. If the requested
Taylor scale exceeds the fixed supported range, the helper returns a visible
nonfinite numerical failure instead of silently using an inaccurate
polynomial. Inactive inputs remain zeroed before scale selection and all
physical tested lanes remain finite.

The rotated p14/p15 regression uses a fixed orthogonal matrix and contracts
the density with the frontier eigenprojector. Its value and first three
derivatives therefore have the same independent scalar occupation oracle
without differentiating eigenvectors. It verifies both sides of the p14/p15
transition under `vmap`, reverse mode, forward mode, and reverse-over-forward.

After correction, at p14 the reverse, forward, and mixed third derivatives
were `123738865.63536978`, `123738865.63536975`, and `123738865.63536984`;
their largest relative error was `6.1e-16`. At p15 they were
`123738865.63536973`, `123738865.63536966`, and `123738865.63536972`; their
largest relative error was `8.5e-16`. Values, first derivatives, and second
derivatives also agreed with the scalar oracle to the printed precision.

Across all nine points below, at, and above the p12, p13, and p14 transitions
(selecting powers 12 through 15), the maximum absolute/relative errors were:

| Transform quantity | Maximum absolute error | Maximum relative error |
|---|---:|---:|
| Value | 5.55e-17 | 1.39e-16 |
| First reverse derivative | 1.43e-13 | 5.62e-16 |
| Second reverse derivative | 5.10e-11 | 9.70e-16 |
| Third reverse derivative | 1.20e-7 | 9.64e-16 |
| Third forward derivative | 1.64e-7 | 1.33e-15 |
| Reverse-over-forward third derivative | 1.50e-7 | 1.21e-15 |

## Chemical-potential and tmpda controls

The Newton-correction code was not changed. With corrected matrix-Fermi
values, the detailed tmpda serial diagnostic no longer observes a negative or
near-zero electron-response slope. The nine matrix-Fermi calls recorded
electron counts `[139, 139]` in both channels; observed response slopes were
`0.177753412997` for the first spectral state and `23.3342360952` for later
states. The final free-energy setup had target counts `[139, 139]`, zero
chemical-potential shifts, finite spectral and occupation matrices, and no
empty/full/too-many/gap classification. Thus no Newton correction was needed
in this one-trip primal diagnostic. The former `-7.4074416548e-251` response
slope is not reproduced.

One- and two-trip tmpda serial and one-lane mapped runs completed with finite
energies, free energies, residuals, and position gradients. Serial and mapped
values matched at printed precision:

| Trips | Energy (Eh) | Free energy (Eh) | Gradient norm | Converged | Result iterations |
|---:|---:|---:|---:|---|---:|
| 1 | -26.593551333776166 | -0.000243471705459834 | 45.75374846749224 | false | 2 |
| 2 | -28.78419627915915 | -0.000363705698021727 | 42.01318653382842 | false | 3 |

These fixed-trip results are deliberately unconverged and are compared only
between serial and mapped executions with the same trip budget. The one-trip
diagnostic peak RSS was 2.59 GiB mapped and 2.84 GiB serial; two-trip peaks
were 3.26 GiB mapped and 3.65 GiB serial. These do not replace the W2b
memory diagnosis or establish full-workload performance.

The stable kernel's isolated no-grad timings, five repetitions at one thread,
were 7.03 ms versus 6.90 ms for the old kernel at 32 orbitals and 157.31 ms
versus 151.06 ms at 255 orbitals. These short timings indicate a small cost
increase on these fixtures; they are diagnostic and not W2 performance
acceptance.

The five instrumented tmpda runs are preserved under
[`benchmarks/results/e6c1-w2a2/`](../../benchmarks/results/e6c1-w2a2/),
including the detailed serial no-grad response-slope probe.

## Validation

All listed tests ran with explicit `PYTHONPATH=src`. No reference, assertion,
or tolerance was weakened.

| Selection | Python 3.11 / Torch 2.8 | Python 3.14 / Torch 2.14.1 |
|---|---:|---:|
| Invariant spectral density, including p14/p15, rotated, and explicit range failure | 57 passed | 57 passed |
| E7a padding, padded GFN1/GFN2 spectral/SCF, implicit-root controls | 147 passed | 147 passed |
| E6bcompat, retained legacy composition, policy | 64 passed | 64 passed |
| Mapped padded integral/spectral and GFN2 SCF | 119 passed | 119 passed |
| E1.3/E6a D3 and transform acceptance | 146 passed | 146 passed |
| Invariant free energy and functional implicit root | 68 passed | 68 passed |

The Python 3.14 runs emitted the known `torch.jit.script` FutureWarning.
PyTorch 2.8 E7a/E6bimpl selections emitted the established
`count_nonzero` vmap performance warning. D3 0.7 emitted its existing
`chunk_size` deprecation warnings. No CUDA run was performed. Black, isort,
`py_compile`, and `git diff --check` pass; Ruff and mypy were unavailable in
the project environments.

## Independent review and next work

An independent read-only review confirmed the Taylor coefficients through
degree 12, the exponential-doubling identity, lane-local detached scaling,
the rotated scalar oracle, and retained inactive-pair masking. The reviewer
identified the need to make out-of-range behavior explicit; that is now
covered by a regression that confirms unsupported extreme spectra produce
nonfinite failure output instead of a finite but inaccurate result. No
blocking review finding remains.

W2b's measured fixed-trip memory growth remains the governing performance
diagnosis: approximately 0.64 GiB per trip from 2 to 16 trips, with 8-trip
differentiated RSS 6.90 GiB and 16-trip RSS 12.05 GiB. No 32-, 179-, or
300-trip run was repeated. The likely next bounded design study should
prototype converged-root implicit differentiation because an accepted root
attachment already exists and intentionally removes the primal trajectory
graph. This is not interchangeable with fixed-trip differentiation and
requires a separately accepted physical-root contract. Checkpointing and
per-lane convergence-aware execution remain unproven under the complete
transform contract.

- W2a.2 Fermi numerical correction: **accepted at the tested supported
  range**.
- W2b fixed-trip memory diagnosis: **complete and preserved**.
- W2 performance and W3: **open**.
- E6bcompat: **preserved**.
- Public Calculator routing: **unchanged**.
- E6broute: **blocked**.
- Remote publication: **not yet performed**.
