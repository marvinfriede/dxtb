# E6c1-W2c6 — Tanh-Doubling Solve Reformulation

## Decision

**Production integration accepted for the narrow tanh-doubling expression.**
The candidate computes the same matrix function and preserves the S0.1b
scaling, masks, numerical-range checks, solver and correction schedule. It
passed value, boundary, transform, padding, mapped-lane, physical-root and
energy/force checks. On a captured physical tmpda density, it reduced
matrix-Fermi correction-core time by about 17% forward and 20% reverse. A
bounded one-conformer energy/force run completed about 11% faster than the
accepted W2c2a reference, with matching energies and forces and a lower RSS
high-water mark.

This is a kernel optimization result only. W2 and W3 remain open, E6broute
remains blocked, and public Calculator routing is unchanged. The complete
eight-conformer W2 acceptance workload was not run.

## Source and environment

| Field | Value |
|---|---|
| Required starting SHA | `45f2e9ec98368c51521146559e2c3c615ae09d2b` |
| Final production implementation SHA | `f14ca105e15251b4dcc370f2232fdb7f64839b06` |
| Starting branch | `claude/tender-dijkstra-cda42s` |
| Starting remote SHA | `45f2e9ec98368c51521146559e2c3c615ae09d2b` at package start |
| CPU | AMD EPYC 7763 64-Core Processor |
| Main benchmark environment | Python 3.11.15, PyTorch 2.8.0+cpu |
| Additional tested environments | Python 3.12/PyTorch 2.11.0 and 2.13.0; Python 3.14/PyTorch 2.14.1 |
| Dependencies | tad-mctc 0.11.3; tad-multicharge 0.8.0; tad-libcint 0.4.0; tad-dftd3 0.7.0; tad-dftd4 0.8.0 |
| Thread settings | PyTorch intra/inter-op 1; OMP, MKL, OpenBLAS 1 |
| Production files changed | `src/dxtb/_src/scf/functional/spectral.py` only |
| Public Calculator routing | Unchanged |

The Python 3.10/PyTorch 2.6.0 environment has tad-mctc 0.9.1 rather than the
accepted 0.11.3 stack. Test collection fails before dxtb tests load because
that installed package does not export `tad_mctc.autograd.attach_gradient`,
which the source imports. No dependency was changed to work around this
environment mismatch.

## Mathematical identity

The tanh doubling step is

\[
T_{n+1}=2T_n(I+T_n^2)^{-1}.
\]

In exact arithmetic, `I + T²` is a polynomial in `T`, so it commutes with `T`.
Where the denominator is invertible, its inverse also commutes with `T`:

\[
2T(I+T^2)^{-1}=2(I+T^2)^{-1}T.
\]

For a real symmetric tangent, the eigenvalues of `I + T²` are `1 + t²`, hence
are positive and at least one. In the supported finite-precision path the
matrix may have small symmetry error from solves; the existing dense solver,
symmetrization of the final density, and all validity handling remain in
place. The change is algebraically exact but not bitwise identical: solve
rounding and multiplication order differ.

The old loop formed `solve(I + T @ T, I)` and then multiplied `T` by the
result. The new loop solves directly with `T` as the right-hand side. It
retains one `T @ T` multiplication and one dense solve per loop iteration
while removing the explicit multiplication of `T` by the inverse. There is
no new scaling or failure policy.

## Implementation and numerical coverage

The production change is limited to `_matrix_fermi()`'s doubling loop:

```python
denominator = eye + tangent @ tangent
doubled = 2.0 * dense_solve(denominator, tangent)
```

The existing initial solve, fixed 29-step loop, `scale_power` update mask,
S0.1b maximum power 30, invalid-input sanitization, active-orbital masking,
and accepted `dense_solve` backend are unchanged. A scoped private diagnostic
also runs the direct-RHS function under the existing tmpda physical
benchmark and restores `_matrix_fermi` in `finally`.

The focused tests compare against a test-only copy of the accepted
pre-optimization identity-RHS recurrence. This reference shares the bounded
scaling and input-safety preparation, but retains the original
`2 * T @ dense_solve(I + T @ T, I)` update. Thus the matrix-function parity
checks do not call the modified production implementation to generate their
expected values. Separate assertions verify that the production path follows
the candidate and that p30 is finite while p31 returns NaNs.

The focused tests exercise:

- One-, two-, eight-, 32-, 128- and 255-orbital values; diagonal and rotated
  spectra; float32 and float64; exact and near degeneracy; a generalized
  overlap preparation; and an independent scalar Fermi occupation oracle.
- Exact scale selection at p14, p15, p21, p22, p23, p29 and p30. A
  partially occupied frontier is retained while the remote diagonal level
  sets the selected power, so the first-through-third frontier derivatives
  are nontrivial at each scale.
- The p30-to-unsupported boundary, the H=1000/μ=990 case, large finite
  cancellation, zero temperature, nonfinite active inputs, nonfinite
  inactive padded entries, mixed valid/invalid lanes and noncontiguous masks.
- `grad`, `jvp`, `jacfwd`, `jacrev`, HVP, forward-over-forward and
  reverse-over-forward second derivatives, directional and reverse third
  derivatives, nested `vmap`, mixed scaling powers, lane permutations and
  zero cross-lane Jacobian blocks.

An early rotated extreme-spectrum fixture lost frontier precision when
constructing a matrix with a very remote eigenvalue. The power-boundary tests
therefore use a diagonal matrix with a separated frontier. Rotated matrices
are used for ordinary spectral ranges, degeneracy, transform and lane tests.
No tolerance was relaxed.

All JSONL records whose method is `production` are historical pre-change
measurements: they were recorded at the package starting source SHA
`45f2e9ec98368c51521146559e2c3c615ae09d2b`, before the production recurrence
was modified. The benchmark harness was present in a dirty worktree for those
runs, so the source revision identifies the accepted source base rather than
a committed benchmark snapshot. The `direct-rhs` records use the experimental
implementation. New runs can select `identity-rhs` explicitly to reproduce
the old recurrence after the production switch.

The profiler uses fresh processes per method. `aten::matmul` can be an
aggregate event around `aten::mm`/`aten::bmm`; the operator-level comparison
uses the underlying `mm` or `bmm` count and reports the wrapper separately.
For one matrix-Fermi call, the 29 doubling steps reduce underlying matrix
products by 29 and leave the 30 dense-solve calls unchanged. In a two-spin
three-evaluation density correction, this is 87 fewer batched matrix
products; the observed dense-solve count remains 90.

## Performance

Measurements were on the same AMD EPYC 7763 host with float64, one PyTorch
intra/inter-op thread, and OMP/MKL/OpenBLAS set to one. Matrix-function
timings are median wall time after warm-up; small cases use five samples and
255-orbital cases generally use two or three. The synthetic fixture is a
rotated symmetric matrix. The physical fixture is the archived tmpda
conformer 0 spectral input from W2c5a, 255 orbitals and two spin channels.

| Fixture/mode | Original | Direct RHS | Difference |
|---|---:|---:|---:|
| N=32, p16, forward | 10.08 ms | 9.93 ms | −1.5% |
| N=128, p16, forward | 42.34 ms | 42.56 ms | +0.5% |
| N=255, p11, forward | 211.51 ms | 179.99 ms | −14.9% |
| N=255, p16, forward | 223.98 ms | 176.54 ms | −21.2% |
| N=255, p22, forward | 214.19 ms | 175.68 ms | −18.0% |
| N=255, p29, forward | 209.02 ms | 176.27 ms | −15.7% |
| N=255, p30, forward | 210.64 ms | 177.48 ms | −15.7% |
| N=255, p16, reverse | 590.73 ms | 521.24 ms | −11.8% |
| N=255, p16, JVP | 565.31 ms | 466.06 ms | −17.6% |
| N=255, p16, second directional | 1.545 s | 1.337 s | −13.5% |
| N=255, p16, third directional | 4.441 s | 3.506 s | −21.1% |
| Captured tmpda, forward | 410.30 ms | 340.50 ms | −17.0% |
| Captured tmpda, reverse | 1.328 s | 1.021 s | −23.2% |
| Captured tmpda density correction, forward | 1.242 s | 1.031 s | −17.0% |
| Captured tmpda density correction, reverse | 3.372 s | 2.688 s | −20.3% |

The captured tmpda density-correction run executes the existing initial
matrix-Fermi evaluation and two chemical-potential correction evaluations.
It does not use matrix-shift reuse. The full correction trace changes
underlying batched matrix-multiply counts from 218 to 131 and retains 90
dense solves. Reverse-mode process peak RSS in this isolated correction
measurement was 1,926 MiB for the original formulation and 1,493 MiB for the
candidate. Forward process peaks were 347 MiB and 352 MiB respectively; those
small differences are allocator/high-water variation, not a memory claim.

For the complete one-conformer tmpda implicit energy/force diagnostic, the
candidate used the frozen W2 settings and the exact `result.energy.sum()`
observable. It converged in 66 map evaluations with independent residual
norm `6.0120e-9`, electron-count error zero, and finite density/energy/forces.
The W2c2a baseline also converged in 66 evaluations with residual norm
`6.0119e-9`.

| Measurement | W2c2a accepted baseline | W2c6 candidate |
|---|---:|---:|
| Total elapsed time | 164.44 s | 145.78 s |
| Peak RSS | 2,534.72 MiB | 2,305.92 MiB |
| Energy | −29.758437462176378 Eh | −29.758437462176317 Eh |
| Free-energy sum | −0.001391311654298255 Eh | −0.001391311654408855 Eh |
| Maximum force difference | — | `1.82e-13 Eh/bohr` |
| Response linearize calls | 13 | 13 |
| Time in linearize calls | 37.9 s recorded in W2c2a | 33.66 s |

The candidate total was 11.35% lower and the measured RSS high-water mark
was 9.0% lower. This is one candidate run against the previously accepted
same-host baseline, not a repeated A/B series. It is promising evidence for
the kernel and response path, not W2 performance clearance. The recorded
physical trace is `benchmarks/results/e6c1-w2c6/tmpda0-candidate-energy-position.jsonl`.

## Test evidence

All validation used `-n logical --maxprocesses=1 --dist worksteal`, with one
CPU available to the container. This keeps the required logical-worker mode
while capping it to a single worker to avoid duplicating memory-heavy
processes. The following finalized selections passed:

| Selection | Python/PyTorch | Result |
|---|---|---:|
| W2c6 candidate tests | 3.11/2.8 | 27 passed |
| W2c6 candidate tests | 3.12/2.11 | 27 passed |
| W2c6 candidate tests | 3.12/2.13 | 27 passed |
| W2c6 candidate tests | 3.14/2.14.1 | 27 passed |
| Invariant density and free-energy suites | 3.11/2.8 | 102 passed |
| Invariant density and free-energy suites | 3.14/2.14.1 | 102 passed |
| Functional implicit-root and W2c1a mapped-root tests | 3.11/2.8 | 34 passed |
| Functional implicit-root and W2c1a mapped-root tests | 3.14/2.14.1 | 34 passed |
| E7a padding and higher-order module | 3.11/2.8 | 13 passed |
| E7a padding and higher-order module | 3.14/2.14.1 | 13 passed |
| E6bcompat frozen 64-node selection | 3.11/2.8 | 64 passed |
| E6bcompat frozen 64-node selection | 3.14/2.14.1 | 64 passed |
| Mapped GFN1/GFN2 119-node selection | 3.11/2.8 | 119 passed |
| Mapped GFN1/GFN2 119-node selection | 3.14/2.14.1 | 119 passed |
| E6a conformer/component/transform controls | 3.11/2.8 | 38 passed |
| E6a conformer/component/transform controls | 3.14/2.14.1 | 38 passed |

An attempted combined spectral/implicit command was interrupted before its
summary and is not counted. The component suites were rerun in fresh,
bounded processes and the rows above report those completed runs. The Python
3.10/PyTorch 2.6 profile did not reach collection because its installed
tad-mctc 0.9.1 environment is incompatible with this source, as described
above. No production dependency or tolerance was changed.

Black, isort, Python byte compilation and `git diff --check` passed for the
changed Python files. Ruff was unavailable in the selected development
environment. No relevant `torch.compile` acceptance test exists in the
spectral/SCF focused modules; eager AD and `vmap` are tested directly.

## Scope, review and remaining gates

No eigendecomposition, new solver, branch, clipping, custom autograd rule,
cache, or change to public routing was added. The existing free-energy
implementation was not rewritten; its invariant suite passed with the
production kernel change, and the physical tmpda run produced a free-energy
sum within `1.11e-13 Eh` of the accepted reference. A separate
energy-plus-free-energy force benchmark was not run.

Independent review requested with Luna/high reasoning effort; the runtime does
not expose the actual reviewer model identity. The first read-only review
found the algebra and production scope sound, with one MAJOR test issue:
expected values used the post-change production function. That issue was
corrected by adding the independent pre-change recurrence described above,
plus explicit p30 and p31 assertions. Follow-up review returned **NO
FINDING** on the corrected regression reference and benchmark provenance. The
reviewed code consists of diagnostic commit
`41540c8afa7d896d4015b46a81590c740fe49f55` and production plus test commit
`f14ca105e15251b4dcc370f2232fdb7f64839b06`. The reviewer later identified a
bookkeeping hash mismatch caused by updating this report after the reviewed
aggregate diff; the intervening change was documentation-only. No code, test,
or benchmark blocker remained.

The production edit is only the direct-RHS tanh-doubling expression. The
reviewer found no blocker in the numerical, transform, padding, lane, or
scope changes. W2 and W3 remain open, E6broute remains blocked, and public
Calculator routing is unchanged.

The W2c6 code change affects only the algebraic right-hand side of each
tanh-doubling solve. The full eight-conformer W2 benchmark, W3, and E6broute
remain outside this package's clearance and remain open/blocked respectively.
