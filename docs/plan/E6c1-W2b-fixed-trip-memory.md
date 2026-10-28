# E6c1-W2b — Fixed-trip memory diagnosis

## Decision

**W2 memory owner identified; W2 remains OPEN.** Differentiated memory grows
approximately linearly with the number of fixed SCF trips, while the same
forward calculation under `no_grad` stays near its setup footprint. This is
consistent with saved autograd state for the unrolled functional trajectory,
not an unintended Python cache or a `vmap`-specific leak. No memory optimization
was implemented because checkpointing or changing the derivative contract
needs separate transform-compatibility acceptance.

The W2a adaptive Fermi scaling correction remains **not fully verified**. A
new independent scalar-oracle boundary probe finds third-order reverse-mode
and mixed-transform error at the upper end of the tested room-temperature
spectral range. The independent review found no issue in the scaling algebra
or masked-input correction, but this probe is a derivative acceptance blocker.
The correction is not published and no W2 clearance is claimed.

## Repository and environment

- Starting SHA: `8368371c4abc404d0b7218aa01205bde1921cd01` (local W2a commit).
- Ending code/test SHA: `e7d0943e94b83500984c61c8fcb23208d6836e3c`.
- Report-only documentation commit follows; final local repository SHA is in
  the delivery summary. The code/test commit is an unpublished diagnostic
  continuation, not an accepted W2a result.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote `origin/claude/tender-dijkstra-cda42s` was verified at
  `5dc157bd12e854d2a8df9bee084d768eaa270a35`; no changes were pushed.
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0, D3 0.7.0,
  D4 0.8.0.
- Measurements: Python 3.11.15, PyTorch 2.8.0+cpu, AMD EPYC 7763,
  one intra-op and inter-op thread, `OMP_NUM_THREADS=1`.
- Differentiated processes were run in fresh subprocesses with a 16 GiB
  `RLIMIT_AS` ceiling and a 180-second timeout. The completed 16-trip run
  peaked at 12.05 GiB. The original historical measurements remain unchanged.

Raw JSONL diagnostic records are in
[`benchmarks/results/e6c1-w2b/`](../../benchmarks/results/e6c1-w2b/).
They use the actual `System.singlepoint()` route, unchanged tmpda geometry
generation, D3 and all production classical terms. `--no-grad` is diagnostic
only and is not a substitute for the differentiable path.

## Fixed-trip measurements

| Route / trips | Grad mode | Forward (s) | Backward (s) | Total (s) | Peak RSS (GiB) | Energy (Eh) | Free energy (Eh) | Gradient norm |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Serial / 1 | enabled | 5.09* | 8.94* | 14.03* | 2.41* | -26.5935513338* | -0.0002434717061* | 45.75375* |
| Serial / 2 | enabled | 6.52 | 11.65 | 18.17 | 3.05 | -28.7841962792 | -0.0003637056984 | 42.01319 |
| Serial / 4 | enabled | 9.77 | 17.81 | 27.58 | 4.34 | -29.1806661396 | -0.0002404001863 | 41.54216 |
| Serial / 8 | enabled | 16.08 | 30.05 | 46.13 | 6.90 | -26.6957239809 | -0.00008900465658 | 101.01078 |
| Serial / 16 | enabled | 29.27 | 56.09 | 85.36 | 12.05 | -29.7297899174 | -0.0014639383993 | 43.05936 |
| Mapped / 4 | enabled | 9.97 | 17.95 | 27.92 | 4.10 | -29.1806661396 | -0.0002404001863 | 41.54216 |
| Serial / 1 | disabled | 3.86 | — | 3.86 | 0.36 | -26.5935513338 | -0.0002434717061 | — |
| Serial / 8 | disabled | 12.42 | — | 12.42 | 0.37 | -26.6957239809 | -0.00008900465658 | — |

`*` The one-trip row is the previously recorded W2a measurement on the same
profile; its raw diagnostic predates this W2b artifact folder. “Total” is the
forward plus backward duration (or forward alone for no-grad); process elapsed
also includes imports and setup and was 21.29, 30.57, 50.39, 89.14, and
32.31 seconds for serial 2/4/8/16 and mapped 4 trips, respectively. All current
JSONL measurements are individual runs, not medians. The 16-trip run completed
with finite position gradients. Its peak was already 77% of the 16 GiB process
address-space limit. A 32-trip differentiable run was therefore not attempted.

Peak RSS increases from 3.05 GiB at two trips to 12.05 GiB at sixteen trips,
about 0.64 GiB per additional trip over that interval. Eight trips with
gradients peak at 6.90 GiB; the corresponding no-grad forward peaks at 0.37
GiB. Energy and free energy are identical between those two eight-trip runs.
This distinguishes the retained differentiation graph from the physical
forward intermediates. The mapped four-trip peak (4.10 GiB) is comparable to
serial four-trip (4.34 GiB), so this growth is not primarily attributable to
`vmap`.

The sixteen-trip run performs 17 complete spectral evaluations: the initial
spectral state plus one after each of 16 fixed trips. It calls
`_matrix_fermi()` 54 times, `dense_solve()` 1,188 times, and `torch.matrix_exp`
90 times, including finalization. The SCF result remains unconverged at this
budget. The reported 179 completed spectral evaluations in the earlier
300-second run therefore corresponded to 178 completed map trips plus the
initial state; the next spectral evaluation was still in
`fermi_density_matrix()` when the timeout occurred.

## Numerical correction boundary probe

The independently reviewed W2a schedule selects a detached, lane-local
power-of-two scale. The adaptive choice remains outside the derivative graph;
the selected matrix-function calculation is differentiated normally. A new
regression sweeps below, exactly at, and above the p12–p14 transitions and
compares the matrix Fermi value and derivatives against the analytic
scalar-eigenvalue Fermi-Dirac function. It exercises `vmap`, reverse mode,
forward mode, reverse-over-forward, and third derivatives.

The highest exercised transition (p14 to p15) exposes numerical sensitivity.
For `kt=9.5e-4`, frontier energy `0.4 kt`, and a second active diagonal
eigenvalue just above `2**16 * kt`, the required power is 15. The observed
value error in the direct single-lane probe is `1.34e-11`; its reverse-mode
third derivative has absolute error `-17.7009` (`-1.43e-7` relative) and
reverse-over-forward has absolute error `-23.5955` (`-1.91e-7` relative).
Forward third order differs by `0.00582` (`4.70e-11` relative). In the full
15-lane boundary regression, the largest reverse third-order error is `11.7889`
absolute (`9.53e-8` relative). Both exceed the established
`2e-8` relative third-order tolerance. This shows transform-composition and
batch-size sensitivity and prevents accepting the W2a numerical correction
at this boundary. No tolerance or reference was changed.

The likely sensitive sequence is the subtraction
`matrix_exp(2 * scaled) - I` followed by repeated tanh doubling: the small
frontier component is formed by cancellation and then magnified by repeated
doubling. This is a diagnosis from the scalar reproducer, not yet a proven
root-cause fix. A trial schedule remap did not satisfy the established third
order tolerance and was discarded. The production adaptive schedule is
unchanged. A separate change now masks inactive exponent entries before the
matrix exponential; its direct ±1e8 noncontiguous-padding regression passes
through the third directional derivative.

## Memory owner and next strategy

No evidence indicates a persistent cache or Python reference leak. `no_grad`
completes eight trips at approximately 0.37 GiB RSS, whereas the differentiated
calculation grows with each trip and retains that footprint until backward
has consumed the graph. The dominant owner is therefore the unrolled SCF
autograd graph, including saved dense-solve and matrix-exponential state.
Final trace free-energy evaluation adds about 1.5 seconds per completed
forward and is not the source of the per-trip memory slope.

The next optimization proposal is to investigate recomputation/checkpointing
of per-trip spectral state while preserving nested `torch.func` transforms.
It requires a dedicated proof for JVP, `jacfwd`, `jacrev`, HVP, third order,
parameter gradients and mapped lanes before implementation. Static
transform-compatible early stopping is a separate solver-policy decision;
the current controller correctly evaluates every configured fixed trip.
Implicit differentiation at a converged root is not interchangeable with the
currently accepted fixed-trip derivative when the primal state has not
converged.

## Validation and status

- Independent source review: the reviewer confirmed the new supported-range
  p14→p15 third-order discrepancy is a W2a acceptance blocker and advised
  localizing the reverse/forward/mixed transform mismatch without relaxing
  tolerances. The reviewer found no issue with pre-exponential inactive
  masking.
- Python 3.11 / PyTorch 2.8: existing invariant spectral selection excluding
  the new boundary reproducer: 54 passed. The exact boundary regression fails
  at reverse third order as recorded above; the combined boundary and mask
  selection reports 1 passed, 1 failed in 23.70 s. The direct inactive-input
  mask test, including third-order inactive-coordinate derivatives, passes
  separately (1 passed in 42.23 s).
- The attempted combined spectral/E7a/padded-spectral run did not produce a
  final pytest summary and was interrupted; it is not counted as passing.
- Python 3.14 / PyTorch 2.14.1: boundary regression and inactive-input test:
  1 passed, 1 failed in 40.35 s; the same reverse third-order boundary
  assertion fails with maximum relative error `9.53e-8`. The inactive-input
  test passes separately (1 passed in 46.50 s).
- E6bcompat remains accepted at its prior tested boundary. No public routing,
  fixed-trip controller, or physical term changed.
- E6c1-W2 and W3 remain OPEN. No 300-trip result, completed eight-conformer
  result, or performance clearance exists.
