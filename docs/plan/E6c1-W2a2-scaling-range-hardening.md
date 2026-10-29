# S0.1b — Matrix-Fermi scaling-range hardening

## Decision

The outer matrix-Fermi scale and inner matrix-expm1 range now have one
bounded contract. For finite active inputs, `_matrix_fermi()` requires an
induced infinity norm no greater than `4 * 2**30` before scaling; the detached
power selects a scale with scaled norm at most four. The fixed reconstruction
executes up to 29 tanh doublings, which is the `p - 1` work required at the
maximum supported power `p=30`. Its initial matrix-expm1 argument has norm at
most eight, inside `_matrix_expm1()`'s supported range. Unsupported and
nonfinite active lanes use safe internal operands and return NaNs. Padded
inactive entries are zeroed before validity and scale decisions.

The accepted p14→p15 cancellation correction remains in place. No change was
made to the SCF controller, chemical-potential Newton correction, physical
formula, public API, Calculator routing, dependencies, or E6bcompat policy.

## Source and review

- Starting SHA: `6bb357f41733e24aae4367880c05a14f1b7fa84b`.
- Branch: `claude/tender-dijkstra-cda42s`.
- Starting remote ref matched the SHA above.
- Changed production file: `src/dxtb/_src/scf/functional/spectral.py`.
- Changed regression file: `test/test_scf/test_invariant_spectral_density.py`.
- Raw bounded tmpda diagnostics: `benchmarks/results/e6c1-s0.1b/`.
- The independent read-only reviewer rechecked the actual diff after two
  requested hardenings: zeroing unsupported `_matrix_expm1()` lanes before
  polynomial/doubling work, and checking reverse, forward and
  reverse-over-forward third derivatives at high scaling boundaries. Final
  review found no blocker. Reviewer identity: `/root/s0_1b_review`; exact model
  identity is not exposed by the runtime.

## Scaling calculation and tests

For `kt=9.5e-4 Eh`, the two-orbital diagonal fixture uses a partially occupied
frontier at `0.4 kt` and a distant level with dimensionless magnitude
`4 * 2**(p - 0.5)`. Its induced infinity norm therefore selects `p` for every
`p=20..30`. Alternating signs test both nearly empty and nearly full distant
states. For all these lanes, the inner expm1 argument norm is `4*sqrt(2)` and
the helper uses five internal halvings. The outer loop performs `p-1`
doublings. The frontier occupation and its first, second, reverse third,
forward third, and reverse-over-forward third derivatives are compared with
the independent scalar Fermi-Dirac formula.

| Target p | `||exponent||∞` | Required / selected p | Inner expm1 p | Outer doublings | Distant-state occupation |
|---:|---:|---:|---:|---:|---:|
| 20 | 2,965,820.80076 | 20 / 20 | 5 | 19 | 0 |
| 21 | 5,931,641.60152 | 21 / 21 | 5 | 20 | 1 |
| 22 | 11,863,283.2030 | 22 / 22 | 5 | 21 | 0 |
| 23 | 23,726,566.4061 | 23 / 23 | 5 | 22 | 1 |
| 24 | 47,453,132.8121 | 24 / 24 | 5 | 23 | 0 |
| 25 | 94,906,265.6243 | 25 / 25 | 5 | 24 | 1 |
| 26 | 189,812,531.249 | 26 / 26 | 5 | 25 | 0 |
| 30 | 3,037,000,499.98 | 30 / 30 | 5 | 29 | 0 |
| 31 limit probe | 4,294,967,300.29 | 31 / rejected (safe schedule cap 30) | 0 on sanitized zero input | safe 29-step zero recurrence | NaN |

The power-30 limit corresponds to an exponent bound `4 * 2**30`. For the
p30→unsupported boundary, points below and at the threshold require p30; a
point just above requires p31 and returns NaNs. The capped p30 schedule is
used only to keep control flow finite after the invalid lane's operands have
been replaced with safe values; it does not evaluate that unsupported input.
The p21→p22 and p29→p30 transitions exercise all three third-order transform
compositions on their valid lanes. The existing rotated symmetric p14/p15
fixture remains the independent rotated-matrix oracle and now uses
order-specific absolute tolerances: `5e-11`, `2e-9`, `2e-7`, and `2e-4` for
value through third order. Its relative tolerance remains `2e-8`.

The inner `_matrix_expm1()` regression includes a valid small matrix, a finite
dense matrix whose maximum element is bounded but whose row norm exceeds the
helper's limit, and a nonfinite lane. Rejected lanes are sanitized before the
Taylor polynomial and doubling loop. Separate tests verify active over-limit
and nonfinite Fermi lanes return NaNs, valid lanes stay finite under `vmap`,
inactive nonfinite padded entries have no effect, and invalid occupation
nonfiniteness reaches the free-energy matrix and trace paths. The free-energy
propagation test clears the fixture's detected-gap flag to exercise the raw
matrix-Fermi error path; production status values are not modified.

## tmpda one- and two-trip controls

The unchanged tmpda 69-atom geometry was run with one and two fixed SCF trips,
position gradients enabled, independently in serial and one-lane mapped
execution. Each log records stage timing and process RSS. All four runs
completed with finite energy, free energy, gradients, residuals, and metadata.
The four logs were rerun after the final subtraction-overflow precheck change;
they retain the same energies, free energies, residuals, iterations, and
gradient norms.

| Trips / route | Energy (Eh) | Free energy (Eh) | Residual norm | Converged / iterations | Gradient norm | Peak RSS |
|---|---:|---:|---:|---|---:|---:|
| 1 / serial | -26.593551333776166 | -0.000243471705459834 | 9.41189244 | false / 2 | 45.75374847 | 3.50 GiB |
| 1 / mapped | -26.593551333776166 | -0.000243471705459834 | 9.41189244 | false / 2 | 45.75374847 | 3.24 GiB |
| 2 / serial | -28.78419627915915 | -0.000363705698021727 | 8.22971887 | false / 3 | 42.01318653 | 4.50 GiB |
| 2 / mapped | -28.78419627915915 | -0.000363705698021727 | 8.22971887 | false / 3 | 42.01318653 | 4.11 GiB |

Maximum elementwise serial/mapped position-gradient differences were
`7.99e-15` at one trip and `7.61e-15` at two trips. Both channels retained
electron count 139. The observed response slopes were positive: approximately
`0.177753413` at the first map state, `23.3342361` at the next state, and
`39.9720525` for the additional two-trip state. Chemical-potential shifts
were zero. The previous `-7.4074416548e-251` slope did not recur. These
unconverged fixed-trip results are compared only between routes with equal
trip counts; they do not establish full-convergence parity or W2 performance.

## Validation

All pytest commands used `OMP_NUM_THREADS=1`, `PYTHONPATH=src`,
`-n logical --dist worksteal` unless marked diagnostic.

| Selection | Python 3.11 / Torch 2.8 | Python 3.14 / Torch 2.14.1 | Additional profiles |
|---|---:|---:|---:|
| Invariant spectral density | 65 passed | 65 passed | 3.10/2.7: 65; 3.12/2.11: 65 with two workers and serial diagnostic; 3.12/2.13: 65 with two workers |
| E7a padding, padded GFN1/GFN2, implicit-root controls | 147 passed | 147 passed | — |
| E6bcompat frozen selection | 64 passed | 64 passed | — |
| Mapped padded GFN1/GFN2 selection | 119 passed | 119 passed | — |
| E1.3/E6a D3 and transform selection | 146 passed | 146 passed | — |
| Invariant free energy + functional implicit root | 68 passed | 68 passed | — |

The PyTorch 2.11 and 2.13 environments stalled during `-n logical` worker
startup. The 2.11 suite passed serially and with two workers; the 2.13 suite
passed with two workers. The final five-profile spectral selection contains 65 tests; these were
diagnostic accommodations for the startup issue, not reported as `-n logical`
passes. All required package
groups passed on both primary profiles. Warnings were the existing Python
3.14 `torch.jit.script` warning, existing `count_nonzero` vmap fallback, and
D3 `chunk_size` deprecation notices. Black, isort, `py_compile`, and
`git diff --check` pass; Ruff and mypy are not installed in the approved
environments.

## Limits and verdict

The fixed-trip memory diagnosis is unchanged: differentiated memory grows
approximately linearly with trip count. No long-trip or 300-trip run was
attempted here. W2c1 remains a separate proof of concept; E6bcompat remains
accepted, E6broute remains blocked, and W2/W3 performance clearance remain
open. Public Calculator routing is unchanged.

- Scaling-power policy: **verified**.
- p14→p15 regression: **pass**.
- p22→p23 and p29→p30 regressions: **pass**.
- Higher-power spectral oracle: **pass**.
- Finite over-limit and nonfinite-lane propagation: **pass**.
- Higher-order transforms: **preserved** over the tested supported range.
- E6bcompat: **preserved**.
- Original tmpda one-/two-trip parity: **preserved**.
- Public Calculator routing: **unchanged**.
- W2 performance clearance: **open**.
