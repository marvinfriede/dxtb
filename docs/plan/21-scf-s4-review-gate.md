# SCF S4 review gate — native Broyden evidence

## Candidate and ancestry

The validated S4 candidate remains `708fabe94a2bded8dd8ff0d6d6ff7cd403c138ac`.
Its first S4 commit is `0cdbcb7743b4fa1007b55225eadc9fc31919a4e4`, whose
parent is `e33fcf100b52d291cca815aab870b4cd4b632525`. The range-diff from
`124a464e952e230ac8489ced48c5ef9e9295dcb6..56b6fdaee04614290e52670cd89f69cff9ca869b`
to
`7d71e930e5bdbd52a252bf23d21803b94d2695f4..e33fcf100b52d291cca815aab870b4cd4b632525`
matched all 27 prior SCF commits one-for-one.

The remote branch remains at `56b6fdaee04614290e52670cd89f69cff9ca869b`.
The exact S4 commit is not yet present on GitHub. Git fetch/push from the
available shell still exits during local UID/NSS resolution (`yp_bind_client_create_v3`),
before remote contact. No ref update was attempted. The candidate has not been
rebased or amended.

## Equation and safeguard review

The method uses the controller-projected `context.proposal_x` and forms
`F=x-G(x)`, so its initial proposal is `-H0 F = 0.5*(G(x)-x)` for
`H0=0.5 I`. Controller damping then gives
`x_next=x+context.damp*proposal`. The first-step test checks the exact sign and
factor, including the default quarter-step.

For later observations it computes
`s=x_k-x_(k-1)`, `y=F_k-F_(k-1)`, `v=H.T s`, and `c=s-H y`, then appends
`c d.T` with `d=v/(y.T v)`. This is Broyden's first inverse update. The
observation update occurs before the current step is proposed. A direct
comparison with vendored `BroydenFirst` over a deterministic nonlinear
10-variable sequence measured maximum absolute differences of:

| Quantity | Maximum difference |
| --- | ---: |
| Proposed iterate | `1.67e-16` |
| Inverse action on probe vector | `2.67e-15` |
| `c` correction | `2.09e-16` |
| `d` correction | `2.14e-14` |

The safeguard uses
`quality=abs(y.T v)/(||y|| ||v||)` and accepts a secant only when the quality
is at least `sqrt(machine epsilon)` by default. An optional explicit setting
can override that threshold. It rejects zero/non-finite norms, denominator,
quality, or secant/intermediate vectors. The denominator is replaced by one
with `where` before division when the update is invalid. Rejected updates clear
the correction arrays, reset the count, store the current observation, and
increment `restart_count`; the proposal uses `H0`. The focused degenerate and
non-finite secant tests confirm finite, empty correction storage.

History has fixed shapes `(rank,n)`, with default rank 8. When full, the method
clears the corrections first, then forms the current secant against `H0` and
inserts it if valid. It does not use a rolling window. An inactive transition
selects every previous state field unchanged. There is no dense inverse in
flattened SCF-variable space, no line search, and no method-local damping;
only `context.damp` scales the step.

## Rank study

Runs used Python 3.14/PyTorch 2.14.1, GFN2, potential representation, 300 K,
float64, `maxiter=120`, and `x_atol=x_atol_max=1e-8`. Timing covers the solve
after case/integral preparation and is a single-run characterization. Each row
had one certification evaluation; map count equals the accepted iteration
count. Memory is the full Broyden state estimate, including previous `x/F`,
boolean/integer diagnostics, and both correction arrays.

### H2O/GFN2

| Rank | Iterations/maps | Restarts | Fixed L2/max | Selected stationarity | Energy+free energy (Eh) | Time (s) | State bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 25 | 5 | `8.96e-9 / 4.44e-9` | `3.45e-9` | -5.104158799410 | 1.8502 | 4,341 |
| 8 | 25 | 2 | `9.64e-9 / 5.69e-9` | `3.74e-9` | -5.104158799410 | 1.8507 | 7,801 |
| 16 | 25 | 1 | `9.58e-9 / 4.64e-9` | `3.57e-9` | -5.104158799410 | 1.8527 | 14,721 |
| 32 | 25 | 0 | `9.55e-9 / 5.16e-9` | `3.40e-9` | -5.104158799410 | 1.8551 | 28,561 |

### NO2/GFN2, neutral doublet

| Rank | Iterations/maps | Restarts | Fixed L2/max | Selected stationarity | Energy+free energy (Eh) | Time (s) | State bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 27 | 6 | `7.57e-9 / 5.58e-9` | `3.39e-9` | -10.953075803272 | 2.0899 | 4,341 |
| 8 | 27 | 3 | `6.43e-9 / 3.92e-9` | `3.18e-9` | -10.953075803272 | 2.1020 | 7,801 |
| 16 | 27 | 1 | `5.98e-9 / 2.89e-9` | `3.12e-9` | -10.953075803272 | 2.0831 | 14,721 |
| 32 | 27 | 0 | `6.11e-9 / 3.14e-9` | `3.19e-9` | -10.953075803272 | 2.0977 | 28,561 |

### C4H5NCS/GFN2

| Rank | Iterations/maps | Restarts | Fixed L2/max | Selected stationarity | Energy+free energy (Eh) | Time (s) | State bytes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 33 | 7 | `8.09e-9 / 2.53e-9` | `1.63e-9` | -19.694059708019 | 2.9274 | 17,301 |
| 8 | 30 | 3 | `6.32e-9 / 1.54e-9` | `1.51e-9` | -19.694059708019 | 2.6654 | 31,129 |
| 16 | 29 | 1 | `7.38e-9 / 2.52e-9` | `1.86e-9` | -19.694059708019 | 2.5683 | 58,785 |
| 32 | 29 | 0 | `8.16e-9 / 2.20e-9` | `1.94e-9` | -19.694059708019 | 2.5675 | 114,097 |

All ranks certified. Here “matched rank 32” means the same iteration count on
H2O and NO2 and the same certified energy to the shown precision; it does not
mean the same residual, stationarity, or restart count. Rank 8 used one extra
iteration on C4H5NCS. The rank-32 reference did not reach its capacity in these
cases.

## Fock memory and stability

Runs used GFN2, Fock representation, float64, 300 K, and explicit
`x_atol=x_atol_max=1e-8`. Times cover the solve only after preparation.
Anderson is included as a controller/method comparison, not as a rank-memory
reference.

| System/method | AO count | Flattened `n` | Rank | C/D bytes | Previous x/F bytes | Total state bytes | Iterations/maps | Core certifications | Restarts | Certified | Fixed L2/max | Stationarity | Count error | Energy+free energy (Eh) | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| H2O Broyden | 6 | 36 | 8 | 4,608 | 576 | 5,209 | 26 | 1 | 3 | yes | `5.12e-9 / 2.02e-9` | `1.73e-9` | `1.78e-15` | -5.104158799410 | 1.9093 |
| H2O Anderson | 6 | 36 | — | — | — | — | 12 | 1 | 0 | yes | `1.30e-9 / 5.39e-10` | `5.04e-10` | `1.78e-15` | -5.104158799410 | 0.9206 |
| NO2 Broyden | 12 | 144 | 8 | 18,432 | 2,304 | 20,761 | 27 | 1 | 3 | yes | `6.05e-9 / 3.25e-9` | `3.11e-9` | `-7.11e-15` | -10.953075803272 | 2.0372 |
| NO2 Anderson | 12 | 144 | — | — | — | — | 17 | 1 | 0 | yes | `4.96e-9 / 3.02e-9` | `2.39e-9` | `-7.11e-15` | -10.953075803272 | 1.3199 |
| C4H5NCS Broyden | 38 | 1,444 | 8 | 184,832 | 23,104 | 207,961 | 30 | 1 | 3 | yes | `5.28e-9 / 1.55e-9` | `8.39e-10` | `1.42e-14` | -19.694059708019 | 2.6365 |
| C4H5NCS Anderson | 38 | 1,444 | — | — | — | — | 26 | 1 | 0 | yes | `7.47e-9 / 2.02e-9` | `1.49e-9` | `1.42e-14` | -19.694059708019 | 2.2936 |

The Fock state remains bounded and small for these cases; the largest measured
rank-8 state is about 203 KiB. All three Broyden Fock roots certified. On this
sample Broyden took more iterations and wall time than Anderson, but no
instability or unacceptable memory growth appeared. A fixed-trip Fock test on
H2O under JVP, jacfwd, and vmap returned finite derivatives/values; vmap and
serial results agreed exactly for the tested two guesses.

## Legacy short-trajectory selection

`test_production_singlepoint_matches_legacy_full` expands to these 12 cases:

```text
[zero-k-charge-simple]       [300-k-charge-simple]
[zero-k-charge-anderson]     [300-k-charge-anderson]
[zero-k-potential-simple]    [300-k-potential-simple]
[zero-k-potential-anderson]  [300-k-potential-anderson]
[zero-k-fock-simple]         [300-k-fock-simple]
[zero-k-fock-anderson]       [300-k-fock-anderson]
```

The test compares legacy and functional FULL outputs after only two SCF
iterations. It fails on the S4 package base as well as on S4; the observed
failure occurs while comparing short-trajectory densities/orbital outputs,
before a physical root is reached. Thus it is pre-existing compatibility debt,
not evidence for or against native Broyden, and is not included as S4
acceptance evidence. The direct kernel test
`test_broyden_low_rank_update_matches_vendored_first_method` is the
discriminating Broyden parity check. It now compares `c` and `d` history
corrections as well as proposals and inverse actions. The independent numerical
probe above found correction differences below `2.2e-14`.

## Python 3.10 dependency classification

The failing Python 3.10/PyTorch 2.6 environment contains `tad-mctc 0.9.1`,
which does not provide `tad_mctc.autograd.attach_gradient`. This environment is
outside the branch's declared stack: `setup.cfg` requires PyTorch `>=2.7.0`
and `tad-mctc==0.11.3`; `tox.ini` only enumerates Python 3.10 with PyTorch
2.7 and newer. The Python 3.10/PyTorch 2.7 environment has `tad-mctc 0.11.3`
and exports `attach_gradient`. There the Broyden suite passed (24 tests),
and both focused implicit Broyden route tests passed. The prior 2.6 failure is
therefore a stale unsupported environment, not an S4 compatibility failure.

## Publication status

The remote `scf/redesign` ref was verified through GitHub at
`56b6fdaee04614290e52670cd89f69cff9ca869b`. Shell Git transport still fails
locally during UID/NSS resolution before contacting GitHub. The exact S4
candidate remains unpublished pending a transport-capable runtime. No S5
work has started.
