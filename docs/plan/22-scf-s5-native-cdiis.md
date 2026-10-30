# SCF S5: native Fock-space CDIIS

## Scope and architecture

S5 adds conventional commutator DIIS as an internal, first-class
`SCFMethod`. It is usable with an explicit single-stage
`SCFStrategy((SCFStage(CDIISMethod()),))` and an
`SCFRepresentation.FOCK` problem. This package does not add a public mixer
choice, convert charge/potential states to Fock, or introduce strategy
switching. CDIIS uses the S3 shared controller and S2 physical certification.

The physical observation is paired with the input Fock that generated its map.
For that observation, the density comes from
`physical_state.map_result.spectral.density`, while the stored Fock is the
controller-projected fresh mapped value `context.proposal_x`. CDIIS performs no
additional physical map and does not call certification or inspect
convergence.

## Stationarity error and basis invariance

The current fresh Fock and its input-derived density define

```text
R_AO = F_fresh P S - S P F_fresh
R_orth = X.T R_AO X
```

where `X` is the accepted symmetric inverse square root held in
`problem.inputs.inverse_cholesky`. The pure AO and orthogonalized matrix
helpers now live in `functional/stationarity.py`; S2 certification delegates
to the same helpers, so its numerical policy is unchanged.

CDIIS forms its Frobenius Gram from `R_orth`. A deterministic nonsingular AO
basis transformation test verifies invariant orthogonalized-error Gram and
Pulay coefficients, and verifies the extrapolated Fock transforms
consistently. The raw AO error Gram changes under the same transformation.
This supports the orthogonalized metric choice for both physical meaning and
coefficient stability.

An exploratory same-basis raw-AO error comparison also reached the same roots
on H2O/GFN2 and NO2/GFN2. Both forms took 6 H2O and 11 NO2 iterations, with
the same final energies to the displayed precision and the same prune/restart
counts (2/0 and 0/0). This same-basis comparison alone does not distinguish
the metrics; the transformed-basis Gram/coefficient test does. Raw AO errors
are therefore not retained as a production mode.

## History and Pulay solve

`CDIISState` is a frozen tensor-only pytree with fixed newest-first arrays:

```text
fock_history: (max_history, nao, nao)
error_history: (max_history, nao, nao)
history_count: scalar integer tensor
```

The remaining tensor fields track restarts, pruning and the latest solve
diagnostics. New observations are inserted before extrapolation. At capacity,
the oldest entry rolls out. With fewer than two usable observations, CDIIS
uses ordinary controller-damped simple mixing toward the fresh Fock.

For errors `E_i`, the physical Gram is `B_ij = <E_i,E_j>_F`. The affine Pulay
minimization uses a relative diagonal regularizer and solves the equivalent
constrained system through an SPD Cholesky factorization:

```text
z = (B + lambda I)^-1 1
c = z / (1.T z)
F_DIIS = sum_i c_i F_i
F_next = simple_mix(F_in, F_DIIS, context.damp)
```

The normalized Gram is used only to diagnose linear dependence, separating
error scale from conditioning. Its condition limit defaults to
`eps(dtype)^(-1/2)`. Relative regularization defaults to `eps(dtype)^(3/4)`
times the mean active Gram diagonal. For float64 these are approximately
`6.7e7` and `1.8e-12`; for float32 approximately `2.9e3` and `6.4e-6`.
Tests cover separated and near-dependent vectors in both dtypes, exact
dependence, the affine constraint, and parity with the regularized KKT system.

Pruning drops the oldest entry in a statically bounded loop. If two entries
remain unhealthy, CDIIS restarts to the current observation and falls back to
simple mixing for that transition. Zero or non-finite current errors are not
inserted. All divisions use safe denominators before selection; inactive lanes
freeze every state leaf. There is no tensor-to-Python branch, dynamic history
slice, dense inverse, or AO-fourth-rank object.

## Characterization

The following exploratory runs used the shared functional controller, GFN2,
300 K, float64, `x_atol=x_atol_max=1e-8`, and a 120-map budget. CDIIS timings
are single-run characterization, not performance claims. CDIIS used damping
1.0 except the noted H2O comparison; Broyden and Anderson used damping 0.5.
One certification evaluation is counted for each completed result.

| System | Method / history | Iterations / maps | Core certs | Prunes | Restarts | History bytes | Energy (Eh) | Final stationarity | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| H2O/GFN2 | CDIIS 4, damp 1.0 | 6 / 6 | 1 | 1 | 0 | 2,304 | -5.1041587994 | 3.62e-10 | 0.522 |
| H2O/GFN2 | CDIIS 6, damp 1.0 | 6 / 6 | 1 | 2 | 0 | 3,456 | -5.1041587994 | 3.62e-10 | 0.527 |
| H2O/GFN2 | CDIIS 8, damp 1.0 | 6 / 6 | 1 | 2 | 0 | 4,608 | -5.1041587994 | 3.62e-10 | 0.540 |
| H2O/GFN2 | CDIIS 12, damp 1.0 | 6 / 6 | 1 | 2 | 0 | 6,912 | -5.1041587994 | 3.62e-10 | 0.548 |
| NO2/GFN2 | CDIIS 4, damp 1.0 | 11 / 11 | 1 | 0 | 0 | 9,216 | -10.9530758033 | 7.77e-10 | 0.920 |
| NO2/GFN2 | CDIIS 6, damp 1.0 | 9 / 9 | 1 | 0 | 0 | 13,824 | -10.9530758033 | 4.35e-9 | 0.774 |
| NO2/GFN2 | CDIIS 8, damp 1.0 | 11 / 11 | 1 | 0 | 0 | 18,432 | -10.9530758033 | 3.93e-10 | 0.928 |
| NO2/GFN2 | CDIIS 12, damp 1.0 | 12 / 12 | 1 | 9 | 0 | 27,648 | -10.9530758033 | 2.49e-9 | 1.028 |
| C4H5NCS/GFN2 | CDIIS 4, damp 1.0 | 41 / 41 | 1 | 0 | 0 | 92,416 | -19.6940597080 | 1.49e-9 | 3.682 |
| C4H5NCS/GFN2 | CDIIS 6, damp 1.0 | 27 / 27 | 1 | 0 | 0 | 138,624 | -19.6940597080 | 1.35e-9 | 2.496 |
| C4H5NCS/GFN2 | CDIIS 8, damp 1.0 | 23 / 23 | 1 | 0 | 0 | 184,832 | -19.6940597080 | 2.63e-10 | 2.143 |
| C4H5NCS/GFN2 | CDIIS 12, damp 1.0 | 18 / 18 | 1 | 0 | 0 | 277,248 | -19.6940597080 | 1.47e-9 | 1.749 |

All listed roots certified and had independent electron-count errors below
`2.2e-14`. The rank-8 provisional choice is a compact middle ground; rank 12
was faster for the difficult C4H5NCS case, while rank 4 was faster than rank 8
on NO2. These data do not identify one universally fastest capacity. The
internal default remains provisional, not a public setting.

Additional rank-8 references, all certified with one core certification:

| System | Model / temperature | Iterations / maps | Energy (Eh) | Final stationarity | Count error |
| --- | --- | ---: | ---: | ---: | ---: |
| H2 | GFN1 / 0 K | 2 / 2 | -1.0585983567 | about 1e-16 | below 1e-14 |
| H2O | GFN1 / 300 K | 4 / 4 | -5.8052487051 | 1.08e-9 | below 1e-14 |
| H2O+ doublet | GFN2 / 300 K | 6 / 6 | -4.4312251114 | 6.75e-11 | 8.88e-16 |
| Stretched H2 (2 Å) | GFN1 / 0 K | 2 / 2 | -1.0048258218 | below 1e-14 | below 1e-14 |

For H2O/GFN2, rank-8 CDIIS with damping 0.5 took 24 iterations and 4
restarts, versus 6 iterations at damping 1.0. On the three GFN2 systems above,
damping-0.5 Anderson took 11/16/24 iterations and Broyden took 26/27/30;
CDIIS at damping 1.0 took 6/11/23. These are exploratory comparisons and not
a default-method recommendation. CDIIS is effective near the root but is not
claimed as a difficult-start strategy; the C4H5NCS run did converge directly
within 41 iterations.

## Memory and transforms

For float64 and provisional history length 8, the two dominant history arrays
occupy `2 * 8 * nao^2 * 8 bytes`; the previous tensor-free counters and scalar
diagnostics add only a small constant amount. Representative dimensions and
history storage:

| System | Active AO | Fock dimension | Fock + error history | Approx. full method state |
| --- | ---: | ---: | ---: | ---: |
| H2O/GFN2 | 6 | 36 | 4,608 B | 4.7 kB |
| NO2/GFN2 | 12 | 144 | 18,432 B | 18.5 kB |
| C4H5NCS/GFN2 | 38 | 1,444 | 184,832 B | 184.9 kB |

The Pulay matrices are only `8 x 8`. Tests exercise lane-local `vmap` with
different history counts, a fixed-trip Fock solve under `vmap` and JVP, and
Pulay algebra under JVP, `jacfwd`, reverse mode, and one nested reverse-mode
second derivative. Stable unrolled CDIIS supports these focused transforms;
broader higher-order guarantees remain outside S5.

## Boundaries and validation

CDIIS is Fock-only in S5. Charge and potential inputs fail with a clear
representation error. There is no routing/configuration change, xitorch
dependency, EDIIS/ADIIS, public CDIIS selection, or S6 stage switching.
Implicit-root preparation receives only the certified physical root; CDIIS
history does not enter the response. S2 certification remains method
independent.

The focused CDIIS suite passed on Python 3.14 / PyTorch 2.14.1 and Python
3.11 / PyTorch 2.8.0. S2 certification plus S4 Broyden passed (85 combined),
the lifecycle/E6a.2 selection passed (262), implicit/root passed (58), the
spectral/free-energy/W2c5b/W2c6 selection passed (156), and Phase-A Groups 1,
2, and 3 completed with 228 passed plus the inherited Anderson parity failure,
58 passed, and 128 passed respectively. The Anderson-only selection had that
same single inherited failure at `2.5007065862503453e-11` against `2e-13`;
its tolerance and arithmetic were unchanged. Focused CDIIS test count and
final static checks are recorded in the handoff.
