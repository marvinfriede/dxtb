# S10 — xTB-consistent energy DIIS startup study

## Decision

**Decision C: adopt neither EDIIS nor ADIIS in production.** The exact-energy
xTB density interpolation is mathematically consistent and its small simplex
optimizer works, but it did not improve the existing startup portfolio enough
to justify production complexity and cost. ADIIS is only an approximate
Roothaan–Hall secant model and showed no incremental benefit over the existing
methods. No production SCF code, public configuration, or default changed.

Active-smearing calculations keep the S9 policy (Broyden, Anderson, or CDIIS).
This study did not apply an electronic-energy-only objective at active smearing
and did not add a thermal EDIIS implementation. Fixed/Aufbau calculations keep
the existing first-order choices and optional near-root SOSCF.

## Provenance and scope

- Starting S10 commit: `23c3fda56d4718dbf4ca7698dd9a70240bdd9d4c`.
- General-refactor synchronization: `8ed902f762804a9da7f4537e8d40bab80822ed1d`.
  The published general branch had not advanced, so no rebase was needed.
- Final benchmark-source commit: `c04728db6ccaa2aa83e702eb72a7f07305d2ffdf`.
- Benchmarks ran on Python 3.14.8, PyTorch 2.14.1+cpu, float64, one thread,
  Linux x86_64 (`Linux-6.4.0-150600.23.103-default-x86_64-with-glibc2.38`).
  JSON records the source SHA and a clean source-tree flag for each run.
- Reference runs are single-process, single-thread measurements. Wall times
  are characterization, not statistically controlled performance claims.
- Main artifacts are in [`benchmarks/results/scf-s10/`](../../benchmarks/results/scf-s10/).
  `final-*.json` files are the clean-source reruns used below; earlier raw
  exploratory traces are retained alongside them.

The package adds mathematical tests and benchmark-only prototypes. It does not
modify `src/dxtb`. SciPy is used only by the benchmark/reference optimizer.

## Variational energy/Fock identity

For a symmetric AO density `P`, the tests evaluate the existing density energy
`E(P) = electronic_energy_from_density(P, inputs).sum()` and rebuild the
physical Fock from its charges. The observed dxtb convention is

```text
dE(P)[V] = sum_ij F(P)[i,j] V[i,j] = Tr(F(P).T V).
```

The JVP identity passes for arbitrary symmetric directions at non-self-
consistent densities in GFN1 and GFN2, and with an applied field. The exercised
GFN2 interaction set includes ES2, AES2, ES3, and D4SC. Centered finite
differences over a step sweep agree with the same contraction. The tests also
check the coefficient derivative `dE/dc_i = <F(P(c)), P_i>` against AD and
finite differences. The electronic-energy decomposition is shared with the
accepted spectral energy helper; no SOSCF-specific energy expression is
introduced.

A low-positive-temperature fixed-occupation check confirms that, for both an
integer-electron and fractional-charge setup with `smear=False`, the entropy
contribution is constant across physical history states. This licenses the
electronic energy as a fixed-occupation interpolation objective for the
prototype; it does not license it for active smearing.

## Why the conventional quadratic model is not the xTB specification

The classic EDIIS reference follows PySCF's stored-energy/stored-density/
stored-Fock heuristic ([PySCF SCF DIIS source](https://pyscf.org/_modules/pyscf/scf/diis.html)). The test suite separates that coefficient-1 expression from
the exact pairwise identity for a quadratic functional: with the tested dxtb
convention `dE = <F, dP>`, the exact quadratic correction has coefficient
`1/4`. The `1/4` identity matches the direct quadratic toy energy to numerical
precision. The PySCF coefficient-1 heuristic does not match even that toy
energy, and the test locks this distinction rather than claiming PySCF's
heuristic passes the exact quadratic parity gate. This is an explicit
qualification of the conventional reference, not a claim that the PySCF
algorithm is an exact energy interpolant. The xTB candidate instead optimizes
and evaluates the actual energy of each convex density combination.

For the tested nonlinear xTB histories, the PySCF-style model's predicted
minimum can be materially different from the minimum of the actual xTB energy.
At history size four, representative results were:

| Case | Maximum absolute classic-model error | Exact-energy gap at classic minimum | Fock interpolation mismatch at `c=0.37` |
|---|---:|---:|---:|
| H2O/GFN1 | 9.23e-4 | 4.83e-4 | 3.17e-6 |
| H2O/GFN2 | 1.58e-3 | 8.65e-4 | 2.59e-4 |
| NO2/GFN2 | 2.07e-1 | 4.68e-2 | 2.95e-2 |
| C4H5NCS/GFN2 | 3.34 | 4.25e-1 | 1.66e-1 |

The units of the energy errors are the code's atomic energy units. These are
history-dependent examples, not universal error bounds. Removing ES3 and D4SC
from the controlled Fock interpolation case reduces the Fock mismatch to
roughly `1e-15`, identifying the nonlinear charge terms as the source in that
experiment. This directly rules out using `sum_i c_i F_i` as the exact xTB
proposal for a convex density.

The audit also includes the exact quadratic control with coefficient `1/4`; it
closely matches the exact energy for sampled histories in several small cases
but does not define the nonlinear xTB functional. For C4H5NCS even this model
has an exact-energy minimum gap of about `1.9e-3` at history size four.

## Exact-energy prototype and constraints

The research-only candidate minimizes

```text
P(c) = sum_i c_i P_i
c_i >= 0, sum_i c_i = 1
c* = argmin E_xTB(P(c))
```

using deterministic multi-start SLSQP (uniform, newest vertex, best-energy
vertex) for history capacity at most 12. The analytic coefficient gradient is
the contraction of the rebuilt physical Fock with each history density. The
selected target is `F(P(c*))`, rebuilt from the interpolated density, never a
linear combination of stored Focks. The objective optimization performs no
SCF map and no generalized eigensolve. When optimization fails, is non-finite,
or returns an objective worse than the best stored vertex, the prototype
returns that best vertex. The prototype records optimizer evaluations, time,
failures, selected coefficients, simplex feasibility, and vertex fallbacks.

Tests verify density symmetry, electron count, padding zeros, coefficient
simplex feasibility, and optimizer parity against a dense two-state simplex
grid. The histories preserve the physical electron count without repair. The
prototype rejects active smearing explicitly and has `supports_fixed_trip =
False`; it is not in `src/dxtb`.

The ADIIS reference reproduces PySCF's augmented Roothaan–Hall coefficient
formula on its test case. Its local coefficient-space gradient and Hessian are
compared against AD of the exact xTB density energy. At history size four the
GFN2 local gradient/Hessian errors were approximately `5.3e-3`/`8.0e-3` for
H2O, `2.61e-1`/`7.80e-1` for NO2, and `8.49e-1`/`4.16` for C4H5NCS. Thus it is
an approximate secant model for these nonlinear interactions, not an exact
xTB energy. It did not demonstrate an advantage sufficient for production.

## Startup and difficult-case results

All listed final runs certified and converged to the same reference energy for
the case. Counts are physical maps, not coefficient objective evaluations.
Timings include the method's total wall time after subtracting benchmark-only
map-diagnostic overhead; the coefficient optimizer cost remains included.

| Case / method | Physical maps | Wall seconds |
|---|---:|---:|
| H2O/GFN2 Broyden -> CDIIS | 5 | 0.111 |
| H2O/GFN2 Anderson -> CDIIS | 8 | 0.133 |
| H2O/GFN2 exact EDIIS -> CDIIS | 5 | 0.542 |
| H2O/GFN2 ADIIS -> CDIIS | 4 | 0.130 |
| NO2/GFN2 Broyden -> CDIIS | 8 | 0.179 |
| NO2/GFN2 exact EDIIS -> CDIIS | 8 | 1.194 |
| NO2/GFN2 ADIIS -> CDIIS | 8 | 0.249 |
| H2O+/GFN2 Broyden -> CDIIS | 4 | 0.091 |
| H2O+/GFN2 exact EDIIS | 3 | 0.125 |
| Issue #187 H2O/GFN1 Anderson | 10 | 0.057 |
| Issue #187 H2O/GFN1 exact EDIIS -> CDIIS | 4 | 0.194 |
| Issue #187 H2O/GFN1 ADIIS -> CDIIS | 4 | 0.081 |
| C4H5NCS/GFN2 Broyden -> CDIIS | 11 | 0.393 |
| C4H5NCS/GFN2 exact EDIIS -> CDIIS, capacity 4 | 16 | 3.439 |
| C4H5NCS/GFN2 ADIIS -> CDIIS, capacity 4 | 23 | 2.152 |

Additional runs cover NO2, perturbed NO2/H2O/stretched-H2 starts, switch
thresholds, and EDIIS history capacities 4/6/8/12. Perturbation can improve
map counts for exact EDIIS versus a single Broyden run, but the measured
optimizer cost remains substantial. No tested case required EDIIS or ADIIS to
rescue a failure of the existing portfolio. On C4H5NCS, exact EDIIS reduced
maps versus standalone CDIIS but was much slower than Broyden -> CDIIS. On
H2O+ and Issue #187, ADIIS was competitive in this one-thread sample, but did
not establish a broader reliability benefit beyond the existing Broyden and
Anderson hybrids.

The EDIIS -> CDIIS -> SOSCF exploratory H2O run certified during CDIIS; it did
not enter SOSCF. Coefficient/stage thresholds affected map allocation but did
not change the adoption conclusion. Auxiliary energy reduction is not SCF
certification; the existing S2 controller remains authoritative.

Separate integration tests compare exact-EDIIS -> CDIIS against standalone
Broyden and CDIIS roots for H2O/GFN2 and open-shell NO2/GFN2. They require
certification and parity of density, charges, and rebuilt physical Fock.

## Thermal scope and final policy

The exact fixed-occupation objective was not evaluated with active-smearing
occupations. The thermal channel-density/entropy objective, one-hot parity, and
thermal variational-stationarity gates were therefore not claimed or passed.
No active-smearing energy-only interpolation was run. S9's policy remains
unchanged. A future thermal interpolation candidate would need both spin
channel densities, the accepted symmetric-overlap orthogonalization, the
matrix entropy, one-hot parity with the existing spectral free energy, and a
thermal stationarity demonstration before benchmarking or adoption.

**Decision C:** keep the current Broyden/Anderson/CDIIS startup portfolio and
fixed-occupation optional SOSCF. Reject production EDIIS and ADIIS for S10.
There is no production simplex optimizer, public API/default, or EDIIS/ADIIS
method state. S11 was not started.
