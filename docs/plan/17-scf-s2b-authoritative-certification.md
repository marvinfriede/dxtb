# SCF S2b authoritative physical certification

## Scope and synchronization

S2b started from SCF `456fe680dfbe3a318941c49f62921394208b6140` and rebased
onto the published general-refactor head
`45f2e9ec98368c51521146559e2c3c615ae09d2b`. The intervening general-refactor
changes were confined to W2c5b benchmark, documentation, and test files; no
production SCF source changed. The B3b inverse-Cholesky test expectation was
reported in the shared coordination log for the general-refactor owner; it was
not changed here.

S2b makes the S2a physical certification result authoritative for functional
SCF. It does not migrate Anderson, change method arithmetic, or change the
public `SCFStatus` schema.

## Threshold resolution and diagnostics

The standard profile uses the actual effective root thresholds carried by the
current solver settings:

| Value | Standard policy |
| --- | --- |
| Fixed-point L2 | caller's effective `x_atol` |
| Fixed-point max | caller's effective `x_atol_max` |
| Stationarity | internal standard profile: `1e-5` in float64/float32 |
| Independent electron count | effective fixed-point L2 threshold |

Tight and higher-order/reference profiles remain explicit opt-ins. Their root
pairs are `1e-10 / 1e-10` and `1e-12 / 1e-12`, respectively, with the frozen
stationarity and electron-count policies described in plan 16. No threshold is
selected from gradient state or transform mode.

Solver settings preserve both requested and dtype-resolved root tolerances.
Certification diagnostics report requested and effective fixed-point L2/max,
stationarity, and electron-count thresholds. For float32, a requested `1e-12`
root tolerance resolves to the existing `5e-5` floor; stationarity remains at
the explicit `1e-5` profile floor. The count threshold follows the effective
root L2 threshold.

The authoritative controller path calls `certify_scf_core_state`. It computes
fixed-point, fresh-Fock stationarity, independent `Tr(P S)` count, finite, and
root-state Fermi checks. It does not evaluate electronic energy, entropy, or
free energy. The full `certification_metrics` and `certify_scf_state` diagnostic
APIs remain available for optional observables.

## Physical and operational status

`SCFSolveResult.converged` now means the physical root
`SCFPhysicalState(root_x, G(root_x))` passed all core checks. It is independent
of mixer history and accumulated trajectory status. `trajectory_status`
remains an operational historical diagnostic; the separate final-spectrum
Fermi status remains part of finalization/return validity. A stationarity or
independent-count failure is not converted into a public Fermi failure code.

The internal immutable certification diagnostics include component booleans,
the selected failure code, raw and orthogonalized stationarity observations,
electron count/error, finite/Fermi validity, and effective/requested
thresholds. They are not added to the public `Result` pytree or public status
schema.

## Controller behavior

Both controllers reuse the existing shared lifecycle and transition code.

### Eager

The initial setup map is not a convergence candidate, preserving iteration
ordering. Each later raw map is checked with the existing projected L2/max
criterion. Only a raw candidate receives core certification. A certified
candidate stops at its exact input `x`, paired with `G(x)`. A finite candidate
that fails stationarity, independent count, or recoverable Fermi convergence
continues through the normal method transition while budget remains. A
non-finite, invalid-temperature, or too-many-electrons state is terminal and
is not mixed into the next iterate.

The eager host policy still handles existing subsolve/public-status failures
separately from physical convergence. `maxiter=0` remains the accepted one-shot:
one initial map, `converged=False`, and no implicit-root attachment.

### Fixed-trip

The transform-compatible fixed-trip controller uses tensor masks only. It
evaluates core certification for every statically scheduled loop candidate and
performs a final certification recomputation on the selected root/map pair.
Terminal failures become inactive without becoming certified; final
`converged` is the explicit `certified` state, never `~active`. Dummy loop maps
cannot replace the selected physical pair. Unsupported/non-finite candidates
do not reach a method update with NaNs. Zero budget keeps the S1.1 final-only
status behavior and is always non-converged.

Certification rebuilds `F[P]` algebraically. It performs neither a raw SCF map
nor a diagonalization, so it changes neither physical-map counts nor the
reported iteration budget.

## Implicit root attachment

Implicit preparation now requires an S2-certified primal outcome and valid
final/root Fermi state. Historical trajectory status is not part of the root
acceptance predicate. A non-certified outcome is rejected before preparing the
transform adapter. The independent exact-parameter raw-map revalidation still
checks finite residual, L2, and max against the same effective caller
thresholds before IFT attachment.

## Performance observation

The following single-thread CPU measurements used Python 3.14 / PyTorch 2.14.1,
a 40-step budget, `x_atol=1e-4`, `x_atol_max=1e-5`, and three timed samples after
one warmup. They report the complete eager/fixed-trip solve including normal
finalization. Eager certification ran once at the accepted raw candidate for
the first three cases. Fixed-trip certification ran 41 times (40 static loop
slots plus final selected-pair diagnostics), even when a lane was inactive.

| Case | Controller | Median wall time | Reported iterations | Raw-map calls | Core-cert calls | Converged |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| H2 / GFN1 | eager | 0.081 s | 16 | 16 | 1 | yes |
| H2 / GFN1 | fixed-trip | 0.246 s | 16 | 41 | 41 | yes |
| H2O / GFN2 | eager | 0.148 s | 19 | 19 | 1 | yes |
| H2O / GFN2 | fixed-trip | 0.441 s | 19 | 41 | 41 | yes |
| NO2 / GFN2, 300 K | eager | 1.300 s | 17 | 17 | 1 | yes |
| NO2 / GFN2, 300 K | fixed-trip | 3.236 s | 17 | 41 | 41 | yes |
| C4H5NCS / GFN2, 300 K | eager | 3.626 s | 41 | 41 | 1 | no |
| C4H5NCS / GFN2, 300 K | fixed-trip | 3.700 s | 41 | 41 | 41 | no |

As expected, eager avoids certification work except at a raw candidate. The
fixed-trip path pays for every certification algebra call because masking does
not skip matrix operations. It also retains its static raw-map schedule. This
is a correctness-first tradeoff; future performance work may optimize it only
with the transform and higher-order contract intact. The S2a.2 core-only
microbenchmarks measured 0.27–0.52 raw-map equivalents for small cases and
0.05–0.37 for NO2/larger cases; those are useful context, not a substitute for
end-to-end timing.

## Validation

Python 3.14 / PyTorch 2.14.1 results:

| Selection | Result |
| --- | ---: |
| Certification | 53 passed |
| S1/lifecycle/E6a.2 selection | 259 passed |
| Anderson | 32 passed, 1 inherited failure |
| Implicit/root (`not slow`) | 54 passed |
| Spectral/free-energy | 117 passed |
| W2c5b shift contract | 12 passed |
| Phase-A Group 1 | 225 passed, 1 inherited failure |
| Phase-A Group 2 | 54 passed |
| Phase-A Group 3 | 131 passed |

The only failure is
`test_anderson_solver_real_position_parameter_and_field_gradients`: maximum
absolute difference `2.5007065862503453e-11` against `2e-13`. Its test
tolerance and Anderson implementation were not changed. Python 3.11 / PyTorch
2.8.0 passed the selected standard-threshold, float32 count, vmap/JVP/jacfwd/
nested metric, implicit-root, and fixed-trip transform smoke tests (7 passed).

## Deferred work

S2b does not change public status categories or add physical failure codes to
`SCFStatus`. It does not migrate Anderson to `SCFMethod`, implement Broyden,
CDIIS, or SOSCF, change convergence thresholds, redesign implicit response, or
start S3. Future public status work may expose the internal physical failure
reason separately from historical trajectory and final-spectrum Fermi status.
