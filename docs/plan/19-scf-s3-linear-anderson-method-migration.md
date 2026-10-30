# SCF S3: Linear and Anderson share the method/controller architecture

## Package boundary

S3 started from SCF commit `6d5cfd1b123b299e9be71570c2e2805761e9567c` and
rebased once onto general-refactor synchronization `124a464e952e230ac8489ced48c5ef9e9295dcb6`.
The resulting pre-edit SCF head was `f845602ea7547f056000c0fdb7bf599867656cb8`.
The general-refactor worktree was not modified. Its later W2c9 claim concerns
`implicit.py`, `spectral.py`, `map.py`, and `data.py`; S3 did not edit those
files or rebase during the package.

## Method and settings ownership

`setup_solver_settings(config, dtype)` now creates the common immutable
`SCFSolverSettings` for either method. It owns the iteration budget, initial
mixing choice, ordinary damping and dynamic-damping settings, and effective
raw-root tolerances. `setup_simple_solver_settings` remains only as a thin
compatibility alias; production routing uses the neutral builder.

`AndersonMethodSettings` contains only `damp_init`, `generations`,
`soft_start`, and `diagonal_offset`. `AndersonState` remains an immutable
fixed-size pair of history tensors. The new `AndersonMethod.propose()` uses
the paired physical input and `SCFMethodContext.proposal_x`; it performs no
map evaluation and does not make convergence decisions. Its update delegates
to the existing `anderson_mix` arithmetic. Accelerated updates consume the
controller's dynamic damping, while warm-up keeps `damp_init`.

## Shared execution path

The configuration boundary `_prepare_solver()` resolves the effective mixer
once and builds a one-stage `SCFStrategy`. The charge, potential, Fock, and
implicit-primal paths pass that strategy to the shared `_solve_map()` and
existing eager/fixed-trip controllers. The controller has no Anderson branch.
The Anderson-specific `_solve_map_anderson` driver and its nested transition
closure were removed. Production `mix_fn` callbacks were removed as well.

`solve_scf_problem()` centralizes representation-specific finalization for the
common strategy path. Existing simple representation functions are thin
delegators. Potential padding projection still occurs before method proposals;
Anderson history therefore records the projected controller proposal. Generic
trace state is now named `method_state`.

The strategy remains single-stage. Dynamic damping and all convergence,
terminal-failure, status, root-pairing, and S2 certification policies remain in
the shared controller. Implicit preparation still sees only the certified
root and its exact raw map; no Anderson history enters the implicit boundary.

## Validation

The Anderson kernel/history tests, common-controller transform tests, all
three representation tests, terminal-history freeze test, lifecycle/E6a.2,
certification, implicit-root, padded-trace, and W2c5b/W2c6 tests were run on
Python 3.14 / PyTorch 2.14.1. Focused method/controller, certification, and
implicit tests also passed on Python 3.11 / PyTorch 2.8.0. The slow NO2
potential implicit response test passed. The existing diagnostic E30 fixed-N
Anderson derivative test at 18 maps passed.

Phase-A results after S3:

| Selection | Result |
| --- | ---: |
| Functional map, solver, Anderson, eager compatibility | 225 passed, 1 failed |
| Implicit root and root derivatives (`not slow`) | 54 passed |
| Public status, invariant spectral density, invariant free energy | 128 passed |

The sole Phase-A Group 1 failure remains
`test_anderson_solver_real_position_parameter_and_field_gradients`, with
maximum absolute difference `2.5007065862503453e-11` against `2e-13`, matching
the inherited strict-parity failure. No tolerance was changed. The E6b padded
trace selection passed 51 tests; the W2c5b/W2c6 selections passed 39 tests.

The W2c2 benchmark still imported the removed Anderson driver and settings
helper. Its primal setup was updated to call `_prepare_solver()` and the common
`_solve_map()` with the resulting strategy. This is a call-site migration only;
it does not alter W2 calculations.

## H2O/GFN2 performance smoke

The same H2O/GFN2 Anderson case was run once at S2b.1 (`6d5cfd1`) and once at
S3. Both runs used `maxiter=40`, `mix_guess=True`, one CPU thread, and the same
initial state. These are smoke timings, not a performance benchmark.

| Controller | S2b.1 time | S3 time | Iterations | Maps | Core certifications | Energy (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 0.0850 s | 0.0850 s | 10 | 10 | 1 | -5.1041587994095075 |
| Fixed-trip | 0.4611 s | 0.4545 s | 10 | 41 | 41 | -5.1041587994095075 |

The map and certification counts matched between revisions. The small timing
difference is within the noise expected from a single run.

## Review-gate accounting

The earlier Phase-A Group 3 summary reported 131 tests, while the S3 summary
reported 128. Collection was repeated with the same explicit selection at
the S1.1 tree (`0053e265`), the published S2b.1 tree (`6d5cfd1`), the pre-S3
parent (`f845602`), and the S3 tree. Each collected the same 128 node IDs; the
before/after node-ID diff is empty. The three test files are unchanged between
the S1.1 and S2b.1 trees. The Group 3 selection was rerun at S3 and completed
with 128 passed. Thus S3 removed no Group 3 coverage; the historical 131
summary was an unreproducible collection/count discrepancy, and the original
command that produced that number was not preserved.

The exact Group 3 command used for the comparison was:

```bash
python -m pytest -q -n 1 \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_scf/test_invariant_spectral_density.py \
  test/test_scf/test_invariant_free_energy.py
```

The earlier broad xdist invocation was interrupted during worker shutdown and
did not produce a final pytest summary; its complete command and concurrency
value were not preserved. The same combined file selection was rerun without
xdist and with `-n 2`, using one thread for each BLAS/OpenMP library and
`-m 'not slow'`. Both runs completed normally with 562 passed, one inherited
Anderson strict-gradient-parity failure, and four deselected. The maximum
difference remained `2.5007065862503453e-11` against `2e-13`. No pytest worker
processes remained after either run. The shutdown hang is therefore not
reproduced at serial or low xdist concurrency; the earlier incomplete run is
classified as a transient/high-concurrency runner event, not an S3 numerical
regression.

## Scope status

S3 adds no switching or multi-stage execution. Broyden, CDIIS, SOSCF, and later
strategy packages were not started. S2 certification and public status policy
were not redesigned.
