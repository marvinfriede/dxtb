# SCF S1.1 zero-budget status correction

## Review finding

The S1 fixed-trip controller accidentally recorded the initial raw-map Fermi
status when `maxiter=0`. The E6a.2 public status contract instead treats that
one-shot as having no physical trajectory. The raw map still runs to produce
the result, but its status is not trajectory status.

The regression escaped S1 validation because the focused S1 matrix omitted
`test/test_calculator/test_e6a2_public_status.py`.

## Accepted E6a.2 contract

For fixed-trip/System `maxiter=0`:

```text
trajectory_status = None
status source = final spectrum only
converged = False
iterations = 1
residual = G(guess) - guess diagnostic
```

For `maxiter > 0`, the initial physical raw-map status seeds trajectory
status, and subsequent active raw maps accumulate status. Fixed-trip dummy
maps after convergence remain excluded by the existing active mask.

Eager Calculator compatibility is intentionally distinct: its zero-budget
initial raw-map status behavior remains unchanged. Physical outputs and the
iteration budget remain compatible; status ownership need not match.

## Correction

In `_solve_map_control`, initialization now sets
`record_initial_status=maxiter > 0`. The eager controller initialization was
not changed. No configuration, convergence, status-code, or S1 architecture
changes were made. `functional_subsolves_ok` already treats absent trajectory
status as structurally valid and checks final Fermi status, so it required no
change.

Potential and Fock zero-budget tests now assert that the raw and final spectra
were evaluated while only final status is exposed. The S1 contract test
explicitly checks fixed-trip/eager status distinction, one iteration,
nonconvergence, and compatible physical values. The E6a.2 public regression
checks both zero-trip Aufbau and finite-temperature final Fermi failure.

## Validation

The regression reproduced before the fix in
`test_public_zero_trip_has_final_only_status`: the finite-temperature row
reported trajectory and final failure code `FERMI_NONCONVERGENCE` instead of
`NONE` and `FERMI_NONCONVERGENCE` respectively. The zero-trip Aufbau row
passed.

After correction:

| Selection | Result |
| --- | ---: |
| S1 contracts, solver/eager compatibility, E0 iteration, potential, Fock, and full E6a.2 public status | 259 passed |
| Implicit root and root derivatives (`not slow`) | 54 passed |
| Invariant spectral density and free energy | 117 passed |
| Phase-A Group 1 | 225 passed, 1 inherited failure |
| Phase-A Group 2 | 54 passed |
| Phase-A Group 3 | 131 passed |
| Python 3.11 / PyTorch 2.8.0 E6a.2 zero-trip node | 2 passed |

The Group 1 failure remains
`test_anderson_solver_real_position_parameter_and_field_gradients`, with
maximum absolute difference `2.5007065862503453e-11` against `2e-13`; this
matches the documented inherited failure. The Python 3.14 runs emitted only
the existing TorchScript deprecation warning.

The S0.1c synthetic unsupported-kernel lifecycle probe was repeated with a
matrix-Fermi input forced beyond the supported envelope. For fixed-trip
`maxiter=0` and `force_convergence=True`, the public result remained
`converged=False`, `iterations=1`, with non-finite density and energy. Both
public failure codes remained `NONE`; there is no dedicated numerical-envelope
status, which remains deferred to S2. No false successful convergence was
reported.

Static checks passed: `compileall`, isort, Black, and `git diff --check`.

Future SCF packages that touch solver lifecycle, status accumulation,
finalization, or fixed-trip behavior must include
`test/test_calculator/test_e6a2_public_status.py` in their validation matrix,
in addition to the existing Phase-A selections.

## Coordination

The SCF agent read the shared `agent-coordination.md`, recorded the completed
S1 handoff, and claimed S1.1 before fetching or editing production code. The
published general-refactor branch had advanced from `a0f35c9` to `66e9ce8`;
the intervening W2c3 commits changed benchmark and planning files only. The
coordination log marked that package boundary safe, and the clean SCF branch
was rebased to it before this correction. No general-refactor worktree files
were modified. Before publication, the general branch advanced again to
`573f706df2047d7ee96fdfeee25f8a0378cc5966`; its W2c4 commits changed only
benchmark and planning files, and the completed handoff marked synchronization
safe. The clean SCF package was rebased to that newer published head. The
resulting S1.1 branch contains no unpublished general-refactor dependency.
