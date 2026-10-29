# E6c1-W2c1 — Converged-root implicit-SCF prototype

## Scope and verdict

This is a private proof of concept using the existing eager physical-root
preparation and upstream implicit attachment. It changes no production code,
public Calculator route, or fixed-trip differentiation behavior.

The prototype is **promising for converged scalar energy observables on the
tested small GFN1 systems**. It is not a W2 clearance and does not authorize a
production route change. `W2` and `W3` remain open, and `E6broute` remains
blocked.

- Starting SHA: `9c0be0f28a39d00fac0c86eaea369b626e9a167e` (accepted S0.1b).
- Final tested implementation SHA: `a7f9851ce37627fb61cf3ec48790d99c5baf7bb2`.
  This report and the roadmap record are the following documentation-only
  closure commit.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote at start: `origin/claude/tender-dijkstra-cda42s` matched the starting
  SHA. S0.1b is already published there; W2c1 has not been published.
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0, D3 0.7.0,
  D4 0.8.0.
- Environments: Linux CPU, Python 3.11.15 / PyTorch 2.8.0+cpu and Python
  3.14.8 / PyTorch 2.14.1+cpu, one intra-op thread, one inter-op thread,
  OMP/MKL/OpenBLAS threads set to one.

The harness is [e6c1_w2c1_implicit.py](../../benchmarks/e6c1_w2c1_implicit.py).
It uses the existing root-derivative test fixture and the existing
`run_prepared_functional_implicit()` API. It times setup, root preparation,
backward, reports current/peak process RSS, and records BiCGSTAB response
status, iterations, and residual from a scoped diagnostic wrapper. Raw JSONL
results are under `benchmarks/results/e6c1-w2c1/`.

## Physical root and derivative comparison

The main comparison is GFN1 H2O, neutral, charge representation, Anderson
mixer, float64, PyTorch integrals, 300 K, and `1e-12` SCF tolerances. The
independent fixed-trip calculation executes 16 configured trips (the initial
map plus 16 map evaluations); its convergence counter reaches 12. The eager
root preparation stops after 12 map evaluations. Both are converged to the
same physical root, so comparing their derivatives is meaningful.

| Measurement | Python 3.11 / Torch 2.8 | Python 3.14 / Torch 2.14.1 |
|---|---:|---:|
| Implicit root iterations / residual norm | 12 / `1.79e-13` | 12 / `1.78e-13` |
| Fixed-trip map evaluations / residual norm | 17 / `1.62e-13` | 17 / `1.61e-13` |
| SCF energy, both routes (Eh) | `-5.805248705084274` | `-5.805248705084273` |
| Implicit free energy (Eh) | `-1.22e-16` | `-1.22e-16` |
| Fixed-trip free energy by subtraction (Eh) | `0` at printed precision | `0` at printed precision |
| Position-gradient norm, both routes | `0.17228881228261` | `0.17228881228261` |
| Model `h0_setup.selfenergy` gradient | `[1.7678134680, 0.1307298606, 1.7701366754, 5.4954860244]` | same to about `1e-14` |
| Model-derived `n0` directional gradient | `-0.001059756350794` | same to about `1e-17` |
| `n0` second derivative | `0.0178708537728` | `0.0178708537728` |
| `n0` third derivative | `1.3384874e-6` | `1.3384874e-6` |

The elementwise position-gradient difference is below `4e-16`. The largest
component difference for the shared model-parameter gradient is below
`3e-14`. The difference in the `n0` second derivative is below `6e-12`; the
third derivative agrees to approximately `4e-13` absolute. Both derivative
paths are finite. These are observed cross-route differences, not new frozen
reference tolerances.

On PyTorch 2.14.1, root preparation plus first through third derivative work
took about 8.2 seconds after input setup for implicit differentiation, versus
about 10.1 seconds for fixed 16-trip unrolled differentiation. Peak RSS was
about 553 MB implicit versus 793 MB unrolled. On PyTorch 2.8, corresponding
peak RSS was about 515 MB versus 769 MB; backward took 7.45 seconds versus
9.55 seconds. This is a 31–33% peak reduction on this small three-atom case.
It does not establish a scaling factor for tmpda or for eight conformers.

The implicit energy backward's three captured response solves reported
`CONVERGED` (upstream status 0), iteration counts 0, 1, and 0, and residual
norms `9.64e-15`, `3.37e-15`, and `1.64e-14` on PyTorch 2.14.1. The existing
failure tests also pass: a failed response solve remains observable as NaN,
including under `vmap`.

The H2O occupation count matches the target exactly; Fermi and trajectory
convergence statuses are true. The final Fermi residual is zero at reported
precision. This was a converged root, not a fixed-trip approximation to an
unconverged trajectory.

## Mapped roots and lane independence

Two independent physical roots were prepared from different initial charge
guesses. On H2 at 0 K, eager preparation took 9 and 13 iterations, with raw
residual norms `5.49e-16` and `2.60e-15`. On H2O at 300 K, it took 12 and 15
iterations, with residual norms `1.61e-13` and `5.88e-14`.

The roots were attached through `torch.func.vmap` using the existing
`implicit_root`. Per-System `n0` directions were batched; all other floating
SCF inputs, including shared model-derived data, were passed as shared
arguments. The mapped root values match independent serial attachments
exactly. For scalar energy responses:

- H2 maximum mapped/serial derivative difference: `3.4e-21`.
- H2O maximum mapped/serial derivative difference: `6.8e-17`.
- Off-diagonal cross-System energy derivative blocks: exactly zero.
- Mapped versus serial shared `selfenergy` parameter-gradient difference:
  below `9e-16` for H2 and `3e-16` for H2O.

The full root-state Jacobian probe on H2O at 300 K contains NaNs in some
components, in both serial and mapped evaluation. The scalar energy gradient
and its lane-wise Jacobian are finite and match serial results. This indicates
that some arbitrary internal-state response directions do not converge even
though the energy adjoint directions used here do. The solver returns NaNs for
those failed directions, as required; callers must not consume a full
root-state Jacobian as if it were valid. Further work would be needed before
claiming a general all-state Jacobian contract.

## Existing transform and failure controls

On each primary environment, the focused implicit-root selection passed 8
tests, covering the converged H2 higher-response comparison, model-derived
`n0` response, prepared H2 implicit route against finite differences, mapped
prepared roots, rejected unconverged roots, and float32 response. A separate
failure selection passed 3 tests on each environment, covering explicit
failed linear-solve signaling and `vmap` failure isolation. Existing tests
also cover JVP, `jacfwd`, `jacrev`, nested forward/reverse transforms, HVPs,
and third-order field responses.

Exact commands (environment prefixes omitted):

```bash
python -m pytest -q -n logical --dist worksteal \
  test/test_scf/test_functional_implicit_root.py \
  -k 'h2_implicit_higher_responses or implicit_reference_occupation_parameter or prepared_h2_implicit_route_matches_direct_energy_fd or prepared_roots_vmap_matches_loop_and_nonleaf_input_gradient or implicit_preparation_rejects_nonconverged_and_one_shot_roots or public_functional_implicit_h2_float32_response'

python -m pytest -q -n logical --dist worksteal \
  test/test_scf/test_functional_implicit_root.py \
  -k 'unconverged_linear_solve_is_explicit_and_never_used_as_a_derivative or linear_solve_failure_signal_is_vmap_compatible'
```

Results were 8/8 and 3/3 in both Python 3.11/Torch 2.8 and Python
3.14/Torch 2.14.1. Python 3.14 emitted the existing `torch.jit.script`
FutureWarning. No broad suite was run. The diagnostic uses no no-grad
substitute for its measured derivative paths.

## Review and limitations

Source review confirms the harness invokes the existing eager preparation,
root validation, `implicit_root`, and upstream BiCGSTAB; no second solver or
production monkey patch was added. The only monkey patch is a process-local
wrapper in the diagnostic to copy solver status/iteration/residual records,
then restore the original function. The numerical `torch.func` implementation
is unchanged. Public Calculator routing and E6bcompat behavior are unchanged.

No separate external reviewer was engaged for this diagnostic-only prototype.
The implementation-author review checked the harness against the existing
root-preparation, transform, and failure-contract tests; no production diff
was present for a numerical reviewer to assess.

The W2b tmpda measurement remains the governing memory evidence: 8 trips
differentiated peaked at 6.90 GiB versus 0.37 GiB no-grad, and 16 differentiated
trips peaked at 12.05 GiB. The H2O proof demonstrates reduced memory for one
small converged root; it does not establish acceptable tmpda runtime/memory,
eight-conformer scaling, or legacy packed performance. No tmpda run was made
here, and no full 300-trip run was attempted.

- Converged-root physical parity: **pass** on H2O.
- Position derivatives: **pass** against converged fixed-trip results.
- Model parameter and `n0` derivatives: **pass** at the tested boundary.
- Higher-order derivatives: **pass** for the tested energy responses.
- Mapped lane independence: **pass** for scalar energy values/derivatives;
  arbitrary full-state Jacobian has failed directions as described above.
- Implicit response convergence: **pass** for tested energy response solves;
  failures remain observable as NaNs.
- Memory improvement: **verified** on small H2O, 31–33% peak RSS reduction.
- Performance viability: **promising for further study**, not validated for W2.
- W2 clearance: **open**.
- W3 clearance: **open**.
- E6broute: **blocked**.
- Public Calculator routing: **unchanged**.

The W2c1 prototype is accepted at this bounded small-system energy-observable
boundary. It is not an authorization to substitute implicit differentiation
into production and does not close W2/W3.
