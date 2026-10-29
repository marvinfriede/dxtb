# E6c1-W2c1a — Mapped implicit-root validation

## Scope and status

This package corrects the private W2c1 mapped-root diagnostic. It does not
change public Calculator routing or replace the fixed-trip SCF derivative
contract. W2c1a is **accepted at the validated physical-observable boundary**:
two independently prepared H2O geometries pass mapped/serial energy, force,
reference-occupation, shared-parameter, and lane-isolation checks on the
primary CPU profiles. Arbitrary full root-state Jacobians remain outside the
accepted observable contract and are explicitly unverified; the bounded
directional probe is classified below as a pre-solver execution-cost limit.

- Starting SHA: `6ae03ac15a7d9c78739c62cdae8ab6a32e6d26e2`.
- Tested implementation SHA: `40e878ceee0e2a5aaefaf0b59c19958bd584fde5`; the
  report and roadmap commit follows the implementation/tests.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: unchanged at the accepted starting baseline; no publication made.
- Implementation commits: `87efa84` root validation, `a3ccde5` regressions,
  and `40e878c` staged diagnostics. `74369d7` adds the requested `AGENTS.md`.
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0, D3 0.7.0,
  D4 0.8.0.

## Root/parameter consistency

`_prepare_scf_root()` now evaluates the raw physical residual against the exact
parameter tuple intended for implicit attachment. It checks finite residuals,
the vector norm against `x_atol`, and the maximum component against
`x_atol_max`; an inconsistent root raises `SCFConvergenceError` before
implicit differentiation. The functional eager caller passes its configured
tolerances. The negative regression changes `n0` while retaining the old root
and confirms that this boundary rejects the mismatch.

The mapped geometry test uses neutral GFN1 H2O at 300 K with PyTorch
integrals. Lane 0 uses the fixture geometry; lane 1 displaces one atom by
0.2 bohr. It constructs position-dependent integrals and interaction inputs
for each lane, prepares and validates each physical root independently, then
maps all varying floating leaves. Both lanes share the same Model parameter
owner. The independent raw residual norms are at most `1e-12`; both roots
report converged Fermi and electron-count status, and their SCF iteration
counts differ.

The test also compares mapped attached root values directly with independent
serial attachments. A separate `n0` energy Jacobian compares mapped and serial
responses and asserts zero off-lane blocks. The existing
`test_implicit_reference_occupation_parameter_matches_tight_unroll_fd` passes
on both primary profiles, providing an independent reference for the physical
scalar reference-occupation response.

At the corrected benchmark's lane-specific `n0` experiment, each finite
parameter value now has its own prepared root before differentiation. The
diagnostic separates root preparation, attachment, energy, and state-response
stages. The energy stage asserts mapped/serial total-energy parity rather than
only logging a difference.

## Numerical results

The two-geometry automated test passes on Python 3.11 / Torch 2.8 and Python
3.14 / Torch 2.14.1. It compares mapped attached roots directly against
independent serial attachments, energy and free energy separately, position
derivatives against independent serial implicit evaluations, shared Model
parameter gradients against the serial sum, verifies zero off-diagonal
position blocks, and verifies lane-order equivariance. Mapped/serial energy
parity in the staged H2O benchmark was within `8.9e-16 Eh`.

A separate GFN1 H2 case at 10,000 K has nonzero Fermi free energy
(`|F_fermi| > 1e-4 Eh`) and fractional occupations. Electron count, total and
electronic energies, position gradient, Model parameter gradient, and finite
temperature derivatives through third order were checked. First, second and
third temperature derivatives agree with five-point finite differences whose
four perturbed points were independently converged. Position and shared
Model-parameter gradients agree with a separately prepared serial implicit
root. This is a deliberately conditioned proof case; it does not alter the
frozen W2 temperature.

Test results, run in separate processes to avoid stacking memory peaks:

| Selection | Python 3.11 / Torch 2.8 | Python 3.14 / Torch 2.14.1 |
|---|---:|---:|
| Mapped physical lanes + root/parameter parity + `n0` Jacobian | 1 passed | 1 passed |
| Mismatched-root rejection | 1 passed | 1 passed |
| Nonzero finite-temperature entropy and responses | 1 passed | 1 passed |
| Existing implicit-root focused transform/failure selection | 5 passed, 26 deselected | 5 passed, 26 deselected |
| Reference-occupation energy derivative vs tight unroll FD | 1 passed | 1 passed |
| H2 implicit first/second/third responses vs converged unroll | 1 passed | 1 passed |
| E7a mixed finite-temperature trace/padding control | 1 passed | 1 passed |
| E6bcompat packed mode-2 control | 1 passed | 1 passed |

After changing physical root preparation to `no_grad`, the mapped-geometry
test with the reference-occupation Jacobian passes in 14.23 seconds on Python
3.11 / Torch 2.8 and 13.62 seconds on Python 3.14 / Torch 2.14.1. Peak RSS was
606 MB and 637 MB. The mismatch rejection takes 0.37 seconds and 0.25
seconds. The entropy test passes in 40.46 seconds and 37.07 seconds; peak RSS
was 771 MB and 802 MB. The separate reference-occupation tests peak at 387 MB
and 448 MB, and the implicit higher-response tests at 485 MB and 532 MB. The
E7a trace/padding controls pass in 16.61 seconds and
15.88 seconds; E6bcompat mode-2 controls pass in 1.14 seconds and 0.90
seconds. Each test ran in a separate process. Root preparation no longer
retains the unrolled solve graph; derivatives are evaluated after implicit
attachment.

## Full root-state response probe

The earlier NaNs in the full root-state Jacobian came from attaching a lane's
perturbed `n0` to a root prepared for unperturbed inputs. Those NaNs are
withdrawn as evidence about a valid physical response. The corrected
diagnostic uses lane-isolated directional JVPs and emits solver status,
iterations, RHS norm and residual when the response solve is reached.

A bounded Python 3.11 / Torch 2.8 run prepared both roots, checked exact
mapped/serial attachment parity, and began lane 0's JVP. It reached a
60-second timeout at 1.80 GiB peak RSS without an
`implicit_response_solver_entered` event. The implicit-root forward rule forms
the parameter-direction RHS before calling `implicit_solve`; the timeout is
therefore in the JVP of the raw physical SCF residual, before BiCGSTAB. It is
not a linear-solver failure and produced no NaN or response status. An earlier
uninstrumented 90-second attempt peaked at 2.56 GiB and also produced no solve
result. No further full-Jacobian or directional retry was made.

Arbitrary full root-state Jacobians are not required for the validated energy,
free-energy, force and shared Model-parameter observables. Their general
validity remains unverified and is not claimed. The prior NaNs are explained
by an invalid root/parameter pairing; the corrected internal-state JVP is
classified as too costly for the bounded diagnostic, before the response
solver begins. Supported scalar `n0` response, mapped position derivatives,
shared parameter gradients and finite-temperature derivatives pass
independent references.

## Memory-aware validation boundary

No unrestricted suite or unrolled third-order batch was run. Existing
W2b measurements show differentiated fixed-trip SCF memory growing by about
0.64 GiB per trip over the measured 2–16 trip interval. The full root-state
Jacobian is outside the accepted observable contract. The bounded JVP is
recorded as a diagnostic cost, not substituted for physical derivative
acceptance.

Formatting, import ordering, Python compilation, and `git diff --check` pass.
The focused mapped, entropy, reference-occupation, higher-response, E7a and
E6bcompat controls listed above pass on both primary profiles. The corrected
two-lane `n0` energy stage passes on Python 3.11: mapped/serial total energies
differ by `8.88e-16 Eh`, with 643 MiB peak RSS. The directional internal-state
probe timed out before the linear solver began. The large E7a mixed
third-order selections and the full E6bcompat 64-node selection were not
repeated because of their memory footprint; prior acceptance remains their
evidence. No five-profile rerun was made.

## Verdicts

- S0.1b: **preserved**; no spectral production code changed.
- W2c1 single-system implicit prototype: **preserved** at its previously
  accepted small-system observable boundary.
- Mapped root/parameter consistency: **verified** for tested cases.
- Two distinct physical configurations: **verified** for H2O geometries.
- Mapped scalar-energy parity: **pass**.
- Position derivatives: **pass** for the tested lanes.
- Model parameter derivatives: **pass** for shared parameters.
- Nonzero finite-temperature free-energy response: **pass** for conditioned H2.
- Full root-state Jacobian failures: **explained and classified**; earlier
  NaNs came from mismatched roots/parameters. Correct-root arbitrary internal
  state derivatives remain unverified; the bounded JVP timed out before the
  response solver.
- Implicit response convergence: **verified** for tested physical scalar
  energy/free-energy, position, reference-occupation and Model-parameter
  responses; no claim for arbitrary full-state directions.
- Mapped cross-System independence: **pass** for position derivatives and
  scalar observables.
- Independent review: accepted at the documented physical-observable
  boundary. The reviewer confirmed the lane/parameter pairing, serial
  references, and bounded timeout classification; arbitrary full-state
  Jacobian support remains explicitly unverified.
- W2c1a: **accepted at the documented observable boundary**; no production
  implicit route migration is authorized.
- W2c2: **not started**.
- W2/W3: **open**.
- E6bcompat: unchanged; no regression observed in the tests run here.
- Public Calculator routing: **unchanged**.
- Remote publication: **not pushed**.
