# E6bimpl.2c — Final mixed-System numerical acceptance

**Status:** private PyTorch numerical implementation **COMPLETE**; overall
E6bimpl **OPEN** for the policy-assigned compatibility composition listed
below.

Starting SHA: `2a477b9b181e6e84909e82b204853b3a203ad38c`.

This acceptance pass makes no production-code changes. It corrects one stale
C8b test expectation and adds a full-energy GFN2 Hessian-vector comparison.
Public Calculator routing, legacy classes, and dependency files remain
unchanged. `AGENTS.md` remains untouched and untracked.

## Starting-point and independent-review reconciliation

HEAD matched the required SHA. The preceding commit `2a477b9` added an
independent transform review to the GFN2 D4 EEQ cutoff regression. The
reviewer found that the prior test differentiated a lower-level fixed-trip
helper, so its `jacrev` did not cross the actual `evaluate_padded_systems()`
`vmap` boundary. The correction changed the test to differentiate the public
private orchestration entry point, with a JVP control, and added no production
change. This was a test-coverage gap, not a false-positive result. The current
test calls `evaluate_padded_systems()` for both mapped `jacrev` and JVP and
compares against an independently evaluated serial D4 expression. The
reviewer reported no remaining blocker or major finding.

The final source review found no additional defect. One older C8b.3a test
still asserted that a GFN2 method identifier was rejected as “GFN1 cores
only,” although canonical padding now intentionally supports both methods.
That stale expectation is replaced with assertions that method identity
changes the signature and partitions the two entries into distinct buckets.
Actual production GFN2 setup and canonical evaluation remain covered by the
GFN2 tests; the method-edited core in the C8b test is explicitly documented as
an invalid fixture used only to inspect the static signature.

## Frozen E6bimpl contract

The 16 requirements are taken from `E6bcompat-policy.md`. This matrix records
the implementation owner, direct evidence, tested boundary and limits.

| # | Requirement | Owner and evidence | Accepted configuration / tolerance | Limit |
|---:|---|---|---|---|
| 1 | Numerical values | `padded_integrals.py`; mixed GFN1/GFN2 tests | Float64 PyTorch. H0/overlap and multipole physical blocks: `atol=rtol=2e-12`; interaction kernels: `5e-9`; GFN2 full Result fields: `5e-8`; established energy/gradient-specific assertions are retained. | No CUDA acceptance. |
| 2 | Energy, SCF, classical and optional Result fields | `functional/padded.py`; GFN1 and GFN2 Result parity tests | Complete supported Results; tree schema, labels/order, optional `None`, energy, density, H0, overlap, multipoles, spectrum, iterations, convergence, residual and status are checked. GFN2 tensor fields use `5e-8`; occupations in degenerate subspaces are compared by energy cluster. | Eigenvectors are not compared without gauge fixing. |
| 3 | Charge and spin | GFN1 H2/H3+ and open-shell cases; GFN2 charged/open-shell acceptance | Physical source atoms only; serial electron count, occupations, density and energy parity. GFN2 includes H2+ and H2− lanes and lane-local over-capacity status. | Only valid configurations supported by the existing production method are accepted. |
| 4 | Masks and padding | `PaddedCoreValidity`; `test_e6bimpl1a_padded_spectral.py`; `test_e6bimpl2b_gfn2_scf.py`; scatter tests | Noncontiguous AO validity, inactive atom/shell/orbital masks, multiple dummy states, and zero padded-coordinate gradients. Exact zeros are asserted for inactive blocks where the representation contract specifies zero. | Arbitrary external mutation of a padded core is not a public API. |
| 5 | Atom/shell/AO maps | `canonical_padding.py`; C8b tests; H2/H3+, H2/H2O, H2/LiH and reversed GFN2 fixtures | Explicit source-to-canonical and inverse mappings; per-entry matrix parity. | GFN0 canonical mapping remains outside this package. |
| 6 | Integrals and H0 | `padded_integrals.py`; E6bimpl.0/2a suites | Production PyTorch kernels; every physical H0/overlap entry, reference occupation and supported dipole/quadrupole block compared. H0/overlap use `2e-12` absolute and relative tolerances. | Libcint whole-System mapping is not claimed. |
| 7 | No fictitious physical contribution | Explicit atom/shell/orbital/pair masks; adversarial padding tests; GFN1/GFN2 decoded Result parity | Padded coordinates, dummy orbitals, invalid pair entries and padded multipoles do not alter physical observables. | The shared dummy spectral level is not treated as the validity mask. |
| 8 | No cross-System derivative blocks | GFN1/GFN2 energy-vector Jacobians and mapped integrals/spectral tests | Position Jacobian diagonal blocks match independent serial derivatives; off-diagonal and inactive-coordinate blocks are zero within the test's stated tolerance. | Verified for the listed PyTorch mapped fixtures. |
| 9 | Position gradients | GFN1 `51`-case module; GFN2 mixed derivative tests | GFN1 and GFN2 float64 production energies. GFN2 gradient parity `2e-6`; GFN1 uses its existing established assertions. | The known unrelated legacy/functional Anderson discrepancy remains separately owned. |
| 10 | Supported parameter gradients | GFN1 D3 `s6`, H0 and ES2; GFN2 H0 `wexp`, ES2 `gam`, ES3 `gam3`, AES2 `dmp3`, D4SC `d4.a1`, classical D4 `d4.a1`, Repulsion `arep` | Mapped total-energy gradient equals sum of independent serial gradients; selected family tests require nonzero sensitivity. GFN2 component gradients `atol=2e-7, rtol=2e-6`; mixed position/ES2 response `3e-6 / 3e-5`. | No differentiable cutoff contract; unsupported custom parameter families are not implied. |
| 11 | Convergence and failure metadata | Shared fixed-trip solver; SCFStatus tests and mixed lane tests | Zero-trip, positive-budget, different lane convergence, finite-temperature status and lane-local too-many-electrons are covered. Tensor status retains separate trajectory/final fields. | No stable physical full-System GFN2 Fermi-nonconvergence fixture; lower-level status projection tests cover that code. |
| 12 | Custom component policy | Padded evaluator preflight and `E6bcompat-policy` controls | Unsupported custom descriptors fail explicitly in mapped evaluation; existing legacy execution remains available pending route migration. | Arbitrary callbacks are not claimed transform-compatible. |
| 13 | Argument and option preservation | E6bcompat policy tests; public dispatch unchanged | Shared field `(3,)` and field-gradient `(3,3)` behavior is accepted for GFN1. Existing legacy option forwarding and error tests pass. | The private evaluator is not a new Calculator option surface. |
| 14 | Deterministic ordering | Canonical bucket tests, noncontiguous IDs `(4,19)`/`(19,4)`, decoded Results | Reversed source order preserves decoded lane order and values; duplicate or invalid maps reject. | Mixed-method orchestration is not exposed; incompatible methods separate into different buckets. |
| 15 | Source-System immutability | Frozen registered cores, out-of-place padding, E6b2a/2b parameter snapshots | Model-owned parameter identities and exact values are unchanged around setup/evaluation/gradient calls. | The private API does not promise a general mutation-proof wrapper for arbitrary user extensions. |
| 16 | One Model parameter owner | C8b parameter-tree controls; GFN1/GFN2 parameter-gradient tests | Parameter identity is checked before/after; mapped gradients match the sum from independent serial Systems; no per-System parameter table is created. | Incompatible parameter owners/configurations cannot share a bucket. |

## Cross-method numerical and component matrix

| Method | Production fixtures and controls | Supported built-in contribution inventory | Result/solver coverage |
|---|---|---|---|
| GFN1 | H2/H3+, H2/H2O, reversed H2O/H2, H2O/CH4; neutral and open-shell; float64 | H0, ES2, ES3, Repulsion, official D3; shared electric field and field gradient | Charge/potential/Fock SCP × linear/Anderson, zero/finite temperature, fixed-trip and zero-trip, trace parity, lane-local residual/status, complete Result decode, position/parameter/forward derivatives. |
| GFN2 | H2/H2, H2/LiH, H2/H2O, reversed H2O/H2, H2O/CH4; charged/open-shell and finite temperature | H0, ES2, ES3, AES2, D4SC, Repulsion, classical D4 | All six SCP/mixer combinations on the accepted H2/LiH fixture, full fixed-trip Result and status, lane-local over-capacity case, finite-T mask/free-energy parity, gradients and higher derivative below. |

GFN1 default terms are active exactly once. AES2, D4SC and ALPB/GB are
explicitly unsupported by the private GFN1 evaluator; nondefault GFN1
classical configurations without parity evidence are not advertised as
supported. GFN2 default ES2/ES3/AES2/D4SC, Repulsion and classical D4 are
individually checked against independent serial setup/kernel references and
in the complete SCF result. Custom D4SC model extras and arbitrary custom
interaction/classical subclasses remain explicit mapped-path errors. The
legacy public path is unchanged.

GFN1 shared electric field and field-gradient terms are covered in both
source orders with response derivatives and required dipole/quadrupole
levels. This report does not infer GFN2 explicit-field acceptance from that
GFN1 evidence. Non-SCF/classical-only behavior remains its separate existing
route.

The GFN2 custom EEQ policy is one static cutoff per bucket. Default 25 Bohr,
smaller 0.8 Bohr and larger 4.0 Bohr values are compared to serial EEQ/D4;
the test verifies that the charges actually passed into D4 change with the
cutoff. Equal custom cutoffs are compatible; mixed cutoff buckets reject
before mapped execution. The cutoff is not differentiable.

## Trajectory, results and derivatives

The GFN1 and GFN2 paths reuse the shared fixed-trip solver and its opt-in
trace. GFN1 trace checks cover initial and intermediate states, raw maps,
mixed iterates, residual L2/max norms, active masks, iterations, damping,
Anderson history, Fermi status and final evaluation. State/mixer tolerances
are `2e-9`, Hamiltonian/density `2e-8`, and controller/residual `2e-10`.
GFN2 trace checks compare the production mapped trace with independent serial
Systems over 30 trips and include different lane activity. Ordinary
evaluation has tracing disabled. Traced and ordinary final results agree.

GFN2 Result parity compares total and SCF energies, classical labels/order
and atom-resolved terms, free energy, monopoles/dipoles/quadrupoles, density,
Hamiltonian, H0, overlap, dipole/quadrupole integrals, orbital energies,
occupations, potential, iterations, convergence, residual and both status
codes. Physical source ordering is decoded through canonical inverse maps.
Fermi validity is separate from AO validity; several high-energy dummy
levels remain unoccupied, including a positive fractional-occupation
frontier case. Charge/spin counts exclude padded atoms/orbitals.

GFN2 `jacrev`, JVP, `jacfwd`, mapped gradients, parameter gradients and a
mixed position/ES2-parameter derivative are compared against independent
serial calls. This package adds a directional Hessian-vector regression on
full GFN2 energy for H2/LiH, with two fixed SCF trips. It computes
`jvp(grad(E))` through `evaluate_padded_systems()` (including its internal
`vmap`) and compares each physical lane to a separate unpadded
`System.singlepoint()` reference. Across the five profiles the maximum
absolute HVP discrepancy is `5.551115123125783e-17`; maximum relative L2
discrepancy is `4.199545e-16` on Torch 2.6/2.8 and `3.464583e-16` on Torch
2.11/2.13/2.14. The asserted tolerance is `atol=rtol=2e-12`, justified by
the five-profile serial/mapped measurements and consistent with accepted
integral/H0 parity. The HVP test passes in all five CPU profiles. GFN1 second/
higher derivative and D3 transform acceptance remains in the accepted E6a
and E6bimpl.1b suites. No finite-difference third derivative is claimed for
the complete GFN2 SCF route.

## Test execution and failures

Every test command used `OMP_NUM_THREADS=1` and `pytest -n logical --dist
worksteal`. Official packages were present in all profiles:

| Python | Torch | `tad-mctc` | `tad-dftd3` | `tad-dftd4` | GFN1 mapped SCF/spectral | GFN2 mixed SCF |
|---|---|---|---|---|---:|---:|
| 3.10.22 | 2.6.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 51 passed | 41 passed |
| 3.11.15 | 2.8.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 51 passed | 41 passed |
| 3.12.10 | 2.11.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 51 passed | 41 passed |
| 3.12.10 | 2.13.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 51 passed | 41 passed |
| 3.14.8 | 2.14.1+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 51 passed | 41 passed |

The additional GFN2 HVP node passed in all five profiles. In the primary
profile, completed module-sized commands covered **431 unique acceptance
nodes: 425 passed and 6 failed**. The exact node list and manifest checksum
are in [`E6bimpl2c-acceptance-manifest.txt`](E6bimpl2c-acceptance-manifest.txt).
The 52-case C8b/schema/index/scatter group passed after the stale method
expectation was corrected. E6a0–E6a4 plus E1.3 D3-forward tests passed
73/73; E6bcompat-policy passed 22/22; E6bimpl.0/2a integral/canonical tests
passed 26/26.

The six failures are the same baseline failures recorded by E6bimpl.2b-F,
not new mixed-System failures:

1. `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[0]` — existing expectation of a trajectory status at zero outer trips.
2. `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[1]` — same status-policy discrepancy.
3. `test_functional_fock.py::test_fock_valid_trajectory_with_invalid_final_spectrum` — same zero-trip trajectory/final status distinction.
4. `test_functional_potential.py::test_potential_maxiter_zero_tracks_raw_and_final_fermi_status` — same zero-trip trajectory status expectation.
5. `test_functional_fock.py::test_fock_final_free_energy_matches_legacy[300K-atomic]` — absolute difference `2.634295391673461e-13` versus existing `2e-13` tolerance on Torch 2.14.1; it passes on earlier profiles.
6. `test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients` — max absolute position-gradient difference `2.5006982595776606e-11` versus `2e-13` tolerance.

All six were previously reproduced at the E6bimpl.2b-F starting commit
`f675c9d7`; the E6bimpl.2c start adds only tests and documentation relative
to that baseline and contains no changes to these functional solver/test
files. They remain assigned to the functional SCF status/Anderson baseline,
not this mixed-System implementation. No tolerance or numerical reference
was changed.

A single 601-node exploratory combined run did not reach a pytest summary;
it stopped after progress output near the end. It is not counted. The failure
was not attributed to an individual test or conclusively to worker shutdown.
The acceptance selection was therefore run as disjoint module-sized commands
with completed exit codes. The broad 601-node attempt and a C8b parameter-tree
module rerun remain incomplete; the predecessor's accepted C8b parameter
tree and broad regression evidence remain the reference for those unchanged
subsystems. CUDA and whole-System libcint mapping were not run and are not
claimed.

Black, isort, compileall and `git diff --check` pass. Ruff and mypy are not
installed in the primary environment. PyTorch emits its existing vmap
batching-fallback warnings for `matrix_exp`/`count_nonzero` and TorchScript
deprecation warnings in newer profiles.

## Compatibility classification and handoff

| Policy case | Classification | Owner / evidence |
|---|---|---|
| Same-method, supported GFN1/GFN2, PyTorch, built-in registered descriptors | **Mapped PyTorch** | `evaluate_padded_systems()`; the test modules listed in the manifest; serial independent reference and derivative comparisons. |
| Different method or incompatible static setup/cutoff | **Unsupported with explicit error or separate method-specific bucket** | Canonical signature and cutoff preflight; no GFN1/GFN2 shared bucket. |
| GFN0, unsupported optional setup, custom callbacks/model extras | **Unsupported with explicit error** in the private mapped path | Padded evaluator preflight. Existing legacy behavior remains untouched. |
| Public `Calculator(batch_mode=1/2)`, legacy FULL, low-level options, negative-temperature/Fermi routing, packed outputs and field validation | **Retained compatibility** | Existing Calculator/SCF path; `test_e6bcompat_policy.py` passed 22/22. Public dispatch was not changed. |
| Non-SCF/classical-only route | **Separate existing route** | Existing `solve_nonselfconsistent`/Calculator ownership; not routed through mapped FULL. |
| Whole-System libcint, if packed replacement is unavailable | **Retained compatibility composition still required** | E6bimpl owns an outer ordered loop of independent legacy single-System calls under the frozen policy; no such replacement composition has been implemented or accepted here. |
| Custom interaction/classical extension batch preservation before retirement | **Retained compatibility composition still required** where the legacy extension contract applies | E6bimpl policy composition owner; custom callbacks are not accepted by mapped core. Existing public packed path remains active pending E6broute. |

The private entry point for E6bcompat replacement checks is
`dxtb._src.scf.functional.padded.evaluate_padded_systems` (and the
lower-level `evaluate_padded_fixed_trip` for already prepared call data).
Its accepted domain is one method/static-configuration bucket of
independently set-up Systems, PyTorch integrals, compatible dtype/device,
supported registered components, and per-System geometry/charge/spin with
the established shared field contract. It returns a registered padded
Result plus source-ordered decoded physical Results and public SCFStatus.
Its derivative contract covers positions, the listed Model-owned parameters,
JVP/forward Jacobian, reverse gradients and selected higher derivatives.
Unsupported descriptors/methods fail explicitly. Independent serial
`System.singlepoint()` calls are the value and derivative references.

E6bcompat can begin differential validation of this accepted PyTorch
replacement and the still-active legacy public path. It cannot declare the
full frozen policy closed until E6bimpl supplies/tests the retained
compatibility composition or the architecture owner explicitly changes that
binding policy. Public route activation remains E6broute and is blocked on
E6bcompat, E7 batch safety and E6c1 W2/W3 performance gates.

## Decisions

**E6bimpl numerical implementation: COMPLETE.** Supported mixed GFN1/GFN2
PyTorch execution, default physical contributions, Result/status, padding,
charge/spin, derivatives and independent serial parity are covered. The six
remaining suite failures are matched baseline failures, and no unexplained
new numerical regression was found.

**Overall E6bimpl: OPEN.** The finite remaining implementation item is the
policy-assigned ordered legacy composition for cases that cannot enter the
mapped PyTorch bucket (notably retained libcint compatibility where packed
replacement is unavailable, and retained custom extension composition).
Required tests are the corresponding E6bcompat-policy serial-result,
ordering, error/option forwarding, cache freshness, force-CPU and custom
contribution controls. Do not activate this path in Calculator here.

**E6bcompat readiness: PARTIAL / handoff prepared.** The PyTorch replacement
boundary is ready for differential validation. Final compatibility
acceptance depends on the retained-composition item above and the policy
matrix tests. E6broute is not unblocked.
