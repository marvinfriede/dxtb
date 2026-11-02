# SCF S8 — fixed-occupation Fock-space SOSCF

## Scope and status

S8 adds an internal, eager-only `SOSCFMethod` for fixed integer occupations in
the Fock representation. It does not add a public mixer or strategy preset.
Active Fermi smearing and fractional occupations are rejected explicitly;
nonzero electronic temperature remains eligible only when the functional
occupation branch is still fixed/Aufbau. Fixed-trip/vmap execution of SOSCF is
not supported.

The method is a final stage after a first-order or compact quasi-Newton method.
The shared S2 controller still owns the global budget and physical
certification. SOSCF owns only orbital steps and their local diagnostics.

## Orbital coordinates and energy

The orbital model uses the current spectral coefficients and spin-channel
occupations from the paired Fock observation. Total spatial occupations are
`n = n_alpha + n_beta`. Active coordinates are all ordered pairs `(a, i)` with
`n_i > n_a`; equal-occupation gauge rotations are omitted. This includes all
`2 -> 1`, `2 -> 0`, and `1 -> 0` sectors in fixed open-shell cases.

Each packed coordinate creates an antisymmetric generator with
`K[a, i] = k_ai` and `K[i, a] = -k_ai`. The finite rotation is
`C(k) = C exp(K)`, and the density is `C(k) diag(n) C(k).T`. Therefore metric
orthogonality and electron count are preserved without a repair step.

`density_to_charges_and_core_energy()` is now shared by ordinary spectral
evaluation and the orbital model. `electronic_energy_from_density()` uses the
same core and interaction energy partition as the accepted spectral energy.
Tests verify exact density-energy parity on an ordinary spectral state.

For the controller convention, the fresh gradient Fock is the projected
`context.proposal_x`, paired with the density produced by the current input
Hamiltonian. The analytic local orbital gradient is

```text
g_ai = 2 (n_i - n_a) (C.T Ffresh C)[a, i]
```

The one-coordinate H2 finite-difference test locks the sign and factor. The
analytic gradient matches `torch.func.grad(E(k))` at the rotation origin for
H2/GFN1, H2O/GFN1, H2O/GFN2, NO2/GFN2, H2O+/GFN2, and stretched H2/GFN1. Its
norm has the analytically expected scaling to the S2 orthogonalized
commutator.

## Matrix-free orbital Hessian

`_SOSCFHessianOperator` applies the Hessian of the fixed-occupation energy by
JVP of its coordinate gradient. Every product differentiates through the
rotated density, Mulliken multipoles, physical xTB interaction response, and
Fock construction. It does not diagonalize or materialize an orbital Hessian.
Dense Hessians are test-only references. The checked production module has no
`torch.func.hessian`, `torch.func.jacfwd`, dense linear solve, or map call.

For H2/GFN1, H2O/GFN1, H2O/GFN2, and NO2/GFN2, local HVPs match dense test
Hessians and have symmetric bilinear products. A centered gradient difference
also agrees for H2O/GFN2. The HVP is the actual energy Hessian, not the
orbital-energy denominator. The denominator is used only as a positive
preconditioner. The production proposal uses the analytic origin gradient;
the independent AD gradient identity remains a test gate and is not recomputed
on every macroiteration.

## Inner solve and step acceptance

The inner system is `(H + lambda I) s = -g`, solved with restarted
preconditioned conjugate gradients. The diagonal preconditioner uses gaps from
`diag(C.T Ffresh C)` and `2 (n_i - n_a) (epsilon_a - epsilon_i)`. A dtype-aware
gap floor rejects differently occupied near-degeneracies; equal-occupation
degeneracies remain harmless gauge freedom.

The curvature rejection threshold is relative to
`||p|| ||(H + lambda I)p||`, so it scales down with near-root directions.
An absolute floor proportional to machine epsilon incorrectly rejected valid
small positive-curvature steps; a tiny-direction SPD regression guards this
case. Negative curvature increases the level shift geometrically and restarts
the inner solve. If retries fail, the method uses a recorded preconditioned
orbital-gradient descent fallback. No CDIIS fallback is hidden inside SOSCF.

The step is bounded by an internal trust/step-radius bound and backtracked from
the controller-provided damping; the radius is not adapted by a predicted vs
actual reduction ratio. Acceptance uses an Armijo energy condition with an
energy-scale and dtype-aware roundoff allowance. The stored predicted
reduction is the linear directional decrease `-g.dot(s)` used by Armijo, not
the quadratic Newton-model reduction. Each accepted trial must preserve the
spectral ordering of the differently occupied Fock blocks. The state records
gradient/step norms, inner residual, microiterations, HVP count, level shift,
line-search alpha/trials, negative curvature, and fallback use.

## Fock seed and physical certification

The seed uses the accepted trial density's freshly rebuilt physical Fock. In
the accepted rotated MO basis, it retains the full fresh-Fock blocks within
each equal-occupation subspace and zeros only cross-occupation blocks:

```text
Hseed = S C FMO_block C.T S
```

This is a gauge-preserving refinement of a diagonal seed. Equal-occupation
block rotations leave the density unchanged, while retaining those blocks
avoids a spurious same-occupation Fock fixed-point residual. At stationarity,
cross-occupation Fock blocks vanish, so the seed equals the fresh physical
Fock. Tests verify that a normal Fock map on the seed reproduces the accepted
density, charges, and mapped Fock; the certified-root test verifies the seed
and fresh Fock agree to the tight-root tolerance. Seed production does not
perform an eigensolve or an additional physical map.

The shared S2 certification remains authoritative and still checks both the
Fock fixed-point residual and physical stationarity. An orbital gradient alone
cannot declare convergence.

SOSCF performs no additional physical SCF map or generalized AO eigensolve in
its proposal. S8.1 adds small symmetric eigenspectra of equal-occupation MO
blocks for Aufbau-sector safety; these block checks are not an AO diagonalize
step.

## Strategy and implicit response

`SOSCFMethod.required_representation` is Fock, and its generic method
capability marks it eager-only. The shared controller rejects a fixed-trip
SOSCF strategy before numerical work. No concrete SOSCF branch was added to
the controller.

The focused H2O/GFN2 `CDIIS(Fock) -> SOSCF(Fock)` strategy certifies and then
passes the selected Fock root through the prepared implicit-root adapter. A
JVP and nested second derivative with respect to an explicit hcore direction
match a standalone CDIIS root. The implicit root response to one nuclear
coordinate, computed by reverse-mode autograd through integral construction,
also agrees between SOSCF and CDIIS roots. The prepared response record has no
method-history field; the response is attached to the exact raw Fock map and
certified root.

The H2O/GFN2 `Broyden(Potential) -> SOSCF(Fock)` strategy also certifies. In a
characterization run it used seven total physical maps (three compact-stage
maps and four Fock-stage maps); the switch itself did not perform a map.

Additional fixed-occupation root checks cover H2/GFN1, H2O/GFN1,
H2O+/GFN2, and stretched H2/GFN1. H2O/GFN1 and H2O+/GFN2 enter the SOSCF
stage and certify. The two H2 cases certify during CDIIS startup before the
stationarity switch is needed, so they validate the physical roots but do not
count as SOSCF macroiteration evidence. Separate near-root SOSCF runs start
from a controlled small orbital displacement of an independently converged
H2/GFN1 root; both equilibrium and stretched H2 certify through the SOSCF
controller. The C4H5NCS/GFN2 orbital dimension smoke has `nao=38` and
`nrot=360`; each float64 AO matrix occupies 11,552 bytes, while the method
stores only O(nrot) inner vectors and scalar diagnostics. A C4H5NCS macro-SCF
timing/convergence study was not run.

The open-shell NO2/GFN2/0 K `CDIIS(Fock) -> SOSCF(Fock)` run used an explicit
`1e-3` stationarity entry threshold and certified in 11 total physical maps
(five CDIIS-stage maps and six SOSCF-stage maps). The final fixed-point L2
residual was `5.8230e-8`, the orthogonalized stationarity L2 per square-root
orbital was `1.7918e-8`, and electron-count error was below `4e-15`. Its active
pair set includes all three required occupation sectors.
An open-shell NO2/GFN2 implicit hcore response JVP also matches standalone
CDIIS at the same certified root.

## Preflight and validation record

Before SOSCF edits, the S7 GFN0 H2O JVP/jacfwd smoke and open-shell O2 JVP smoke
both passed. Forward/reverse agreement was at float64 roundoff (`3.9e-16` for
H2O and `1.3e-16` for O2); no GFN0 production change was needed. Two S7 stale
comments were corrected: custom interaction subclasses are explicitly
rejected, and the packed FULL compatibility class is not described as the
implicit engine.

The focused SOSCF validation passes 43 tests on Python 3.14/PyTorch 2.14.1 and
43 tests on Python 3.11/PyTorch 2.8.0. A combined S2-S7 regression selection
passed 659 tests; the only failure was the unchanged inherited Anderson
gradient-parity result `2.5007065862503453e-11` against `2e-13`. The slow
implicit-root and spectral cases in that selection completed. The focused
SOSCF validation covers rotation/energy/gradient/HVP identities, open-shell
sectors, Fock-seed parity, equal-occupation gauge invariance, fixed-trip
rejection, active-smearing rejection, low-temperature fixed occupations,
immutable state freezing, synthetic inner solves, float32 finite-step
behavior, stationary reseeding from stale input eigenvalues, the additional
fixed-occupation root checks, near-root equilibrium/stretched H2 SOSCF, and
CDIIS/Broyden stage integration, and method-independent H2O/NO2 implicit
responses. The NO2
strategy uses the explicit experimental switch stationarity threshold
`1e-3`; it reaches an S2-certified zero-temperature open-shell root. The
final focused rerun after the finite-curvature guard is recorded in the
package handoff.

No public default/configuration changes, EDIIS/ADIIS, finite-temperature
SOSCF, S9, S10, S11, or transform support for SOSCF were started.

## S8.1 precision and occupation-sector hardening

The post-S8 review reproduced an accuracy ceiling in the stationary reseed
shortcut. On the S8 baseline, the threshold was
`sqrt(eps) * max(abs(Ffresh), 1)`, which is `1.4901e-8` for the tested H2O/GFN2
Fock scale. A `1e-9` rotation from an independently converged CDIIS root had
orbital-gradient norm `3.0065552e-9` and fixed-point residual
`1.0055848e-9`, but produced an SOSCF proposal with zero HVPs. Repeating the
old proposal three times left the fixed-point residual at `1.00558e-9`, with
zero HVPs and only `1.4e-15` to `3.0e-15` changes in the proposed seed.

The shortcut now uses the separate `stationary_atol_factor` setting, default
32, and

```text
floor = factor * eps * max(abs(Ffresh)) * sqrt(max(nrot, 1))
```

with the Fock scale clamped to at least one. At H2O/GFN2, `nrot=8` and this
floor is `2.0097e-14`. The controlled `1e-9` displacement now performs HVPs,
reduces the physical residual, and certifies at `x_atol=x_atol_max=1e-10`.
Float64 and float32 tests characterize factors 10, 32, and 100. The genuine
zero-gradient/stale-input-eigenvalue test tightens its CDIIS root to `1e-12`;
the gradient remains below the epsilon-scale floor and still reseeds with
zero HVPs.

A second review finding showed that ordering the diagonal entries of
`C.T @ Ffresh @ C` is gauge-dependent and insufficient when the seed retains
full equal-occupation blocks. The adversarial `[2, 2, 0]` block
`[[-1, 2], [2, -1]]` has diagonal entries below the virtual value zero, but
its eigenvalues `[-3, 1]` interleave the virtual sector. The new
`_occupation_sector_spectra()` and `_occupation_sector_order_valid()` helpers
compute each active same-occupation block spectrum and require the maximum
eigenvalue of every higher-occupation sector, plus the gap floor, to remain
below the minimum eigenvalue of the next lower sector. The gap floor is scaled
from the AO Fock magnitude and dtype, so same-sector MO gauge rotations do not
change the threshold or verdict. This check runs at method entry, for each
line-search trial, and defensively before seed construction. Unsafe trials
backtrack; unsafe seeds raise. No diagonal-only seed was introduced.

The open-shell NO2/GFN2 seed test rotates all `2->1`, `2->0`, and `1->0`
coordinates by `1e-4`. Its sector gaps are positive and the next Fock map
reproduces the accepted density, charges, and mapped Fock. A separate
same-sector gauge test confirms unchanged block spectra and seeded physical
map. A synthetic crossing trial that passes Armijo is rejected at the full
step; the shorter `alpha=0.125` trial is accepted after the block spectra
separate. The adversarial `[2,2,0]` fixture also diagonalizes the unsafe
block-preserving seed: its occupied block eigenvalues interleave the virtual
level, and Aufbau filling the global sorted spectrum changes the represented
density. This demonstrates the physical failure mode behind the block check.

### Tight-root and larger-system characterization

The controlled H2O/GFN2 tight strategy uses three CDIIS maps followed by six
SOSCF maps and certifies at `1e-10`. It takes 17 HVPs/microiterations, has a
final fixed-point L2/max of `2.2622e-11` / `8.2266e-12`, stationarity
`2.1358e-11`, orbital-gradient norm `7.3986e-11`, and energy
`-5.104158799409648`. The SOSCF gradient trajectory decreases from
`1.1214e-3` through `6.9874e-5`, `4.4193e-6`, `2.8127e-7`, `1.7970e-8`,
and `1.1515e-9`.

The open-shell NO2/GFN2/0 K strategy uses five CDIIS maps and eleven SOSCF
maps and certifies at `1e-9`. It has 63 HVPs/microiterations, fixed-point
L2/max `6.8309e-10` / `2.6521e-10`, stationarity `2.3547e-10`, electron-count
error `7.1e-15`, final orbital-gradient norm `1.1536e-9`, and energy
`-10.953075803272055`. The final occupation-sector spectral gaps are
`0.1460373` for 2/1 and `0.0482472` for 1/0.

The NO2 implicit geometry check perturbs the x coordinate of atom 0 and
compares the sum of the prepared implicit Fock root from tight CDIIS and
CDIIS-to-SOSCF roots. First derivatives are `-0.2725750100164` and
`-0.2725750097151` (difference `3.0133e-10`). Second derivatives are
`-1.5599331805206` and `-1.5599331798598` (difference `6.6083e-10`). This is
reverse-over-reverse for the geometry-built PyTorch-integral path; the existing
H2O geometry and hcore response controls remain in place.

The C4H5NCS/GFN2/0 K CDIIS-to-SOSCF run exercises the full `nrot=360`
production path (`nao=38`) and certifies in 17 physical maps: ten CDIIS and
seven SOSCF. It performs 60 HVPs/microiterations, seven line-search trials,
no negative-curvature retries, and no gradient fallback. Wall time is 2.46 s
and the process peak is 1757.1 MiB on the recorded Python 3.14/PyTorch 2.14.1
CPU environment. A float64 AO matrix is 11,552 bytes. The persistent method
state has only O(nrot) vectors/scalars and no `nrot x nrot` Hessian. The final
fixed-point L2/max is `7.9726e-8` / `2.8339e-8`, stationarity is `2.8367e-8`,
and energy is `-19.694059708019115`.

The S8.1 characterization script and raw JSON records are retained under
`benchmarks/soscf_s81_characterize.py` and
`benchmarks/results/soscf-s81/`; their SHA-256 hashes are recorded in the
package handoff. An independent read-only review found no blocker or major
issue; its minor request for explicit density reassignment evidence is covered
by the synthetic regression above.

### S8.1 regression record

The final focused SOSCF file passed **55 tests** on Python 3.14/PyTorch 2.14.1
and **55 tests** on Python 3.11/PyTorch 2.8.0, including the new synthetic
seed-density reassignment assertion. A broader combined SCF/calculator
selection initially completed with **630 passed and 5 failed**. Four failures
were stale executable assertions in `test_e6a0_system_vmap.py` for pre-S4/S7
routing behavior, not accepted baseline failures. S8.1a replaces them with
tests for the current explicit validation, retired experimental mode, and
native Broyden contracts. The one remaining numerical failure is the accepted
Anderson parity discrepancy `2.5007065862503453e-11` versus `2e-13`.

The production proposal no longer computes a second AD gradient on every
macroiteration. `torch.func.grad(energy_at)` remains in the HVP operator and
in the independent gradient/Hessian test references. This avoids redundant
reverse-mode work while preserving the analytic-gradient identity gate.

### S8.1a review gate — public routing tests

The four stale E6a.0 expectations were reproduced on the S8.1 starting tree
as the following exact failing nodes:

- `test_system_legacy_fallback_constructs_only_eager_guess`
- `test_system_static_legacy_routes_skip_transform_guess[negative-fermi-maxiter]`
- `test_system_static_legacy_routes_skip_transform_guess[non-full-mode]`
- `test_system_strict_mixer_error_precedes_initial_guess`

The tests now require negative electronic temperature and negative Fermi
iteration count to raise their explicit `ValueError`s before a guess or
functional solve is entered. Experimental single-shot raises the documented
retirement `NotImplementedError` before guessing. Strict Broyden is exercised
through a real public `System.singlepoint()` call, with a probe confirming
`BroydenMethod.propose()` is reached. No production SCF code changed.

The complete E6a.0 file passes **14 tests** on Python 3.14/PyTorch 2.14.1 and
**14 tests** on Python 3.11/PyTorch 2.8.0. Focused S8.1 checks for the tight
H2O stationary-floor regression, synthetic block-spectrum counterexample,
and NO2 open-shell seed pass **3 tests** on Python 3.14. The exact command for
the earlier 630-test combined selection was not preserved in the package
record, so that exact selection is not claimed as rerun here; the prior
630/5 result is historical, and the completed E6a.0 and focused SOSCF results
above are the S8.1a evidence.

An expanded reproducible selection of all `test_functional_*.py` SCF modules
plus `test_e6a0_system_vmap.py` collected 602 tests and completed with 596
passed and 6 failed. Besides the accepted Anderson gradient discrepancy, the
additional failures were three `test_functional_temperature_cutoff.py`
assertions and `test_functional_map_has_no_legacy_stateful_hooks`, whose
directory-wide `.tolist()` scan also sees the eager-only SOSCF implementation.
This expanded command was not the historical 630-test selection and does not
replace its acceptance result. The four rewritten E6a.0 routing tests passed
in both primary Python environments.

### S8.1b review gate — reproducible SCF regression selection

At `72ccc23739b85fd1422d6434c06c6fddd902ccf9`, the reproducible selection
`test/test_scf/test_functional_*.py` plus `test/test_calculator/test_e6a0_system_vmap.py`
collected 602 nodes and completed with 596 passed and six failed. The exact
failures were the accepted Anderson gradient discrepancy, the directory-wide
functional architecture scan, and all four nodes in
`test_functional_temperature_cutoff.py`:

- `test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`
- `test_functional_map.py::test_functional_map_has_no_legacy_stateful_hooks`
- `test_functional_temperature_cutoff.py::test_positive_subcutoff_temperature_preserves_legacy_free_energy`
- `test_functional_temperature_cutoff.py::test_subcutoff_fractional_charge_gradient_matches_legacy`
- `test_functional_temperature_cutoff.py::test_temperature_cutoff_boundaries_preserve_legacy_split`
- `test_functional_temperature_cutoff.py::test_outer_convergence_and_fermi_status_remain_separate`

The entire temperature file was run unmodified with Python 3.14/PyTorch 2.14.1
at S7 (`17eba1a`), S8 (`2167f19`), and current (`72ccc23`). All four tests
failed identically at all three revisions. The first three compared against
the retired single-System SCF oracle or required bitwise legacy free-energy
equality. Their replacements use
`filling.get_fermi_entropy_low_temperature(occupation, kt)` as an independent
occupation-space physical reference. They retain equal atom partitioning,
the fractional-charge entropy derivative, exact-zero settings/zero entropy,
positive sub-cutoff non-smearing, and cutoff-equality/above-cutoff smearing.
The temperature test file no longer calls the legacy engine for its reference.

The fourth failure was not an entropy-oracle issue. Its raw fixed-point
component passes while Fermi validity fails; S2 certification intentionally
requires both, as well as stationarity and electron-count checks. The test now
asserts those component verdicts and the resulting non-certified state rather
than claiming the full physical result is converged with failed Fermi status.

The architecture guard is split. All functional modules, including SOSCF,
remain checked for legacy mutable-owner hooks. The host tensor-extraction scan
continues to cover raw maps, certification, solver, and transform-capable
method paths. It excludes `implicit.py` for its custom transform boundary and
the eager-only SOSCF orbital path (`soscf.py` plus its `orbital.py` sector-pair
helper). `SOSCFMethod.supports_fixed_trip is False`, and the existing test
confirms fixed-trip SOSCF is rejected before a physical map. Removing host
control from this eager-only method path belongs to S12.

After these test-only changes the same glob selection collected 603 nodes:
602 passed and the only failure was the inherited Anderson discrepancy
`2.5007065862503453e-11` versus `2e-13`. The node count increased by one
because the original single architecture test was split into the two required
independent guards. The E6a.0 routing tests were included and pass. No
production SOSCF code changed, and W2c10 was not synchronized.
