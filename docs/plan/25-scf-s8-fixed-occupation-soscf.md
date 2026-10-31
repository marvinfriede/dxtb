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
preconditioner.

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

The step is bounded by an internal trust radius and backtracked from the
controller-provided damping. Acceptance uses an Armijo energy condition with
an energy-scale and dtype-aware roundoff allowance. Each accepted trial must
preserve the ordering of differently occupied Fock orbital energies. The
state records gradient/step norms, inner residual, microiterations, HVP count,
level shift, line-search alpha/trials, negative curvature, and fallback use.

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
