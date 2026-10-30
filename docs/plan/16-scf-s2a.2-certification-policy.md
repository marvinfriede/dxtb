# SCF S2a.2 certification policy closure

## Scope

S2a.2 freezes diagnostic metrics and proposed internal profiles. It does not
make certification authoritative in a controller, alter a stopping decision,
change public status, or change implicit-root acceptance. No map or eigensolve
was added. The certification core performs fixed-point algebra, fresh Fock
assembly, stationarity matrix products, an independent density trace, and
validity checks. Energy and entropy observables remain optional diagnostics.

## Repository-wide lazy-loader equality audit

The audit searched `src`, `test`, `benchmarks`, and `examples` for every
`GFN1_XTB`/`GFN2_XTB` reference, then inspected equality, inequality,
conditional, membership, tuple/list/set/dictionary dispatch, and method-helper
contexts. The loader objects are distinct lazy parameter loaders whose
`__eq__` is not a safe model discriminator. `labels.GFN1_XTB` and
`labels.GFN2_XTB` are resolved integer identifiers and ordinary equality is
correct for those values.

Three test helpers had unsafe loader equality and could silently label GFN2
fixtures as GFN1:

| Location | Classification | Correction |
| --- | --- | --- |
| `test/test_calculator/test_c6_descriptor_tree_audit.py` | test helper | identity dispatch plus explicit unsupported-loader error |
| `test/test_calculator/test_b3b_layout_contract.py` | test helper | identity dispatch for the known GFN1/GFN2 loader pair |
| `test/test_calculator/test_c8b_schema_preflight.py` | test helper | identity dispatch, explicit loader annotation, and both-model regression |

No production misdispatch was found. Production comparisons in CLI, padding,
SCF, and calculator code use resolved integer labels. Other inspected loader
references are direct arguments, tuple parameterization, or string-keyed
dispatch (`"gfn1"` / `"gfn2"`) and do not compare lazy loaders. The example
`Param.from_file(...) == GFN1_XTB` is data-object equality, not method dispatch.
The characterization helper now explicitly accepts `LazyLoaderParam` and
asserts config label, H0 method, and resolved parameter model before emitting a
row.

## Stationarity scalar

The raw nonorthogonal diagnostic remains

```text
R_AO = F[P] P S - S P F[P]
```

with the fresh Hamiltonian rebuilt from density-derived charges. The
orthogonalized diagnostic is

```text
R_orth = X.T R_AO X,  X = S^(-1/2)
```

and is invariant under the tested general AO basis rescaling in Frobenius
norm. Its unscaled Frobenius norm is extensive under direct sums. The selected
primary candidate is therefore

```text
stationarity_orthogonalized_l2_per_sqrt_orbital
    = ||R_orth||_F / sqrt(n_active_orbitals)
```

where `n_active_orbitals` comes from the structural orbital mask. Raw AO L2,
raw AO max, orthogonalized L2, and orthogonalized max remain diagnostics.
Orthogonalized max is not the primary invariant because its value changes under
general AO transformations.

A synthetic nonzero 3-by-3 residual block was repeated 1, 2, 4, and 8 times.
For k copies, orthogonalized L2 followed `sqrt(k) * base`, max stayed constant,
and L2 divided by `sqrt(active_orbitals)` stayed constant. L2 divided by
active orbital count and by `sqrt(active_pair_count = n_active^2)` both fell
as `1/sqrt(k)`. The normalized scalar passed JVP and jacfwd checks.

The real-system size sample at tight roots includes 6 active AOs (GFN2/H2O),
12 (GFN2/NO2), 20 (GFN1/NH3-dimer), and 38 (GFN2/C4H5NCS). The 38-AO system
shows no abnormal growth in the selected normalized metric. This is useful
coverage, not a claim about arbitrarily large systems. Earlier basis-rescaling
evidence in the S2a.1 record remains: raw AO L2 changed from 0.79980 to 1.04922
while orthogonalized L2 stayed 0.475056.

| Model / system | Active AO | Root tol | Fixed L2 / max | Raw AO L2 / max | Orthogonalized L2 / max | Selected scalar |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GFN2/H2O | 6 | 1e-10 | 9.04e-11 / 5.15e-11 | 5.59e-12 / 2.29e-12 | 6.87e-12 / 2.88e-12 | 2.81e-12 |
| GFN2/NO2, 300 K | 12 | 1e-8 | 9.30e-9 / 4.41e-9 | 6.88e-10 / 2.26e-10 | 8.82e-10 / 2.68e-10 | 2.55e-10 |
| GFN1/NH3-dimer | 20 | 1e-10 | 9.03e-11 / 3.05e-11 | 1.26e-13 / 3.76e-14 | 1.53e-13 / 4.45e-14 | 3.42e-14 |
| GFN2/C4H5NCS | 38 | 1e-10 | 8.94e-11 / 7.65e-11 | 6.10e-12 / 1.16e-12 | 7.36e-12 / 1.38e-12 | 1.19e-12 |

## Tight GFN2/H2O representations

All rows are independently constructed, verified GFN2/H2O, float64, Aufbau,
charge 0, spin 0, with the same linear solver setup. Values are
`L2 / max`; selected stationarity is the orthogonalized L2 divided by
`sqrt(active AOs)`.

| Representation | Root tol | Fixed-point | AO stationarity | Orthogonalized stationarity | Selected scalar | Count error | Energy (Eh) | Maps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| charge | 1e-10 | 9.04e-11 / 5.15e-11 | 5.59e-12 / 2.29e-12 | 6.87e-12 / 2.88e-12 | 2.81e-12 | 0 | -5.104158799409648 | 101 |
| charge | 1e-12 | 8.25e-13 / 4.75e-13 | 5.10e-14 / 2.11e-14 | 6.29e-14 / 2.66e-14 | 2.57e-14 | 0 | -5.104158799409649 | 125 |
| potential | 1e-10 | 9.76e-11 / 8.18e-11 | 6.65e-11 / 2.72e-11 | 8.18e-11 / 3.42e-11 | 3.34e-11 | 0 | -5.104158799409648 | 88 |
| potential | 1e-12 | 8.80e-13 / 7.45e-13 | 6.11e-13 / 2.50e-13 | 7.52e-13 / 3.15e-13 | 3.07e-13 | 1.78e-15 | -5.104158799409647 | 112 |
| Fock | 1e-10 | 9.13e-11 / 4.20e-11 | 5.47e-11 / 2.24e-11 | 6.73e-11 / 2.81e-11 | 2.75e-11 | 0 | -5.104158799409647 | 89 |
| Fock | 1e-12 | 9.98e-13 / 4.65e-13 | 6.11e-13 / 2.50e-13 | 7.51e-13 / 3.15e-13 | 3.07e-13 | 0 | -5.104158799409648 | 112 |

The 1e-10-to-1e-12 selected scalar ratios are 0.0091 (charge), 0.0092
(potential), and 0.0112 (Fock), so no representation stalls as the root is
tightened. At 1e-10, stationarity/fixed-point L2 ratios are about 0.076,
0.84, and 0.73 respectively; raw root and physical stationarity remain
distinct gates.

All representations converge toward lower stationarity as the raw root is
tightened. The ratio is representation-dependent; there is no fixed conversion
factor. Energy remains consistent to displayed precision. This supports
representation-specific fixed-point thresholds and one common physical
stationarity scalar.

## Charged/open-shell and NO2 reference evidence

The verified GFN2 water cation doublet is charge +1, spin 1, with
`nel=[4,3]`. At root tolerances 1e-8 and 1e-10, respectively, the selected
stationarity scalar is 1.64e-10 and 1.86e-12; independent count error is
8.88e-16 in both cases. Both spin-channel populations are nonzero, and the
energy is -4.431225111368876 / -4.431225111368877 Eh.

For the higher-order GFN2/NO2 doublet at 300 K, the prior verified sweep gives
selected stationarity about 2.54e-10 at root tolerance 1e-8 and 2.54e-14 at
1e-12. Raw/orthogonalized unnormalized L2 values at the tight root are
approximately 6.87e-14 / 8.79e-14; independent count error remains at
roundoff. This preserves the explicit higher-order reference target without
using autodiff mode to select a tolerance.

## Fractional float32 electron count

The production matrix-Fermi path was characterized with GFN2/NO2, charge 0,
spin 1, both spin-channel electron targets nonzero, in float32 and float64.
Occupations are reported only when strictly between `1e-3` and `1-1e-3`.

| T | dtype | fractional occupation range | count | target N | Tr(P S) error | effective root tolerance | Fermi residual (alpha, beta) | Fermi/count validity |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1000 K | f32 | none in selected interval | 0 | 17 | -3.81e-6 | 5e-5 | 4.66e-9, 4.71e-12 | valid / independent error below floor |
| 2000 K | f32 | 0.0212–0.9788 | 2 | 17 | -3.81e-6 | 5e-5 | 9.13e-8, 2.18e-6 | valid / independent error below floor |
| 3000 K | f32 | 0.0697–0.9303 | 2 | 17 | -3.81e-6 | 5e-5 | 2.24e-7, 1.25e-6 | valid / independent error below floor |
| 5000 K | f32 | 0.0124–0.9955 | 5 | 17 | -3.81e-6 | 5e-5 | 3.32e-5, 6.36e-5 | valid / independent error below floor |
| 2000 K | f64 | 0.0212–0.9788 | 2 | 17 | -3.55e-15 | 1e-8 | 2.64e-12, 2.46e-16 | valid |
| 3000 K | f64 | 0.0697–0.9303 | 2 | 17 | -7.11e-15 | 1e-8 | 2.36e-18, 7.56e-13 | valid |

GFN2/H2O in float32 is a second finite-temperature control with saturated
occupations and independent error 0 at the printed precision. The f32 NO2
error is below the existing effective root floor and persists across these
fractional occupations; the float64 controls are at roundoff. This evidence
supports `electron_count_tolerance = effective fixed-point L2 tolerance` as
the simplest S2b rule, alongside Fermi's own status when smearing is active.
It gives 1e-4 for the standard float64 profile and 5e-5 for float32 under the
existing resolver; tight/reference float64 profiles use 1e-10/1e-12. A
separate certification test perturbs a finite, Fermi-valid density and verifies
that the independent count gate rejects it.

A float32 synthetic density/overlap block was replicated 1, 2, 4, and 8 times.
The trace error was 6.22e-9, 1.24e-8, 2.49e-8, and 4.97e-8 respectively: it
scaled linearly for this constructed sum. This is below the 5e-5 float32 root
floor at the tested scale; large padded batches still need monitoring.

The synthetic direct-sum test validates stationarity normalization, not a
general growth law for float32 summation error. The molecular data do not
justify an extra multiplier proportional to AO count or electron count.
S2b should start with the effective-root rule and retain the independent count
as a separate gate; revisit only if larger padded/batched evidence exceeds the
effective floor.

## Internal profiles frozen for S2b

These are internal policy values, not public configuration. The raw root pair
is tested per representation; the same numeric defaults are retained because
`G(x)-x` units differ by representation and S2b must preserve current route
settings unless a caller explicitly selects a tighter profile.

| Profile | Representation | Fixed-point L2 / max | Selected stationarity scalar | Independent count threshold |
| --- | --- | ---: | ---: | --- |
| standard | charge/potential/Fock | existing ConfigSCF values: 1e-4 / 1e-5 | 1e-5 | effective fixed-point L2 tolerance |
| tight | charge/potential/Fock | 1e-10 / 1e-10 | 1e-9 | effective fixed-point L2 tolerance |
| higher-order/reference | charge/potential/Fock | 1e-12 / 1e-12 | 1e-12 | effective fixed-point L2 tolerance |

For float32, each requested root tolerance resolves through the existing
`_resolve_tolerance`; e.g. requested 1e-12 becomes 5e-5. The explicit
float32 stationarity floor is 1e-5 for all three profiles; this is above the
measured 1.48e-6 worst selected value and does not claim float64-level physical
accuracy. S2b diagnostics must record both requested and effective values. The profiles are explicit; there
is no `requires_grad`, grad-mode, or transform detection. Finite and Fermi
validity are mandatory in every profile. The standard profile accepted all four explicitly characterized standard roots (GFN1/NH3-dimer, GFN2/H2O, GFN2/NO2, and GFN2/C4H5NCS): 4 passed, 0 stationarity-only, 0 count-only, and 0 validity failures. Their selected metrics range from 1.75e-7 to 2.41e-6. Tight and reference roots are separated from the standard raw-root policy by stricter fixed-point thresholds. Historical higher-order evidence
continues to support the explicit 1e-12 reference profile for NO2/GFN2.

## Core timing

CPU medians used two warmups and seven samples on Python 3.14/PyTorch 2.14.1.
Core excludes energy and entropy; each operation consumes an already paired
physical state. “Free-energy correction” is the entropy/free-energy component.

| Case | Active AO | Raw map | Core certification | Electronic energy | Free-energy correction | Core/map |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2/GFN1 | 4 | 4.38 ms | 1.16 ms | 0.34 ms | 0.003 ms | 0.27 |
| H2O/GFN2 | 6 | 6.77 ms | 3.51 ms | 2.07 ms | 0.003 ms | 0.52 |
| NO2/GFN2, 300 K | 12 | 71.80 ms | 3.62 ms | 2.06 ms | 32.05 ms | 0.05 |
| C4H5NCS/GFN2 | 38 | 10.02 ms | 3.73 ms | 2.08 ms | 0.003 ms | 0.37 |

The entropy correction dominates the finite-temperature full diagnostic. The
S2b gate should use core metrics and reuse finalized energy/free-energy values
for optional progress reporting.

## S2b integration design

### Eager controller

1. Evaluate the authoritative raw map and form the paired state
   `(root_x, G(root_x))`.
2. Apply the existing cheap representation-projected L2/max candidate test.
3. If it is not a candidate, continue without certification.
4. On a candidate, evaluate core certification on that exact paired state.
5. Accept only if fixed-point, selected stationarity, independent count,
   finite, and Fermi checks pass. If the state is finite and Fermi-valid but
   stationarity/count fail, leave it active and continue; do not promote
   `mapped_x` to the root.
6. A non-finite or invalid-Fermi active candidate fails closed under explicit
   S2b policy. Existing final public-spectrum status remains a separate
   finalization check.

The candidate branch is host-side eager control, so expensive core work can be
skipped when raw residual is not a candidate.

### Fixed-trip controller

The fixed-trip path cannot branch on tensor values in Python. The preferred
initial S2b integration is to evaluate core diagnostics on every active loop
state, combine the tensor verdict with the raw candidate mask, and continue
active lanes when a raw candidate is finite/Fermi-valid but stationarity/count
are not yet acceptable. A candidate mask cannot skip the certification
matrix work: `torch.where` selects results after evaluation. The implementation
must report the full per-active-iteration added cost. If this
per-iteration cost proves unacceptable, an alternative is final-only
certification after the fixed raw solve, with `converged=False` on certification
failure; that cannot use remaining budget to refine and is therefore a
behavioral tradeoff, not equivalent continuation.

Certification is neither an SCF map nor an iteration: it rebuilds `F[P]` but
does not diagonalize or evaluate `G`. In both controllers the certified state
is the exact accepted `SCFPhysicalState(root_x, G(root_x))`. Potential and Fock
routes still retain their existing extra final potential/Fock spectrum for
public result finalization; that final-spectrum Fermi validity is separate from
root certification.

Proposed internal failure categories for S2b are raw root not converged,
stationarity failed, independent electron count failed, Fermi subsolver failed,
non-finite numerical state, and final public spectrum failed. The public
`SCFStatus` mapping and `force_convergence` behavior remain S2b work.

## Validation and boundaries

The direct-sum primitive, certification suite, loader helper regressions,
characterization harness, and policy tests are run in this package. The
combined loader selection exposed one unrelated existing expectation mismatch
in `test_scf_orbital_mask_and_conditioning_keep_indexhelper_padding`, which
also reproduces in isolation. It compares `SCFInputs.inverse_cholesky` with
`storch.inv_cholesky_factor`, while the accepted S2a.1 implementation uses the
symmetric inverse square root for this field. The 189/289 element mismatch is
in that B3b test expectation, outside the loader fixes and diagnostic changes;
no production adjustment was made in S2a.2.
No production stopping/status policy was changed. The general-refactor
worktree was not modified.
