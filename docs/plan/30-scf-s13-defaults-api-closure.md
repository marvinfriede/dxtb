# SCF S13 — defaults and public strategy API

## Decision status

This document records the final S13 policy. The final local candidate is
`5b622a58c394839a2284f18fea34804805995615`; remote publication and the
post-publication verification remain the release gate before marking the full
redesign complete.

The final-head one-thread benchmark matrix in
`benchmarks/results/scf-s13/defaults-py314-torch2141.json` contains 45
method/case runs from source SHA
`5b622a58c394839a2284f18fea34804805995615` with a clean worktree: 15 GFN1/GFN2,
charge, temperature, and poor-start cases, each run with Anderson, Broyden,
and CDIIS. All three methods certified all 15 cases with the public 100-map
budget. Anderson/Potential used a median of 9 maps and nearest-rank p90 of 12,
compared with Broyden/Potential at 15/43 and CDIIS/Fock at 13/17. Median and
nearest-rank p90 wall times were 0.075/0.615 s for Anderson, 0.138/1.013 s for
Broyden, and 0.151/0.905 s for CDIIS. The benchmark records runtime, dtype,
thread count, tolerances, certification, physical maps, memory, and raw rows.

This matrix supports Anderson/Potential as the default on the tested set. It
does not establish universal superiority. Broyden remains a supported native
method, and CDIIS remains the explicit Fock-space choice.

## Public configuration

The stable namespace is `dxtb.scf`. It exports `SCFStrategy`, `SCFStage`,
`SCFSwitchPolicy`, `SCFRepresentation`, the Linear, Anderson, Broyden, CDIIS,
and SOSCF methods, their supported settings, and the differentiation,
convergence-profile, and response-setting types.

```python
from dxtb.config import Config

default = Config.create()
robust = Config.create(strategy="broyden", differentiation="implicit")
fock_diis = Config.create(strategy="cdiis", differentiation="implicit")
```

An explicit fixed-occupation SOSCF strategy is eager and uses implicit
differentiation:

```python
from dxtb.config import Config

accurate = Config.create(
    strategy="cdiis-soscf",
    differentiation="implicit",
    convergence="higher_order",
    fermi_etemp=0.0,
)
```

The default preset is `anderson`: Anderson in Potential coordinates, five
generations, a 0.1 startup damping value, soft startup, and a 0.01 diagonal
offset. The common iteration budget remains 100. The default differentiation
is `unrolled`, which preserves fixed-trip and same-shape `vmap` use for the
single-stage default.

`ConfigSCF.create()` and `Config.create()` accept the named presets. Advanced
users can construct `SCFStrategy` and `SCFStage` from `dxtb.scf`; each stage
owns its method and representation. There is no top-level representation
setting in the final strategy model. Temporary `mixer`, `scp_mode`, and
`scf_mode` parser arguments are converted immediately and are not stored on
`ConfigSCF`; deprecated Anderson damping parser arguments are converted into
`AndersonMethodSettings`. Ambiguous conflicts are rejected. `f_atol` is
retired because physical fixed-point and stationarity certification own SCF
convergence.

```python
from dxtb.config import Config
from dxtb.scf import (
    BroydenMethod,
    CDIISMethod,
    SCFRepresentation,
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)

strategy = SCFStrategy(
    (
        SCFStage(
            BroydenMethod(),
            representation=SCFRepresentation.POTENTIAL,
            switch_policy=SCFSwitchPolicy(fixed_point_l2=1.0e-2),
        ),
        SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),
    )
)
custom = Config.create(strategy=strategy, differentiation="implicit")
```

The conceptual migration is:

| Retired input | Strategy API |
|---|---|
| `mixer="simple"` | `strategy="linear"` |
| `mixer="anderson"` | `strategy="anderson"` |
| `mixer="broyden"` | `strategy="broyden"` |
| global `scp_mode=...` | `SCFStage(..., representation=...)` |
| `scf_mode="full"` | `differentiation="unrolled"` |
| `scf_mode="implicit"` | `differentiation="implicit"` |

## Presets and numerical policies

| Preset | Stages | Representation | Scope |
|---|---|---|---|
| `linear` | Linear | Potential | Public reference method |
| `anderson` | Anderson | Potential | Default, transform-compatible |
| `broyden` | Broyden | Potential | Public native method |
| `cdiis` | CDIIS | Fock | Public orbital-stationarity method |
| `broyden-cdiis` | Broyden → CDIIS | Potential → Fock | Eager/implicit only |
| `anderson-cdiis` | Anderson → CDIIS | Potential → Fock | Eager/implicit only |
| `cdiis-soscf` | CDIIS → SOSCF | Fock | Eager/implicit, fixed occupations |
| `broyden-cdiis-soscf` | Broyden → CDIIS → SOSCF | Potential → Fock | Eager/implicit, fixed occupations |

The current public settings retain Anderson generations 5, Broyden rank 8,
and CDIIS history 8. The `broyden-cdiis` switch uses physical fixed-point L2
of `1e-2`; the SOSCF-entry policy uses stationarity `1e-3`. These are frozen
preset values, not automatic choices based on temperature, gradients, or
tensor values. There is no implicit fallback or method substitution.

SOSCF remains genuine Fock-space orbital second order. It is limited to fixed
Aufbau occupations, rejects active Fermi smearing, and is eager-only. It is
opt-in; automatic SOSCF is disabled. EDIIS/ADIIS and finite-temperature
Newton–Krylov remain research prototypes and are not imported by production
SCF code.

## Differentiation, response, and accuracy

The public differentiation values are `unrolled` and `implicit`. `unrolled`
is the default for transform-compatible single-stage methods. `implicit` is
recommended for eager high-order derivatives, dynamic strategies, and SOSCF;
its response is attached to the certified physical root and is independent of
solver history.

The implicit response coordinate defaults to Potential, with BiCGSTAB
settings `rtol=1e-10`, `atol=1e-12`, and no explicit iteration cap. These
settings are available through the public `SCFResponseSettings` type in
`dxtb.scf`. The converted Potential root is independently certified and is
never refined or silently replaced with the primal coordinates. Explicit
`charge`, `potential`, `fock`, and `primal` response coordinates are supported.

SOSCF is constructible through `SOSCFMethod()` and the named SOSCF presets.
Its epsilon-scale stationary and occupation-sector safety controls remain
internal implementation details rather than public settings.

Convergence profiles are explicit and are never selected from
`requires_grad` or transform state:

| Profile | Fixed-point L2 | Fixed-point max | Stationarity gate |
|---|---:|---:|---:|
| `standard` | `1e-4` | `1e-5` | `1e-5` |
| `tight` | `1e-10` | `1e-10` | `1e-9` |
| `higher_order` | `1e-12` | `1e-12` | `1e-12` |

Explicit `x_atol` and `x_atol_max` override the selected profile. Effective
thresholds are shown by `ConfigSCF.info()`. Dtype resolution and S2 checks
remain authoritative.

## Transform and temperature policy

| Method/policy | Fixed-trip | `vmap` | Active Fermi smearing |
|---|---|---|---|
| Linear | Supported | Supported | Supported |
| Anderson | Supported | Supported | Supported |
| Broyden | Supported | Supported | Supported |
| CDIIS | Supported, Fock only | Supported | Supported |
| SOSCF | Eager-only | Unsupported | Rejected |
| Dynamic multi-stage strategy | Eager-only | Unsupported | Supported unless it contains SOSCF |

Single-stage fixed-trip methods are the supported same-shape mapping path.
Dynamic representation-changing stages and SOSCF use eager primal execution
with implicit differentiation. A fixed-trip or active-smearing SOSCF
combination is rejected before physical SCF setup.

Automatic `torch.compile` is off. S12 found method-transition compile
characterization but continued raw-map and whole-solver graph breaks, without
optimized-backend speed evidence. Small same-shape `vmap` batches are
supported; streaming remains the memory-bounded choice for large derivative
batches. The default fixed-trip budget may evaluate more maps than an eager
early-terminating solve, so eager high-order properties should prefer the
documented implicit route where supported.

## Results and status

The result contract remains: convergence is S2 primal certification;
iterations and residual describe the primal controller; strategy diagnostics
describe the primal stages; and response failure is a derivative failure, not
a retrospective primal failure. GFN0 remains non-SCF and does not enter the
strategy controller.

The final architecture is:

```text
physical problem
    → SCF strategy and method stages
    → shared S2 certification
    → certified physical root
    → unrolled or implicit differentiation
    → physical result
```

The remaining follow-up areas are future work rather than S13 requirements:
thermal second-order theory, optimized `torch.compile` execution, dynamic
multi-stage `vmap`, and public mixed-System routing owned by its existing
package.

## Final S13 decision

| Policy | Frozen choice |
|---|---|
| Default primal | Anderson / Potential |
| Default iteration budget | 100 |
| Default convergence profile | `standard`: fixed-point L2 `1e-4`, max `1e-5`, stationarity `1e-5` |
| Default differentiation | `unrolled` |
| Recommended eager high-order differentiation | `implicit` |
| Default implicit response | Potential, BiCGSTAB, `rtol=1e-10`, `atol=1e-12`, no iteration cap |
| Automatic SOSCF | Disabled; fixed-occupation eager opt-in only |
| Active-smearing second order | None; retain first-order methods |
| Automatic `torch.compile` | Disabled |
| Same-shape batch policy | Fixed-trip `vmap` for single-stage methods; streaming for memory-bounded large derivative batches |

The current-head benchmark gives the Anderson default equal certification
success (15/15) and lower median and p90 cost than Broyden on this decision
set. This is a portfolio choice from measured cases, not a claim of universal
superiority. Broyden remains a supported native limited-memory first method;
CDIIS remains a supported Fock-only commutator method.

The public construction surface is `Config.create()` with `strategy`,
`differentiation`, and `convergence` arguments, plus the immutable method and
stage types in `dxtb.scf`. Stable named strategies are `linear`, `anderson`, `broyden`,
`cdiis`, `broyden-cdiis`, `anderson-cdiis`, `cdiis-soscf`, and
`broyden-cdiis-soscf`; dynamic and SOSCF-containing presets require eager
implicit execution, and SOSCF-containing presets reject active smearing.
There is no automatic solver substitution.

`mixer`/`scp_mode`/`scf_mode` inputs remain only as parser-edge migration
adapters; they are not stored in `ConfigSCF`. Anderson-specific tuning is
stored in `AndersonMethodSettings`. `strict` and `f_atol` have no core owner;
`f_atol` is rejected as retired, and convergence always uses physical S2
certification. The default is never inferred from `requires_grad`.
