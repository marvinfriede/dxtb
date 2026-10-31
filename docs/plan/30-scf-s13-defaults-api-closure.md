# SCF S13 — defaults and public strategy API

## Decision status

This document records the S13 candidate policy. The implementation and
regression gates are still under review; the S0–S13 redesign is not marked
complete until the validated candidate is published.

The final-head one-thread benchmark matrix in
`benchmarks/results/scf-s13/defaults-py314-torch2141.json` contains 45
method/case runs: 15 GFN1/GFN2, charge, temperature, and poor-start cases,
each run with Anderson, Broyden, and CDIIS. All three methods certified all
15 cases with the public 100-map budget. Anderson/Potential used a median of 9
maps and a p90 of 12, compared with Broyden/Potential at 15/42 and CDIIS/Fock
at 13/17. Median and p90 wall times were 0.078/0.191 s for Anderson,
0.144/0.396 s for Broyden, and 0.154/0.266 s for CDIIS. The benchmark records
its source revision, runtime, dtype, thread count, tolerances, and raw rows.

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
`ConfigSCF`; ambiguous conflicts are rejected. `f_atol` is retired because
physical fixed-point and stationarity certification own SCF convergence.

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
settings `rtol=1e-10`, `atol=1e-12`, and no explicit iteration cap. The
converted Potential root is independently certified and is never refined or
silently replaced with the primal coordinates. The alternate `charge`,
`potential`, `fock`, and `primal` response choices remain represented by the
internal response configuration pending the full public-settings decision.

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
