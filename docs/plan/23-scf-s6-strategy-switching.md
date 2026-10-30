# SCF S6: eager multi-stage strategy switching

## Scope and controller ownership

S6 extends the immutable `SCFStrategy` from a single executable stage to an
ordered eager program. Each non-final `SCFStage` declares an exit policy; the
last stage has no exit policy. One global `SCFSolverSettings.maxiter` budget
covers every stage. The fixed-trip controller continues to accept one stage
and raises `NotImplementedError` for multiple stages.

The shared eager controller owns stage selection, stage-local physical map
counts, method-state reset, stage-entry damping reset, and final S2
certification. Methods remain unaware of the next stage. Switching is
considered only after a candidate has failed certification and is not terminal.
It can be triggered by all configured residual/stationarity thresholds or by
an explicitly configured maximum stage count. No energy or free-energy
observable is computed for a switch.

The result's internal strategy diagnostics record the final stage, switch
count, per-stage map counts, switch reason codes, and fixed-point/stationarity
metrics at each configured boundary. A switch itself performs no map and does
not increment the global iteration count. The new method receives `None` state
at local step 1 and starts with its declared damping (or the common solver
damping); dynamic damping's `switched` flag is cleared. `mix_guess` applies
only to solver initialization.

## Fixed-trip inactive-lane sanitation

Fixed-trip execution still evaluates dummy maps for inactive lanes. The
controller now retains the last finite, paired method observation separately
from the selected diagnostic result. Inactive method calls receive that stable
finite pair and its projected proposal. The selected terminal map, root pair,
and certification remain unchanged, so failure diagnostics still expose the
actual bad candidate. Tests cover Linear, Anderson, Broyden, and CDIIS with
terminal non-finite candidates and check finite method state, JVP, `jacfwd`,
reverse Jacobian, and mapped lanes.

## Representation conversion

The only cross-representation conversion in S6 is compact CHARGE/POTENTIAL to
FOCK. For the current observation, the Fock input is the Hamiltonian that
generated its density, `candidate.spectral.hamiltonian`. The converted map
output is rebuilt from that same spectrum's charges. The spectrum is reused
and no diagonalization is performed at the switch. CDIIS is seeded immediately
from that exact pair. A converted pair may be certified in Fock space before a
new map is evaluated. Same-representation transitions, including FOCK to
FOCK, are supported. FOCK to a compact representation is rejected explicitly.

Parity tests compare the converted pair with a separate `fock_map` call for
charge and potential observations; that extra evaluation is test-only. The
controller's global iteration counts show no additional map at transition.

## Strategy and threshold characterization

The experimental policy used for the runs below was:

```text
min stage iterations: 3
fixed-point L2:        1e-2
stationarity:          1e-2
forced stage limit:    15 (H2O, NO2, H2O+) / 20 (C4H5NCS, stretched H2)
stage-0 damping:       0.5
CDIIS-entry damping:   1.0
global maxiter:        50 (H2O), 60 (other GFN2 cases), 100 (stretched H2)
root L2/max:           1e-7
```

The forced limits were a characterization guard, not a selected production
preset. Metric thresholds caused the observed switches in H2O, NO2, H2O+ and
C4H5NCS. Stretched H2 was already certified after its first loop candidate,
so certification correctly took precedence and no switch occurred.

### Switch-threshold sweep

A follow-up Broyden(potential) → CDIIS(Fock) sweep used matching fixed-point
L2 and orthogonalized stationarity thresholds, with `min_stage_iterations=3`,
stage damping 0.5 → 1.0, and the same certified-root tolerances. Each row
switched on the configured physical threshold; no forced stage budget was
used. The switch energy was measured by a benchmark-only observer and is not
computed by the controller. CDIIS history count is measured immediately after
the converted observation seeds the new stage.

| System | Threshold | Maps (Broyden+CDIIS) | Switch residual L2 | Switch stationarity | Switch energy (Eh) | CDIIS entry history |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H2O/GFN2 | 1e-1 | 3+5 | 3.653e-2 | 1.618e-2 | -5.103837606281 | 1 |
| H2O/GFN2 | 1e-2 | 5+4 | 9.088e-3 | 4.053e-3 | -5.104138621045 | 1 |
| H2O/GFN2 | 1e-3 | 9+3 | 5.675e-4 | 2.556e-4 | -5.104158718315 | 1 |
| NO2/GFN2, 300 K | 1e-1 | 5+7 | 5.378e-2 | 2.564e-2 | -10.951492308608 | 1 |
| NO2/GFN2, 300 K | 1e-2 | 8+6 | 6.850e-3 | 3.252e-3 | -10.953050491001 | 1 |
| NO2/GFN2, 300 K | 1e-3 | 11+5 | 8.755e-4 | 4.134e-4 | -10.953075397862 | 1 |
| C4H5NCS/GFN2, 300 K | 1e-1 | 8+17 | 6.165e-2 | 7.283e-3 | -19.693711660216 | 1 |
| C4H5NCS/GFN2, 300 K | 1e-2 | 11+14 | 8.341e-3 | 9.529e-4 | -19.694053443776 | 1 |
| C4H5NCS/GFN2, 300 K | 1e-3 | 15+12 | 5.178e-4 | 1.345e-4 | -19.694059627742 | 1 |

All nine runs certified at the same final energy as the standalone methods.
The `1e-2` pair gives a practical middle ground: it switches after the compact
method has materially reduced the residual, while avoiding the extra compact
iterations of `1e-3`; `1e-1` gives a one-to-two map advantage on the smaller
systems but switches farther from the root. This remains an explicit
experimental policy, not a public default. CDIIS enters with one paired Fock
observation in every run.

### Physical results

All rows below used the eager shared controller on the PyTorch 2.14.1
environment. `Maps` is the reported global iteration count, including the
initial physical map under the established controller contract. Stage counts
sum to the same global count; a conversion adds no map.

| System | Strategy | Maps / per-stage maps | Certified | Final orthogonalized stationarity | Electron-count error | Energy (Eh) | Wall time (s) |
| --- | --- | ---: | :---: | ---: | ---: | ---: | ---: |
| H2O/GFN2 | Broyden | 22 / 22 | yes | 3.33e-8 | — | -5.104158799410 | 0.195 |
| H2O/GFN2 | Anderson | 12 / 12 | yes | 8.36e-9 | — | -5.104158799410 | 0.108 |
| H2O/GFN2 | CDIIS | 7 / 7 | yes | 2.14e-10 | — | -5.104158799410 | 0.084 |
| H2O/GFN2 | Broyden → CDIIS | 9 / 5+4 | yes | 1.25e-9 | — | -5.104158799410 | 0.120 |
| H2O/GFN2 | Anderson → CDIIS | 11 / 8+3 | yes | 1.07e-8 | — | -5.104158799410 | 0.142 |
| NO2/GFN2, doublet, 300 K | Broyden | 25 / 25 | yes | 2.86e-8 | 7.11e-15 | -10.953075803272 | 1.887 |
| NO2/GFN2, doublet, 300 K | Anderson | 16 / 16 | yes | 4.46e-9 | 7.11e-15 | -10.953075803272 | 1.221 |
| NO2/GFN2, doublet, 300 K | CDIIS | 10 / 10 | yes | 2.15e-8 | 7.11e-15 | -10.953075803272 | 0.840 |
| NO2/GFN2, doublet, 300 K | Broyden → CDIIS | 14 / 8+6 | yes | 2.23e-9 | 3.55e-15 | -10.953075803272 | 1.147 |
| NO2/GFN2, doublet, 300 K | Anderson → CDIIS | 14 / 8+6 | yes | 4.26e-8 | 7.11e-15 | -10.953075803272 | 1.141 |
| H2O+/GFN2, doublet, 1000 K | Broyden | 20 / 20 | yes | 3.58e-8 | 8.88e-16 | -4.431210313834 | 1.443 |
| H2O+/GFN2, doublet, 1000 K | Anderson | 12 / 12 | yes | 3.62e-10 | 1.78e-15 | -4.431210313831 | 0.902 |
| H2O+/GFN2, doublet, 1000 K | CDIIS | 6 / 6 | yes | 1.37e-8 | 0 | -4.431210313832 | 0.513 |
| H2O+/GFN2, doublet, 1000 K | Broyden → CDIIS | 7 / 4+3 | yes | 3.77e-8 | 0 | -4.431210313834 | 0.596 |
| H2O+/GFN2, doublet, 1000 K | Anderson → CDIIS | 10 / 6+4 | yes | 3.40e-10 | 0 | -4.431210313831 | 0.814 |
| C4H5NCS/GFN2, 300 K | Broyden | 29 / 29 | yes | 2.40e-8 | 1.42e-14 | -19.694059708019 | 2.427 |
| C4H5NCS/GFN2, 300 K | Anderson | 24 / 24 | yes | 1.55e-8 | 1.42e-14 | -19.694059708019 | 2.012 |
| C4H5NCS/GFN2, 300 K | CDIIS | 22 / 22 | yes | 1.15e-8 | 2.13e-14 | -19.694059708019 | 1.926 |
| C4H5NCS/GFN2, 300 K | Broyden → CDIIS | 25 / 11+14 | yes | 5.21e-9 | 1.42e-14 | -19.694059708019 | 2.193 |
| C4H5NCS/GFN2, 300 K | Anderson → CDIIS | 24 / 11+13 | yes | 6.59e-9 | 1.42e-14 | -19.694059708019 | 2.117 |
| stretched H2/GFN1, 3.0 Å | Broyden | 2 / 2 | yes | 1.32e-15 | — | -0.924088449850 | 0.017 |
| stretched H2/GFN1, 3.0 Å | Anderson | 2 / 2 | yes | 1.32e-15 | — | -0.924088449850 | 0.016 |
| stretched H2/GFN1, 3.0 Å | CDIIS | 2 / 2 | yes | 1.32e-15 | — | -0.924088449850 | 0.019 |
| stretched H2/GFN1, 3.0 Å | Broyden → CDIIS | 2 / 2+0 | yes | 1.32e-15 | — | -0.924088449850 | 0.016 |
| stretched H2/GFN1, 3.0 Å | Anderson → CDIIS | 2 / 2+0 | yes | 1.32e-15 | — | -0.924088449850 | 0.016 |

The switch path reduced maps materially for H2O, NO2 and H2O+ versus the
corresponding compact methods, and reached the same certified root observables
as standalone CDIIS. C4H5NCS showed parity or a modest improvement, not a
universal acceleration. These are experimental internal strategies, not public
defaults. The initial threshold study did not sweep a broad grid, so it does
not establish one universal switch threshold.

## W2c9 adapter reconciliation

The synchronized W2c9 batch-implicit module still imported the pre-S3
`_solve_map_anderson` driver and selected controllers by settings-bundle type.
The focused W2c9 test therefore failed collection at the S6 baseline. The
batch adapter now prepares the common `(SCFSolverSettings, SCFStrategy)` pair
and calls `_solve_map` for its existing one-stage route. Its implicit
finalizers pass the exact attached root as required by S2b.1. This is a narrow
compatibility reconciliation; it does not change W2c9 spectral mathematics or
batch routing policy.

## Validation status

The focused S6 strategy plus S2 certification selection passed **86 tests**
on Python 3.14 / PyTorch 2.14.1 and Python 3.11 / PyTorch 2.8.0. It includes
compact-to-Fock pair parity, forced and metric-triggered switching, global map
accounting, stage-entry method state/damping reset, same-representation Fock
switches, fixed-trip rejection of multi-stage strategies, and post-terminal
dummy-trip JVP/Jacobian checks for Linear, Anderson, Broyden, and CDIIS. The
switched-versus-standalone CDIIS implicit-boundary regression passed for H2O
and finite-temperature NO2. NO2 used `1e-12` fixed-point tolerances; first- and
second-order parameter derivative comparisons through the attached root
matched standalone CDIIS in both cases.

The Phase-A groups completed with **228 passed and one inherited failure** in
Group 1, **58 passed** in Group 2, and **128 passed** in Group 3. The only
failure is `test_anderson_solver_real_position_parameter_and_field_gradients`:
maximum difference `2.5007065862503453e-11` against the unchanged `2e-13`
tolerance. The lifecycle/E6a.2 selection passed **262 tests**. The combined
Group 3, spectral/free-energy, W2c5b/W2c6, and W2c9 batch-implicit selection
passed **188 tests**; Group 3 was also rerun alone and passed all 128 tests.
The non-slow implicit/root selection passed **58 tests**.

The W2c9 reconciliation updates the stale batch adapter to use the current
common one-stage strategy interface at both setup and refresh. Its focused
selection passes after both call sites were migrated. This is compatibility
work required by the synchronized W2c9 source; no W2c9 spectral mathematics
changed. The switched NO2 check reaches a tight `1e-12` certified root and
compares first- and second-order attached-root responses against standalone
CDIIS at the same finite-temperature state. The accepted slow NO2
hyperpolarizability reference also passed in 460 seconds; it remains the
independent physical higher-order observable control. No SOSCF, EDIIS/ADIIS,
public strategy defaults, or dynamic fixed-trip stage switching were started.
