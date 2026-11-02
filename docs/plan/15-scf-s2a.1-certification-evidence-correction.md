# S2a.1 certification evidence correction

## Review point and scope

This diagnostic package starts from SCF `87e0a28e5be4dfb086b763ed7e2653d7598bf3b5`, synchronized to general-refactor `4a6d8100d28332aedc455065962e7ed76b506a60`.
It corrects the S2a characterization provenance, adds raw/orthogonalized
stationarity evidence, and recommends S2b metric choices. It does not change
production stopping, status, or implicit-root behavior.

## Model provenance correction

The original characterization helper selected its parameter set with:

```python
"gfn1" if method == GFN1_XTB else "gfn2"
```

`GFN1_XTB` and `GFN2_XTB` are distinct lazy parameter-loader objects, but their
overloaded equality comparison returns true across those objects. Thus the
helper used GFN1 parameters for a requested GFN2 calculation. The displayed
`-5.805... Eh` H2O result is GFN1; the `-5.104... Eh` result is GFN2. The
apparent second GFN2 root was a model-labeling error, not evidence of root
multiplicity.

The test fixture now dispatches by object identity. The characterization
utility independently asserts all of:

| Requested model | Resolved Calculator/System config | H0 method | Parameter model | H2O energy |
| --- | ---: | --- | --- | ---: |
| GFN1-xTB | `1` | `gfn1` | `GFN1-xTB` | `-5.8052487050843 Eh` |
| GFN2-xTB | `2` | `gfn2` | `GFN2-xTB` | `-5.1041587994096 Eh` |

The corresponding established references are `-5.8052489623704 Eh` and
`-5.1041590251073 Eh`. The test checks model identity and uses a broad 1 mEh
energy discrimination, rather than imposing a tighter new reference.
`test/test_scf/scf_certification_characterization.py` emits the resolved model
on every row and aborts before recording a mismatched case.

The identity probe used H2O, neutral singlet, zero temperature, quadrupole
integrals. GFN1 had `nel=[4,4]`, occupation rows `[1,1,1,1,0,0,0,0]`,
`n0=[2,4/3,4/3,4/3,1,0,1,0]`, and initial packed-state SHA256
`c045d9f1dd685578cdc65c6111632994911b05f9dffa9847400a558c10714919`.
Its H0 norm/sum were `2.0541320208` / `-7.5352610718`, and overlap
norm/sum were `3.1509904224` / `11.7919406554`.
GFN2 had `nel=[4,4]`, occupation rows `[1,1,1,1,0,0]`,
`n0=[2,4/3,4/3,4/3,1,1]`, and initial packed-state SHA256
`f31e0af5cdb091b5b31e008cfd4c72fec4e2be84645eb6c5118a5bf2f0641c39`.
Its H0 norm/sum were `1.7686719852` / `-6.1247020733`, and overlap
norm/sum were `2.7566791809` / `9.2952919699`. The distinct basis sizes
(8 versus 6 AOs) reflect the different parameter models.

## Setup-order and shared-state probe

The sequences `GFN1 -> GFN2`, `GFN2 -> GFN1`, `GFN1 -> GFN2 -> GFN1`,
`GFN2 -> GFN1 -> GFN2`, and `GFN2 -> GFN2 -> GFN2` were run in one process.
Each sequence was checked using both the normal test helper and independently
constructed fresh `ParamModule` objects. Parameter tensors were snapshotted
around construction and SCF solving.

For each resolved model, repeated construction produced exact-equal parameter
digests, initial packed states, `n0`, occupation, `nel`, H0, overlap,
interaction setup tensors, converged root, energy, and iteration count. Shared
parameter snapshots were unchanged. The result was identical in Python
3.14/PyTorch 2.14.1 and Python 3.11/PyTorch 2.8.0. The old order-dependence
claim does not reproduce; no production shared-state pollution was found.

## Correct GFN2/H2O root-tolerance sweep

Rows were independently constructed from the same verified GFN2/H2O charge
initial state, at zero electronic temperature. Every row resolves to config
`2`, H0 `gfn2`, and parameter model `GFN2-xTB`.
The shared initial packed-state SHA256 is
`f31e0af5cdb091b5b31e008cfd4c72fec4e2be84645eb6c5118a5bf2f0641c39`; all
rows use the same quadrupole integral level, damping, method, and temperature.

| Model / system | Charge / spin | Representation | T / dtype | Requested/effective tolerance | Fixed-point L2 / max | Raw AO stationarity L2 / max | Orthogonalized stationarity L2 / max | Electron-count error | Energy / total free energy (Eh) | Maps |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GFN2 / H2O | `0 / 0` | charge | `0 K / f64` | `1e-6 / 1e-6` | `9.18e-7` / `4.92e-7` | `5.44e-8` / `2.23e-8` | `6.69e-8` / `2.80e-8` | `1.8e-15` | `-5.104158799409646` / same | 54 |
| GFN2 / H2O | `0 / 0` | charge | `0 K / f64` | `1e-8 / 1e-8` | `9.97e-9` / `5.57e-9` | `6.08e-10` / `2.49e-10` | `7.48e-10` / `3.13e-10` | `1.8e-15` | `-5.104158799409648` / same | 77 |
| GFN2 / H2O | `0 / 0` | charge | `0 K / f64` | `1e-10 / 1e-10` | `9.04e-11` / `5.15e-11` | `5.59e-12` / `2.29e-12` | `6.87e-12` / `2.88e-12` | `0` | `-5.104158799409648` / same | 101 |
| GFN2 / H2O | `0 / 0` | charge | `0 K / f64` | `1e-12 / 1e-12` | `8.25e-13` / `4.75e-13` | `5.10e-14` / `2.11e-14` | `6.29e-14` / `2.66e-14` | `0` | `-5.104158799409649` / same | 125 |

Fermi status is structurally valid (Aufbau) and all rows are finite. The energy
is consistent with the established GFN2/H2O result.

## GFN2/H2O representation comparison

All rows use independently prepared GFN2/H2O inputs and requested raw
tolerance `1e-8`.

| Model / system | Charge / spin | Representation | T / dtype | Requested/effective tolerance | Fixed-point L2 / max | Raw AO stationarity L2 / max | Orthogonalized L2 / max | Electron-count error | Energy (Eh) | Maps |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GFN2 / H2O | `0 / 0` | charge | `0 K / f64` | `1e-8 / 1e-8` | `9.97e-9` / `5.57e-9` | `6.08e-10` / `2.49e-10` | `7.48e-10` / `3.13e-10` | `1.8e-15` | `-5.104158799409648` | 77 |
| GFN2 / H2O | `0 / 0` | potential | `0 K / f64` | `1e-8 / 1e-8` | `9.04e-9` / `7.45e-9` | `5.95e-9` / `2.43e-9` | `7.32e-9` / `3.06e-9` | `0` | `-5.104158799409649` | 65 |
| GFN2 / H2O | `0 / 0` | Fock | `0 K / f64` | `1e-8 / 1e-8` | `8.52e-9` / `3.82e-9` | `4.89e-9` / `2.00e-9` | `6.02e-9` / `2.52e-9` | `1.8e-15` | `-5.104158799409650` | 66 |

The energies agree to displayed precision. Raw fixed-point norms and the
stationarity residual reached at a given nominal root tolerance vary by
representation, so the fixed-point gate must remain representation-specific.

## Stationarity definitions and basis behavior

The retained raw AO diagnostic is:

```text
R_AO = F[P] P S - S P F[P]
```

where `F[P]` is rebuilt from the density-derived charges. The candidate
orthogonalized diagnostic uses `X = S^(-1/2)`:

```text
R_orth = X.T R_AO X
```

An independent SPD-overlap algebra test verifies that this equals
`F_orth P_orth - P_orth F_orth`, with `F_orth = X.T F X` and
`P_orth = X^(-1) P X^(-T)`. Simultaneous AO permutation preserves L2 and max
for both residuals.

A synthetic diagonal basis rescaling (`diag(0.35, 1.7, 3.2)`) changed raw AO
L2/max from `0.79980` / `0.46933` to `1.04922` / `0.60244`, while
orthogonalized Frobenius/L2 stayed at `0.475056` to roundoff. The
orthogonalized max changed from `0.281981` to `0.267179` (about 5.5%) because
symmetric orthogonalization after a general basis rescaling can differ by an
orthogonal rotation; an elementwise maximum is not invariant to that rotation.
Thus orthogonalized L2
is the more basis-independent stationarity scalar. Keep both raw and
orthogonalized max values as diagnostics, but do not describe either max as
fully basis invariant.

Size observations at requested `1e-8`:

| Model/system | Charge / spin | Representation | T / dtype / tolerance | Active AO | Raw AO L2 / max | Orthogonalized L2 / max | Raw L2 / sqrt(active AO pairs) |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| GFN2/H2 | `0 / 0` | charge | `0 K / f64 / 1e-8` | 2 | `2.22e-16` / `1.11e-16` | `1.34e-16` / `6.68e-17` | `1.11e-16` |
| GFN2/H2O | `0 / 0` | charge | `0 K / f64 / 1e-8` | 6 | `6.08e-10` / `2.49e-10` | `7.48e-10` / `3.13e-10` | `1.01e-10` |
| GFN2/NO2 | `0 / 1` | charge | `300 K / f64 / 1e-8` | 12 | `6.88e-10` / `2.26e-10` | `8.82e-10` / `2.68e-10` | `5.74e-11` |

This small, chemically varied sample does not show a monotonic raw L2 increase
with AO count; the H2 projector is trivially stationary, while H2O and NO2
have different spectra and occupation behavior. It is not enough to validate
one absolute L2 threshold over arbitrary system sizes. The max values are
similar for these two nontrivial roots, but more systems are needed before
claiming size robustness.

`SCFProblem` currently represents one unbatched `SCFInputs`, whereas canonical
mixed-System padding is handled by the separate `PaddedSCF`/`PaddedSCFSpectralResult`
path. S2a.1 does not force those types together. Existing padded tests verify
physical masks and inactive AO blocks; the certification unit test verifies
packed residual tails are excluded. Full certification over a mixed padded
System remains a downstream boundary, not evidence claimed here.

## Charged/open-shell, finite-temperature, and float32 cases

Verified GFN2 water cation doublet (`charge=+1`, `spin=1`, `nel=[4,3]`,
7 electrons):

| Model/system | Charge / spin | Representation, T, dtype | Requested/effective tolerance | Fixed-point L2 / max | Raw stationarity L2 / max | Orthogonalized L2 / max | `Tr(P S)` error | Energy (Eh) | Maps |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GFN2/H2O+ | `+1 / 1` | charge, 0 K, f64 | `1e-6 / 1e-6` | `9.06e-7` / `3.50e-7` | `2.85e-8` / `1.22e-8` | `3.53e-8` / `1.50e-8` | `0` | `-4.431225111368876` | 47 |
| GFN2/H2O+ | `+1 / 1` | charge, 0 K, f64 | `1e-10 / 1e-10` | `8.92e-11` / `3.66e-11` | `3.67e-12` / `1.55e-12` | `4.56e-12` / `1.90e-12` | `8.9e-16` | `-4.431225111368877` | 91 |

Both spin targets are positive and the independent count uses their sum. The
finite-temperature GFN2/NO2 doublet (`nel=[9,8]`, 300 K) has independent count
error `-7.1e-15`; the separate Fermi status reports convergence and its own
electron-count check. At raw tolerance `1e-8`, raw/orthogonalized stationarity
L2 is `6.88e-10` / `8.82e-10`; at `1e-12`, it is `6.87e-14` / `8.79e-14`.
Energy/free energy are `-10.953075803271599` / `-10.953075803272070 Eh` at
`1e-8`, and `-10.953075803271602` / `-10.953075803272073 Eh` at `1e-12`.
Fermi status is valid in both cases.

Verified GFN2/H2O and GFN2/NO2 doublet in float32 at 300 K and 1 K, each with
requested `1e-12`, which resolves to `5e-5` under `_resolve_tolerance`
(`50 * finfo(float32).resolution`):

| Model/system | Charge / spin | T | Representation, requested/effective tol | Fixed-point L2 / max | Raw stationarity L2 / max | Orthogonalized L2 / max | Independent count error | Maps; Fermi / finite |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| GFN2/H2O | `0 / 0` | 300 K | charge, `1e-12 / 5e-5` | `4.82e-5` / `2.41e-5` | `2.69e-6` / `1.13e-6` | `3.24e-6` / `1.39e-6` | `0` | 34; valid / valid |
| GFN2/H2O | `0 / 0` | 1 K | charge, `1e-12 / 5e-5` | `4.82e-5` / `2.41e-5` | `2.69e-6` / `1.13e-6` | `3.24e-6` / `1.39e-6` | `0` | 34; valid / valid |
| GFN2/NO2 | `0 / 1` | 300 K | charge, `1e-12 / 5e-5` | `4.20e-5` / `1.38e-5` | `3.65e-6` / `1.07e-6` | `4.75e-6` / `1.33e-6` | `-3.81e-6` | 36; valid / valid |
| GFN2/NO2 | `0 / 1` | 1 K | charge, `1e-12 / 5e-5` | `4.20e-5` / `1.38e-5` | `3.65e-6` / `1.07e-6` | `4.75e-6` / `1.33e-6` | `-3.81e-6` | 36; valid / valid |

The H2O gap leaves its Fermi occupations saturated, so it checks the smeared
path but not fractional occupations. NO2's independent count error shows why
float32 must not inherit a float64 threshold. Fermi status separately reports
its internal electron-count flag; for NO2 the Fermi status reports convergence
and its internal electron-count check passes despite the independent trace
error. The energy/free-energy observations were `-5.104158401489258 Eh` for
H2O and `-10.953073501586914 Eh` for NO2. The sampled float64 errors were
below `7.2e-15`.

## S2b metric recommendation

- **Fixed point:** retain the existing projected `G(x)-x` L2/max pair and
  representation-specific tolerance profiles. The corrected GFN2 rows show
  similar raw convergence order but different physical stationarity at a
  common nominal tolerance.
- **Stationarity:** use orthogonalized Frobenius/L2 as the leading physical
  diagnostic candidate because it is invariant under the tested basis
  rescaling; retain raw AO L2/max and orthogonalized max in diagnostics. Do not
  freeze a universal absolute L2 threshold yet: the sample is small and the
  root/molecule dependence is material. S2b should consider a max gate plus a
  size-aware L2/RMS condition, or gather more systems first.
- **Electron count:** require both the independent `Tr(P S)` difference and
  the Fermi subsolver's own validity when smearing is active. Float64 samples
  were within `7.2e-15`; float32 H2O was exactly integral at the reported
  precision but does not bound less saturated systems. A candidate scale may
  depend on dtype epsilon, active AO count, and target electron count, but the
  current evidence does not choose a safe multiplier. Add an open-shell,
  fractionally occupied float32 case before fixing that rule.
- **Profiles:** standard can begin from current representation-aware root
  tolerances, but stationarity ceilings remain provisional. Tight/reference
  roots at `1e-10` to `1e-12` bring H2O and NO2 stationarity to approximately
  `1e-12` to `1e-14`; higher-order/reference remains explicit and should not
  be inferred from gradient mode. No public profile or tolerance is introduced
  in S2a.1.

## Timing and remaining limits

Median CPU timing used two warmups and seven measurements per component in
Python 3.14/PyTorch 2.14.1. `Core` includes projected fixed-point metrics,
fresh Fock assembly, AO stationarity, and electron count; it excludes energy
and entropy. `Full` calls the unchanged S2a metric function.

| Case | One raw map | Core metrics | Electronic energy | Free-energy correction | Full metrics |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2/GFN1, 4 AO | 4.33 ms | 1.13 ms | 0.33 ms | 0.003 ms | 1.42 ms |
| H2O/GFN2, 6 AO | 6.62 ms | 5.51 ms | 2.03 ms | 0.003 ms | 5.38 ms |
| NO2/GFN2, 12 AO, 300 K | 73.01 ms | 5.66 ms | 2.08 ms | 31.68 ms | 38.86 ms |

At finite temperature the entropy trace dominates the full certification cost.
S2b should reuse already finalized energy observables or keep optional energy
progress separate from the authoritative physical gate. No extra map or
eigensolve is added by these diagnostics. The timing artifact is kept outside
the repository at `/tmp1/friede/dxtb-agentic/scf-s2a.1/timing-breakdown.json`.

The sampled padded boundary is limited as described above. No production
certification or stopping behavior was changed, no autodiff-dependent
tolerance selection was added, and no spectral algorithm or existing physical
certification formula was changed.
