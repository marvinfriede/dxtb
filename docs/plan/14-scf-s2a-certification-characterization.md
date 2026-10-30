# S2a physical certification primitives and characterization

## 1. Review point and scope

S2a starts at SCF commit `0053e26519578d134070cb762e5b17df973fe5a5`, based
on general-refactor synchronization point
`573f706df2047d7ee96fdfeee25f8a0378cc5966`. During S2a the published branch
advanced to `4a6d8100d28332aedc455065962e7ed76b506a60` through two W2c5
diagnostic/documentation commits. Their file list contains no SCF source or
SCF test changes, and the coordination log had no unresolved blocker. S2a was
rebased at the package boundary onto that published head. The general-refactor
worktree was not changed.

This package adds diagnostic physical metrics only. No controller stopping
decision, `SCFSolveResult` status, public status, implicit-root acceptance,
force-convergence policy, or solver tolerance changed. The spectral module is
untouched. `SCFProblem` and its paired `SCFPhysicalState` supply the inputs to
certification; method, method state, strategy, and iteration count are not
inputs.

## 2. Metric ownership and formulas

`SCFPhysicalState.delta` remains the sole raw fixed-point delta:

```text
G(state.x) - state.x
```

`SCFProblem.project_residual` applies the same structural charge, potential,
and Fock padding rules used by the fixed-trip controller. Packed PADNZ tails
are zeroed in the measured residual. The two retained norms are raw Euclidean
L2 and max absolute; they are labeled with the representation because charge,
potential, and Fock residuals have different units.

`SCFProblem.rebuild_hamiltonian(charges)` is the canonical physical rebuild:

```text
charges -> existing potential_from_charges -> existing
potential_to_hamiltonian
```

It performs no occupation calculation or diagonalization. Certification uses
the spectral density `P` paired with the tested raw input and the freshly
rebuilt `F[P]`:

```text
R = F[P] P S - S P F[P]
```

The Frobenius/L2 and max-absolute norms are unnormalized. The orbital pair mask
from `SCFInputs.orbital_mask` removes padded AO rows and columns before either
norm is computed. A regression shows the Hamiltonian that produced `P` has a
near-zero commutator by construction while the fresh-Fock residual is
materially nonzero for the initial H2O state. An actual nonorthogonal H2O
overlap and a simultaneous AO permutation check verify the generalized-basis
formula and norm covariance.

The independent electron count is:

```text
N[P] = sum_spin Tr(P S)
target = sum(inputs.nel)
error = N[P] - target
```

It is measured separately from `FermiStatus.electron_count_ok`. Fermi
convergence, inner electron-count status, invalid temperature, and too-many
electrons remain separate tensor flags. Aufbau states with no Fermi status are
structurally Fermi-valid. `finite_ok` inspects the paired state, map and
spectral tensors, charges and driving potential, Fermi residual/core state,
fresh Hamiltonian, continuous metrics, and energy/free-energy values. A real
out-of-envelope matrix-Fermi evaluation, made by scaling the synthetic H2
core Hamiltonian by `1e20` at 300 K, retained a valid-looking Fermi status but
returned non-finite density/energy. Certification correctly reports
`finite_ok=False` and `certified=False`.

## 3. Explicit certification and progress values

`SCFCertificationThresholds` requires caller-provided values for one explicit
representation. There are no defaults or public options. A representation
mismatch raises `ValueError`. `certify_scf_state` returns component verdicts
for fixed point, stationarity, independent electron count, finite values, and
Fermi validity; their conjunction is a testable `certified` tensor. Nothing
consumes that verdict in production in S2a.

`electronic_energy_from_spectral` and `free_energy_from_spectral` now live in
`functional/energy.py`; the existing solver imports/re-exports these helpers
and finalization calls the same formulas. The electronic energy remains the
existing atom-partitioned core plus interaction energy. `electronic_free_energy`
is explicitly the total `electronic_energy + sum(fenergy)`, not `fenergy` alone.
Equal and atomic entropy partitions change per-atom distribution while
preserving the total within numerical precision.

`scf_progress_metrics` compares energy and total electronic free energy for
two already paired physical states. It does not run another raw SCF map. A
focused guard also confirms certification does not call `SCFProblem.map` or
diagonalize the fresh Hamiltonian.

## 4. Characterization results

> **S2a.1 correction:** The GFN2/H2O rows in this section are superseded.
> The test helper selected GFN1 parameters for both `GFN1_XTB` and
> `GFN2_XTB`: distinct lazy parameter loaders compare equal with `==`. The
> `-5.805... Eh` H2O result is GFN1; it is not a GFN2 root. Some later rows
> also lacked a model-identity guard and must not be treated as verified GFN2
> evidence. The helper now uses identity dispatch, and the corrected,
> model-verified characterization is in
> `15-scf-s2a.1-certification-evidence-correction.md`.

All rows below use the Python 3.14 / PyTorch 2.14.1 CPU environment unless
otherwise shown. The raw root solver's reported iteration count includes its
initial physical map. Certification adds no outer map. Numerical values are
diagnostic observations, not new references or tolerance requirements.

### GFN1/H2 root-tolerance sweep

| Requested/effective tolerance | Fixed-point L2 / max | Fresh-Fock L2 / max | Electron-count error | Energy / free energy | Maps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-6` | `9.77e-7` / `4.81e-7` | `2.99e-16` / `2.22e-16` | `-6.7e-16` | `-1.058598356724007` / same | 27 |
| `1e-8` | `9.84e-9` / `4.85e-9` | `2.99e-16` / `2.22e-16` | `-6.7e-16` | `-1.058598356724007` / same | 36 |
| `1e-10` | `9.92e-11` / `4.89e-11` | `2.58e-16` / `2.22e-16` | `-4.4e-16` | `-1.058598356724007` / same | 45 |
| `1e-12` | `9.99e-13` / `4.92e-13` | `3.05e-16` / `2.22e-16` | `-6.7e-16` | `-1.058598356724007` / same | 54 |

H2's closed-shell projector is stationary to eigensolver precision throughout
this sweep, so it is not a useful system for estimating a stationarity
tolerance from raw residuals alone.

### GFN2/H2O charge representation, isolated fixture setup

Each tolerance row was rerun in a fresh process from the same prepared initial
guess. This avoids carrying mutable test-fixture setup state between rows.

| Requested/effective tolerance | Fixed-point L2 | Fresh-Fock L2 | Electron-count error | Energy / free energy | Maps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `1e-6` | `6.17e-7` | `4.03e-11` | `0` | `-5.805248705084273` / same | 29 |
| `1e-8` | `6.22e-9` | `4.06e-13` | `0` | `-5.805248705084274` / same | 38 |
| `1e-10` | `6.26e-11` | `4.38e-15` | `0` | `-5.805248705084273` / same | 47 |
| `1e-12` | `6.31e-13` | `1.47e-15` | `0` | `-5.805248705084274` / same | 56 |

At `1e-8`, the three GFN2/H2O working representations converged to the same
total energy (`-5.805248705084274` within the displayed digits):

| Representation | Fixed-point L2 | Fresh-Fock L2 | Fresh-Fock max | Electron-count error | Maps |
| --- | ---: | ---: | ---: | ---: | ---: |
| charge | `6.22e-9` | `4.06e-13` | `1.64e-13` | `0` | 38 |
| potential | `9.14e-9` | `1.12e-10` | `4.54e-11` | `0` | 27 |
| Fock | `6.11e-9` | `6.73e-11` | `2.73e-11` | `0` | 28 |

The fixed-point norms are not interchangeable across representations. Potential
and Fock stationarity agree closely here, while charge reaches a tighter
fresh-Fock residual at the same nominal raw tolerance.

### Additional physical cases

| Case | Model / representation | T / dtype | Requested/effective root tolerance | Fixed-point L2 / max | Stationarity L2 / max | Electron-count error | Energy / free energy | Maps |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GFN2/H2O | charge | 0 K / f64 | `1e-8` | `9.11e-9` / `5.74e-9` | `5.81e-10` / `2.37e-10` | `0` | `-5.1041587994` / same | 41 |
| GFN1/H2+ doublet | charge | 0 K / f64 | `1e-8` | `9.91e-9` / `4.30e-9` | `2.24e-16` / `1.11e-16` | `-3.3e-16` | `-0.3137104655` / same | 36 |
| stretched GFN1/H2 | charge | 0 K / f64 | `1e-8` | `9.64e-9` / `4.85e-9` | `5.83e-16` / `4.44e-16` | `-1.1e-15` | `-0.9705205335` / same | 37 |
| GFN2/NO2 doublet | charge | 300 K / f64 | `1e-8` | `6.79e-9` / `3.13e-9` | `1.07e-12` / `3.06e-13` | `0` | `-12.4611743257` / same to shown digits | 38 |
| GFN2/NO2 doublet | charge | 300 K / f64 | `1e-12` | `7.50e-13` / `3.65e-13` | `3.56e-15` / `9.99e-16` | `3.6e-15` | `-12.4611743257` / `-12.4611743257` | 57 |
| GFN1/H2 | charge | 300 K / f64 | `1e-8` | `9.84e-9` / `4.85e-9` | `2.10e-16` / `1.11e-16` | `-6.7e-16` | `-1.0585983567` / same | 36 |

The original setup-order-dependence claim is withdrawn. It came from the
unverified model-labeling path described above and was not supported by a
model-identity-controlled comparison. S2a.1 reran alternating setup sequences
with both the corrected test helper and independent fresh `ParamModule`
instances; those results are recorded in the follow-up document.

At 300 K the NO2 entropy correction was about `-1e-15` for these orbital
occupations, so electronic energy and total free energy coincide at displayed
precision. At 1000 K, the test's NO2 open-shell state has a resolvable entropy
correction; the equal and atomic partitions return different per-atom values
but the same total. A separate H2/GFN1 case at 0.1 K, just above the configured
matrix-Fermi smear boundary, has finite certification metrics and valid Fermi
status.

### Float32 floor

For GFN1/H2, a requested `1e-12` raw tolerance resolves through the current
`_resolve_tolerance` rule to `5e-5` in float32 (`50 * finfo(float32).resolution`).
With that effective tolerance the solve converged in 20 maps. Observed values
were fixed-point L2/max `3.49e-5` / `1.72e-5`, stationarity L2/max
`1.36e-7` / `1.19e-7`, and independent electron-count error `-7.15e-7`.
These data do not support requesting float64-style certification from float32.

## 5. Transform and validity checks

The new continuous stationarity metric was exercised under `vmap`, JVP,
`jacfwd`, and a nested second derivative on small GFN1/H2. Value and transform
outputs were finite. Explicit tests cover all three residual representations,
nonorthogonal overlap, AO permutation norm invariance, structural packed-row
padding, Fermi validity versus independent electron count, an actual
out-of-envelope non-finite matrix-Fermi state with an otherwise valid-looking
Fermi status, explicit threshold pass/fail, and representation mismatch
rejection. A loose/tight GFN2/H2O root comparison demonstrates that a loose raw
residual can coexist with materially larger fresh-Fock stationarity.

No certification function accepts a method/history or performs another SCF
map. Fresh-Fock assembly does not diagonalize. The energy helper extraction is
formula-preserving and finalization imports the same helper implementation.
Equal and atomic free-energy partition tests verify the same total.

## 6. Certification overhead

Median CPU timings used two warmups and five alternating measurements in the
Python 3.14 / PyTorch 2.14.1 environment. The raw-map column is one call to the
existing charge map; certification includes fresh potential/Hamiltonian
assembly, metrics, electronic energy, and free energy. No eigenspectrum is
added by certification.

| Case | One raw map | Certification | Ratio |
| --- | ---: | ---: | ---: |
| H2 / GFN1 | 4.33 ms | 1.44 ms | 0.33 |
| H2O / GFN2 | 6.78 ms | 5.45 ms | 0.80 |
| NO2 / GFN2, 300 K | 72.80 ms | 38.94 ms | 0.53 |

The measurement is a single-System diagnostic, not a performance guarantee.
Profiling attributes most of this cost to recomputing the entropy/free-energy
observable: the existing Fermi free-energy trace setup repeats matrix-Fermi
evaluations and chemical-potential correction steps. Certification does not
diagonalize or run another SCF map, but this is still substantial work. S2b
should avoid paying it unconditionally when solver/finalization already has
the same physical state's `fenergy`; pass/reuse that observable through an
explicit physical-observable value or request progress metrics separately.
Keep a direct-from-state diagnostic path and verify it against finalization.

## 7. S2b evidence and recommendation (superseded by S2a.1)

The model-labeled evidence and recommendation below are retained as the
historical S2a record only. Do not use the GFN2/H2O observations or these
threshold suggestions for S2b; see the provenance-checked correction in
`15-scf-s2a.1-certification-evidence-correction.md`.

This evidence is enough to design explicit profiles, but not enough to publish
universal tolerance defaults.

- **Standard:** retain the existing representation-specific raw residual
  tolerances as the initial candidate. At nominal `1e-8`, sampled GFN2/H2O and
  NO2 charge roots had fresh-Fock L2 below `6e-10`, while the H2O potential
  example was around `1e-10`; the Fock example was around `7e-11`. A stationarity
  ceiling near `1e-8` appears conservative for these examples, but more GFN1
  open-shell and charged GFN2 cases are needed before adopting it.
- **Tight:** `1e-10` raw charge tolerance produced H2O fresh-Fock L2 around
  `4e-15`; `1e-12` gave NO2 around `4e-15`. These support a candidate
  stationarity threshold at or above the float64 noise floor, but do not
  establish a single threshold across AO dimension and representation.
- **Higher-order/reference:** the accepted E3.1 NO2 response evidence requires
  approximately `1e-12` raw root tolerance for the tested third-order response;
  the S2a NO2 stationarity at that root was about `4e-15`. Preserve the explicit
  higher-order profile idea and do not infer it from autodiff state.
- **Electron count:** observed errors were at most `4e-15` in the sampled
  float64 cases and about `7e-7` in float32 under its effective raw tolerance
  floor. A dtype-aware threshold is needed; this sample is too small to choose
  its multiplier safely.

Before freezing S2b profiles, add at least one charged/open-shell GFN2 case,
one larger AO system with padding, and repeated float32 finite-temperature
evidence. Tolerances must remain explicit and representation-aware.

## 8. Explicit non-goals

S2a does not make `certified` authoritative. It does not change controller
masks, stopping, iterations, maxiter behavior, convergence tolerances, public
status/failure codes, or implicit-root preparation. It adds no response map or
spectral diagonalization. The known inherited Anderson strict-parity failure
is not an S2a target.
