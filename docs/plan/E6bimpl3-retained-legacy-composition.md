# E6bimpl.3 — Retained Legacy Composition

## Decision

**E6bimpl.3: COMPLETE. Overall E6bimpl: COMPLETE at the implementation
boundary. E6bcompat: READY for its independent differential acceptance.**

This package adds a private, ordered composition of independent retained
single-System evaluations. It does not change `Calculator` dispatch, activate
public batching, or replace the accepted mapped PyTorch GFN1/GFN2 path.

Starting SHA: `9a9864b4aa6a19a0fde6f0d21b8bbc24d93bb488`.
Implementation commit: `0a0352dda55c56ef36000c294269dfb15a95bb55`.
The final package SHA is reported in the delivery summary after the
documentation update.

## Implementation and route ownership

The private entrypoint is
`dxtb._src.calculators.legacy_composition.compose_legacy_systems`. It accepts
aligned tuples of independently prepared `System`s, positions, charges and
spins, optional shared field tensors, source IDs, and typed
`LegacySCFOptions`. It returns ordered ordinary physical `Result`s. Different
atom/AO dimensions and optional `None` fields remain intact; no synthetic
padded result schema is created. Source IDs remain metadata, never tensor
indices.

`classify_legacy_system` selects an owner from static setup and extension
metadata before any numerical call:

| Case | Owner |
|---|---|
| Supported exact PyTorch FULL setup with eligible registered terms | MAPPED; the retained composer rejects it |
| Libcint, custom interaction/classical extensions, or explicit low-level engine extras | LEGACY_COMPOSED |
| SCF-excluded or iteration-free calculation | SEPARATE_ROUTE through its existing single-System behavior |
| Packed façade, missing exact setup, or unsupported integral driver | REJECTED |

There is no exception-driven mapped-to-legacy fallback. All lanes are
classified before execution. Numerical errors propagate with their original
exception object and category; the failing source ID is attached as context.

`scf.solve_legacy` shares preparation and numerical code in
`scf.iterator` and forces the existing legacy constructor for retained FULL
calls. It does not duplicate SCF equations. Low-level positional and keyword
extras are passed to that constructor without filtering. Non-SCF behavior
remains on its established separate path.

Each lane uses its source System, component instances, captured setup, and
Model parameter owner. Interaction call data is prepared per evaluation;
custom classical extension setup is refreshed through its existing
`get_cache` hook. No Model, ParamModule, IndexHelper, or mutable System is
created per lane. The accepted mapped route is not modified.

## Numerical and compatibility evidence

The dedicated test module contains 24 tests, all passing in every approved
CPU environment. It exercises:

- Actual PySCF/tad-libcint H2 and H2O calculations, independent serial
  comparisons for energy/H0/overlap, source IDs `(4, 19)`, and reversed order.
- Per-System charge/spin, shared electric field and field gradient, field
  response, and nonzero position response.
- A custom nonzero Interaction with geometry-dependent cache data, preserved
  contribution, ordering, and gradient.
- A custom named Classical term with refreshed setup, atom-resolved energy,
  repeated-call parity, and position-gradient parity.
- Duplicate field-label rejection, mapped-eligible route rejection,
  negative-temperature and negative-Fermi-maxiter route ownership, negative
  outer-maxiter behavior, and classical-only routing.
- Legacy unknown keyword and positional-extra failures, consumed
  `fwd_options`/`bck_options`/`eigen_options`, and numerical exception
  propagation with source context. Supplied option mappings retain value
  identity at the legacy constructor boundary; the instantiated engine
  retains configured overrides and default merging. Malformed named option
  mappings fail with the original constructor error.
- Heterogeneous per-System classical labels remain distinct in returned
  Results, with no fabricated cross-lane classical entry.
- Invalid-geometry error parity against an independent serial call, plus an
  injected exception identity check proving source context does not replace
  the underlying exception.
- A nonzero Model-owned `kpol` gradient compared with an independent serial
  libcint calculation, with parameter identity preserved.

The focused command in each environment was:

```bash
OMP_NUM_THREADS=1 <environment-python> -m pytest -q \\
  -n logical --dist worksteal \\
  test/test_calculator/test_e6bimpl3_legacy_composition.py
```

| Python / Torch | Official D3 / mctc / D4 | Result |
|---|---|---:|
| 3.10.22 / 2.6.0+cpu | 0.7.0 / 0.9.1 / 0.8.0 | 24 passed |
| 3.11.15 / 2.8.0+cpu | 0.7.0 / 0.9.1 / 0.8.0 | 24 passed |
| 3.12.10 / 2.11.0+cpu | 0.7.0 / 0.9.1 / 0.8.0 | 24 passed |
| 3.12.10 / 2.13.0+cpu | 0.7.0 / 0.9.1 / 0.8.0 | 24 passed |
| 3.14.8 / 2.14.1+cpu | 0.7.0 / 0.9.1 / 0.8.0 | 24 passed |

All five environments had `tad-libcint==0.3.0` and a working PySCF
installation; the primary environment used PySCF 2.14.0. Therefore libcint
acceptance is based on real numerical evaluation, not mocks. A no-libcint
profile and CUDA were unavailable/not run; neither is claimed as passed.

Affected regression selections in the primary profile completed:

| Selection | Result |
|---|---:|
| E6bimpl.0, .1a, .2a, .2b plus C8b mixed padding and parameter-tree modules | 184 passed, 11 warnings |
| E6bcompat policy and functional eager compatibility | 124 passed |
| E6a status/transforms and field inputs | 48 passed, 3 existing Torch warnings |

The larger combined run that included the entire libcint integral module did
not produce a pytest summary and was interrupted; it is not counted as a
pass. The dedicated composer tests separately exercised real libcint. The
six inherited functional-SCF failures and interrupted 601-node run remain as
recorded in [E6bimpl2c final acceptance](E6bimpl2c-final-numerical-acceptance.md);
this package changed neither their references nor tolerances. No new failure
was observed in the completed selections.

Black, isort, compileall, and `git diff --check` pass. Ruff and mypy are not
installed in the primary environment. No public Calculator route changed.
The required independent read-only review found no remaining blocker or
major issue after the added forwarding, heterogeneous-label, and error-parity
tests. The requested reviewer was Luna/high; actual model identity is not
exposed by the runtime.

## Frozen policy coverage and remaining boundary

The E6bcompat policy matrix is divided by execution owner:

| Policy case | Implementation owner | Evidence / next owner |
|---|---|---|
| Supported mixed GFN1/GFN2 PyTorch with registered terms | Mapped evaluator `evaluate_padded_systems` | Accepted E6bimpl.0–2c tests; E6bcompat performs differential validation |
| Libcint retained single-System calls | `compose_legacy_systems` and existing libcint/System path | Actual CPU numerical composer tests; E6bcompat repeats policy comparisons |
| Custom interactions/classical terms and cache-dependent extensions | Existing extension methods within per-System legacy composition | Nonzero contribution, cache refresh, order/repeat, and gradient controls; arbitrary extension state remains subject to its existing API limits |
| Non-SCF/classical-only behavior | Existing separate single-System route | Regression controls; no mapped FULL routing |
| Direct low-level FULL extras and legacy FULL options | `LegacySCFOptions` → `solve_legacy` → existing constructor | Unknown/positional failures and named option consumption covered |
| Public packed Calculator modes, output/field/charge compatibility | Existing public Calculator route | E6bcompat validates; E6broute owns activation/migration |
| Unsupported integral driver or malformed/private request | Explicit `REJECTED` or validation error | No silent component loss or exception fallback |
| Optional no-libcint installation behavior | Existing backend import/error owner | No approved no-libcint environment is available; E6bcompat owns that optional-profile check |

The composer returns one source-ordered physical Result per System, so
optional tensors and heterogeneous classical labels are preserved without
padding. Shared fields retain `(3,)` and `(3,3)` semantics through the
existing single-System validation. Public packed charge/spin broadcasting,
legacy GFN2 Fock-selection warnings/strict behavior, and public error policy
remain owned by the unchanged Calculator compatibility route.

The E6bimpl-owned retained implementation obligation is complete. E6bcompat
must now independently compare the mapped and retained routes against their
serial references, validate the frozen error/argument contracts, and verify
optional-backend behavior. This report does not claim that compatibility
validation or public route activation is complete.

## Decisions

- **E6bimpl.3 — COMPLETE.** The retained composition boundary and required
  numerical compatibility controls pass.
- **Overall E6bimpl — COMPLETE.** The accepted private mapped GFN1/GFN2
  implementation and the policy-assigned retained composition both have
  defined, tested owners. The six matched functional-SCF baseline failures
  remain separately tracked, not unexplained regressions from this package.
- **E6bcompat — READY, not complete.** Differential acceptance is the next
  package. Public Calculator routing remains unchanged.
- **E6broute — NOT UNBLOCKED.** It remains gated on E6bcompat, E7 batch
  safety, and E6c1 W2/W3 performance clearance. E4.6 legacy cleanup remains
  separate.
