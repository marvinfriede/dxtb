# E6bcompat — Differential acceptance after M1.1

## Decision

**E6bcompat: ACCEPTED at the frozen compatibility boundary.** The unchanged
packed Calculator, independent serial Systems, private mapped PyTorch path,
and retained legacy composition were evaluated under the upgraded dependency
stack. Public Calculator dispatch remains unchanged.

**E6broute: BLOCKED.** The public packed Result adapter remains route-owned;
batch-critical E7 and E6c1-W2/W3 are independent prerequisites. In
particular, W2 remains open after the M1.1 one-conformer diagnostic timed out.

## Repository and environment

- Starting SHA: `e127ea47d880d53537b7ae726938aa3f0a74f24d`.
- Test-hardening implementation SHA: `1ff17de2ad82055b9d95e8ddb8827119a21a99f1`.
- Final acceptance-documentation SHA: recorded in the delivery summary.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: `origin` → `git@github.com:marvinfriede/dxtb.git`.
- The remote branch exists and resolved to the starting SHA before edits.
- The U1–U4 merge, M1.1 commits, and accepted D3 source are ancestors of
  this starting point. The merge parent is `4546417c4a1174ce9322a0c18014eb552c678f7a`.
- Dependency versions in the tested profiles: mctc 0.11.3, multicharge
  0.8.0, libcint 0.4.0, D3 0.7.0, D4 0.8.0.
- Tested CPU profiles: Python 3.11/Torch 2.8.0, Python 3.12/Torch 2.11.0,
  Python 3.12/Torch 2.13.0, and Python 3.14/Torch 2.14.1. The Python
  3.10/Torch 2.6 environment is below the accepted PyTorch 2.7 floor and was
  not treated as supported.

## Route classification

| Frozen behavior | Owner | Acceptance disposition |
|---|---|---|
| Supported GFN1 mode 1/mode 2 and GFN2 Fock PyTorch | `MAPPED_PYTORCH` | Compared public packed, independent serial, and private mapped results where applicable. |
| Libcint and custom interaction/classical/cache extensions | `RETAINED_LEGACY` | Independent per-System legacy composition and real CPU libcint 0.4 tests passed. |
| Classical-only, non-SCF, and C8b structural-core calls | `SEPARATE_ROUTE` | Existing execution owners remain separate from mapped FULL. |
| Explicit unavailable libcint request; mapped-eligible System passed to the legacy composer | `EXPLICITLY_REJECTED` | Clear existing errors preserved; no silent PyTorch substitution or legacy fallback. |
| Unknown public packed Calculator keyword behavior | `APPROVED_MIGRATION` | Existing ignore/no-engine-forwarding behavior remains characterized. E6broute owns the approved explicit `TypeError` migration. |

The full frozen case inventory remains in
[`E6bcompat-final-acceptance.md`](E6bcompat-final-acceptance.md); its 64-node
manifest is unchanged. The acceptance gates below are the 16 requirements in
`E6bcompat-policy.md`.

## Sixteen requirement acceptance matrix

| # | Requirement | Evidence exercised | Tolerance / contract | Result |
|---:|---|---|---|---|
| 1 | Numerical values | Public GFN1 mode 1/mode 2 and GFN2 Fock differential tests; E6bimpl.0–2b mapped tests | Existing per-test tolerances; no references or tolerances changed | PASS |
| 2 | Complete Result fields | Packed-vs-decoded field assertions; mapped Result/status suites; M1.1 separate energy and free-energy assertions | Public packed `(B,N)` atom-energy shape and established `None` status semantics retained | PASS |
| 3 | Charge and spin | Public charge-shape policy; GFN1 charged/open-shell mapped controls; libcint composer per-System charge/spin test | Per-System physical inputs match independent serial calls; scalar batch contract unchanged | PASS |
| 4 | Masks and structural padding | E6bimpl padded spectral/integral tests; corrected M1.1 finite-temperature H2/H2O test | Inactive atoms/orbitals and padded-coordinate derivatives excluded | PASS |
| 5 | Atom/shell/AO mappings | Canonical source-map and GFN2 canonical tests; reversed public H2O/H2 case | Explicit maps and decoded source order; AO masks remain distinct from sorted spectral prefixes | PASS |
| 6 | Integral and H0 semantics | E6bimpl.0/2a production-kernel tests; actual libcint 0.4 composition | Physical H0/overlap/multipole blocks compared with independent paths at existing tolerances | PASS |
| 7 | No fictitious padded contribution | M1.1 checks padded atom energy/free energy and sorted dummy occupations; existing invalid-pair and padding controls | Structural zeros only; no fabricated physical component | PASS |
| 8 | No cross-System derivatives | E6bimpl mapped Jacobian controls and M1.1 lane Jacobian/HVP checks | Off-diagonal and inactive-coordinate contributions zero under the tested contract | PASS |
| 9 | Position derivatives | Public mode-2/shared-field response; libcint and custom-classical composer gradients; mapped position controls | Existing serial comparison tolerances retained | PASS |
| 10 | Supported parameter derivatives | Model-owned mapped GFN1/GFN2 parameter controls; libcint composition `kpol` gradient | Mapped/legacy gradients connected to their source parameter owner | PASS |
| 11 | Convergence and failure metadata | Public GFN2 strict/warn policy; GFN2 lane-local status; low-level error-policy tests | Legacy packed `None` metadata and functional tensor status remain distinct | PASS |
| 12 | Custom component policy | Nonzero custom interaction and custom classical/cache tests | Subclass, hooks, contribution order, atomwise values, and supported gradients preserved | PASS |
| 13 | Options and errors | Low-level unknown/positional options, named option merge/errors, malformed integrals, explicit backend errors | Original constructor exception categories and public no-forwarding behavior preserved | PASS |
| 14 | Deterministic ordering | Public reversed composition fixture; canonical source IDs; composer source-order and heterogeneous-label tests | Outputs remain aligned with source order | PASS |
| 15 | Source-System immutability | Existing E6bimpl.2c A/B/A and parameter snapshot acceptance; composer repeated/reversed-call/cache tests | No observed source setup or prior Result mutation | PASS |
| 16 | Shared Model parameter ownership | C8b Model parameter-tree controls; mapped component gradients; retained libcint parameter identity/gradient test | No per-System Model parameter table; source owner remains connected | PASS |

The accepted E6bimpl.2c source-immutability and parameter-tree evidence is
reused for unchanged private invariants; it was not duplicated solely to
increase this package's test count.

## Differential and Result evidence

The E6bcompat differential module compares the current public packed
Calculator (A), independent physical `System.singlepoint()` calls (B), and
the private mapped evaluator (C) for supported GFN1/GFN2 cases. It checks
public packed energy shape, physical lane values, classical labels,
charges, density/Hamiltonian/overlap/H0, dipole/quadrupole integrals,
occupations, iterations, ordering, shared fields, and GFN2 warning/strict
behavior. Controller differences remain explicit: public packed legacy SCF
may stop early while private `System.singlepoint()` uses fixed trips. No
controller change was made to force numerical trajectories to match.

The private retained composer is tested separately from mapped execution.
Its current suite exercises actual libcint 0.4 H2/H2O calculations, serial
energy/integral/force comparisons, field responses, Model parameter
gradients, custom interactions and classical terms, geometry-dependent
cache refresh, repeated/reversed ordering, error identity/source context,
and direct low-level option forwarding. No libcint-to-PyTorch fallback is
claimed.

The public packed Result adapter has a proven specification but is not
implemented here. It must scatter physical atom/shell/AO fields through
explicit source maps, preserve configured classical-label identity and
optional `None` fields, and retain legacy packed metadata semantics. That
adapter remains E6broute-owned.

## Test results

All commands used `OMP_NUM_THREADS=1`, two pytest workers, and
`--dist worksteal`.

The 64-node command is the one recorded in
[`E6bcompat-acceptance-manifest.txt`](E6bcompat-acceptance-manifest.txt).
The 119-node mapped selection ran these files:

```text
test/test_calculator/test_e6bimpl0_padded_integrals.py
test/test_calculator/test_e6bimpl1a_padded_spectral.py
test/test_calculator/test_e6bimpl2a_gfn2_canonical.py
test/test_calculator/test_e6bimpl2b_gfn2_scf.py
```

The M1.1 boundary regression was
`test/test_calculator/test_e7a_padding_safety.py::test_finite_temperature_trace_is_padding_safe_through_mixed_gfn1`.
The baseline constrained-environment selection ran the implicit-root,
D3-0.7, libcint, masked-trace, and public-status nodes shown in the
M1.1 report's validation record; it completed with 9 passed.

| Selection | Python/Torch | Result |
|---|---|---:|
| E6bcompat differential, retained composition, and low-level policy modules (64 nodes) | 3.11 / 2.8.0 | 64 passed |
| Same 64-node selection | 3.12 / 2.11.0 | 64 passed |
| Same 64-node selection | 3.12 / 2.13.0 | 64 passed |
| Same 64-node selection | 3.14 / 2.14.1 | 64 passed |
| E6bimpl padded integral/spectral, GFN2 canonical and SCF files (119 nodes) | 3.11 / 2.8.0 | 119 passed |
| Same 119-node selection | 3.12 / 2.11.0 | 119 passed |
| Same 119-node selection | 3.12 / 2.13.0 | 119 passed |
| Same 119-node selection | 3.14 / 2.14.1 | 119 passed |
| Corrected M1.1 mixed finite-temperature trace regression | Each of the four profiles above | 1 passed per profile |
| Baseline constrained install representative execution | 3.11 / 2.14.0 CPU | 9 passed |

The baseline selection included an implicit-root transform, D3 0.7 trainable
`s9=0`, real libcint 0.4 single-System integrals, the noncontiguous masked
trace comparison, and a public Result/status regression. The fresh editable
install used `.[tox,libcint,pyscf] -c constraints/baseline.txt`, then
installed D3/D4 with the documented `--no-deps` exception. `pip check`
reported only the accepted D3/D4 metadata conflicts. The baseline had
pytest 8.4.2 and collected the focused spectral selection successfully.

The M1.1 test now asserts energy and free energy independently and snapshots
the mapped trace-call count before serial reference calls. Occupation checks
document that occupations follow energy-sorted molecular-orbital order and
are not indexed using a potentially noncontiguous canonical AO mask.

## Inherited limitations and independent gates

- M1.1's one-conformer/one-trip tmpda mapped W2 diagnostic timed out at
  300 seconds. W2 is **OPEN**; this is not evidence that the trace change
  cleared W2 or that the old quadrature remains the active bottleneck.
- E6c1-W3 remains **OPEN**.
- Broader E7 remains separate from the tested batch-critical padding gate.
- No public Calculator route changed or activated.
- No CUDA or no-libcint installation profile is claimed.
- No full general suite was repeated; the focused acceptance files and
  previously accepted E6bimpl.2c immutability/Model-tree evidence define this
  bounded package.

## Independent review and verdicts

An independent read-only source review by `/root/e6bcompat_review` found no
blocking or nonblocking issue. It confirmed the trace counter is captured
before serial evaluation, separate energy/free-energy assertions preserve the
existing tolerances, occupation indexing is correctly distinguished from AO
masking, and the report keeps the public Result adapter and route gates open.
No public Calculator routing or numerical production implementation changed;
the only source change in this package is test hardening.

- **E6bcompat: ACCEPTED.**
- **Private mapped GFN1/GFN2: VERIFIED** for the accepted configurations and
  tested CPU profiles.
- **Retained libcint/custom composition: VERIFIED** with real libcint 0.4
  calculations in the focused suites.
- **M1.1 test hardening: COMPLETE.**
- **Baseline constrained installation and execution: VERIFIED** with only
  accepted D3/D4 metadata conflicts.
- **E6c1-W2: OPEN. E6c1-W3: OPEN.**
- **E6broute: BLOCKED** on its public Result adapter and the independent E7
  and E6c1 gates.
- **Public Calculator routing: UNCHANGED.**
- **Public release readiness: BLOCKED** by separate release/dependency gates.
- **Remote publication: pushed and verified on `origin/claude/tender-dijkstra-cda42s`;** the exact published SHA is in the delivery summary.
