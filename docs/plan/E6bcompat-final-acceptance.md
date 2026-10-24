# E6bcompat — Differential Acceptance

## Decision

**E6bcompat: COMPLETE. E6broute: BLOCKED.** The frozen behavior has been
classified and differentially checked without changing public Calculator
routing. Mapped PyTorch and retained legacy composition remain distinct
execution families. The exact public packed Result adaptation is specified
below; its production implementation belongs to E6broute. E6broute must also
wait for the independent E7 batch-safety and E6c1-W2/W3 performance gates.

Starting SHA: `75e78f251faf1da16eebff1016e1e0b768474948`.
No production source or dependency was changed. `AGENTS.md` remains untouched
and unstaged. The seven E6bimpl.3 changed files were confirmed with
`git diff --name-status 9a9864b4aa6a19a0fde6f0d21b8bbc24d93bb488
75e78f251faf1da16eebff1016e1e0b768474948`; they match the predecessor
report. The implementation commits inspected were `0a0352dda55c56ef36000c294269dfb15a95bb55`
and `75e78f251faf1da16eebff1016e1e0b768474948`.

The exact seven-file status list was:

```text
M docs/plan/07-current-state-and-execution-plan.md
A docs/plan/E6bimpl3-retained-legacy-composition.md
A src/dxtb/_src/calculators/legacy_composition.py
M src/dxtb/_src/calculators/singlepoint.py
M src/dxtb/_src/scf/__init__.py
M src/dxtb/_src/scf/iterator.py
A test/test_calculator/test_e6bimpl3_legacy_composition.py
```

## Frozen policy acceptance matrix

Every substantive row in `E6bcompat-policy.md` is assigned below. Test node
IDs are in [`E6bcompat-acceptance-manifest.txt`](E6bcompat-acceptance-manifest.txt).
The 64-node manifest SHA-256 is
`9b260ae47835e78b3a30b8f11f03e21b99fb9ec8e6a0d97e846efd54d2744836`.
“Accepted” means the existing owner was independently exercised; it does not
mean public dispatch was migrated.

| Policy row | Class | Evidence / acceptance | Remaining owner or limit |
|---|---|---|---|
| GFN1 mode 1, equal-size PyTorch | `MAPPED_PYTORCH` | Public packed H2/H2 compared with private mapped output and independent serial Systems; per-atom and reduced energy shapes checked. | Public route remains legacy pending E6broute. |
| GFN1 mode 1, different-size padded systems | `MAPPED_PYTORCH` | H2/H3+, H2/H2O, reversed H2O/H2; physical atom/AO blocks, classical terms, energies and ordering compared. | Public façade is not the canonical C8b mapper. |
| GFN1 mode 2 conformers | `MAPPED_PYTORCH` | H2 conformers compared against serial and mapped results; energy and position gradients match, with nonzero response. | Public mode 2 stays on legacy dispatch. |
| Public modes 1 and 2 packed API | `MAPPED_PYTORCH` | Both packed shapes exercised while existing Calculator route remains unchanged. | Route selection/retirement belongs to E6broute. |
| GFN2 packed SCP behavior | `RETAINED_LEGACY` | Fock, charge and potential; strict/non-strict; linear/Anderson. Non-strict warning and Fock switch inspected; strict rejection and direct Fock control checked. GFN2 packed Fock H2/H2O compared with mapped and serial results. | Migration to broader private SCP behavior requires an explicit E6broute decision. |
| C8b registered padded cores | `SEPARATE_ROUTE` | Existing C8b and E6bimpl canonical tests remain the authority; Calculator does not dispatch these objects. | No route substitution inferred from structural acceptance. |
| PyTorch packed integrals/H0 | `MAPPED_PYTORCH` | Accepted E6bimpl.0/2a parity tests compare physical matrices and supported multipoles. | Legacy builder retirement is E6broute/E2-legacy work. |
| CPU libcint composition | `RETAINED_LEGACY` | Actual tad-libcint calculations compare independent single-System and ordered composition results, including fields and derivatives. | Whole-System libcint `vmap` is not claimed. |
| Explicit unavailable libcint | `EXPLICITLY_REJECTED` | Fresh-process simulated absence with explicit string `"libcint"` produces the established clear configuration error. | No dedicated no-libcint installation profile was available. |
| Default integer libcint selector when absent | `RETAINED_LEGACY` | The fresh-process simulated-absence test verifies integer default `0` warns and falls back, while explicit string request rejects. | Preserve this distinction unless E6broute approves a migration. |
| Mixed logical/physical libcint device | `RETAINED_LEGACY` | Existing force-CPU ownership and CPU behavior retained. | CUDA/libcint behavior not claimed; CPU-only validation. |
| Built-in GFN1/GFN2 components | `MAPPED_PYTORCH` | E6bimpl differential tests cover H0, ES terms, classical terms, GFN2 AES2/D4SC/D4, labels and physical partitions. | Only the accepted built-in configurations are mapped. |
| Custom interactions | `RETAINED_LEGACY` | Nonzero custom interaction, setup/cache hooks, energy/potential and gradients preserved by per-System composer. | Arbitrary custom transforms are not promised. |
| Custom classical components | `RETAINED_LEGACY` | Nonzero named atom-resolved contribution, total energy, gradient, repeat and ordering checks. | Heterogeneous labels remain separate per-System schemas. |
| Geometry-dependent legacy caches | `RETAINED_LEGACY` | Adversarial cache data, repeated calls and reversed order establish per-System refresh/isolation. | Cache policy remains extension-owned. |
| Direct low-level `unexpected_option` | `RETAINED_LEGACY` | Actual batched `scf.iterator.solve()` reaches legacy constructor; exception type and keyword preserved. | No Calculator engine-option API is added. |
| Direct low-level `solver_options` | `RETAINED_LEGACY` | Same constructor/error path verified. | None. |
| Other direct low-level unknown kwargs | `RETAINED_LEGACY` | Multiple distinct unknown names preserved to constructor and error. | None. |
| Direct positional engine extra | `RETAINED_LEGACY` | Identity reaches constructor; unsupported arity retains TypeError. | None. |
| `fwd_options` | `RETAINED_LEGACY` | Identity, mapping validation, merge precedence and effective legacy configuration checked. | Functional equivalence is not claimed. |
| `bck_options` | `RETAINED_LEGACY` | Identity, mapping validation and legacy default merge checked. | FULL forward execution does not establish a numerical backprop effect. |
| `eigen_options` | `RETAINED_LEGACY` | Identity, merge and constructor consumption checked. | None. |
| Malformed named option mappings | `RETAINED_LEGACY` | Malformed `fwd_options`, `bck_options`, `eigen_options` retain constructor error category. | None. |
| Negative electronic temperature | `RETAINED_LEGACY` | Legacy route classification and behavior match the frozen policy. | No global normalization of temperature policy. |
| Negative Fermi maxiter | `RETAINED_LEGACY` | Legacy route classification and behavior match the frozen policy. | No conversion or clamp. |
| Negative outer maxiter on eligible single FULL | `MAPPED_PYTORCH` | Existing functional error owner retains its ValueError. | No shared global validator. |
| Negative outer maxiter on legacy-owned batch/extensions | `RETAINED_LEGACY` | Existing route and error/result behavior checked. | Legacy loop semantics remain. |
| Missing H0/overlap | `RETAINED_LEGACY` | Low-level owner raises the policy's ValueError; composer does not fabricate matrices. | None. |
| Malformed/nonfinite matrices | `RETAINED_LEGACY` | Shape and eigensolver failures remain at their original layer; no zero substitute or fallback. | Exact backend error remains library-dependent. |
| Scalar batch charge/spin | `RETAINED_LEGACY` | Default scalar charge, nondefault scalar rejection, spin behavior and per-System tensors covered by policy tests. | No arbitrary scalar broadcasting added. |
| Per-System charge/spin | `MAPPED_PYTORCH` | H3+ and GFN2 charge/spin controls compare physical electron counts and serial values; established E6a/E6bimpl tests add open-shell coverage. | Only supported physical states. |
| Shared field/field-gradient | `MAPPED_PYTORCH` | `(3,)` and `(3,3)` shared inputs, public validation, response gradients and field-label conflicts checked. | Per-System public fields remain unsupported. |
| Classical-only | `SEPARATE_ROUTE` | Classical-only execution preserves zero SCF/fenergy and existing optional fields. | Not routed through mapped FULL. |
| Non-self-consistent methods | `SEPARATE_ROUTE` | Existing early route is retained; no FULL replacement semantics are inferred. | Separate route remains owner. |
| Public unknown Calculator kwargs | `APPROVED_MIGRATION` | Current packed path accepts/ignores unknown extras; tests prove they do not reach SCF. | E6broute owns approved clear `TypeError` migration; do not forward engine options. |

The table rows map to exact tests as follows (full node IDs are listed in the
manifest): rows 1–4 use `test_public_gfn1_mode1_matches_serial_and_private_mapped`,
`test_public_gfn1_mode2_matches_serial_and_private_mapped`, and
`test_public_gfn2_packed_fock_matches_private_and_serial`; row 5 uses
`test_public_packed_gfn2_nonfock_policy_is_preserved` and
`test_public_packed_gfn2_fock_route_has_no_nonfock_warning`; rows 6–7 and 12
use the accepted E6bimpl C8b, integral/H0, component and Result suites named
in the E6bimpl.2c manifest; rows 8–11 use the libcint tests and
`test_missing_libcint_dependency_is_not_replaced_by_pytorch`; rows 13–15 use
`test_custom_interaction_is_retained_and_order_independent`,
`test_custom_classical_cache_is_refreshed_per_composition_call`,
`test_legacy_composition_preserves_custom_classical_position_gradient`, and
`test_heterogeneous_classical_results_keep_per_system_schema`; rows 16–23
use `test_batched_low_level_*`, `test_batched_legacy_named_*`, and
`test_legacy_engine_*` nodes; rows 24–29 use the composer Fermi/maxiter
nodes, `test_batched_missing_integrals_*`,
`test_batched_malformed_integral_shapes_fail_at_their_owner`, and
`test_batched_nonfinite_integrals_keep_eigensolver_error`; rows 30–32 use
`test_public_packed_charge_shape_contract`,
`test_public_packed_field_is_shared_not_per_system`, and
`test_public_shared_fields_match_serial_and_private_mode2`; rows 33–34 use
`test_classical_only_calculation_retains_separate_route` and the accepted
non-self-consistent route controls in `test_functional_eager_compatibility.py`;
row 35 uses `test_public_packed_calculator_does_not_forward_engine_options`.

New public-versus-decoded value assertions use `atol=rtol=2e-9` for fields
and components; the GFN2 total-energy serial control uses `5e-8`, and the
position/field response controls use `2e-8`. Existing E6bimpl and composer
assertions retain their established tolerances. No tolerance was changed.

## Differential results

The added public differential tests compare three paths where applicable:
the unchanged packed Calculator (A), independent physical `System.singlepoint`
calls (B), and the private mapped evaluator (C). Public packed Results are
compared field-by-field with decoded mapped Results for per-atom energy, SCF
and classical partitions, charge/potential, density, H0, overlap,
dipole/quadrupole integrals, orbital energies, occupations, iteration
metadata, ordering, and packed padding tails. In this new public comparison
module, decoded-versus-serial assertions directly compare energy and SCF energy
for GFN1 and energy and density for GFN2. The broader mapped-versus-serial
field parity for the accepted GFN1/GFN2 numerical routes is supplied by the
E6bimpl.0–2c test suites, not inferred from those narrower assertions. The
public packed result keeps
`energy.shape == (B,N)` and `Calculator.energy()` reduces to `(B,)` or `(B,1)`
with `keepdim=True`. Mode 2 and shared field/field-gradient gradients were
also compared with independent mapped/serial derivatives.

GFN1 mode 1 covers equal-size H2/H2, H2/H3+, H2/H2O, reversed H2O/H2, and
H2O/CH4.
Mode 2 covers two H2 conformers. GFN2 packed Fock mode covers H2/H2O and
matches mapped and independent serial physical results. Public GFN2 legacy
charge/potential switching and strict behavior are checked separately across
mixers. GFN2 legacy non-Fock output is intentionally not compared to private
non-Fock output: the public owner switches to Fock or raises by policy.

### Result representation and E6broute adapter contract

The private retained composer returns an ordered tuple of physical per-System
Results, preserving each physical atom/AO dimension, labels, and `None`
fields. The private mapped evaluator returns a registered bucket-sized Result
plus decoded physical Results. The legacy packed public Result is a third
representation. To adapt physical Results to the established public packed
schema, E6broute must, per compatible bucket:

1. scatter physical atom-valued fields into `(B,N)` using source-to-canonical
   atom maps, zeroing only structural padding;
2. scatter square AO matrices into `(B,NAO,NAO)` using explicit inverse AO
   maps, and orbital/shell fields through their corresponding maps;
3. preserve the existing static classical label order, with structural
   padding zeros only for a label actually present in that compatible output
   schema;
4. preserve `None` for optional fields absent from the bucket schema;
5. preserve legacy metadata semantics: packed legacy `status`, `converged`,
   and `residual` are currently `None`, while the private mapped Result has
   tensor-valued per-lane SCFStatus/convergence data. Any migration of that
   public metadata is an explicit E6broute change, not inferred parity.

The current public route tests establish the reference shape and physical
values, and E6bimpl tests establish the private source maps and decoded values.
No public adapter was added or activated in this package.

## Retained composition and source state

The E6bimpl.3 composer was reviewed at source level. Its route classification
is static and deterministic; mapped-eligible PyTorch Systems are rejected by
the composer; each retained lane uses the existing legacy engine and its own
System/cache; custom components remain active; and no broad exception-driven
fallback exists. Public `Calculator` dispatch remains unchanged. The
dedicated composer suite uses actual CPU libcint and validates custom
interaction/classical contributions, repeated calls, reversed ordering,
source IDs, source context on errors, and gradients. The existing suite also
validates source parameter identity and no stale cache reuse.

All five environments report the accepted installed versions:

| Python | PyTorch | tad-mctc | tad-dftd3 | tad-dftd4 | tad-libcint | Device | Focused result |
|---|---|---|---|---|---|---|---:|
| 3.10.22 | 2.6.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 0.3.0 | CPU | 64 passed |
| 3.11.15 | 2.8.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 0.3.0 | CPU | 64 passed |
| 3.12.10 | 2.11.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 0.3.0 | CPU | 64 passed |
| 3.12.10 | 2.13.0+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 0.3.0 | CPU | 64 passed |
| 3.14.8 | 2.14.1+cpu | 0.9.1 | 0.7.0 | 0.8.0 | 0.3.0 | CPU | 64 passed |

The official D3 wheel identity is the already verified PyPI wheel
`tad_dftd3-0.7.0-py3-none-any.whl`, SHA-256
`44bb0ddec5fdce42f4c50f0f8270a52a996390a25db827ef2e370110bd1bde7e`.
The metadata conflict with `tad-mctc==0.9.1` remains the documented
controlled `--no-deps` distribution policy, not a numerical failure. Actual
CPU libcint tests ran in each supported environment. CUDA was not run.

## Inherited failures and limitations

The six known functional-SCF failures remain exactly as classified at the
E6bimpl.2c baseline; this package changed no solver or numerical reference:

1. `test/test_scf/test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[0]`
2. `test/test_scf/test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[1]`
3. `test/test_scf/test_functional_fock.py::test_fock_valid_trajectory_with_invalid_final_spectrum`
4. `test/test_scf/test_functional_potential.py::test_potential_maxiter_zero_tracks_raw_and_final_fermi_status`
   — zero-trip trajectory/final status expectation differences.
5. `test/test_scf/test_functional_fock.py::test_fock_final_free_energy_matches_legacy[300K-atomic]`
   — primary absolute difference `2.634295391673461e-13` versus `2e-13`; passes in earlier profiles.
6. `test/test_scf/test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`
   — primary max absolute position-gradient difference `2.5006982595776606e-11` versus `2e-13`.

All six were reproduced at the E6bimpl.2b-F baseline and remain unrelated to
this test-only package. The earlier 601-node combined run was interrupted and
is not counted as passing. The focused 64-node selection passes in all five
environments. Supplemental primary commands completed
`test_functional_eager_compatibility.py` (102 passed),
`test_nonselfconsistent.py` (10 passed), and
`test_c6_legacy_packed_acceptance.py` (1 passed). Larger combined extension
and field-input selections did not produce pytest summaries before
interruption; neither is counted as passing. Field response is covered by the
64-node differential suite and five-profile E6bimpl.3 field controls; the
prior E6bimpl.3 primary group also reported 48 E6a/status/field-input tests
passing with three existing warnings. No new unexplained failure was found.
Limitations are explicit: no CUDA evidence, no dedicated installed
without-libcint profile, no whole-System libcint transforms, and no public
packed Result adapter/dispatch migration in this package.

## Decisions and handoff

The required independent read-only review found no blocker or major issue.
Its minor finding asked this report to distinguish the new public field-by-
field comparison from broader serial parity inherited from E6bimpl. That
clarification is included above. Requested reviewer: Luna/high; actual model
identity is not exposed by the runtime.

- **E6bcompat — COMPLETE.** Every frozen policy case has a retained,
  mapped, separate-route, explicit-rejection, or approved-migration owner and
  test evidence. Public modes and result semantics are characterized against
  the existing route; private mapped and retained paths are independently
  checked. No public route changed.
- **E6broute — NOT READY.** Implement and test the public packed Result
  adapter described above, then activate only after E7 batch-critical safety
  and E6c1-W2/W3 performance gates pass. Preserve the existing strict GFN2
  Fock switch/warning/rejection and explicitly implement the approved
  unsupported-public-keyword `TypeError` migration without engine forwarding.
- **Independent gates — OPEN.** E7, E6c1-W2 conformer performance, and
  E6c1-W3 mixed-System performance remain separate and unaccepted.

Next work is the E7/E6c1 prerequisites; E6broute remains blocked until all
three prerequisites and its output-adapter work are complete.
