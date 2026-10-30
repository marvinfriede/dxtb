# SCF S7 — functional authority and xitorch retirement

## Scope and source inventory

S7 started from SCF `745ea14bb06e786d91feb7b044e05fbc38d5e9e8`, already
synchronized to general-refactor `7b946a21fcd27807127d54b4262b5ec5177d607d`.
No later general-refactor commits were imported.

A fresh `rg` audit found production xitorch consumers in the old implicit root
solver and `BaseSCF` type annotation, plus the GFN0 padded generalized
_eigensolver. The iterator and eager adapter selected legacy implicit, FULL,
and experimental engines for compatibility cases. S7 removed those implicit
and single-shot paths, removed override-driven legacy selection, migrated GFN0
to ordinary tensor eigensolver results, then deleted the vendored package.

| Route or consumer | S7 disposition |
| --- | --- |
| Eligible iterative FULL | Functional `SCFProblem` / `SCFStrategy` / shared controller with S2 certification |
| Eligible iterative IMPLICIT | Functional primal, exact selected-root validation, functional IFT attachment |
| Public packed IMPLICIT | Existing tensor-batched implicit behavior retained through the xitorch-free functional response; distinct from W2c9's internal converged-root conformer API |
| Experimental single-shot | Explicit `NotImplementedError`; derivative path was not exact |
| GFN0 / non-self-consistent | Retained one generalized eigensolve, now through `storch.eighb` |
| Public packed FULL batch | Existing unrolled compatibility route retained pending explicit public batch-policy work; it imports no xitorch |
| W2c9 converged-root batch implicit | Retained unchanged and remains on functional solver/implicit machinery |
| Legacy engine `args` / `kwargs` and option dictionaries | Explicitly rejected; no engine fallback |
| Unsupported custom interaction/classical extensions | Explicit `NotImplementedError` naming unsupported interaction extensions or classical extension category |
| Invalid negative temperature/Fermi budget/outer budget | Explicit configuration errors |
| Missing hcore/overlap | Explicit setup error |

## Implicit final representation ownership

The initial working representation still comes from `config.scp_mode`. After
primal solve, final-spectrum validation, adapter preparation, exact raw-map
revalidation, attached-map evaluation, and finalization now use
`outcome.final_problem.representation`, `.inputs`, and
`.potential_layout`. A switched compact-to-Fock primal is therefore validated
and differentiated as a Fock root without mutating the original configuration.

The representation-to-label conversion is centralized in
`_representation_label`. S6's physical root/map pairing and no-extra-map
switch conversion remain unchanged.

## GFN0 generalized eigensolver

The old GFN0 path wrapped padded Hamiltonian and overlap matrices in xitorch
`LinearOperator`s and called `lsymeig` after shifting ghost diagonal entries.
The retained path now calls `storch.eighb` on ordinary tensors with
`aux=True, sort_out=True`. That solver appends ghost orbitals and reports their
eigenvalues as zero, preserving the padded orbital ordering needed by
occupation masking; the artificial `PADDING_SHIFT` was removed. `get_overlap`
and its operator wrapper were deleted. The pure GFN0 density, Mulliken charge,
energy, and free-energy calculations remain the same.

Derivative scope around exact orbital degeneracies remains the existing
`storch.eighb` contract; S7 does not add a stronger promise.

## Retained compatibility modules

The only production legacy iterative owner is the public packed FULL batch
route. `SelfConsistentFieldFull` uses `BaseTSCF` and `BaseSCF`, and its
controller uses the legacy `Simple`/`Anderson` mixer implementations for
batch culling and restoration. This route was kept so S7 does not silently
change public packed-batch semantics. It has no xitorch dependency. W2c9's
internal converged-root batch implementation remains separate and functional.

`scf/pure` remains only where consumed by retained packed FULL or GFN0:
`data.py`, `conversions.py`, `iterations.py`, and `energies.py` remain; the old
`ovlp_diag` operator API was replaced by the tensor-only GFN0 diagonalizer.
The implicit class hierarchy, experimental single-shot class, and unused
legacy Broyden mixer were deleted. No public ConfigSCF defaults or S2
certification definitions changed.

Public packed IMPLICIT remains supported through the tensor-batched functional
response because no approval to retire that API was present. Its existing
zero-gradient-lane regression passes on both the S6 base and S7: the active
lane has a finite nonzero derivative, the inactive lane derivative is zero,
and no dense-solve warning is emitted. The public route does not call the
W2c9 internal conformer orchestrator. Public packed FULL remains at its
separate, xitorch-free unrolled compatibility boundary.

## Architecture checks

S7 removes the lazy xitorch export, coverage omit entry, vendored source tree,
old implicit module tree, and examples that depended on those implementations.
Architecture tests assert the vendored path is absent, no production SCF
source imports xitorch or names the retired implicit/single-shot engines, and
iterator/singlepoint have no `solve_legacy` / `force_legacy_scf` selection
hook.

## Validation

The main SCF selection passed **490 tests** with the one accepted inherited
Anderson strict gradient-parity failure. The value remains
`2.5007065862503453e-11` against `2e-13`; no Anderson arithmetic or tolerance
changed. This selection included S2 certification, lifecycle/E6a.2, native
Anderson/Broyden/CDIIS, S6 strategies, implicit roots, and GFN0. The exact
Phase-A groups on Python 3.14 / PyTorch 2.14.1 completed as follows:

| Group | Result |
| --- | ---: |
| 1: functional map/solver/Anderson/eager compatibility | 214 passed, 1 inherited failure |
| 2: implicit root and root derivatives (`not slow`) | 59 passed |
| 3: public status, spectral density, free energy | 128 passed |

Group 1's accepted S6 run was 228 passed plus the same inherited failure. A
node-ID collection diff against S6's exact `745ea14` base explains the count
change: 30 legacy-ownership tests were removed or replaced and 16 tests were
added for the explicit S7 policies, for a net collection decrease of 14. No
S2/S3-S6 controller or numerical-method tests were removed. The complete
node-ID change is recorded below.

The broader E6bcompat/E6bimpl/E7 plus functional-SCF memory selection passed
**204 tests** after updating the E7 free-energy trace spy to its current
`functional.energy` owner. The GFN0/non-self-consistent and route smoke passed
**21 tests**. On Python 3.11 / PyTorch 2.8.0, architecture, GFN0 position
gradient, functional FULL, switched implicit final-representation, and W2c9
batch-implicit controls passed **27 tests**. The earlier W2c9-only Python 3.14
baseline passed **21 tests**.

The functional-SCF memory checks now warm up the implementation and test for
continued tensor-storage growth across repeated calls. This avoids treating a
small fixed set of framework cache tensors as an accumulating leak. All 16
functional implicit memory variants passed. The old xitorch-specific cases
were removed with their implementation; they are not reported as fixed.

Static checks passed: `compileall`, Black, isort, and `git diff --check`.
`rg -n 'xitorch' src/dxtb` returned no matches, and the vendored directory is
absent. No S2 certification definition or S3-S6 method mathematics changed.
The deleted vendored tree contained 36 Python files (about 234 kB of source).
A single primary-environment `import dxtb` measurement took 3.08 seconds;
there is no before/after benchmark, so this is recorded only as an observation.

### Phase-A Group 1 node accounting

The exact command was the four-file selection in the table above. Relative to
the S6 base, the 30 removed node IDs were:

```text
test_arbitrary_full_kwargs_preserve_legacy_typeerror[adapter-solver_options]
test_arbitrary_full_kwargs_preserve_legacy_typeerror[adapter-unexpected_option]
test_arbitrary_full_kwargs_preserve_legacy_typeerror[production-solver_options]
test_arbitrary_full_kwargs_preserve_legacy_typeerror[production-unexpected_option]
test_batched_numbers_remain_legacy_owned
test_batched_production_dispatch_uses_legacy_engine
test_core_index_owner_keyword_remains_legacy_owned[None]
test_core_index_owner_keyword_remains_legacy_owned[object]
test_core_index_owner_keyword_remains_legacy_owned_for_implicit[None]
test_core_index_owner_keyword_remains_legacy_owned_for_implicit[object]
test_core_index_remains_the_preexisting_internal_keyword
test_custom_interaction_routes_to_legacy_and_keeps_extension_behavior
test_full_engine_overrides_remain_legacy_owned[bck_options]
test_full_engine_overrides_remain_legacy_owned[eigen_options]
test_full_engine_overrides_remain_legacy_owned[fwd_options]
test_full_engine_overrides_remain_legacy_owned[positional]
test_full_engine_overrides_remain_legacy_owned[solver_options]
test_full_engine_overrides_remain_legacy_owned[unexpected_option]
test_legacy_engine_options_are_retained_by_full_engine
test_legacy_owned_routes_precede_negative_scf_maxiter
test_missing_integrals_remain_legacy_owned[hcore-No core Hamiltonian provided]
test_missing_integrals_remain_legacy_owned[overlap-No Overlap provided]
test_negative_fermi_maxiter_is_legacy_owned
test_negative_scf_maxiter_invalid_spin_precedes_e0_error
test_negative_temperature_uses_legacy_eager_behavior[default-broyden]
test_negative_temperature_uses_legacy_eager_behavior[simple]
test_non_full_modes_remain_legacy_owned[experimental]
test_non_full_modes_remain_legacy_owned[implicit]
test_nonselfconsistent_legacy_kwargs_are_forwarded
test_nonselfconsistent_method_remains_legacy_owned
```

The 16 added node IDs were:

```text
test_custom_interaction_extension_is_explicitly_unsupported
test_experimental_mode_has_a_specific_retirement_error
test_fixed_trip_experimental_mode_has_the_same_retirement_error
test_gfn0_remains_a_nonselfconsistent_special_case
test_invalid_temperature_and_fermi_budget_are_explicit_errors
test_legacy_engine_overrides_are_retired[bck_options]
test_legacy_engine_overrides_are_retired[eigen_options]
test_legacy_engine_overrides_are_retired[fwd_options]
test_legacy_engine_overrides_are_retired[unexpected_option]
test_legacy_engine_positional_override_is_retired
test_missing_integrals_are_explicit_setup_errors[hcore]
test_missing_integrals_are_explicit_setup_errors[overlap]
test_negative_scf_maxiter_invalid_spin_is_rejected_explicitly
test_nonfull_route_resolution_rejects_wrong_mode
test_packed_full_batch_is_an_explicit_compatibility_boundary
test_single_system_full_does_not_instantiate_legacy_engine
```

Every removed node exercised a retired override/fallback, prior legacy route
selection, or the old FULL batch dispatch boundary. Replacements explicitly
assert the retirement error, preserve the packed-batch compatibility boundary,
or verify functional single-System routing. Core-index ownership is covered
by the unchanged trusted-core integration tests outside this four-file
selection. No coverage was silently dropped.
