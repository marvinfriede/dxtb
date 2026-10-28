# SCF S0 audit and implementation baseline

## 1. Review point

S0 was performed on `scf/redesign` at kickoff commit
`1794ab895d007c0dceee7c1cee99f1edfe4c9129`, based on general-refactor
commit `f8297423a363305d0233307bc92d1c6c947b63a8`. The branch was pushed to
`origin/scf/redesign`. The general-refactor worktree was not changed. Its
untracked `AGENTS.md` remains present.

The architectural target is [08-scf-redesign-overview.md](08-scf-redesign-overview.md).
The existing physical map, accepted E3.1 root boundary, functional core, and
compatibility behavior are the baseline to migrate. This audit proposes
structure only; it does not begin S1.

## 2. Environment and dependency state

Both environments use Python 3.14.8, PyTorch 2.14.1+cpu, tad-mctc 0.11.3,
tad-libcint 0.4.0, and tad-multicharge 0.8.0. The SCF environment imports
`dxtb` from `dxtb-scf/src/dxtb`. The source and SCF environments also have
tad-dftd3 0.6.0 and tad-dftd4 0.8.0; direct imports of these packages fail
against tad-mctc 0.11.3 (`coordination_number` and `tad_mctc.typing.Any` are
missing). This is the already documented temporary dispersion compatibility
arrangement, outside S0's repair scope.

The SCF code imports `AtomCGTOBasis` and `CGTOBasis` from
`tad_libcint.basis` (`src/dxtb/_src/exlibs/libcint/__init__.py:25`), then
expects `tad_libcint.interface.integrals.int1e`/`overlap` and
`interface.wrapper.LibcintWrapper`. Installed tad-libcint 0.4.0 does not
export `AtomCGTOBasis` from `tad_libcint.basis`; the import fails before
integral setup. The same failure reproduces in the general-refactor checkout
and configured environment. No package version was changed.

## 3. Executable baseline

Commands were run in the SCF worktree with all four requested thread
variables set to one and `pytest -n 1`. Raw outputs are outside the repository
at `/tmp1/friede/dxtb-agentic/scf-s0/`.

| Group | Result | Warning |
| --- | --- | --- |
| functional map, solver, Anderson, eager compatibility | 195 passed, 30 failed | `torch.jit.script` warns that Python 3.14+ support may change |
| implicit root and root derivatives, `not slow` | 13 passed, 40 failed | Same Python 3.14 JIT warning |
| public status and invariant spectral/free-energy | 88 passed | Same Python 3.14 JIT warning |

The three commands and exact selections are recorded in the raw logs named
`functional-baseline.txt`, `implicit-baseline.txt`, and
`invariant-baseline.txt`. The control group is green; the JIT message is a
warning, not a test failure.

## 4. Failure classification

### F1 — finite-temperature and parity checks stricter than observed numerical agreement (29 tests)

Twenty-seven of the 30 first-group failures are tensor parity assertions
between the functional route and the legacy physical/solver route. The common
assertion is `torch.testing.assert_close(..., rtol=0.0, atol=2e-13)` in
`test/test_scf/test_functional_map.py:238`. Cases span finite-temperature raw
maps, simple/Anderson trajectories, Broyden adapter parity, and short-budget
representation parity. The representative H2O/GFN1/300 K raw-map mismatch is
`3.205213872092827e-13`; other observed map deltas reach `1.2491e-11`, and the
LiH fractional-temperature row reaches `3.4997e-10`. These are float64 CPU
runs. One first-order energy/gradient test is also grouped here only insofar
as it compares the two solver paths; it needs its own interpretation in S1's
test review. The baseline does not establish that every parity failure is the
same low-order delta, so the raw node list is retained for follow-up.

The test helper explicitly sets zero relative tolerance and a `2e-13`
absolute threshold. This is a solver/legacy parity regression bound, not a
published physical accuracy threshold. No energy, force, or independent
physical finite-difference bound is asserted by these failing parity checks.
The representative test was rerun at the exact general-refactor checkout and
failed with the same tensor values and tolerance. Thus the discrepancy is
inherited at this code/dependency point. Keep these tests strict until a
focused test-design decision separates parity checks from physical-equivalence
checks; do not loosen them merely to clear the audit.

Affected nodes are listed in the raw log. Main families are:

- six parameterizations of `test_finite_temperature_raw_map_matches_full_engine`;
- `test_potential_map_matches_legacy_physical_evaluation` for the fractional,
  300 K case;
- eleven finite-temperature simple solver/budget/convergence checks;
- three Anderson gradient/free-energy/converged-parity checks;
- two default Broyden adapter parity checks;
- six 300 K short-budget representation/mixer parity checks.

### F2 — damping warning text assertion (1 test)

`test_float32_damping_warning_uses_configured_decimal` fails in the eager
compatibility file. This is an output-contract assertion rather than a
physical-root failure. It is inherited from the kickoff base; its expected
versus actual warning is present in the raw functional log. Owner: later
compatibility/status cleanup, after routing decisions; no warning text changed
in S0.

### F3 — libcint API import incompatibility (40 tests)

All 40 failures in the non-slow implicit/root group fail during case setup at
the same missing `tad_libcint.basis.AtomCGTOBasis` import. Thirteen other tests
in that selection pass without reaching the incompatible setup. The traceback
shows no independent failure after the import point among the failed nodes;
those assertions were not executed and remain unclassified at runtime, rather
than being claimed to pass. A representative node was rerun in the general
checkout and fails identically. Owner: the separately coordinated dependency
integration point, not S0 or the SCF redesign.

## 5. Functional SCF inventory

| Path / symbols | Role and layer | Planned fate / owner |
| --- | --- | --- |
| `functional/data.py`: `SCFInputs`, `SCFMapResult`, `SCFSpectralResult`, charge/potential layouts and Fermi settings | Immutable call inputs and shared physical result values | Retain physical tensor/setup contracts; S1 maps them into explicit problem/state ownership without cloning the physics |
| `functional/core.py`: `CoreIndexData`, `FunctionalInteractionSet` | Frozen single-System index and built-in interaction snapshots | Retain as physical setup values; broader interaction ownership remains with general refactor |
| `functional/map.py`: `scf_map`, `potential_map`, `fock_map`, conversion/initial-guess helpers | Physical raw map and representation conversions | Retain as one physical map backend; S1 adapter consumes it, S2 defines shared residual/certification |
| `functional/spectral.py`: density/free-energy matrix functions and spectral status | Physical spectral evaluation, occupations, derivatives | Retain; do not fork a second spectral implementation |
| `functional/interactions.py`, `functional/padded.py` | Physical built-in interaction capture and padded batch compatibility | Retain existing role; no broad batching redesign in S1 |
| `functional/solver.py`: `SCFSolveResult`, simple solver step/loop, finalization | Functional linear/simple method plus method-local convergence | Split method stepping from controller/convergence over S1-S3; preserve current accepted route until parity contract is explicit |
| `functional/anderson.py`: `AndersonSettings`, `AndersonState`, mixing and solver entry points | Functional Anderson history and method | Adapt as second method after the minimum linear consumer; S3 |
| `functional/eager.py`: `resolve_full_scf_route`, status conversion, full eager, fixed-trip and prepared implicit routes | Eager routing, compatibility adapter, controller-like loops and public status | S1 introduces a narrow adapter; preserve legacy routes until later migration/retirement packages |
| `functional/implicit.py`: `_PreparedSCFRoot`, `_SCFMapTensorTree`, `implicit_root`, matrix-free solve | Functional prepared-root attachment and derivative solve | Preserve E3.1 boundary and reuse; S1 must not move differentiation inside method state |
| `result.py`, `status.py` | Public SCF fields and status/failure codes | Retain public result contract initially; S2 coordinates common certification diagnostics |
| `singlepoint.py` and calculator `scf.solve` call boundary | Calculator construction, integration setup, SCF routing, result snapshot | Keep shared Model/System/Result refactor ownership; change only the smallest SCF call adapter when coordinated |

## 6. Legacy SCF inventory

| Path / symbols | Role and layer | Planned fate / owner |
| --- | --- | --- |
| `scf/base.py`: `BaseSCF`, old orbital/eigenproblem and integrals access | Legacy physical bridge and shared class hierarchy; xitorch overlap operator | Keep as compatibility boundary during migration; retire in S7 after all consumers are named |
| `scf/pure/{data,conversions,iterations,energies,ovlp_diag}.py` | Legacy physical map/conversions, energy and eigenproblem | Compare against functional physical kernel; migrate useful equations, not object/cache structure. `ovlp_diag` eigen route is a separate xitorch consumer |
| `scf/iterator.py`, `nonselfconsistent.py` | Legacy iteration/status and GFN0 path; iterator retains old xitorch eigen compatibility | Preserve GFN0 semantics; S1/S3 adapter, S7 purge only when all methods routed |
| `scf/mixer/{base,simple,anderson,broyden}.py` | Mutable legacy mixer hierarchy and histories | Replace as solver engine in S3/S4; Broyden remains a required method, not a removal target |
| `scf/unrolling/{base,default,singleshot}.py` | FULL/autograd trajectory and single-shot gradient routes | S3 adapter first; define migration/retirement explicitly in S7/S11 after transform tests |
| `scf/implicit/{base,default,fixed_point}.py` | Legacy xitorch root attachment and custom adjoint/implicit solve | Keep fallback until functional eligibility is complete; retire only after E3.1 consumers and Broyden are covered, S7 |
| `exlibs/xitorch/**` | Vendored operator, eigensolver, root solver, linear algebra, transforms | Delete only in S7 after zero production consumers and architecture guard tests |

## 7. xitorch consumer inventory

Search covered `src`, `test`, `examples`, and `docs`. Production consumer
families, excluding the vendored implementation itself, are:

| Consumer | Present purpose | Replacement / owner |
| --- | --- | --- |
| `scf/mixer/broyden.py` | Legacy Broyden calls vendored root solver | Native functional Broyden while retaining Broyden behavior; S4 |
| `scf/implicit/fixed_point.py`, `implicit/default.py` | Legacy forward root acquisition, method table and custom implicit adjoint | Functional prepared-root attachment already exists for eligible calls; finish fallback coverage then retire in S7/S11 |
| `scf/base.py`, `scf/implicit/base.py` | `LinearOperator` overlap/eigensolver API types | Replace with tensor/standard torch linear algebra where needed; S7 |
| `scf/pure/ovlp_diag.py`, plus iterator compatibility use | Generalized symmetric eigensolve for legacy path | Existing functional spectral route uses invariant matrix functions and ordinary reporting eigensystem; legacy consumer retires with old engine, S7 |
| `exlibs/__init__.py` and vendored package internals | Optional dependency exposure and internal support | Remove with last production consumer; S7 |

There are no non-SCF production consumers identified. Remaining matches are
vendored implementation text, tests, documentation/example (`examples/limitation_xitorch.py`),
exception wording, and lint/coverage configuration. They should be classified
again before deletion. The generalized eigensolver should not be replaced by a
new xitorch-shaped abstraction.

## 8. Functional-to-legacy fallback matrix

Current routing is primarily in `functional/eager.py::resolve_full_scf_route`
and `singlepoint.py`:

| Condition | Why / current behavior | Planned action and owner |
| --- | --- | --- |
| `scf_mode != FULL` for this full-functional entry | implicit/experimental routes have separate owners | Migrate eligible implicit contract without merging derivative policy; S1/S3 |
| Non-SCF method (GFN0) | no iterative SCF | Explicitly retain separate non-self-consistent physical route |
| `IndexHelper.batch_mode != 0` or batched numbers | old engine owns current batched compatibility | Replace only at relevant batch package; S12 coordination |
| Legacy engine `args`/`kwargs` supplied or `force_legacy_scf` | extension/backward-compatibility API | Inventory consumers; explicit adapter or retirement in S7 |
| Interaction not in exact supported built-in functional tuple | custom extension preserves old hooks | Migrate supported interactions or explicitly retire; scope per interaction, S7 |
| Negative electronic temperature or negative Fermi maxiter | preserves legacy semantics | decide/document as supported input or reject; S2/S7, no silent route |
| Missing hcore or overlap | functional map requires both matrices | resolve setup contract; S1 adapter, no separate physics fallback long-term |
| Broyden selected with FULL | `strict=False` warns and changes effective mixer to Anderson; `strict=True` raises | silent numerical substitution is prohibited by target; S4 or explicit error until Broyden exists |
| `mixer` outside linear/Anderson/Broyden labels | errors | keep fail-closed validation |
| implicit default/Broyden and legacy overrides | current E3.1 compatibility keeps legacy path; global Broyden is ignored by legacy implicit engine in favor of its own simple forward behavior | preserve accepted E3.1 contract; native Broyden and explicit method routing S4/S7 |
| fixed-trip ineligible System route | delegates to legacy through `run_system_fixed_trip` | document fixed-trip support limits; S3/S12 |

The principal silent substitution is FULL+Broyden with `strict=False`: it
executes Anderson after warning. The legacy unrolling constructor has the same
Broyden-to-Anderson fallback for non-implicit use. Legacy implicit Broyden
forward execution is separately not the configured functional Broyden method;
its `Simple` refinement behavior and ignored configuration are compatibility
semantics that must be made explicit or retired.

## 9. Configuration migration matrix

`ConfigSCF` currently has 20 fields and `ConfigFermi` has 5 fields (25 total).
`method` selects the electronic model (GFN0/1/2), while the remaining fields
mix solver, convergence, derivative, occupation, and compatibility concerns.

| Current field | Current meaning | Target owner / note |
| --- | --- | --- |
| `strict` | raise instead of some compatibility warning/substitution | legacy compatibility policy; remove once substitutions/fallbacks are explicit |
| `method` | GFN electronic model | physical/model settings, likely outside solver strategy |
| `guess` | EEQ/SAD initial state | initial guess |
| `maxiter` | outer SCF iteration budget | controller/global evaluation budget |
| `mixer` | linear/Anderson/Broyden selection | method/strategy; replace old mixer label |
| `mix_guess` | put initial guess in history | method-specific option |
| `damp`, `damp_init`, `damp_dynamic`, `damp_dynamic_factor`, `damp_soft_start`, `damp_generations`, `damp_diagonal_offset` | old mixer damping and history regularization | method-specific options; do not pass to methods that do not use them |
| `scf_mode` | FULL/unrolled, implicit, experimental single-shot derivative route | differentiation policy (and legacy route selector today); separate from primal method |
| `scp_mode` | charge/potential/Fock fixed-point variable | primal working representation / response representation, which must become distinct concepts |
| `x_atol`, `x_atol_max` | solver state update/residual tolerances | common convergence/certification profile after S2 |
| `f_atol` | solver output/function tolerance | common convergence diagnostics after S2; determine current exact usage before migration |
| `force_convergence` | permit forced/continued handling | compatibility/failure policy; likely retire in favor of explicit status policy |
| `fermi.etemp` | electronic temperature | physical occupation/free-energy setting |
| `fermi.maxiter`, `fermi.thresh` | Fermi chemical-potential iteration controls | occupation solver settings and occupation validity diagnostics |
| `fermi.diff_order` | differentiable Fermi Newton-step order | differentiation/occupation numerical setting |
| `fermi.partition` | free-energy partition convention | physical free-energy setting |

## 10. Representation/map inventory

| Representation | State and raw map | Initialization/final state | Convergence and derivative status | Method fit |
| --- | --- | --- | --- | --- |
| Charge | packed AO-indexed mono/dipole/quadrupole charge tensor; map builds potential/Hamiltonian, evaluates spectrum, then maps Mulliken charges back | EEQ/SAD charge guess; final charges come from selected converged-map result, with physical spectrum finalized consistently | residual is legacy solver's configured tensor norm; E3.1 implicit test references include H2/GFN1 charge, finite-T charge, and NO2/GFN2 high-order selected response. F1 parity is stricter than observed numerical delta | linear/Anderson/Broyden natural; CDIIS/SOSCF need a separate density/Fock history or conversion, not an implicit relabel |
| Potential | packed AO-indexed mono/dipole/quadrupole potential; potential is assembled into Hamiltonian and map returns potential from spectral charges | charge guess converted to potential; final potential/Hamiltonian from final physical reevaluation | E3.1 has H2 charge/potential/Fock dense IFT and higher response, plus NO2/GFN2 300 K potential. Kickoff finite-T parity includes one failing row | linear/Anderson/Broyden natural; CDIIS/SOSCF need explicit conversion to AO Fock/density and independent method state |
| Fock | full AO Hamiltonian/Fock matrix; diagonalization and density/charge map produces next Fock | Fock initial guess from charge-derived potential plus core Hamiltonian; final physical Hamiltonian is spectral output | E3.1 H2 dense IFT/FD includes Fock representation. Functional map includes invariant spectral/free-energy paths | linear/Anderson/Broyden natural as matrix-space updates; CDIIS natural in AO Fock/error history; SOSCF changes orbital state and needs a shared physical state conversion |

A method's working coordinate is an optimizer/mixing choice. The implicit
response coordinate is a choice of equation used for `x = G(x)` and must be
validated independently. S0 does not select one universal response
representation.

## 11. Convergence semantics

- **Functional simple:** map current `x`, form raw residual/update and test
  configured x/f criteria in solver loop; retain map result associated with
  accepted/tested input for output. Eager finalization may reevaluate physical
  output. No common method-independent final certification exists yet.
- **Functional Anderson:** same physical map, but method-owned immutable
  history and candidate selection; map/update convergence remains coupled to
  Anderson loop policy. Its root values agree with simple in accepted H2 tests;
  finite-T legacy parity rows fail at the strict bound.
- **Legacy Broyden/rootfinder:** legacy mixer/root solver stopping and
  xitorch root/adjoint settings; legacy status/result conversion owns
  convergence. This is distinct from physical root certification.
- **Implicit preparation:** E3.1 first acquires root eagerly, checks solver
  convergence and final Fermi validity, rejects maxiter-zero/nonconverged
  preparation, then attaches the implicit response at retained `root_x`.
  The prepared transform does no primal iteration/status inspection.
- **FULL/unrolled:** solver convergence is checked in the primal execution;
  autograd tracks the selected trajectory. `strict` controls compatibility
  warnings/errors, not a shared physical convergence definition.

Current checks include map/update norms and Fermi status. Energy/free-energy
progress can be available, but is not a universal method-independent
criterion. A rebuilt Fock-density commutator/orbital-gradient diagnostic and
final method-independent root certification belong to S2. The conflation is
at each solver loop/status adapter where “the method stopped” is represented
as “the physical result is valid”; implicit preparation already provides a
stronger separate boundary for derivative attachment.

## 12. Differentiation boundaries

| Route | Current support and boundary |
| --- | --- |
| ordinary eager/no derivative | legacy-compatible calculator call; functional FULL eager route for eligible built-in, single-System inputs |
| FULL/unrolled | autograd follows selected functional simple/Anderson or legacy trajectory, depending on eligibility |
| implicit | E3.1: primal acquisition -> primal/Fermi validation -> prepared-root attachment -> ordinary physical finalization; eligible single-System implicit Simple/Anderson use functional route, compatibility cases stay legacy |
| `torch.autograd` | supported for ordinary eager and accepted functional prepared-root path |
| `torch.func.jvp`, `jacrev`, `jacfwd` | prepared-root transform and matrix-free solve tested; public whole-System transform routing belongs to E6a |
| `vmap` | root primitive/failed-lane semantics have local coverage; public System route and full strategy independence remain broader E6/S12 scope |

Legacy implicit custom autograd and vendored xitorch root machinery still
couple root attachment to old engine conventions. The accepted E3.1 contract
is a hard boundary for S1: do not differentiate through solver histories when
requesting the physical implicit derivative.

## 13. Proposed S1 structural migration

Minimum viable S1 is a contract shell around the already implemented physical
map, with linear mixing as the only consumer. It must keep numerical defaults
and routing unchanged until separately accepted migrations.

| Concept | Minimal proposed representation | Existing material to reuse |
| --- | --- | --- |
| `SCFProblem` | frozen container/call boundary for `SCFInputs` and `evaluate(state) -> SCFEvaluation` | `SCFInputs`, `scf_map`, `SCFMapResult`, `SCFSpectralResult` |
| `SCFState` | consistent physical values from one evaluation: raw working tensor plus spectral state/status | existing immutable map/spectral results; avoid duplicate density/eigensolver state |
| `SCFMethod` | plain protocol or named pure step function taking problem/state/method state | linear/simple update in `functional/solver.py` |
| `SCFMethodState` | immutable method-specific history (empty linear state first) | later `AndersonState`; do not put it in Problem |
| `SCFStrategy` / stage | frozen method choice and method options; one linear stage for S1 | `ConfigSCF` adapter, no generic dict option bag |
| `SCFController` | small eager loop that requests a method proposal and convergence check | extract only enough existing simple loop to demonstrate contract |
| `SCFConvergence` | result/decision data boundary, initially mirrors current solver-local criteria | no new threshold, certification, or commutator policy in S1; S2 owns semantics |
| `SCFDifferentiation` | route/attachment policy outside method/state | reuse E3.1 prepared-root path, no new autograd Function |

Likely files: add `functional/problem.py`, `functional/method.py` for frozen
method/state/stage contracts, and `functional/controller.py` for the eager
controller. Keep physical equations in `map.py`/`spectral.py`, prepared-root
logic in `implicit.py`, and compatibility routing in `eager.py` until
subsequent packages. Linear is the first consumer. Anderson history migration
is S3; native Broyden S4; CDIIS/SOSCF later packages. Do not duplicate map,
spectral filling, Fermi validity, result conversion, or the legacy engine.

S1 must not edit broader System/Result or batching abstractions owned by the
general refactor. Preserve an adapter at the calculator boundary and document
any required shared contract for coordination. S2 owns convergence redesign;
S1 only moves the existing numerical predicate into an explicit boundary.

## 14. Synchronization gate

**S1 production work is blocked** until all of these conditions hold:

1. A specific committed general-refactor/integration synchronization SHA is
   selected by the architecture owner.
2. That SHA includes or explicitly resolves the intended `tad-libcint 0.4`
   compatibility work.
3. `scf/redesign` is rebased onto that SHA at a clean package boundary.
4. The focused SCF baseline is rerun in the isolated SCF environment.
5. Every remaining failure is in a documented inherited category (including
   intentional legacy behavior slated for retirement, external dependency
   issues, strict parity/reference issues, or known xitorch defects).
6. There are **zero unexplained focused baseline failures**.

This does not require every legacy or xitorch test to pass. It does require
knowing what every remaining failure means before S1 changes the contracts.
The current gate is blocked by the unselected dependency/integration SHA and
the focused failure categories F1-F3 still requiring a package-by-package
disposition.

## 15. Explicit non-goals

S0 changes no production code, public API, dependencies, tolerances, solver
defaults, or test expectations. It does not implement `SCFProblem`, redesign
`ConfigSCF`, implement Broyden/CDIIS/SOSCF, patch libcint, cherry-pick
integration work, delete xitorch, or modify the general-refactor worktree.
