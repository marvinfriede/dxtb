# B3b-Final.0 System layout and transform-critical gather inventory

Audit base: `24a72b97965d7d4b8c01ee230ce4d28ea46ac6d8`

Branch: `claude/tender-dijkstra-cda42s`
Scope: source and compatibility audit; no production change is justified by
this review.

## Milestone state and dependency order

B2, B4, B5, B5.6, B6b-core, C5, C7.0-C7.4, and C8a are accepted. This audit
completes only the B3b-Final.0 inventory. Final B3b remains open pending the
implementation packages below. Broader C6 remains partial and is an independent
C8b prerequisite. C8b follows final B3b and the already accepted B6b-core, C5,
C6, C7, and B2. E6a remains separately open; E6bcompat-policy precedes
E6bimpl, and E6bimpl precedes E6broute. C9 remains downstream of E4.6. No
ordering or completion claim is made for E1.3.

## Current `System` field inventory

“Core candidate” means data needed by a future exact single-System stackable
core. It is a design target, not a `System` Node introduced here. Tensor fields
are dynamic children if/when that Node is implemented; immutable non-tensor
settings are static metadata. Model-wide parameter tables stay on shared
`Model`/`ParamModule`, rather than being copied into each System.

| Current field | Current owner and use | Core / legacy disposition |
|---|---|---|
| `numbers` | System integer tensor; composition, shapes and device authority | Core dynamic integer child; exact single-System path requires rank 1. Packed shape semantics stay in legacy adapter. |
| `config` | Shared immutable effective `Config`; method, integral level and SCF settings | Share immutable configuration or encode its required static fields; it has no dtype/device/batch setting. |
| `batch_mode` | System scalar; selects packed cache and setup paths; SCF/Calculator legacy code generally reads `ihelp.batch_mode` | Remove from future exact core. Keep mode 1/2 in explicit packed/legacy ownership. Do not fake mode 0 for packed data. |
| `ihelp` | Full mutable `IndexHelper`; basis, species, shell/orbital maps, padding and packed semantics | Do not retain in a core Node. Preserve for legacy Calculator/SCF compatibility until narrower consumers are migrated. Reuse existing C5/C7 maps. |
| `classicals` | Mutable `ClassicalList`; component identity, insertion order, extension hooks | Legacy compatibility owner. Core should use ordered immutable term descriptors plus setup Nodes, without storing mutable components as Node children. |
| `interactions` | Mutable `InteractionList`; identity, ordering and custom hooks | Legacy compatibility owner. Core should use ordered descriptors and captured setup Nodes; custom extension policy remains explicit. |
| `es2_setup` | `ES2CoreSetup` for eligible exact single System | Authoritative dynamic Node; retain `None` when absent. |
| `es2_interaction` | Associated mutable exact/extension object used for identity and compatibility | Legacy association outside core Node; replace core use with immutable term descriptor when C8b policy is fixed. |
| `es3_setup` | `ES3CoreSetup` | Authoritative dynamic Node; `None` when absent. |
| `es3_interaction` | Associated interaction object | Legacy association outside core Node; immutable descriptor is future core representation. |
| `aes2_setup` | `AES2CoreSetup` | Authoritative dynamic Node; `None` when absent. |
| `aes2_interaction` | Associated interaction object | Legacy association outside core Node; immutable descriptor is future core representation. |
| `d4sc_setup` | `D4SCCoreSetup` | Authoritative dynamic Node; `None` when absent. |
| `d4sc_interaction` | Associated D4SC component; identity binds functional setup authority | Legacy association outside core Node; preserve exact identity contract. |
| `generalized_born_setup` | `GeneralizedBornCoreSetup` | Authoritative dynamic Node; `None` when absent. |
| `generalized_born_interaction` | Associated mutable component/extension hook | Legacy association outside core Node; immutable term descriptor is future core representation. |
| `integral_setup` | Outer `IntegralSetup` compatibility dataclass with active PyTorch or libcint Node | Active backend setup remains authoritative; outer container remains a compatibility wrapper unless a supported transform requires it directly. |
| `h0_setup` | `H0SetupNode` on exact single System; legacy `H0Setup` for packed | Core dynamic Node for single System. Packed legacy setup remains outside core. Calculator H0 adapter is isolated by graph-preserving copies (C7.2a). |
| `_classical_extension_cache` | Persistent setup-time cache for custom classical extensions only | Legacy extension ownership; do not put arbitrary extension cache or component objects in core Node. |
| `_packed_classical_cache` | Complete `ClassicalListCache` for packed modes | Packed-only legacy owner; never a core child. |
| `repulsion_setup` | `RepulsionCoreSetup` or legacy setup for packed | Core Node for exact single System; legacy setup for packed. |
| `repulsion_classical` | Associated component, used for identity/order and cache adapter | Legacy association; core term identity must become an immutable descriptor. |
| `srb_setup` | `ShortRangeBondCoreSetup` or legacy setup | Core Node for exact single System; legacy setup for packed. |
| `srb_classical` | Associated component | Legacy association; immutable descriptor needed in future core. |
| `d4_setup` | `DispersionD4CoreSetup` or legacy setup | Core Node for exact single System; legacy setup for packed. |
| `d4_classical` | Associated component | Legacy association; immutable descriptor needed in future core. |
| `ies_setup` | `IESCoreSetup` or legacy setup | Core Node for exact single System; legacy setup for packed. |
| `ies_classical` | Associated component | Legacy association; immutable descriptor needed in future core. |
| `halogen_setup` | `HalogenCoreSetup` or legacy setup | Core Node for exact single System; legacy setup for packed. |
| `halogen_classical` | Associated component | Legacy association; immutable descriptor needed in future core. |
| `dd` | Mutable typed device/dtype dictionary; setup override and no-floating-setup dtype fallback; Calculator.type updates dtype | Do not move the dictionary into Node context. Preserve an explicit effective floating dtype scalar in future core, updated by type conversion. Device validation remains anchored to `numbers.device`; Calculator metadata remains legacy-owned. |
| `d3_setups` | Ordered `(DispersionD3 component, DispersionD3CoreSetup)` pairs | D3 setup Nodes are authoritative dynamic core values; component refs/order remain compatibility metadata. Future core needs ordered labels/descriptors paired with captured setup Nodes. |

The following lifecycle matrix makes the per-field ownership and transform
implications explicit. “Captured” means during `Model.setup`; “per call” refers
to pure System evaluation unless marked legacy.

| Field | Tensor leaves / static state | Setup-time construction | Per-call consumer | Stackability implication |
|---|---|---|---|---|
| `numbers` | Integer tensor leaf | Passed to `Model.setup` | Shape/device checks, integral and SCF data | Pad/stack only under C8b’s explicit mixed-composition policy. |
| `config` | Frozen static configuration; no tensor leaves | Shared from `Model` with effective integral level | Integral, SCF, and feature selection | Static context or selected immutable scalar fields. |
| `batch_mode` | Python integer | Inferred/validated by `Model.setup` | Compatibility view; legacy Calculator/SCF routing | Exclude from exact core; packed modes stay outside it. |
| `ihelp` | Many integer maps, masks, and counts; mutable wrapper | `IndexHelper.from_numbers` in setup | SCF charge layout and legacy drivers; some fixed maps copied for functional inputs | Not an opaque child; expose only narrow already captured maps. |
| `classicals` | Mutable component list; components can own tensors/caches | Built/resolved by Model | Ordered result dispatch, compatibility and custom hooks | Keep out of tree; encode built-in terms in ordered descriptors. |
| `interactions` | Mutable component list; extensions may own arbitrary state | Built/resolved by Model | Interaction preparation/dispatch and extension hooks | Keep out of tree; ordered built-in descriptors plus explicit extension boundary. |
| `es2_setup` | Dynamic ES2 parameter tensors; static validated options | ES2 setup factory in `Model.setup` | ES2 matrix and potential construction | Optional Node child; compatible stacking requires matching static options. |
| `es2_interaction` | Mutable interaction object, possible component tensors | Selected by Model/config | Identity and legacy/custom dispatch | Exclude object; retain a static immutable term descriptor. |
| `es3_setup` | Dynamic ES3 tensor parameters | ES3 setup factory in `Model.setup` | Per-call ES3 geometry/interaction data | Optional Node child with static mode aligned across stack. |
| `es3_interaction` | Mutable interaction object | Selected by Model/config | Identity and extension dispatch | Exclude object; immutable descriptor only. |
| `aes2_setup` | Dynamic AES2 tensors and structural values | AES2 setup factory in `Model.setup` | Per-call multipole geometry data | Optional Node child; stack only like-configured terms. |
| `aes2_interaction` | Mutable interaction object | Selected by Model/config | Identity and compatibility dispatch | Exclude object; immutable descriptor only. |
| `d4sc_setup` | Dynamic model/parameter tensors; static validated metadata | D4SC setup factory in `Model.setup` | Functional interaction preparation and D4SC energy/potential | Optional Node child; model mode/static structure must match. |
| `d4sc_interaction` | Mutable D4SC component; identity binds authority | Selected by Model/config | Identity test when preparing interaction data | Exclude object; preserve descriptor and setup association. |
| `generalized_born_setup` | Dynamic Born/model tensors; static method/cutoff descriptors | GeneralizedBorn setup factory in `Model.setup` | Per-call geometry-dependent radii/matrix | Optional Node child; static method metadata must agree. |
| `generalized_born_interaction` | Mutable component/extension state | Selected by Model/config | Identity and compatibility route | Exclude object; descriptor for exact built-in path. |
| `integral_setup` | Active PyTorch/libcint Node tensor leaves; outer dataclass metadata | One active backend setup in `setup_integral_evaluation` | Integral evaluation on each geometry | Stack active setup Node only; inactive backend remains `None`. |
| `h0_setup` | Dynamic H0 parameter/index leaves on exact System; legacy object for packed | `setup_h0_core` on eligible single System | Raw/weighted H0 and occupations | Stack compatible method/CN descriptors and optional fields only. |
| `_classical_extension_cache` | Arbitrary legacy cache values, potentially tensors/objects | Custom extension `get_cache` during setup | Extension compatibility path | Not stackable by default; remains outside exact core. |
| `_packed_classical_cache` | Family legacy setups, potentially tensors and external objects | Full list cache only for packed setup | Packed Calculator/classical evaluation | Packed adapter owns it; do not stack as a core Node. |
| `repulsion_setup` | Dynamic repulsion parameters/masks; legacy dataclass for packed | Family setup builder once | Repulsion energy/potential | Optional Node child; shape and static options constrain stacking. |
| `repulsion_classical` | Mutable component reference and possible extension state | Chosen by ClassicalList resolution | Identity/order and legacy update/reset hooks | Exclude object; use ordered descriptor. |
| `srb_setup` | Dynamic SRB parameters, maps, masks | Family setup builder once | SRB energy/potential | Optional Node child; compatible shell/atom shape required. |
| `srb_classical` | Mutable component reference | Chosen by ClassicalList resolution | Identity/order and legacy hooks | Exclude object; use descriptor. |
| `d4_setup` | Dynamic D4 parameters, model tensors and cutoff tensors | D4 setup builder and core conversion once | Call-local model/cutoff/damping reconstruction and energy | Optional Node child; sparse parameter structure/static modes must match. |
| `d4_classical` | Mutable component reference | Chosen by ClassicalList resolution | Identity/order and legacy hooks | Exclude object; use descriptor. |
| `ies_setup` | Dynamic EEQ tensors and IES parameters; static CN mode | IES family setup builder once | IES energy/potential | Optional Node child; static strategy must agree. |
| `ies_classical` | Mutable component reference | Chosen by ClassicalList resolution | Identity/order and legacy hooks | Exclude object; use descriptor. |
| `halogen_setup` | Ten dynamic tensors including integer numbers and Boolean masks | Halogen family setup builder once | Halogen energy and nearest-neighbor selection | Optional Node child; mask shapes/static label must agree. |
| `halogen_classical` | Mutable component reference | Chosen by ClassicalList resolution | Identity/order and legacy hooks | Exclude object; use descriptor. |
| `dd` | Mutable Python device/dtype mapping, no tensor children | Resolved from `par.dd` or explicit Model override | Dtype fallback and Calculator metadata/type path | Do not stack dictionary; carry only effective dtype metadata in core. |
| `d3_setups` | Dynamic D3 setup tensor leaves plus mutable component references | One D3 core setup per exact selected D3 component | D3 energy dispatch keyed by component identity/ordered label | Stack corresponding setup leaves and immutable descriptors in matching order. |

Per-call positions, charge, spin, field, field gradient, interaction geometry
data, SCF iterates, integral matrices, result tensors and call-local libcint
wrappers are not persistent System setup fields. They remain evaluation inputs,
local work or outputs.

## `batch_mode` and `dd` decisions

`Model.setup` accepts/infer modes 0, 1 and 2, constructs an `IndexHelper` with
that mode, and branches to exact captured Nodes only for mode 0. Modes 1 and 2
retain padded-batch/conformer shapes and the established legacy cache and
Calculator drivers. Pure `System.singlepoint` rejects non-1D numbers and does
not read `System.batch_mode` directly; the `classical_cache` property uses it to
choose the on-demand single-System compatibility view versus the persistent
packed cache. Functional SCF gates use `ihelp.batch_mode`, and legacy drivers,
mixers and analytical paths also depend on the helper’s mode.

Disposition: future core is defined for one exact unbatched composition and
contains no batch-mode selector. A separate outer packed/Calculator adapter
owns modes 1 and 2 and provides their established legacy route. Keep the
current façade and compatibility property until that adapter is implemented
and tested; deleting the scalar globally now would break known callers.

`Model.setup(dd=...)` uses an explicit override in preference to `par.dd`, and
passes it into setup factories. `singlepoint` validates position device against
`numbers.device`; expected dtype comes from the first floating setup tensor,
or from `system.dd["dtype"]` when no floating setup exists. The latter behavior
is covered by `test_core_singlepoint_validates_dtype_without_integral_setup`.
`Calculator.type(dtype)` converts integral/H0 setup values and replaces the
stored dtype in `dd`. Thus `dd` cannot simply disappear: the core needs the
effective dtype even for SCF-excluded/no-integral cases, while the mutable
dictionary and runtime device selection stay in the compatibility boundary.
Preserve explicit `dd` overrides; do not infer dtype from integer numbers.

The initial focused model/singlepoint/H0 validation did not complete cleanly:
its ten known GFN1 production/legacy density-parity cases failed, while the
remaining tests passed. The audit did not change dtype/device policy.

## B3b-Final.0 validation evidence

Primary environment: Python 3.14.8 / PyTorch 2.14.1, CPU. All pytest runs used
`OMP_NUM_THREADS=1 -n logical --dist worksteal`.

- `test/test_calculator/test_model.py test/test_calculator/test_singlepoint_core.py test/test_calculator/test_general.py test/test_hamiltonian/test_h0_setup_node.py`: 122 collected, 112 passed, 10 failed. All ten failures are parameterizations of `test_production_singlepoint_matches_legacy_full`, the tracked GFN1 density-parity issue; they are not attributed to this documentation-only audit.
- The same model/general/singlepoint/H0 tests, excluding `production_singlepoint_matches_legacy_full`, plus `test/test_integrals/test_pytorch_setup_node.py test/test_integrals/test_libcint_setup_node.py`: 142 passed.
- `test/test_scf/test_functional_map.py::test_b6a_functional_interaction_family_matrix test/test_scf/test_functional_map.py::test_b6a_functional_interaction_boundary_is_history_independent`: 8 passed.
- `test/test_calculator/test_b3b_layout_contract.py`: 6 passed in each approved environment: Python 3.10.22/PyTorch 2.6.0; Python 3.11.15/PyTorch 2.8.0; Python 3.12.10/PyTorch 2.11.0; Python 3.12.10/PyTorch 2.13.0; Python 3.14.8/PyTorch 2.14.1. The module covers modes 0/1/2 ownership, explicit dtype fallback without floating setup, no IndexHelper reconstruction during evaluation, mutation sensitivity of the captured shell map, overlap A/B/A plus JVP/jacfwd/jacrev/vmap and a real Slater parameter gradient, full singlepoint A/B/A outputs/position gradients, and per-call ES2/ES3 interaction-data history.
- `test/test_integrals/test_pure_evaluation.py test/test_integrals/test_pure_multipole.py test/test_integrals/test_param_grad.py test/test_integrals/test_transforms.py`: 64 collected, 62 passed, 1 skipped, 1 failed. The failure is `test_exponent_gradient_matches_fd[gfn2-LiH]` with a NaN H0 matrix followed by `torch.linalg.eigh` convergence failure. This is the previously classified C7.1 GFN2 LiH issue; this audit did not repeat the original pre-C7.1 matched-base run, so it is reported as the known inherited signature rather than a newly proven comparison.

The primary environment emitted the existing Python 3.14 TorchScript
`FutureWarning`; Torch 2.11 and 2.13 emitted the existing TorchScript
deprecation warnings in the new module. CUDA was not run under the
repository's CPU-only policy. No five-environment libcint matrix was run;
these focused layout tests exercise the PyTorch setup and CPU-only metadata.

## Structural masks, maps and gathers

| Source / operation | Inputs and dependency | Shape / host effect / transform relevance | Current owner and disposition |
|---|---|---|---|
| `Model.setup`: `torch.unique(numbers)`, `IndexHelper.from_numbers` | Atomic numbers only | May perform structural grouping and host-side setup; defines species/shell/orbital layout | Correct setup-time owner. Do not repeat during exact geometry evaluation. |
| `basis/bas.py`: Basis parameter gathers (`ngauss`, `pqn`, `slater`, `valence`, shell metadata) | Fixed composition + ParamModule leaves | Structural setup; parameter-dependent tensors retain graph | Captured by integral setup Nodes. libcint CGTO conversion is deliberately call-local from captured Slater data. |
| `xtb/h0.py`: `setup_h0` `torch.unique`, valence, element and pair gathers, optional EEQ values | Fixed numbers + trainable parameter tables | Setup-time tensor preparation; some host structural bookkeeping occurs before captured Node creation | Correct setup owner; `H0SetupNode` and `H0IndexNode` make used state visible. No evaluation-time ParamModule access in pure H0. |
| `components/classicals/shortrangebond/factory.py`: three `torch.unique(numbers)` gathers | Numbers and SRB parameter leaves | Repeated setup-time grouping; no geometry dependency or transformed evaluation | Correct but redundant setup work; captured in `ShortRangeBondCoreSetup`. Optimizing the repeated unique is optional and not a correctness blocker. |
| Classical and interaction factories: family `par.get*` calls | Composition and differentiable ParamModule leaves | Setup-time parameter-to-element/shell gathering | Captured by accepted family setup Nodes. Custom extension hooks remain legacy-owned and may access mutable components by contract. |
| `scf/functional/data.py`: orbital mask from `ihelp.orbitals_per_shell`; padding diagonal | Composition, integral shape/device/dtype; no geometry or trainable parameter | Per-call fixed-shape map operation; no host scalar extraction or shape change; padding semantics affect overlap conditioning | Legitimate evaluation-local operation. A future narrow `H0IndexNode`/CoreIndexData consumer can reuse the captured map if behavior and packed adapters are preserved; no correctness defect established. |
| `scf/functional/data.py`: clones of `shells_to_atom` and `orbitals_to_shell` into `CoreIndexData` | Integer IndexHelper maps only | Per-call copy, fixed shape, no host synchronization, no trainable graph | Repeated numbers-only snapshot/conversion. Candidate for B3b-Final.1 consolidation if the active backend Node map is a valid owner for all callers. No production fix is justified without an instrumented parity/identity reproducer. |
| `integral/driver/pytorch/impls/pairs.py`: `select_pairs` `torch.unique` after screening | Geometry-dependent distances and screening threshold | Deliberately data-dependent selected-pair shape, discrete selection; known restriction for vmap/compile screening | Correct call-local numerical selection, not setup-only composition structure. Preserve accepted limitations. |
| `basis/bas.py` / `LibcintBasisNode.create_libcint_cgtos`: `.item()`/`.tolist()` on valence/angular structure | Captured numbers-only basis metadata plus captured Slater tensors | Call-local external CGTO construction; host scalar extraction, outside the PyTorch pure transform guarantee | Accepted libcint compatibility boundary from C7.3; no ParamModule regather or persistent wrapper. Do not move wrappers/CGTOs to System. |
| H0 and functional/index-map `wrap_gather` operations | Captured fixed index tensors and numerical tensors | Fixed-shape differentiable gathers; no shape discovery | Intentionally retained and transform-visible. A gather is not misplaced merely because it runs per evaluation. |

## B3b-Final.1 — SCF structural-map ownership

Status: complete for the per-call `CoreIndexData` construction question;
final B3b remains open for the later core/packed and cross-family boundary.

`setup_scf_inputs()` has two production callers through the prepared
functional runners: eager FULL and the implicit-root path, both dispatched by
`scf.iterator.solve()`. It also remains a direct functional preparation API
used by tests/diagnostics and compatibility callers. Both runners pass through
the direct fallback unless `System.singlepoint()` supplied a captured map.
Legacy packed Calculator paths do not qualify for the functional unbatched
route and do not pass a map.

For exact single-System setups, both backend Nodes already capture the two
maps SCF reductions need:

| Owner | Map fields | Construction and device policy | Relation to `System.ihelp` |
|---|---|---|---|
| `PytorchIntegralSetupNode.index` | `shells_to_atom`, `orbitals_to_shell` | Captured once by C7.1 from the active setup; positions-device placement | Equal in shape, integer dtype, device, ordering and contents; separate tensor storage |
| `LibcintIntegralSetupNode.index` | `shells_to_atom`, `orbitals_to_shell` | Captured once by C7.3; CPU when `force_cpu=True` | Equal in structural values; CPU placement can differ from the SCF helper on non-CPU systems |
| `H0SetupNode.index` | Same two maps plus atom/species/shell maps | Captured once by C7.2 on the System setup device | Equal values; separately owned structural tensors |
| `System.ihelp` | Full mutable IndexHelper state including `orbitals_per_shell` | Legacy compatibility owner | Remains required for charge/reference-occupation setup and padding-aware orbital-mask construction |

On the exact singlepoint path, `System.singlepoint()` selects the active
backend Node's `CoreIndexData` and supplies it only to `scf.solve()`. The
prepared FULL and implicit runners pass it to `setup_scf_inputs()`, where
shape, dtype, and device are checked before reuse. A mismatched backend device
falls back to the old isolated map snapshot. Direct calls with no map still
construct one `CoreIndexData` from two graph-independent integer clones; that
boundary preserves isolation from a caller's mutable IndexHelper. Passing a
legacy `PytorchIntegralSetup`/`LibcintIntegralSetup` without a Node `index`
also follows that fallback. Packed modes do not enter this route.
The optional `core_index` is an internal trusted-input boundary: its values
come from the active captured setup Node, whose map equivalence is tested at
setup. Per-call validation checks shape, dtype, and device without synchronizing
map values; direct callers should omit the optional argument.

Instrumentation on GFN1/H2 observed, per eligible functional SCF preparation:
zero `CoreIndexData` constructions and zero clones of the two IndexHelper
maps; `SCFInputs.index` is the exact active backend Node object. A direct
`setup_scf_inputs()` call observed one construction and exactly two map clones.
Both the prepared FULL and implicit-root singlepoint paths now have direct
identity assertions for the captured `SCFInputs.index`.
System setup still performs the existing backend-specific captured map
construction once. No speed claim is made from eliminating these two small
integer copies. `orbital_mask` continues to use
`ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell)`; the captured
reduction Node does not encode the orbital counts or padding semantics.

The map equivalence test covers GFN0, GFN1, and GFN2 with H2, LiH, H2O, and
SiH4 for PyTorch, plus the installed libcint route with H2O. It checks shape,
dtype, device, values, independent source identity/storage, and H0 map parity.
The direct fallback checks its two clones. The eligible System test checks
Node identity and zero per-call construction/cloning. This closes no
parameter-gather issue: instrumentation of `ParamModule.get()` and
`get_elem_param()` after GFN1/H2 setup observed no calls over two geometries.
The existing fixed-shape gathers and the call-local padding mask remain
intentional. No physics change was made.

The accepted A/B/A history test now differentiates geometry A, evaluates B,
then differentiates A again before comparing position gradients. No packed
path or direct functional signature was removed. Effective `dd` dtype fallback
still applies with no floating setup, including explicit float32/float64
overrides, and mismatched device/dtype checks remain active.

The small structural-copy reduction was characterized by construction and
clone counts; no wall-clock performance measurement was performed. The
primary selected functional control run passed 13 tests and had one existing
Anderson real-position-gradient parity failure at
`test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`
(maximum absolute difference 2.5e-11 against 2e-13). That test directly builds
SCF inputs without the optional captured map; it does not exercise the new
reuse branch. No matched-base execution was performed for that numerical
failure. One initial integration run exposed a candidate-only compatibility
failure when a legacy `PytorchIntegralSetup` was substituted into a System;
the backend map lookup now falls back when the object has no captured `index`,
and all three affected legacy setup parameterizations pass after the fix.

## B3b-Final.1 focused validation

`test/test_calculator/test_b3b_layout_contract.py`: 26 passed in each approved
CPU environment. PyTorch versions were 2.6.0, 2.8.0, 2.11.0, 2.13.0, and
2.14.1 respectively; the libcint map test ran (not skipped) in all five. CUDA
was not tested under the CPU-only policy. Primary environment model,
singlepoint, PyTorch setup Node, and H0 setup Node selection excluding the ten
known GFN1 density-parity cases: 114 passed. Selected functional-map/root
controls: 13 passed; the separate Anderson parity case above failed. The E0
iteration-compatibility module passed 9 tests. Full E3 was not run; no E3
acceptance claim is made.

Parameter lookup audit found no exact single-System pure evaluator that
regathers from `ParamModule`: setup factories gather and capture graph-bearing
values; evaluation consumes those values plus per-call geometry/charge. D3/D4,
repulsion, SRB, IES, Halogen, ES2/ES3/AES2/D4SC/GeneralizedBorn and H0 use their
captured setup Nodes. Custom extension hooks remain an explicit legacy
exception. No detached cache or newly persistent evaluation state was found.

## Compatibility boundaries and recommended packages

- `System.classical_cache` remains an on-demand view for single Systems, with
  setup Node identity preserved; `_classical_extension_cache` remains
  extension-only and `_packed_classical_cache` remains packed-only. Do not
  reconstruct exact built-in setups to populate the view.
- `System.classicals` and `System.interactions` remain mutable compatibility
  containers carrying component identity, ordering and custom hooks. They are
  not Node children. Keep direct APIs and packed Calculator behavior until the
  corresponding compatibility policies retire them.
- `System.ihelp` remains the full IndexHelper for SCF and Calculator legacy
  behavior. `CoreIndexData`, `H0IndexNode`, `PairPlanNode`, and the
  family-specific Nodes are narrower authorities, not a universal replacement.
- No persistent Model, live integral container, mutable PairPlan, Basis,
  ParamModule, libcint wrapper, or per-call matrix belongs in the future core.

Recommended follow-up split:

1. **B3b-Final.1 — transform-critical maps and gathers:** complete. Functional
   SCF reuses an active backend Node when its structural maps match the
   evaluation device; direct and device-mismatched compatibility paths retain
   isolated clones. Parameter-gather instrumentation found no unintended
   per-call lookup. The padding-aware orbital mask remains IndexHelper-owned.
2. **B3b-Final.2 — core/packed layout boundary:** introduce an explicit exact
   single-System layout without `batch_mode` or mutable `dd`; preserve packed
   modes 1/2, Calculator.type, explicit setup dtype overrides, and direct
   compatibility adapters. This package must settle effective dtype ownership
   for the no-floating-setup case.
3. **B3b-Final.3 — cross-family compatibility and acceptance:** exercise custom
   classical/interaction extensions, ordered labels, direct compatibility
   views, reset/type behavior, PyTorch/libcint, C6 and all six classical setup
   families. Close B3b only after both core and packed contracts pass.

C8b should begin only after that final layout is accepted and broader C6 is
resolved to its existing acceptance contract. E6a and the later E6b policy,
implementation, and route milestones remain independent in their documented
order. No C8b Node conversion, production change, numerical reference update,
or tolerance change is part of B3b-Final.0.
