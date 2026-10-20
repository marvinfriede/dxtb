# Track C: `Node` migration and retiring `TensorLike`

**Purpose.** Replace `TensorLike` (reflective cloning over `__slots__`) with `Node`: frozen dataclasses registered as pytrees. Conversion, batching (`vmap` over stacked systems) and parameter training (`partition`/`combine`) then work through standard tree operations.

## Current upstream state

The original C1/C1b plan has already been overtaken by tad-mctc 0.9.1.
`Node`, `ModuleNode`, `partition`, `combine`, `stack` and `leaf_paths` are
available and were probed in T0. They satisfy the dxtb requirements relevant
to C5-C8, with the limitations recorded in `T0-baseline-report.md` section 10.

Do not implement another Node base in dxtb.

Additional helpers still needed (found in T0):

- `in_dims_like(node, **fields)` for constructing per-field `in_dims` from the
  existing tree spec;
- an explicit serialization policy for trained Node models rather than relying
  on default `torch.load(weights_only=True)` behavior.

C1 and C1b are complete prerequisites. C2-C4 are retained as dependency
coordination history and as current external cleanup, to be done only after
verifying the exact current state in those repositories (the dispersion
packages currently need the `mctc_shim` bridge, see `00-overview.md` section 2).

**Relation to the design review.** The `TensorLike` design review (revision 8) remains the reference for the problems with `TensorLike` and the probe results. This track supersedes it where they differ:

| Design review (rev. 8) | This plan |
| --- | --- |
| `derived` field kind for caches and setup state | dropped: TB removes that state first |
| interim cache-write helper using `object.__setattr__` | dropped |
| compile minimum torch 2.8 and wrapper | moved to TD (D4) |
| `.to(device)` interim fix | dropped; documented as known issue (T0.6) |
| deleting `TensorLike` in the release that adds `Node` | deletion is a separate final step (C9) |
| caches as explicit values (step 4) | replaced by TB |

**Order.** C1/C1b are done. C2–C4 are external cleanup (verify state first).
C8a is the separate Result-output pytree prerequisite and can follow
B5/B5.6/C1 without waiting for C5-C7. The structural core values (C5-C7)
and C8b Model/System stackability proceed bottom-up after their TB
mutation-removal prerequisites.

**Current status (B3b-Final.3).** B2 and C7.0-C7.4 are complete. C7 setup
Node acceptance is closed at its scoped boundary; legacy drivers remain under
their separate compatibility packages. B3b-Final.0 inventories the System
layout, B3b-Final.1 closes functional-SCF captured-map reuse and the
per-call parameter-gather audit, and B3b-Final.2 adds a frozen exact-mode
layout value with effective dtype while preserving packed modes 1/2 and the
legacy facade. B3b-Final.2a restores arbitrary legacy SCF keyword forwarding,
checks exact-core setup-reference coherence, and prevalidates floating setup
device/dtype. B3b-Final.3 adds ordered immutable classical and interaction
descriptors over the already-owned setup Nodes. Exact built-in classical
evaluation now dispatches from descriptor kind and setup; functional
interaction preparation receives descriptor kinds while retaining the accepted
`FunctionalInteractionSet`. Compatibility identity associations remain beside
the System façade, outside `ExactSystemCore`. Packed modes and extension hooks
remain on their legacy routes. The result is the accepted staged input layout
for C8b, not a registered/stackable System Node. Evidence and the inherited
Anderson gradient classification are in [`B3b final layout inventory`](B3b-Final-layout-inventory.md).
Final B3b is complete at this compatibility boundary. C6-Final.0 found that persistent
ordered descriptors hid their nested setup Nodes from standard pytree
traversal. C6-Final.1 registers those descriptors, making the existing setup
Nodes dynamic children while leaving ExactSystemCore and the call-local
FunctionalInteractionSet unchanged. C6-Final.1a removes temporary descriptor
Node construction from exact-core coherence checks by comparing expected
metadata tuples with the persistent descriptors; setup, owner, ordering, and
index-identity checks remain. C6-Final.2 passed the available cross-family
Node, functional-map, implicit-root, E0, and pure integral controls; libcint
numerical controls remain blocked by the missing PySCF interface in the
available environment at that time. C6-Final.3 has since exercised current
source descriptor/model, packed/cache, interaction-family, and selected
PySCF-backed libcint controls in Python 3.12/Torch 2.14.1. Focused
current-source tests cover family Nodes, Calculator/result behavior, packed
mode energy/gradient parity, and selected libcint/System paths; only this one
of the five approved environments was available, and broader selections that
did not finish are not counted. C6 is complete at its exact-core/legacy
compatibility boundary; C8b may begin. B2, B3b, C6, and C7 are satisfied
prerequisites.
The Anderson gradient discrepancy remains inherited and unresolved; no
numerical policy was changed.

**Current C8b status.** C8b.0 established and tested the schema, alias,
parameter-partition, and stacking preflight. C8b.1 registered production
`ExactSystemCore` and its selected integral backend wrapper. C8b.2 now exposes
the Model-owned registered parameter tree and functional setup boundary while
retaining the public `Model` and `System` façades. C8b.3 owns padded mixed-size
P9 stacking. See [`C8b schema preflight`](C8b-schema-preflight.md) for
production evidence. CUDA validation of mixed-device libcint core trees is
still unavailable in the current environment.

---

## C1 `Node` base in tad-mctc: done

Delivered in tad-mctc 0.9.1 (`Node`, `partition`, `combine`, `stack`, `leaf_paths`). Nothing to implement in dxtb.

## C1b `nn.Module` wrapper for ML terms: done

Delivered as `ModuleNode` in tad-mctc 0.9.1. Unblocks F3.

---

## C2 tad-mctc classes

**Goal.** Migrate tad-mctc's own `TensorLike` classes (`Mol`, the neighbor list).

**Done when.** No `TensorLike` subclass left in tad-mctc; released.

**Needs.** C1. **Size.** S.

---

## C3 tad-multicharge

**Goal.** Migrate `ChargeModel` (and any other `TensorLike` subclass); pin the new tad-mctc.

**Done when.** Released; tests green.

**Needs.** C1. **Size.** S.

---

## C4 tad-dftd3 and tad-dftd4

**Goal.** Migrate `Cutoff` (tad-dftd3) and `BaseModel`, `DispTerm`, `Disp` (tad-dftd4); pin the new tad-mctc and tad-multicharge.

**Done when.** Both released; tests green.

**Needs.** C1, C3. **Size.** M.

---

## C5 dxtb leaf classes: `IndexHelper` and `Basis`

**Goal.** Migrate the classes at the bottom of dxtb's object tree.

**Preconditions.** B6b-core has established an immutable structural value for
the functional single-System core; structural values are built in `setup`
(B3). The legacy batch adapter may still use its existing mutable,
batch-aware `IndexHelper` until E6broute retires that route; E4.6a then removes
its cull/restore machinery.

**Steps.**

1. Migrate the surviving immutable core structural value to `Node`; integer
   index tensors are children; per-molecule counts are tensors, not context
   (P9).
2. Remove `allowed_dtypes` (integer handling now comes from the strict conversion rule: integer tensors keep their dtype).

`batch_mode` and cull/restore are not migrated into the core Node. Do not
remove those methods from the separately isolated legacy `IndexHelper` while
the production batch fallback still uses it. C5 migrates the core value; it
does not require deleting the legacy adapter.

**Done when.** Both migrated; conversion moves their tensors with the parent's `.to(device)`; T0.2 reproduced.

**Needs.** C1, B3a, B6b-core, E2-core, E4core. **Size.** M.

---

## C6 Components and component lists

**Goal.** Migrate all classical and interaction components and their list containers.

**Progress.** C6.1 is complete: the single-System ES2/ES3 setup values are
stored as `ES2CoreSetup` and `ES3CoreSetup` tad-mctc Nodes. C6.2 is complete:
the System-owned AES2 setup is an `AES2CoreSetup` Node. Legacy setup
dataclasses and cache/component routes remain available. C6 remains partial;
other component families and the complete interaction collection are separate
migrations.

C6.3 is complete: the eligible single-System GeneralizedBorn setup is stored
as `GeneralizedBornCoreSetup`; the legacy `GeneralizedBornSetup`, component,
and cache APIs remain available. This migrates only the immutable setup value
and does not complete broader C6.

C6.3a is complete: core conversion transfers only CPU zero-dimensional tensor
`born_scale` and `born_offset` values to the setup device. Same-device scalar
tensors retain identity; foreign-device arrays and `descreening` retain the
upstream shared-device validation behavior. This does not establish general
mixed-device Node support.

C6.4 is complete: eligible single-System D4SC setup is stored as a
`D4SCCoreSetup` Node. Ordered parameter names and recursive model-extra
container metadata remain static, while tensor-valued parameters and model
state remain dynamic children. The legacy `D4SCSetup`, component, cache, and
functional-interaction preparation routes remain available and unchanged.
Broader C6 remains partial; C7 scoped setup Node acceptance is complete.

C6.4a is complete: single-System functional FULL and implicit D4SC preparation
now derive model configuration from the same System-owned setup snapshot used
to build call-local dispersion data. Direct compatibility callers still
capture setup from their supplied component, and solver override routing is
unchanged.

C6.4b is complete: `_interaction_data()` attaches the System-owned D4SC setup
only when the resolved interaction list contains that exact component object.
Alternate exact D4SC components retain component-based compatibility setup,
and selections without D4SC carry no D4SC authority. Broader C6 remains partial.

C6.4c investigated the reported NO2 third-derivative worker-mode NaN using
matched base/candidate runs. The exact test passed in the controlled serial,
one-worker, two-worker, logical-worker, larger randomized-selection, and
insertion-order diagnostic runs. No candidate-specific cause was established;
the original worker-only discrepancy remains unresolved pending its exact
failing invocation and stage-level output. The original failure was not fixed,
and its first non-finite derivative remains unknown. No source or test change
was made. The set-based component ordering observed during diagnosis is a
separate C6 follow-up: the roadmap requires component lists to preserve order,
but the NO2 NaN was not reproduced with or without temporary order control and
ordering is not established as its cause. C6.4b remains accepted on source and
functional validation; broader C6 remains partial and full E3 remains blocked
by E1.3.

C6.5 is complete: the shared `ComponentList` retains constructor order while
deduplicating by the existing Python hash/equality contract and retaining the
first occurrence. Classical and interaction lists, `ComponentList.type()`,
and functional interaction entries inherit that order. The insertion-order
correction is independent C6 work; it did not reproduce or establish a cause
for the unresolved NO2 worker-mode NaN. B7 field values remain per-evaluation
inputs, with no persistent field setup Nodes added. Broader C6 remains partial.

C6.6 is complete: eligible exact single-System Repulsion setup values are
stored as a `RepulsionCoreSetup` Node. The System's classical cache entry and
`repulsion_setup` refer to that same Node, while direct component/cache APIs
and packed legacy batches continue using `RepulsionSetup`. The setup conversion
preserves tensor identity, the Boolean pair mask, parameter gradients, and the
existing cutoff isolation clone. Broader C6 remains partial; B7 remains
complete, and the unresolved C6.4c NO2 worker-mode diagnosis is not changed.

C6.7 is complete: eligible exact single-System ShortRangeBond setup values are
stored as a `ShortRangeBondCoreSetup` Node, aliased by the System classical
cache entry. The legacy `ShortRangeBondSetup`, direct component/cache APIs, and
packed batches remain supported. Conversion preserves the setup tensors and
counting-function identity without another clone; the legacy and core setups
use the same SRB builder. The existing intermittent small-gap order-2 E0
gradcheck observation remains under tracking; the C6.7 canonical selection
passed, which does not resolve the earlier intermittent observation. Broader
C6 remains partial.

C6.8 is complete: eligible exact single-System IES setup values are stored as
an `IESCoreSetup` Node, and the System's `ies_setup` and classical-cache entry
refer to that same Node. The nested `EEQModel` remains a dynamic Node child;
its `chi`, `kcn`, `eta`, and `rad` tensors are visible pytree leaves and retain
their setup-time parameter graph. Direct `setup_ies()`/`get_cache()` and packed
batch paths continue using `IESSetup`, and both setup forms share the existing
IES energy builder. The IES Node tests passed in the five approved CPU
environments. Broader C6 remains partial. The C6.4c NO2 worker-mode NaN, the
intermittent E0 small-gap gradcheck observation, and the ten inherited GFN1
density-parity failures remain unresolved. The C6.7 broad-suite report of 138
passes and 38 failures remains not fully classified because the complete
failure list was unavailable. Full E3 remains blocked by E1.3.

C6.9 is complete: eligible exact single-System Halogen setup values are held
by `HalogenCoreSetup`, with all ten setup tensors retained as dynamic leaves.
The integer atomic numbers and five Boolean masks preserve their dtypes during
floating-point conversion. Conversion reuses the legacy setup tensors, and
the System's classical-cache entry aliases its core setup. Direct and packed
legacy paths still use `HalogenSetup`; custom subclasses remain on their
legacy route. The installed tad-multicharge 0.7.0 review confirmed that
`EEQModel` exposes its four parameter tensors and nested coordination-model
tensors through its Node tree; no C6.8 correction was indicated. All ten
historical NO2/GFN2 reference rows were validated across low-order,
second-order, and third-order selections: 8 passed and 2 expected XFAIL. The
third-order pair required a separate 495-second two-worker run. Broader C6
remains partial. The NO2 worker-mode NaN, intermittent E0 gradcheck, ten GFN1
density-parity failures, and unclassified C6.7 broad-suite failures remain
separate unresolved issues. C7 scoped setup Node acceptance is complete; C8b remains
blocked by its prerequisites, C9 is deferred, and full E3 remains blocked by
E1.3. The next ownership slice is B6a-Cache.1.

C6.10 is complete: eligible exact single-System classical D4 setup values are
held by `DispersionD4CoreSetup`. Ordered sparse parameters, model tensor
tables, all four cutoff tensors, and optional compatibility charges remain
dynamic pytree leaves; immutable scalar configuration and the built-in
counting/damping strategies remain static descriptors. Each evaluation builds
fresh external model, cutoff, and damping objects from the captured Node.
Direct legacy setup/cache APIs, custom model and damping extensions, and
packed-batch routes remain legacy-owned. The System classical-cache entry
aliases its authoritative D4 Node. Broader C6 remains partial; DFT-D3 and
other legacy component/cache owners remain. The ten inherited GFN1
density-parity failures, intermittent E0 small-gap gradcheck, original C6.4c
NO2 worker-mode NaN, and incompletely classified C6.7 broad-suite failures
remain unresolved and were not addressed here. The classical D4 suite passed
109 tests, and the focused core Node suite passed 9 tests in each of the five
approved CPU environments. The C6.10 E0 selection first had the tracked
`eigensolver.small_gap_aufbau:order2` failure (23/24); a repeat passed 24/24.
The exact base passed 24/24, and isolated reruns passed on base and candidate.
The intermittent observation remains tracked rather than resolved.

C6.11 reviewed the accepted C6.1-C6.10 family contracts and hardened the
classical D4 core path to validate dtype/device before reconstructing call-local
external objects. The established invalid-cache TypeError text is retained.
The acceptance inventory records the remaining generic-cache/B6a-B3b work;
DFT-D3 migration is outside this package. See
[`C6.11 acceptance inventory`](C6.11-C6-acceptance-inventory.md). Broader C6
remains partial.

B6a-D3.1 is complete: exact single-System DFT-D3 setup values are captured in
`DispersionD3CoreSetup` and evaluated by the pure
`dispersion_d3_energy()` path. Direct component/cache APIs, custom subclasses,
and packed routes remain legacy-compatible. B6a-D3.1a lets
`DispersionD3.get_energy()` consume either the legacy cache or the captured
core setup, so the `ClassicalList` compatibility route works without
rebuilding setup from mutable component state. It also rejects distinct exact
D3 components sharing any label while preserving the canonical-label error,
same-object first-occurrence deduplication, and custom labels when unique.
Same-label collisions involving a D3 subclass or a different classical
family retain the existing container policy; resolving those would require a
broader cache-key uniqueness decision.
The legacy distinction between numbers used to gather setup arrays and
numbers used by D3 evaluation remains explicit. B6a-Cache.1 now constructs
each exact built-in setup directly for single Systems and removes persistent
generic built-in cache ownership. `System.classical_cache` remains as an
on-demand compatibility view; custom extension data and packed batches retain
their explicit legacy ownership. Broader B6a/B3b and C6 remain partial, and
E1.3 tad-dftd3 forward-AD support remains blocked upstream.
The single-System boundary rejects duplicate active labels and multiple exact
instances of a family represented by one setup field; D3 remains multi-instance
because its captured setups are associated with components by identity.

Static averaging-callable context does not expose arbitrary tensor-valued
state held by custom callables as dynamic Node leaves. The built-in averaging
functions do not rely on such hidden state; consider separate C6.1 hardening
only if a supported custom-callable use case demonstrates a concrete issue.

**Preconditions.** Core component/setup values satisfy B6b-core. Separately
isolated legacy-only component/cache adapters may remain until E4.6a/E4.6b.

**Steps.** Convert per component (can be split into one PR per component family: Coulomb, dispersion, repulsion, halogen/SRB/IES, solvation, fields). Settings changes go through `replace()`. Component lists keep order (no sets).

**Done when.** No `TensorLike` subclass among components; T0.2 reproduced.

**Needs.** C5, B6b-core. **Size.** L (split per family).

---

## C7 Integrals, Hamiltonian and driver

**Status: C7 setup Node acceptance complete; legacy driver retirement remains
separate.** The schema inventory, production ownership, matched-base
numerical outcomes, and transform evidence are recorded in
[`C7 integral Node schema`](C7.0-integral-node-schema.md). Exact single-System
PyTorch setup owns registered PairPlan and setup Nodes, and the exact
single-System H0 setup owns `H0SetupNode` with explicit `H0IndexNode` maps.
Exact pure single-System libcint evaluation owns `LibcintIntegralSetupNode`
and ordered `LibcintBasisNode` values. Its external wrapper is built without a
live IndexHelper and derives shell/AO maps from the captured shell order;
`LibcintCallData` carries `CoreIndexData` for dxtb's origin shifts. The full
System IndexHelper remains outside the Node for SCF and legacy Calculator
responsibilities. Direct setup dataclasses, Calculator drivers, and packed
routes retain their existing compatibility paths. The accepted-base GFN2 LiH
exponent-gradient failure remains unchanged. No production physics,
references, or tolerances changed.

**Goal.** Migrate the integral description objects, the Hamiltonian and the driver.

**Preconditions.** Builders are pure and objects hold no matrices (B4); driver setup returns a value (B6a/B6b-core).

**Done when.** No `TensorLike` subclass among them; T0.2 reproduced.

**Needs.** C5, B4, B6b-core. **Size.** M.

**Packages.** C7.1 PyTorch setup/PairPlan Nodes, C7.2 H0/index Nodes, C7.3
libcint structural setup/compatibility Nodes, and C7.4 combined acceptance are
complete. C7.4 reviewed whether transforms require the outer `IntegralSetup`
as a pytree input or `IntegralMatrices` as a direct transformed output. The
supported API requires neither: transforms capture setup and return tensor
fields, while public singlepoint results use the registered `Result` pytree.
The validated frozen dataclasses remain unchanged. C7 acceptance covers the
scoped single-System Node setup boundary; legacy Calculator drivers, integral
containers, and packed routes remain in their separately accepted
compatibility-retirement packages.

---

## C8a Result pytree representation

**Status: complete.**

**Goal.** Make the immutable public `Result` returned by
`System.singlepoint()` a valid `torch.func` pytree before E6a. This package is
independent of C5/C6/C7 and does not add mixed-System stackability.

**Steps.** Register or migrate `Result`, `ChargeResult`, `PotentialResult`,
`Multipoles`, and nested result containers as pytree/Node values. `None` fields
and classical-term string labels must live in static pytree context, never as
ordinary mapped leaves; tensors, including each classical energy, remain
dynamic leaves.

**Acceptance.** Keep this package strictly about the output container. Use a
transform-safe probe that builds the complete Result tree from tensor inputs,
independently of SCF or `System.singlepoint` control:

```python
def result_probe(x):
    return Result.snapshot(
        energy=..., scf=..., classical=(("probe", ...),), fenergy=...,
        iterations=..., charges=..., density=..., coefficients=..., emo=...,
        occupation=..., potential=..., hamiltonian=..., overlap=...,
        hcore=..., dipole_integrals=..., quadrupole_integrals=...,
        overlap_norm=..., converged=..., residual=..., status=...,
    )

mapped = torch.func.vmap(result_probe)(xs)
```

Require flatten/unflatten round-trip and default `out_dims=0`; `mapped` remains
a `Result`; every tensor leaf has exactly one new leading dimension; optional
static `None` fields remain `None`; classical labels remain static pytree
context while energy tensors are mapped; nested charges, potential, and
`SCFStatus` reconstruct; and autograd through tensor leaves is preserved.
Cover the complete schema from B1/B5.6, including distinct optional-field
layouts. No `System.singlepoint`, SCF-excluded special case, private tuple
workaround, or custom `out_dims` is part of C8a acceptance. The exact public
B1 call is tested only in E6a after both this pytree and the fixed-trip
controller exist.

Specifically, `trajectory_failure` and `final_failure` are dynamic integer
Tensor leaves: each gains one leading mapped dimension, reconstructs in
`SCFStatus`, preserves dtype/device, and is not stored in pytree context.

**Needs.** B5, B5.6, C1. **Unblocks.** E6a. **Size.** M.

## C8b Model/System stackability

**Goal.** The B3 Model/System objects become stackable nodes so mixed systems
can be composed and models partitioned. C8a separately establishes Result's
transform-compatible pytree representation for E6a.

**Preconditions.** B3b, C5, C6, C7, C8a, and B2 are accepted. C8b.0 records
the source inventory and executable preflight in
[`C8b schema preflight`](C8b-schema-preflight.md). The staged
`ExactSystemCore` remains an opaque frozen dataclass; registered
descriptor/setup Nodes are ready to be enclosed by a bounded core Node.

**Schema and ownership contract.** The registered exact core exposes atomic
numbers, optional H0 setup, one selected integral backend Node, and ordered
classical/interaction descriptor Nodes as dynamic children. Effective dtype,
immutable `Config`, selected driver/integral level, and ordered descriptor
metadata are static context. The integral index is a derived alias of the
selected backend's `CoreIndexData`; do not register it again. Family setup
Nodes are owned by their descriptors; System façade setup fields and identity
associations remain compatibility adapters. `IndexHelper`, component lists,
drivers, caches, positions, charge, spin, and fields stay outside the core.
`IntegralSetup` must be adapted through a registered wrapper with its selected
backend as a child, not retained as opaque context.

**Model parameters.** Keep one Model-level `ParamModule` owner. The production
`Model.parameter_tree` is a registered `ModelParameterNode` containing one
`ModuleNode.from_module(par)` child; named parameters and buffers are dynamic
leaves, while module architecture is static context. The additive
`Model.setup_functional()` boundary applies replacement values through
`torch.func.functional_call` around the existing `Model.setup()` code. Tests
prove that Repulsion and ES2 replacements reach captured setup Nodes and
change singlepoint energies while the original ParamModule remains unchanged.
The public Model/System/Calculator façades and custom component ownership
remain outside this parameter Node, and no parameter tables are copied into
Systems.

Since `functional_call` temporarily substitutes values on that shared
`ParamModule`, Model-mediated `parameter_tree`, `setup()`, and
`setup_functional()` calls using the same owner are serialized by a shared
reentrant lock. This protects parameter snapshots, setup reads, and
restoration. Direct concurrent `ParamModule` access and later evaluation
involving mutable legacy extension state remain outside this guarantee.

Functional state validation compares floating/integer/bool dtype categories
with the owning module's registered state while allowing real floating dtype
conversion. It uses one shallow mapping snapshot for validation and
`functional_call`, retaining the caller's tensor references and gradient
paths.

**Implementation split.** C8b.1 registers the exact-core and selected-backend
wrapper schema and proves conversion/alias behavior. C8b.2 registers the
Model parameter tree and integrates genuine functional parameter substitution
behind the existing Model/System façades. C8b.3 implements compatible
mixed-composition padding/partitioning and the different-composition P9 stack
test. Each stage retains C8a `Result`.

**C8b completion boundary.** C8b registers the core and appropriate Model
representation, partitions Model parameters without duplicating them, stacks
different-composition Systems padded to a common valid shape bucket, and runs
the accepted simple mapped energy kernel over that stack. This is structural
and pure-kernel acceptance, not production mixed-System SCF batching (E6b) or
public `vmap(system.singlepoint)` (E6a). `Calculator.type()` and legacy packed
modes stay at their compatibility boundary; their retirement/replacement is
owned by E4.6/E6b policy, not C8b.

**Preflight result.** tad-mctc 0.9.1 supports Node, ModuleNode,
partition/combine, stack, and conversion. A test-local core wrapper exposes
the selected PyTorch/libcint setup Nodes without duplicating their tensors;
production ExactSystemCore and outer IntegralSetup remain opaque. Independent
GFN1/C2 and GFN1/N2 cores stack and map the existing Repulsion descriptor
kernel with position-gradient parity. GFN2/H2 stacks as a same-composition
control. GFN2/H2 and GFN2/LiH have incompatible pair-plan and shell/AO leaf
shapes despite equal atom count. Full padded mixed-size P9 support remains
C8b.3 because current exact setup maps are unpadded. See the preflight report
for exact tests and limitations.

**Needs.** B3b, C5, C6, C7, C8a, B2. **Unblocks.** E6bimpl after its other
prerequisites; E6a remains separately owned. **Size.** M.

---

## C9 Delete `TensorLike` and `ModuleLike`

**Goal.** Remove the old base classes and every piece of scaffolding.

**Preconditions.** C2-C4 and C8a/C8b are complete, and E4.6a/E4.6b have retired
every legacy batch/FULL fallback object that still inherits from or requires
`TensorLike`/`ModuleLike`. C9 is not complete while such a compatibility
object remains, and C9 does not block E6b.

**Steps.**

1. Delete `TensorLike`, `ModuleLike` and `_clone_tensorlike` in tad-mctc.
2. Delete dxtb's remaining conversion scaffolding.
3. Coordinated releases: tad-mctc, then tad-multicharge, tad-dftd3, tad-dftd4, then the dxtb restructuring release (`00-overview.md` section 8), each pinning the others exactly.

**Done when.** `rg -n "TensorLike|ModuleLike" src/dxtb` finds no remaining
runtime references, and the corresponding source-tree search has been run in
each participating external package (tad-mctc, tad-multicharge, tad-dftd3,
tad-dftd4) before its coordinated release. External scaffolding is removed
and coordinated releases are complete. C9 cannot be marked done while any
runtime reference remains in dxtb or a participating package; all legacy
owners must first be retired or migrated under E4.6a/E4.6b.

**Needs.** C2–C4, C8a/C8b, E4.6a, E4.6b. **Size.** S.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| C1 | `Node` base in tad-mctc (done, 0.9.1) | none | M |
| C1b | `nn.Module` wrapper (done, `ModuleNode`) | C1 | S |
| C2 | tad-mctc classes | C1 | S |
| C3 | tad-multicharge | C1 | S |
| C4 | tad-dftd3, tad-dftd4 | C1, C3 | M |
| C5 | Core structural value for `IndexHelper`/`Basis` | C1, B3a, B6b-core, E2-core, E4core | M |
| C6 | Core components and lists | C5, B6b-core | L |
| C7 | Integrals, Hamiltonian, driver | C5, B4, B6b-core | M |
| C8a | Result pytree representation | B5, B5.6, C1 | M |
| C8b | Model/System stackability of the final System layout | B3b, C5, C6, C7, B2 | M |
| C9 | Final repository-wide `TensorLike`/`ModuleLike` purge | C2–C4, C8a/C8b, E4.6a, E4.6b | S |
