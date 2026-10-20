# C8b.0 — Exact-core Node schema and stacking preflight

Audit base: `3ae533847b93e09c99b61bf3d5afb796cac60dcb`.

This package establishes a source-backed schema and executable prototype. It
does not register or replace production `System`, `ExactSystemCore`, or
`Model`, and it does not implement mixed-System SCF batching.

## Production field and ownership inventory

`ExactSystemCore` currently owns references to `numbers`, effective dtype,
frozen `Config`, outer `IntegralSetup`, optional H0 setup, the selected
`CoreIndexData` alias, and ordered descriptor tuples. The core is a frozen,
non-Node dataclass, so standard pytree traversal treats it as one opaque leaf.
Its `numbers`, `integral_setup`, and `h0_setup` are identity aliases of fields
on the mutable compatibility façade. Descriptor setup children refer to the
same family setup Nodes that the façade and compatibility associations expose.

| Current field(s) | Classification | C8b disposition |
| --- | --- | --- |
| `ExactSystemCore.numbers` | Authoritative integer tensor; aliases `System.numbers` | Dynamic `child(keep_dtype=True)` |
| `ExactSystemCore.effective_dtype` | Immutable validation/configuration scalar | Static context, updated by Node conversion |
| `ExactSystemCore.config` | Frozen static configuration | Static context after recursive tensor/mutability audit |
| `ExactSystemCore.integral_setup` | `IntegralSetup` compatibility value selecting one backend | Registered wrapper with driver/level context and selected backend Node child |
| `ExactSystemCore.h0_setup` | Optional authoritative H0 Node; aliases the System field | Optional dynamic child |
| `ExactSystemCore.index` | Alias of selected backend `.index`, or absent without a Node backend | Derived property; do not register a second child |
| `ExactSystemCore.classical_terms` | Ordered registered descriptor Nodes; each owns its existing setup Node | Dynamic tuple child; no family setup tensor copies |
| `ExactSystemCore.interaction_terms` | Ordered registered descriptor Nodes; each owns its existing setup Node | Dynamic tuple child; call-local fields are excluded |
| `ExactSystemCompatibility` | Component identity tokens, labels, order and setup associations | Legacy façade only; exclude from the core tree |
| `System.batch_mode`, `System.dd` | Legacy setup/runtime compatibility state | Keep on façade; absent from exact-core Node |
| `System.ihelp` | Full mutable/batch-aware `IndexHelper` | Keep on façade; never static Node context |
| `System.classicals`, `System.interactions` | Mutable public component lists | Keep on façade and packed/custom routes |
| `_classical_extension_cache`, `_packed_classical_cache`, `classical_cache` | Extension or packed cache owners / derived view | Compatibility boundary only |
| Family setup fields and `d3_setups` | Repeated aliases plus component identity associations | Descriptors own exact-core dynamic setup references; façade fields remain validated aliases |
| Positions, charge, spin, field, field-gradient | Per-evaluation numeric inputs | Never persist in the core |

The selected backend Node contains the authoritative SCF `CoreIndexData`; the
exact core's index is therefore derived from the selected child. The H0 Node
also contains its `H0IndexNode`, required by H0 evaluation. Those H0 and
backend index tensors have equal values on the tested fixture but are
separately captured setup data, not Python aliases; the prototype adds no
third copy. `System._active_exact_core()` and the legacy compatibility
identity/provenance checks remain façade behavior and are not used as Node
static context.

## Node and integral findings

The active environment reports tad-mctc `0.9.1`, whose supported APIs include
`Node`, `ModuleNode`, `child`, `context`, `Node.to`, `stack`, `partition`, and
`combine`. Nested Nodes, tuples, and `None` children flatten and reconstruct
through the standard PyTorch pytree registry. Integer/Boolean leaves marked
`keep_dtype=True` retain dtype under conversion.

`IntegralSetup` in `src/dxtb/_src/integral/evaluation.py` is an unregistered
frozen dataclass. For both PyTorch and libcint fixtures,
`tree_flatten(system.integral_setup)` returns the outer dataclass as one leaf,
while flattening its selected `PytorchIntegralSetupNode` or
`LibcintIntegralSetupNode` exposes all backend setup tensors. The smallest
adapter is a registered integral wrapper with `driver_type` and `intlevel` in
context and only the selected backend Node as a dynamic child. This preserves
the existing backend Node by identity, carries libcint `force_cpu` through
the backend's own context/conversion override, and neither stores the outer
dataclass opaquely nor copies basis tensors.

The test-local `_IntegralCoreNode` and `_ExactCoreNode` in
`test/test_calculator/test_c8b_schema_preflight.py` demonstrate this boundary.
For PyTorch and libcint, they prove dynamic leaf visibility, no tensor-valued
Node context, flatten/unflatten reconstruction, optional-child handling,
derived index alias coherence, descriptor/setup identity on initial capture,
CPU float64-to-float32 conversion, and preservation of integer and Boolean
leaf dtypes. The production `ExactSystemCore` and `IntegralSetup` remain
opaque, as required in C8b.0.

`Config` and its nested `ConfigIntegrals`, `ConfigSCF`, and `ConfigFermi` are
frozen dataclasses containing scalar, tuple, optional path, and string/integer
configuration. The audit found no tensor or mutable list/dict/set fields.
They can be static context. A stable structural bucket key should use explicit
configuration values plus method, driver, integral level, effective dtype,
libcint device/force-CPU policy, ordered descriptor `(kind, label, position,
optional setup, family configuration)` metadata, and dynamic leaf
shape/dtype/device. Nested Node static contexts also carry shell/pair/angular
and family configuration; canonicalize their typed scalar, tuple, path, enum,
function, and Node-type values recursively. Do not use `repr()` or Python
object identity.

## Model parameter owner

`ParamModule` is an `nn.Module` and is not itself a registered Node. The
preflight wraps its one model-owned parameter module with
`ModuleNode.from_module(par)`. `ModuleNode.params` contains the original
named parameter/buffer tensors (identity checked), and tad-mctc `partition`
and `combine` operate on those paths. A gradient probe reaches selected
floating leaves. The `ModuleNode.module` architecture is static context per
the tad-mctc contract; its dynamic `params` child is the functional state
passed through `functional_call`. This wrapper remains once at Model scope,
never copied into each System. A future Model Node must preserve that
single-owner boundary and test setup-to-energy gradients using real
ParamModule leaves.

Recommended representation: retain the public `Model` façade and expose a
registered Model core/parameter tree containing one `ModuleNode`, immutable
configuration, and explicit immutable metadata. Do not store `ParamModule`
only as static context and do not put a Model parameter table in System.
Legacy custom component prototypes remain outside this core tree under the
E4.6/E6b compatibility policy.

## Structural stack and transform evidence

The focused preflight tests build independent production Systems and wrap
their existing setup Nodes:

| Pair | Result | Evidence |
| --- | --- | --- |
| GFN2/H2 and independent GFN2/H2 | Compatible positive control; stack succeeds | Equal tree specs, leaf shapes/dtypes, leading system dimension, and derived backend-index alias |
| GFN1/C2 and GFN1/N2 | Different compositions with compatible exact-size setup shapes; stack succeeds | Existing Repulsion descriptor kernel under `torch.func.vmap` matches serial energies and position gradients; independent setup storage is asserted |
| GFN2/H2 and GFN2/LiH | Incompatible, despite equal atom count | Pair-plan metadata and shell/AO-resolved dynamic leaf shapes differ; stack is rejected |
| GFN1/H2 and GFN1/LiH | Incompatible | Structural stack rejects the incompatible tree/leaf dimensions |

The C2/N2 positive case proves a different-composition exact-size stack, but
does not satisfy the original P9 requirement for different-size compositions
padded into one atom/shell/AO bucket. Current exact setup maps are unpadded;
therefore C8b.3 still must design and test valid structural padding or
compatible-bucket partitioning for that case. No dummy physical terms may be
inserted. A pure Repulsion kernel maps over the compatible descriptor; this
does not establish full System `vmap`, physical SCF mapping, E6a, or E6b.

## Test and environment record

The focused test file contains seven test functions (nine cases with the
PyTorch/libcint parameterizations). It passed **9/9** in the available Python
3.12.10/Torch 2.14.1 CPU tox environment with tad-mctc 0.9.1, tad-libcint
0.3.0, and PySCF 2.14.0; both optional libraries imported successfully. The
same test passed **7**, skipped **2 libcint cases** by the existing optional
dependency availability in the Python 3.12.10/Torch 2.14.1 no-libcint tox
profile; `has_libcint=False` and `has_pyscf=False` there, and importing
`tad_libcint` failed with `ModuleNotFoundError`. These are two dependency
profiles of one Python/Torch environment, not two of the five approved
Python/Torch combinations. CUDA was unavailable. The focused combined
selection of this module, C6
descriptor audit, B3b layout contract, Model, non-parity singlepoint core,
Result pytree, PyTorch/libcint setup Nodes, and H0 setup Nodes passed **221
tests**, with **12 intentionally deselected** GFN1 legacy-parity cases and
one existing TorchScript deprecation warning. A preceding unfiltered run
reported the ten established GFN1 production/legacy density-parity failures;
the two other parameterizations passed. No current run was performed in the
other four approved environments, so their historical evidence is not
presented as current C8b.0 execution. The descriptor audit and layout/model
controls exercise the accepted setup/ownership paths. No production source,
physics, references, tolerances, or XFAILs changed.

## C8b execution boundary and owners

The original C8b contract is retained as three bounded obligations:

1. **C8b.1 — exact core and backend schema:** register the enclosing exact
   core (or equivalent registered tree), integrate selected backend visibility,
   preserve aliases and dtype/device conversion, and keep the legacy façade
   separate.
2. **C8b.2 — Model parameter partition and façade integration:** use one
   Model-level parameter owner, prove `partition`/`combine` and
   parameter-to-setup-to-energy gradients, and keep setup identities and
   public Calculator/System APIs compatible.
3. **C8b.3 — P9 mixed structure:** define a stable compatible partition key,
   implement supported padding or bucket grouping without dropping setup
   state, then stack different-composition Systems and map a pure energy
   kernel against serial results and gradients.

The original C8b statement that a stacked padded batch runs under a mapped
physical System evaluation is assigned to E6a (whole-System
`vmap(system.singlepoint)`) and E6b (production mixed-System SCF batching).
Those are not silently removed from the roadmap. The original instruction to
remove `Calculator.type()` conflicts with the accepted direct/packed legacy
surface; its retention/retirement policy belongs to E4.6/E6b, while C8b must
preserve the current façade behavior and document known conversion limits.
E1.3, E6a, E6b, E4.6, and legacy component retirement remain independent.

**Disposition:** C8b.0 schema proof is complete. Production C8b is not
complete. The source-backed schema is suitable for bounded C8b.1 work, while
the unpadded different-size P9 stack remains an explicit C8b.3 acceptance
requirement.

## C8b.1 production exact-core implementation

Base: `87e915f42d84200075df9351ac73eb148cd22f7b`.

`ExactSystemCore` is now a registered tad-mctc `Node`. Its dynamic children
are integer-preserving `numbers`, an optional `IntegralBackendNode`, optional
H0 setup, and ordered classical and interaction descriptor tuples. Its static
context is the effective floating dtype and frozen `Config`. The backend
wrapper stores driver and integral level plus the selected backend Node only;
it never stores the legacy outer `IntegralSetup` or the inactive backend.
The existing PyTorch/libcint setup Node is retained by identity at initial
capture. `core.index` is a derived alias of `core.integral.backend.index`, so
the backend remains the sole owner of this SCF map.

`core.integral_setup` remains a compatibility property that constructs a
lightweight outer `IntegralSetup` view around the same selected backend Node.
Evaluation continues to use the System façade's original `IntegralSetup`
after `_active_exact_core()` verifies selected-backend identity, driver,
integral level, H0 identity, descriptor metadata/setup identity and
IndexHelper provenance. The outer facade is not a pytree child. `Calculator.type`
rebuilds the wrapper around its converted backend Node and updates effective
dtype without duplicating setup tensors. Packed modes retain `exact_core=None`.

The backend wrapper carries a logical System-device context so a force-CPU
libcint Node can remain a CPU child while the core is on the requested System
device. This is required by tad-mctc `Node`'s device-homogeneity validation;
`ExactSystemCore.to()` synchronizes that logical context from converted
`numbers` before the enclosing Node is reconstructed. CPU and structural
meta-device conversion are exercised; the latter confirms force-CPU backend
leaves remain CPU while the enclosing core moves, without numerical meta
execution. CUDA was unavailable, so construction and conversion with CUDA
positions plus CPU libcint leaves remain unverified on hardware. The legacy
libcint `force_cpu` setting itself is preserved.

Current validation used Python 3.12.10/Torch 2.14.1 CPU. The focused schema,
descriptor, and B3b layout selection passed 73 tests with libcint and 70
passed/3 skipped in the no-libcint profile. Model/System, C7 setup Node, and
family Node controls passed 252 tests and reproduced the ten established
GFN1 FULL density/orbital parity failures; two other parity parameterizations
remain passing. Functional map controls passed 57; non-slow implicit-root and
E0 controls passed 33 with 3 slow cases deselected. Pure integral evaluation
passed 31 with 1 skip. Libcint overlap, overlap-gradient, and multipole
controls passed 209; the integral-level libcint selection passed 8. No
tolerances or XFAILs changed. The broad regression selection had 10 failures,
all matching the already classified GFN1 parity family; no new failure was
identified. One attempted combined map/root/E0/integral command was
interrupted without a completed result and is not counted.
The broad E3.0 diagnostics module was also interrupted during a long-running
selection; it is not counted. The non-slow functional-root derivative
selection above supplies the completed implicit-root transform controls.

The libcint profile imported tad-libcint 0.3.0 and PySCF 2.14.0 successfully;
the no-libcint profile reported both unavailable and raised
`ModuleNotFoundError` on imports. These are two dependency profiles of one
Python/Torch combination, not two of the five approved combinations. No other
approved Python/Torch environments were available. CUDA was unavailable, so
mixed CUDA-System/CPU-libcint construction and conversion remain unverified.

This closes only C8b.1's exact-core/backend schema scope. Model parameter
partition and façade migration remain C8b.2; different-size P9 padding and
partition acceptance remain C8b.3. Whole-System `vmap` and production mixed
System batching remain E6a and E6b, respectively.

## C8b.2 production Model parameter tree and functional setup

`Model.parameter_tree` now returns a registered `ModelParameterNode` with a
single `ModuleNode.from_module(self.par)` dynamic child, frozen `Config` and
`auto_int_level` static metadata. The ModuleNode's `params` mapping exposes
all `ParamModule.named_parameters()` and `named_buffers()` by their real
module paths. Construction preserves the original tensor identities; no
parameter table is copied into `ExactSystemCore` or any System.

`Model.setup_functional(numbers, parameter_tree, ...)` is the explicit
functional boundary. It validates that the tree's ModuleNode architecture,
configuration and auto-integral-level metadata belong to that Model, then
uses `torch.func.functional_call` with a short-lived `nn.Module` adapter. The
adapter registers the same existing `ParamModule` object and delegates to the
existing `Model.setup()` implementation. Substituted parameter and buffer
values therefore remain active while numbers-only setup Nodes are gathered;
the call returns the usual legacy `System` façade, with its registered
`ExactSystemCore` containing the captured replacement-derived setup Nodes.
`functional_call` restores the original module state after the call. The
adapter and Model façade are not stored in either registered Node.

The C8b.2 hardening review reproduced a shared-module race on the reviewed
implementation: two `Model` façades sharing one `ParamModule` could overlap
`functional_call`, and a bounded interleaving captured call B's Repulsion
parameter in call A's System while call B captured the original module value.
For H2 in Python 3.12.10/Torch 2.14.1 CPU, the requested replacements were
2.5097 and 3.1097: call A captured 3.1097 and call B captured the original
2.2097. The original parameter was still listed but its object identity and
value were not restored; named buffers retained their identities and values
in this interleaving. A weak owner-keyed registry now supplies one `RLock`
per `ParamModule`;
`Model.parameter_tree`, `Model.setup_functional()`, and ordinary
`Model.setup()` hold it. This makes Model-mediated tree snapshots and setup
operations sharing an owner serial and supports nested Model-mediated setup
calls through reentrancy. It does not synchronize direct concurrent mutation
or reads through the raw `ParamModule` API. It does not claim thread safety
for later System evaluation, Calculator cache operations, or arbitrary
extension state. Existing `torch.func.grad` and mixed-derivative
controls run with the lock in place. A small setup-inclusive
`torch.compile(..., backend="eager")` probe (integral-free Repulsion setup)
succeeded on both the reviewed base and candidate with the same value
4.4194; both emitted graph-break warnings from the existing
`IndexHelper.from_numbers(...number.item())` and `Parameter.expand()` paths.
This shows no additional failure in that probe, but is not fullgraph evidence.
A separate full H2 singlepoint compile probe emitted the same tensor-scalar
graph-break and Node-recompile warnings, then was interrupted after more than
two minutes without a result; it is not counted as passing or failing
evidence. The repository's informational transform probe separates
setup-inclusive compilation from compilation of an already-built System's
per-call energy.

The hardening also validates the `ModuleNode` child before dereferencing it,
checks that its state keys and tensor shapes match the owning ParamModule, and
preserves support for converted floating dtypes and non-Parameter tensor
leaves. It accepts real floating replacements for floating state and requires
nonfloating structural state to retain its registered dtype; in particular,
float64-to-float32 trees remain supported while integer, Boolean, and complex
substitutions for floating parameters are rejected before setup. State
validation and `functional_call` share one shallow snapshot of the input
mapping and its tensor references. These metadata-only checks precede
`functional_call`; no tensor values are synchronized or compared. A float64
parameter tree converted to float32 has a numerical acceptance control against
a separately constructed float32 Model, including setup values, H2 energy,
and a real parameter gradient. The GFN1 `hamiltonian.xtb.kpol` path similarly
proves functional substitution through `H0SetupNode`, H2 energy, and gradients
against a direct ParamModule reference.

Failure-path controls cover invalid batch mode after substitution begins,
original parameter and buffer object/value restoration, subsequent ordinary
and functional setup, mode-1 packed setup, and a custom Repulsion subclass
retaining its legacy cache hook and component identity. CUDA was unavailable;
converted CUDA setup remains unverified.

The concurrency regressions now observe entry into the owner-lock acquisition
and the protected `IndexHelper.from_numbers`/`ModuleNode.from_module` stages.
A lock-bypassed test-only run makes each ordinary-setup and parameter-snapshot
regression fail at its corresponding overlap assertion. The synchronization
contract remains limited to Model-mediated setup and parameter-tree snapshots.

Executable proof uses the actual GFN1 `parameter_tree.element.H.arep.param`
path. The replacement reaches `RepulsionCoreSetup.arep`, changes the full H2
singlepoint energy, matches an independently evaluated Model whose
`ParamModule.get("element.H.arep")` value was updated, and supports
`torch.func.grad`, autograd, position gradients and a mixed
position/parameter derivative. The `p`, `p + delta`, `p` sequence recovers the
initial setup and energy without mutating the source module. A separate GFN1
ES2 `charge.effective.gexp` replacement reaches `ES2CoreSetup.gexp`, changes
the H2O singlepoint energy, and has a nonzero parameter gradient. These tests
exercise the existing Model setup and numerical evaluation paths; they do not
establish whole-System `vmap` or mixed-System batching.

The C8b.1 wrapper now checks that a present backend Node has the concrete type
selected by its driver. `_active_exact_core()` also requires a non-absent
selected backend and matching logical System device before trusting the core.
Node construction already rejects a wrapper logical device inconsistent
with the core number tensor; the regression proves this through `Node.replace`.
The logical device remains distinct from libcint force-CPU tensor placement.
An absent integral setup remains represented by `core.integral is None`.

**Independent hardening rerun.** The focused Model parameter-tree module
passed 24 tests in the libcint-enabled profile. In the no-libcint profile,
the module plus schema tests reported 29 passed and 4 libcint-specific skips.
C8b schema plus B3b layout contracts reported 84 passed. Model/singlepoint
controls reported 74 passed and the same ten GFN1 production/legacy parity
failures already listed below. Non-slow functional-map, implicit-root and E0
controls reported 95 passed and one deselected. PyTorch/H0/libcint setup-node
plus PyTorch integral controls reported 68 passed; libcint overlap,
overlap-gradient, multipole and integral controls reported 217 passed; packed
acceptance and B3b layout controls reported 52 passed. All execution used
Python 3.12.10/Torch 2.14.1+cpu and tad-mctc 0.9.1. The enabled profile had
importable tad-libcint 0.3.0 and PySCF 2.14.0; neither is installed in the
no-libcint profile. CUDA and the other four approved Python/Torch combinations
were unavailable.

Focused C8b.2 execution on Python 3.12.10/Torch 2.14.1 CPU with
tad-mctc 0.9.1 passed the new Model parameter-tree and setup tests plus the
C8b schema and B3b layout contracts: **70 passed**. The no-libcint profile
passed the new Model tests and schema preflight: **15 passed, 4 skipped**;
the skips are the existing libcint-only and cross-backend cases. The
Calculator/Model/singlepoint and C7 setup-node selection completed with
**124 passed and 10 failed**; all ten failures are the previously classified
GFN1 production/legacy density-orbital parity cases, with no new failure.
Non-slow functional-map, implicit-root derivative and E0 iteration controls
completed with **119 passed, 4 deselected**. The selected libcint integral,
overlap, overlap-gradient, and multipole controls completed with **239
passed**; packed legacy acceptance passed **1** test. The libcint profile had
tad-libcint 0.3.0 and PySCF 2.14.0 installed and importable; CUDA was
unavailable. These are two dependency profiles of one Python/Torch
combination, not two of the five approved Python/Torch combinations. The
other four approved combinations are unavailable in this checkout and are
not represented by current C8b.2 execution.

**Disposition:** C8b.2 is implemented at the Model parameter ownership and
functional setup boundary while retaining the public Model/System/Calculator
façades. C8b.3 still owns different-size P9 padding/partitioning and full
structural stacking acceptance. C8b remains incomplete.
