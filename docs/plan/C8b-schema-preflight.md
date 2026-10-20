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
