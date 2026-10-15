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
Broader C6 remains partial; C7 remains independently unblocked.

C6.4a is complete: single-System functional FULL and implicit D4SC preparation
now derive model configuration from the same System-owned setup snapshot used
to build call-local dispersion data. Direct compatibility callers still
capture setup from their supplied component, and solver override routing is
unchanged.

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

**Goal.** Migrate the integral description objects, the Hamiltonian and the driver.

**Preconditions.** Builders are pure and objects hold no matrices (B4); driver setup returns a value (B6a/B6b-core).

**Done when.** No `TensorLike` subclass among them; T0.2 reproduced.

**Needs.** C5, B4, B6b-core. **Size.** M.

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

**Precondition.** B3b must already have removed `model`, `batch_mode`, `dd`,
live mutable integrals, and legacy Calculator state from the final System
layout. C8b verifies that boundary and migrates it to Node; it does not own
setup/per-call physics refactoring. Stacking a System must not stack complete
Model parameter tables.

**Steps.**

1. Model and System become `Node` subclasses; retain the Result representation
   established by C8a.
2. Verify stacking: two systems of different composition, padded to the same size bucket, have identical tree structure (P9).
3. Remove `Calculator.type` and any remaining custom conversion; `.to()` comes from the base.

**Done when.** A stacked batch of padded systems runs under `vmap`; `partition` of a model works for training; T0.2 reproduced.

**Needs.** B3b, C5, C6, C7, B2. **Unblocks.** E6bimpl (stacked mixed-System batches), F2c. **Size.** M.

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
