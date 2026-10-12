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

**Order.** C1/C1b are done. C2–C4 are external cleanup (verify state first). dxtb's own classes (C5–C8) migrate after the TB packages that remove their mutable state, from the bottom of the object tree up.

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
batch-aware `IndexHelper` until E6b retires that route; E4.6a then removes
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

**Needs.** C1, B3, B6b-core, E2, E4. **Size.** M.

---

## C6 Components and component lists

**Goal.** Migrate all classical and interaction components and their list containers.

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

## C8 Model, system and result as nodes

**Goal.** The B3/B5 objects become nodes, so systems can be stacked and models partitioned.

Before converting System to Node, remove `model`, `batch_mode`, `dd` and all
live mutable builder/cache objects. Stacking a System must not stack complete
Model parameter tables.

**Steps.**

1. Model, system and result become `Node` subclasses.
2. Verify stacking: two systems of different composition, padded to the same size bucket, have identical tree structure (P9).
3. Remove `Calculator.type` and any remaining custom conversion; `.to()` comes from the base.

**Done when.** A stacked batch of padded systems runs under `vmap`; `partition` of a model works for training; T0.2 reproduced.

**Needs.** C6, C7, B5, B2. **Unblocks.** E6b (stacked mixed-System batches), F2c. **Size.** M.

---

## C9 Delete `TensorLike` and `ModuleLike`

**Goal.** Remove the old base classes and every piece of scaffolding.

**Preconditions.** C2-C4 and C8 are complete, and E4.6a/E4.6b have retired
every legacy batch/FULL fallback object that still inherits from or requires
`TensorLike`/`ModuleLike`. C9 is not complete while such a compatibility
object remains, and C9 does not block E6b.

**Steps.**

1. Delete `TensorLike`, `ModuleLike` and `_clone_tensorlike` in tad-mctc.
2. Delete dxtb's remaining conversion scaffolding.
3. Coordinated releases: tad-mctc, then tad-multicharge, tad-dftd3, tad-dftd4, then the dxtb restructuring release (`00-overview.md` section 8), each pinning the others exactly.

**Done when.** `rg -n "TensorLike|ModuleLike" src/dxtb` finds no remaining
runtime references; external scaffolding is removed and coordinated releases
are complete. C9 cannot be marked done while any runtime reference remains;
all legacy owners must first be retired or migrated under E4.6a/E4.6b.

**Needs.** C2–C4, C8, E4.6a, E4.6b. **Size.** S.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| C1 | `Node` base in tad-mctc (done, 0.9.1) | none | M |
| C1b | `nn.Module` wrapper (done, `ModuleNode`) | C1 | S |
| C2 | tad-mctc classes | C1 | S |
| C3 | tad-multicharge | C1 | S |
| C4 | tad-dftd3, tad-dftd4 | C1, C3 | M |
| C5 | Core structural value for `IndexHelper`/`Basis` | C1, B3a, B6b-core, E2, E4 | M |
| C6 | Core components and lists | C5, B6b-core | L |
| C7 | Integrals, Hamiltonian, driver | C5, B4, B6b-core | M |
| C8 | Model, system, result as nodes | C6, C7, B5, B2 | M |
| C9 | Final repository-wide `TensorLike`/`ModuleLike` purge | C2–C4, C8, E4.6a, E4.6b | S |
