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

**Preconditions.** `IndexHelper.cull`/`restore` are gone (E4); both classes are built in `setup` (B3).

**Steps.**

1. Convert to `Node`; integer index tensors are children; per-molecule counts are tensors, not context (P9).
2. Remove `allowed_dtypes` (integer handling now comes from the strict conversion rule: integer tensors keep their dtype).

`batch_mode` is not migrated. IndexHelper becomes a single-system structural
Node after E4 deletes cull/restore.

**Done when.** Both migrated; conversion moves their tensors with the parent's `.to(device)`; T0.2 reproduced.

**Needs.** C1, B3, E4. **Size.** M.

---

## C6 Components and component lists

**Goal.** Migrate all classical and interaction components and their list containers.

**Preconditions.** No caches or mutation methods remain (B6).

**Steps.** Convert per component (can be split into one PR per component family: Coulomb, dispersion, repulsion, halogen/SRB/IES, solvation, fields). Settings changes go through `replace()`. Component lists keep order (no sets).

**Done when.** No `TensorLike` subclass among components; T0.2 reproduced.

**Needs.** C5, B6. **Size.** L (split per family).

---

## C7 Integrals, Hamiltonian and driver

**Goal.** Migrate the integral description objects, the Hamiltonian and the driver.

**Preconditions.** Builders are pure and objects hold no matrices (B4); driver setup returns a value (B6).

**Done when.** No `TensorLike` subclass among them; T0.2 reproduced.

**Needs.** C5, B4, B6. **Size.** M.

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

**Needs.** C6, C7, B5, B2. **Unblocks.** E6 (stacked batches), F2c. **Size.** M.

---

## C9 Delete `TensorLike` and `ModuleLike`

**Goal.** Remove the old base classes and every piece of scaffolding.

**Steps.**

1. Delete `TensorLike`, `ModuleLike` and `_clone_tensorlike` in tad-mctc.
2. Delete dxtb's remaining conversion scaffolding.
3. Coordinated releases: tad-mctc, then tad-multicharge, tad-dftd3, tad-dftd4, then the dxtb restructuring release (`00-overview.md` section 8), each pinning the others exactly.

**Done when.** `grep -r TensorLike` finds nothing in any of the packages; all released.

**Needs.** C2–C4, C8. **Size.** S.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| C1 | `Node` base in tad-mctc (done, 0.9.1) | none | M |
| C1b | `nn.Module` wrapper (done, `ModuleNode`) | C1 | S |
| C2 | tad-mctc classes | C1 | S |
| C3 | tad-multicharge | C1 | S |
| C4 | tad-dftd3, tad-dftd4 | C1, C3 | M |
| C5 | `IndexHelper`, `Basis` | C1, B3a, E4 | M |
| C6 | Components and lists | C5, B6b | L |
| C7 | Integrals, Hamiltonian, driver | C5, B4, B6b | M |
| C8 | Model, system, result as nodes | C6, C7, B5, B2 | M |
| C9 | Delete `TensorLike` | C2–C4, C8 | S |
