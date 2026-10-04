# Track C: `Node` migration and retiring `TensorLike`

**Purpose.** Replace `TensorLike` (reflective cloning over `__slots__`) with `Node`: frozen dataclasses registered as pytrees. Conversion, batching (`vmap` over stacked systems) and parameter training (`partition`/`combine`) then work through standard tree operations.

**Relation to the design review.** The `TensorLike` design review (revision 8) remains the reference for the problems with `TensorLike` and the probe results. This track supersedes it where they differ:

| Design review (rev. 8) | This plan |
| --- | --- |
| `derived` field kind for caches and setup state | dropped: TB removes that state first |
| interim cache-write helper using `object.__setattr__` | dropped |
| compile minimum torch 2.8 and wrapper | moved to TD (D4) |
| `.to(device)` interim fix | dropped; documented as known issue (T0.6) |
| deleting `TensorLike` in the release that adds `Node` | deletion is a separate final step (C9) |
| caches as explicit values (step 4) | replaced by TB |

**Order.** C1–C4 can start immediately. dxtb's own classes (C5–C8) migrate after the TB packages that remove their mutable state, from the bottom of the object tree up.

---

## C1 `Node` base in tad-mctc

**Goal.** Ship the base class in tad-mctc 0.8 alongside the unchanged `TensorLike`.

**Specification.**

1. **Field kinds.** Two only:
   - `child`: tensor, `None`, node, or a list/tuple/dict of tensors and nodes;
   - `context`: static Python values for method-level settings; hashable and value-comparable; never a tensor, node or per-molecule value (P9).
2. **Class creation.** `__init_subclass__` applies `dataclasses.dataclass(frozen=True, eq=False, kw_only=True)` itself, builds the field layout, checks it and registers the class as a pytree node. Subclasses don't write the decorator. Every subclass is registered automatically (registration is per exact type).
3. **Layout check at class definition.** Each class declares only its own fields; the base combines them along the MRO. Raises on: untagged fields, annotated names that aren't fields (excluding `ClassVar` and `InitVar`; no exemption for names starting with `_`), double-underscore field names. Uses `inspect.get_annotations` (Python 3.14 hides annotations from `__dict__`).
4. **Pytree registration.** Flatten returns child fields as leaves and context fields as context; unflatten uses `object.__new__` and assigns fields directly. Provide `flatten_with_keys_fn` and `serialized_type_name` (needed by `torch.export`).
5. **Construction checks** (run in the generated `__init__`, never in unflatten):
   - no tensors or nodes in context fields;
   - no Python scalars in child fields;
   - all floating children share one dtype;
   - all tensors in the object and its child nodes share one device (no exemption for child nodes);
   - metadata only: types, dtypes, devices, shapes; never tensor values, so construction stays traceable (`vmap`, compile).
6. **`dtype` and `device`.** Read from the object's own first floating (or any) tensor; recurse into children only if the object has none; raise a clear error if there is none.
7. **Conversion.** `.to(device=None, dtype=None)` and `.type(dtype)` are tree maps over leaves. Floating tensors get the dtype; all tensors get the device. Any leaf that isn't a tensor or `None` raises (strict map).
8. **`replace(**changes)`** returns a checked copy. Return type `Self` for `replace`, `.to` and `.type`.
9. **Training helpers.** `partition(tree, predicate)` and `combine(params, rest)`.
10. **Type checking.** `typing.dataclass_transform` on the base, with field specifiers whose signatures declare `default`, `default_factory`, `init` and `kw_only`, so pyright sees defaults correctly. A pyright check in CI over a usage file.
11. **Rules enforced in CI.** `object.__setattr__` appears only in the `Node` base (grep check).
12. **Support.** Python ≥ 3.10 for tad-mctc 0.8.

**Tests.**

- Layout check fires for each error case.
- Conversion round trip for dtype and device; strict map raises on foreign leaves.
- `vmap` over stacked nodes (`in_dims=0`) and over a shared node (`in_dims=None`).
- `jacrev` with respect to a partitioned subset of leaves, compared with finite differences.
- Stacking two nodes with equal structure; stacking with different context raises.

**Done when.** tad-mctc 0.8 released with `Node`; `TensorLike` unchanged.

**Needs.** None. **Unblocks.** everything else in TC; F2c (partition). **Size.** M.

---

## C1b `nn.Module` wrapper for ML terms

**Goal.** A defined way to hold a user's `nn.Module` inside a node tree (needed by F3).

**Rule.**

- The module's architecture (the module object, stripped of state, or a factory) is context.
- Its parameters and buffers are children (a dict of tensors).
- The module is always called through `torch.func.functional_call` with the parameters from the tree, never with its own stored parameters.

**Steps.** Implement `ModuleNode` in tad-mctc (or dxtb) with `from_module(module)` and `__call__` via `functional_call`; test conversion, `vmap`, `jacrev` with respect to the module's parameters, and that two copies share architecture context.

**Done when.** A small MLP inside a node converts with `.to()`, trains through `partition`/`combine`, and works under `vmap`.

**Needs.** C1. **Unblocks.** F3. **Size.** S.

---

## C2 tad-mctc classes

**Goal.** Migrate tad-mctc's own `TensorLike` classes (`Mol`, the neighbor list).

**Done when.** No `TensorLike` subclass left in tad-mctc; released in 0.8 or 0.8.x.

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

1. Delete `TensorLike`, `ModuleLike` and `_clone_tensorlike` in tad-mctc 0.9.
2. Delete dxtb's remaining conversion scaffolding.
3. Coordinated releases: tad-mctc 0.9, then tad-multicharge, tad-dftd3, tad-dftd4, then dxtb release 2, each pinning the others exactly.

**Done when.** `grep -r TensorLike` finds nothing in any of the packages; all released.

**Needs.** C2–C4, C8. **Size.** S.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| C1 | `Node` base in tad-mctc | none | M |
| C1b | `nn.Module` wrapper | C1 | S |
| C2 | tad-mctc classes | C1 | S |
| C3 | tad-multicharge | C1 | S |
| C4 | tad-dftd3, tad-dftd4 | C1, C3 | M |
| C5 | `IndexHelper`, `Basis` | C1, B3, E4 | M |
| C6 | Components and lists | C5, B6 | L |
| C7 | Integrals, Hamiltonian, driver | C5, B4, B6 | M |
| C8 | Model, system, result as nodes | C6, C7, B5, B2 | M |
| C9 | Delete `TensorLike` | C2–C4, C8 | S |
