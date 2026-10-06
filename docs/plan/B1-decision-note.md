# B1 Decision note: evaluation API and state

Status: **proposal, for agreement.** Package B1 of `02-TB-evaluation-api.md`.
Written against `main` at `3b6a90f` plus the Track 0 branch. Evidence for each
point is in `T0-baseline-report.md` (section numbers in brackets).

The note fixes the shape of the API so that B2-B8 and C8/F1 can start. Where a
choice is open, the proposal comes first, then the alternative and why it was
not taken. Items marked **[decide]** need an explicit answer before B2 starts.

## 0. Constraints from the baseline

| Finding | Consequence for the API |
| --- | --- |
| Caches gave wrong results (stale graphs, zeros for numerical field derivatives, views) [4] | No cache of any kind in the core. Reuse is an explicit value. |
| `Calculator.forces` fails for batches; `functorch` modes return `(nb, nb, ...)` blocks [3.1] | Property functions are single-system. Batches go through `vmap` or a batch entry point with per-system semantics. |
| `vmap` stops at `data_ptr` cache keys, batch-size heuristics on `numbers`, data-dependent control flow [5] | Everything that depends on `numbers` alone moves into `setup`. The per-call path has fixed shapes. |
| `reset()` detaches user tensors; field updates mutate a component [9] | Field, charge and spin are call inputs. |
| Implicit SCF is right under autograd but not under `torch.func` [3.1, 9 E0.1] | The result and property functions must not assume which SCF mode produced them. The mode is a model setting. |
| Parameter gradients reach every leaf except `refocc` (since #271) [7] | The model keeps parameters as leaves a transform can differentiate. `refocc` must be fixed before F1 relies on it. |
| Batched eigensolver stall solved by a per-call `l_inv` [6] | The overlap-dependent factorisation belongs to the per-call data, not to the system. |

## 1. Levels (question 1)

**Proposal: three levels, as in P1.**

| Level | Holds | Built by | Methods |
| --- | --- | --- | --- |
| `Model` | parameters (per-element tables as tensors, trainable or not), frozen configuration, the list of terms | `dxtb.GFN1(...)`, `dxtb.GFN2(...)`, `dxtb.GFN0(...)`, or `Model(par, config, terms)` | `setup(numbers)`, `replace(...)`, `to`-style tree maps (C8) |
| `System` | everything that depends on `numbers` alone: index helper, basis, gathered per-atom and per-shell parameters, classical-term setup data | `model.setup(numbers)` | `singlepoint(positions, chrg, spin, field, field_grad)`, `energy(...)` |
| `Result` | outputs of one SCF (section 2) | `system.singlepoint(...)` | none; plain data |

- `Model` and `System` are frozen `Node`s (C1) once `Node` lands. Until then
  they are frozen dataclasses with the same fields, so C8 only changes the base.
- `System` keeps a reference to the model's parameters, not a copy, so a
  gradient through `setup` reaches the parameter leaves (T0.9 test).
- `Model` carries no `numbers`, no device, no dtype (item 7).

**Alternative rejected:** keep one `Calculator(numbers, par)` and make it
immutable. It would keep `numbers` and parameters in one object, so stacking
systems of different composition (C8) and training with `torch.func` (F1) would
still need the split.

## 2. Result contents (question 2)

**Proposal: everything, always.** Memory is the only cost, and the SCF holds
these tensors anyway.

| Group | Fields |
| --- | --- |
| Energies | `energy` (total), `scf` (per atom), `classical` (dict by term), `fenergy` (free-energy term) |
| SCF state | `charges` (atom, shell, orbital), `multipoles` (dipole, quadrupole, where the method has them), `density`, `coefficients`, `emo`, `occupation`, `potential`, `hamiltonian` |
| Integrals | `overlap`, `h0`, dipole and quadrupole integrals when built |
| Diagnostics | `iterations`, `converged` (bool tensor per system), `residual` |

The current `Result` (`calculators/result.py`) already has most of these;
`converged`, `iterations` and `multipoles` are new. The result is a frozen
`Node`; `integrals` becomes a plain tuple of tensors, not the live
`IntegralMatrices` object (B4).

Analytical forces and bond orders take a result as input (B8 step 2); they read
`density`, `emo`, `coefficients`, `overlap` from it, never from a store.

## 3. Property functions (question 3)

**Proposal: both forms, with one implementation.**

```python
# primary: pure functions of the system and the call inputs
dxtb.forces(system, positions, chrg=0, spin=None, field=None)
dxtb.hessian(system, positions, ...)            # (nat, 3, nat, 3)
dxtb.third_order(system, positions, ...)        # (nat, 3, nat, 3, nat, 3)
dxtb.dipole(system, positions, ...)
dxtb.polarizability(system, positions, ...)
dxtb.hyperpolarizability(system, positions, ...)
dxtb.dipole_derivative / polarizability_derivative / ir / raman / vibration
dxtb.forces_numerical(system, positions, step=..., ...)   # same signature

# shortcut when a result already exists (analytical forces, bond orders)
dxtb.forces_analytical(system, result)
dxtb.bond_orders(system, result)
```

- The primary form recomputes the SCF inside the transform. That is what
  `torch.func` needs, and it is the only form that composes to higher order.
- Shortcuts exist only where the derivative is a closed expression in the
  result (analytical forces, bond orders). Their input is a result.
- `ir`/`raman`/`vibration` compose the lower-order functions; no manual resets
  (B6 step 5).
- Names follow the current methods (`dipole_deriv` becomes `dipole_derivative`,
  `pol_deriv` becomes `polarizability_derivative`) because the migration guide
  lists renames anyway **[decide: keep the short names?]**.
- Method versions on `System` (`system.forces(positions)`) are thin
  delegates, documented as sugar. They hold no logic.

**Alternative rejected:** property functions of a result only. A result is a
value; it cannot be differentiated through, so higher orders would need the
SCF repeated anyway.

## 4. Fields (question 4)

**Confirmed.** `field` and `field_grad` are arguments of `singlepoint` (B7).
`None` means no field and no field term in the graph. Charge and spin are
arguments too, so `d/d chrg` works with the fractional electron count of #271.

## 5. Batching (question 5)

**Proposal.**

1. Core functions accept one system, possibly padded. No batch dimension in
   physics code (P2).
2. Conformers: `torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`.
   Property functions are written so the same `vmap` pattern gives per-system
   results with no cross-system blocks.
3. Mixed compositions: `dxtb.batch.setup(model, numbers_batch)` returns stacked
   padded systems (C8). `dxtb.batch.singlepoint` and the property variants
   take them and loop or `vmap` internally. Batch size is never inferred
   from shapes.
4. `batch_mode` and the shape-based guessing are removed (104 references today).
5. The SCF driver loops over entries until `vmap` rules exist for the
   eigensolver (E6a). Until then `dxtb.batch.*` is the supported path and
   `vmap` over the whole singlepoint is the target.

**[decide]** whether a batched call with a single composition and many
geometries gets a dedicated `setup(numbers)` + `positions[nb, nat, 3]` entry
(the common CREST case) or only the generic `dxtb.batch.*`. Proposal: the
former is just `vmap` over positions and needs no extra function.

## 6. Parameter storage (question 6)

**Proposal.**

- The model holds per-element tables as tensors, indexed by atomic number, one
  row per element up to `max_element`. The tables are the pytree leaves.
- Trainable subsets (F2c): `partition(model, selector)` returns a model-like
  tree holding only the selected leaves and the remainder; `combine` rebuilds.
  `setup` gathers from the combined tables. The selection is by tree path
  (`element.H.refocc`), not by index, so the existing `ParamModule` names
  survive.
- Structural parameters (shell layout, number of shells, `max_element`) are
  context fields, never leaves (F2a decides which parameters are structural).
- The gradient of the energy with respect to a gathered table row must equal
  the T0.9 table; B3 gates on it.

`Param`/`ParamModule` stay as the file format and the default constructor input
until F1; the model converts once in its constructor.

## 7. Dtype and device (question 7)

**Confirmed.** Neither is stored in the configuration or on the model. Both are
read from the input tensors (`positions`, then the parameter tables).
`dxtb.GFN1(dtype=..., device=...)` stays as a constructor convenience that
creates the tables; afterwards `model.to(...)` is a tree map (C8). Float64 is
the documented requirement; float32 inputs raise a clear error in the SCF, not
silently run.

## 8. Numerical derivatives (question 8)

**Confirmed.** `*_numerical` functions take the same arguments as the analytical
ones plus `step`, call the primary functions with perturbed inputs, and mutate
nothing. The T0.5 finding (zeros for numerical field derivatives with the cache)
disappears with the cache.

## 9. ASE (question 9)

**Confirmed.** Out of the core, optional adapter in B9, holding the last result.

## 10. Removal list (question 10)

Basis of the migration guide. Replacement in the second column.

| Removed | Replacement |
| --- | --- |
| `Calculator(numbers, par, opts=...)` | `model = dxtb.GFN1(config)`, `system = model.setup(numbers)` |
| `calc.singlepoint(positions, chrg, spin, **store_kwargs)` | `system.singlepoint(positions, chrg, spin, field, field_grad)` |
| `calc.cache`, `CalculatorCache`, `@cdec.cache`, `opts.cache`, `store_*` | `Result` fields |
| `calc.get_energy`, `get_forces`, ... (`GetPropertiesMixin`) | `dxtb.energy`, `dxtb.forces`, ... |
| `calc.forces_analytical(positions)` reading the cache | `dxtb.forces_analytical(system, result)` |
| `calc.reset()`, `Component.update`, `reset`, `cache_*`, `ComponentList.reset_all`, `Integrals.reset_all` | none; objects are frozen |
| `interactions.update_efield`, `update_efield_grad`, `requires_efield*` | `field=`, `field_grad=` arguments |
| `opts.batch_mode`, shape-based mode guessing | `dxtb.batch.*` or `vmap` |
| `opts.device`, `opts.dtype`, `Calculator.type()` | tree maps on the model |
| `Config` mutation (`calc.opts.scf.x = ...`) | `config.replace(...)` |
| `IntDriver.is_latest`, `_positions`, `invalidate` | `setup` returns a value |
| `TensorLike`, `ModuleLike` | `Node` (Track C) |
| `scf_mode="nonpure"` | already removed (#272) |
| `dxtb.calculators.GFN1Calculator` etc. | `dxtb.GFN1`; the old names raise with a pointer for one release |

## 11. Settings that stay in the configuration

Configuration (frozen, hashable, pytree context): method, SCF settings (mode,
tolerances, mixer, damping, maximum iterations, fermi settings, guess), integral
settings (driver, level, cutoffs), exclusions, `max_element`. These are the
context part of the tree: two models with different settings have different
tree structure and are never stacked.

## 12. Open questions the proposal leaves

1. **[decide]** Short or long names for the derivative properties (section 3).
2. **[decide]** Whether `System` also holds the *geometry-independent
   interaction data* of the self-consistent terms (ES2 shell hardness gather,
   ES3 parameters) or only the classical terms. Proposal: both; the
   T0.9 test already covers them.
3. **Per-call data.** The Coulomb matrices, the Cholesky factor `l_inv`, the
   integrals depend on positions. They live in the per-call interaction cache
   inside `singlepoint` and in the result (B6), not in the system. The
   Cholesky factor is the one place where this costs repeated work in a
   transformed function; measure before adding anything (P10).
4. **The `refocc` derivative** must be fixed or the leaf excluded before F1.
5. **Release placement.** The new API is dxtb release 1 (overview, section 8).
   The old API raises with a migration pointer for that release only.

## 13. Acceptance

The note is agreed when sections 1-10 are confirmed or amended and the items
marked **[decide]** are answered. Then `00-overview.md` links it and B2 starts.
