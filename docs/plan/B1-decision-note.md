# B1 Decision note: evaluation API and state

Status: **agreed** (conformers: plain `vmap`, no extra entry point; short property names, setup data of
self-consistent terms in `System`, `refocc` fix as requirement T0.12). Package B1 of `02-TB-evaluation-api.md`.
Written against `main` at `3b6a90f` plus the Track 0 branch. Evidence for each
point is in `T0-baseline-report.md` (section numbers in brackets).

The note fixes the shape of the API so that B2-B8 and C8a/C8b/F1 can start. Where a
choice is open, the proposal comes first, then the alternative and why it was
not taken. Items marked **[decide]** needed an explicit answer before B2 starts; the remaining ones are listed in section 5 and 12.

## 0. Constraints from the baseline

| Finding | Consequence for the API |
| --- | --- |
| Caches gave wrong results (stale graphs, zeros for numerical field derivatives, views) [4] | No cache of any kind in the core. Reuse is an explicit value. |
| `Calculator.forces` fails for batches; `functorch` modes return `(nb, nb, ...)` blocks [3.1] | Property functions are single-system. Batches go through `vmap` or a batch entry point with per-system semantics. |
| `vmap` stops at `data_ptr` cache keys, batch-size heuristics on `numbers`, data-dependent control flow [5] | Everything that depends on `numbers` alone moves into `setup`. The per-call path has fixed shapes. |
| `reset()` detaches user tensors; field updates mutate a component [9] | Field, charge and spin are call inputs. |
| Implicit SCF is right under autograd but not under `torch.func` [3.1, 9 E0.1] | The result and property functions must not assume which SCF mode produced them. The mode is a model setting. |
| Parameter gradients reached every leaf except `refocc` (#271 until T0.12, fixed) [7] | The model keeps parameters as leaves a transform can differentiate. |
| Batched eigensolver stall solved by a per-call `l_inv` [6] | The overlap-dependent factorisation belongs to the per-call data, not to the system. |

## 1. Levels (question 1)

**Proposal: three levels, as in P1.**

| Level | Holds | Built by | Methods |
| --- | --- | --- | --- |
| `Model` | parameters (per-element tables as tensors, trainable or not), frozen configuration, the list of terms | `dxtb.GFN1(...)`, `dxtb.GFN2(...)`, `dxtb.GFN0(...)`, or `Model(par, config, terms)` | `setup(numbers)`, `replace(...)`, `to`-style tree maps (C8b) |
| `System` | everything that depends on `numbers` alone: index helper, basis, gathered per-atom and per-shell parameters, classical-term setup data | `model.setup(numbers)` | `singlepoint(positions, chrg, spin, field, field_grad)`, `energy(...)` |
| `Result` | outputs of one SCF (section 2) | `system.singlepoint(...)` | none; plain data |

- `Model` and `System` are frozen values. C8b makes them stackable `Node`s;
  C8a separately gives the immutable Result tree transform-compatible pytree
  structure, including static optional/string metadata.
- `System` does not keep the complete Model or complete parameter tables.
  `setup` gathers the tensors needed by this composition, and those tensors
  retain their autograd relationship to the Model parameter leaves. A
  differentiated/training evaluation therefore performs `setup` from the
  current Model inside the loss. This avoids duplicating complete Models when
  Systems are stacked (T0.9 test).
- `Model` carries no `numbers`, no device, no dtype (item 7).

**Alternative rejected:** keep one `Calculator(numbers, par)` and make it
immutable. It would keep `numbers` and parameters in one object, so stacking
systems of different composition (C8b) and training with `torch.func` (F1) would
still need the split.

## 2. Result contents (question 2)

**Decision: close the schema before C8a freezes its pytree definition.** The
canonical names and representations are:

| Name | Representation and pytree role |
| --- | --- |
| `energy` | Actual `Tensor` field; dynamic total-energy leaf. |
| `scf` | Actual `Tensor | None` field; dynamic leaf when present, optional static structure when absent. Per-atom SCF energy. |
| `classical` | Actual tuple of `(label, Tensor)` pairs; tuple order/labels are static structure and each energy is a dynamic leaf. It is not a dictionary field. |
| `fenergy` | Actual `Tensor | None` field; dynamic leaf when present, optional static structure when absent. |
| `charges` | Actual nested `ChargeResult | None` field; monopole is a dynamic leaf; optional dipole/quadrupole are static structure when absent and dynamic leaves when present. |
| `multipoles` | Derived property over `charges`, not an independent field or duplicated pytree leaf. It returns the dipole/quadrupole view as `Multipoles`. |
| `density`, `coefficients`, `emo`, `occupation` | Actual optional tensor fields; dynamic leaves when present. |
| `potential` | Actual nested `PotentialResult | None` field; each present component is a dynamic leaf and absent components are static structure. |
| `hamiltonian`, `overlap`, `hcore`, `dipole_integrals`, `quadrupole_integrals`, `overlap_norm` | Actual optional tensor fields; dynamic leaves when present. `hcore` is canonical; `h0` is not a Result field. |
| `iterations`, `converged`, `residual` | Actual tensor fields for functional per-System results; dynamic leaves. `converged` means outer raw-residual convergence only. Legacy routes may retain `None` for convergence/residual. |
| `status` | Actual `SCFStatus | None` field. Functional results carry two dynamic integer-tensor leaves, `trajectory_failure` and `final_failure`; legacy routes may use `None`. Their category meanings are defined by the canonical `dxtb.SCFFailureCode` API enum from B5.6. |

`Result`, `ChargeResult`, `PotentialResult`, `Multipoles`, and `SCFStatus` are
immutable values. Optional `None` fields and classical labels are static
pytree structure; status-code tensors and other tensor values remain dynamic
leaves. The `dxtb.SCFFailureCode` integer-to-category mapping is static public
API metadata, not pytree context or a tensor leaf. It is defined once in
`src/dxtb/_src/scf/status.py` and re-exported from `src/dxtb/__init__.py` as
`dxtb.SCFFailureCode`; its numeric values are
stable public API values. `SCFStatus` preserves the
first physical-trajectory failure separately from the final-public-spectrum
failure, so the eager adapter can report trajectory failure first, final
spectrum failure second, and outer nonconvergence last without exposing
internal `SCFSolveResult` state. The legacy `h0` spelling is resolved to the
canonical `hcore` field before C8a.

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
- **Decided: short names stay** (`dipole_deriv`, `pol_deriv`, as in the
  current methods), so the migration guide lists fewer renames.
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
2. Conformers: the public call is
   `torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`. Property
   functions use the same pattern for per-system results without cross-system
   blocks. C8a first proves Result pytree behavior with a tensor-built probe;
   the exact System integration call is tested in E6a after the fixed-trip
   SCF boundary exists.
3. Mixed compositions: `dxtb.batch.setup(model, numbers_batch)` returns stacked
   padded systems (C8b). The accepted mixed-System implementation uses stack
   plus `vmap` for the authoritative PyTorch route. The generic public mixed
   batch API remains compatible while the legacy packed route is migrated;
   after E6broute it uses the E6bimpl route or an explicitly retained outer
   libcint compatibility composition selected by E6bcompat-policy. Batch size
   is never inferred from shapes.
4. `batch_mode` and the shape-based guessing are removed (104 references today).
5. E6a owns conformer mapping for one fixed System using the exact public
   `torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)` call. E6a
   does not retire the heterogeneous mixed-System packed batch route. Mixed
   `dxtb.batch.*` compatibility and its entry-loop fallback remain until
   E6broute activates the E6bimpl mixed-System implementation or an explicitly
   retained libcint outer-loop compatibility path.

The temporary `System.batch_mode` used by the legacy Calculator during the
migration is transitional legacy metadata, not part of the public/core
batching contract. E6a must not branch on, infer semantics from, or otherwise
depend on this field for the public fixed-System conformer `vmap` path; E6a
does not require B3b. B3b removes `batch_mode` from the final System layout
before C8b freezes that layout and E6bimpl stacks mixed Systems. Any separately
isolated legacy batch adapter may retain batch state until E6broute retires the
mixed-System route after its correctness and activation gates pass. E6a
conformer mapping does not delete mixed-System entry loops.

**Decision.** A batched call with one composition and many geometries (the
common CREST case) uses `setup(numbers)` once and then the conformer call in
the public contract above. It needs no additional public function. Mixed
compositions use the generic `dxtb.batch.*` API.

**Canonical contract and E6a target.** The public conformer call remains
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)`; E6a does not
add a public conformer entrypoint. B5.6 first closes Result's public
solver-status schema. C8a is then strictly an output-container package: it
proves the closed Result tree with a tensor-built probe, independently of
`System.singlepoint` and SCF control. The exact default-`out_dims` public call
is tested only in E6a, after its fixed-trip controller exists.

E6a puts fixed-trip numerical execution and tensor-valued status under
`System.singlepoint`; the private fixed-trip boundary is an implementation
detail, not a new public API. The ordinary eager `Calculator.singlepoint`
adapter owns eager early stop and warning/error policy outside transformed
execution and no longer obtains its eager result simply by delegating to
`System.singlepoint`. Both routes share the same physical SCF map, spectral
and Fermi primitives, solver step/mixer formulas, and finalization; only
controller and policy ownership differ. Direct eager calls to
`System.singlepoint` intentionally use the fixed-trip controller, run the
configured trips after convergence, return tensor-valued validity status, and
do not perform eager Fermi/convergence warning or error inspection. Tensor-
detected spectral failures are returned in `SCFStatus`; statically invalid
configuration may still fail during setup. Negative configured temperature
and negative Fermi `maxiter` remain legacy-owned while the global
E4.6b configuration-policy decision is open; E6a does not change this policy
or Calculator behavior. This is an explicit behavior change to the
direct-System call,
which E6a must cover with migration notes and separate tests for direct
System and Calculator semantics. No functorch detection is permitted and the
public conformer call does not change.

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
until the training work (F1, now the last package of the plan); the model converts once in its constructor.

## 7. Dtype and device (question 7)

**Confirmed.** Neither is stored in the configuration or on the model. Both are
read from the input tensors (`positions`, then the parameter tables).
`dxtb.GFN1(dtype=..., device=...)` stays as a constructor convenience that
creates the tables; afterwards `model.to(...)` is a tree map (C8b). Float64 is
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

1. ~~Short or long names~~ Decided: short names (section 3).
2. ~~Setup data of the self-consistent terms~~ Decided: `System` holds the
   element-only data of the classical **and** the self-consistent terms (ES2
   hardness, ES3 derivatives); see section 14.
3. **Per-call data.** The Coulomb matrices, the Cholesky factor `l_inv`, the
   integrals depend on positions. They live in per-call interaction data
   inside `singlepoint` and in the result (B6), not in the system. The
   Cholesky factor is the one place where this costs repeated work in a
   transformed function; measure before adding anything (P10).
   These values are local explicit values. They are never reused by identity
   or value comparison and are not caches in the architectural sense.
4. **Requirement (decided, done): the `refocc` derivative is fixed in T0.12**,
   before B3. It was not excluded and not left to F1.
5. **Release placement.** The new API ships in the single public restructuring release (overview, section 8).
   The old API raises with a migration pointer for that release only.

## 13. Acceptance

The note is agreed when sections 1-10 are confirmed or amended and the items
marked **[decide]** are answered (none remain). Then `00-overview.md` links it and B2 starts.

## 14. Explanation: what "geometry-independent data of the self-consistent terms" means

Every energy term has two kinds of input. Take the self-consistent terms, which
change the Fock matrix and are iterated in the SCF:

- **ES2** (second-order electrostatics) builds a Coulomb matrix `J[i, j]` between
  atoms (or shells). Its entries combine two things: the *hardness* of each atom
  or shell, which depends only on the element (`hubbard`, `lhubbard`, looked up
  per atom with `ihelp.spread_uspecies_to_atom`), and the *distance* between
  them, which depends on `positions`.
- **ES3** (third-order) needs one vector of Hubbard derivatives per atom (or
  per shell, times a shell scaling). It has no position dependence at all.

Today both are built lazily in `get_cache(numbers, positions, ihelp)`, with a
value-based cache. For ES3 the cache depends on `numbers` only. For ES2 the
element lookup is buried inside the function that also computes distances
(`get_atom_coulomb_matrix`, `get_shell_coulomb_matrix`), so the element
part is recomputed on every call.

The choice is where the element-only part lives after B3:

| Option | `setup(numbers)` holds | Per call |
| --- | --- | --- |
| **Both (proposed)** | classical setup data **and** the gathered ES2 hardness vectors and ES3 derivative vectors | the distance part of ES2, combined with the stored hardness |
| Classical only | classical setup data | all of ES2 and ES3, including the element lookup |

Why both:

1. **P3 and `vmap`.** The element lookup uses the index helper, which is
   data-dependent. It must not run inside a transformed function. Having the
   system hold the gathered vectors keeps the per-call path free of it.
2. **Training.** The gather is the one place where gradients reach the Hubbard
   parameters. If it sits in `setup`, `setup` is the single differentiable
   gateway from parameter tables to everything else, and the T0.9 test checks
   one place, not several.
3. **Cost.** The lookup is small, but in a transformed function it would be
   repeated per call (and per batch entry); in `setup` it runs once per
   composition.

Cost of choosing both: `System` grows by the term list. A system then belongs to
one model's set of terms, so adding a term (for example an external ML term,
F3) means building a new model and a new system, not mutating the old one. That
matches P4 and is why it is the proposal; "classical only" would keep `System`
smaller, but leaves the lookup in the per-call path and the gather outside the
single gateway.

What does **not** go into the system under either option: the Coulomb matrix
itself, ALPB and D4SC data, and anything else that depends on `positions`. Those
are per-call data (B6).
