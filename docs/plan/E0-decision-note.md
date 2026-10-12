# E0 decision note: SCF and spectral differentiation

Status: **accepted** after independent mathematical and dependency review.
This is a decision package, not an SCF implementation. It fixes the
mathematical contract and package boundaries for E3, E4, and E6. The current
mutable SCF engines remain transitional until those packages land.

## 1. Canonical SCF physics and solver separation

The canonical physical map is conceptually:

```python
scf_map(x, inputs) -> SCFMapResult
```

`x` is the configured self-consistent variable: charges, potential, or a Fock
matrix according to `scp_mode`. `inputs` are immutable per-call physical and
setup values. `SCFMapResult` contains at least `mapped_x = G(x)` and the
physical observables produced by that same evaluation: density, charges,
potential, Hamiltonian, orbital energies, coefficients, occupations,
free-energy data, and other observables needed by the result. Exact field and
class names remain E4core implementation details.

The physical fixed-point equation is:

```text
x* = G(x*; inputs)
R(x*) = G(x*; inputs) - x* = 0
```

This equation contains no mixer history. A separate pure solver transition is
conceptually:

```python
solver_step(solver_state, map_result, solver_settings) -> new_solver_state
```

`solver_state` may contain the current iterate, Anderson/Broyden/simple-mixer
history, dynamic damping state, active/converged masks, and solver bookkeeping.
The mixer is an algorithm for approaching the physical root, not part of the
physical equation. Changing Simple ↔ Anderson ↔ Broyden may change convergence
speed and the finite-iteration trajectory, but it must not change the SCF
equation whose root defines the physical solution.

The functional unrolled reference evaluates `G(x)`, computes the raw residual,
applies the functional solver/mixer transition, and repeats. It differentiates
the actual finite forward solver trajectory. The canonical equation used by
future implicit differentiation is instead `G(x) - x = 0`; these are related
but not identical mathematical objects. The unrolled composition around the
common physical map is the G1 correctness/reference differentiation path.
Ordinary PyTorch composition provides the clearest derivative semantics, so
its higher memory use is accepted for this reference implementation. This
does not bless today's `SelfConsistentFieldFull` implementation wholesale:
the mutable `_Data`, culling, mixer objects, Lorentzian broadening, and hidden
inverse-Cholesky cache are not authoritative architecture.

The implicit function theorem applies to the raw physical equation:

```text
F(x, p) = G(x, p) - x = 0
```

It does not apply to Anderson history, Broyden history, dynamic damping, or the
entire finite-iteration solver state. Apart from numerical convergence error,
the implicit derivative of one converged physical solution must be
independent of whether Simple, Anderson, or Broyden found it. Implicit
differentiation is mathematically attached only to a converged fixed point. A
nonconverged state must not silently receive a derivative presented as the
derivative of the SCF root. E3 will choose an explicit runtime policy: eager
failure/reporting, returning status without attaching the implicit derivative,
or a documented fallback to an unrolled derivative.

The physical map result and solver state have distinct ownership. `SCFMapResult`
must not contain mixer history, damping generations, residual history, or
solver counters. Conversely, physical outputs must come from the same map
evaluation that supplies the convergence residual; they must not be recomputed
from an unrelated mixed iterate after convergence.

Current behavior is transitional and diverges between strategies:

| Current path | Current spectral implementation | Status |
| --- | --- | --- |
| Full/unrolled | `tad_mctc.storch.eighb`; when gradients are enabled it selects Lorentzian eigenvector broadening | Not the final derivative definition |
| Implicit | vendored xitorch `lsymeig` through the pure/implicit route | Separate spectral derivative behavior; must converge to the common primitive |

That divergence must disappear. In particular, current forward-mode behavior
must not select different derivative mathematics based on `requires_grad`.

The source currently separates the raw physical evaluation from solver
mixing. In full mode, `SelfConsistentFieldFull.scf()` evaluates
`q_new = fcn(guess)` and then optionally mixes it at
`src/dxtb/_src/scf/unrolling/default.py:81-97`; each loop repeats the raw map,
then calls `mixer.iter(q_new, q)` at lines 101-115. In `Simple.iter()`, the
returned update is `x_old + (x_new - x_old) * damp`, while the stored
convergence delta is exactly `x_new - x_old` at
`src/dxtb/_src/scf/mixer/simple.py:142-148`. `BaseSCF.iterate_charges()`
increments `_data.iter` once per raw map call at
`src/dxtb/_src/scf/base.py:636-651`. In implicit mode,
`SelfConsistentFieldImplicit.scf()` sends its stateless map to `equilibrium()`
at `src/dxtb/_src/scf/implicit/default.py:63-69`; the root callback is
`y - fcn(y)` at `src/dxtb/_src/scf/implicit/fixed_point.py:516-522`. Thus the
current physical equation already excludes mixer history even though the old
full-mode class and this note previously described a single mixed “step.”

## 2. Current implicit mode and its future boundary

The current `scf_mode="implicit"` remains publicly available during the
transition for backwards compatibility. Its ordinary-autograd reverse-mode
results are an established baseline, but the implementation is not a G1
transform path: it uses custom `_ImplicitFixedPoint` and `_AdjointSolve`
autograd functions and discovers dependencies by walking `grad_fn` leaves
(`_grad_leaves`) and Python closure/GC referents (`_reaches_grad`). It does not
provide the required general `jvp`, `jacfwd`, `vmap` composition, or explicit
functional parameter-tree semantics.

Treat it as a legacy reverse-autograd optimization only. Do not extend it
piecemeal before E4core and do not silently change the public default in E0.
After E4core, an implicit strategy must apply the implicit-function theorem to the exact
same raw physical `scf_map` used by unrolling, using `G(x, p) - x = 0`; it
must not wrap or differentiate the mixer/solver transition. All
differentiable quantities are explicit input-tree leaves; no parameter
discovery by graph or object introspection is allowed.

E0 does not delete the current implementation. A future custom
`torch.autograd.Function`, if proposed, requires a separately reviewed E3/E3b
package and the complete repository gate: reverse mode, required
reverse-over-reverse orders, `jvp`, `jacfwd`, `vmap`, parameter gradients,
mixed derivatives, and padding safety. “Only an optimization” is not an
exception.

## 3. One spectral and occupation primitive

Every SCF strategy uses one conceptual primitive, with exact naming deferred:

```python
spectral_state(
    hamiltonian,
    overlap,
    electron_count,
    spin,
    electronic_temperature,
    orbital_mask,
    settings,
) -> SpectralState
```

Its outputs may include orbital energies, coefficients, occupations, density
matrix, and electronic free-energy data. Alpha/beta channels, the chemical
potential/Fermi solve, and entropy/free-energy response belong to this same
spectral contract. The occupation used for density and the occupation used
for free energy must be consistent, including for charge and reference
occupation derivatives.

The SCF needs physical, gauge-invariant quantities: density/occupation
matrix, charges, potential, energy/free energy, and response properties.
Individual coefficient columns are not a unique derivative target at a
degeneracy. `Result.coefficients` may remain a forward-value output, but no
unique derivative is promised for arbitrary eigenvector columns inside an
exactly degenerate eigenspace.

The forward implementation may use an ordinary eigendecomposition to report
eigenvalues and coefficients, while derivative-critical density and energy
response is formulated in an invariant way. This does not permit detached
eigenvectors, straight-through estimators, or hand-zeroed gradients as a
replacement for the correct density response.

## 4. Degeneracies, gaps, and the derivative contract

Lorentzian replacement of `1 / (epsilon_i - epsilon_j)` is **not** the final
derivative definition. Do not tune the broadening factor, epsilon,
`degen_atol`, or `degen_rtol` to hide known failures. Existing evidence is
physics-level: exact-degeneracy second and third derivatives are wrong, and
first-order response is biased at sufficiently small gaps.

For finite electronic temperature, the target is the spectral-matrix-function
density. In an orthogonal basis, write
`P = U f(epsilon) U.T`. The Fréchet derivative uses the divided difference

```text
F_ij = (f_i - f_j) / (epsilon_i - epsilon_j),  epsilon_i != epsilon_j
F_ij = f'(epsilon_i),                            epsilon_i == epsilon_j
```

The equal-energy limit applies to every entry in a degenerate block, including
off-diagonal entries between distinct eigenvectors in that block. The same
smooth matrix-function response applies when a finite-temperature degeneracy lies at the chemical
potential. This divided-difference kernel is evaluated at the current
chemical potential; the response of that chemical potential required by the
fixed electron-count and spin constraints is also part of the same Fermi
solve and must be differentiated. The final code mechanism for evaluating
this response is left to E3; the mathematical limit is fixed here.

The supported degeneracy cases are distinct:

1. **Degeneracy within an equally occupied subspace:** rotations inside the
   eigenspace do not change the density. Its physical response must be finite
   and gauge-invariant.
2. **Finite-temperature degeneracy, including at the chemical potential:**
   Fermi occupation is a smooth matrix function and uses the divided-
   difference limit above.
3. **Strict zero-temperature occupied/unoccupied crossing:** occupation is
   discontinuous. At an exact crossing its derivative may be non-unique or
   undefined. Do not manufacture a unique derivative with broadening. The
   differentiable zero-temperature contract requires a nonzero
   occupied/unoccupied spectral gap; internal degeneracies among equally
   occupied states remain allowed.

For a small but nonzero gap, the result must approach the correct mathematical
response continuously and agree with analytic or finite-difference evidence.
Numerical conditioning may be reported, but a broadening parameter may not
replace the limit.

Preserve the minimal spectral targets in
`test/test_baseline/components.py`:

```text
eigensolver.degenerate_aufbau
eigensolver.degenerate_fermi
eigensolver.small_gap_aufbau
eigensolver.small_gap_fermi
```

They cover exact occupied degeneracy, finite-temperature exact degeneracy,
small-gap zero-temperature Aufbau, and small-gap finite-temperature/Fermi.
Preserve the molecular benzene and Fe(CO)5 regressions as well. E3 must verify
physics against analytic/finite-difference results, not only finite outputs.
Acceptance ultimately includes reverse orders 1–3 where defined, `jvp`,
`jacfwd`, `vmap`, parameter and geometry dependence, padding, and small-gap
continuity. At an exact zero-temperature occupied/unoccupied crossing, test
the documented nondifferentiability policy instead of asserting a fabricated
derivative. If any listed target is not stable and standalone when E3 starts,
record and repair that test in E3 before changing the mathematics.

## 5. Generalized eigenproblem and structural padding

Overlap is fixed during one SCF iteration sequence but depends on geometry and
therefore belongs to one singlepoint call. Compute its inverse-Cholesky or
orthogonalization factor once for that call, place it in immutable
`SCFInputs`, and reuse it in every step. Do not retain it on an SCF engine,
System, or other persistent object. The current `BaseTSCF._inv_cholesky`
instance cache is transitional and must disappear in E4core.

The spectral primitive receives an explicit structural `orbital_mask` derived
during setup. It must not infer padding from numerical equality such as
`overlap == 0` at evaluation time. Keep fixed tensor shapes; do not dynamically
delete padded orbitals. A shifted padding block may be used for forward
diagonalization if needed, while the mask remains explicit structural data.

## 6. Fixed-trip SCF and Fermi semantics

The transform-critical Fermi solve and SCF iteration use fixed-shape tensor
control flow. Their maximum iteration count is static. Active/converged flags
are tensor masks; converged entries freeze their values. There is no
`Tensor.item()`, tensor-dependent Python `break`, `bool(tensor)`, or Python
`any()` over tensor-derived flags to control the numerical loop.

Conceptually, the future fixed-trip solver around the physical map is:

```python
solver_state = initial_solver_state
active = initial_active_mask
final_map_result = scf_map(solver_state.x, inputs)
iterations = ones_like_systems(dtype=int64)  # initial map evaluation
residual = residual_values(final_map_result.mapped_x - solver_state.x)

if maxiter > 0:  # static configuration branch; zero is supported
    if mix_guess:
        solver_state = initial_mix(solver_state, final_map_result)
    else:
        solver_state = with_x(solver_state, final_map_result.mapped_x)

    for iteration in range(maxiter):
        map_result = scf_map(solver_state.x, inputs)
        raw_residual = map_result.mapped_x - solver_state.x
        newly_converged = active & convergence_test(raw_residual)
        final_map_result = tree_where(active, map_result, final_map_result)
        residual = tree_where(active, residual_values(raw_residual), residual)
        iterations = where(active, iterations + 1, iterations)
        transition_active = active & ~newly_converged
        candidate_solver_state = solver_step(
            solver_state, map_result, solver_settings, transition_active
        )
        solver_state = tree_where(
            transition_active, candidate_solver_state, solver_state
        )
        active = transition_active

converged = zeros_like(active)
if maxiter > 0:
    converged = ~active
```

This sketch describes the eventual E4core structure; its public iteration-count
compatibility is specified below. `SCFInputs` contains immutable per-call
physical/setup inputs. `SCFMapResult` contains `G(x)` and the observables from
that physical evaluation. `SCFSolverState` owns the current iterate, mixer
history, active/converged masks, and solver bookkeeping. `SCFSolveResult`
contains the selected final physical map result and convergence/status values.
The raw residual is tested before the mixer changes the next iterate. A
converged system's final physical observables are those from the `G(x_k)` map
evaluation whose raw residual converged; the subsequently mixed iterate is
never reported as though it were that converged physical evaluation.

The loop trip count is static, and no tensor-dependent Python `break` is used.
The candidate physical result is accepted for currently active systems,
including systems that converge on that evaluation; later evaluations freeze
their physical result and status. Anderson/Broyden/simple mixer history and
dynamic damping state belong only to `SCFSolverState`; mixer settings belong to
a separate static solver-settings value, never `SCFInputs`. The solver
transition is a pure function.

The canonical functional control deliberately uses the post-convergence
`transition_active = active & ~newly_converged` mask for the mixer transition.
Current mutable `SelfConsistentFieldFull` instead calls `mixer.iter(q_new, q)`
before checking `mixer.converged`, so it advances the mixer on a converged raw
map. E4core removes this dead transition. The selected converged physical map
result is unchanged, but this is not exact legacy side-effect/error parity:
legacy Anderson may fail while solving a mixer system after the raw residual
already meets convergence; canonical E4core accepts that map without the physical
transition. E4.5 must review this deliberate compatibility decision before
production routing.

The raw convergence residual is exactly:

```text
R(x) = G(x) - x
```

It is measured before damping or mixer history changes the next iterate. Its
L2 and maximum-absolute values are computed from that raw physical residual,
in that order, matching the existing pair of convergence tests. Convergence
must not be defined from `mixed_x - x`, whose value depends on the solver
settings. In current full mode this is the distinction represented by the
mixer's stored raw delta `x_new - x_old`.

The canonical E4core/full functional convergence test preserves the current
full-mode tolerance semantics exactly:

```text
L2(R(x)) < x_atol
L∞(R(x)) < x_atol_max
```

Both conditions must hold. The future `Result.residual` contains exactly two
values per system: `[..., 0]` is `L2(G(x)-x)` and is compared with `x_atol`;
`[..., 1]` is the maximum-absolute norm and is compared with `x_atol_max`.
There is no third `f_atol` residual or convergence condition. Current
`SelfConsistentFieldFull` uses `Mixer.converged`, which checks these two x
tolerances and does not consult `f_tol`. `BaseSCF.fwd_options` still maps
`ConfigSCF.f_atol` to `f_tol` for the transitional implicit/root-solving
path. That legacy compatibility behavior remains unchanged. When that path is
rebuilt around E4core, the public fate and meaning of `f_atol` must be decided
explicitly; it must not silently become a full-mode convergence criterion.
Other legacy modes retain their existing behavior in this decision package.

### Iteration and result compatibility

The terms below are distinct:

- `maxiter` is the user-configured budget for loop cycles after the initial
  physical-map evaluation in current full mode.
- A physical-map evaluation is one call to the raw `G(x)` map.
- A solver/mixer transition updates the next iterate and its history; current
  full mode has an initial mix transition only when `mix_guess=True`, then one
  transition after each loop map evaluation.
- `Result.iterations` currently reports the number of physical-map
  evaluations, including the initial evaluation. It is not the number of
  mixer transitions or the configured `maxiter` value.
- A future first-converged-step index is a separate status concept and must
  not be silently substituted for current public `Result.iterations`.

Current full mode deliberately supports `maxiter=0`: it evaluates `G(guess)`
once, returns that raw physical map result, performs no mixer transition, and
reports one iteration/map evaluation. `mix_guess` has no effect in this case.
For positive `maxiter`, the implementation starts with one raw evaluation and
then performs up to `maxiter` further map evaluations, stopping early if the
raw map delta meets convergence. Thus with no early convergence the current
full-mode reported count is `maxiter + 1`. This compatibility contract is
characterized by `test/test_scf/test_e0_iteration_compat.py` and is the first
behavior E4core must preserve. The characterization also records that `mix_guess`
changes the finite trajectory and returned physical values for positive
budgets, but not the physical fixed-point equation. On the current deterministic
CPU GFN1 H2O case with Simple mixing and damping `0.25`, the characterization
is:

| `maxiter` | `mix_guess` | raw map evaluations / reported iterations | mixer transitions | returned total energy | unconverged warning |
| ---: | :---: | ---: | ---: | ---: | :---: |
| 0 | true | 1 | 0 | -5.7674748965126055 | no |
| 0 | false | 1 | 0 | -5.7674748965126055 | no |
| 1 | true | 2 | 2 | -5.768234535786348 | yes |
| 1 | false | 2 | 1 | -5.768064929714654 | yes |
| 2 | true | 3 | 3 | -5.768491792471992 | yes |
| 2 | false | 3 | 2 | -5.768438426259488 | yes |

Exact returned orbital charges are locked alongside energy in the focused
compatibility test. The positive-budget cases use `f_atol=x_atol=x_atol_max=
1e-12`; they do not converge in one or two cycles, so the warning is expected.
The initial map is not convergence-tested by the current full-mode algorithm.
Consequently `maxiter` is non-negative, not positive, and the zero case is a
separately preserved one-shot operation.

The tolerance compatibility test also characterizes a five-cycle CPU GFN1
H2O run with Simple mixing, `damp=0.25`, and `mix_guess=True`. With both x
tolerances fixed at `1e-12`, `f_atol=1e3` and `f_atol=1e-12` each report six
map evaluations, the same energy (`-5.768616757320126`), identical orbital
charges (`[0.304379237811349, -0.6666666666666672,
-0.3137346056586603, 0.0038987582914078445, 0.36813476975763415,
-0.03207313164634945, 0.3681347697576336, -0.032073131646349326]`), and
the same unconverged warning. With `f_atol=1e-8` held fixed, changing both x
tolerances from `1e-12` to `1e2` changes the result from six evaluations,
energy `-5.768616757320126`, and an unconverged warning to two evaluations,
energy `-5.768234535786348`, and no warning. This confirms that the full-mode
convergence point is controlled by the x tolerances, not by `f_atol`.

Negative `maxiter` is outside the canonical E4core contract and must be rejected
as invalid static configuration when an otherwise functional-eligible,
single-System full-SCF route is selected. Common `get_refocc` and `get_guess`
preparation still precede that route decision. Batch, custom-interaction,
non-full, non-SCF, and negative-Fermi-maxiter legacy-owned routes remain under
their existing behavior; E4core does not seize ownership of their negative SCF
budgets. This does not preserve accidental legacy behavior for a negative
budget on the canonical functional route.

The future core returns tensor-valued per-system `iterations`, `converged`, and
`residual`. `converged` and `residual` describe the raw map criterion and the
map evaluation selected as the final physical result. For positive budgets,
`iterations` first preserves the characterized current full-mode
map-evaluation-count semantics, including the initial evaluation. A system
that converges records the count at its first converged raw map evaluation; a
system that exhausts the budget records the number of map evaluations actually
performed (up to `maxiter + 1`). For `maxiter=0`, the observable compatibility
requirement is one raw map evaluation and `Result.iterations == 1`; the current
API has no convergence-status field for this one-shot case. The future E4core
status is explicitly conservative: `converged=False` because no outer
convergence check was performed, and `residual` contains the raw diagnostic
`G(guess)-guess` from that initial map. A small diagnostic residual does not
retroactively turn the one-shot operation into a convergence test. This also
prevents implicit root differentiation from attaching to a zero-budget result.
Any different zero-budget status convention requires an explicit API
behavior-change decision. For mapped systems, later fixed-trip evaluations
freeze each system's physical result and status.

Current implicit mode uses a different root solver and reports its own
function-evaluation count. It remains transitional; do not force its current
count to match full mode. The common future E4core solver/status contract applies
to the canonical functional solver, while legacy implicit mode keeps its
existing behavior until separately rebuilt around the common map.

The eager compatibility wrapper may inspect returned status to issue a
configured warning/error or user-facing message. The transform-critical
numerical kernel returns values and status; it does not synchronize to the
host to decide whether it may continue. Invalid static structure may still be
rejected eagerly before evaluation.

## 7. NO2 third-order isolation

The open-shell GFN2 NO2 third-order failure remains a separate G1 correctness
blocker. Do not infer its cause from the final molecular number and do not
change tolerances. E3 diagnoses it in this order, stopping at the first layer
whose third-order finite-difference comparison becomes wrong:

1. Test occupation/Fermi response alone using NO2-representative electron
   counts, spin channels, temperature, and a frozen orbital-energy spectrum.
2. Test the common spectral-density primitive with frozen Hamiltonian/overlap
   and the alpha/beta occupations.
3. Test one pure SCF step at a fixed converged input.
4. Compare GFN1 and GFN2 term activation around that same state to locate the
   first GFN2-specific contribution where third order diverges.
5. Only then investigate the full fixed point.

Use multiple finite-difference step sizes and demonstrate a stable region;
one third-derivative step is not evidence. Keep the `KNOWN_WRONG` NO2 entry
until the nested derivative and a stable finite-difference estimate agree.
Do not weaken tolerances, regenerate references, or remove NO2 to make the
matrix green.

## 8. Package boundaries and status

**E4core owns** both (A) the pure physical `scf_map(x, inputs) -> SCFMapResult` and
(B) the pure functional solver around that map: immutable `SCFInputs`, explicit
`SCFSolverState`, functional mixers, a fixed-shape solver loop, fixed-trip
tensor-masked SCF/Fermi control flow, convergence/status tensors, removal of
`cull`/`restore` from the canonical functional single-System SCF path, and
per-call spectral inputs such as the inverse-Cholesky factor. E4core establishes
fixed-shape masked control; it does not require deleting separately isolated
`cull`/`restore` machinery still owned by the legacy production batch FULL
adapter. E4core may reuse numerical formulas, but it must not combine the physical
map and solver transition into one opaque state update. This note does not
declare today's `BaseSCF._Data`, `_MapData`, or `pure.iterations` suitable as
the final explicit API.

E6b retires the legacy production batch route after the batch-critical E7
padding-safety gate and both E6c1 performance clearances (W2 conformers and
W3 mixed Systems); the broader E7 audit may continue. An explicit
architecture-review waiver may allow route retirement after a documented CPU
regression, but it does not satisfy the final G2 throughput target. E4.6a
then removes the now-unowned batch cull/restore machinery.
E4.6b separately deletes the remaining legacy FULL loop only after every
other compatibility fallback is migrated, rejected, or explicitly retired.

**E3 owns**, after E0 is accepted and E4core supplies the raw physical map,
invariant spectral/density differentiation, the NO2 isolation and fix,
higher-order SCF derivative validation, unrolled-versus-implicit comparison,
and any optional transform-compatible implicit strategy. Its implicit path
consumes `scf_map` (or the corresponding raw-map component of the E4core API),
never the selected mixer. A required later acceptance test is that the same
converged physical point reached by Simple, Anderson, and Broyden gives the
same implicit physical derivative within solver convergence error. If the
invariant spectral primitive must land before E4core can function, split that work
into a narrowly approved spectral-prerequisite package; E4core must not improvise
its derivative contract.

**E6a owns** conformer mapping over one fixed System using the B1 public
`vmap(system.singlepoint)` contract, after the transform-safe controller and
Result pytree prerequisites exist. **E6b owns** heterogeneous mixed-System
construction, stacking/vmap composition, and production batch-route
migration. The B7 field kernels are already vmap-safe; the B7 whole-System
failure is generic host scalar extraction in the occupation/Fermi search and
is resolved on the E4core fixed-trip path before E6a. This does not start E4core or
E6.

Keep `scf_mode="implicit"` publicly available during transition as a legacy
ordinary-autograd/reverse-oriented optimization. Do not advertise it as the
G1 transform path or change its public default in this decision package.
Replace its internals only after the common E4core map exists. E0 makes no choice
of a custom eigensolver/autograd primitive; plain PyTorch is preferred, and
any custom spectral primitive is a separately reviewed design/implementation
package subject to the complete transform ladder.

This note does not start E3, E4, E6, B8, Node migration, D3 work, or GPU work.
It changes no runtime behavior, references, tolerances, `KNOWN_WRONG` entries,
SCF defaults, mode defaults, eigensolver factors, or Fermi thresholds.
