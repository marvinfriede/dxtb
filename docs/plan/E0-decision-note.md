# E0 decision note: SCF and spectral differentiation

Status: **accepted** after independent mathematical and dependency review.
This is a decision package, not an SCF implementation. It fixes the
mathematical contract and package boundaries for E3, E4, and E6. The current
mutable SCF engines remain transitional until those packages land.

## 1. Canonical SCF physics and differentiation

There is one canonical SCF physics definition:

```python
scf_step(state, inputs) -> new_state
```

`state` and `inputs` are explicit immutable values. `scf_step` includes the
physical interaction evaluation, spectral solution, occupations, density,
charges/potential, free-energy data, and mixer transition needed for one
fixed-point step. The fixed-point map includes Fermi/occupation physics; it is
not merely a charge-to-Hamiltonian-to-charge map with occupations computed by
an unrelated side path.

The exact same step definition is used by fixed-iteration unrolling, any future
implicit differentiation, eager early-stop wrappers if retained, and mapped
execution. There may be different solve/differentiation strategies around that
step, but no independently evolving “full physics” and “implicit physics.”

The **functional unrolled composition of the common step** is the G1
correctness and reference differentiation path. Ordinary PyTorch composition
provides the clearest derivative semantics, so its higher memory use is
accepted for the reference path. This does not bless today's
`SelfConsistentFieldFull` implementation wholesale: the current mutable
`_Data`, culling, mixer objects, Lorentzian broadening, and hidden inverse
Cholesky cache are not authoritative architecture.

Current behavior is transitional and diverges between strategies:

| Current path | Current spectral implementation | Status |
| --- | --- | --- |
| Full/unrolled | `tad_mctc.storch.eighb`; when gradients are enabled it selects Lorentzian eigenvector broadening | Not the final derivative definition |
| Implicit | vendored xitorch `lsymeig` through the pure/implicit route | Separate spectral derivative behavior; must converge to the common primitive |

That divergence must disappear. In particular, current forward-mode behavior
must not select different derivative mathematics based on `requires_grad`.

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
piecemeal before E4 and do not silently change the public default in E0. After
E4, an implicit strategy must wrap the exact same pure fixed-point map used by
unrolling. All differentiable quantities are explicit input-tree leaves; no
parameter discovery by graph or object introspection is allowed.

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
instance cache is transitional and must disappear in E4.

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

Conceptually, one fixed-trip loop is:

```python
state = initial_state
active = initial_active_mask
iterations = zeros_like_systems(dtype=int64)
residual = zeros_like_systems(dtype=inputs.dtype, trailing=2)

for iteration in range(maxiter):
    candidate = scf_step(state, inputs)
    newly_converged = active & convergence_test(candidate)
    state = tree_where(active, candidate, state)
    iterations = where(newly_converged, iteration + 1, iterations)
    residual = tree_where(active, candidate.residual, residual)
    active = active & ~newly_converged

converged = ~active
iterations = where(converged, iterations, maxiter)
```

`maxiter` is a positive static configuration value. The Python iteration
count is static. The candidate state is accepted for
currently active systems, including systems that converge on that step; later
steps preserve their accepted state. Anderson/Broyden/simple mixer history is
part of explicit `SCFState`; mixer settings are part of `SCFInputs` or its
static configuration, and each mixer transition is a pure function.

The convergence residual is the raw fixed-point delta `g(x) - x` for the
configured iterated quantity, measured before mixer damping/history changes
the next iterate. This preserves the current distinction between the map
residual and a damped state update.

The numerical core returns tensor-valued per-system `iterations`,
`converged`, and `residual` (for one System the system dimensions may be
zero-dimensional). `residual` has a final length-two axis containing the raw
L2 and maximum-absolute fixed-point deltas, in that order, matching the two
current convergence tests. The configured tolerances are applied componentwise.

- `converged` is the tensor flag for the configured SCF residual criterion.
- `iterations` is one-based and gives the first step at which that criterion
  became true; if it never becomes true, it equals `maxiter`.
- `residual` records both criterion values at the first converged step for a
  converged system, and after `maxiter` for a nonconverged system.

The eager compatibility wrapper may inspect those returned flags to issue the
configured warning/error or user-facing message. The transform-critical
numerical kernel returns values and status; it does not synchronize to the
host to decide whether it is allowed to continue. Invalid static structure
may still be rejected eagerly before evaluation.

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

**E4 owns** explicit immutable `SCFInputs` and `SCFState`, the common pure
`scf_step`, functional mixers, the fixed-trip masked loop, removal of
SCF/cache `cull`/`restore`, transform-compatible Fermi control flow, and
per-call spectral inputs such as the inverse-Cholesky factor. E4 may reuse
current numerical formulas, but this note does not declare today's
`BaseSCF._Data`, `_MapData`, or `pure.iterations` suitable as the final explicit
state/input API.

**E3 owns**, after E0 is accepted and E4 supplies the pure map, invariant
spectral/density differentiation, the NO2 isolation and fix, higher-order SCF
derivative validation, unrolled-versus-implicit comparison, and any optional
transform-compatible implicit strategy. If the invariant spectral primitive
must land before E4 can function, split that work into a narrowly approved
spectral-prerequisite package; E4 must not improvise its derivative contract.

**E6 owns** external batching and `vmap` of complete Systems. E4 must make the
single-System SCF transform-compatible enough for E6. The B7 field kernels
are already vmap-safe; the B7 whole-System failure is generic host scalar
extraction in the occupation/Fermi search and is deferred to E4/E6 under this
decision. This does not start E4 or E6.

Keep `scf_mode="implicit"` publicly available during transition as a legacy
ordinary-autograd/reverse-oriented optimization. Do not advertise it as the
G1 transform path or change its public default in this decision package.
Replace its internals only after the common E4 map exists. E0 makes no choice
of a custom eigensolver/autograd primitive; plain PyTorch is preferred, and
any custom spectral primitive is a separately reviewed design/implementation
package subject to the complete transform ladder.

This note does not start E3, E4, E6, B8, Node migration, D3 work, or GPU work.
It changes no runtime behavior, references, tolerances, `KNOWN_WRONG` entries,
SCF defaults, mode defaults, eigensolver factors, or Fermi thresholds.
