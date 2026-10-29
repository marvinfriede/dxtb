# SCF S1 core contracts

## 1. Review point

S1 started from the accepted published synchronization point
`5365b16ca48ee2d7bac0326eeaad779993f244f6`, with SCF starting commit
`aa47203f5810280d7cd9645d763ec3fbb1d33dfc`. During S1, the published
general-refactor branch advanced to `a0f35c92e7357752e2f4cc8ec2ba2b3f16ba4423`.
Its only intervening commits were W2c2a mapped diagnostic/performance
documentation and touched none of `src/dxtb/_src/scf/**`,
`config/scf.py`, or `test/test_scf/**`. At the clean S1 package boundary the
SCF branch was rebased onto that unrelated published head. The general-refactor
worktree was not modified.

## 2. Pre-S1 baseline

The three Phase-A selections were rerun with the isolated SCF environment,
single-threaded math libraries, and the repository-required logical xdist
workers:

| Selection | Result |
| --- | ---: |
| Functional map, solver, Anderson, eager compatibility | 225 passed, 1 failed |
| Implicit root and root derivatives (`not slow`) | 54 passed |
| Public status, invariant spectral density, invariant free energy | 131 passed |

The sole failure is the documented inherited
`test_anderson_solver_real_position_parameter_and_field_gradients`: maximum
absolute difference `2.5007065862503453e-11` against strict parity tolerance
`2e-13`. Its magnitude matches the accepted S0.1c classification. The
Python 3.14 TorchScript deprecation warnings are unchanged warnings, not test
failures. There were zero unexplained pre-S1 failures.

## 3. New contract modules

### `functional/problem.py`

`SCFProblem` is a frozen value containing only `SCFInputs`, a typed
`SCFRepresentation` (`CHARGE`, `POTENTIAL`, or `FOCK`), and the optional
potential `PotentialLayout`. `map(x)` delegates directly to the established
`scf_map`, `potential_map`, or `fock_map`. `evaluate(x)` returns the input
paired with that map result as `SCFPhysicalState`.

`SCFPhysicalState` stores only `x` and its `SCFMapResult`; `delta` exposes
`G(x) - x`. It does not copy spectral data or store convergence policy. The
controller adapter constructs it from the exact input used for the candidate
map and that candidate's returned map result.

The problem contains no public configuration, iteration budget, tolerances,
damping, controller masks, method state, or differentiation state.

### `functional/method.py`

`SCFMethod` is the future-facing protocol:

```text
propose(problem, physical_state, method_state, context)
    -> SCFMethodProposal
```

`SCFMethodContext` carries the current step, active-transition tensor,
controller-resolved damping, and the mapped proposal after the existing
representation projection. The projection remains controller-owned. The
method can inspect the full physical observation, including density, Fock,
energy inputs, and problem data, without constraining future methods to a
two-tensor mixer signature.

`LinearMethod` is the only method implementation in S1. It uses the existing
`simple_mix(old, mapped, damp)` arithmetic, with no method history; its state
is `None`. `SCFMethodState` is a marker protocol for later immutable,
method-specific dataclasses, not a universal history container.

### `functional/strategy.py`

`SCFStage` binds a method and `SCFStrategy` holds an immutable stage tuple.
The S1 execution boundary calls `single_stage()` and rejects empty or
multi-stage strategies with `NotImplementedError`. There is no method
switching or fallback policy.

### `functional/convergence.py`

`SCFConvergence` and `SCFResidual` expose the existing stopping decision at a
small boundary. The residual remains the projected raw `G(x)-x`; convergence
still requires strict `L2 < x_atol` and `maxabs < x_atol_max`. Existing Fermi
validity accumulation remains in the solver. No threshold, dtype resolution,
timing, or final certification semantics changed.

## 4. Controller and differentiation ownership

There is no new controller loop or controller module. The linear method is
adapted by `_linear_transition_adapter` at the transition boundary and uses
the existing `_solve_map_control` / `_solve_map_control_eager` loops plus
their shared `_initialize_solver_control` and `_solver_control_step` logic.
Anderson continues to use its private legacy-shaped transition callback. The
callback signature was not adopted as the `SCFMethod` contract. The private
`_simple_mixer_controller_adapter` remains only for synthetic/private
`_solve_map` callers that exercise the pre-S1 transition path.

The controller state now calls its history slot `method_state`; the existing
public trace field name `mixer_state` and its tensor layout are unchanged.
Controller lifecycle, damping selection, active masks, projection, map counts,
and root selection remain shared.

The fixed-trip controller now retains the initial Fermi trajectory status for
the accepted `maxiter=0` one-shot, matching the eager controller and existing
status contract. The one raw map still produces `converged=False`; the extra
status records the map that was already evaluated and does not change the
iteration budget or convergence decision.

Differentiation was not reimplemented. The accepted E3.1 boundary remains the
eager `_prepare_scf_root` validation followed by prepared-root `attach()`,
which differentiates only the physical equation `G(x,p)-x=0` and finalizes
after attachment. No method, history, strategy, or controller state enters
that equation.

## 5. Production migration scope

The functional linear/simple charge, potential, and Fock solve entry points
now construct one `SCFProblem` and one single-stage `LinearMethod` strategy.
All three continue to use the existing raw maps and finalizers. The existing
eager route selects these entry points through the unchanged configuration
adapter. `ConfigSCF`, `ConfigFermi`, public routing, damping settings,
`mix_guess`, `maxiter`, projections, trace, and implicit implementation were
not redesigned.

Anderson remains operational on the prior controller transition interface
and was not migrated. Broyden and legacy routing remain unchanged.

## 6. Test hygiene and contract coverage

Five S0.1c isolation assertions that demanded `atol=rtol=0` were changed to
float64 numerical bounds: value comparisons use `1e-13` absolute and relative
tolerance, and JVP comparisons use `1e-12`. These tests check that a bad or
heterogeneous sibling lane does not materially alter a supported lane; they
do not specify LAPACK bitwise identity. The spectral-oracle tolerances were
not changed. Direct-batch/explicit, `vmap`/standalone, and supported-lane JVP
differences were measured as exactly zero on both PyTorch 2.14.1+cpu and
2.13.0+cpu.

The new `test_functional_s1_contracts.py` verifies all three map delegates in
Aufbau and finite-temperature cases, physical input/result pairing, linear
mixing parity for float32/float64 and multiple residual/damping values,
single-stage rejection, eager/fixed-trip controller parity, dynamic damping,
`mix_guess`, `maxiter` boundaries, trace fields, the tested-input root
invariant, and `jvp`/`jacfwd`/`jacrev`/`vmap` use. The existing iteration,
potential, Fock, eager compatibility, implicit-root, spectral, free-energy,
and Anderson tests remain the behavioral acceptance suite.

An independent same-environment comparison between the accepted general
source and the S1 source found identical results for H2/GFN1, H2O/GFN2, and
NO2/GFN2 at 300 K. For these cases the iteration counts were 16, 19, and 17;
all converged. Energy, free energy, residual, charges, density norm, and
Hamiltonian norm matched exactly at printed precision. No production
matrix-Fermi code changed.

## 7. Deferred work

S1 does not migrate Anderson or implement Broyden, CDIIS, EDIIS/ADIIS, SOSCF,
method switching, fallback heuristics, unified final certification, new
status categories, public configuration, response-representation selection,
xitorch retirement, or batch redesign. S2 owns convergence and final physical
certification; it should distinguish a numerical spectral failure from
ordinary SCF nonconvergence. S3 owns unified linear/Anderson method-state and
configuration migration.

## 8. Validation status

The focused S1 contract file passes, as do the map/solver/eager-compatibility,
E0 iteration, Fock and potential selections. The Phase-A groups reproduce 225
passed plus the inherited single failure, 54 passed, and 131 passed
respectively. The Anderson selection has 32 passing tests plus its one
inherited parity failure. Compileall, Black, isort, and `git diff --check`
pass on the changed files. Exact final validation counts after the package
boundary synchronization are recorded in the completion report. The inherited
Anderson strict-parity failure is not relaxed or otherwise modified.
