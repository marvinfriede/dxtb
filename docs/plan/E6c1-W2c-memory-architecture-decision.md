# E6c1-W2c — Fixed-trip memory architecture decision

## Decision status

**The converged-root proof of concept is promising for scalar energy
observables on small GFN1 systems; production route selection remains OPEN.**
See [E6c1-W2c1](E6c1-W2c1-implicit-root-prototype.md). This is a different
derivative contract from the current finite fixed-trip trajectory. The
current fixed-trip path remains authoritative. No checkpointing,
implicit-route substitution, or convergence-aware loop change is implemented.

## Evidence carried forward

The accepted W2b measurements show differentiated RSS increasing from 3.05
GiB at two trips to 12.05 GiB at sixteen trips, approximately 0.64 GiB per
trip over that interval. Eight trips peak at 6.90 GiB with gradients and 0.37
GiB in the no-grad diagnostic. Serial and mapped four-trip peaks are similar
(4.34 and 4.10 GiB). Those measurements identify retained unrolled
differentiation state as the dominant owner, while not proving every byte is a
saved autograd tensor. W2a.2 repeated one- and two-trip checks only; it did not
repeat the unsafe long-trip runs. See the W2b report and
[`E6c1-W2a2-third-order-correction.md`](E6c1-W2a2-third-order-correction.md).

The current fixed-trip controller in
`src/dxtb/_src/scf/functional/solver.py` evaluates `map_fn` on every configured
trip, then masks solver-state updates for inactive lanes. Masking the update
does not skip spectral, integral, or interaction work. The controller is
static and transform-compatible. An eager single-System controller already
uses host convergence checks, but that behavior cannot be copied into a
`vmap` path where lanes may converge at different trips.

The repository has an accepted implicit-root attachment in
`src/dxtb/_src/scf/functional/implicit.py`. It requires a separately prepared
and validated primal root; it attaches derivatives of `G(x,p)-x=0` and uses
the upstream matrix-free BiCGSTAB response solve. Existing functional
implicit-root tests pass, but the full tmpda W2 workload has not been run with
that route and its derivatives are not the derivatives of an unconverged
finite fixed-trip trajectory.

## Option comparison

| Option | Contract and expected benefit | Transform and numerical status | Decision |
|---|---|---|---|
| A. Fixed-trip recomputation/checkpointing | Preserves the finite unrolled trajectory if replay is exact. Checkpointing can trade saved activations for recomputation; memory should depend on checkpoint boundaries rather than every map's intermediates. | No project prototype establishes `vmap`, JVP, `jacfwd`, `jacrev`, HVP, mixed or third-order support. Non-reentrant checkpointing is not assumed to satisfy these transform requirements. Runtime and memory have not been measured. | Keep as a comparison candidate. Do not integrate before a transform prototype proves the required boundary. |
| B. Validated converged root with implicit derivatives | Removes the primal iteration graph from the derivative path. The physical derivative is the implicit response at a converged root. This is the strongest available route to reduce memory based on existing code. | The root attachment and response solves have focused transform tests. W2-specific primal convergence, parameter response, mapped lane independence, response cost, and third-order performance remain unmeasured. It differs mathematically from finite-trip differentiation whenever the primal has not converged. | **Selected for the next bounded proof of concept**, with the root contract made explicit and no production substitution. |
| C. Transform-compatible convergence-aware execution | Could skip map evaluations after convergence and reduce work, if supported control flow can stop each lane independently. | State masking alone does not skip `map_fn`. The eager controller is single-System and host-controlled. A mapped dynamic loop is not established across supported PyTorch versions or required derivatives. | Defer. Do not introduce tensor-to-Python control flow or a new dynamic loop here. |

## Next bounded package

Create a W2c1 proof-of-concept on a small GFN1 system before tmpda. Keep the
existing fixed-trip result as the reference and evaluate only converged cases
when comparing implicit derivatives. The prototype should:

1. Validate primal convergence, electron count, final residual, status, and
   failure behavior before attaching the root derivative.
2. Compare energy, position gradient, and Model parameter gradient with a
   tightly converged independent serial reference.
3. Check HVP and the supported third-order transform boundary.
4. Exercise one lane and a mapped pair with different convergence histories;
   verify no cross-System derivatives.
5. Measure fresh-process forward/backward time and peak RSS at bounded
   workloads, then at tmpda only if the small case passes.
6. Preserve the fixed-trip route and its finite-trajectory derivative
   contract. Any eventual route selection requires separate architecture
   approval.

Checkpointing remains a fallback comparison if the implicit-root prototype
cannot meet the accepted derivative or convergence contract. A checkpoint
prototype must separately demonstrate all required `torch.func` transforms
before memory savings are considered.

## Remaining limits

- The H2O proof reduced peak RSS by 31–33% at a 16-trip fixed/unrolled
  comparison and matched converged energy, position, model-derived parameter,
  and tested higher-order responses. This small-case result does not predict
  tmpda/eight-conformer behavior.
- Per-System mapped H2/H2O energy Jacobians matched serial results and had
  zero cross-System blocks. Some arbitrary H2O full-root-state Jacobian
  directions produced NaNs in both serial and mapped evaluation, while the
  tested energy adjoint remained converged. A general state-Jacobian contract
  is not established.
- The prototype is a feasibility result only; it does not authorize replacing
  fixed-trip derivatives in the production path.
- No candidate architecture has a measured full-W2 memory/time curve.
- No 32-, 179-, or 300-trip differentiation run was repeated.
- No full eight-conformer mapped result exists.
- W2 and W3 remain OPEN; E6broute remains BLOCKED.
- Public Calculator routing is unchanged.
