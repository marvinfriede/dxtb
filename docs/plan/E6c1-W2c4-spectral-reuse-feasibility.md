# E6c1-W2c4 — Spectral Reuse Feasibility

## Status

**Feasibility: COMPLETE; no production optimization selected.** Candidate A
reproduces density values for smooth spectra but fails the supported derivative
contract near and at degeneracy. Candidate B reproduces smooth primal values
but omitting the matrix-level chemical-potential corrections loses electron
count and Hamiltonian derivatives. The experimental implementations remain in
the diagnostic harness only. No production SCF, spectral, dependency, or public
Calculator code changed.

W2/W3 remain OPEN; E6broute remains BLOCKED. The published S0.1b, W2c1a,
W2c2/W2c2a and E6bcompat boundaries are unchanged.

## Source and environment

- Starting SHA: `66e9ce8d7501b32835eeec2d201f7b250454ae42`
- Validated harness/results SHA:
  `9a3acb2cdf0c41a7dbc626ee0b1fd1ffab50cfa6`
- Branch: `claude/tender-dijkstra-cda42s`
- Verified starting remote SHA: `66e9ce8d7501b32835eeec2d201f7b250454ae42`
- CPU: AMD EPYC 7763
- Profiles: Python 3.11.15 / Torch 2.8.0+cpu and Python 3.14.8 / Torch
  2.14.1+cpu
- Intra-op/inter-op PyTorch threads: 1 / 1
- OMP/MKL/OpenBLAS threads: 1 / 1 / 1
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0,
  dftd3 0.7.0, dftd4 0.8.0
- Imported dxtb source asserted as this worktree's `src/dxtb/__init__.py`.

The experimental records are `benchmarks/results/e6c1-w2c4/`.

## W2c3 thread-count qualification

The baseline tmpda profile recorded one intra-op thread and used the accepted
one-thread inter-op setup. The W2c3 Cholesky end-to-end trace records
`torch_threads=1` and `torch_interop_threads=128`; its OMP, MKL, and OpenBLAS
environment variables were all one. The W2c2a baseline used one inter-op
thread. The historical timing values in the W2c3 report are unchanged, but
the 3.5% end-to-end regression and 27% `linearize` increase are not a
perfectly controlled comparison. The Cholesky candidate is not rerun.

## Existing data flow

`evaluate_hamiltonian()` first calls `_transitional_diagonalize()`. The
returned `emo` values and AO-space generalized eigenvectors are retained in
`SCFSpectralResult`. For smearing, the function then constructs a
`_FermiCoreState` from spin-expanded orbital energies and the energy-sorted
valid-orbital prefix. The state retains scalar occupations/status, a detached
chemical-potential reference and offset, gap/saturation flags, and the
energy-sorted mask. The result also retains the Hamiltonian, density,
coefficients, occupations, and this Fermi state.

For the general orthogonalizer `T` used by functional SCF,
`T S T.T = I`, `U` diagonalizes `T H T.T`, and the AO coefficients are
`C = T.T U`. The functional input currently uses a symmetric inverse-square-
root transform, so `T.T = T`. A Cholesky inverse transform is a special case:
`T = L^{-1}`, giving `C = L^{-T} U`. Thus the value identity is
`P = C diag(f) C.T`, summed across spin channels. The accepted matrix path
constructs the same AO density as `T.T F T`. A nonidentity-overlap fixture
using the Cholesky transform matched the accepted density to `2.22e-16`; its
metric electron count `trace(P S)` was `1.9000000000000004` for a target of 1.9.

The scalar filling search and polish determine the primal chemical potential
`ref + offset` without a graph. `state.result.occupation` receives implicit
derivatives by differentiable scalar Newton steps. The stored `ref` and
`offset` remain detached. The accepted `fermi_density_matrix()` starts from
those values, performs two differentiable matrix electron-count corrections
for derivative order three, and evaluates a final matrix-Fermi density. It
also applies empty/full/gap handling, transforms back through `L^{-1}`, masks
inactive AO pairs, and sums spin channels. The separate free-energy setup
repeats its own two corrections and final matrix occupation; equal-partition
free energy uses the trace route but still builds that setup.

Consequently, the matrix density function does not currently receive the
already retained eigensystem or corrected matrix-Fermi intermediates. It does
receive the Hamiltonian, overlap transform, structural AO mask, electron
counts, thermal energy, and scalar primal Fermi state. Matrix-level
corrections enforce the count for the matrix approximation and provide the
required derivatives; they are not redundant merely because scalar filling
has a small residual.

## Candidate A — retained eigensystem density

The harness evaluated `C diag(sum_spin(f)) C.T` using the eigensystem from the
same test call and an independent bisection scalar Fermi-Dirac value oracle.
The rotated nondegenerate and exact-degenerate density values agreed with the
invariant matrix density to `2.22e-16` and `7.63e-17`, respectively. The
nonidentity-overlap case also matched to `2.22e-16`.

For the rotated nondegenerate spectrum `[-0.5, 0.6, 4.0]`, at `kT=0.2` and
spin electron counts `[1.3, 0.8]`, Candidate A matched the invariant
Hamiltonian gradient, second directional derivative, and third directional
derivative within `1.7e-16`, `9.1e-17`, and `2.1e-16` maximum absolute error.
This is the smooth case where reuse appears viable.

At the near-degenerate spectrum `[-0.3, -0.2999999999, 0.5]`, Candidate A's
third directional derivative was `-2.7487790686e11`, versus `0.03937090405`
for the invariant route. Its second derivative tensor differed by up to
`3.323e3`, and its third derivative tensor by up to `9.734e12`. At exact
degeneracy `[-0.3, -0.3, 0.5]`, the eigensystem route produced NaNs in reverse
and forward derivatives while the invariant route remained finite. This is
the expected eigenvector-gauge singularity; value agreement does not rescue
the derivative contract.

A deliberately noncontiguous structural-padding fixture also demonstrated
that AO validity cannot be applied as a prefix occupation mask. Its artificial
eigenvalue vector contained the padded zero eigenvalue between the two active
levels, so the naive prefix-filled eigensystem density differed from the
accepted masked density by `0.329`. This fixture is a mapping constraint, not
a claim about the production canonical packing path. A valid reuse design
would need the exact source-to-spectral eigenpair mapping and still would not
solve the demonstrated degeneracy failure.

**Candidate A is rejected.** It is not added to production and no full
first-through-third transform, mapped, padded physical, or tmpda performance
matrix is warranted after the reproduced higher-order failure.

## Candidate B — scalar chemical-potential reuse

The diagnostic evaluated one matrix-Fermi call at the scalar filling
`ref + offset`, with no differentiable matrix correction. In the smooth
fixture, its density value matched to `1.11e-16`; its electron-count Jacobian
was `[0, 0]`, while the accepted matrix path returned `[1, 1]`. The scalar
state intentionally stores detached chemical-potential values; its
derivatives are carried by the scalar occupation result, not by those fields.
Consequently, substituting the detached value into the matrix function
silently drops the electron-count response.

For the same smooth rotated test, Candidate B's Hamiltonian gradient differed
by `0.2213` maximum absolute value, its second derivative by `0.1621`, and its
third derivative by `0.1292`. The directional first/second/third derivatives
were respectively `0.09036`, `-0.05536`, and `-0.04245`, versus accepted
values `0.06233`, `-0.02208`, and `0.0007943`. These are correctness failures,
not tolerance questions. Candidate B is rejected.

The scalar filling and matrix correction share the same physical electron
constraint in the exact mathematical Fermi function. The numerical algorithms
are not interchangeable as implemented: the scalar search uses stable
orbital-wise occupation/tail expressions, gap classification, and a scalar
implicit derivative attachment; the matrix route differentiates the
matrix-function count and adds its own correction. Candidate B did not
preserve those derivatives. Reusing the scalar occupation itself would revert
to Candidate A and inherit its eigenvector degeneracy failure.

## Execution and limitations

Diagnostic command, run once in each primary profile:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0/bin/python -u \
  benchmarks/e6c1_w2c4_spectral_reuse.py \
  --output benchmarks/results/e6c1-w2c4/spectral-py311-torch280.jsonl
```

The same command used the Python 3.14 / Torch 2.14.1 interpreter and a
separate JSONL file. Both profiles produced the same pass/fail findings.
Python 3.14 emitted the existing upstream TorchScript deprecation warning.
No timing claim is made: both candidates failed numerical feasibility before
performance testing. No 32/128/255-orbital benchmark, tmpda response probe,
W2 rerun, or candidate physical-System sweep was performed. No pytest suite
was rerun because only a diagnostic harness and planning records changed; the
harness exercised the accepted matrix density, scalar filling, and generalized
overlap path directly.

Black, isort, Python compilation, and `git diff --check` passed. Ruff was not
run because it is not installed in the selected environment.

## Independent review

Requested reviewer: Luna / high. Actual model identity: not exposed by
runtime. The reviewer returned one MINOR finding: generalize the overlap
transform notation because functional SCF uses a symmetric inverse square
root, not the Cholesky form. This was corrected above; the Cholesky expression
is now identified as a special case. The reviewer confirmed the Candidate A
degeneracy failure, Candidate B derivative failures, the synthetic padding
caveat, W2c3 thread qualification, and absence of production SCF edits. No
blocking findings remain.

## Decision

No production optimization is implemented. Candidate A offers the more
direct duplication removal but cannot preserve the invariant method's
degenerate-spectrum derivatives. Candidate B cannot use the detached scalar
chemical-potential fields without losing electron-count and Hamiltonian
responses; using scalar occupations instead returns to Candidate A. No
implicit-response or end-to-end performance benefit is established.

The next optimization proposal should retain a gauge-invariant matrix
function and address reuse within its matrix correction/free-energy setup,
or develop a new invariant approximation with independently verified
degenerate first-through-third derivatives. That requires a new gated package
and coordination before any change to `spectral.py` during the active S1.1
work. No public route, E6bcompat behavior, or production solver changed.

## Verdicts

- W2c3 performance qualification: **CORRECTED**; timings retained, comparison
  explicitly qualified.
- Existing eigensystem reuse: **value identity verified for unpadded standard
  and generalized cases**; unsupported as a replacement under the derivative
  and mapped-padding constraints.
- Candidate A density values: **PASS** for tested smooth/generalized values;
  **FAIL** for exact synthetic noncontiguous padding mapping.
- Candidate A third derivatives: **FAIL** near/exact degeneracy.
- Candidate A degeneracy behavior: **FAIL**.
- Candidate B chemical-potential correction: **FAIL** as a one-call detached
  scalar-mu replacement.
- Candidate B electron-count derivatives: **FAIL** (`[0,0]` versus `[1,1]`).
- Candidate B third derivatives: **FAIL**.
- Padding and mixed-lane behavior: **NOT TESTED as a candidate production
  route**; synthetic noncontiguous mapping probe exposed the need for explicit
  eigenpair mapping. The accepted production path was not changed.
- Scaling-range contract: **PRESERVED**; no production kernel edits.
- Implicit-response parity: **NOT TESTED**; candidates failed earlier gates.
- Isolated spectral performance: **NOT TESTED**; no candidate passed numerical
  feasibility.
- Implicit-response performance: **NOT TESTED**.
- Production optimization: **NOT IMPLEMENTED**.
- One-/two-conformer tmpda performance: **NOT TESTED**.
- Peak memory: **NOT TESTED**.
- E7a: **PRESERVED; NOT RERUN**.
- E6bcompat: **PRESERVED; NOT RERUN**.
- W2/W3 performance clearance: **OPEN**.
- E6broute: **BLOCKED**.
- Public Calculator routing: **UNCHANGED**.
