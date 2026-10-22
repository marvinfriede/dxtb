# E6a.4 — Public conformer transform and derivative acceptance

Status: the selected GFN1/GFN2 PyTorch transforms pass with official
`tad-dftd3==0.7.0` under the controlled `--no-deps` installation and local
tad-mctc compatibility shim. E1.3-L2 adds the trainable-zero-`s9` correction
and official-release validation. Overall E6a remains open for its remaining
method/component/derivative obligations.

The candidate-backed runs below are historical comparison evidence only.
The candidate branch and wheel are no longer supported or used; current
official-release test results are recorded in
[`E1.3-L2-official-d3.md`](E1.3-L2-official-d3.md).

Local starting commit: `9cc581d6deeef8bf5675d90b533222f16b4c69aa`.
The complete E1.3-U2 change from the reviewed pre-integration baseline was
preserved at `/tmp1/friede/tmp/E6a4-E1.3-U2-review.patch` with SHA-256
`7bf5a3bcec60333e1afa4ea973c8ab180121fb486fc7afd4d32059734219f7c7`.
The upstream candidate source identity is
`814f55795b1fe17cb07aba36bb81e53c004ad7b2`; this package made no changes to
that source tree.

## Dependency profiles

The candidate profile used the already built ordinary wheel
`tad_dftd3-0.7.0-py3-none-any.whl`, SHA-256
`c8277577155f1a6b14306a4700f63824a5e8db568acc5ab6261b1b62b41b7479`. Its
metadata requires `tad-mctc==0.9.1`; the five active CPU environments report
that exact mctc version and import D3 from their installed site-packages. No
upstream source path or shim participates in this candidate profile. The wheel
is an unreleased candidate, so these results do not constitute released
dependency acceptance.

The legacy profile remains tad-dftd3 0.6.0. The local tox Python 3.12/Torch
2.14.1 environment has D3 0.6.0 with mctc 0.9.1; direct `import tad_dftd3`
fails because the old package imports the removed
`tad_mctc.ncoord.coordination_number` name. The repository compatibility shim
allows its existing D3 path to run. The E6a.0 AtomicC6 blocker-characterization
node passed under that profile by observing the expected missing-JVP
`NotImplementedError`. This is a legacy control, never counted as candidate
forward-mode acceptance.

The candidate wheel digest and installed versions are recorded separately so
the two profiles cannot be confused. The candidate profile passed direct D3
import and the public dxtb integration tests. The populated candidate dxtb
environment's `pip check` still reports separately owned conflicts: D4 0.8.0
requires mctc 0.7.0 and multicharge 0.5.0, while D4 and libcint require NumPy
below 2 and this environment has NumPy 2.5.3. These do not arise from the
candidate D3 wheel.

## Original E6a boundary

The original E6a contract in roadmap section 17 is public conformer mapping of
one fixed production System using exactly:

```python
torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
```

It requires the PyTorch integral driver, the real registered `Result` and
`SCFStatus`, default `out_dims=0`, serial-loop parity for GFN1 and GFN2,
correct optional fields, fixed-trip status semantics, position and supported
parameter derivatives, fields/charges where batched, and independent
conformer lanes. Calculator remains the eager-policy owner. E6a is not mixed
System batching and does not promise whole-System libcint vmap.

The required public method boundary is GFN1/GFN2. GFN0 is non-self-consistent;
its IES/SRB path reaches the legacy xitorch padded eigensolver and is not
transform-safe. The original E6a public acceptance fixtures and E6a.1/.3
method matrix require GFN1/GFN2, not GFN0. GFN0 vmap is therefore a separately
scoped future transform task, not a newly imposed E6a prerequisite. Its
established non-SCF spin validation remains covered by E6a.1a.

## Public method/configuration evidence

All calculations below use production `ParamModule`, `Config`, `Model.setup`,
`System.singlepoint`, and public `Result` values, with float64 CPU tensors and
the PyTorch integral driver. Serial references are separate calls to the same
public fixed-trip `System.singlepoint` method.

| Method and composition | Configuration / evidence |
| --- | --- |
| GFN1 H2, H2O, CH4 | Linear mixing, charge SCP, zero temperature; complete result parity; 2- or 3-lane batches. H2 is also tested with 3 conformers for 2 atoms. |
| GFN1 H2 | Zero and finite temperature, positive budget, `maxiter=0`, explicit charge/spin/field mapping, public status and Calculator eager-policy controls. |
| GFN1 H2O | Active D3/Repulsion/ES2/ES3 aggregate; Halogen and ALPB public controls; D3-enabled JVP, jacfwd, jacrev, vmap, mapped jacfwd/grad, and D3 parameters `s6`, `s8`, `s9`, `a1`, `a2`. The GFN1 `s9` Model leaf is exactly zero and differentiable. |
| GFN2 H2 and H2O | Anderson/Fock SCP, zero temperature; complete Result parity with D4, Repulsion, ES2, AES2, ES3 and D4SC active; status, position gradients and cross-lane Jacobian evidence. |
| GFN2 H2 and H2O | Public JVP, jacfwd, jacrev and mapped jacfwd of the complete production energy; diagonal blocks agree with serial gradients and off-diagonal blocks are zero. |
| GFN1 and GFN2 H2 | Each combination of linear/Anderson mixer and charge/potential/Fock SCP at positive one-trip budget; complete Result serial/vmap parity (12 combinations). |

The added E6a.4 test module checks complete GFN2 Result tree parity for both
H2 and H2O, and the actual GFN2 descriptors, not a reduced electronic-only
path. Existing E6a.1
and E6a.3 tests retain the full GFN1/GFN2 result, charge/status, fields,
parameter ownership, and component checks. No contribution is dropped to
enable mapping.

## Derivative accuracy

The exact public GFN2/H2O energy path passed `jvp`, `jacfwd`, and `jacrev`; the
forward and reverse gradients agree. Differentiating the public two-lane
energy vector gives a Jacobian of shape `(2, 2, 3, 3)`. Its diagonal equals
the independent serial gradients and its off-diagonal blocks are zero. The
existing GFN2 vmap-gradient suite also verifies serial gradients and lane
independence.

For the public GFN1 and GFN2 H2O energies, reverse-over-reverse Hessians were
compared with central finite differences of the first derivative at step
`1e-4`. The exploratory maximum absolute/relative errors were respectively
`5.54e-9 / 1.01e-8` for GFN1 and `5.89e-9 / 1.10e-8` for GFN2. The regression
asserts these independently computed tensors with `atol=rtol=1e-7` and runs
in all five CPU profiles.

For full public GFN1 with nonzero D3 energy, a directional third derivative
was compared to a five-point finite difference of second derivatives at step
`1e-3`. AD gave `-2.038295593860998`; finite differences gave
`-2.0382955938976486`; absolute error was `3.67e-11`, relative error
`1.80e-11`. This check is part of the new regression test. The upstream
candidate independently has its own s9-zero forward/mixed derivative and
third-order accuracy tests. E6a.3's aggregate Hessian symmetry tests remain
in place.

Mapped GFN1 D3 parameter gradients reach the original Model-owned leaves for
`s6`, `s8`, `s9`, `a1`, and `a2` and match summed serial gradients. Existing
E6a.3 tests cover Model-owned Repulsion and ES2 leaves. No parameter tables
are copied into a System and no source parameters are mutated.

## Status, Calculator, and compatibility

The E6a.0–E6a.3 controls remain passing: public `Result` pytree parity,
tensor-valued `SCFStatus`, zero/finite temperature, outer nonconvergence,
`maxiter=0`, lane-local too-many-electrons/Fermi statuses, fields, charge and
spin mapping, and Calculator eager policy. The original fixed-trip System
and eager Calculator controllers remain distinct. Candidate D3 did not alter
status projection or Calculator routing.

The legacy 0.6.0 blocker test passes only as an expected-error control. The
candidate 0.7.0 public GFN1 JVP/jacfwd tests pass without excluding D3. No
whole-System libcint mapping, GFN0 vmap, custom callback transform support,
or mixed-System padding is claimed.

## Failure triage

The exact 10 known GFN1 production/legacy parity rows were run in Python
3.14.8/Torch 2.14.1 with the candidate D3 wheel both at current E6a.4 code
and at local start commit `9cc581d6deeef8bf5675d90b533222f16b4c69aa`.
Both runs produced the identical 10 failures, 2 passes, same exact parameter
IDs, and same maximum differences. They are inherited.

`test_anderson_solver_real_position_parameter_and_field_gradients` failed
with the same maximum absolute difference `2.5006982595776606e-11` against
the existing `2e-13` tolerance at current code and start commit. It is
inherited.

`test_implicit_d4sc_uses_system_setup_and_preserves_gradients` failed at
current code and the exact start commit with the same assertion: its implicit
legacy `Result.status` is `None`. The D4SC setup/gradient tests otherwise
passed. This is an existing implicit-result status limitation, not a public
conformer regression.

A larger non-slow xdist selection was attempted but interrupted near the end
without a pytest summary; it is not counted and its transient progress marks
are not classified. The focused public E6a suites and independent exact-node
comparisons above completed. The five previously tracked GFN2 force/Hessian
reference failures were not rerun in this package; their earlier matched-base
classification remains unchanged.

## Validation

Focused public suite, run in each of the five environments:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e13_d3_forward.py
```

Result: 59 passed per environment. The new E6a.4 module passed 17 tests in
each environment:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  test/test_calculator/test_e6a4_transform_acceptance.py
```

| Python / PyTorch | E6a.0–E6a.3 + D3 forward | E6a.4 new tests |
| --- | ---: | ---: |
| 3.10.22 / 2.6.0+cpu | 59 passed | 17 passed |
| 3.11.15 / 2.8.0+cpu | 59 passed | 17 passed |
| 3.12.10 / 2.11.0+cpu | 59 passed | 17 passed |
| 3.12.10 / 2.13.0+cpu | 59 passed | 17 passed |
| 3.14.8 / 2.14.1+cpu | 59 passed | 17 passed |

All five report tad-mctc 0.9.1 and candidate tad-dftd3 0.7.0. CUDA was
unavailable. The no-libcint tox profile is the legacy D3 0.6.0 environment;
candidate numerical acceptance used the five libcint-enabled CPU profiles.
After adding an explicit assertion that the differentiable GFN1 `s9` leaf is
exactly zero, the E1.3 public D3 file passed 12 tests in each of the same five
profiles.

Additional completed primary-environment controls:

- D3 core, energy, position-gradient, parameter-gradient, and E1.3 tests:
  78 passed.
- D4/D4SC setup and gradient controls: 121 passed, one inherited D4SC
  implicit-status failure described above.
- Legacy D3 0.6.0 AtomicC6 blocker node: 1 passed (expected error).
- Black, isort, compileall, and `git diff --check`: passed. Ruff and mypy are
  unavailable.
- The broader E6a/C8b/E6bcompat selection was attempted but interrupted and
  is not counted. Prior accepted C8b and E6bcompat package results remain the
  prerequisite evidence.

## Readiness decision

E6a.4's candidate-backed public transform and derivative preparation boundary
is demonstrated for the selected stable GFN1/GFN2 configurations, including
the full default GFN2 interactions and D3-enabled GFN1 forward AD. No new
dxtb production defect was reproduced. GFN0 IES/SRB mapping is not a required
GFN1/GFN2 E6a gate and remains separately scoped.

Overall E6a remains OPEN. The full-method E1.3 candidate is not a released
dependency, the broader non-slow suite did not complete, and the complete
project-wide method/component/derivative matrix has not been certified against
a release. No C8b, E6b, E7, or E4.6 ownership is changed.
