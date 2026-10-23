# E6bimpl.1b — Mixed GFN1 fixed-trip SCF

Status: implementation slice delivered; **E6bimpl.1b remains open**.

Starting commit: `fbf3bf3ce6d78d1254ee226013ba17821494b4b8`.

This package adds a private mapped GFN1/PyTorch fixed-trip evaluation path.
It does not activate `Calculator(batch_mode=1/2)`, alter packed batching,
claim GFN2 mixed-System support, or complete E6bimpl.

## Numerical boundary

`evaluate_padded_systems()` validates the source values, stacks independently
created registered `PaddedExactSystemCore` values, and invokes
`torch.func.vmap(evaluate_padded_fixed_trip)` with the default output axis.
The mapped lane consumes the registered exact core plus explicit padded
positions, charge, and spin. It does not construct a mutable System,
IndexHelper, InteractionList, or per-System parameter table.

The lane reuses `evaluate_padded_integrals`, padded SCF input preparation,
the existing EEQ/SAD guess primitive, `_solve_representation(...,
eager=False)`, shared raw maps/mixers/Fermi/finalization, the public
`SCFStatus` projection, and the existing exact-System classical descriptor
evaluator. Captured default GFN1 ES2 and ES3 interactions are prepared from
their registered setup descriptors. Unsupported interaction descriptors
raise explicitly. Repulsion and official-release D3 remain active and are
added once outside the SCF contribution.

`PaddedCoreIndexData` carries the registered shell and orbital validity
masks. Spreads zero invalid mixer state and reductions use the E6bimpl.1a
negative-index-safe reductions. The decoder uses canonical AO maps for
AO-indexed matrices and rows, while energy-ordered molecular orbitals use the
physical spectral columns. Coefficients are checked for physical shape and
dtype; numerical comparisons use gauge-invariant density/spectrum values.
The wrapper requires source input tuples to follow `bucket.indices` order but
accepts noncontiguous source IDs. The reversed H2O/H2 integration case also
exercises a bucket whose source IDs are `(1, 3)`.

## Production parity evidence

Independent production `Model.setup()` Systems were used for:

| Pair | Inputs | Result |
| --- | --- | --- |
| H2 / H3+ | Neutral H2 and valid charged closed-shell H3+ | Mapped Result parity |
| H2 / H3 | Neutral H2 and valid open-shell neutral H3 | Mapped Result parity |
| H2 / H2O | Different elements, shells, and AO counts | Mapped Result parity |
| H2O / H2 | Reversed source order | Mapped Result parity |
| H2O / CH4 | Three-element union and noncontiguous canonical AO map | Mapped Result parity |

The charge/linear cases use positive fixed-trip budgets and compare decoded
Results with independent `System.singlepoint()` calls using identical
configuration and explicit charge/spin. Comparisons cover actual Result
reconstruction and tree structure, energy, SCF and free-energy contributions,
classical labels/values, charges, potential, density, orbital energies and
occupations, Hamiltonian, overlap, Hcore, overlap normalization, convergence,
residual, iteration count, and both public status fields. Every mapped tensor
leaf has the source count as its leading dimension. Coefficient values are not
compared elementwise because eigenvector sign and degenerate-subspace gauge
are not invariant.

Dipole-level and quadrupole-level tests additionally compare the optional
integral tensors against serial System Results. Absent quadrupoles remain
`None` at dipole level. The fixed-Hcore level retains absent multipole fields.

## Interactions, classical terms, and controllers

The production default GFN1 captured ES2/ES3 descriptors are active in the
mapped SCF map. The Result comparisons include their effect on energy,
density, charges, and potential. Model-owned `element.H.gam` (ES2) gradients
match the sum of independent serial gradients, remain finite and nonzero, and
do not mutate the parameter. The existing registered D3 term is active;
mapped gradients for the Model-owned D3 `s6` parameter match serial gradients
and leave the source parameter unchanged. Repulsion and D3 component values
and ordering match the serial result.

Charge SCP with linear and Anderson mixing and potential/Fock SCP with both
linear and Anderson mixing match their independent serial fixed-trip Results
for energy, density, iteration count, and status. No mixer or physical map was
copied. Other interaction descriptor families remain explicitly unsupported
by this private evaluator until a later E6bimpl package adds their exact
registered setup handling.

## Status and padding checks

The zero-trip one-shot case has `converged=False` and no fabricated spectral
failure. At positive budget, the finite-temperature H2/H2O fixture exercises
different outer convergence outcomes (`True`/`False`) while both Fermi status
codes remain `NONE`; serial and mapped iteration/status values match. A
fractionally charged H2 lane with `fermi_maxiter=0` reports
`FERMI_NONCONVERGENCE` in both trajectory and final-spectrum codes, while the
neutral H2O lane reports `NONE` in both fields. This confirms lane-local
status projection and distinguishes outer nonconvergence from Fermi failure.

An adversarial test perturbs only inactive padded coordinates while retaining
the full EEQ, ES2/ES3, D3, repulsion, and SCF call graph. Physical energy,
density, and charges remain unchanged; inactive-coordinate gradients are
zero. The mixed Jacobian has shape `(B, B, padded_natom, 3)`, diagonal
physical gradients match serial gradients, and off-diagonal System blocks
are zero. Reverse-mode `grad`/`jacrev`, forward-mode `jvp`/`jacfwd`, and
`vmap(grad)` agree for the tested positions. The H0 `kpol`, ES2 `gam`, and D3
`s6` parameter gradients retain Model ownership and serial-sum parity.

The added nonzero dummy Hamiltonian level and explicit orbital mask remain
the E6bimpl.1a convention. Tests do not differentiate dummy eigenvectors.
Padding invariance is established for the canonical layouts exercised here;
the suite does not establish arbitrary padded-capacity mutation beyond the
canonical bucket contract.

## Validation

The full padded SCF/spectral test module completed in all five approved CPU
profiles with official `tad-dftd3==0.7.0` and `tad-mctc==0.9.1`:

| Python | PyTorch | Focused module |
| --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 35 passed |
| 3.11.15 | 2.8.0+cpu | 35 passed |
| 3.12.10 | 2.11.0+cpu | 35 passed |
| 3.12.10 | 2.13.0+cpu | 35 passed |
| 3.14.8 | 2.14.1+cpu | 35 passed |

Command in each environment:

```bash
export OMP_NUM_THREADS=1
python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6bimpl1a_padded_spectral.py
```

On Python 3.14.8 / PyTorch 2.14.1, the combined focused E6bimpl.0/1a,
canonical padding, Model parameter tree, E6a.0–E6a.4, padding-safe
scatter, functional solver/status, and E6bcompat-policy selection completed
with **248 passed, 9 warnings** in 137.90 seconds. The warnings are the
existing Python 3.14 `torch.jit.script` future warnings. The selected padding
safe-scatter test file separately passed **10/10**. The combined command was:

```bash
export OMP_NUM_THREADS=1
python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6bimpl0_padded_integrals.py \
  test/test_calculator/test_e6bimpl1a_padded_spectral.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e6a4_transform_acceptance.py \
  test/test_indexhelper/test_padding_safe_scatter.py \
  test/test_scf/test_functional_solver.py \
  test/test_scf/test_public_scf_status.py \
  test/test_scf/test_e6bcompat_policy.py
```

A previous combined attempt had two fixture-construction failures because
hand-built `PaddedCoreIndexData` values lacked validity-mask arguments. The
fixtures were updated to provide masks matching their maps; the final
combined run passed. This was test setup only, not a production failure.

Black, isort, compileall, and `git diff --check` are run on the final files.
Ruff and mypy are not installed in the approved primary environment. PyTorch
emits expected performance warnings for the finite-temperature batching
fallbacks (`matrix_exp` and `count_nonzero`); tests pass. CUDA is outside the
CPU-only scope.

## Remaining acceptance work

This package remains open. The current evidence does not yet include explicit
instrumentation and comparison of the initial guess, first SCF map, selected
intermediate mixed iterates, and final trajectory state for each fixture.
The default ES2 descriptor and selected parameter derivative are covered,
but all interaction descriptor/resolution families and their position
derivatives are not. Optional multipole values are tested, but external field
and field-gradient call data are not integrated into this mixed path. The
parameter matrix covers `kpol`, ES2 `gam`, and D3 `s6`, not every parameter
family. SCFStatus tests cover valid status, one-shot, lane-local Fermi
failure, and differing outer convergence; not every public status category
has a reachable mixed production fixture. Additional malformed-core and
unsupported-component negative cases remain for follow-up.

GFN2 mixed execution, other component families not captured by this GFN1
boundary, public Calculator route activation, and legacy packed-route
retirement remain separately scoped. The next implementation work should
complete the GFN1 trajectory and component matrix while retaining the private
boundary; only a later authorized E6broute package may activate Calculator
batch routing. **Overall E6bimpl remains OPEN.**

## E6bimpl.1b-F final acceptance addendum

The follow-up acceptance package closes the original private GFN1/PyTorch
mixed-System boundary. The earlier status above is retained as the historical
partial-package report. The final trace and padding regressions cover initial
guess/map, intermediate states, residuals, damping, mixer history, and
lane-local convergence. Shared electric-field and field-gradient paths are
also compared to independent serial calls, including response derivatives.
Source identifiers `(4, 19)` and `(19, 4)` confirm that result order follows
the input sequence rather than treating IDs as dense tensor positions.

The final focused acceptance module passed 51 tests in each of the five
approved CPU profiles. A completed broader primary selection reported 360
passed and one failed. The sole Anderson position/parameter/field-gradient
node reproduces on the required starting SHA with identical dependencies and
the same `2.5006982595776606e-11` maximum absolute discrepancy against
`2e-13`; it is inherited. See
[`E6bimpl1b-final-acceptance.md`](E6bimpl1b-final-acceptance.md) for exact
scope and validation results.

E6bimpl.1b is complete at its private GFN1 numerical boundary. Overall
E6bimpl remains open for GFN2 mixed execution and remaining component,
compatibility, and route acceptance. No Calculator batch dispatch changed.
