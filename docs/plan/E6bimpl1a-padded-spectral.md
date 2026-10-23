# E6bimpl.1a — Padded SCF Inputs and Spectral Evaluation

**Starting SHA:** `e9f2deef0a8988e4862c7211ae1ad598baf597a0`
**Scope:** private GFN1/PyTorch mixed-System SCF preparation and spectral
evaluation. Public Calculator routing remains unchanged.

## E6bimpl.0 review

The starting tree is the accepted E6bimpl.0 commit. Its padded integral
evaluator consumes `PaddedExactSystemCore`, explicit positions and charge,
and registered setup Nodes. It reuses the existing overlap and H0 builders;
it does not construct a mutable `System` or `IndexHelper` in mapped code.
The two-axis gather change in that package handles transformed axes without
tensor-to-Python branching. No setup tensor mutation or public Calculator
routing change was found. The official dependency profile remains
`tad-dftd3==0.7.0` with `tad-mctc==0.9.1`; no candidate dependency is used.

## Negative-index reductions

The mandatory probe reproduced contamination in the inherited wrapper:
indices `[0, -1, 1]` and values `[3, 999, 4]` reduced to `[1002, 4]`.
The invalid source was redirected to destination zero after its index was
clamped. Sum reductions now zero invalid source values before scatter, which
produces `[3, 4]`. One-axis and two-axis cases, including a leading channel
dimension, all-valid and multiple-invalid maps, batched `vmap`, reverse
gradients, forward JVP, and the non-sum legacy behavior are covered in
`test_padding_safe_scatter.py`.

The generic utility accepts an optional static `output_size`. Padded index
reductions provide it explicitly so the mapped path does not call `.item()`
to infer an output extent. Other reduction modes retain their previous
invalid-index behavior; a general neutral-element/counting redesign is not
part of this package.

## Padded SCF input and spectral boundary

`setup_padded_scf_inputs()` in
`src/dxtb/_src/scf/functional/padded.py` consumes the registered padded core,
the E6bimpl.0 integral result, and explicit charge/spin tensors. The
`PaddedCoreIndexData` uses the core's registered shell/orbital maps and static
physical target extents. Reference occupations and electron counts use only
valid physical shells. `PaddedCoreValidity.orbitals` is the authoritative
orbital mask, including noncontiguous canonical layouts; validity is not
inferred from matrix zeros or eigenvalues.

The overlap is masked to physical entries and receives identity on inactive
orbitals. This leaves the physical overlap block unchanged, removes all
physical/inactive and inactive off-diagonal couplings, and makes the padded
matrix positive definite for the supported fixtures. The existing
inverse-Cholesky primitive is reused. The Hamiltonian keeps the physical block
unchanged and assigns inactive diagonal elements the fixed value
`1.0e4` Hartree. This parameter-independent dummy level is far above the
physical spectra measured in the fixtures, including a positive physical
orbital energy and charged H2. It prevents dummy states from being occupied
without assumptions that all physical levels are negative.

`evaluate_padded_spectrum()` calls the existing `evaluate_hamiltonian()`;
Aufbau and finite-temperature Fermi evaluation, density construction,
charge reductions, and Fermi status therefore use the shared production
primitives. The composed `evaluate_padded_scf_spectrum()` can be vmapped over
stacked registered cores. Physical numerical evaluation is not a Python loop.
Source charge/spin validity is checked eagerly once per independent source by
`validate_padded_charge_spin()` before mapping; this preserves established
validation without tensor-to-Python inspection inside `vmap`.

This is a spectral preparation boundary, not a complete SCF step. Interactions
are intentionally empty in `SCFInputs`; interaction call data, solver
trajectory, public `SCFStatus`, energy finalization, and `Result` construction
belong to E6bimpl.1b. At zero temperature the shared
`SCFSpectralResult.occupation` field carries the SCFInputs Aufbau seed, not a
post-diagonalization occupation vector. The padded seed is generated in
canonical AO order; this stage checks its per-spin electron totals and zero
padded entries, while physical occupied spectral states are validated through
density, energy, and electron counts. E6bimpl.1b must define how AO-ordered
seed metadata is projected through the explicit source↔canonical AO map.
Density, populations, electron counts, and eigenvalues are compared in
physical mapped order. GFN2 remains unsupported by the existing canonical
mapper and is not routed through GFN1 layout assumptions.

## Fixtures and evidence

The production fixtures use independent GFN1 `Model.setup()` calls and the
PyTorch driver:

| Pair | Purpose |
| --- | --- |
| H2 / H3+ | Different atom/AO/pair sizes with a valid charged open-shell H3 state |
| H2 / H2O | Different elements, shells and AO dimensions |
| H2O / H2 | Reversed source order |
| H2O / CH4 | Noncontiguous physical AO validity in canonical order |
| H2+ / H2O | Charged lane and dummy-state occupation check |

Tests compare independently built serial spectra with mapped spectra after
source-to-canonical AO mapping. They check H0/overlap physical blocks,
positive-definite overlap and inverse-Cholesky parity, density, charges,
eigenvalues, reference populations, electron/spin counts, status, and zero
padded populations. Zero and coincident padded coordinates leave physical
outputs unchanged. Position gradients agree for serial and mapped execution;
inactive-coordinate gradients and cross-System Jacobian blocks are zero.
JVP, `jacfwd`, and reverse differentiation are exercised. A gradient through
Model-owned `kpol` agrees with independent serial evaluations, and the source
parameter is unchanged.

Finite-temperature lane tests verify valid Fermi status, `TOO_MANY_ELECTRONS`,
and deterministic Fermi nonconvergence with `fermi_maxiter=0`; statuses remain
lane-local under mapping. The overfilled-electron status uses an extreme H2
anion input specifically to exercise the existing numerical status path. It
does not expand the accepted physical input domain. Invalid spin/electron
configurations are rejected by eager pre-map validation.

## Validation

The focused padding/spectral matrix (the two new test modules, E6bimpl.0
integral tests, and canonical-padding tests) passed in all five official-D3
CPU environments, 58 tests each:

| Python | PyTorch | D3 / mctc | Result |
| --- | --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 0.7.0 / 0.9.1 | 58 passed |
| 3.11.15 | 2.8.0+cpu | 0.7.0 / 0.9.1 | 58 passed |
| 3.12.10 | 2.11.0+cpu | 0.7.0 / 0.9.1 | 58 passed |
| 3.12.10 | 2.13.0+cpu | 0.7.0 / 0.9.1 | 58 passed |
| 3.14.8 | 2.14.1+cpu | 0.7.0 / 0.9.1 | 58 passed |

The broader official-D3 focused selection on Python 3.14.8 / PyTorch 2.14.1
completed in 682.96 s with **684 passed, 8 failed, 3 skipped**. Each failed node was then
rerun at the exact starting SHA in the same environment and failed there as
well. The five established GFN2 force/Hessian nodes were:

- `test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[LiH-md]`
- `test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[LiH-os]`
- `test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[SiH4-md]`
- `test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[SiH4-os]`
- `test_integrals/test_pytorch_gfn2.py::test_hessian`

Three additional inherited nodes also failed identically at the base:
`test_integrals/test_param_grad.py::test_exponent_gradient_matches_fd[gfn2-LiH]`
(SCF reaches a NaN Hamiltonian),
`test_integrals/test_quadrupole_getter.py::test_quadrupole_pytorch_matches_libcint_driver[LiH]`
(the same nonconvergent `eigh` path), and
`test_scf/test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`
(largest gradient difference `2.5006982595776606e-11` against a `2e-13`
absolute threshold). No new failure was attributed to this package.

After the channel-leading test was added, direct reduction consumers
(IndexHelper, Mulliken, and libcint overlap gradients) passed **138 tests with
1 existing xfail**; `test_functional_map.py` passed **57 tests**. A larger
ad-hoc combination including broad implicit-matrix and legacy-finalization
files was interrupted before pytest emitted its final summary and is not
counted as passing or acceptance evidence. The completed focused commands are
the package's matrix and direct consumer evidence.

All five environments used CPU builds and the official D3 release; CUDA was
unavailable. Ruff and mypy were not installed. Black, isort, compileall, and
`git diff --check` passed. No GFN2 mixed-System result is claimed.

## Next stage

E6bimpl.1b consumes this padded input and spectral boundary to implement
mixed-GFN1 fixed-trip SCF, immutable interaction preparation, energy
finalization, tensor status, and mapped `Result` reconstruction. Public
Calculator routing and legacy packed batching remain unchanged here.
