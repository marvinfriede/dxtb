# E6bimpl.2b-F — mixed GFN2 final acceptance

Status: **E6bimpl.2b complete at the private GFN2 numerical boundary; overall
E6bimpl remains open.**

Starting SHA: `f675c9d7f3d22e133e398d47e4d1d7bed5f6f255`.

This acceptance package adds regression evidence only; it found no production
defect requiring a code change. Public `Calculator` batch routing, legacy
packed execution, and the official D3/D4 dependencies are unchanged. GFN0,
custom components, GFN2 public batch routing, and E6bimpl.2c's broader
cross-method acceptance remain outside this package.

## Fermi spectral mask

The source audit confirms that `SCFInputs.orbital_mask` is an AO-index mask,
while finite-temperature filling receives energy-sorted eigenvalues. The
generalized eigensolver places the physical spectral subspace before the
high-energy inactive subspace. `evaluate_hamiltonian()` therefore constructs
the Fermi mask from the physical orbital count in sorted spectral order, and
continues to use the AO mask when projecting density matrices. The same AO
mask is not reused as the Fermi mask.

The acceptance test covers H2/LiH, H2/H2O, reversed H2O/H2, and H2O/CH4.
At 10,000 K, the H2/H2O pair has a positive physical frontier eigenvalue with
a fractional occupation. The mapped and serial calculations agree for the
full Result, electron count, physical occupations, density, free energy,
entropy, reconstructed chemical potential, and Fermi convergence status.
Padded orbitals have energy above 9,000 Hartree and exactly zero occupation.
The finite-temperature position gradient also matches serial for both source
orders. These explicit occupation checks prevent a matching total energy from
hiding a mask permutation.

The dummy level is retained as spectral separation, while explicit validity
masks enforce physical filling and density. The tests cover several inactive
orbitals and reversed canonical mappings; arbitrary dummy eigenvector
derivatives are not used as a reference.

## D4 EEQ cutoff policy

The mapped route uses **one shared static EEQ cutoff per canonical bucket**.
Before `vmap`, it reads each D4 descriptor's static cutoff and rejects a
bucket whose cutoffs differ. The agreed scalar is then passed as a captured
ordinary Python float into the mapped evaluator. `get_eeq_guess()` uses the
upstream `CNModel.replace(cutoff=float(cutoff))` contract while the helper is
called inside `vmap`, but the conversion sees only that captured static float,
never a BatchedTensor. Different cutoffs must be placed in separate buckets.
There is no derivative contract for this hard structural cutoff.

The regression evaluates 0.8 and 4.0 Bohr plus the official 25.0-Bohr default
on H2O. EEQ charges differ across the selected cutoffs; the observed charges
are explicitly compared to the charges received by the mapped classical D4
adapter, so selecting the default or first lane silently fails the test. For
each cutoff, classical D4 atom-resolved energy matches an independent
unmapped call within `atol=2e-8, rtol=2e-8`. The chosen GFN2 classical D4
parameters give very small charge sensitivity, so cutoff sensitivity is
by claiming a large D4 energy change. Position `jacrev` and JVP at fixed
4.0-Bohr cutoff pass through `evaluate_padded_systems()` and its internal
`vmap`, and match the independent serial expression. Mixed cutoffs in one
bucket raise the established static-configuration error.

## Independent interaction references

For H2/LiH, H2/H2O, and reversed H2O/H2, serial references are built from
ordinary production Systems using their unpadded `_interaction_data()` and
`prepare_functional_interactions()` boundary. They are compared against
mapped preparation and evaluation after explicit atom, shell, and AO inverse
maps. Trial monopoles, dipoles, and quadrupoles are nonzero.

ES2, ES3, AES2, and D4SC geometry-data tensors match the independent source
System tensors within `atol=5e-9, rtol=5e-9`. Each interaction's monopole,
dipole, and quadrupole potential channels and atom-resolved energy match the
unmapped functional kernels at the same tolerance. The fixture asserts that
each tested contribution is nonzero. A separate derivative-consistency test
compares each interaction potential against the gradient of its corresponding
interaction energy with respect to monopole, dipole, and quadrupole trial
charges (`atol=2e-8, rtol=2e-8`). The multipole values use the production six
component quadrupole convention.

## Classical and interaction parameter gradients

Mapped total-energy gradients are compared to the sum of independent serial
gradients for differentiable Model-owned parameters:

| Family | Parameter | Result contribution selected |
| --- | --- | --- |
| ES2 | `element.O.gam` | total energy |
| ES3 | `element.O.gam3` | total energy |
| AES2 | `multipole.damped.dmp3` | total energy |
| D4SC | `dispersion.d4.a1` | SCF energy |
| Classical D4 | `dispersion.d4.a1` | `DispersionD4` |
| Repulsion | `element.O.arep` | `Repulsion` |

Each derivative is finite and nonzero; mapped and serial sums match within
`atol=2e-7, rtol=2e-6`. An additional H0 `wexp` gradient matches at
`atol=2e-6, rtol=2e-6`. Parameter identity and value are checked before and
after calls to verify a single Model owner and no mutation. A mixed position /
ES2-parameter directional derivative agrees with the independent serial
composition within `atol=3e-6, rtol=3e-5`.

## Charged Systems, open shell, and statuses

The full mixed calculation checks H2+ / H2− lanes with distinct charges and
spin 1 against separate serial `System.singlepoint()` calls. A second test
places neutral H2 and an over-capacity H2 lane (charge −3, spin 1) in one
bucket. Decoded energy, iterations, convergence, Result fields, and both
SCFStatus codes match serial. The neutral lane has status code 0; the
over-capacity lane has `TOO_MANY_ELECTRONS` (code 2) in trajectory and final
status, without changing the other lane.

A marginal H2/H2O run demonstrates lane-local outer convergence (`[True,
False]`) with independent iteration counts/residuals; padding-capacity
invariance is checked on the same marginal setting. Fixed-trip `maxiter=0`
and positive-budget nonconvergence remain covered by the inherited GFN2
acceptance module. A stable GFN2 fixture that produces finite-temperature
Fermi *nonconvergence* was not found: reducing Fermi iterations still
converged in the tested fixtures. The generic public SCFStatus projection
suite independently covers `FERMI_NONCONVERGENCE`, invalid temperature,
electron-count failure, too-many-electrons, and separate trajectory/final
codes. Mixed GFN2 success and too-many-electrons statuses are directly
verified lane by lane here; no unstable physical fixture was introduced for
the remaining Fermi failure category.

## Regression results

The focused E6bimpl.2b module contains 41 collected cases and passes in all
five CPU profiles with the official packages installed (`tad-mctc==0.9.1`,
`tad-dftd3==0.7.0`, `tad-dftd4==0.8.0`):

| Python | PyTorch | Result |
| --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 41 passed |
| 3.11.15 | 2.8.0+cpu | 41 passed |
| 3.12.10 | 2.11.0+cpu | 41 passed |
| 3.12.10 | 2.13.0+cpu | 41 passed |
| 3.14.8 | 2.14.1+cpu | 41 passed |

The completed broader relevant primary-environment selection had **500
passed and 6 failed**. All six exact nodes were rerun on a detached worktree
at this package's starting SHA and reproduced there:

- `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[0]`
- `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[1]`
- `test_functional_fock.py::test_fock_valid_trajectory_with_invalid_final_spectrum`
- `test_functional_potential.py::test_potential_maxiter_zero_tracks_raw_and_final_fermi_status`
- `test_functional_fock.py::test_fock_final_free_energy_matches_legacy[300K-atomic]`
- `test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`

Five are the already characterized maxiter-zero trajectory/final status and
Anderson gradient discrepancies. The 300 K atomic-partition free-energy
control misses its existing `2e-13` absolute tolerance by
`6.34e-14` on Python 3.14 / Torch 2.14.1. It passes on the two earlier CPU
profiles. These are unchanged baseline failures, outside the new GFN2
interaction/derivative implementation. No existing test reference or
tolerance was modified.

An exploratory 300 K, five-trip H2/LiH finite-temperature serial reference
produced NaNs on Python 3.11 / Torch 2.8. Raising the finite temperature to
10,000 K yielded a stable, finite fractional-occupation case and the complete
suite then passed. This was a fixture stability correction; it did not relax
parity tolerances or alter production code.

## Remaining scope

No production defect was confirmed in this acceptance pass. The following
remain separately owned: GFN2 mixed-System public Calculator routing and
legacy retirement; GFN0; arbitrary custom component extensions; CUDA and
libcint whole-System `vmap`; and E6bimpl.2c's broader cross-method/component
acceptance, numerical derivative accuracy matrix, and final compatibility
readiness. E6bimpl.2b is complete at its private mixed-GFN2 fixed-trip
boundary; overall E6bimpl remains **OPEN**.

Formatting and source checks completed: Black, isort, `compileall`, and
`git diff --check` pass. Ruff and mypy are not installed in the primary CPU
environment. The full 41-case module was rerun after the final cutoff-transform
change in all five profiles; the cutoff regression itself was also rerun
independently across the same five profiles. PyTorch emitted its known
`matrix_exp`/`count_nonzero` batching fallback warnings; no numerical
tolerance or reference data was changed.
