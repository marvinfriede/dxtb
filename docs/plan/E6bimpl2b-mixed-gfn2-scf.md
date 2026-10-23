# E6bimpl.2b — mixed GFN2 interactions and fixed-trip SCF

Status: **partial; E6bimpl remains open**.

Starting SHA: `82d7a84cf405ba0cd1ddf21557272b2b8816e446`.

This package extends the private canonical GFN2 numerical path. It does not
change public `Calculator` batch routing, retire legacy packed execution, or
modify the official D3/D4 dependencies. Runtime tests use official
`tad-dftd3==0.7.0`, `tad-mctc==0.9.1`, and the installed `tad-dftd4==0.8.0`.

## Implementation changes

The padded SCF preparation boundary now accepts matching GFN1 and GFN2 exact
cores. GFN2 interaction preparation retains the production ES2 and ES3 entries
and builds geometry call data for AES2 and D4SC from their captured setup
Nodes. Potential and interaction-energy calculations reuse the existing
functional interaction kernels. The configured default GFN2 interaction order
remains ES2, AES2, ES3, D4SC; the classical order remains D4, Repulsion.

The GFN2 default D4 classical term obtains its EEQ charges through the existing
`get_eeq_guess` and D4 adapter. The D4 EEQ cutoff is static for a compatible
bucket, so the bucket evaluator validates and extracts it before `vmap` and
passes the resulting charges to the production D4 kernel. The direct D4 helper
accepts optional call-local charges; existing callers retain the original
default behavior. Canonical D4 static compatibility now includes dispersion,
coordination-number, and EEQ cutoffs.

The first full mixed-GFN2 test exposed a spectral mask mismatch. Canonical AO
validity can be noncontiguous, while Fermi masks index energy-sorted orbitals.
Applying the AO mask to the sorted spectrum caused wrong padded occupations
and density for finite-temperature mixed GFN2. The shared map now constructs a
prefix energy-space mask from the valid-orbital count. Fermi matrix routines
use the explicit AO mask for matrix projection while retaining the sorted
mask for filling/status. The projector similarly applies its active-count
mask in sorted eigenvalue order. Ordinary unpadded behavior is unchanged.

No second SCF solver, D3/D4/AES2 physical kernel, mutable System, IndexHelper,
or per-lane parameter table was introduced. Public Calculator routes were not
changed.

## Public numerical evidence

The new tests construct production GFN2 Systems independently and compare
`evaluate_padded_systems` Results against serial `System.singlepoint` Results.
The fixtures cover H2/H2, H2/LiH, H2/H2O, reversed H2O/H2, and H2O/CH4. H2,
LiH, and H2O use zero electronic temperature; methane uses 300 K because its
zero-temperature Aufbau occupation is gauge-dependent within a degenerate
frontier. The methane comparison groups occupations by degenerate energy
clusters and also requires density, spectrum, Hamiltonian, energy, charge,
potential, and status parity. A separate GFN2 zero-trip test checks the shared
one-shot contract.

Result checks include energy and SCF partition, classical labels/order and
per-atom contributions, free energy, charges, density, orbital energies,
degeneracy-invariant occupations, potential, Hamiltonian, overlap, H0,
dipole/quadrupole integrals, overlap normalization, iteration count,
convergence, residual, and both SCFStatus codes. Raw MO coefficients are not
compared because eigenvector signs and degenerate-subspace gauges are not
unique; density and the energy-ordered spectrum are compared instead.

A nonzero monopole/dipole/quadrupole trial state exercises each default GFN2
interaction separately through its potential and atom-resolved energy kernel.
Mapped values agree with direct lane evaluations. A separate call-data check
compares ES2, AES2, ES3, and D4SC data under mapping. Full serial/mapped SCF
parity confirms that the default interaction and classical contributions are
included in the production call graph without duplication.

The trace regression uses the existing fixed-trip solver trace and independent
serial System calls. For H2/H2O it compares mapped and serial Hamiltonians,
density matrices, residual norms, active masks, iteration counts, damping,
Fermi status, and final Result energies through 30 configured trips. At least
one lane changes its active/converged state before the other; the fixed-trip
loop remains lane-local.

Position reverse gradients for mixed H2/LiH match independent serial gradients;
inactive-coordinate and cross-System Jacobian blocks vanish, while physical
gradients are nonzero. JVP and jacfwd agree with reverse mode. The mapped
total-energy gradient through Model-owned GFN2 H0 `wexp` matches the sum of
independent serial gradients, remains nonzero, and does not mutate or replace
the Model parameter leaf.

The six charge/potential/Fock × linear/Anderson combinations are compared
against serial fixed-trip Results on H2/LiH. These tests establish numerical
execution for that fixture and do not assert that every method-specific
configuration is supported by the original GFN2 configuration policy.

## Scope limits and remaining work

This is not the complete E6bimpl.2b acceptance matrix. Direct component tests
compare mapped lane execution with direct evaluation on the same canonical
setup; independent unpadded references for every AES2/D4SC subterm and all
their parameter derivatives have not been established here. The new derivative
coverage is position plus one active H0 parameter. ES2, AES2, ES3, D4SC, and
classical D4 parameter-gradient matrices, charged/open-shell fixtures,
lane-local failure cases, additional finite-temperature failures, and
cross-System parameter derivatives remain for E6bimpl.2c or a narrowly scoped
follow-up. No GFN2 component is silently dropped; unsupported descriptors
continue to raise explicitly.

No public mixed-System route is activated. GFN0, libcint whole-System mapping,
custom component extensions, and legacy route retirement remain separately
owned.

## Validation

The final E6bimpl.2b module passes **22 tests in each** approved CPU
environment with official D3 0.7.0. This includes noncontiguous source IDs
`(4, 19)` and reversed `(19, 4)`, with mapped Result decoding preserving the
input lane order. The combined focused subsystem selection also ran in every
environment before those two source-ID cases were added; the new cases were
then run separately in all five environments. Its inherited failures vary slightly by Torch
version:

| Python / PyTorch | Focused combined result |
| --- | ---: |
| 3.10.22 / 2.6.0+cpu | 368 passed, 5 failed |
| 3.11.15 / 2.8.0+cpu | 368 passed, 5 failed |
| 3.12.10 / 2.11.0+cpu | 367 passed, 6 failed |
| 3.12.10 / 2.13.0+cpu | 373 passed, 6 failed |
| 3.14.8 / 2.14.1+cpu | 373 passed, 6 failed |

The combined selection contains the new package tests plus E6bimpl.0/1a,
E6bimpl.2a, C8b canonical/parameter-tree, E1.3 D3, classical D4/D4SC,
AES2, functional map/solver/SCF-mode/mixer, and public SCFStatus controls.
Across the versions, the failed nodes are the existing maxiter-zero
trajectory-status expectations, the tight Anderson gradient comparison, and
the 300 K atomic-partition free-energy comparison. All six exact nodes fail
on starting SHA `82d7a84` under Python 3.14/Torch 2.14.1 with the same
configuration; the 300 K atomic-partition node passes on the 3.10/3.11
profiles. No new failure was identified in the GFN2 or padded spectral paths.
The reproduced nodes are:

- `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[0]`
- `test_functional_fock.py::test_fock_maxiter_zero_uses_one_raw_and_one_final_spectrum[1]`
- `test_functional_potential.py::test_potential_maxiter_zero_tracks_raw_and_final_fermi_status`
- `test_functional_anderson.py::test_anderson_solver_real_position_parameter_and_field_gradients`
- `test_functional_map.py::test_fock_final_free_energy_matches_legacy[300K-atomic]`
- `test_functional_fock.py::test_fock_valid_trajectory_with_invalid_final_spectrum`

The focused regression selection used:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6bimpl2b_gfn2_scf.py \
  test/test_calculator/test_e6bimpl2a_gfn2_canonical.py \
  test/test_calculator/test_e6bimpl1a_padded_spectral.py \
  test/test_calculator/test_e6bimpl0_padded_integrals.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_e13_d3_forward.py \
  test/test_classical/test_dispersion/test_d4.py \
  test/test_classical/test_dispersion/test_d4sc_grad.py \
  test/test_coulomb/test_aes2.py \
  test/test_scf/test_functional_map.py \
  test/test_scf/test_functional_solver.py \
  test/test_scf/test_functional_potential.py \
  test/test_scf/test_functional_fock.py \
  test/test_scf/test_functional_anderson.py \
  test/test_scf/test_public_scf_status.py
```

The shared spectral change is covered by the padded spectral and functional
SCF suites. Torch's existing `matrix_exp` and `count_nonzero` vmap
batching-fallback performance warnings remain; no numerical tolerance or
reference was changed. PyTorch 3.12+ also emits its existing TorchScript
deprecation notice.

The mapped default D4 path is validated at its shipped cutoffs. A custom
nondefault EEQ cutoff is accepted only as a bucket-static value and is not a
separate numerical acceptance fixture in this package. Ruff and mypy were not
available in the primary environment.

`AGENTS.md` remains untouched and untracked. The final local SHA is recorded
with the package commit.

## E6bimpl.2b-F final acceptance addendum

The original report above remains the implementation-stage record. The
follow-up acceptance in [`E6bimpl2b-final-acceptance.md`](E6bimpl2b-final-acceptance.md)
closes its remaining interaction-reference, cutoff, component-gradient,
charged/open-shell, and lane-local failure-status gates. The focused module
now has 41 cases and passes in all five approved CPU environments. Added
independent unpadded interaction call-data and potential/energy references,
custom EEQ cutoff forwarding and bucket rejection controls, active Model
parameter gradients for ES2/ES3/AES2/D4SC/D4/Repulsion, mixed position/ES2
parameter response, charged/open-shell lanes, and a lane-local
too-many-electrons status case.

No production change was needed in E6bimpl.2b-F. The spectral mask remains
split between AO-index validity and energy-sorted Fermi validity. The D4 EEQ
policy is one shared static cutoff per bucket. The primary combined
regression selection reproduces six existing functional SCF failures on the
exact `f675c9d` package base; the individual nodes and the measured Torch
2.14 discrepancy are listed in the final acceptance report. These failures
are not attributed to the GFN2 mixed-System work. E6bimpl.2b is complete at
its private numerical boundary; E6bimpl.2c retains the broader
cross-method/component and derivative acceptance. Public Calculator routing
remains unchanged.
