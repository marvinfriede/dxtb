# E6bimpl.2a — GFN2 canonical numerical setup

Status: **complete at the canonical setup and integral/H0 boundary**. Full
mixed-GFN2 SCF, interaction evaluation, and public Calculator batching remain
open and belong to later packages.

Starting SHA: `ff265d54c0bb886a4620058300e4f2d7fe0eb0c7`.
Final SHA: recorded by the implementation commit.

The implementation uses the official `tad-dftd3==0.7.0` distribution with
`tad-mctc==0.9.1`. No dependency or public Calculator route was changed.

## Production GFN2 schema inventory

Independent production `Model.setup()` runs use the PyTorch integral backend.
For the required default GFN2 setup, `H0SetupNode.cn_kind` is
`cn_d3(gfn2_count)`. The GFN2 H0 optional fields (`kq`, `kqat`, `h0rad`,
`kdiff`, `enshell`, `enscale4`, `eeq_model`, `cn_radii`, `cn_cutoff`,
`cn_max`, and `cn_kcn`) are absent in the examined production configurations;
the mapper rejects nonempty optional H0 state rather than inventing values.

The dimensions with physical indexing domains are:

| Domain | GFN2 setup leaves | Canonical operation |
| --- | --- | --- |
| Atoms | Core/H0/component numbers; H0 atom-to-element; D4/D4SC atom numbers, model numbers, radii; AES2 atom vectors; Repulsion atom-pair tables; ES2 atom masks; AES2 and D4SC pair data | Scatter through the source atom map; pair tensors scatter both atom axes |
| Shells | Backend/H0 shell-to-atom maps; H0 shell-to-type; H0 `kcn`, reference occupation, self-energy, polynomial, valence; shell ES2; shell ES3 | Scatter/remap through source shell and shell-type maps |
| AOs | Backend/H0 orbital-to-shell; PairPlan AO destinations | Rebuild from canonical shell offsets and angular widths |
| Unique elements | H0 unique values, `en`, `rad`, `kpair` | Canonical element union and source element map |
| Unique shell types | H0 angular classes, `hscale`, and shell vectors | Canonical `(element, shell ordinal, angular momentum)` union |
| Shell-pair plan | PairClass shell indices, AO block indices, weights | Rebuild canonical shell pairs, AO destinations, and per-source validity |
| Atom-pair/reference tables | D4 and D4SC `model_rc6` | Scatter only the leading two atom axes; retain the two reference-table axes |
| Independent reference dimensions | D4/D4SC trailing `(7, 7)` C6 reference dimensions | Kept unchanged; never treated as atom dimensions |
| Global/static | H0 enscale; D4/D4SC damping values, scalar model weights, cutoffs, parameter metadata; Config, integral algorithm/level, CN and damping descriptors | Kept unchanged and included in compatibility signatures |

The default GFN2 descriptors remain in original order: classical D4 and
Repulsion, then ES2, AES2, ES3, and D4SC interactions. Production D4SC model
extras are empty; nonempty custom D4SC extras are explicitly rejected by this
canonical mapper until their indexing semantics are supported. GFN1 and GFN2
have distinct static configurations and cannot share a bucket.

## Canonical remapping and numerical results

The mapper now accepts GFN1 or GFN2 only when the registered Config and H0
method agree. It preserves GFN2's `cn_d3(gfn2_count)` metadata. H0 element
tables (`kpair`, `en`, `rad`) and shell tables (`hscale`, `kcn`, `refocc`,
`selfenergy`, `shpoly`, `valence`) use their respective canonical maps.
Default D4 and D4SC data scatter their per-atom and atom-pair axes while
preserving the model C6 reference axes. AES2 atom vectors and pair masks,
ES2 atom/shell mappings, ES3 shell Hubbard derivatives, and Repulsion pair
matrices are also remapped. Dynamic tensor values retain autograd edges.

Complete registered default cores stack with `Node.stack`; the transformed
path calls `torch.func.vmap(evaluate_padded_integrals)` and existing production
integral/H0 kernels. There is no per-System numerical evaluation loop inside
the mapped call and no reconstructed System façade.

Fixtures were independently constructed for H2/H2, H2/LiH, H2/H2O,
H2O/H2, and H2O/CH4. Canonical bucket dimensions were:

| Pair | Atoms | Shells | AOs |
| --- | ---: | ---: | ---: |
| H2/H2 | 2 | 2 | 2 |
| H2/LiH | 3 | 4 | 6 |
| H2/H2O | 3 | 4 | 6 |
| H2O/CH4 | 6 | 8 | 12 |

Every physical H0 and overlap entry was selected through the explicit
source-to-canonical AO map and compared to independent unpadded
`build_integral_matrices()` output. Exploratory max absolute errors were:

| Pair | H0 max error | Overlap max error |
| --- | ---: | ---: |
| H2/H2 | 0 | 0 |
| H2/LiH | `5.6e-17` | `1.2e-16` |
| H2/H2O | `1.2e-16` | `1.2e-16` |
| H2O/H2 | `1.2e-16` | `1.2e-16` |
| H2O/CH4 | `1.2e-16` | `1.2e-16` |

Tests use `2e-13` value tolerances. Canonical reversed input order yields
serial parity in the reversed lane order. The H2O source AO map is
noncontiguous and is explicitly exercised. PairPlan validity bits reconstruct
the source shell-pair set; H0 reference occupations match exactly after
shell-type mapping. Dipole and quadrupole blocks match serial results where
requested.

Inactive atom coordinates are zeroed by the existing numerical evaluator;
an adversarial change to padded coordinates leaves H0/overlap unchanged and
has zero derivative. Invalid PairPlan entries remain masked. Position JVP,
`jacfwd`, and `jacrev` agree with serial derivatives within `2e-10`; mapped
cross-System blocks and inactive-coordinate derivatives are zero, while
physical diagonal derivatives are nonzero and match serial calls.

The active GFN2 H0 parameter is `hamiltonian.xtb.wexp` (GFN2 does not use the
GFN1 `kpol` parameter for this H0 path). Its mapped gradient equals the sum of
independent serial gradients, is nonzero, and leaves the Model-owned
parameter unchanged. No parameter table is copied into an exact core.

## Validation

The new GFN2 canonical module passed **10 tests in each** environment. The
combined focused selection passed **109 tests in each** environment:

| Python | PyTorch | Official tad-mctc / tad-dftd3 | Combined result |
| --- | --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 0.9.1 / 0.7.0 | 109 passed |
| 3.11.15 | 2.8.0+cpu | 0.9.1 / 0.7.0 | 109 passed |
| 3.12.10 | 2.11.0+cpu | 0.9.1 / 0.7.0 | 109 passed |
| 3.12.10 | 2.13.0+cpu | 0.9.1 / 0.7.0 | 109 passed |
| 3.14.8 | 2.14.1+cpu | 0.9.1 / 0.7.0 | 109 passed |

The combined command was:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_calculator/test_e6bimpl2a_gfn2_canonical.py \
  test/test_calculator/test_e6bimpl0_padded_integrals.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_e6bimpl1a_padded_spectral.py \
  test/test_hamiltonian/test_gfn2.py
```

The new GFN2 module was also rerun by itself after the final test assertions:
**10 passed** in all five profiles. The PyTorch 3.12+ environments emitted
only existing TorchScript deprecation notices; finite-temperature GFN1 tests
also emit PyTorch batching fallback performance warnings.

An additional GFN2 PyTorch force/Hessian selection reported **124 passed,
5 failed** on Python 3.14.8/Torch 2.14.1. Exact failing nodes:

- `test/test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[LiH-md]`
- `test/test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[LiH-os]`
- `test/test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[SiH4-os]`
- `test/test_integrals/test_pytorch_gfn2.py::test_energy_and_forces[SiH4-md]`
- `test/test_integrals/test_pytorch_gfn2.py::test_hessian`

All five exact nodes were rerun on starting SHA `ff265d5` with the same
environment and all five reproduce. They fail in the existing single-System
functional SCF eigensolver with NaN Hamiltonian entries and
`torch.linalg.eigh` ill-conditioned/nonconvergence errors. They are inherited
GFN2 numerical failures, not canonical padding regressions. GFN2 mixed SCF is
not implemented or claimed here.

Black, isort, compileall, and `git diff --check` are run on the final changed
Python files. Ruff and mypy availability is recorded with final checks.

## Boundary and next work

E6bimpl.2a accepts complete default GFN2 registered-core padding/stacking,
PyTorch overlap/H0 evaluation, optional multipole evaluation, and selected
position/parameter derivative behavior. Custom D4SC model extras and
non-default/custom component configurations remain explicitly unsupported
until mapped by a later package. No GFN2 interaction kernel or SCF solver is
run by this package. The next GFN2 numerical stage is E6bimpl.2b: prepare the
mapped interaction data and route the existing fixed-trip SCF solver over
these canonically padded cores. Public Calculator batching remains owned by
E6broute.
