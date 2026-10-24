# E7a — batch-critical padding safety

**Status: OPEN.** The first-order value, masking, and derivative selection
passes in all five supported CPU environments. E7a cannot close because the
new full-energy third-directional derivative comparison disagrees between
mapped and independent serial GFN1/GFN2 execution. The mismatch is reproducible
in both PyTorch 2.6 and 2.14. No tolerance or reference was changed.

## Baseline and scope

- Starting SHA: `efebc399a6fbda8810df3ef02ba3986c4e1c8f15`.
- Final SHA: recorded after the local commit.
- Public Calculator routing is unchanged.
- The E6broute public packed Result adapter remains out of scope.
- E6bcompat's retained/mapped numerical boundary is reviewed in its accepted
  report and remains the boundary exercised here.
- `AGENTS.md` was not modified or staged.

## Singular-operation inventory

| Operation family and production owner | Validity input | Risk and current handling | Evidence |
|---|---|---|---|
| Integral coordinates and pair kernels: `evaluate_padded_integrals`, `build_integral_matrices` | `PaddedCoreValidity.atoms`, `pair_entries`, `orbitals` | Inactive coordinates are replaced by zero before the shared integral kernels; outputs are masked by AO validity. | E6bimpl.0 integral suite; E7a full-SCF inactive-coordinate test. |
| Coordination numbers: `ncoord.legacy.coordination_number`, GFN2 `cn_d3(..., counting_function=gfn2_count)` | Real-number pair mask; padded fixed-trip atom mask | Pair counts are excluded by their real-pair masks. The mapped fixed-trip boundary now removes arbitrary inactive coordinates before interaction/classical preparation. | E6bimpl.2b GFN2 component/cutoff tests; E7a full-SCF test. |
| EEQ and classical D4: `get_eeq_guess`, `dispersion_d4_energy` | Padded atom numbers, pair masks, shared static EEQ cutoff | The fixed-trip boundary supplies sanitized coordinates; cutoff compatibility is checked before `vmap`. | E6bimpl.2b cutoff and mixed-D4 tests. |
| AES2: `build_aes2_data`, `_build_atom_coulomb_matrix` | `AES2CoreSetup.pair_mask` plus padded atom mask | A defect was found: invalid pairs had nonzero isotropic dipole-dipole coupling. Invalid pair displacement is now made nonsingular before the norm/reciprocal powers, and all three output matrices are zeroed after construction. | New adversarial AES2 matrix/contraction test. |
| D4SC: `build_d4sc_data` | Padded numbers and internal pair/reference masks | Call-local preparation receives sanitized positions. Reference-weight normalization is covered by E6bimpl.2b independent component tests. | E6bimpl.2b interaction references and padding tests. |
| Overlap/H0 and multipoles: `evaluate_padded_integrals`, `_mask_integrals` | Atom, shell, AO, and PairPlan masks | Inactive atoms are sanitized before shared kernels; invalid AO blocks are removed afterward. | E6bimpl.0, E6bimpl.2a and E7a selections. |
| Spectral decomposition: `spectral._prepare_matrix`, `map.diagonalize` | Canonical AO validity and energy-sorted spectral validity | Physical overlap is completed with inactive identity; spectral routine constructs a finite detached inactive block; explicit masks govern occupations and density. | E6bimpl.1a padded spectral and E6bimpl.2b Fermi selections. |
| Fermi occupations and density: `spectral.fermi_density_matrix`, `map` occupation path | Sorted spectral-state mask | Occupations and density are masked through the registered orbital validity contract. | E6bimpl.1a and E6bimpl.2b finite-temperature/status tests. |
| Charge reductions: `CoreIndexData`, `utils.scattergather.wrap_scatter_reduce` | Source-to-shell/atom maps and explicit output extents | Negative source indices are filtered before sum accumulation. | `test_padding_safe_scatter.py`, including nested `vmap`, gradients, and JVP. |
| SCF residual/mixer state: `functional.solver` projection and residual functions | SCP-specific atom/shell/AO masks | Inactive solver state is projected before residual and mixer use. | E6bimpl.1a adversarial inactive-state test and E6bimpl.2b lane-local/marginal-capacity tests. |
| Result decoding: `decode_padded_result` | Explicit source atom/shell/AO maps | Decoding indexes canonical maps, not source identifiers as dense offsets. | E6bimpl.1b-F, E6bimpl.2b-F and E6bimpl.2c acceptance tests. |

This table covers the current accepted mapped GFN1/GFN2 route. It does not
claim safety for future components, arbitrary custom extensions, libcint
transforms, CUDA, or the public packed Result adapter.

## Confirmed correction

`_build_atom_coulomb_matrix` previously used `pair_mask` only to replace the
invalid displacement with machine epsilon. Its dipole-dipole isotropic term
still used the unmasked reciprocal distance, so padded pairs could produce a
nonzero matrix entry. With nonzero dummy multipoles this could enter a
contraction. The kernel now substitutes a unit displacement before computing
the norm and reciprocal powers, then zeros invalid monopole-dipole,
dipole-dipole, and monopole-quadrupole entries. Physical pair formulas are
unchanged.

`evaluate_padded_fixed_trip` and `_prepare_padded_interactions` now replace
inactive coordinates with zero before geometry-dependent preparation. This
prevents arbitrary caller-provided padding coordinates from reaching CN,
AES2, D4SC, initial-guess, and classical calculations; the pre-existing
integral boundary already performed this sanitization.

The AES2 regression uses nonzero inactive pair contractions, displaced
inactive coordinates, and checks finite physical coordinate derivatives.
It passes in every tested CPU environment.

## Numerical and derivative evidence

The following focused first-order acceptance selection completed with
`120 passed` in each environment:

- Python 3.10.22 / PyTorch 2.6.0.
- Python 3.11 / PyTorch 2.8.0.
- Python 3.12 / PyTorch 2.11.0.
- Python 3.12 / PyTorch 2.13.0.
- Python 3.14.8 / PyTorch 2.14.1.

It includes the new E7a coincident/displaced padding and AES2 tests, existing
GFN1/GFN2 padded spectral and full SCF tests, GFN2 canonical/integral tests,
negative-index scatter reductions, and the historical
`test_padding_ghosts.py` regression. Existing GFN2 position/HVP and active
parameter-gradient comparisons passed in this selection. The 3.14 profile also
ran the dedicated AES2 regression successfully; package import through dxtb
works in the prescribed profile.

The full-energy third directional derivative test gave the following
independent serial versus mapped discrepancies on PyTorch 2.6.0:

| Method | Serial | Mapped | Absolute error | Relative error |
|---|---:|---:|---:|---:|
| GFN1 water lane | -1.3799981115381825 | -1.3629367193659156 | 1.7061392173e-2 | 1.2363e-2 |
| GFN2 lane | 4.6024227236975856e-2 | 4.59887134389816e-2 | 3.5513797994e-5 | 7.7163e-4 |

PyTorch 2.14.1 reproduced both mismatches: GFN1 absolute error
`1.7061392172e-2`; GFN2 absolute error `3.7892206204e-5`. The test uses
float64, positive one-trip linear fixed-trip SCF, independent source Systems,
and the same normalized physical coordinate direction. It does not compare
eigenvectors. A homogeneous H2/H2 control agrees to machine precision, while
a homogeneous water control reproduces the GFN1 difference, so this result
is not specific to inactive atom capacity. Component inspection localized the
GFN1 difference to the SCF energy; D3 and Repulsion third directional
derivatives matched the serial values to displayed precision. The exact first
divergent SCF operation has not yet been isolated.

Both GFN1 and GFN2 parity failures were reproduced by running the same
third-order test against the required starting SHA in a temporary detached
worktree with the same Python 3.10.22 / PyTorch 2.6.0 dependencies. Both
failed at the second source lane (`lane=1`). This is an inherited
mapped/serial higher-order discrepancy, not introduced by the AES2 or
inactive-coordinate safety fixes. It still blocks the binding E7a acceptance
criterion because the behavior remains unsupported at the required boundary.

The test is retained as a failing acceptance regression. No XFAIL, tolerance
change, or reference rewrite was used. The test executes third-order
cross-System and inactive-coordinate controls before reporting the parity
mismatches. This unresolved mapped/serial third-order disagreement blocks
E7a completion.

Independent review requested Luna / high and found no remaining major issue
in the AES2 correction or updated test ordering. Actual reviewer model
identity is not exposed by the runtime. The inherited third-order parity
blocker remains open.

## Test manifest and environment notes

See `E7a-acceptance-manifest.txt` for the exact new nodes and the reproducible
focused command. Standard runs use `OMP_NUM_THREADS=1`,
`-n logical --dist worksteal`, the five repository CPU environments, official
`tad-dftd3==0.7.0`, and `tad-mctc==0.9.1`.

The broader focused primary-profile selection completed with `264 passed` on
Python 3.14.8 / PyTorch 2.14.1. It added E6bimpl.0, E6bimpl.3, E6bcompat,
C8b registered-core/padding, and Model parameter-tree controls. The new AES2
invalid-pair test, including reverse gradient, JVP, and jacfwd, passed
individually in all five environments.

Exact installed versions were PyTorch `2.6.0+cpu`, `2.8.0+cpu`,
`2.11.0+cpu`, `2.13.0+cpu`, and `2.14.1+cpu`; every profile had
`tad-mctc==0.9.1`, `tad-dftd3==0.7.0`, and `tad-dftd4==0.8.0`. Black, isort,
compileall, and `git diff --check` passed. Ruff and mypy are not installed in
the primary environment. CUDA was not run under the repository's CPU-only
work policy.

The first-order command passed 120 tests in each profile. The full E7a module
additionally runs two third-order nodes; both fail as described above on
PyTorch 2.6.0 and 2.14.1. Higher-order tests in the other three profiles have
not been run. The six inherited functional-SCF failures tracked in the
E6bimpl reports were not part of this focused selection. CUDA was not run.

## Completion decisions

- **E7a batch-critical gate: OPEN.** First-order padding safety passed, but the
  required third-order mapped/serial parity does not.
- **Broader E7: OPEN.** This work audits the current mapped GFN1/GFN2 route
  only.
- **E6broute: BLOCKED.** E6c1-W2/W3 and the E6broute-owned public packed
  Result adapter remain outstanding, in addition to this E7a blocker.

Next work is to isolate the first GFN1/GFN2 SCF operation responsible for the
third-order discrepancy, correct it without changing physical references, and
rerun the complete E7a derivative selection.
