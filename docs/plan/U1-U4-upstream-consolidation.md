# U1–U4 upstream dependency and libcint consolidation

## Status

This is a partial implementation record for the isolated integration branch
`integration/u1-u4-4546417`, based on `4546417c4a1174ce9320a0c18014eb552c678f7a`.
The local commits are:

| Package | Commit | Status |
|---|---|---|
| U1 metadata | `bdd102ab1eaf445b799ff2b2c6682ee27055bf69` | Implemented; acceptance blocked by official D3/D4 metadata conflict and inherited implicit Result status failures. |
| U2 libcint 0.4 | `ee116f84d479fa3bbbb83e00c5abcfb6fde893ab` | New API migration and focused Linux CPU numerical checks pass; full platform/API matrix incomplete. |
| U3 multicharge | `4713b28797379f8c3a5197829582cff436ee642c` | Default Schur behavior retained; optional solver modes characterized. |
| U4 consolidation | pending | Final report and branch-plan update only; no package acceptance claim. |

The base contains the requested C8b, E6a, and implicit-SCF history. The work
was performed in a separate worktree; the original checkout and its untracked
`AGENTS.md` were not modified. No public Calculator routing or upstream source
was changed.

## Dependency and installation findings

The intended package index supplied these exact distributions on the current
Linux x86-64 / CPython 3.12 platform:

| Distribution | Installed version | Wheel | SHA-256 |
|---|---:|---|---|
| `tad-mctc` | 0.11.3 | `py3-none-any` | `7e8593a0efbb28ce8578bd0974d7d781f620b8b75b0bde232f18b1dfe00d2e79` |
| `tad-multicharge` | 0.8.0 | `py3-none-any` | `0f7e494f4dbbffd235601e3c8767c029c4f4348055060af99b3323e1428bf634` |
| `tad-libcint` | 0.4.0 | `py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64` | `d273c4ae05a577fd87eb8790fcfdfe5d816b28b7963d274a7dfe65fe650de5bb` |

The installed metadata confirms the new upstream floor is PyTorch 2.7.0 and
Python `>=3.10,<3.15`. The dxtb metadata, docs pin, tox factors, and Ubuntu CI
matrix were updated accordingly. Python 3.10–3.14 were retained where the
configured release matrix has corresponding wheels. Only the Linux x86-64
libcint wheel was installed here; macOS, Windows, and other architecture wheel
availability was not tested.

The clean core installation (`dxtb[libcint]`, mctc 0.11.3, multicharge 0.8.0,
libcint 0.4.0, Torch 2.14.1+cpu) passed `pip check`. dxtb imports the D3 module
during package import, so omitting D3 is not a usable runtime installation.
Installing the documented official `tad-dftd3==0.6.0` and
`tad-dftd4==0.8.0` wheels with `--no-deps` made dxtb import and the tested
runtime paths work, but correctly made `pip check` fail:

- D3 0.6.0 requires `numpy<2` and `tad-mctc==0.7.0`.
- D4 0.8.0 requires `numpy<2`, `tad-mctc==0.7.0`, and
  `tad-multicharge==0.5.0`.
- The integration requires `tad-mctc==0.11.3` and
  `tad-multicharge==0.8.0`; the tested Torch 2.14 wheel installs NumPy 2.5.4.

This is a packaging metadata incompatibility, not evidence of a runtime API
failure. The selected D3/D4 controls passed with those exact official wheels,
but a clean complete environment cannot pass `pip check` without compatible
official D3/D4 metadata/releases or a separately approved dependency policy.
The release candidate is therefore **not package-clean**. No dependency was
patched and no D3/D4 tests were disabled. This does not resolve the separate
E1.3 requirement for released D3 forward-mode support.

The following imports resolved from the installed releases:

- `tad_mctc.storch.implicit.implicit_fixed_point`
- `tad_mctc.storch.linsolve.bicgstab`, `dense_solve`
- `tad_multicharge.model.eeq.EEQModel`
- `tad_libcint.Basis`, `Shell`, `int1e`, `overlap`
- `Structure` from `tad_mctc.io.structure`

The old `LibcintWrapper`, `AtomCGTOBasis`, `CGTOBasis`,
`tad_libcint.interface.wrapper`, and `tad_libcint.interface.integrals` names
are absent from production and test sources after migration. `Structure` is
not exported by `tad_libcint` in 0.4.0; it is imported from tad-mctc.

## U1 implicit SCF and free energy

The dependency pin set in this branch was tested with Python 3.12.10,
PyTorch 2.14.1+cpu, NumPy 2.5.4, pytest 8.4.2, and official D3 0.6.0 / D4
0.8.0 installed without dependencies for runtime checks. The focused command

```text
pytest -q test/test_scf/test_functional_implicit_root.py test/test_scf/test_invariant_free_energy.py
```

reported **65 passed, 2 failed**. The two failures are:

- `test_public_singlepoint_routes_eligible_implicit_mode`
- `test_public_functional_implicit_h2_float32_response`

Both fail because the dictionary-returning implicit SCF path is normalized to
`converged=None`, `residual=None`, and `status=None`, while these tests expect a
successful convergence flag. The normalization branch predates the required
base commit; the tests were already present at the base. They are an existing
SCF Result metadata gap, not a tad-mctc 0.11.3 Krylov regression. They remain
unresolved and prevent full U1 acceptance.

The rest of this selected implicit/free-energy suite passed, including the
matrix-free nonsymmetric solve reference, explicit failed-solve NaN derivative
signal, mixed-success/failure `vmap`, and the trace free-energy value/derivative
controls through the tested third order. A separate `torch.compile(...,
backend="aot_eager")` smoke test of `solve_linear` matched `torch.linalg.solve`.
This is a narrow compile control only; it is not a claim of compiled nested
higher-order transform support.

The trace implementation and dense/matrix-free solve were inspected as
separate changes. The selected invariant free-energy tests passed; no change to
the trace formula, tolerance, or reference was made.

## U2 libcint 0.4 API and numerical boundary

The adapter now constructs call-local `tad_libcint.Basis` and
`tad_mctc.Structure` values from captured dxtb shell parameters. `Shell`
entries are normalized once with the new API. Exponents and coefficients stay
connected to the model parameter graph. Integer shell/element metadata is
materialized only to build the external static Basis; geometry is supplied as
the current positions tensor in a fresh Structure on each call. The dxtb
`IndexHelper` and `CoreIndexData` remain intact.

For the new backend, dipoles use native `j`; quadrupoles use native packed
`jj` in `xx, yx, yy, zx, zy, zz` order. The libcint path does not apply the
PyTorch backend's origin shift and does not reduce an already packed tensor a
second time. PyTorch origin shifts and quadrupole reduction remain unchanged.

Focused actual-backend tests on the official Linux wheel reported:

- Basis/setup, matrix, shape, symmetry, map and geometry-history selection:
  158 passed, 48 skipped.
- New native `j`/`jj` parity against PyTorch for H2O and SiH4, including
  position derivative parity: included in the 158 passed set.
- GFN1 H2O and GFN2 H2/H2O complete fixed-trip singlepoints with PyTorch versus
  libcint: 3 passed; maximum observed total-energy discrepancy was
  `8.9e-16 Eh` in GFN1 H2O and `4.4e-16 Eh` / `1.1e-16 Eh` in GFN2 H2/H2O.
  Maximum force discrepancy was below `3.3e-16 Eh/bohr`.
- A selected basis coefficient gradient/gradgrad and libcint setup/multipole
  set: 77 passed, 132 skipped. PySCF-only reference cases were skipped because
  PySCF was unavailable.
- `test_libcint.py` single and batch integral manager tests exercised
  `force_cpu_for_libcint=True/False` on CPU.
- D3/D4 and 2-body reference selection under the required no-deps runtime
  installation: 33 passed.

GFN2 LiH full SCF did not provide a libcint comparison: the first PyTorch
single-System call failed in `torch.linalg.eigh` with error code 4 on the
starting implementation's GFN2 spectral path. This is not evidence of a
libcint-specific failure, but the LiH full-calculation row remains unaccepted.
The official new wheel and its tested Linux CPU ABI work; wheel availability on
other platforms, full packed Calculator singlepoints, all supported shell
angular momenta, and the complete mixed position/basis derivative matrix
remain unverified. Available GFN basis fixtures exercised s, p, and d shells;
an f-shell dxtb parameter fixture was not found or accepted.

The broad combined coefficient/overlap/multipole run was interrupted before a
pytest summary and is not counted. Its smaller completed replacements above
are the evidence used here.

## U3 multicharge and IES

IES continues to call `EEQModel.solve` without a `solve_mode`, preserving the
upstream default `schur`. No new mode is enabled in production.

The IES and Repulsion core selection, with the added optional-mode test,
reported **61 passed, 2 skipped**. This includes nested EEQ Node structure,
float conversion, parameter substitution, charge/position derivatives, and
mapped setup behavior on the configured GFN0/SiH4 IES fixture.

An additional nonzero-charge H2O setup probe compared `schur`, `cg`, and
`matrix-free`. Charge sums matched `+0.23 e`; charges, energy partitions and
position gradients agreed with Schur within `1e-10` for values and `1e-9` for
gradients. This characterization is committed as a test. No production switch
was made. Periodic upstream EEQ behavior was not tested and is not enabled in
dxtb.

## U4 status and remaining work

Formatting checks available in this environment: isort, compileall, and
`git diff --check` passed. Black, Ruff, and mypy were unavailable. The focused
checks were run only in Python 3.12 / Torch 2.14.1 on Linux CPU; the requested
five-environment matrix, Windows/macOS, CUDA, and a true no-libcint install
were not run. A controlled import-blocking check confirmed that dxtb and a
PyTorch Calculator import without libcint, but it is not a substitute for an
installed no-libcint environment.

Completed commands (run from the repository root with
`env -u PYTHONPATH /tmp/dxtb-u1-clean/bin/python -m pytest -q`):

```text
test/test_integrals/test_libcint_setup_node.py test/test_integrals/test_libcint.py
test/test_basis/test_setup.py test/test_libcint/test_shape.py
test/test_libcint/test_symmetry.py test/test_libcint/test_overlap.py
test/test_libcint/test_overlap_grad.py
  158 passed, 48 skipped
test/test_libcint/test_coeff_grad.py::test_gradcheck[gfn1-H2]
test/test_libcint/test_coeff_grad.py::test_gradgradcheck[gfn1-H2]
test/test_libcint/test_gradcheck.py::test_grad
test/test_integrals/test_libcint_setup_node.py test/test_libcint/test_multipole.py
  77 passed, 132 skipped
test/test_integrals/test_libcint_setup_node.py::test_libcint_full_singlepoint_matches_pytorch
  3 passed
test/test_classical/test_ies test/test_classical/test_repulsion/test_core_node.py
  61 passed, 2 skipped
test/test_classical/test_dispersion/test_d3.py
test/test_classical/test_dispersion/test_d4.py
test/test_classical/test_dispersion/test_d4_2b.py
  33 passed
test/test_scf/test_functional_implicit_root.py
test/test_scf/test_invariant_free_energy.py
  65 passed, 2 failed
```

The broad libcint test command that also included all coefficient, gradcheck,
and multipole parameterizations was interrupted and is intentionally not
counted as a passing run. The four passing groups above completed with normal
pytest summaries.

No physical reference values, tolerances, XFAIL markers, or upstream sources
were changed. Public routing was not touched. E1.3 remains separately open.

**U1–U4 is not accepted.** The finite blockers are:

1. Official D3/D4 metadata conflicts with the new mctc/multicharge baseline;
   the complete documented environment cannot pass `pip check`.
2. Implicit singlepoint Result convergence/status fields remain `None` in the
   dictionary output path and fail two existing controls.
3. The five-environment matrix and remaining packed/full libcint, platform
   wheel, and broader derivative requirements were not completed.

The next action is to resolve the official dependency metadata/runtime policy

## Development-integration follow-up

This follow-up uses the task's accepted policy: published D3/D4 metadata
conflicts may remain in controlled development installs. That does not make
the environment resolver-clean for public release. The current candidate is
based on target branch `f8297423a363305d0233307bc92d1c6c947b63a8`; the four
earlier U1–U4 commits remain intact above that base.

The specified upstream runtime set was installed in five isolated CPU
profiles:

| Python / Torch | NumPy | mctc | multicharge | libcint | D3 / D4 | pytest |
|---|---:|---:|---:|---:|---:|---:|
| 3.10.22 / 2.7.0+cpu | 2.2.6 | 0.11.3 | 0.8.0 | 0.4.0 | 0.6.0 / 0.8.0 | 8.4.2 |
| 3.11.15 / 2.8.0+cpu | 1.26.4 | 0.11.3 | 0.8.0 | 0.4.0 | 0.6.0 / 0.8.0 | 9.1.1 |
| 3.12.10 / 2.11.0+cpu | 2.5.4 | 0.11.3 | 0.8.0 | 0.4.0 | 0.6.0 / 0.8.0 | 9.1.1 |
| 3.12.10 / 2.13.0+cpu | 2.5.4 | 0.11.3 | 0.8.0 | 0.4.0 | 0.6.0 / 0.8.0 | 9.1.1 |
| 3.14.8 / 2.14.1+cpu | 2.5.3 | 0.11.3 | 0.8.0 | 0.4.0 | 0.6.0 / 0.8.0 | 9.1.1 |

The clean Python 3.10 environment uses the tox-supported minimum profile
(Python 3.10 / Torch 2.7), not the old Torch 2.6 profile that is below the new
`torch>=2.7` contract. Its first clean install exposed that the local D3 memory
shim imports `psutil`, which was not a declared runtime dependency. `psutil`
is now explicit in `install_requires` and docs requirements. A pytest 9.2
trial also failed the existing scalar-Tensor `pytest.approx` comparisons under
Torch 2.7; pytest 8.4.2 passed. Development extras now pin `pytest<9` to keep
the supported current test contract reproducible.

The controlled development install is:

```text
python -m venv .venv
. .venv/bin/activate
python -m pip install 'torch==2.7.0'  # select CPU/CUDA wheel for the profile
python -m pip install -e '.[dev,libcint]'
python -m pip install --no-deps --index-url https://pypi.org/simple \
    tad-dftd3==0.6.0 tad-dftd4==0.8.0
python -c 'import dxtb; import tad_dftd3, tad_dftd4; print(dxtb.__version__)'
python -m pip check
```

Import `dxtb` before D3/D4 so the compatibility shim runs. Importing the new
`tad-mctc`, `tad-multicharge`, or `tad-libcint` modules first is supported.
Direct D3 import before dxtb bypasses the shim and is not supported. The
remaining shim supplies removed compatibility names (`memory_device`,
`is_batched`, old safe-operation aliases, legacy typing names, and legacy
coordination-number callables). The coordination-number implementation
remains local to dxtb; no upstream files or scientific routines were changed.

`pip check` in all five profiles reported only the accepted D3/D4 metadata
conflicts: D3 0.6.0 requires NumPy `<2` and mctc `==0.7.0`; D4 0.8.0 requires
NumPy `<2`, mctc `==0.7.0`, and multicharge `==0.5.0`. Python 3.11 uses NumPy
1.26.4 and therefore reported only mctc/multicharge conflicts. The other
profiles also reported the declared D3/D4 NumPy conflicts. No unrelated
`pip check` issue remained. For Python 3.14 / Torch 2.14.1, the exact output
was:

```text
tad-dftd3 0.6.0 has requirement numpy<2, but you have numpy 2.5.3.
tad-dftd3 0.6.0 has requirement tad-mctc==0.7.0, but you have tad-mctc 0.11.3.
tad-dftd4 0.8.0 has requirement numpy<2, but you have numpy 2.5.3.
tad-dftd4 0.8.0 has requirement tad-mctc==0.7.0, but you have tad-mctc 0.11.3.
tad-dftd4 0.8.0 has requirement tad-multicharge==0.5.0, but you have tad-multicharge 0.8.0.
```

### Completed focused validation

The focused command below completed in all five profiles with **82 passed,
2 skipped** per environment:

```text
pytest -q -n logical --dist worksteal -m 'not large' \
  test/test_classical/test_dispersion/test_d3.py \
  test/test_classical/test_dispersion/test_d4.py \
  test/test_classical/test_dispersion/test_d4_2b.py \
  test/test_classical/test_ies \
  test/test_integrals/test_libcint_setup_node.py::test_libcint_full_singlepoint_matches_pytorch
```

This covers D3/D4 execution, IES Schur and optional-mode parity, and
representative GFN1 H2O / GFN2 H2 and H2O libcint-versus-PyTorch energy/force
controls. The Python 3.10 / Torch 2.7 run followed a fresh editable install and
installation of the specified official D3/D4 wheels with `--no-deps`.

The broader libcint selection completed with **316 passed** under Python 3.10
/ Torch 2.7 and Python 3.14 / Torch 2.14.1:

```text
pytest -q -n logical --dist worksteal -m 'not large' \
  test/test_integrals/test_libcint_setup_node.py \
  test/test_integrals/test_libcint.py \
  test/test_integrals/test_pyscf_reference.py \
  test/test_libcint/test_overlap.py \
  test/test_libcint/test_overlap_grad.py \
  test/test_libcint/test_multipole.py \
  test/test_libcint/test_coeff_grad.py \
  test/test_libcint/test_gradcheck.py
```

The PySCF reference helper now reorders only the AO axes from its
`px,py,pz` convention to the dxtb/tad-libcint 0.4 `py,pz,px` convention; it
does not reorder Cartesian derivative axes. Native ket-centered `j` and
already-packed `jj` are checked directly at the integral level. The libcint
container uses `r0` and `r0r0` only when origin-centered moments are requested,
avoiding a redundant shift or six-component reduction. Existing full energy
and force parity controls remain active. The D3 0.6 reference test uses its
tensor cutoff API; the production D3 calculation and cutoff were not changed.

The two former implicit-SCF failures were invalid test expectations, not an
integration regression. `System.singlepoint()` keeps the legacy implicit
engine; its dictionary result intentionally reports `None` for `converged`,
`residual`, and `status` under the B5.6 rule that legacy routes may have
`status=None`. The tests expected fixed-trip FULL-mode status on that legacy
route. They now assert the three `None` fields and still run their energy/field
response and mixed finite-difference checks. Both corrected tests pass in
isolation. No production SCF or status code changed. Reproducing from the
original `4546417` archive was blocked at import because its source requests
`tad_mctc.autograd.attach_gradient`, absent from the available mctc 0.9.1
environment; source and policy inspection confirm the result normalization
and B5.6 rule are unchanged by this integration.

### Remaining validation boundary

The primary `not large and not slow and not baseline` suite was attempted in
Python 3.14 / Torch 2.14.1 but did not complete: the distributed run reached
about 99% with numerous failures and stalled before a pytest summary, so it is
not counted as passing. A single-worker `-x` diagnostic stopped at
`test/test_a_memory_leak/test_scf.py::test_xitorch[False-False-dtype0]` after
20 passed and 2 skipped. The failure reports two retained 512-byte tensors
across iterations. Comparing this test against the pre-integration target
source was blocked because the old source imports removed
`tad_libcint 0.3` names (`AtomCGTOBasis`/`CGTOBasis`) while the candidate
environment contains 0.4. This memory-leak failure remains unclassified
against an executable pre-integration environment. The broad selection must
be rerun in disjoint groups with supported baseline dependencies before
claiming broad regression clearance.

Black, isort, compileall, and `git diff --check` pass for the follow-up
changes. CUDA and Windows/macOS were unavailable. Resolver-clean public
installation remains blocked until compatible official D3/D4 metadata is
published. The integration is **focused-development validated, but overall
U1–U4 acceptance remains open pending classification and completion of the
broader regression matrix**. The branch has not been pushed.
