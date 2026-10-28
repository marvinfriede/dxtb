# U1–U4 merge into `claude/tender`

## Repository and ancestry

- Local target before integration: `f8297423a363305d0233307bc92d1c6c947b63a8` on `claude/tender-dijkstra-cda42s`.
- Accepted dependency source: `68d5609fad4515deb50222d7c17224bd74f333f5` on `integration/u1-u4-current`.
- The accepted source descends directly from the local target; the source/target merge base is `f8297423a363305d0233307bc92d1c6c947b63a8`. Thus all 21 local refactor commits are retained.
- Before validation, the remote target had advanced by `4546417c4a1174ce9322a0c18014eb552c678f7a` from shared base `fbf4531dc11b284a8e25829dd661e4eedb7ce4c1`. That implicit-SCF update is included as the second parent of the merge.
- Validated merge commit: `eb3fab5a9444d85c8f3a1eaaf722d2049e8bebd7`.

The merge had one textual conflict: `src/dxtb/_src/scf/functional/spectral.py`. The resolution retained the incoming implicit-root and trace free-energy paths, the target's explicit AO-mask precedence and energy-sorted active-prefix handling, and its Cholesky projector safeguard. The Fermi matrix function uses the incoming temperature-adaptive scaling schedule with `tad_mctc.storch.linsolve.dense_solve` for its matrix solves. This combination was selected after the uniform-scale/Cholesky combination failed the incoming small-gap trace test, and adaptive Cholesky failed the wide-gap quadrature case.

Independent review also found that `constraints/baseline.txt` still pinned the previous core dependency versions while CI installs the package against those constraints. The constraints now match `setup.cfg`: mctc 0.11.3, multicharge 0.8.0 and libcint 0.4.0. The baseline workflow continues to install D3 0.7.0 and D4 0.8.0 with `--no-deps`, as accepted for the development profile.

## Dependency and route status

The validated environments imported dxtb and the dispersion packages with:

| Python | PyTorch | mctc | multicharge | libcint | D3 | D4 |
|---|---:|---:|---:|---:|---:|---:|
| 3.11.15 | 2.8.0+cpu | 0.11.3 | 0.8.0 | 0.4.0 | 0.7.0 | 0.8.0 |
| 3.14.8 | 2.14.1+cpu | 0.11.3 | 0.8.0 | 0.4.0 | 0.7.0 | 0.8.0 |

Public `Calculator` routing is unchanged. D3/D4 metadata conflicts remain the documented development-only `--no-deps` exception; public release readiness remains blocked. The accepted U1–U4 report retains the full dependency, libcint and IES implementation history: [`U1-U4-upstream-consolidation.md`](U1-U4-upstream-consolidation.md).

## Focused validation

Commands ran in isolated repository worktrees with `PYTHONPATH` pointing to the merge candidate, `OMP_NUM_THREADS=1`, and at most two xdist workers.

| Selection | Python / PyTorch | Result |
|---|---|---:|
| E1.3/E6a D3, conformer mapping, Result/status, transforms (the documented 11-file selection) | 3.11 / 2.8.0 | 146 passed |
| Same E1.3/E6a selection | 3.14 / 2.14.1 | 146 passed |
| `test_scf/test_functional_implicit_root.py` + `test_scf/test_invariant_free_energy.py` | 3.11 / 2.8.0 | 67 passed |
| Same implicit/free-energy selection | 3.14 / 2.14.1 | 67 passed |
| E7a third-derivative and positive-definite controls (`-k 'third_directional or third_derivative or positive_definite'`) | 3.14 / 2.14.1 | 9 passed |
| D3/D4, IES and representative libcint-vs-PyTorch singlepoints (`-m 'not large'`) | 3.14 / 2.14.1 | 82 passed, 2 skipped |
| E6bimpl padded integral/spectral, GFN2 canonical/SCF and retained libcint composition files | 3.14 / 2.14.1 | 143 passed |

The D3/E6a selection retains the trainable `s9=0` derivative, forward and reverse transforms, nested transforms, and public GFN1 conformer `vmap`. The 67-test SCF selection includes the small-gap and wide-gap free-energy trace third derivatives. The E7a selection includes the mixed GFN1/GFN2 full-energy third-derivative controls. The E6bimpl selection exercises GFN1/GFN2 padding, result decoding, D3/D4, and retained libcint composition.

The accepted source branch had separately completed the 316-test libcint selection on Python 3.10/PyTorch 2.7 and Python 3.14/PyTorch 2.14.1. That full selection was not repeated because the merge had no libcint conflict; the merged run above includes actual libcint numerical comparisons and the E6b retained libcint composition tests.

Static checks passed: Black, isort, `compileall`, and `git diff --check`. Ruff and mypy were not run. CUDA was not run; CPU-only behavior is the accepted scope.

## Review and decision

Independent review initially identified the stale baseline constraints; they were corrected and re-reviewed. The re-review found the constraints consistent with `setup.cfg` and the controlled D3/D4 installation. No remaining blocking source findings were reported. The spectral conflict resolution passed the focused SCF, E7a and E6bimpl tests listed above.

**U1–U4 merge into `claude/tender`: ACCEPTED.**

**Upgraded dependency stack: VERIFIED** on both primary environments.

**E1.3/E6a derivative contract: PRESERVED.**

**Functional SCF and E6bimpl: PRESERVED** at the tested boundaries.

**Public Calculator routing: UNCHANGED.**

**Public release readiness: BLOCKED** by the separate D3/D4 metadata and release gates.

The next package is E6bcompat, using the merged target branch as its baseline. Do not activate E6broute before E6bcompat, batch-critical E7 and E6c1-W2/W3 acceptance.
