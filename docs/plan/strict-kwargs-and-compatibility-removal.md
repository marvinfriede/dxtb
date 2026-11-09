# Strict keyword arguments and removal of the remaining compatibility paths

Status: complete on `claude/tender-dijkstra-cda42s`. These are the
follow-ups to [`glossary-alignment-and-legacy-removal.md`](glossary-alignment-and-legacy-removal.md).
The architecture owner decided (2026-11-09) to make Calculator keyword
arguments strict and to remove the three remaining compatibility paths.
All changes are hard breaks without aliases.

## Packages

| Commit | Package |
|---|---|
| `317713f7` | **Strict kwargs.** `Calculator()` takes only `auto_int_level`, `timer` and `batch_mode` besides its named arguments. `singlepoint` (1D and 2D) and `evaluate_packed_systems` take only `write_overlap`, `write_dipole`, `write_quadrupole`, `write_hcore` and `cuda_sync_in_scf`; property methods consume their named arguments and forward the rest, so every unknown keyword raises `TypeError`. `energy` takes `keepdim` by name. `calculate`/`get_property` give each property method only the keyword arguments it takes, and raise if no requested property takes one. Removed: `reject_removed_store_kwargs`, the `cache` message and `ignore_unknown_kwargs`. |
| `1a328bc2` | **No Calculator field defaults.** `Calculator(interaction=[ElectricField(f)])` is rejected by `Model.setup`; fields are `field=`/`field_grad=` call inputs only. Omitting them means no field, and the integral level is no longer raised implicitly. `_field_default`, the `_UNSET` sentinel of the Calculator methods and the dead custom field-label branches are removed. The CLI, the W5 benchmark and the module example pass `field=` per call. Six tests of the default mechanism are deleted. |
| `522b68f9` | **No padded System for a batched Calculator.** `calc.system`, `calc.classicals` and `calc.interactions` are `None` for a 2D Calculator. Its options come from `Model.resolve_config`, and it builds its padded `IndexHelper` directly, exactly as `Model.setup` would. Construction sets up each distinct Lane composition once for validation. The label rename/restore around per-call setups (`_sync_lane_system_labels`) is removed. Tests that need component-level access to a padded batch set up one with `calc.model.setup`. |
| `eb3329fa` | **No stateful integral layer.** `DriverManager`, the stateful drivers, the integral classes with `build()` state, their factories, the `Integrals` container and `dxtb.integrals.factories/types` are deleted. `dxtb.integrals.wrappers` (`overlap`, `dipint`, `quadint`, `hcore`) is rebuilt on the stateless core builders, with one convention for both drivers: multipoles about the origin, all nine quadrupole components, normalized with the overlap norm. With libcint, `dipint`/`quadint` used to be ket-centred (`quadint` with 6 packed components), contrary to their docstring. Tests of the stateful API are deleted; value tests use the core builders, bitwise equal. |

## Decisions recorded

- `calculate([...], **kwargs)`: a keyword argument must be taken by at
  least one requested property. Methods that forward `**kwargs` take the
  energy keyword arguments (`field`, `field_grad`, `keepdim`, single-point
  options). For example, `calculate(["energy", "hessian"],
  use_functorch=False)` works, but `calculate(["energy"],
  use_functorch=False)` raises.
- Integral wrappers: keep them stateless, using the documented convention
  for both drivers (owner's choice over keeping the libcint output, and
  over deleting `dxtb.integrals`).

## Validation

Local workstation, CPU only, `.venv-grouped-cpu` (Python 3.12, torch
2.14.1). A peer session was editing `src/tad_dftd` in the main checkout
at the same time, so every package was validated in a clean git worktree
of its own commit. With `PYTHONPATH` pointing at the worktree's `src`,
pytest imports the worktree's `dxtb` and `tad_dftd`.

Broad suite (`-n logical --dist worksteal -m "not large and not slow"
--random-order-bucket=global`, one 20G scope):

| After | Result | Failing ids |
|---|---|---|
| `317713f7` | 92 failed, 4 errors, 5471 passed | the known 96 |
| `1a328bc2` | 92 failed, 4 errors, 5466 passed | the known 96 |
| `522b68f9` | 120 failed before the test re-pointing below; the 28 new ids pass after it | the known 96 + 28 tests that read the padded System (fixed in the same commit) |
| `eb3329fa` | 90 failed, 4 errors, 5383 passed | the known 96 minus `test_integrals/test_driver/test_pytorch.py::test_batch_mode_fail[dtype0,dtype1]` (file deleted: it tested the stateful driver) |

The known 96 are the pre-existing failures recorded in
[`test-suite-cost.md`](test-suite-cost.md) (libcint precision,
`test_basis` export, `test_baseline` cells, missing `tblite`, two
batch-lifecycle tests, `refocc` NO2 and a few SCF cases).

No tolerance was changed and no golden reference was regenerated.
Independent review was not run (skipped on this workstation). The
`/tmp1` environments are absent here.

## Remaining

- The `get_cache`/`get_energy` component API that the built-in
  Components use to build their setups.
- `xtb/BaseHamiltonian` and `GFN*Hamiltonian` are now used only by tests.
- Three integral tests compare against a stale p-shell permutation
  (`PERM_BY_L`): `test_multipole_matrix::test_matches_libcint`,
  `test_pytorch_gfn2::test_integral_wrappers` and
  `test_driver_precision::test_overlap_matches_libcint`. They fail before
  and after this change, and fixing them means changing the tests.
