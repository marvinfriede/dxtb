# Test-suite memory and run time

Status: complete on `claude/tender-dijkstra-cda42s` (commits `9dd5cb37`,
`8c840c67` and the Fermi search commit after them).

## Memory

With `-n logical`, every xdist worker's resident memory ratcheted up from
0.35 GB to about 3.4 GB over the run. The cause was glibc keeping freed
heap memory, not live tensors. The whole-suite run was OOM-killed under
`MemoryMax=20G` at about 40 %. An autouse fixture in `test/conftest.py`
now runs `gc.collect()` and `malloc_trim(0)` after every test. Workers stay
near 0.45 GB, and the run time is unchanged (415 s vs 418 s on the
measured directory). No single test peaks above 4.7 GB.

## Routine selection of the two largest grids

- `test_scf/test_implicit_matrix.py`: 7 representative cells per
  differentiation mode stay routine (H2O for every derivative kind, plus
  the batched H2O/CH4 gradient and HVP). All other cells are `slow`.
  Routine cells drop from 47 to 17.
- `test_scf/test_resumable_fixed_trip.py`: every configuration at
  `maxiter <= 2`, plus the 12-trip cases of one physics and one
  configuration, stay routine (100 cases); the other 50 are `slow`.
- Both files: routine selection 142 passed in 73 s; slow selection 126
  passed, 1 xfailed, 1 xpassed (existing markers).

## Fermi search stops when done (eager CPU only)

The SCF map fills with `_get_fermi_occupation_core_state`, whose search
always ran `fermi_maxiter + 1` (201) trips. That was about 65 % of each
single point of the Fermi-smeared LiH property tests. In eager CPU
evaluation, `_fermi_energy_search_until_done` now reads one convergence
flag per trip and stops once every channel is done. Done channels are
frozen by `_fermi_search_step`, so the result is bitwise that of all
trips. `test_functional_filling.py` checks this with `torch.equal` on the
occupations, status and search state, including unconverged searches.
Trip counters check that `vmap`, `jacrev` and `torch.compile` (`aot_eager`)
keep all trips. The compile test fails if the compile guard is removed (5
instead of 13 trips).

`_fermi_search_stops_early` keeps all trips, without host reads, inside
`torch.func` transforms, `torch.compile`, `torch.jit.trace`, dispatch
modes (fake tensors, `make_fx`) and on non-CPU devices (CUDA cost not
measured). `_fermi_energy_search_fixed` stays free of host reads, and the
source contract in `test_transform_critical_fermi_sources_have_no_host_control`
is unchanged.

The host read sits in the transform-critical SCF map, which AGENTS.md
otherwise rules out. It is a deliberate, guarded exception approved by
the architecture owner (2026-11-09). It follows the same pattern as the
chunked fixed-trip controller of mapped groups.

Same 7 property tests, same conditions (`-n logical`):

| Test | Before | After |
|---|---|---|
| `test_hyperpol::test_single[LiH]` | 146 s | 86 s |
| `test_hyperpol::test_single_pytorch_driver` | 129 s | 68 s |
| `test_hessian::test_single[LiH]` | 123 s | 82 s |
| `test_vibration::test_single[LiH]` | 106 s | 66 s |
| `test_dipole_deriv::test_single[LiH]` | 60 s | 40 s |
| `test_pol::test_single[LiH]` | 40 s | 31 s |
| wall time of all 7 | 153 s | 92 s |

## Validation

- Broad CPU suite, single run in one 20G scope (`.venv-grouped-cpu`,
  Python 3.12, torch 2.14.1): 5456 passed, 92 failed, 4 errors in 16:04,
  down from 20:41 after the memory fix alone. All 96 failing ids also fail
  with the Fermi change reverted; they are the known pre-existing set.
- `test_wavefunction`, `test_scf/test_map.py`,
  `test_finite_temperature_newton.py`, `test_invariant_free_energy.py` and
  `test_properties` also pass in `.venv` (Python 3.12, torch 2.14.1+cu130,
  CPU) and `.venv-torch27` (Python 3.10, torch 2.7.1).
- No tolerance changed, no reference regenerated. Independent review not
  run (skipped on this workstation).
