# Glossary alignment and legacy compatibility removal

Status: complete on `claude/tender-dijkstra-cda42s` (commits `46d2c17b`
to `da0255d5`). Vocabulary: [`.agent-docs/GLOSSARY.md`](../../.agent-docs/GLOSSARY.md).

## Decisions (architecture owner)

- Public renames are a hard break, without aliases.
- Removed outright: the batched-Calculator legacy metadata, the old
  `ConfigSCF.create` arguments, shims that only re-reject already removed
  arguments, the custom-Component extension path and D4's setup-time `q=`.
- `dxtb._src.scf.functional` is flattened into `dxtb._src.scf`.

## Packages

| Commit | Package |
|---|---|
| `46d2c17b` | Docstrings use Structure/System, Padded/Conformer batch, Padding, Parametrization/Method, "Fermi smearing at nonzero electronic temperature". |
| `4e99136b` | `legacy_compat` removed: batched `Calculator` Results carry per-Lane `iterations`/`converged`/`residual`/`status`. Scalar charges are broadcast, charge/spin dtypes are checked, and `ignore_unknown_kwargs` works on its own. The Calculator error policy for mapped Lane groups is `eager_status_policy`. |
| `5169be54` | `mixer`, `scp_mode`, `scf_mode`, `damp_init/_generations/_soft_start/_diagonal_offset` and `f_atol` are no longer accepted. The `SCF_MODE_*`, `SCP_MODE_*` and `MIXER_*` labels are deleted, and the core uses `SCFRepresentation`/`SCFDifferentiation` directly. Also deleted: the SCF engine override shims, the cache-option messages, D3 `chunk_size`, `INTDRIVER_AUTOGRAD`, `Result.total`/`.iter` and the solver aliases. Callers use the `scf_strategy` test helper. |
| `bb3f59b4` | `Model.setup` accepts only the built-in Classical (D3, D4, Halogen, IES, Repulsion, SRB) and Interaction (AES2, ES2, ES3, D4SC, GB) types. The extension cache becomes `_packed_classical_setups`, which holds only the D3 setups of padded/conformer Systems. `compatibility_q` is removed. The "Creating New Components" tutorial is deleted. |
| `a1e987de` | `FullSCFRoute` → `SCFPlan`, `resolve_full_scf_route` → `resolve_scf_plan`, `EagerFullResult` → `EagerSCFResult`, `LegacyLibcintSetup` → `LibcintDriverSetup`. "Legacy", "compatibility" and "route" wording on live code is reworded. |
| `eb0f69e1` | `SCFRepresentation` → `SCFVariable` (`representation` → `variable` throughout). `*Method` → `*Converger`, with `SCFStage.method` → `SCFStage.converger`. `SCFResponseSettings` → `CPSCFSettings`, with `ConfigSCF.response` → `cpscf`. |
| `62f912e9` | `scf/functional/*` moves to `scf/`; `method.py` becomes `converger.py` and `response.py` becomes `cpscf.py`. |

## Validation

Local workstation, CPU only, `.venv-grouped-cpu` (Python 3.12, torch
2.14.1). The five `/tmp1` environments do not exist on this machine.

- `pytest -n logical --dist worksteal -m "not large and not slow"
  --random-order-bucket=global` runs one test directory at a time under
  `MemoryMax=20G`. The single whole-suite run was OOM-killed at 40 % on
  this 31 GB machine.
- The baseline at `91c10d36` had 96 failing or erroring node ids, all
  pre-existing: libcint multipole/overlap precision, `test_basis` export,
  `test_baseline` cells, the missing `tblite` module, two
  batch-lifecycle tests, `refocc` NO2 and a few SCF cases.
- Every package was checked against that baseline. Full per-directory
  runs were done after `5169be54`, `bb3f59b4`, `eb0f69e1` and
  `62f912e9`, and focused module runs after each fix. After the final
  fixes (`da0255d5`), `test_scf`, `test_calculator`, `test_classical`
  and `test_config` were rerun on HEAD with the standard flags:

  | Directory | Baseline (`91c10d36`) | HEAD |
  |---|---|---|
  | `test_scf` | 11 failed, 1476 passed | 11 failed, 1473 passed |
  | `test_calculator` | 2 failed, 484 passed | 2 failed, 476 passed |
  | `test_classical` | 575 passed | 572 passed |
  | `test_config` | 46 passed | 35 passed |

  The failing node ids are identical except for renamed parametrize ids
  (`refocc` `full` → `unrolled`). The lower pass counts are tests of
  removed behaviour that were deleted, or merged into one rejection test
  per file. All other directories matched the baseline in the
  `62f912e9` run.
- `dxtb.__main__` imports a non-existent `dxtb.cli` already at
  `91c10d36`. This is a pre-existing bug and is not touched here.
- No tolerance was changed and no golden reference was regenerated.
- Independent review was not run (skipped on this workstation).
  Multi-environment runs are pending.

## Kept on purpose

- Private solver parameters that receive a converger are still named
  `method`. Internal `response_*` names inside `cpscf.py` are unchanged.

## Follow-up

`reject_removed_store_kwargs`, the Calculator field defaults and the
stateful `DriverManager` layer were removed afterwards. The padded System
of a 2D Calculator is kept: it is the batch-agnostic view of the
Calculator, not a compatibility path. See
[`strict-kwargs-and-compatibility-removal.md`](strict-kwargs-and-compatibility-removal.md).
The `get_cache`/`get_energy` component API that the built-in Components
use to build their setups remains.
