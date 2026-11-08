# Glossary alignment and legacy compatibility removal

Status: complete on `claude/tender-dijkstra-cda42s` (commits `e02f70a5`
to `0ae13b72`). Vocabulary: [`.agent-docs/GLOSSARY.md`](../../.agent-docs/GLOSSARY.md).

## Decisions (architecture owner)

- Public renames are a hard break, without aliases.
- Removed outright: the batched-Calculator legacy metadata, the old
  `ConfigSCF.create` arguments, shims that only re-reject already removed
  arguments, the custom-Component extension path and D4's setup-time `q=`.
- `dxtb._src.scf.functional` is flattened into `dxtb._src.scf`.

## Packages

| Commit | Package |
|---|---|
| `e02f70a5` | Docstrings use Structure/System, Padded/Conformer batch, Padding, Parametrization/Method, "Fermi smearing at nonzero electronic temperature". |
| `4afadd66` | `legacy_compat` removed: batched `Calculator` Results carry per-Lane `iterations`/`converged`/`residual`/`status`. Scalar charges are broadcast, charge/spin dtypes are checked, and `ignore_unknown_kwargs` works on its own. The Calculator error policy for mapped Lane groups is `eager_status_policy`. |
| `4cdc1397` | `mixer`, `scp_mode`, `scf_mode`, `damp_init/_generations/_soft_start/_diagonal_offset` and `f_atol` are no longer accepted. The `SCF_MODE_*`, `SCP_MODE_*` and `MIXER_*` labels are deleted, and the core uses `SCFRepresentation`/`SCFDifferentiation` directly. Also deleted: the SCF engine override shims, the cache-option messages, D3 `chunk_size`, `INTDRIVER_AUTOGRAD`, `Result.total`/`.iter` and the solver aliases. Callers use the `scf_strategy` test helper. |
| `15816c28` | `Model.setup` accepts only the built-in Classical (D3, D4, Halogen, IES, Repulsion, SRB) and Interaction (AES2, ES2, ES3, D4SC, GB) types. The extension cache becomes `_packed_classical_setups`, which holds only the D3 setups of padded/conformer Systems. `compatibility_q` is removed. The "Creating New Components" tutorial is deleted. |
| `c8a90810` | `FullSCFRoute` → `SCFPlan`, `resolve_full_scf_route` → `resolve_scf_plan`, `EagerFullResult` → `EagerSCFResult`, `LegacyLibcintSetup` → `LibcintDriverSetup`. "Legacy", "compatibility" and "route" wording on live code is reworded. |
| `c53073af` | `SCFRepresentation` → `SCFVariable` (`representation` → `variable` throughout). `*Method` → `*Converger`, with `SCFStage.method` → `SCFStage.converger`. `SCFResponseSettings` → `CPSCFSettings`, with `ConfigSCF.response` → `cpscf`. |
| `0ae13b72` | `scf/functional/*` moves to `scf/`; `method.py` becomes `converger.py` and `response.py` becomes `cpscf.py`. |

## Validation

Local workstation, CPU only, `.venv-grouped-cpu` (Python 3.12, torch
2.14.1). The five `/tmp1` environments do not exist on this machine.

- `pytest -n logical --dist worksteal -m "not large and not slow"
  --random-order-bucket=global` runs one test directory at a time under
  `MemoryMax=20G`. The single whole-suite run was OOM-killed at 40 % on
  this 31 GB machine.
- The baseline at `0ed13b38` had 96 failing or erroring node ids, all
  pre-existing: libcint multipole/overlap precision, `test_basis` export,
  `test_baseline` cells, the missing `tblite` module, two
  batch-lifecycle tests, `refocc` NO2 and a few SCF cases.
- After `0ae13b72` there are no new failures against that baseline. The
  only differences are renamed parametrize ids (`full` → `unrolled`) of
  the same pre-existing failures, and tests that were deleted or
  converted together with the removed behaviour.
- No tolerance was changed and no golden reference was regenerated.
- Independent review was not run (skipped on this workstation).
  Multi-environment runs are pending.

## Kept on purpose

- `reject_removed_store_kwargs` (`store_*`, `grad_mode`) stays. Calculator
  methods ignore unknown keyword arguments, so without it those inputs
  would be accepted silently.
- Private solver parameters that receive a converger are still named
  `method`. Internal `response_*` names inside `cpscf.py` are unchanged.

## Remaining compatibility behaviour (not part of the decision)

- `ElectricField`/`ElectricFieldGrad` in `Calculator(interaction=...)`
  still turn into per-call field defaults.
- A 2D `Calculator` still builds a padded "compatibility" System for
  `calc.system`/`calc.ihelp` beside its per-Lane Systems.
- The stateful `DriverManager` path behind `dxtb.integrals`.
- The `get_cache`/`get_energy` component API that the built-in
  Components still use to build their setups.
