# SCF S0–S13.1 Redesign and General Functional-Refactoring Integration Report

## 1. Branch Provenance and Merge Ancestry

| Item | Branch / Ref | Commit SHA |
|---|---|---|
| Completed SCF Redesign (S0–S13.1 authority) | `scf/redesign` | `e78d3f6ac05a970e7871fa65a358b8f3d028cd9e` |
| General Functional Refactoring (3 newer commits) | `claude/tender-dijkstra-cda42s` | `e694117b87287e6557d1f4256a9f3e9175e3211d` |
| Merge Base | `git merge-base` | `b9b311a234868a1d339795eb0846fb117a5ce1f0` |
| Dedicated Integration Branch | `integration/scf-s13-general-final` | Merge of `e694117b87287e6557d1f4256a9f3e9175e3211d` into `e78d3f6ac05a970e7871fa65a358b8f3d028cd9e` |

Divergence at merge base `b9b311a234868a1d339795eb0846fb117a5ce1f0`:
- **SCF redesign (`scf/redesign`)**: 95 commits implementing S0–S13.1 (`SCFStrategy`, `SCFStage`, `SCFRepresentation`, `SCFDifferentiation`, `SCFConvergenceProfile`, `SCFResponseSettings`, native `LinearMethod`, `AndersonMethod`, `BroydenMethod`, `CDIISMethod`, `SOSCFMethod`, S2 physical convergence certification, S11 implicit response architecture, S12 transform compatibility, S13 defaults, and S13.1 parser/response-root consistency).
- **General refactoring (`claude/tender-dijkstra-cda42s`)**: 3 commits:
  1. `fb2bf0479aba3d2ed1d7870f9375454dc8fa716b` — `fix(calculator): finalize packed singlepoint routing gate, option separation, and Result compatibility`
  2. `a8fe9c5c82a0c4ae235fbf4c7bc71637c3fb48b9` — `test(properties): add primal vs derivative fixed-point convergence qualification`
  3. `e694117b87287e6557d1f4256a9f3e9175e3211d` — `docs(plan): add functional route final acceptance report and update roadmap`

No published commits on either branch were rebased or rewritten. `scf/redesign` remains untouched at `e78d3f6ac05a970e7871fa65a358b8f3d028cd9e`.

---

## 2. Source and Semantic Conflicts & Resolutions

### 2.1 Textual Merge Conflicts (6 files)

1. `src/dxtb/_src/scf/iterator.py`
   - **Conflict**: Commit `fb2bf04` forwarded `*args` into `solve_nonselfconsistent(...)` inside `_solve_impl`, whereas S13 (`b872709` / `5b622a5`) retired all low-level `*args` and `**kwargs` overrides at the top of `solve()` (`raise NotImplementedError("Legacy SCF engine positional/keyword overrides are no longer supported by functional SCF.")`) and removed `*args, **kwargs` from `_solve_impl`.
   - **Resolution**: Preserved the S13 `solve()` rejection of `*args, **kwargs` and kept `_solve_impl` free of legacy override forwarding.

2. `src/dxtb/_src/scf/nonselfconsistent.py`
   - **Conflict**: Commit `6f932cc` on `scf/redesign` migrated GFN0 off xitorch and removed `**kwargs` and `data.eigen_options`, while `fb2bf04` on the general branch added `*args, **kwargs` forwarding into `_Data(...)` to reject unknown low-level options.
   - **Resolution**: Kept the S13 xitorch-free generalized eigensolve and `_Data` construction, and accepted `*args: Any, **kwargs: Any` on `solve_nonselfconsistent` solely to raise the authoritative S13 `NotImplementedError("Legacy SCF engine positional/keyword overrides are no longer supported by functional SCF.")` if any direct caller passes legacy engine overrides.

3. `src/dxtb/_src/calculators/legacy_composition.py`
   - **Conflict**: Commit `fb2bf04` moved the `int_level < INTLEVEL_HCORE` / missing `h0_setup` check (Stage 4) ahead of the `legacy_options` check (Stage 5) in `classify_legacy_system`, while S13 updated `classify_legacy_system` to reject `legacy_options` with `NotImplementedError` and check `system.config.scf.differentiation`.
   - **Resolution**: Preserved the 6-stage failure precedence from `fb2bf04` (Stage 4 `int_level < INTLEVEL_HCORE` / missing `h0_setup` raises `NotImplementedError` before Stage 5 `legacy_options` raises `NotImplementedError`), preserved S13 `NotImplementedError` on `legacy_options`, and reconciled `classify_legacy_system` with the complete S13 strategy/method/representation/differentiation matrix (see Section 4 below).

4. `src/dxtb/_src/calculators/packed_result.py`
   - **Conflict**: Commit `fb2bf04` added `write_*` integral saving, `cuda_sync_in_scf` `try`/`finally` restoration, public vs low-level option separation, `build_calculator_lane_systems`, `evaluate_calculator_packed_singlepoint`, `atom_masks`, `target_atoms`, and `_normalize_system_for_e6bcompat` (which referenced retired `config.scf.scp_mode` and `config.scf.strict`). S13 removed `scp_mode` and `strict` from `ConfigSCF` and made `Anderson/Potential` the default strategy.
   - **Resolution**: Preserved all `packed_result.py` features from `fb2bf04` (`pack_system_results`, `evaluate_packed_systems`, `build_calculator_lane_systems`, `evaluate_calculator_packed_singlepoint`, `write_*`, `cuda_sync_in_scf` `try`/`finally`, `atom_masks`, `target_atoms`, duplicate classical slots) and updated `_normalize_system_for_e6bcompat` so it never reads retired `scp_mode`/`strict` attributes and never silently converts the S13 default (`Anderson/Potential`) or `Charge` representation into `Fock`.

5. `test/test_calculator/test_e6bcompat_differential.py`
   - **Conflict**: Differential tests in `fb2bf04` used retired `f_atol` and `strict` options, expected `TypeError` from `BaseSCF._Data.__init__` on `LegacySCFOptions`, and tested old non-strict GFN2 `scp_mode -> fock` reset warnings.
   - **Resolution**: Adapted test configurations to S13 (`f_atol` and `strict` removed, `LegacySCFOptions` verified to raise `NotImplementedError`, GFN2 tested across S13 default `Anderson/Potential`, `Charge`, and `Fock` on the functional packed route while verifying `NotImplementedError` on the legacy `Calculator.singlepoint` `batch_compat` route for non-Fock GFN2).

6. `docs/plan/07-current-state-and-execution-plan.md`
   - **Conflict**: Roadmap updates from S13 closure and `e694117`.
   - **Resolution**: Combined both status records and added the S0–S13.1 + General-Refactoring integration closure section.

### 2.2 Semantic Conflicts Resolved Outside Conflict Markers

1. **Retired `f_atol` in `test/test_calculator/test_b8_e8_functional_properties.py`**:
   - Commit `a8fe9c5` passed `"f_atol": x_atol` in `test_section5_primal_vs_derivative_fixed_point_convergence`. S13 explicitly retires `f_atol` at `Config.create`. Removed `"f_atol"` from `test_b8_e8_functional_properties.py` so the test uses S13 physical certification via `x_atol` and `x_atol_max`.
2. **Legacy `ConfigSCF(..., mixer=...)` in `test/test_scf/test_nonselfconsistent.py`**:
   - Updated direct dataclass instantiation with legacy `mixer` keyword to `ConfigSCF.create(..., mixer=...)`.
3. **E6bcompat 32-Row Manifest (`docs/plan/E6bcompat-final-routing-manifest.json`)**:
   - Reclassified all 32 rows into the 5 S13 categories (Section 3 below) and aligned all `test_nodes` with the integrated test suite.

---

## 3. Compatibility-Policy Reconciliation (`E6bcompat-final-routing-manifest.json`)

Each of the 32 rows in `docs/plan/E6bcompat-final-routing-manifest.json` has been classified into one of the five authoritative categories:

| Category | Rows | Summary |
|---|---|---|
| **1. Preserved public contract** | `E6B-06`, `E6B-24`, `E6B-26`, `E6B-27`, `E6B-28`, `E6B-29`, `E6B-32` | Padded cores, `maxiter < 0` `ValueError`, missing/malformed integrals, charge/spin broadcasting, shared/per-lane external fields, and public `write_*` / `cuda_sync_in_scf` / `store_*` / `ignore_unknown_kwargs` contracts remain intact. |
| **2. S13-supported functional replacement** | `E6B-01`, `E6B-02`, `E6B-03`, `E6B-04`, `E6B-05`, `E6B-07`, `E6B-10`, `E6B-31` | GFN1/GFN2 mode 1 & mode 2 packed evaluation, native GFN2 `Potential` (S13 default) / `Charge` / `Fock` execution on `MAPPED`, PyTorch batch integrals, built-in interactions (ES2, ES3, AES2, D4SC, GB/ALPB), and xitorch-free GFN0 generalized eigensolve. |
| **3. Explicitly retained compatibility adapter** | `E6B-12`, `E6B-13`, `E6B-30` | Custom `Classical` subclasses, duplicate-label classical contributions, per-call classical cache refresh (`LEGACY_COMPOSED`), and classical-only `exclude="scf"`/`"all"` (`SEPARATE_ROUTE`). |
| **4. Intentionally retired S13 behavior** | `E6B-11`, `E6B-14`..`E6B-23`, `E6B-25` | Custom `Interaction` subclasses on iterative SCF (`NotImplementedError`), low-level `LegacySCFOptions` positional/keyword overrides including `fwd_options`/`bck_options`/`eigen_options` (`NotImplementedError`), negative `fermi_etemp` / `fermi_maxiter` (`ValueError`), and uniform `maxiter < 0` rejection across composed routes (`ValueError`). |
| **5. Not yet supported by mixed-System functional routing** | `E6B-08`, `E6B-09` | `libcint` single-System functional evaluation (`INTDRIVER_LIBCINT` and `force_cpu_for_libcint`) composed via `LEGACY_COMPOSED` + `pack_system_results`, plus valid eager-only/implicit S13 strategies (`SCFDifferentiation.IMPLICIT`, multi-stage `broyden-cdiis`/`cdiis-soscf`, and fixed-occupation `SOSCFMethod`). |

---

## 4. Final Mapped-Route (`MAPPED`) vs Single-System Composed (`LEGACY_COMPOSED`) Eligibility

`classify_legacy_system` in `src/dxtb/_src/calculators/legacy_composition.py` and `evaluate_padded_fixed_trip` in `src/dxtb/_src/scf/functional/padded.py` enforce the exact numerical boundary of the mapped fixed-trip kernel:

| SCF Configuration / Method / Representation | Differentiation | Route in `evaluate_packed_systems` | Notes |
|---|---|---|---|
| `LinearMethod` (`CHARGE`, `POTENTIAL`, `FOCK`), GFN1 & GFN2 | `UNROLLED` | `LegacySystemRoute.MAPPED` | Native fixed-trip `vmap` + S2 certification |
| `AndersonMethod` (`POTENTIAL` [S13 default], `CHARGE`, `FOCK`), GFN1 & GFN2 | `UNROLLED` | `LegacySystemRoute.MAPPED` | Native fixed-trip `vmap` + S2 certification; never converted to Fock |
| `BroydenMethod` (`POTENTIAL`, `CHARGE`, `FOCK`), GFN1 & GFN2 | `UNROLLED` | `LegacySystemRoute.MAPPED` | Native fixed-trip `vmap` + S2 certification |
| `CDIISMethod` (`FOCK`), GFN1 & GFN2 | `UNROLLED` | `LegacySystemRoute.MAPPED` | Native Fock-space commutator DIIS in fixed-trip `vmap` |
| Any valid single-stage or multi-stage strategy (`broyden-cdiis`, `anderson-cdiis`, `cdiis-soscf`, `broyden-cdiis-soscf`) | `IMPLICIT` | `LegacySystemRoute.LEGACY_COMPOSED` | Evaluates each `System.singlepoint` independently (preserving S11/S13.1 implicit response & alternate response coordinates) and packs via `pack_system_results` |
| `SOSCFMethod` with `fermi_etemp == 0` | `IMPLICIT` | `LegacySystemRoute.LEGACY_COMPOSED` | Single-System eager/implicit SOSCF + `pack_system_results` |
| `SOSCFMethod` with active Fermi smearing (`kt >= 3e-7`) | Any | Raises `ValueError` | Rejected at both `ConfigSCF` and `classify_legacy_system` |
| Multi-stage or `SOSCFMethod` strategy | `UNROLLED` | Raises `ValueError` | Rejected at both `ConfigSCF`, `classify_legacy_system`, and `evaluate_padded_fixed_trip` |
| `INTDRIVER_LIBCINT` or custom/duplicate-label `Classical` | `UNROLLED` / `IMPLICIT` | `LegacySystemRoute.LEGACY_COMPOSED` | Independent single-`System` functional evaluation + `pack_system_results` |
| `GFN0-xTB` (`requires_iterations=False`) or `exclude="scf"`/`"all"` | Any | `LegacySystemRoute.SEPARATE_ROUTE` | One-shot non-SCF eigensolve or classical-only evaluation + `pack_system_results` |

---

## 5. S13.1 Response-Root Semantics and GFN0 Verification

- **S13.1 Response-Root Semantics**:
  - Full separation between primal root/representation/status and response root/representation/diagnostics is preserved in `src/dxtb/_src/scf/functional/response.py`, `implicit.py`, and `eager.py`.
  - Default implicit response representation remains `SCFRepresentation.POTENTIAL` (`rtol=1e-10`, `atol=1e-12`, `maxiter=None`), with independent S2 certification of converted Potential roots for `CDIIS/Fock` primal calculations, no hidden refinement, explicit rejection of unsupported padded coordinate conversion, and lane-local NaN derivative failure signaling.
- **GFN0 Verification**:
  - `solve_nonselfconsistent` uses the S13 xitorch-free tensor generalized eigensolve (`hamiltonian_to_density`), retains geometry gradients, rejects density-dependent `Interaction` components with `ValueError`, and rejects retired low-level `*args`/`**kwargs` overrides (`unexpected_option`, `eigen_options`, malformed `eigen_options=None`, positional extras) with `NotImplementedError`.

---

## 6. Validation Summary Across Environments

### 6.1 Primary S0–S13.1 & Extended Functional Suites (`py3.14-torch2.14.1`)

| Suite | Files | Result | Duration |
|---|---|---|---|
| **S13 668-test gate** | `test/test_scf/test_functional_*.py`, `test/test_calculator/test_e6a0_system_vmap.py` | **668 passed, 0 failed** | `854.88s` |
| **Extended S13.1 + E3 + E6a + E6bimpl + E7a gate** | `test_s13_public_strategy.py`, `test_e6c1_w2c9_batch_implicit.py`, `test_s7_architecture.py`, `test_e3_end_to_end_acceptance.py`, `test_e6a1_conformer.py`, `test_e6a1a_gfn0_spin_validation.py`, `test_e6a2_public_status.py`, `test_e6a3_component_vmap.py`, `test_e6a4_transform_acceptance.py`, `test_e6bimpl0_padded_integrals.py`, `test_e6bimpl1a_padded_spectral.py`, `test_e6bimpl2a_gfn2_canonical.py`, `test_e6bimpl2b_gfn2_scf.py`, `test_e7a_padding_safety.py` | **251 passed, 0 failed** | `139.46s` |

### 6.2 Five-Environment Integrated Acceptance Matrix

Suite executed across all five supported CPU environments:
- `test/test_scf/test_s13_public_strategy.py`
- `test/test_scf/test_e6c1_w2c9_batch_implicit.py`
- `test/test_scf/test_nonselfconsistent.py`
- `test/test_calculator/test_e6bimpl3_legacy_composition.py`
- `test/test_calculator/test_e6bcompat_differential.py`
- `test/test_calculator/test_e7a_padding_safety.py`
- `test/test_calculator/test_e3_end_to_end_acceptance.py`
- `test/test_calculator/test_b8_e8_functional_properties.py`

| Environment | Result | Duration |
|---|---|---|
| `py3.10-torch2.7.0` | **149 passed, 0 failed** | `1030.40s` |
| `py3.11-torch2.8.0` | **149 passed, 0 failed** | `1036.41s` |
| `py3.12-torch2.11.0` | **149 passed, 0 failed** | `1063.57s` |
| `py3.12-torch2.13.0` | **149 passed, 0 failed** | `1057.38s` |
| `py3.14-torch2.14.1` | **149 passed, 0 failed** | `936.22s` |

---

## 7. Independent Review Summary

An independent read-only audit of the integrated worktree (`integration/scf-s13-general-final`, merging `e694117b87287e6557d1f4256a9f3e9175e3211d` into `e78d3f6ac05a970e7871fa65a358b8f3d028cd9e`) was performed by a dedicated reviewer agent (`439f9f3c-adeb-4736-bbef-f68f5f219010`) against all 14 mandatory checklist items:

- **Verdict**: **`APPROVED`** (`0 BLOCKER`, `0 MAJOR`, `0 MINOR`, `1 NIT`)
- **Checklist Verification (14/14 PASS)**:
  1. No S13 public API regression (`SCFStrategy`, `SCFStage`, `SCFRepresentation`, `SCFDifferentiation`, `SCFConvergenceProfile`, `SCFResponseSettings`, native methods, and S13 defaults preserved).
  2. No reintroduction of retired `ConfigSCF` fields (`mixer`, `scp_mode`, `scf_mode`, `strict`, `f_atol`, or xitorch mixer string keys).
  3. Correct S13.1 string parser (`scf_mode`, `scp_mode`, `mixer` translation + mixed-input rejection) and response-root separation (`prepare_response_root`, `run_prepared_functional_implicit`, `batch_implicit.py`).
  4. Correct GFN2 strategy/representation handling in `packed_result.py` and `legacy_composition.py` (`_normalize_system_for_e6bcompat` never converts GFN2 `Anderson/Potential` or `Charge` to `Fock`).
  5. Correct method/stage/differentiation routing in `classify_legacy_system` and `evaluate_padded_fixed_trip` (`Linear`, `Anderson`, `Broyden` in `CHARGE`/`POTENTIAL`/`FOCK`, `CDIIS` in `FOCK` with single-stage `UNROLLED` -> `MAPPED`; `SOSCF` + smearing -> `ValueError`; multi-stage/eager-only + `UNROLLED` -> `ValueError`; `IMPLICIT` and valid multi-stage/eager-only -> `LEGACY_COMPOSED`).
  6. Preservation of `pack_system_results` and non-contiguous/trailing atom-mask fixes (`atom_masks`, `target_atoms`).
  7. Preservation of duplicate classical energy contribution slots (`_classical_contributions`) and collision-safe `Result._classical_display_keys`.
  8. Preservation of `write_*` integral disk-export and `cuda_sync_in_scf` `try`/`finally` handling.
  9. Preservation of GFN0 option separation (`solve_nonselfconsistent` rejects `*args, **kwargs` with `NotImplementedError`) and Stage 4 (`int_level < INTLEVEL_HCORE` / missing `h0_setup`) precedence before Stage 5 (`legacy_options`).
  10. Preservation of functional property fixed-point qualification (`test_b8_e8_functional_properties.py` Section 5 tests).
  11. Reconciled compatibility manifest (`docs/plan/E6bcompat-final-routing-manifest.json` schema `2.0-s13.1` with `s13_category`, `authority`, `decision`, and valid test node references).
  12. `Calculator.singlepoint` default batched behavior remains legacy sequential (`default_calculator_routing_changed: false`).
  13. Both parent commit chains (`e78d3f6ac05a970e7871fa65a358b8f3d028cd9e` and `e694117b87287e6557d1f4256a9f3e9175e3211d`) represented in the merge state.
  14. Comprehensive test coverage across all resolved conflict areas (reviewer independently executed 61 conflict-area tests with 61 passed, 0 failed).
- **Resolution of Findings**:
  - `NIT 1`: Stripped trailing blank lines at EOF in 4 modified files so `git diff --check` exits with `0`.

---

## 8. Remaining Project-Owner Decision

- **Public `Calculator.singlepoint` Batch Cutover**:
  - In accordance with Section 13 of the integration specification, default `Calculator.singlepoint` batch dispatch (`batch_mode > 0`) has **not** been switched to `evaluate_calculator_packed_singlepoint`.
  - The integrated functional packed harness (`evaluate_packed_systems` and `evaluate_calculator_packed_singlepoint`) is fully reconciled with S0–S13.1 and verified across all five supported CPU environments, ready for the project owner's authorization to activate public Calculator batch routing and proceed with E4.6 legacy deletion.
