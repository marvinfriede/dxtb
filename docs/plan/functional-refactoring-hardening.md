# Functional Refactoring Closure Hardening and Legacy-Routing Readiness

## 1. Executive Summary

This engineering record documents the closure hardening of the functional `Model` / `System` / `Result` refactoring on branch `claude/tender-dijkstra-cda42s` (base SHA `9653ecdca42ea7c276a44f447497ef4bd53329fd`). It resolves all integration gaps identified during post-consolidation source inspection prior to public `Calculator` batch routing cutover and legacy SCF retirement:

1. **Packed Classical-Contribution Multiplicity (`P0`)**: Replaced dictionary coercion (`dict(r.classical)`) in `pack_system_results` (`src/dxtb/_src/calculators/packed_result.py`) with occurrence-indexed slot packing `(label, occ)` that preserves repeated labels, ordering, multiplicity, and individual gradients across `MAPPED`, `LEGACY_COMPOSED`, and `SEPARATE_ROUTE` lanes. Fixed `Result.get_energies()` (`src/dxtb/_src/calculators/result.py`) to sum over `self.classical` without dropping duplicate labels and to disambiguate repeated entries in the `"Classical"["sub"]` summary, and documented the legacy dictionary view semantics of `Result.cenergies`.
2. **Packed Input and Option Compatibility Matrix (`P0`)**: Hardened `evaluate_packed_systems` (`src/dxtb/_src/calculators/packed_result.py`) so unknown public `**kwargs` and unknown `legacy_options.kwargs` raise `TypeError` by default, while forwarding valid low-level engine options (`fwd_options`, `bck_options`, `eigen_options`, and `legacy_options.args`) to `compose_legacy_systems` and ignoring unknown kwargs only when `ignore_unknown_kwargs=True` (and `not strict_e6bcompat`). Completed the `legacy_compat=True` vs `legacy_compat=False` normalization and failure-policy matrix for charges, spins, electric fields, field gradients, and electron-capacity overflows.
3. **Complete Packed `Result` Adapter (`P0`)**: Audited and tested all 20 `Result` dataclass fields plus `Result.multipoles` across mixed SCF (`MAPPED` + `LEGACY_COMPOSED`) and non-SCF (`SEPARATE_ROUTE`, `exclude=["scf"]`) batches, per-lane fields/field gradients, and shared `Model.parameter_tree` gradients.
4. **Functional Property Acceptance — Contract A & Contract B (`P0` / `P1`)**: Established both **Contract A** (finite-trajectory unrolled differentiation at small `maxiter`) and **Contract B** (genuinely converged physical property acceptance at `x_atol=1e-12`, residual $< 10^{-12}$) across all 13 functional property APIs on `System` and `dxtb._src.calculators.properties.functional`, including physical symmetries, translational/rotational invariance, equilibrium vibrational zero-mode structure, and mixed-derivative identities. Documented exact sign conventions, prefactors, units, quadrupole ordering, equilibrium geometry requirements, and `torch.func` transform boundaries.
5. **E3 / SCF Commit Provenance Reaudit (`P1`)**: Corrected the commit provenance record across `abdb4562a0607475e775072277b63f82501363a0..9653ecdca42ea7c276a44f447497ef4bd53329fd`.
6. **Production Routing Cutover & Legacy Retirement Plan (`P1`)**: Defined the exact public `Calculator` routing cutover plan (`E6broute`) and the file-by-file legacy retirement deletion map (`E4.6a`, `E4.6b`, `E2-legacy`, `C9`, `B6`).

---

## 2. Packed Classical-Contribution Handling (`Section 2`)

### 2.1 Reachability of Repeated Classical Labels

1. **Setup-Time Invariants**:
   - `ComponentList.__init__` (`src/dxtb/_src/components/list.py:146`) deduplicates identical component instances by object identity (`dict.fromkeys`), while `Model.setup` (`src/dxtb/_src/calculators/model.py:1596-1603`) rejects distinct `Classical` instances sharing a `.label` at `batch_mode == 0` setup time.
   - However, `Model.setup` explicitly supports multiple identity-associated `DispersionD3` instances (`system.d3_setups: tuple[tuple[DispersionD3, DispersionD3CoreSetup], ...]`).
2. **Reachable Repeated-Label Configurations**:
   - **Post-setup label unification on identity-keyed `DispersionD3` components**: Two `DispersionD3` instances set up with distinct labels (`"D3:first"`, `"D3:second"`) and subsequently assigned `component.label = "DispersionD3"` evaluate via `_singlepoint_impl` (`src/dxtb/_src/calculators/singlepoint.py:618-638`) using `system.d3_setups` identity lookup and append two `"DispersionD3"` entries to `Result.classical`.
   - **Post-setup `system.classicals.components` extension**: Appending multiple custom `Classical` components sharing a label (e.g. `"CustomHarmonic"`) produces repeated labels in `Result.classical`.
   - **Direct `Result.snapshot` / `pack_system_results` composition**: Any `Result` constructed with duplicate labels in `Result.classical: tuple[tuple[str, Tensor], ...]`.
3. **Route Classification Fix (`src/dxtb/_src/calculators/legacy_composition.py:109-113`)**:
   - `classify_legacy_system` now checks `system._active_exact_core() is None` in addition to `len(core.classical_terms) != len(system.classicals.components)`. When a `System`'s classical component labels or setups are mutated after `Model.setup`, `classify_legacy_system` routes the `System` to `LEGACY_COMPOSED` so `evaluate_packed_systems` matches `System.singlepoint` exactly.

### 2.2 Occurrence-Indexed Slot Packing & `Result` Energy Summaries

- **`pack_system_results` (`src/dxtb/_src/calculators/packed_result.py:296-339`)**:
  - Replaces `classical_maps = [dict(r.classical) for r in results]` with occurrence-indexed slot keys `(label, occ)` where `occ` is the 0-based occurrence index of `label` within each lane's `Result.classical`.
  - Preserves the exact ordered sequence of `(label, occ)` slots across all lanes, pads each slot to `(B, max_atoms)`, and preserves independent autograd graphs for every occurrence.
- **`Result.cenergies` (`src/dxtb/_src/calculators/result.py:397-413`)**:
  - Explicitly documented as a legacy compatibility dictionary view (`dict(self.classical)`) that retains the last occurrence when duplicate labels are present, while `Result.classical` is the authoritative ordered tuple representation.
- **`Result.get_energies()` (`src/dxtb/_src/calculators/result.py:527-579`)**:
  - Computes `ctotal = sum(value.sum().item() for _, value in self.classical)` directly over `self.classical` so duplicate labels are never dropped from total classical energy reporting, and disambiguates repeated labels in `"Classical"["sub"]` as `f"{name} ({occ})"`.

---

## 3. Packed Input & Option Compatibility Matrix (`Section 3`)

Implemented in `src/dxtb/_src/calculators/packed_result.py` and `src/dxtb/_src/scf/functional/padded.py`:

| Input / Option Category | `legacy_compat=False` (Functional Packed API) | `legacy_compat=True` (`Calculator` Compatibility Mode) | `strict_e6bcompat=True` Override |
| :--- | :--- | :--- | :--- |
| **Unknown public `**kwargs`** | Raises `TypeError` by default; ignored only if `ignore_unknown_kwargs=True`. | Raises `TypeError` by default; ignored only if `ignore_unknown_kwargs=True`. | Always raises `TypeError` (ignores `ignore_unknown_kwargs`). |
| **Removed `store_*` kwargs** | Always raises `TypeError`. | Always raises `TypeError`. | Always raises `TypeError`. |
| **Valid `legacy_options.kwargs` (`fwd_options`, `bck_options`, `eigen_options`) & `legacy_options.args`** | Routes lane to `LEGACY_COMPOSED` and forwards options to `scf.solve_legacy`. | Routes lane to `LEGACY_COMPOSED` and forwards options to `scf.solve_legacy`. | Same. |
| **Unknown `legacy_options.kwargs`** | Raises `TypeError` by default; filtered out only if `ignore_unknown_kwargs=True`. | Raises `TypeError` by default; filtered out only if `ignore_unknown_kwargs=True`. | Always raises `TypeError`. |
| **GFN2 with non-Fock `scp_mode` (`CHARGE` / `POTENTIAL`)** | Emits `UserWarning` and normalizes `scp_mode` to `SCP_MODE_FOCK` (unless `system.config.strict=True` / `system.config.scf.strict=True`, which raises `NotImplementedError`). | Emits `UserWarning` and normalizes `scp_mode` to `SCP_MODE_FOCK` (unless `system.config.strict=True` / `system.config.scf.strict=True`, which raises `NotImplementedError`). | Raises `NotImplementedError`. |
| **Charge (`chrg`) — `int` / `float` / 0D `Tensor`** | Broadcasts any scalar charge across all $B$ lanes. | Accepts default `0` for `count > 1`; raises `ValueError` for non-default scalar `int`/`float` or any 0D `Tensor` when `count > 1`. | Same as selected `legacy_compat`. |
| **Charge (`chrg`) — 1D `(B,)`, 2D `(B, 1)`, or `Sequence` of length `B`** | Accepted; preserves per-lane autograd and validates dtype/device. | Accepted; preserves per-lane autograd and validates dtype/device. | Same. |
| **Spin (`spin`) — `None`, scalar, 1D `(B,)`, 2D `(B, 1)`, or `Sequence` of length `B`** | `None` derives ground-state parity `(nab[0] - nab[1])` per lane; explicit values validate parity with electron count and raise `ValueError` on mismatch. | Same. | Same. |
| **Electric field (`field`) & gradient (`field_grad`)** | Accepts shared `(3,)` / `(3, 3)`, per-lane `(B, 3)` / `(B, 3, 3)`, or `Sequence` of length $B$ (including mixed `Tensor` and `None` lanes). | Accepts shared `(3,)` / `(3, 3)` (matching public `Calculator.singlepoint`); rejects per-lane sequences/tensors with `ValueError`. | Same as selected `legacy_compat`. |
| **Impossible electron count (`nel > nao`) / Fermi failure** | Preserves lane-local `SCFStatus` (`SCFFailureCode.TOO_MANY_ELECTRONS` / `FERMI_NONCONVERGENCE`) without aborting valid lanes. | Raises eager `ValueError("Number of electrons exceeds the number of orbitals.")` or `RuntimeError("Fermi energy failed to converge.")`. | Same as selected `legacy_compat`. |
| **Outer SCF nonconvergence (`converged=False`, `maxiter > 0`)** | Preserves `Result.converged` (`(B,)`) and `Result.status` without warning/raising. | Emits `SCFConvergenceWarning` or raises `SCFConvergenceError` (with `.source_id`) when `force_convergence=True`. | Same as selected `legacy_compat`. |

---

## 4. Complete Packed `Result` Adapter Audit (`Section 4`)

All 20 `Result` dataclass fields plus the derived `Result.multipoles` property are populated by `pack_system_results` (`src/dxtb/_src/calculators/packed_result.py`) and verified in `test_complete_packed_result_adapter_all_fields_and_model_param_gradients` (`test/test_calculator/test_e6bcompat_differential.py`):

| Field | Single-System Shape | Packed Batch Shape | Padding / Mixed SCF + Non-SCF (`exclude=["scf"]`) Semantics | Differentiability |
| :--- | :--- | :--- | :--- | :--- |
| `energy` | `(nat,)` | `(B, max_atoms)` | Right-padded with `0.0`; sum of `classical + scf + fenergy`. | 1st–3rd order wrt `positions`, `chrg`, `field`, `field_grad`, `Model.parameter_tree` |
| `scf` | `(nat,)` | `(B, max_atoms)` | Right-padded with `0.0`; `0.0` for non-SCF lanes. | Differentiable wrt all active inputs |
| `classical` | `tuple[(str, (nat,)), ...]` | `tuple[(str, (B, max_atoms)), ...]` | Occurrence-indexed `(label, occ)` slots; right-padded with `0.0`. | Differentiable per occurrence slot |
| `fenergy` | `(nat,)` | `(B, max_atoms)` | Right-padded with `0.0`; electronic Mermin free-energy contribution. | Differentiable wrt active inputs |
| `iterations` | `()` (`int64`) | `(B,)` (`int64`) | Stacked per-lane iteration counts (`0` for non-SCF lanes). | Non-differentiable integer metadata |
| `charges` | `ChargeResult` (`(norb,)`, `(nat, 3)`, `(nat, 6)`) | `ChargeResult` (`(B, max_orbs)`, `(B, max_atoms, 3)`, `(B, max_atoms, 6)`) | Right-padded with `0.0`; `None` only if all lanes have `charges=None`. | Differentiable wrt active inputs |
| `multipoles` *(property)* | `(nat, 9)` (`GFN2`) | `(B, max_atoms, 9)` (`GFN2`) | Concatenates atom-resolved `mono`, `dipole`, and `quadrupole` charges. | Differentiable wrt active inputs |
| `density` | `(2, norb, norb)` | `(B, 2, max_orbs, max_orbs)` | Right-padded along both orbital axes with `0.0`. | Differentiable wrt active inputs |
| `coefficients` | `(2, norb, nmo)` | `(B, 2, max_orbs, max_mos)` | Right-padded along AO and MO axes with `0.0`. | Differentiable (gauge-dependent; physical projectors gauge-invariant) |
| `emo` | `(2, nmo)` | `(B, 2, max_mos)` | Right-padded with `0.0`. | Differentiable wrt active inputs |
| `occupation` | `(2, nmo)` | `(B, 2, max_mos)` | Right-padded with `0.0` using spin-channel prefix `(*occ_prefix, max_mos)`. | Differentiable wrt active inputs |
| `potential` | `PotentialResult` (`(norb,)`, `(nat, 3)`, `(nat, 6)`) | `PotentialResult` (`(B, max_orbs)`, `(B, max_atoms, 3)`, `(B, max_atoms, 6)`) | Right-padded with `0.0`; `None` only if all lanes have `potential=None`. | Differentiable wrt active inputs |
| `hamiltonian` | `(2, norb, norb)` | `(B, 2, max_orbs, max_orbs)` | Right-padded along both orbital axes with `0.0`. | Differentiable wrt active inputs |
| `overlap` | `(norb, norb)` | `(B, max_orbs, max_orbs)` | Right-padded along both orbital axes with `0.0`. | Differentiable wrt `positions` and basis parameters |
| `hcore` | `(norb, norb)` | `(B, max_orbs, max_orbs)` | Right-padded along both orbital axes with `0.0`. | Differentiable wrt `positions` and H0 parameters |
| `dipole_integrals` | `(3, norb, norb)` or `None` | `(B, 3, max_orbs, max_orbs)` or `None` | Right-padded along both orbital axes with `0.0` when `int_level >= Dipole`. | Differentiable wrt `positions` |
| `quadrupole_integrals` | `(6, norb, norb)` or `None` | `(B, 6, max_orbs, max_orbs)` or `None` | Right-padded along both orbital axes with `0.0` when `int_level >= Quadrupole`. | Differentiable wrt `positions` |
| `overlap_norm` | `(norb,)` or `None` | `(B, max_orbs)` or `None` | Right-padded with `0.0`. | Differentiable wrt `positions` |
| `converged` | `()` (`bool`) or `None` | `(B,)` (`bool`) or `None` | `None` when `legacy_compat=True`; `(B,)` in `legacy_compat=False` (non-SCF lanes filled with `True`). | Boolean status metadata |
| `residual` | `(nstate,)` or `None` | `(B, max_state)` or `None` | `None` when `legacy_compat=True`; right-padded `(B, max_state)` in `legacy_compat=False` (`0.0` for non-SCF lanes). | Tensor residual diagnostic |
| `status` | `SCFStatus((), ())` or `None` | `SCFStatus((B,), (B,))` or `None` | `None` when `legacy_compat=True`; `(B,)` codes in `legacy_compat=False` (`0` for non-SCF lanes). | Integer failure-code metadata |

---

## 5. Functional Property Numerical Acceptance & API Conventions (`Sections 5 & 6`)

### 5.1 Two-Contract Acceptance Suite (`test/test_calculator/test_b8_e8_functional_properties.py`)

1. **Contract A — Finite-Trajectory Differentiation (`maxiter=3`)**:
   - Tests `test_b8a_first_and_second_order_functional_properties` and `test_b8b_e8_third_order_hyperpolarizability_raman_and_param_forces` (including vibrational/IR parity and finite-temperature free-energy stationarity).
   - Verifies that differentiating the fixed-trip $K$-step trajectory via `torch.func` matches central finite differences of the same $K$-step trajectory and `Calculator` unrolled autograd across GFN1 and GFN2.
2. **Contract B — Converged Physical Property Acceptance (`maxiter=10..15`, `x_atol=1e-12`, residual $< 10^{-12}$)**:
   - Tests `test_contract_b_converged_water_properties_symmetries_and_mixed_derivatives` and `test_contract_b_equilibrium_vibrations_ir_raman_and_third_order` across GFN1 ($\text{H}_2$, $\text{H}_2\text{O}$) and GFN2 ($\text{H}_2$, $\text{H}_2\text{O}$).
   - Verifies:
     - Tight SCF convergence (`res.converged == True`, `res.residual.abs().max() < 1e-12`).
     - Agreement against converged `Calculator` properties and central finite differences across `energy`, `forces`, `hessian`, `dipole`, `quadrupole`, `polarizability`, `dipole_deriv`, `pol_deriv`, `hyperpolarizability`, `third_order_directional`, `vibration`, `ir`, and `raman`.
     - Physical tensor symmetries: Hessian symmetry ($H_{ia,jb} = H_{jb,ia}$), polarizability symmetry ($\alpha_{ab} = \alpha_{ba}$), polarizability derivative field-index symmetry ($\chi_{ab,ic} = \chi_{ba,ic}$), and full 6-permutation Kleinman symmetry of the static first hyperpolarizability ($\beta_{abc}$).
     - Translational invariance of forces ($\sum_A \mathbf{F}_A = \mathbf{0}$), Hessian ($\sum_B \mathbf{H}_{A,B} = \mathbf{0}$), and dipole derivatives ($\sum_A \nabla_A \boldsymbol{\mu} = \mathbf{0}$ for neutral systems at zero external field).
     - Equivalent mixed-derivative constructions: $\text{jacfwd}(\text{grad}(E)) = \text{jacrev}(\text{grad}(E))$, $\partial \boldsymbol{\mu}/\partial \mathbf{R} = \partial \mathbf{F}/\partial \mathbf{F}_{\text{ext}}$, and $\partial \boldsymbol{\alpha}/\partial \mathbf{R} = \partial (\nabla_{\mathbf{R}}\boldsymbol{\mu})/\partial \mathbf{F}_{\text{ext}}$.
     - Stationary equilibrium geometry ($\max |\mathbf{F}| < 10^{-11}\text{ E}_\text{h}/a_0$) vibrational analysis: 5 translational+rotational zero modes and 1 stretching mode for linear $\text{H}_2$ (zero IR intensity, positive Raman intensity), and 6 translational+rotational zero modes and 3 positive internal modes for bent $\text{H}_2\text{O}$ (matching projected vs unprojected frequencies).

### 5.2 Property Convention & Transform-Support Matrix (`src/dxtb/_src/calculators/properties/functional.py`)

All quantities are returned in **atomic units (Hartree / Bohr / $e$)** unless noted otherwise:

| Property Function / `System` Method | Mathematical Definition & Sign Convention | Output Shape & Component Ordering | Outer `torch.func` Transform Support (`grad` / `jacfwd` / `jacrev` / `vmap`) |
| :--- | :--- | :--- | :--- |
| `system_energy` / `System.energy` | $E_{\text{tot}} = E_{\text{classical}} + E_{\text{scf}} + \Omega_{\text{fenergy}}$ (or $E_{\text{tot}} - \Omega_{\text{fenergy}}$ if `include_free_energy=False`) | Scalar `()` ($\text{E}_\text{h}$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_forces` / `System.forces` | $\mathbf{F} = -\frac{\partial E}{\partial \mathbf{R}}$ | `(nat, 3)` ($\text{E}_\text{h}/a_0$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_hessian` / `System.hessian` | $\mathbf{H} = +\frac{\partial^2 E}{\partial \mathbf{R}^2}$ | `(nat, 3, nat, 3)` or `(3*nat, 3*nat)` if `matrix=True` ($\text{E}_\text{h}/a_0^2$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_dipole` / `System.dipole` | $\boldsymbol{\mu} = -\frac{\partial E}{\partial \mathbf{F}_{\text{ext}}}$ (follows $H_{\text{efield}} = +\boldsymbol{\mu}_{\text{op}}\cdot\mathbf{F}_{\text{ext}}$) | `(3,)` ($e\cdot a_0$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_quadrupole` / `System.quadrupole` | $\mathbf{Q} = -3\,\frac{\partial E}{\partial \mathbf{G}_{\text{ext}}}$ (traceless primitive quadrupole) | `(6,)` packed lower-triangular order: `(xx, yx, yy, zx, zy, zz)` ($e\cdot a_0^2$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_polarizability` / `System.polarizability` | $\boldsymbol{\alpha} = +\frac{\partial \boldsymbol{\mu}}{\partial \mathbf{F}_{\text{ext}}} = -\frac{\partial^2 E}{\partial \mathbf{F}_{\text{ext}}^2}$ | `(3, 3)` ($e^2 a_0^2 / \text{E}_\text{h}$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_dipole_deriv` / `System.dipole_deriv` | $\frac{\partial \boldsymbol{\mu}}{\partial \mathbf{R}} = -\frac{\partial^2 E}{\partial \mathbf{R}\,\partial \mathbf{F}_{\text{ext}}}$ | `(3, nat, 3)` with axes `(field_xyz, atom, coord_xyz)` ($e$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_pol_deriv` / `System.pol_deriv` | $\boldsymbol{\chi} = \frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{R}} = -\frac{\partial^3 E}{\partial \mathbf{R}\,\partial \mathbf{F}_{\text{ext}}^2}$ | `(3, 3, nat, 3)` with axes `(field_a, field_b, atom, coord_xyz)` | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_hyperpolarizability` / `System.hyperpolarizability` | $\boldsymbol{\beta} = \frac{\partial \boldsymbol{\alpha}}{\partial \mathbf{F}_{\text{ext}}} = -\frac{\partial^3 E}{\partial \mathbf{F}_{\text{ext}}^3}$ | `(3, 3, 3)` | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_third_order_directional` / `System.third_order_directional` | $D^3 E(\mathbf{R})[\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3]$ via nested `jvp(jvp(grad(E)))` | Scalar `()` ($\text{E}_\text{h}/a_0^3$) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_bond_orders` / `System.bond_orders` | Wiberg bond order $W_{AB} = \sum_{\mu\in A,\nu\in B} [(\mathbf{P}_\alpha + \mathbf{P}_\beta)\mathbf{S}]_{\mu\nu}[(\mathbf{P}_\alpha + \mathbf{P}_\beta)\mathbf{S}]_{\nu\mu}$ (off-diagonal) | `(nat, nat)` (dimensionless) | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_parameter_force_jacobian` | $\frac{\partial \mathbf{F}(\mathbf{R}, \theta_k)}{\partial \theta_k}$ via `jacrev`/`jacfwd` over `Model.setup_functional` for a selected `parameter_key` | `Tensor` of shape `(nat, 3, *param.shape)` | Full support (`grad`, `jacfwd`, `jacrev`, `vmap`) |
| `system_vibration` / `System.vibration` | Mass-weighted Hessian eigenproblem via `vib_analysis` (requires stationary equilibrium geometry $\partial E/\partial \mathbf{R} = \mathbf{0}$) | `VibrationResult(freqs, modes)` (`freqs` in $\text{E}_\text{h}$, $(3N - N_{\text{tr/rot}},)$) | Differentiable wrt `positions`/`field`/`chrg` (`grad`/`jacfwd`/`jacrev`) for fixed atom count; **not `vmap`-compatible across variable rank** because `vib_analysis` uses data-dependent SVD rank detection (`torch.sum(s > 1e-6)`). |
| `system_ir` / `System.ir` | Normal-mode projection of `system_dipole_deriv` + `system_vibration` | `IRResult(freqs, ints, modes)` | Same transform boundary as `system_vibration` (differentiable; no variable-rank `vmap`). |
| `system_raman` / `System.raman` | Normal-mode projection of `system_pol_deriv` + `system_vibration` | `RamanResult(freqs, ints, depol, modes)` | Same transform boundary as `system_vibration` (differentiable; no variable-rank `vmap`). |

---

## 6. E3 & Functional SCF Commit Provenance Reaudit (`Section 7`)

Inspection of `git log -p abdb4562a0607475e775072277b63f82501363a0..9653ecdca42ea7c276a44f447497ef4bd53329fd` confirms:

1. **Present in `abdb4562a0607475e775072277b63f82501363a0` prior to the closure package**:
   - `_cutoff_mask` and `_build_orbital_energy` in `src/dxtb/_src/scf/functional/solver.py`.
   - `_anderson_step` history slot replacement (`replace_idx = (step - 1) % history`) and mixing damping in `src/dxtb/_src/scf/functional/anderson.py`.
2. **Introduced in `d849982491b42be973216944bf5bf6dfae4ebbf1`**:
   - Removed `.item()` from `src/dxtb/_src/scf/functional/spectral.py:342` (`int(scale_power.max())`) and removed multiprocessing from `batch_implicit.py` and `batch_streaming.py`.
3. **Introduced in `eafd0990ec91d5c1a5b950e06448efdef71722c8`**:
   - Added `record_initial_status: bool | None = None` to `_solve_map_control`, `_solve_map`, and `_solve_map_anderson` in `src/dxtb/_src/scf/functional/solver.py` and `anderson.py`, setting `record_initial_status=True` in Potential and Fock solvers so `maxiter=0` records the initial raw map's Fermi status before finalization.
   - Added the low-temperature equal-partition entropy branch in `free_energy_from_spectral` (`src/dxtb/_src/scf/functional/solver.py:1217-1228`) to eliminate $1\text{ ULP}$ ($2.65\times 10^{-23}\text{ E}_\text{h}$) matrix-product roundoff at sub-cutoff positive $k_B T$.
   - Updated stale test assertions in `test/test_scf/test_functional_anderson.py` and `test/test_scf/test_functional_temperature_cutoff.py` that had assumed pre-`abdb456` behavior, and verified Anderson/sub-cutoff gradients against both central finite differences and unbroadened legacy (`broadening_method=None`), isolating the $O(b_f^2 / \Delta\varepsilon^2) \approx 1.1\times 10^{-14}$ Lorentzian eigenvector broadening bias in legacy `BaseTSCF.diagonalize`.
   - *(Correction Note)*: The commit message of `eafd0990ec91d5c1a5b950e06448efdef71722c8` erroneously stated that `_cutoff_mask`, `_build_orbital_energy`, and `_anderson_step` were changed in that commit; those functions were already in their final form in `abdb4562a0607475e775072277b63f82501363a0`.

---

## 7. Broader E7 Padding Safety & Architecture Simplification (`Sections 8 & 9`)

- **Broader E7 Padding Safety (`test/test_calculator/test_e7a_padding_safety.py`)**:
  - Verified across 24 test cases covering GFN1 and GFN2:
    - Zero-valued `(0, 0, 0)` padding coordinates coincident with an active atom at the origin.
    - Padding coordinates coincident with a non-origin active atom.
    - Displaced noncontiguous padding coordinates and noncontiguous canonical source masks (`test_noncontiguous_canonical_masks_preserve_inactive_zero_and_third_order`).
    - Charge and multipole reductions (`mono`, `dipole`, `quadrupole`), overlap factorization (`_symmetric_inverse_sqrt`), `ES2`, `ES3`, `AES2`, `D4SC`, classical `Repulsion`/`SRB`/`D3`/`D4`/`IES`/`Halogen`, and 1st–3rd order derivatives (`test_broader_e7_padding_safety_audit_matrix`).
    - Isolation of invalid Fermi lanes (`SCFFailureCode.TOO_MANY_ELECTRONS`) from valid lanes under `vmap` (`test_mixed_valid_and_invalid_padded_lanes_isolation`).
- **Architecture Simplification**:
  - Public and candidate production paths (`Model`, `System`, `Result`, `singlepoint.py`, `packed_result.py`, `canonical_padding.py`, `padded.py`, `properties/functional.py`) never import or depend on experimental converged-root/streaming modules (`batch_implicit.py`, `batch_streaming.py`).
  - Shared field/field-gradient normalization and per-lane perturbation forwarding are consolidated cleanly across `packed_result.py`, `legacy_composition.py`, and `scf/functional/padded.py`.

---

## 8. Concrete Production Routing Cutover Plan (`Section 10 — E6broute`)

### 8.1 Current vs Target Public `Calculator` Routing

| Call Pattern | Current Production Route | Target `E6broute` Production Route |
| :--- | :--- | :--- |
| `System.singlepoint(pos, ...)` (`batch_mode == 0`) | `system_singlepoint` -> `_singlepoint_impl(..., fixed_trip=True)` -> `dxtb._src.scf.functional` | Unchanged (already on functional core). |
| `System.<property>(pos, ...)` (`energy`, `forces`, `hessian`, `dipole`, `quadrupole`, `polarizability`, `dipole_deriv`, `pol_deriv`, `hyperpolarizability`, `third_order_directional`, `bond_orders`, `vibration`, `ir`, `raman`) | `dxtb._src.calculators.properties.functional` (`torch.func` over `System`) | Unchanged (already on functional core). |
| `Calculator(numbers_1d, ...).singlepoint(pos_2d, ...)` (`batch_mode == 0`, PyTorch, `SCF_MODE_FULL`) | `EnergyCalculator.singlepoint` -> `_singlepoint_eager` -> `scf.solve` -> `run_prepared_functional_full` | Unchanged (already on functional core). |
| `Calculator(numbers_2d, ..., batch_mode=1 \| 2).singlepoint(pos_3d, ...)` | Legacy packed `System(batch_mode=1\|2)` -> `ClassicalList.get_cache` / `InteractionList.get_cache` -> `scf.solve_legacy` (`SelfConsistentFieldFull` / `SelfConsistentFieldImplicit`) | **Target `E6broute`**: Construct per-lane `System(batch_mode=0)` tuple on `Calculator` and dispatch `EnergyCalculator.singlepoint` to `evaluate_packed_systems(systems, positions, chrg, spin, field, field_grad, legacy_options=..., legacy_compat=True, ignore_unknown_kwargs=True, **kwargs)`. |

### 8.2 Exact Cutover Steps When Authorized

1. In `Calculator.__init__` (`src/dxtb/_src/calculators/main.py`): when `numbers.ndim == 2` (`batch_mode in (1, 2)`), unpad each row's positive atomic numbers `nums[nums > 0]` and construct `self._lane_systems = tuple(self.model.setup(row_nums, batch_mode=0) for row_nums in unpadded_numbers)`.
2. In `EnergyCalculator.singlepoint` (`src/dxtb/_src/calculators/types/energy.py`): when `self.ihelp.batch_mode in (1, 2)`, delegate to `evaluate_packed_systems(self._lane_systems, positions, chrg=chrg, spin=spin, field=field, field_grad=field_grad, legacy_options=LegacySCFOptions(args=args, kwargs=kwargs), legacy_compat=True, ignore_unknown_kwargs=True)`.
3. **Project-Owner Decision Options on `W2` / `W3` Gates**:
   - **Option 1 (Correctness-First Cutover / Defer `W2`/`W3` to Performance Phase)**: Activate `E6broute` in `EnergyCalculator.singlepoint` now based on complete `E6bcompat` numerical/exception/warning parity, retiring legacy batched `Calculator` state immediately and addressing conformer/mixed-batch speedups (`W2`/`W3`) in the dedicated performance phase.
   - **Option 2 (Gate Activation on `W2`/`W3` Clearance)**: Keep `evaluate_packed_systems` as the verified `E6broute` entry point while retaining legacy `batch_mode=1|2` dispatch on `Calculator` until the post-refactoring performance phase clears `W2`/`W3`.

---

## 9. Concrete Legacy Retirement Deletion Map (`Section 11`)

Once `E6broute` cutover (Section 8) and `E4.6b` implicit-mode cutover are authorized, the legacy stack can be retired in four deterministic commits:

| Retirement Stage | Files / Classes / Caches to Delete or Narrow | Prerequisite Gate | Tests to Migrate or Retire |
| :--- | :--- | :--- | :--- |
| **Stage 1 (`E4.6a` — Unrolled Full SCF Retirement)** | Delete `SelfConsistentFieldFull`, `SelfConsistentFieldFullMixer`, and `SelfConsistentFieldSingleShot` in `src/dxtb/_src/scf/unrolling.py`; remove `SCF_MODE_UNROLL_MIXER` / `SCF_MODE_EXPERIMENTAL` branches in `src/dxtb/_src/scf/iterator.py` and `src/dxtb/_src/scf/functional/eager.py`. | `E6broute` cutover (or explicit scope restriction of `libcint` / negative `fermi_etemp` to non-SCF or functional paths). | Migrate `test/test_scf/test_scf.py` and `test/test_calculator/test_SP.py` `unroll_mixer`/`experimental` cases to `SCF_MODE_FULL` functional solver tests. |
| **Stage 2 (`E4.6b` — Implicit SCF Retirement)** | Route `SCF_MODE_IMPLICIT` in `src/dxtb/_src/scf/iterator.py` to `run_prepared_functional_implicit` (`src/dxtb/_src/scf/functional/implicit.py`); delete `SelfConsistentFieldImplicit` (`src/dxtb/_src/scf/implicit.py`), `BaseTSCF` (`src/dxtb/_src/scf/base.py`), and `src/dxtb/_src/scf/mixer/` legacy stateful mixers (`Anderson`, `Broyden`, `Simple`). | Stage 1 + routing `SCF_MODE_IMPLICIT` to `run_prepared_functional_implicit`. | Update `test/test_scf/test_implicit.py` to exercise `run_prepared_functional_implicit` / `implicit_root`; retire legacy stateful mixer unit tests in `test/test_scf/test_mixer.py` in favor of `test_functional_anderson.py`. |
| **Stage 3 (`E2-legacy` / `C9` / `B6` — Mutable Component & Packed Cache Retirement)** | Remove `batch_mode > 0` branches from `Model.setup` (`src/dxtb/_src/calculators/model.py`), `_packed_classical_cache`, `ClassicalList.get_cache`, `InteractionList.get_cache`, `Classical.cache_is_latest`, `Interaction.cache_is_latest`, and `reset_all()` hooks on `AutogradCalculator` / `NumericalCalculator`. | `E6broute` cutover on `Calculator` + Stage 2. | Retire legacy cache-mutation assertions in `test/test_classical/` and `test/test_interaction/` while keeping custom subclass extension tests on stateless `get_cache` per-call hooks. |
| **Stage 4 (Experimental W2 Prototype Cleanup)** | Archive or delete experimental `src/dxtb/_src/calculators/batch_implicit.py`, `src/dxtb/_src/calculators/batch_streaming.py`, and `test/test_scf/test_e6c1_w2c9_batch_implicit.py` once the permanent post-refactoring batching optimization replaces them. | Completion of post-refactoring performance phase (`E6c`). | Replace `test_e6c1_w2c9_batch_implicit.py` with tests for the final batching optimization. |

---

## 10. Independent `Luna / high` Review Record

- **Pass 1 (`Luna / high`)**: Identified 1 `BLOCKER` and 1 `MINOR`:
  - `BLOCKER 1`: `_setup_converged_model_and_calculator` in `test/test_calculator/test_b8_e8_functional_properties.py` used `maxiter = 6` for $\text{H}_2$ (`(1, 1)`), which only reached a residual of `1.69e-6` (`GFN1_XTB`) and `3.00e-7` (`GFN2_XTB`) instead of `< 1e-12`. **Resolved** by increasing $\text{H}_2$ `maxiter` to `10` (`residual = 3.64e-16` for GFN1, `0.00` for GFN2).
  - `MINOR 1`: `_normalize_system_for_e6bcompat` in `src/dxtb/_src/calculators/packed_result.py` checked only `strict_e6bcompat` rather than `strict_e6bcompat or system.config.strict or system.config.scf.strict`, and `docs/plan/functional-refactoring-hardening.md` had minor table/test-name discrepancies at lines 52, 97, 99, and 126. **Resolved** in both `packed_result.py` and `functional-refactoring-hardening.md`.
- **Pass 2 (`Luna / high`)**: **`APPROVED` (`NO FINDING`)**.
