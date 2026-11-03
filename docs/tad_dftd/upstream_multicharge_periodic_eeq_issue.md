# Upstream Corrective Specification: Safe Partial-Periodic EEQ Handling in `tad-multicharge`

- **Target upstream repositories**: `grimme-lab/multicharge` and `tad-mctc/tad-multicharge`
- **Observed upstream version in `dxtb` environments**: `tad-multicharge == 0.8.0`
- **Status**: Tracked upstream — 1D and 2D periodic EEQ electrostatics are being implemented upstream in Fortran `multicharge` and will be updated in `tad-multicharge` separately.
- **Classification in `marvinfriede/dxtb` (`tad_dftd` / `tad_dftd.ase`)**: **Inherited upstream dependency behavior** — `tad_dftd` delegates automatic EEQ evaluation directly to `tad-multicharge` via `tad_dftd.charge.get_charges` -> `tad_dftd.charge.eeq.get_eeq_charges` -> `tad_multicharge.model.eeq.EEQModel`, while `tad_dftd.ase` normalizes `pbc=[False, False, False]` cells to `lattice=None, periodic=None` (`DFTD._build_structure` in `src/tad_dftd/ase.py`).

---

## 1. Problem Description

### 1.1 Partial-Periodicity Guard Bypassed Under `vmap` and `torch.compile`

In `tad_multicharge/model/eeq.py` (`EEQModel.get_coulomb_matrix`, lines 346–369 in `v0.8.0`), the Coulomb matrix dispatch branches solely on `if structure.lattice is None:`:

```python
if structure.lattice is None:
    amat = _coulomb_matrix_0d(
        structure.numbers,
        structure.positions,
        radii,
        cutoff=cutoff,
    )
else:
    periodic = structure.periodic
    if periodic is None:
        periodic = torch.tensor(
            [True, True, True],
            dtype=torch.bool,
            device=structure.positions.device,
        )
    if (
        not is_functorch_tensor(periodic)
        and not is_compiling()
        and not bool(periodic.all())
    ):
        raise NotImplementedError(
            "Periodic EEQ is currently only implemented for fully "
            "3D-periodic systems (periodic = [True, True, True])."
        )
    amat = coulomb_matrix_3d(...)
```

This creates two concrete defects:

1. **Silent 3D Ewald evaluation for 1D/2D periodic structures under `vmap` or `torch.compile`**:
   - When `periodic` is batched under `torch.func.vmap` (`is_functorch_tensor(periodic)` is `True`) or traced under `torch.compile` (`is_compiling()` is `True`), the `NotImplementedError` guard is completely bypassed.
   - `coulomb_matrix_3d` in `tad_multicharge/model/_ewald.py` ignores `periodic` altogether, computing the 3D cell volume $V = |\det(L)|$, 3D real-space lattice translations (`get_translations(lattice, ...)` along all 3 lattice vectors), and 3D reciprocal-space lattice vectors $G = 2\pi (L^{-1})^T$.
   - Consequently, 1D or 2D periodic systems (e.g., `periodic = [True, False, False]` or `[True, True, False]`) with a non-singular 3×3 bounding box silently receive **finite, unphysical 3D-periodic Ewald EEQ charges, energies, and derivatives** instead of failing or poisoning the output.
2. **Edge case: `lattice` provided with `periodic = [False, False, False]` (`0D-with-lattice`)**:
   - Because dispatch checks `if structure.lattice is None:` rather than `periodic`, passing an isolated molecule inside a dummy bounding box (`lattice != None`, `periodic = [False, False, False]`) enters the `else:` branch.
   - In eager mode, it raises `NotImplementedError("Periodic EEQ is currently only implemented for fully 3D-periodic systems...")` instead of evaluating the 0D Coulomb matrix `_coulomb_matrix_0d`.
   - Under `vmap` or `torch.compile`, it silently evaluates **3D-periodic Ewald summation** on a non-periodic molecule!

---

## 2. Actual Automatic-EEQ Dispatch in `tad_dftd` and `tad_dftd.ase`

- **Native `tad_dftd` automatic-EEQ call chain**:
  1. `tad_dftd.dftd(structure, param, model="d4" | "d4s" | "d4srev")` dispatches to `dftd4` / `dftd4s` / `dftd4srev`, which instantiates `D4Model` / `D4SModel` / `D4SrevModel` (`src/tad_dftd/model/d4.py`).
  2. When external atomic partial charges `q` are not provided (`q is None`), `D4Model.environment` (and `D4SModel.environment` / `D4SrevModel.environment`) calls `tad_dftd.charge.get_charges(structure.numbers, structure.positions, charge, qmod=self.qmodel, lattice=structure.lattice, periodic=structure.periodic, cutoff=self.cutoff.cn_eeq)`.
  3. `tad_dftd.charge.get_charges` (`src/tad_dftd/charge/__init__.py`) validates element numbers and delegates to `tad_dftd.charge.eeq.get_eeq_charges` (`src/tad_dftd/charge/eeq.py`), which constructs a `tad_mctc.io.structure.Structure` with the passed `lattice` and `periodic` tensors and invokes `tad_multicharge.model.eeq.EEQModel.param2019(...)(struct, return_energy=False)`.
  4. Native `tad_dftd` does **not** perform separate `periodic` pre-validation or strip `lattice` in `src/tad_dftd/disp.py`; all `lattice` / `periodic` branching and the eager `NotImplementedError` check happen inside `tad_multicharge.model.eeq.EEQModel.get_coulomb_matrix`.
- **ASE `tad_dftd.ase` automatic-EEQ call chain**:
  1. `DFTD._build_structure` (`src/tad_dftd/ase.py`) inspects `pbc = [bool(x) for x in atoms.get_pbc()]` and sets `has_any_pbc = any(pbc)`.
  2. When `has_any_pbc` is `False` (`pbc = [False, False, False]`), `DFTD._build_structure` sets `lattice_t = None` and `periodic_t = None` even if `atoms.cell` is non-zero, so `0D-with-cell` ASE `Atoms` objects always reach `_coulomb_matrix_0d` in `tad-multicharge`.
  3. When `has_any_pbc` is `True` and `charges_key is None`, `DFTD.calculate` runs in eager mode and passes `lattice_t` and `periodic_t` to `tad_dftd.dftd`, where `tad_multicharge.model.eeq.EEQModel.get_coulomb_matrix` raises `NotImplementedError` for 1D or 2D `pbc`.
- **Upstream resolution**:
  - Genuine 1D and 2D periodic EEQ electrostatics are being implemented upstream in Fortran `grimme-lab/multicharge` and will be ported to `tad-mctc/tad-multicharge` in a separate upstream update.

---

## 3. Recommended Upstream Fix Strategy (`tad-mctc/tad-multicharge`)

Do not implement genuine 1D/2D Ewald summation in this corrective patch. Instead, combine eager exception raising with a **graph-compatible fail-safe (NaN poisoning) and 0D/3D `torch.where` dispatch** so that unsupported 1D/2D periodicity can never silently produce finite 3D Ewald numbers under `vmap`, `jacrev`, `jacfwd`, `torch.compile`, or higher-order derivatives:

1. **Eager validation (when not transforming `periodic` and not compiling)**:
   - If `not bool(periodic.any())`: route directly to `_coulomb_matrix_0d(structure.numbers, structure.positions, radii, cutoff=cutoff)` even when `structure.lattice is not None` (`0D-with-lattice`).
   - If `bool(periodic.any()) and not bool(periodic.all())`: raise `NotImplementedError("Periodic EEQ is currently only implemented for 0D (periodic = [False, False, False]) and fully 3D-periodic systems (periodic = [True, True, True]).")`.
2. **Graph-compatible dispatch and fail-safe (under `vmap` over `periodic` or `torch.compile`)**:
   - Compute boolean scalar masks (shape `(...)` matching batch prefix):
     - `is_0d = ~periodic.any(dim=-1)`
     - `is_3d = periodic.all(dim=-1)`
     - `is_partial = periodic.any(dim=-1) & (~is_3d)`
   - When `structure.lattice is not None` in a traced/vmapped context:
     - Provide a safe fallback lattice for `is_0d` entries (e.g., `torch.where(is_0d.unsqueeze(-1).unsqueeze(-1), torch.eye(3, dtype=lattice.dtype, device=lattice.device), lattice)`) so `coulomb_matrix_3d` does not divide by zero on zero-padded 0D lattices in a heterogeneous batch.
     - Select between `_coulomb_matrix_0d` and `coulomb_matrix_3d` for valid `is_0d` vs `is_3d` batch elements using `torch.where(is_0d.unsqueeze(-1).unsqueeze(-1), amat_0d, amat_3d)`.
     - **Mask any partial-periodic (`is_partial`) elements with `torch.nan`**:
       ```python
       amat = torch.where(
           is_partial.unsqueeze(-1).unsqueeze(-1),
           torch.full_like(amat, float("nan")),
           amat,
       )
       ```
   - Because `amat` enters `solve(...)`, any `is_partial` structure deterministically yields `NaN` charges, `NaN` energies, and `NaN` first- and higher-order derivatives under `vmap`, `jacrev`, `jacfwd`, and `torch.compile`, rather than silently returning finite 3D Ewald results.

---

## 4. Required Upstream Verification Matrix

The upstream PR in `tad-mctc/tad-multicharge` must verify all five periodicity configurations across all six execution/transformation modes:

| Periodicity Case | `lattice` | `periodic` | Eager | `vmap` | `jacrev` | `jacfwd` | `torch.compile` | Higher-Order (`jacrev(jacrev)`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`0D`** | `None` | `None` | Finite 0D | Finite 0D | Finite 0D | Finite 0D | Finite 0D | Finite 0D |
| **`0D-with-lattice`** | `3×3` tensor | `[False, False, False]` | Finite 0D (matches `lattice=None`) | Finite 0D | Finite 0D (`d/d(lattice) == 0`) | Finite 0D (`d/d(lattice) == 0`) | Finite 0D | Finite 0D |
| **`1D`** | `3×3` tensor | `[True, False, False]` | Raises `NotImplementedError` | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) |
| **`2D`** | `3×3` tensor | `[True, True, False]` | Raises `NotImplementedError` | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) | Raises or `NaN` (never finite 3D) |
| **`3D`** | `3×3` tensor | `[True, True, True]` | Finite 3D Ewald | Finite 3D Ewald | Finite 3D Ewald | Finite 3D Ewald | Finite 3D Ewald | Finite 3D Ewald |
