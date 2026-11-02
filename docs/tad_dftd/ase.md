# ASE Calculator Interface (`tad_dftd.ase`)

`tad_dftd.ase` provides a native, Fortran-free [ASE](https://wiki.fysik.dtu.dk/ase/) `Calculator` interface for the unified PyTorch dispersion library `tad_dftd`. It follows the keyword and parameter conventions of the upstream Grimme-lab Python `dftd.ase` wrapper while adopting standard PyTorch MLIP calculator idioms (`device`, `dtype`, `compute_stress`, and autograd-derived forces and stress).

ASE remains an **optional** dependency of `tad_dftd`: importing `tad_dftd` and calling `tad_dftd.dftd` does not require ASE. Install the optional extra with:

```bash
pip install "dxtb[ase]"
```

---

## Exported Calculators

| Class | Default `model` | Description |
| :--- | :--- | :--- |
| `DFTD` | `"d4"` | Generic dispersion calculator supporting all implemented models (`D1`, `D2`, `D3`, `D3S`, `D4`, `D4S`, `D4Srev`) |
| `DFTD4` | `"d4"` | Convenience subclass defaulting to DFT-D4 (rational Becke-Johnson 2B + ATM 3B damping when `method` is given) |
| `DFTD3` | `"d3"` | Convenience subclass defaulting to DFT-D3 (rational Becke-Johnson 2B damping without ATM 3B unless requested) |

### Supported Model Identifiers (`model`)

Both canonical string names and Fortran `dftd` integer model IDs (`DispersionModelId`) are accepted:

| Model | String Name | Integer ID (`DispersionModelId`) | Status |
| :--- | :--- | :--- | :--- |
| DFT-D1 | `"d1"` | `10` | Supported |
| DFT-D2 | `"d2"` | `20` | Supported |
| DFT-D3 | `"d3"` | `30` | Supported |
| DFT-D3S | `"d3s"` | `31` | Supported |
| DFT-D4 | `"d4"` | `40` | Supported |
| DFT-D4S | `"d4s"` | `41` | Supported |
| DFT-D4Srev | `"d4srev"` | `42` | Supported (`params_tweaks` required) |
| DFT-D5 | `"d5"` | `50` | Excluded (`NotImplementedError`) |

---

## Usage Examples

### 1. Basic Dispersion Calculation

```python
from ase.build import molecule
from tad_dftd.ase import DFTD

atoms = molecule("H2O")
atoms.calc = DFTD(method="PBE", model="d4")

energy = atoms.get_potential_energy()      # eV (float)
atom_energies = atoms.get_potential_energies()  # eV, shape (N,)
forces = atoms.get_forces()                # eV / Angstrom, shape (N, 3)
```

### 2. D3(BJ) Calculation

```python
from ase.build import molecule
from tad_dftd.ase import DFTD3

atoms = molecule("H2O")
atoms.calc = DFTD3(method="PBE")

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### 3. D3 with Explicit Damping

Use `damping_hint` with keys `"2b"` and/or `"3b"` (or pass a `tad_dftd.Damping` container directly) to select non-default damping functions or to enable/disable the three-body Axilrod-Teller-Muto (ATM) term. Both string names/aliases and `TwoBodyDamping` / `ThreeBodyDamping` / `Damping` instances are supported:

```python
from ase.build import molecule
from tad_dftd import Damping
from tad_dftd.ase import DFTD3
from tad_dftd.damping import RationalTwoBody, ZeroThreeBodyD3, ZeroTwoBody

atoms = molecule("H2O")

# D3(0) zero damping (by name or damping object):
atoms.calc = DFTD3(method="PBE", damping_hint={"2b": "zero"})
# Or equivalently:
# atoms.calc = DFTD3(method="PBE", damping_hint={"2b": ZeroTwoBody()})
e_d3_zero = atoms.get_potential_energy()

# D3(BJ)-ATM (enabling three-body dispersion on D3):
atoms.calc = DFTD3(
    method="PBE",
    damping_hint=Damping(two=RationalTwoBody(), three=ZeroThreeBodyD3()),
)
e_d3_atm = atoms.get_potential_energy()
```

### 4. Explicit Parameters (`params_tweaks`)

Provide explicit damping parameters via `params_tweaks` either as standalone parameters (`method=None`) or as selective overrides on top of a functional lookup:

```python
from ase.build import molecule
from tad_dftd.ase import DFTD3

atoms = molecule("H2O")
atoms.calc = DFTD3(
    params_tweaks={
        "s6": 1.0,
        "s8": 0.7875,
        "a1": 0.4289,
        "a2": 4.4407,
    }
)
energy = atoms.get_potential_energy()
```

### 5. D4 with External Atomic Partial Charges (`charges_key`)

For charge-dependent models (`D4`, `D4S`, `D4Srev`), atomic partial charges can be supplied via an entry in `atoms.arrays` named by `charges_key`, bypassing internal EEQ evaluation:

```python
import numpy as np
from ase.build import molecule
from tad_dftd.ase import DFTD

atoms = molecule("H2O")
atoms.set_array("my_charges", np.array([-0.8, 0.4, 0.4]))

atoms.calc = DFTD(
    method="PBE",
    model="d4",
    charges_key="my_charges",
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

When `charges_key` is active, `DFTD.check_state()` tracks changes to `atoms.arrays[charges_key]` (including in-place array modifications) and automatically invalidates cached results.

### 6. Periodic Calculation and Stress Tensor

```python
from ase.build import bulk
from tad_dftd.ase import DFTD

atoms = bulk("Si", "diamond", a=5.43)
atoms.calc = DFTD(
    method="PBE",
    model="d4",
    realspace_cutoff={
        "disp2": 20.0,
        "disp3": 12.0,
        "cn": 12.0,
        "cn_eeq": 15.0,
    },
)

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
stress = atoms.get_stress()  # Voigt 6-vector (xx, yy, zz, yz, xz, xy) in eV / Angstrom^3
```

### 7. Combining with Another ASE Calculator

Combine `DFTD` with any baseline ASE calculator using `SumCalculator` or `DFTD.add_calculator(...)`:

```python
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.calculators.mixing import SumCalculator
from tad_dftd.ase import DFTD

atoms = molecule("H2O")
atoms.calc = SumCalculator([EMT(), DFTD(method="PBE", model="d4")])
# Or equivalently:
# atoms.calc = DFTD(method="PBE", model="d4").add_calculator(EMT())

energy = atoms.get_potential_energy()
forces = atoms.get_forces()
```

### 8. PyTorch Device and Dtype Selection

```python
import torch
from ase.build import molecule
from tad_dftd.ase import DFTD

atoms = molecule("H2O")
atoms.calc = DFTD(
    method="PBE",
    model="d4",
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float64,
)

energy = atoms.get_potential_energy()
```

---

## Unit Conventions

All inputs and outputs at the ASE boundary use standard ASE units, while internal `tad_dftd` kernels operate in atomic units (`Bohr` and `Hartree`), converted using `ase.units.Bohr` and `ase.units.Hartree`:

| Quantity | ASE Unit | Internal `tad_dftd` Unit | Conversion |
| :--- | :--- | :--- | :--- |
| `atoms.positions` | $\text{\AA}$ | $\text{Bohr}$ | $r_{\text{Bohr}} = r_{\text{\AA}} / \text{Bohr}$ |
| `atoms.cell` | $\text{\AA}$ | $\text{Bohr}$ | $L_{\text{Bohr}} = L_{\text{\AA}} / \text{Bohr}$ |
| `realspace_cutoff` | $\text{\AA}$ | $\text{Bohr}$ | $R_{\text{Bohr}} = R_{\text{\AA}} / \text{Bohr}$ |
| `energy`, `free_energy` | $\text{eV}$ | $\text{E}_{\text{h}}$ ($\text{Hartree}$) | $E_{\text{eV}} = E_{\text{Eh}} \cdot \text{Hartree}$ |
| `energies` | $\text{eV}$ | $\text{E}_{\text{h}}$ ($\text{Hartree}$) | $E_{i,\text{eV}} = E_{i,\text{Eh}} \cdot \text{Hartree}$ |
| `forces` | $\text{eV}/\text{\AA}$ | $\text{E}_{\text{h}}/\text{Bohr}$ | $F_{\text{eV/\AA}} = -\nabla_{r_{\text{Bohr}}} E_{\text{Eh}} \cdot (\text{Hartree}/\text{Bohr})$ |
| `stress` (Voigt 6-vector) | $\text{eV}/\text{\AA}^3$ | $\text{E}_{\text{h}}$ (virial $\partial E/\partial \varepsilon$) | $\sigma_{\text{eV/\AA}^3} = (\partial E_{\text{Eh}}/\partial \varepsilon) \cdot (\text{Hartree} / V_{\text{\AA}^3})$ |
| `charge`, `charges_key` | $e$ | $e$ | $1:1$ |

---

## Limitations and Scope

- **Excluded upstream models**: `D5` and `EEQBC` / `EEQBC2025` are unfinished upstream and raise `NotImplementedError`.
- **`D4Srev` parameterization**: `D4Srev` does not have functional lookup tables (`method="..."` is rejected); supply explicit damping parameters via `params_tweaks`.
- **1D and 2D periodic charge-dependent models**: Automatic periodic EEQ is implemented for 0D molecules and 3D periodic crystals (`pbc=[True, True, True]`). For 1D or 2D periodic systems with `D4`, `D4S`, or `D4Srev`, supply external atomic partial charges via `charges_key`.
- **Periodic stress**: Analytical stress is only defined for non-degenerate 3D periodic cells (`pbc=[True, True, True]`) when `compute_stress=True`. Requesting stress on 0D, 1D, or 2D structures or when `compute_stress=False` raises `PropertyNotImplementedError` (also a `ValueError` subclass).

---

## Differences from Upstream Fortran `dftd.ase`

1. **No compiled Fortran extension required**: `tad_dftd.ase` calls `tad_dftd.dftd` directly in PyTorch and computes analytical forces and stress via `torch.autograd.grad`.
2. **No `cache_api` keyword**: The Fortran wrapper's `cache_api` option (which cached C-API handles across calls) is intentionally omitted because `tad_dftd` executes statelessly on PyTorch tensors without a Fortran C-API handle.
3. **PyTorch execution controls**: Adds `device` (`"cpu"`, `"cuda"`), `dtype` (`torch.float64`, `torch.float32`), `qmodel` (`"eeq"`), and `compute_stress` (`True`/`False`) keywords.
