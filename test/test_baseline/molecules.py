# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Molecule set of the reference data (T0.2).

Inputs only; the reference files store their own copy of the inputs, so they
can be compared without this module. Positions are in bohr.

The geometries of benzene, glycine, OH-, NH4+ and Fe(CO)5 were relaxed with
GFN2-xTB (dxtb, libcint driver, max. gradient < 1e-4 Eh/bohr). The others
are taken from ``tad_mctc.data.molecules`` (tad-mctc 0.7.0).

The conformer batch consists of the lysine structure ``LYS_xao`` and seven
copies with Gaussian displacements (0.1 bohr, fixed seed). They stand in for
conformers: same composition, different geometries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from tad_mctc.batch import pack

from ..molecules import mols

__all__ = ["System", "SYSTEMS", "get_system"]


@dataclass(frozen=True)
class System:
    """Input of one reference calculation."""

    name: str
    """Name of the system (file name of the reference)."""

    group: str
    """Group of the molecule set (see T0.2)."""

    numbers: torch.Tensor
    """Atomic numbers, ``(nat,)`` or ``(nbatch, nat)``."""

    positions: torch.Tensor
    """Positions in bohr (float64), ``(nat, 3)`` or ``(nbatch, nat, 3)``."""

    charge: torch.Tensor
    """Total charge, ``()`` or ``(nbatch,)`` (float64)."""

    spin: torch.Tensor | None = None
    """Number of unpaired electrons, ``None`` for the default."""

    batch_mode: int = 0
    """dxtb batch mode (0: single, 1: padded, 2: conformers)."""

    note: str = field(default="")
    """Free text stored with the reference."""

    @property
    def batched(self) -> bool:
        """Whether the system is a batch."""
        return self.batch_mode > 0


_DD = {"dtype": torch.float64}

# fmt: off
_RELAXED: dict[str, dict] = {
    "C6H6": {
        "numbers": [6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
        "positions": [
            [+2.6164731541, +0.0000000002, -0.0000000000],
            [+1.3082365762, +2.2659322195, +0.0000000000],
            [-1.3082365762, +2.2659322194, +0.0000000000],
            [-2.6164731539, +0.0000000002, -0.0000000000],
            [-1.3082365766, -2.2659322197, -0.0000000000],
            [+1.3082365765, -2.2659322198, -0.0000000000],
            [+4.6582615272, -0.0000000000, +0.0000000000],
            [+2.3291307637, +4.0341728202, +0.0000000000],
            [-2.3291307637, +4.0341728202, +0.0000000000],
            [-4.6582615272, -0.0000000000, -0.0000000000],
            [-2.3291307637, -4.0341728202, +0.0000000000],
            [+2.3291307637, -4.0341728202, +0.0000000000],
        ],
    },
    "glycine": {
        "numbers": [6, 6, 8, 8, 1, 7, 1, 1, 1, 1],
        "positions": [
            [-1.2906466594, -0.5067843549, -0.0000000568],
            [+1.5797314521, -0.4052715552, +0.0000000058],
            [+2.8032191515, +1.5085957790, -0.0000000017],
            [+2.6207820481, -2.7084994438, -0.0000000004],
            [+4.4514664880, -2.5888176258, -0.0000000002],
            [-2.4956379335, +1.9448244441, +0.0000002727],
            [-1.9375188940, +2.9453654854, +1.5390230853],
            [-1.9375187962, +2.9453656258, -1.5390233326],
            [-1.8969384361, -1.5673891780, +1.6677239982],
            [-1.8969384205, -1.5673891767, -1.6677239703],
        ],
    },
    "OH-": {
        "numbers": [8, 1],
        "positions": [
            [-0.0000000000, -0.0000000000, -0.9248098731],
            [+0.0000000000, +0.0000000000, +0.9248098731],
        ],
    },
    "NH4+": {
        "numbers": [7, 1, 1, 1, 1],
        "positions": [
            [-0.0000000000, -0.0000000000, +0.0000000000],
            [+1.1233059255, +1.1233059255, +1.1233059255],
            [-1.1233059255, -1.1233059255, +1.1233059255],
            [-1.1233059255, +1.1233059255, -1.1233059255],
            [+1.1233059255, -1.1233059255, -1.1233059255],
        ],
    },
    "Fe(CO)5": {
        "numbers": [26, 6, 8, 6, 8, 6, 8, 6, 8, 6, 8],
        "positions": [
            [-0.0000038966, -0.0000000407, +0.0000359416],
            [-0.0000000213, -0.0000000002, +3.3494795313],
            [+0.0000001493, +0.0000000016, +5.5072354459],
            [-0.0000000218, -0.0000000002, -3.3494862512],
            [+0.0000001480, +0.0000000015, -5.5072542121],
            [+3.4067333543, +0.0000000138, -0.0000013121],
            [+5.5645593513, -0.0000000045, -0.0000021721],
            [-1.7033652431, +2.9503174463, -0.0000013120],
            [-2.7822792914, +4.8190483451, -0.0000021736],
            [-1.7033652498, -2.9503174302, -0.0000013120],
            [-2.7822792789, -4.8190483325, -0.0000021736],
        ],
    },
}
# fmt: on


def _mol(name: str) -> tuple[torch.Tensor, torch.Tensor]:
    if name in _RELAXED:
        numbers = torch.tensor(_RELAXED[name]["numbers"])
        positions = torch.tensor(_RELAXED[name]["positions"], **_DD)
    else:
        numbers = mols[name]["numbers"].clone()
        positions = mols[name]["positions"].to(**_DD).clone()
    return numbers, positions


def _single(
    name: str, group: str, charge: int = 0, spin: int | None = None
) -> System:
    numbers, positions = _mol(name)
    return System(
        name=name,
        group=group,
        numbers=numbers,
        positions=positions,
        charge=torch.tensor(float(charge), **_DD),
        spin=None if spin is None else torch.tensor(float(spin), **_DD),
    )


def _padded() -> System:
    names, charges = ["H2O", "NH4+", "glycine", "C6H6"], [0.0, 1.0, 0.0, 0.0]
    numbers, positions = zip(*(_mol(n) for n in names))
    return System(
        name="padded4",
        group="padded batch",
        numbers=pack(list(numbers)),
        positions=pack(list(positions)),
        charge=torch.tensor(charges, **_DD),
        batch_mode=1,
        note="padded batch of " + ", ".join(names),
    )


def _conformers(nconf: int = 8, sigma: float = 0.1, seed: int = 1) -> System:
    numbers, positions = _mol("LYS_xao")
    gen = torch.Generator().manual_seed(seed)
    disp = torch.randn((nconf - 1, *positions.shape), generator=gen, **_DD)
    confs = torch.cat([positions.unsqueeze(0), positions + sigma * disp])
    return System(
        name="LYS_xao-conformers",
        group="conformer batch",
        numbers=numbers.unsqueeze(0).expand(nconf, -1).clone(),
        positions=confs,
        charge=torch.zeros(nconf, **_DD),
        batch_mode=2,
        note=(
            f"LYS_xao and {nconf - 1} copies with Gaussian displacements "
            f"(sigma={sigma} bohr, torch.Generator seed {seed})"
        ),
    )


SYSTEMS: dict[str, System] = {
    s.name: s
    for s in [
        _single("H2O", "small organic"),
        _single("CH4", "small organic"),
        _single("C6H6", "small organic"),
        _single("glycine", "small organic"),
        _single("OH-", "charged", charge=-1),
        _single("NH4+", "charged", charge=1),
        _single("NO2", "open shell", spin=1),
        _single("Fe(CO)5", "transition metal"),
        _single("PbH4-BiH3", "heavy main group"),
        _padded(),
        _conformers(),
    ]
}
"""All systems of the reference set, by name."""


def get_system(name: str) -> System:
    """Return a system of the reference set by name."""
    return SYSTEMS[name]
