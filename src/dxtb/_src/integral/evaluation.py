# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
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
"""Pure setup and evaluation of overlap, multipole, and H0 matrices."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from tad_mctc.batch import deflate

from dxtb import IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.base import normalize_integral_matrix
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.driver.libcint import (
    LibcintIntegralSetup,
    build_dipole_libcint,
    build_overlap_libcint,
    build_quadrupole_libcint,
    setup_libcint,
)
from dxtb._src.integral.driver.pytorch.multipole import (
    build_dipole as build_dipole_pytorch,
    build_quadrupole as build_quadrupole_pytorch,
)
from dxtb._src.integral.driver.pytorch.overlap import (
    build_overlap as build_overlap_pytorch,
)
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    setup_integrals,
)
from dxtb._src.integral.types.dipole import shift_dipole_origin
from dxtb._src.integral.types.quadrupole import (
    make_quadrupole_traceless,
    reduce_quadrupole_9_to_6,
    shift_quadrupole_origin,
)
from dxtb._src.integral.utils import snorm
from dxtb._src.param import ParamModule
from dxtb._src.typing import Tensor
from dxtb._src.xtb.h0 import H0Setup, build_hcore

__all__ = [
    "IntegralSetup",
    "build_integral_matrices",
    "setup_integral_evaluation",
]


@dataclass(frozen=True, eq=False)
class IntegralSetup:
    """Composition-dependent builder data selected for one integral backend."""

    driver_type: int
    intlevel: int
    pytorch: PytorchIntegralSetup | None = None
    libcint: LibcintIntegralSetup | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> IntegralSetup:
        """Return backend setup data converted without retaining call state."""
        pytorch = (
            None if self.pytorch is None else self.pytorch.to(device, dtype)
        )
        libcint = None
        if self.libcint is not None:
            libcint_device = (
                torch.device("cpu")
                if self.libcint.force_cpu
                else device
            )
            libcint = replace(
                self.libcint,
                numbers=self.libcint.numbers.to(device=libcint_device),
                ihelp=self.libcint.ihelp.to(device=libcint_device),
                basis_setups=tuple(
                    basis.to(device=libcint_device, dtype=dtype)
                    for basis in self.libcint.basis_setups
                ),
            )
        return replace(self, pytorch=pytorch, libcint=libcint)


def setup_integral_evaluation(
    numbers: Tensor,
    par: ParamModule,
    ihelp: IndexHelper,
    *,
    driver_type: int,
    intlevel: int,
    algorithm: str | None = None,
    force_cpu_for_libcint: bool = True,
) -> IntegralSetup | None:
    """Gather immutable backend data for later per-geometry matrix builds."""
    if intlevel < labels.INTLEVEL_OVERLAP:
        return None

    if driver_type == labels.INTDRIVER_PYTORCH:
        basis = Basis(numbers, par, ihelp)
        pytorch_setup = setup_integrals(
            basis,
            **({} if algorithm is None else {"algorithm": algorithm}),
        )
        return IntegralSetup(
            driver_type=driver_type,
            intlevel=intlevel,
            pytorch=pytorch_setup,
        )

    if driver_type != labels.INTDRIVER_LIBCINT:
        raise ValueError(f"Unknown integral driver '{driver_type}'.")

    device = torch.device("cpu") if force_cpu_for_libcint else numbers.device
    backend_numbers = numbers.to(device=device)
    backend_ihelp = ihelp.to(device=device)
    basis_setups = []

    if ihelp.batch_mode == 0:
        basis_ihelp = backend_ihelp
        basis_numbers = backend_numbers
        basis = Basis(
            basis_numbers,
            par,
            basis_ihelp,
            device=device,
            dtype=par.dtype,
        )
        basis_setups.append(basis.setup_data())
    else:
        for batch_index, batch_numbers in enumerate(numbers):
            if ihelp.batch_mode == 1:
                batch_numbers = deflate(batch_numbers)
            batch_numbers = batch_numbers.to(device=device)
            basis_ihelp = IndexHelper.from_numbers(batch_numbers, par).to(
                device=device
            )
            basis = Basis(
                batch_numbers,
                par,
                basis_ihelp,
                device=device,
                dtype=par.dtype,
            )
            basis_setups.append(basis.setup_data())

    libcint_setup = LibcintIntegralSetup(
        numbers=backend_numbers,
        ihelp=backend_ihelp,
        basis_setups=tuple(basis_setups),
        force_cpu=force_cpu_for_libcint,
    )
    return IntegralSetup(
        driver_type=driver_type,
        intlevel=intlevel,
        libcint=libcint_setup,
    )


def build_integral_matrices(
    setup: IntegralSetup,
    h0_setup: H0Setup,
    positions: Tensor,
    charge: Tensor | float | int,
) -> tuple[IntegralMatrices, Tensor, Tensor]:
    """Build all requested matrices from explicit setup and current inputs."""
    if setup.driver_type == labels.INTDRIVER_PYTORCH:
        if setup.pytorch is None:
            raise RuntimeError("PyTorch integral setup is missing.")
        integral_setup = setup.pytorch
        raw_overlap = build_overlap_pytorch(integral_setup, positions)

        def build_dipole() -> Tensor:
            return build_dipole_pytorch(integral_setup, positions)

        def build_quadrupole() -> Tensor:
            return build_quadrupole_pytorch(integral_setup, positions)

        ihelp = integral_setup.ihelp
        integral_positions = positions
    elif setup.driver_type == labels.INTDRIVER_LIBCINT:
        if setup.libcint is None:
            raise RuntimeError("libcint integral setup is missing.")
        call_data = setup_libcint(setup.libcint, positions)
        raw_overlap = build_overlap_libcint(call_data)

        def build_dipole() -> Tensor:
            return build_dipole_libcint(call_data)

        def build_quadrupole() -> Tensor:
            return build_quadrupole_libcint(call_data)

        ihelp = call_data.ihelp
        integral_positions = positions.to(device=raw_overlap.device)
    else:
        raise ValueError(f"Unknown integral driver '{setup.driver_type}'.")

    norm = snorm(raw_overlap)
    overlap = normalize_integral_matrix(raw_overlap, norm)
    dipole = None
    quadrupole = None

    if setup.intlevel >= labels.INTLEVEL_DIPOLE:
        raw_dipole = normalize_integral_matrix(build_dipole(), norm)
        if setup.intlevel >= labels.INTLEVEL_QUADRUPOLE:
            if setup.intlevel > labels.INTLEVEL_MAX:
                raise RuntimeError(
                    f"An integral level of '{setup.intlevel}' is requested, "
                    "but the quadrupole integral is the highest supported "
                    "moment."
                )
            raw_quadrupole = normalize_integral_matrix(
                build_quadrupole(), norm
            )
            quadrupole = reduce_quadrupole_9_to_6(raw_quadrupole)
            orbital_positions = ihelp.spread_atom_to_orbital(
                integral_positions, dim=-2, extra=True
            )
            quadrupole = shift_quadrupole_origin(
                quadrupole, raw_dipole, overlap, orbital_positions
            )
            dipole = shift_dipole_origin(
                raw_dipole, overlap, orbital_positions
            )
            quadrupole = make_quadrupole_traceless(quadrupole)
        else:
            orbital_positions = ihelp.spread_atom_to_orbital(
                integral_positions, dim=-2, extra=True
            )
            dipole = shift_dipole_origin(
                raw_dipole, overlap, orbital_positions
            )

    target_device = positions.device
    overlap = overlap.to(device=target_device)
    if dipole is not None:
        dipole = dipole.to(device=target_device)
    if quadrupole is not None:
        quadrupole = quadrupole.to(device=target_device)

    hcore, refocc = build_hcore(h0_setup, positions, overlap, charge=charge)
    matrices = IntegralMatrices(
        hcore=hcore,
        overlap=overlap,
        dipole=dipole,
        quadrupole=quadrupole,
    )
    return matrices, refocc, norm
