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

import torch
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.integral.container import IntegralMatrices
from dxtb._src.integral.driver.libcint import (
    LibcintIntegralSetup,
    LibcintIntegralSetupNode,
    build_dipole_libcint,
    build_overlap_libcint,
    build_quadrupole_libcint,
    setup_libcint,
    setup_libcint_core,
)
from dxtb._src.integral.driver.pytorch.multipole import (
    build_dipole as build_dipole_pytorch,
)
from dxtb._src.integral.driver.pytorch.multipole import (
    build_quadrupole as build_quadrupole_pytorch,
)
from dxtb._src.integral.driver.pytorch.overlap import (
    build_overlap as build_overlap_pytorch,
)
from dxtb._src.integral.driver.pytorch.setup import (
    PytorchIntegralSetup,
    PytorchIntegralSetupNode,
    setup_integrals_core,
)
from dxtb._src.integral.types.dipole import shift_dipole_origin
from dxtb._src.integral.types.quadrupole import (
    make_quadrupole_traceless,
    reduce_quadrupole_9_to_6,
    shift_quadrupole_origin,
)
from dxtb._src.param import ParamModule
from dxtb._src.typing import Tensor
from dxtb._src.utils.tensors import normalize_device
from dxtb._src.xtb.h0 import H0Setup, H0SetupNode, build_hcore

__all__ = [
    "IntegralSetup",
    "build_integral_matrices",
    "setup_integral_evaluation",
]


class IntegralSetup(Node):
    """Authoritative setup Node selected for one integral backend."""

    driver_type: int = context()
    intlevel: int = context()
    logical_device: torch.device | None = context(default=None)
    backend: (
        PytorchIntegralSetupNode
        | LibcintIntegralSetupNode
        | PytorchIntegralSetup
        | LibcintIntegralSetup
        | None
    ) = child(default=None)

    def _normalize(self) -> dict[str, object]:
        if self.logical_device is not None:
            normalized = normalize_device(self.logical_device)
            assert isinstance(normalized, torch.device)
            if normalized != self.logical_device:
                return {"logical_device": normalized}
            return {}
        if isinstance(
            self.backend, (PytorchIntegralSetupNode, LibcintIntegralSetupNode)
        ):
            device = self.backend.index.device
        elif isinstance(self.backend, PytorchIntegralSetup):
            device = self.backend.ihelp.device
        elif isinstance(self.backend, LibcintIntegralSetup):
            device = self.backend.numbers.device
        else:
            device = torch.device("cpu")
        return {"logical_device": device}

    def _validate(self) -> None:
        """Require the selected backend to match its driver."""
        expected_backends = {
            labels.INTDRIVER_PYTORCH: (
                PytorchIntegralSetupNode,
                PytorchIntegralSetup,
            ),
            labels.INTDRIVER_LIBCINT: (
                LibcintIntegralSetupNode,
                LibcintIntegralSetup,
            ),
        }.get(self.driver_type)
        if expected_backends is None:
            raise ValueError(
                f"Unsupported integral driver '{self.driver_type}'."
            )
        if self.backend is not None and not isinstance(
            self.backend, expected_backends
        ):
            raise TypeError(
                "Integral backend Node does not match driver "
                f"'{self.driver_type}'."
            )

    @property
    def device(self) -> torch.device:
        """Return the logical System device while allowing force-CPU leaves."""
        assert self.logical_device is not None
        return self.logical_device

    @property
    def pytorch(
        self,
    ) -> PytorchIntegralSetupNode | PytorchIntegralSetup | None:
        """Return the PyTorch backend setup when selected."""
        if self.driver_type == labels.INTDRIVER_PYTORCH:
            return self.backend  # type: ignore[return-value]
        return None

    @property
    def libcint(
        self,
    ) -> LibcintIntegralSetupNode | LibcintIntegralSetup | None:
        """Return the libcint backend setup when selected."""
        if self.driver_type == labels.INTDRIVER_LIBCINT:
            return self.backend  # type: ignore[return-value]
        return None

    @property
    def index(self) -> IndexHelper | None:
        """Return the IndexHelper owned by the backend setup, if present."""
        if isinstance(
            self.backend, (PytorchIntegralSetupNode, LibcintIntegralSetupNode)
        ):
            return self.backend.index
        if isinstance(
            self.backend, (PytorchIntegralSetup, LibcintIntegralSetup)
        ):
            return self.backend.ihelp
        return None

    def _convert_child(
        self,
        name: str,
        value: object,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> object:
        if isinstance(value, (PytorchIntegralSetup, LibcintIntegralSetup)):
            return value.to(device=device, dtype=dtype)
        return super()._convert_child(name, value, device, dtype)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> IntegralSetup:
        """Return backend setup data converted without retaining call state."""
        result = super().to(device=device, dtype=dtype)
        if device is not None:
            logical_device = normalize_device(device)
            assert isinstance(logical_device, torch.device)
            if result.logical_device != logical_device:
                result = result.replace(logical_device=logical_device)
        return result


def setup_integral_evaluation(
    numbers: Tensor,
    par: ParamModule,
    ihelp: IndexHelper,
    *,
    driver_type: int,
    intlevel: int,
    algorithm: str | None = None,
    force_cpu_for_libcint: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntegralSetup | None:
    """Gather immutable backend data for later per-geometry matrix builds."""
    if intlevel < labels.INTLEVEL_OVERLAP:
        return None
    if ihelp.batch_mode != 0:
        raise ValueError(
            "setup_integral_evaluation requires a single System (batch_mode == 0)."
        )

    target_device = numbers.device if device is None else device
    target_dtype = par.dtype if dtype is None else dtype

    if driver_type == labels.INTDRIVER_PYTORCH:
        backend_numbers = (
            numbers
            if numbers.device == target_device
            else numbers.to(device=target_device)
        )
        backend_ihelp = (
            ihelp
            if ihelp.device == target_device
            else ihelp.to(device=target_device)
        )
        basis = Basis(
            backend_numbers,
            par,
            backend_ihelp,
            device=target_device,
            dtype=target_dtype,
        )
        pytorch_setup = setup_integrals_core(
            basis,
            **({} if algorithm is None else {"algorithm": algorithm}),
        )
        return IntegralSetup(
            driver_type=driver_type,
            intlevel=intlevel,
            logical_device=target_device,
            backend=pytorch_setup,
        )

    if driver_type != labels.INTDRIVER_LIBCINT:
        raise ValueError(f"Unknown integral driver '{driver_type}'.")

    libcint_device = (
        torch.device("cpu") if force_cpu_for_libcint else target_device
    )
    backend_numbers = (
        numbers
        if numbers.device == libcint_device
        else numbers.to(device=libcint_device)
    )
    backend_ihelp = (
        ihelp
        if ihelp.device == libcint_device
        else ihelp.to(device=libcint_device)
    )
    basis = Basis(
        backend_numbers,
        par,
        backend_ihelp,
        device=libcint_device,
        dtype=target_dtype,
    )
    return IntegralSetup(
        driver_type=driver_type,
        intlevel=intlevel,
        logical_device=target_device,
        backend=setup_libcint_core(basis, force_cpu=force_cpu_for_libcint),
    )


def _evaluate_pytorch_moments(
    backend: PytorchIntegralSetup | PytorchIntegralSetupNode,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None, IndexHelper]:
    """Evaluate normalized overlap and ket-centered moments with PyTorch."""
    overlap = build_overlap_pytorch(backend, positions)
    ihelp = (
        backend.index
        if isinstance(backend, PytorchIntegralSetupNode)
        else backend.ihelp
    )
    dipole = None
    quadrupole = None

    if intlevel >= labels.INTLEVEL_DIPOLE:
        raw_dipole = build_dipole_pytorch(backend, positions)
        orbital_positions = ihelp.spread_atom_to_orbital(
            positions, dim=-2, extra=True
        )
        dipole = shift_dipole_origin(raw_dipole, overlap, orbital_positions)
        if intlevel >= labels.INTLEVEL_QUADRUPOLE:
            raw_quadrupole = build_quadrupole_pytorch(backend, positions)
            quadrupole = shift_quadrupole_origin(
                reduce_quadrupole_9_to_6(raw_quadrupole),
                raw_dipole,
                overlap,
                orbital_positions,
            )
            quadrupole = make_quadrupole_traceless(quadrupole)

    return overlap, dipole, quadrupole, ihelp


def _evaluate_libcint_moments(
    backend: LibcintIntegralSetup | LibcintIntegralSetupNode,
    positions: Tensor,
    intlevel: int,
) -> tuple[Tensor, Tensor | None, Tensor | None, IndexHelper]:
    """Evaluate normalized overlap and ket-centered moments with libcint."""
    call_data = setup_libcint(backend, positions)
    overlap = build_overlap_libcint(call_data)
    dipole = None
    quadrupole = None

    if intlevel >= labels.INTLEVEL_DIPOLE:
        # tad-libcint 0.4 returns native ket-centered j/jj moments;
        # jj is already packed as xx, yx, yy, zx, zy, zz.
        dipole = build_dipole_libcint(call_data)
        if intlevel >= labels.INTLEVEL_QUADRUPOLE:
            raw_quadrupole = build_quadrupole_libcint(call_data)
            quadrupole = make_quadrupole_traceless(raw_quadrupole)

    return overlap, dipole, quadrupole, call_data.ihelp


def build_integral_matrices(
    setup: IntegralSetup,
    h0_setup: H0Setup | H0SetupNode,
    positions: Tensor,
    charge: Tensor | float | int,
) -> tuple[IntegralMatrices, Tensor]:
    """Build all requested matrices from explicit setup and current inputs."""
    if setup.intlevel > labels.INTLEVEL_MAX:
        raise RuntimeError(
            f"An integral level of '{setup.intlevel}' is requested, "
            "but the quadrupole integral is the highest supported "
            "moment."
        )

    if setup.driver_type == labels.INTDRIVER_PYTORCH:
        if setup.pytorch is None:
            raise RuntimeError("PyTorch integral setup is missing.")
        overlap, dipole, quadrupole, ihelp = _evaluate_pytorch_moments(
            setup.pytorch, positions, setup.intlevel
        )
    elif setup.driver_type == labels.INTDRIVER_LIBCINT:
        if setup.libcint is None:
            raise RuntimeError("libcint integral setup is missing.")
        overlap, dipole, quadrupole, ihelp = _evaluate_libcint_moments(
            setup.libcint, positions, setup.intlevel
        )
    else:
        raise ValueError(f"Unknown integral driver '{setup.driver_type}'.")

    target_device = positions.device
    overlap = overlap.to(device=target_device)
    if dipole is not None:
        dipole = dipole.to(device=target_device)
    if quadrupole is not None:
        quadrupole = quadrupole.to(device=target_device)

    h0_ihelp = (
        ihelp
        if ihelp.device == target_device
        else ihelp.to(device=target_device)
    )
    hcore, refocc = build_hcore(
        h0_setup, positions, overlap, charge=charge, ihelp=h0_ihelp
    )
    hcore = hcore.to(device=target_device)
    refocc = refocc.to(device=target_device)
    matrices = IntegralMatrices(
        hcore=hcore,
        overlap=overlap,
        dipole=dipole,
        quadrupole=quadrupole,
    )
    return matrices, refocc
