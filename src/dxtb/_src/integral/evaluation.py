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
)
from dxtb._src.integral.driver.libcint import (
    evaluate_moments as evaluate_libcint_moments,
)
from dxtb._src.integral.driver.libcint import setup_libcint
from dxtb._src.integral.driver.pytorch import (
    PytorchIntegralSetup,
)
from dxtb._src.integral.driver.pytorch import (
    evaluate_moments as evaluate_pytorch_moments,
)
from dxtb._src.integral.driver.pytorch import setup_integrals
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
    backend: PytorchIntegralSetup | LibcintIntegralSetup | None = child(
        default=None
    )

    def _normalize(self) -> dict[str, object]:
        if self.logical_device is not None:
            normalized = normalize_device(self.logical_device)
            assert isinstance(normalized, torch.device)
            if normalized != self.logical_device:
                return {"logical_device": normalized}
            return {}
        device = (
            torch.device("cpu")
            if self.backend is None
            else self.backend.index.device
        )
        return {"logical_device": device}

    def _validate(self) -> None:
        """Require the selected backend to match its driver."""
        expected_backend = {
            labels.INTDRIVER_PYTORCH: PytorchIntegralSetup,
            labels.INTDRIVER_LIBCINT: LibcintIntegralSetup,
        }.get(self.driver_type)
        if expected_backend is None:
            raise ValueError(
                f"Unsupported integral driver '{self.driver_type}'."
            )
        if self.backend is not None and not isinstance(
            self.backend, expected_backend
        ):
            raise TypeError(
                "Integral backend Node does not match driver "
                f"'{self.driver_type}'."
            )

    @property
    def device(self) -> torch.device:
        """Return the logical System device while allowing CPU-resident backend leaves."""
        assert self.logical_device is not None
        return self.logical_device

    @property
    def pytorch(self) -> PytorchIntegralSetup | None:
        """Return the PyTorch backend setup when selected."""
        if self.driver_type == labels.INTDRIVER_PYTORCH:
            return self.backend  # type: ignore[return-value]
        return None

    @property
    def libcint(self) -> LibcintIntegralSetup | None:
        """Return the libcint backend setup when selected."""
        if self.driver_type == labels.INTDRIVER_LIBCINT:
            return self.backend  # type: ignore[return-value]
        return None

    @property
    def index(self) -> IndexHelper | None:
        """Return the IndexHelper owned by the backend setup, if present."""
        return None if self.backend is None else self.backend.index

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
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> IntegralSetup | None:
    """
    Gather immutable backend data for later per-geometry matrix builds.

    Note
    ----
    The returned :class:`IntegralSetup` captures setup-time CGTO ``alphas`` and
    ``coeffs`` derived from ``par``. Reusing one setup across geometries
    supports repeated independent backward passes with respect to ``positions``
    without ``retain_graph=True``. For repeated independent backward passes or
    :mod:`torch.func` transforms (``jvp``, ``jacfwd``, ``jacrev``) with respect
    to basis parameters (such as ``element.*.slater``), rebuild a fresh setup
    from the active parameter tensors on each evaluation (for example via
    :meth:`~dxtb._src.calculators.model.Model.setup_functional` or
    :class:`~dxtb.Calculator`).
    """
    if not isinstance(par, ParamModule):
        raise TypeError(
            "setup_integral_evaluation requires a ParamModule instance, "
            f"got {type(par).__name__}."
        )
    if intlevel < labels.INTLEVEL_OVERLAP:
        return None
    if ihelp.batch_mode != 0:
        raise ValueError(
            "setup_integral_evaluation requires a single System "
            "(batch_mode == 0)."
        )
    if driver_type not in (
        labels.INTDRIVER_PYTORCH,
        labels.INTDRIVER_LIBCINT,
    ):
        raise ValueError(f"Unknown integral driver '{driver_type}'.")

    target_device = numbers.device if device is None else device
    target_dtype = par.dtype if dtype is None else dtype
    backend_device = (
        torch.device("cpu")
        if driver_type == labels.INTDRIVER_LIBCINT
        else target_device
    )

    basis = Basis(
        numbers,
        par,
        ihelp,
        device=backend_device,
        dtype=target_dtype,
    )

    backend: PytorchIntegralSetup | LibcintIntegralSetup
    if driver_type == labels.INTDRIVER_PYTORCH:
        backend = setup_integrals(
            basis,
            **({} if algorithm is None else {"algorithm": algorithm}),
        )
    else:
        backend = setup_libcint(basis)

    return IntegralSetup(
        driver_type=driver_type,
        intlevel=intlevel,
        logical_device=target_device,
        backend=backend,
    )


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
        overlap, dipole, quadrupole = evaluate_pytorch_moments(
            setup.pytorch, positions, setup.intlevel
        )
        ihelp = setup.pytorch.index
    elif setup.driver_type == labels.INTDRIVER_LIBCINT:
        if setup.libcint is None:
            raise RuntimeError("libcint integral setup is missing.")
        overlap, dipole, quadrupole = evaluate_libcint_moments(
            setup.libcint, positions, setup.intlevel
        )
        ihelp = setup.libcint.index
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
