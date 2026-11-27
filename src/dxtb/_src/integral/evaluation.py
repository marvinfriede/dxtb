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
"""Pure setup and evaluation of the overlap and multipole integrals."""

from __future__ import annotations

import torch
from tad_mctc.tree import Node, child, context

from dxtb import IndexHelper, labels
from dxtb._src.basis.bas import Basis
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

__all__ = [
    "IntegralSetup",
    "evaluate_integrals",
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
            torch.device("cpu") if self.backend is None else self.backend.device
        )
        return {"logical_device": device}

    def _validate(self) -> None:
        """Require a supported level and a backend matching the driver."""
        if not labels.INTLEVEL_OVERLAP <= self.intlevel <= labels.INTLEVEL_MAX:
            raise ValueError(
                f"Unsupported integral level '{self.intlevel}'. The "
                "quadrupole integral is the highest supported moment."
            )
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
    if driver_type not in (labels.INTDRIVER_PYTORCH, labels.INTDRIVER_LIBCINT):
        raise ValueError(f"Unknown integral driver '{driver_type}'.")

    target_device = (
        numbers.device if device is None else normalize_device(device)
    )
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


def _make_quadrupole_traceless(quadrupole: Tensor) -> Tensor:
    r"""
    Convert the ket-centered quadrupole integral into the traceless form used
    by the xTB methods (as in tblite).

    .. math::

        \Theta_{ab} = \frac{3}{2} Q_{ab} - \frac{1}{2} \delta_{ab}
                      \left( Q_{xx} + Q_{yy} + Q_{zz} \right)

    Parameters
    ----------
    quadrupole : Tensor
        Six-component quadrupole integral in the order xx, yx, yy, zx, zy, zz
        (shape: ``(..., 6, nao, nao)``).

    Returns
    -------
    Tensor
        Traceless quadrupole integral in the same order and shape.
    """
    # half of the trace: (Q_xx + Q_yy + Q_zz) / 2
    half_trace = 0.5 * (
        quadrupole[..., 0, :, :]
        + quadrupole[..., 2, :, :]
        + quadrupole[..., 5, :, :]
    )

    # 3/2 Q_ab, minus the half trace on the diagonal components only
    return torch.stack(
        [
            1.5 * quadrupole[..., 0, :, :] - half_trace,  # xx
            1.5 * quadrupole[..., 1, :, :],  # yx
            1.5 * quadrupole[..., 2, :, :] - half_trace,  # yy
            1.5 * quadrupole[..., 3, :, :],  # zx
            1.5 * quadrupole[..., 4, :, :],  # zy
            1.5 * quadrupole[..., 5, :, :] - half_trace,  # zz
        ],
        dim=-3,
    )


def evaluate_integrals(
    setup: IntegralSetup,
    positions: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor | None]:
    """
    Evaluate the overlap and the multipole integrals for one geometry.

    Parameters
    ----------
    setup : IntegralSetup
        Backend setup from :func:`setup_integral_evaluation`.
    positions : Tensor
        Cartesian coordinates of all atoms (shape: ``(nat, 3)``).

    Returns
    -------
    tuple[Tensor, Tensor | None, Tensor | None]
        Normalized overlap ``(nao, nao)``, ket-centered dipole
        ``(3, nao, nao)`` and ket-centered traceless quadrupole
        ``(6, nao, nao)`` on the device of ``positions``. Moments above
        ``setup.intlevel`` are ``None``.
    """
    backend = setup.backend
    if isinstance(backend, PytorchIntegralSetup):
        moments = evaluate_pytorch_moments(backend, positions, setup.intlevel)
    elif isinstance(backend, LibcintIntegralSetup):
        moments = evaluate_libcint_moments(backend, positions, setup.intlevel)
    else:
        raise RuntimeError("Integral backend setup is missing.")

    overlap, dipole, quadrupole = moments

    # both drivers return the plain ket-centered quadrupole; the xTB methods
    # use its traceless form
    if quadrupole is not None:
        quadrupole = _make_quadrupole_traceless(quadrupole)

    # libcint evaluates on CPU; return to the device of the positions
    device = positions.device
    return (
        overlap.to(device=device),
        None if dipole is None else dipole.to(device=device),
        None if quadrupole is None else quadrupole.to(device=device),
    )
