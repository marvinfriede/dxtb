# This file is part of tad-dftd.
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
r"""
Damping functions
=================

The damping of the two-body term is a :class:`TwoBodyDamping`, that of the
three-body term a :class:`ThreeBodyDamping`, and the pair of them a
:class:`Damping`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import torch
from tad_mctc.typing import Tensor

from .param import DampingParam, ParamInput, as_damping_param

__all__ = [
    "DFTD_RADIUS_DEFAULTS",
    "Damping",
    "PairData",
    "ThreeBodyDamping",
    "TripleData",
    "TwoBodyDamping",
    "pow6",
    "pow6_pow8",
    "scaled_radius",
    "scaled_radius3",
]


class PairData(NamedTuple):
    """
    What a :class:`TwoBodyDamping` reads about the pairs of atoms.

    Every field has the shape of `distances` or broadcasts against it. The
    pairs that are not kept have positive dummy values, e.g. ones, so that
    evaluating a damping on all entries and masking the output leaves the
    gradient finite.
    """

    distances: Tensor
    """Interatomic distances, in Bohr."""

    qq: Tensor
    r"""Ratio :math:`C_8 / C_6 = 3 \cdot r_4 r_2[i] \cdot r_4 r_2[j]`."""

    c6: Tensor
    """Atomic C6 dispersion coefficients."""

    rvdw: Tensor
    """Pairwise van-der-Waals radii."""

    znum: Tensor
    """Sum of the atomic numbers of the two atoms."""

    rdamp: Tensor | None = None
    r"""Damping radius of the model: :math:`\sqrt{C_8 / C_6}` for D3 and D4
    (computed from `qq` if ``None``)."""


class TripleData(NamedTuple):
    """
    What a :class:`ThreeBodyDamping` reads about the triples of atoms.

    Every field has the shape of `r` or broadcasts against it. As in
    :class:`PairData`, the triples that are not kept have positive dummy
    values.
    """

    r: Tensor
    """Product of the three interatomic distances, ``r_ij * r_ik * r_jk``."""

    r2ij: Tensor
    """Squared distance of the pair ``(i, j)``."""

    r2ik: Tensor
    """Squared distance of the pair ``(i, k)``."""

    r2jk: Tensor
    """Squared distance of the pair ``(j, k)``."""

    rdampij: Tensor
    """Damping radius of the pair ``(i, j)``."""

    rdampik: Tensor
    """Damping radius of the pair ``(i, k)``."""

    rdampjk: Tensor
    """Damping radius of the pair ``(j, k)``."""

    rvdwij: Tensor
    """Van-der-Waals radius of the pair ``(i, j)``."""

    rvdwik: Tensor
    """Van-der-Waals radius of the pair ``(i, k)``."""

    rvdwjk: Tensor
    """Van-der-Waals radius of the pair ``(j, k)``."""

    @property
    def legs(
        self,
    ) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        """``(r2, rdamp)`` of the pairs ij, ik and jk, in this order."""
        return (
            (self.r2ij, self.rdampij),
            (self.r2ik, self.rdampik),
            (self.r2jk, self.rdampjk),
        )


class _Damping:
    """
    Reading a parameter from a :class:`~tad_dftd.damping.DampingParam`, with
    the defaults of this damping.
    """

    defaults: ClassVar[tuple[tuple[str, float], ...]] = ()
    """``(name, value)`` of each parameter that has a default."""

    label: ClassVar[str] = "Damping"
    """Human-readable name of this damping, for error messages."""

    def value(self, param: DampingParam, name: str) -> Tensor | float:
        """
        The value of `name` in `param`, or its default on this damping.

        Raises
        ------
        ValueError
            If `name` is ``None`` in `param` and has no default here.
        """
        v = getattr(param, name)
        if v is not None:
            return v
        for key, default in self.defaults:
            if key == name:
                return default
        raise ValueError(f"{self.label} requires the parameter '{name}'.")


class TwoBodyDamping(_Damping):
    """
    Damping of the two-body term.

    Returns the damped dispersion of each pair divided by its C6: the energy
    of the pair is ``-c6`` times this.
    """

    reads_rvdw: ClassVar[bool] = True
    """Whether the damping reads :attr:`PairData.rvdw`. If ``False``, the
    evaluation of the dispersion fills that field with a placeholder."""

    reads_znum: ClassVar[bool] = True
    """Whether the damping reads :attr:`PairData.znum`. If ``False``, the
    evaluation of the dispersion fills that field with a placeholder."""

    has_d8: ClassVar[bool] = True
    """Whether :meth:`damp` has an eighth-order term when `s8` is set."""

    def __call__(self, pairs: PairData, param: DampingParam) -> Tensor:
        raise NotImplementedError

    def damp(
        self, param: ParamInput, r2: Tensor, rdamp: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Evaluate the bare 2-body damping factors :math:`(d_6, d_8)` as a
        function of squared distance ``r2`` and base damping radius ``rdamp``
        (matching ``get_2b_damp`` in Grimme-lab ``dftd``).
        """
        p = as_damping_param(param)
        mask = r2 > 0
        r = torch.sqrt(torch.where(mask, r2, torch.ones_like(r2)))
        one = torch.ones_like(r)
        zero = torch.zeros_like(r)
        rdamp_safe = torch.where(rdamp > 0, rdamp, one)
        has_s8 = self.has_d8 and (
            p.s8 is not None or any(k == "s8" for k, _ in self.defaults)
        )
        p_d6 = p if has_s8 else p.replace(s8=0.0)
        pairs_d6 = PairData(
            distances=r,
            qq=zero,
            c6=one,
            rvdw=rdamp_safe,
            znum=one,
            rdamp=rdamp_safe,
        )
        d6 = torch.where(mask, self(pairs_d6, p_d6), zero)
        if not has_s8:
            return d6, zero
        pairs_d8 = PairData(
            distances=r,
            qq=one,
            c6=one,
            rvdw=rdamp_safe,
            znum=one,
            rdamp=rdamp_safe,
        )
        d8 = torch.where(mask, self(pairs_d8, p.replace(s6=0.0)), zero)
        return d6, d8


class ThreeBodyDamping(_Damping):
    """
    Damping of the three-body (Axilrod-Teller-Muto) term.

    Returns the damping factor of each triple, including `s9`.
    """

    reads_rdamp: ClassVar[bool] = True
    """Whether the damping reads the ``rdamp*`` fields of :class:`TripleData`.
    If ``False``, the evaluation of the dispersion fills them with
    placeholders."""

    reads_rvdw: ClassVar[bool] = True
    """Whether the damping reads the ``rvdw*`` fields of :class:`TripleData`.
    If ``False``, the evaluation of the dispersion fills them with
    placeholders."""

    def __call__(self, triples: TripleData, param: DampingParam) -> Tensor:
        raise NotImplementedError

    def damp(
        self,
        param: ParamInput,
        r2ij: Tensor,
        r2jk: Tensor,
        r2ik: Tensor,
        rdampij: Tensor,
        rdampjk: Tensor,
        rdampik: Tensor,
    ) -> Tensor:
        """
        Evaluate the 3-body damping factor :math:`s_9 f_{\\text{damp}}` from
        squared distances and damping radii (matching ``get_3b_damp`` in
        Grimme-lab ``dftd``).
        """
        mask = (r2ij > 0) & (r2jk > 0) & (r2ik > 0)
        r2_prod = r2ij * r2jk * r2ik
        r = torch.sqrt(torch.where(mask, r2_prod, torch.ones_like(r2_prod)))
        r2ij_s = torch.where(r2ij > 0, r2ij, torch.ones_like(r2ij))
        r2ik_s = torch.where(r2ik > 0, r2ik, torch.ones_like(r2ik))
        r2jk_s = torch.where(r2jk > 0, r2jk, torch.ones_like(r2jk))
        triples = TripleData(
            r=r,
            r2ij=r2ij_s,
            r2ik=r2ik_s,
            r2jk=r2jk_s,
            rdampij=rdampij,
            rdampik=rdampik,
            rdampjk=rdampjk,
            rvdwij=rdampij,
            rvdwik=rdampik,
            rvdwjk=rdampjk,
        )
        out = self(triples, as_damping_param(param))
        return torch.where(mask, out, torch.zeros_like(out))


@dataclass(frozen=True)
class Damping:
    """
    The damping of the two- and the three-body term.

    Parameters
    ----------
    two : TwoBodyDamping
        Damping of the two-body term.
    three : ThreeBodyDamping | None, optional
        Damping of the three-body term, or ``None`` for no three-body term.
        Defaults to ``None``.
    """

    two: TwoBodyDamping
    three: ThreeBodyDamping | None = None


def pow6(x: Tensor) -> Tensor:
    """``x**6`` from three multiplications."""
    x2 = x * x
    return x2 * x2 * x2


def pow6_pow8(x: Tensor) -> tuple[Tensor, Tensor]:
    """``(x**6, x**8)``, sharing ``x**2`` and ``x**6``."""
    x2 = x * x
    x6 = x2 * x2 * x2
    return x6, x6 * x2


DFTD_RADIUS_DEFAULTS: tuple[tuple[str, float], ...] = (
    ("a1", 1.0),
    ("a2", 0.0),
)
"""Default ``a1`` and ``a2`` of the dampings ported from ``dftd``: the
damping radius of the model as it is."""


def scaled_radius(
    damping: _Damping, param: DampingParam, rdamp: Tensor | None,
    *,
    qq: Tensor | None = None,
) -> Tensor:
    """
    ``a1 * rdamp + a2``, the critical radius of the rational damping and the
    radius adaptation of ``dftd``.
    """
    if rdamp is None:
        assert qq is not None
        rdamp = torch.sqrt(qq)
    return damping.value(param, "a1") * rdamp + damping.value(param, "a2")


def scaled_radius3(
    damping: _Damping, param: DampingParam, rdamp: Tensor
) -> Tensor:
    """
    ``a13 * rdamp + a23`` for three-body dampings, falling back to ``a1`` and
    ``a2`` when ``a13`` or ``a23`` is not set.
    """
    a13 = param.a13 if param.a13 is not None else damping.value(param, "a1")
    a23 = param.a23 if param.a23 is not None else damping.value(param, "a2")
    return a13 * rdamp + a23
