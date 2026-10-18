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
"""Composition-dependent H0 data and pure core-Hamiltonian builders."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Callable

import torch
from tad_mctc import Structure, storch
from tad_mctc.batch import real_pairs
from tad_mctc.convert import any_to_tensor, symmetrize
from tad_mctc.data import radii
from tad_mctc.data.radii import ATOMIC_RADII
from tad_mctc.tree import Node, child, context
from tad_mctc.typing import Tensor
from tad_mctc.units import EV2AU
from tad_multicharge.model.eeq import EEQModel

from dxtb import IndexHelper
from dxtb._src.ncoord import (
    cn_d3,
    coordination_number,
    erf_count,
    exp_count,
    gfn2_count,
)
from dxtb._src.param import Param, ParamModule
from dxtb._src.utils import wrap_gather
from dxtb._src.utils.tensors import structure_charge

PAD = -1


@dataclass(frozen=True, eq=False)
class H0Setup:
    """Immutable H0 data gathered from one composition and parametrization."""

    method: str
    numbers: Tensor
    unique: Tensor
    ihelp: IndexHelper
    hscale: Tensor
    kcn: Tensor
    kpair: Tensor
    refocc: Tensor
    selfenergy: Tensor
    shpoly: Tensor
    valence: Tensor
    en: Tensor
    enscale: Tensor
    rad: Tensor
    cn: Callable[[Tensor, Tensor], Tensor] | None = None
    kq: Tensor | None = None
    kqat: Tensor | None = None
    h0rad: Tensor | None = None
    kdiff: Tensor | None = None
    enshell: Tensor | None = None
    enscale4: Tensor | None = None
    eeq_model: EEQModel | None = None
    cn_radii: Tensor | None = None
    cn_cutoff: Tensor | None = None
    cn_max: Tensor | None = None
    cn_kcn: Tensor | None = None

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> H0Setup:
        """Return converted setup data without casting structural tensors."""
        values: dict[str, object] = {}
        for name in (
            "hscale",
            "kcn",
            "kpair",
            "refocc",
            "selfenergy",
            "shpoly",
            "en",
            "enscale",
            "rad",
            "kq",
            "kqat",
            "h0rad",
            "kdiff",
            "enshell",
            "enscale4",
            "cn_radii",
            "cn_cutoff",
            "cn_max",
            "cn_kcn",
        ):
            value = getattr(self, name)
            values[name] = (
                None if value is None else value.to(device=device, dtype=dtype)
            )
        values["numbers"] = self.numbers.to(device=device)
        values["unique"] = self.unique.to(device=device)
        values["valence"] = self.valence.to(device=device)
        values["ihelp"] = self.ihelp.to(device=device)
        if self.eeq_model is not None:
            values["eeq_model"] = EEQModel(
                chi=self.eeq_model.chi.to(device=device, dtype=dtype),
                kcn=self.eeq_model.kcn.to(device=device, dtype=dtype),
                eta=self.eeq_model.eta.to(device=device, dtype=dtype),
                rad=self.eeq_model.rad.to(device=device, dtype=dtype),
            )
        return replace(self, **values)


class H0IndexNode(Node):
    """Minimal immutable index mappings used by pure H0 evaluation."""

    atom_to_unique: Tensor = child(keep_dtype=True)
    shells_to_ushell: Tensor = child(keep_dtype=True)
    unique_angular: Tensor = child(keep_dtype=True)
    shells_to_atom: Tensor = child(keep_dtype=True)
    orbitals_to_shell: Tensor = child(keep_dtype=True)

    def spread_uspecies_to_atom(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(x, dim, self.atom_to_unique, extra=extra)

    def spread_atom_to_shell(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(x, dim, self.shells_to_atom, extra=extra)

    def spread_uspecies_to_shell(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return self.spread_atom_to_shell(
            self.spread_uspecies_to_atom(x, dim=dim, extra=extra),
            dim=dim,
            extra=extra,
        )

    def spread_ushell_to_shell(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(x, dim, self.shells_to_ushell, extra=extra)

    def spread_shell_to_orbital(
        self,
        x: Tensor,
        dim: int | tuple[int, int] = -1,
        extra: bool = False,
    ) -> Tensor:
        return wrap_gather(x, dim, self.orbitals_to_shell, extra=extra)


class H0SetupNode(Node):
    """Captured, differentiable H0 data without a live IndexHelper."""

    method: str = context()
    cn_kind: str = context()

    index: H0IndexNode = child()
    numbers: Tensor = child(keep_dtype=True)
    unique: Tensor = child(keep_dtype=True)
    hscale: Tensor = child()
    kcn: Tensor = child()
    kpair: Tensor = child()
    refocc: Tensor = child()
    selfenergy: Tensor = child()
    shpoly: Tensor = child()
    valence: Tensor = child(keep_dtype=True)
    en: Tensor = child()
    enscale: Tensor = child()
    rad: Tensor = child()
    kq: Tensor | None = child()
    kqat: Tensor | None = child()
    h0rad: Tensor | None = child()
    kdiff: Tensor | None = child()
    enshell: Tensor | None = child()
    enscale4: Tensor | None = child()
    eeq_model: EEQModel | None = child()
    cn_radii: Tensor | None = child()
    cn_cutoff: Tensor | None = child()
    cn_max: Tensor | None = child()
    cn_kcn: Tensor | None = child()

    @property
    def ihelp(self) -> H0IndexNode:
        """Compatibility name for the narrow H0 index mapping."""
        return self.index

    @property
    def cn(self) -> Callable[[Tensor, Tensor], Tensor] | None:
        """Create the validated stateless built-in CN callable locally."""
        if self.cn_kind == "cn_d3(exp_count)":
            return partial(cn_d3, counting_function=exp_count)
        if self.cn_kind == "cn_d3(gfn2_count)":
            return partial(cn_d3, counting_function=gfn2_count)
        if self.cn_kind == "erf_count":
            return None
        raise RuntimeError(
            f"Unknown H0 coordination descriptor '{self.cn_kind}'."
        )


def _h0_cn_kind(setup: H0Setup) -> str:
    """Validate a legacy setup's CN callable and return its static descriptor."""
    if setup.method == "gfn0":
        if setup.cn is not None:
            raise TypeError(
                "GFN0 H0 Node conversion requires the built-in erf CN."
            )
        return "erf_count"

    expected = {
        "gfn1": ("cn_d3(exp_count)", exp_count),
        "gfn2": ("cn_d3(gfn2_count)", gfn2_count),
    }
    if setup.method not in expected:
        raise TypeError(
            f"Unsupported H0 method '{setup.method}' for Node conversion."
        )

    kind, counting_function = expected[setup.method]
    function = setup.cn
    if not isinstance(function, partial):
        raise TypeError(
            f"H0 Node conversion does not support custom {setup.method.upper()} CN callables."
        )
    if (
        function.func is not cn_d3
        or function.args
        or function.keywords != {"counting_function": counting_function}
    ):
        raise TypeError(
            f"H0 Node conversion requires the built-in {setup.method.upper()} CN strategy."
        )
    return kind


def core_h0_setup(setup: H0Setup) -> H0SetupNode:
    """Capture a legacy setup as an isolated Node, preserving autograd edges."""
    cn_kind = _h0_cn_kind(setup)

    def clone(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.clone()

    eeq_model = None
    if setup.eeq_model is not None:
        eeq_model = EEQModel(
            chi=setup.eeq_model.chi.clone(),
            kcn=setup.eeq_model.kcn.clone(),
            eta=setup.eeq_model.eta.clone(),
            rad=setup.eeq_model.rad.clone(),
        )

    index = H0IndexNode(
        atom_to_unique=setup.ihelp.atom_to_unique.clone(),
        shells_to_ushell=setup.ihelp.shells_to_ushell.clone(),
        unique_angular=setup.ihelp.unique_angular.clone(),
        shells_to_atom=setup.ihelp.shells_to_atom.clone(),
        orbitals_to_shell=setup.ihelp.orbitals_to_shell.clone(),
    )
    return H0SetupNode(
        method=setup.method,
        cn_kind=cn_kind,
        index=index,
        numbers=setup.numbers.clone(),
        unique=setup.unique.clone(),
        hscale=setup.hscale.clone(),
        kcn=setup.kcn.clone(),
        kpair=setup.kpair.clone(),
        refocc=setup.refocc.clone(),
        selfenergy=setup.selfenergy.clone(),
        shpoly=setup.shpoly.clone(),
        valence=setup.valence.clone(),
        en=setup.en.clone(),
        enscale=setup.enscale.clone(),
        rad=setup.rad.clone(),
        kq=clone(setup.kq),
        kqat=clone(setup.kqat),
        h0rad=clone(setup.h0rad),
        kdiff=clone(setup.kdiff),
        enshell=clone(setup.enshell),
        enscale4=clone(setup.enscale4),
        eeq_model=eeq_model,
        cn_radii=clone(setup.cn_radii),
        cn_cutoff=clone(setup.cn_cutoff),
        cn_max=clone(setup.cn_max),
        cn_kcn=clone(setup.cn_kcn),
    )


def legacy_h0_setup(setup: H0SetupNode, ihelp: IndexHelper) -> H0Setup:
    """Build an isolated legacy view for the Calculator-owned H0 adapter.

    Setup tensors are cloned with their autograd edges intact so mutable
    legacy Hamiltonians cannot write through into the System-owned Node.
    """

    def clone(value: Tensor | None) -> Tensor | None:
        return None if value is None else value.clone()

    eeq_model = None
    if setup.eeq_model is not None:
        eeq_model = EEQModel(
            chi=setup.eeq_model.chi.clone(),
            kcn=setup.eeq_model.kcn.clone(),
            eta=setup.eeq_model.eta.clone(),
            rad=setup.eeq_model.rad.clone(),
        )

    return H0Setup(
        method=setup.method,
        numbers=setup.numbers.clone(),
        unique=setup.unique.clone(),
        ihelp=ihelp,
        hscale=setup.hscale.clone(),
        kcn=setup.kcn.clone(),
        kpair=setup.kpair.clone(),
        refocc=setup.refocc.clone(),
        selfenergy=setup.selfenergy.clone(),
        shpoly=setup.shpoly.clone(),
        valence=setup.valence.clone(),
        en=setup.en.clone(),
        enscale=setup.enscale.clone(),
        rad=setup.rad.clone(),
        cn=setup.cn,
        kq=clone(setup.kq),
        kqat=clone(setup.kqat),
        h0rad=clone(setup.h0rad),
        kdiff=clone(setup.kdiff),
        enshell=clone(setup.enshell),
        enscale4=clone(setup.enscale4),
        eeq_model=eeq_model,
        cn_radii=clone(setup.cn_radii),
        cn_cutoff=clone(setup.cn_cutoff),
        cn_max=clone(setup.cn_max),
        cn_kcn=clone(setup.cn_kcn),
    )


def setup_h0_core(
    numbers: Tensor,
    par: Param | ParamModule,
    ihelp: IndexHelper,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> H0SetupNode:
    """Build and capture one H0 setup for the exact single-System path."""
    return core_h0_setup(
        setup_h0(numbers, par, ihelp, device=device, dtype=dtype)
    )


def gather_hscale(
    method: str,
    unique: Tensor,
    ihelp: IndexHelper,
    valence: Tensor,
    par: ParamModule,
    dd: dict,
) -> Tensor:
    angular = ihelp.unique_angular
    labels = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}
    angular_labels = [labels.get(int(ang), PAD) for ang in angular]
    shell = par.get("hamiltonian.xtb.shell")

    if method == "gfn1":
        kpol = par.get("hamiltonian.xtb.kpol")
        kii: list[Tensor] = []
        for i, label in enumerate(angular_labels):
            if valence[i] == 0:
                kii.append(kpol)
            elif f"{label}{label}" in shell:
                kii.append(shell[f"{label}{label}"].param.view(-1)[0])
            else:
                kii.append(torch.tensor(1.0, **dd))
        result = torch.empty((len(angular), len(angular)), **dd)
        for i in range(len(angular)):
            for j in range(i + 1):
                if valence[i] == 1 and valence[j] == 1:
                    key1 = f"{angular_labels[i]}{angular_labels[j]}"
                    key2 = f"{angular_labels[j]}{angular_labels[i]}"
                    if key1 in shell:
                        value = shell[key1].param.view(-1)[0]
                    elif key2 in shell:
                        value = shell[key2].param.view(-1)[0]
                    else:
                        value = (kii[i] + kii[j]) / 2.0
                else:
                    value = (kii[i] + kii[j]) / 2.0
                result[i, j] = value
                result[j, i] = value
        return result

    if method == "gfn2":
        wexp = par.get("hamiltonian.xtb.wexp")
        zeta = par.get_elem_param(unique, "slater", pad_val=PAD)
        zi, zj = zeta.unsqueeze(-1), zeta.unsqueeze(-2)
        zmat = storch.safe_pow(
            2 * storch.safe_divide(storch.safe_sqrt(zi * zj), zi + zj), wexp
        )
        result = torch.ones((len(angular), len(angular)), **dd)
        for i, label_i in enumerate(angular_labels):
            for j, label_j in enumerate(angular_labels):
                key1, key2 = f"{label_i}{label_j}", f"{label_j}{label_i}"
                if key1 in shell:
                    value = shell[key1].param.view(-1)[0]
                elif key2 in shell:
                    value = shell[key2].param.view(-1)[0]
                elif PAD not in (label_i, label_j):
                    key_ii, key_jj = (
                        f"{label_i}{label_i}",
                        f"{label_j}{label_j}",
                    )
                    if key_ii not in shell or key_jj not in shell:
                        raise KeyError(
                            f"GFN2 Core Hamiltonian: missing shell parameters '{key_ii}'/'{key_jj}'."
                        )
                    value = 0.5 * (
                        shell[key_ii].param.view(-1)[0]
                        + shell[key_jj].param.view(-1)[0]
                    )
                else:
                    value = torch.tensor(1.0, **dd)
                result[i, j] = value * zmat[i, j]
        return result

    if method == "gfn0":
        wexp = par.get("hamiltonian.xtb.wexp")
        zeta = par.get_elem_param(unique, "slater", pad_val=1)
        zi, zj = zeta.unsqueeze(-1), zeta.unsqueeze(-2)
        zweight = storch.safe_pow(
            2.0 * storch.safe_divide(storch.safe_sqrt(zi * zj), zi + zj), wexp
        )
        result = torch.ones((len(angular), len(angular)), **dd)
        for i, label_i in enumerate(angular_labels):
            for j, label_j in enumerate(angular_labels):
                key1, key2 = f"{label_i}{label_j}", f"{label_j}{label_i}"
                if key1 in shell:
                    value = par.get(f"hamiltonian.xtb.shell.{key1}")
                elif key2 in shell:
                    value = par.get(f"hamiltonian.xtb.shell.{key2}")
                elif PAD in (label_i, label_j):
                    value = torch.tensor(1.0, **dd)
                else:
                    raise KeyError(
                        f"GFN0 Hamiltonian: missing shell pair '{key1}'."
                    )
                result[i, j] = value
        return result * zweight

    raise ValueError(f"Unsupported H0 method '{method}'.")


def setup_h0(
    numbers: Tensor,
    par: Param | ParamModule,
    ihelp: IndexHelper,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    cn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> H0Setup:
    """Gather all composition-dependent data needed to build H0."""
    if device is not None and (
        numbers.device != device or ihelp.device != device
    ):
        raise ValueError("All input tensors must be on the same device")
    if not isinstance(par, ParamModule):
        par = ParamModule(par, device=device, dtype=dtype)
    if par.is_none("meta.name"):
        raise ValueError("The parametrization must specify a method name.")
    name = par.meta.name.casefold().replace("-xtb", "")
    if name not in ("gfn0", "gfn1", "gfn2"):
        raise ValueError(f"Unsupported Hamiltonian type: {par.meta.name}")
    dd = {
        "device": numbers.device if device is None else device,
        "dtype": par.dtype if dtype is None else dtype,
    }
    unique = torch.unique(numbers)
    if par.is_none("hamiltonian"):
        raise RuntimeError("Parametrization does not specify Hamiltonian.")
    if name == "gfn2":
        valence = torch.ones(
            len(ihelp.unique_angular), device=numbers.device, dtype=torch.bool
        )
    else:
        valence = par.get_elem_valence(unique, pad_val=PAD)
    hscale = gather_hscale(name, unique, ihelp, valence, par, dd)
    en = par.get_elem_param(unique, "en", pad_val=PAD)
    enscale = (
        torch.tensor(0.0, **dd)
        if par.is_none("hamiltonian.xtb.enscale")
        else par.get("hamiltonian.xtb.enscale")
    )
    element_pad = 0 if name == "gfn0" else PAD
    kcn = par.get_elem_param(unique, "kcn", pad_val=element_pad)
    selfenergy = (
        par.get_elem_param(unique, "levels", pad_val=element_pad) * EV2AU
    )
    shpoly = par.get_elem_param(unique, "shpoly", pad_val=PAD)
    refocc = par.get_elem_param(unique, "refocc", pad_val=PAD)
    # Tensor.tolist() requires storage and fails for the GradTrackingTensor
    # produced when Model.setup runs inside torch.func.grad (notably on older
    # supported PyTorch versions). Atomic numbers are structural integer
    # inputs, so extract their scalar values individually for the pair lookup.
    kpair = par.get_pair_param([int(number) for number in unique])
    rad = ATOMIC_RADII(**dd)[unique]
    kcn = kcn * EV2AU

    cn_function = cn
    if name == "gfn1" and cn_function is None:
        cn_function = partial(cn_d3, counting_function=exp_count)
    if name == "gfn2" and cn_function is None:
        cn_function = partial(cn_d3, counting_function=gfn2_count)

    extras: dict[str, object] = {}
    if name == "gfn0":
        if par.is_none("eeq"):
            raise RuntimeError("GFN0 Hamiltonian requires EEQ parameters.")
        if par.get("eeq.cn") != "erf":
            raise ValueError("GFN0 Hamiltonian only supports erf CN.")
        elements = (
            torch.arange(
                int(numbers.max().item()) + 1,
                dtype=numbers.dtype,
                device=numbers.device,
            )
            if numbers.numel()
            else torch.arange(1, dtype=numbers.dtype, device=numbers.device)
        )
        eeq = EEQModel(
            chi=par.get_elem_param(elements, "eeq_chi", pad_val=0).to(**dd),
            kcn=par.get_elem_param(elements, "eeq_kcn", pad_val=0).to(**dd),
            eta=par.get_elem_param(elements, "eeq_eta", pad_val=0).to(**dd),
            rad=par.get_elem_param(elements, "eeq_rad", pad_val=0).to(**dd),
        )
        extras = {
            "kq": par.get_elem_param(unique, "kq", pad_val=0) * EV2AU,
            "kqat": par.get_elem_param(unique, "kqat", pad_val=0) * EV2AU,
            "h0rad": par.get_elem_param(unique, "h0rad", pad_val=1),
            "kdiff": par.get("hamiltonian.xtb.kdiff"),
            "enshell": par.get("hamiltonian.xtb.enshell"),
            "enscale4": par.get("hamiltonian.xtb.enscale4"),
            "eeq_model": eeq,
            "cn_radii": radii.COV_D3(**dd)[numbers],
            "cn_cutoff": par.get("eeq.cutoff"),
            "cn_max": par.get("eeq.cn_max"),
            "cn_kcn": par.get("eeq.kcn"),
        }
        cn_function = None

    return H0Setup(
        method=name,
        numbers=numbers,
        unique=unique,
        ihelp=ihelp,
        hscale=hscale,
        kcn=kcn,
        kpair=kpair,
        refocc=refocc,
        selfenergy=selfenergy,
        shpoly=shpoly,
        valence=valence,
        en=en,
        enscale=enscale,
        rad=rad,
        cn=cn_function,
        **extras,
    )


def build_hcore(
    setup: H0Setup | H0SetupNode,
    positions: Tensor,
    overlap: Tensor,
    charge: Tensor | float | int | None = None,
) -> tuple[Tensor, Tensor]:
    """Build the overlap-weighted H0 matrix and return reference occupations."""
    return _build_hcore(setup, positions, overlap, charge)


def build_hcore_raw(
    setup: H0Setup | H0SetupNode,
    positions: Tensor,
    charge: Tensor | float | int | None = None,
) -> tuple[Tensor, Tensor]:
    """Build the pre-overlap H0 terms for legacy analytical use."""
    return _build_hcore(setup, positions, None, charge)


def _build_hcore(
    setup: H0Setup | H0SetupNode,
    positions: Tensor,
    overlap: Tensor | None,
    charge: Tensor | float | int | None,
) -> tuple[Tensor, Tensor]:
    """Implement the shared H0 formula for true and raw compatibility APIs."""
    ihelp = setup.index if isinstance(setup, H0SetupNode) else setup.ihelp
    if setup.method == "gfn0":
        if charge is None:
            raise ValueError("Total molecular charge is required for GFN0 H0.")
        assert setup.cn_radii is not None and setup.cn_cutoff is not None
        assert setup.cn_max is not None and setup.cn_kcn is not None
        assert setup.eeq_model is not None
        cn = coordination_number(
            setup.numbers,
            positions,
            counting_function=erf_count,
            rcov=setup.cn_radii,
            cutoff=setup.cn_cutoff,
            cn_max=setup.cn_max,
            kcn=setup.cn_kcn,
        )
        total_charge = any_to_tensor(
            charge, device=positions.device, dtype=positions.dtype
        )
        structure = Structure(
            numbers=setup.numbers,
            positions=positions,
            charge=structure_charge(total_charge, setup.numbers),
        )
        charges = setup.eeq_model.solve(structure, cn)
        assert isinstance(charges, Tensor)
        eps0 = ihelp.spread_ushell_to_shell(setup.selfenergy)
        kcn = ihelp.spread_ushell_to_shell(setup.kcn)
        kq = ihelp.spread_ushell_to_shell(setup.kq)
        shell_cn = ihelp.spread_atom_to_shell(cn)
        shell_q = ihelp.spread_atom_to_shell(charges)
        kqat = ihelp.spread_uspecies_to_atom(setup.kqat)
        shell_q2 = ihelp.spread_atom_to_shell(kqat * charges**2)
        selfenergy = eps0 - kcn * shell_cn - kq * shell_q - shell_q2

        atom_pairs = real_pairs(setup.numbers, mask_diagonal=True)
        shell_pairs = ihelp.spread_atom_to_shell(atom_pairs, dim=(-2, -1))
        zero = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
        distances = storch.cdist(positions, positions, p=2)
        h0rad = ihelp.spread_uspecies_to_atom(setup.h0rad)
        reduced = storch.safe_divide(
            distances, h0rad.unsqueeze(-1) + h0rad.unsqueeze(-2)
        )
        root_reduced = ihelp.spread_atom_to_shell(
            torch.where(atom_pairs, storch.safe_sqrt(reduced), zero),
            dim=(-2, -1),
        )
        shpoly = ihelp.spread_ushell_to_shell(setup.shpoly)
        distance_scale = (1.0 + shpoly.unsqueeze(-1) * root_reduced) * (
            1.0 + shpoly.unsqueeze(-2) * root_reduced
        )
        angular = ihelp.unique_angular.clamp(min=0)
        shell_en = setup.enshell[angular]
        enscale = 0.005 * (shell_en.unsqueeze(-1) + shell_en.unsqueeze(-2))
        enscale = ihelp.spread_ushell_to_shell(enscale, dim=(-2, -1))
        en = ihelp.spread_uspecies_to_shell(setup.en)
        den2 = (en.unsqueeze(-1) - en.unsqueeze(-2)) ** 2
        enpoly = 1.0 + enscale * den2 + setup.enscale4 * enscale * den2**2
        kpair = ihelp.spread_uspecies_to_shell(setup.kpair, dim=(-2, -1))
        hscale = ihelp.spread_ushell_to_shell(setup.hscale, dim=(-2, -1))
        scale = hscale * kpair * enpoly
        valence = ihelp.spread_ushell_to_shell(setup.valence)
        both_valence = valence.unsqueeze(-1) & valence.unsqueeze(-2)
        one_nonvalence = valence.unsqueeze(-1) ^ valence.unsqueeze(-2)
        scale = torch.where(
            both_valence,
            scale,
            torch.where(one_nonvalence, scale * setup.kdiff, zero),
        )
        average = 0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2))
        shell_h0 = torch.where(
            shell_pairs, distance_scale * scale * average, zero
        )
        h0 = ihelp.spread_shell_to_orbital(shell_h0, dim=(-2, -1))
        if overlap is not None:
            h0 = h0 * overlap
        diagonal = ihelp.spread_shell_to_orbital(selfenergy)
        h0 = symmetrize(h0, force=True) + torch.diag_embed(diagonal)
        return h0, setup.refocc

    if setup.cn is None:
        cn = torch.zeros_like(
            setup.numbers, device=positions.device, dtype=positions.dtype
        )
    else:
        cn = setup.cn(setup.numbers, positions)
    atom_mask = real_pairs(setup.numbers, mask_diagonal=True)
    shell_mask = real_pairs(
        ihelp.spread_atom_to_shell(setup.numbers), mask_diagonal=False
    )
    shell_diagonal = ihelp.spread_atom_to_shell(atom_mask, dim=(-2, -1))
    zero = torch.tensor(0.0, device=positions.device, dtype=positions.dtype)
    kcn = ihelp.spread_ushell_to_shell(setup.kcn)
    selfenergy = ihelp.spread_ushell_to_shell(
        setup.selfenergy
    ) - kcn * ihelp.spread_atom_to_shell(cn)
    distances = storch.cdist(positions, positions, p=2)
    rad = ihelp.spread_uspecies_to_atom(setup.rad)
    rr = storch.safe_divide(distances, rad.unsqueeze(-1) + rad.unsqueeze(-2))
    rr_shell = ihelp.spread_atom_to_shell(
        torch.where(atom_mask, storch.safe_sqrt(rr), zero), (-2, -1)
    )
    shpoly = ihelp.spread_ushell_to_shell(setup.shpoly)
    var_pi = (1.0 + shpoly.unsqueeze(-1) * rr_shell) * (
        1.0 + shpoly.unsqueeze(-2) * rr_shell
    )
    en = ihelp.spread_uspecies_to_shell(setup.en)
    var_x = torch.where(
        shell_diagonal,
        1.0
        + setup.enscale * torch.pow(en.unsqueeze(-1) - en.unsqueeze(-2), 2.0),
        zero,
    )
    kpair = ihelp.spread_uspecies_to_shell(setup.kpair, dim=(-2, -1))
    hscale = ihelp.spread_ushell_to_shell(setup.hscale, dim=(-2, -1))
    valence = ihelp.spread_ushell_to_shell(setup.valence)
    var_k = torch.where(
        valence.unsqueeze(-1) * valence.unsqueeze(-2),
        hscale * kpair * var_x,
        hscale,
    )
    var_h = torch.where(
        shell_mask,
        0.5 * (selfenergy.unsqueeze(-1) + selfenergy.unsqueeze(-2)),
        zero,
    )
    hcore = ihelp.spread_shell_to_orbital(
        torch.where(shell_diagonal, var_pi * var_k * var_h, var_h), dim=(-2, -1)
    )
    if overlap is not None:
        hcore = hcore * overlap
    return symmetrize(hcore, force=True), setup.refocc
