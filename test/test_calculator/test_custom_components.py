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
"""End-to-end tests for custom Classical and Interaction components."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import pytest
import torch
from tad_mctc.tree import ModuleNode
from tad_mctc.tree import stack as tree_stack
from torch.func import hessian, jacrev, vmap

from dxtb import GFN1_XTB, Calculator, IndexHelper, ParamModule, labels
from dxtb._src.calculators.batch_implicit import (
    evaluate_converged_root_batch,
    prepare_converged_root_batch,
)
from dxtb._src.calculators.lane_evaluation import (
    LaneGroup,
    can_vmap_system,
    evaluate_lane_group,
)
from dxtb._src.calculators.model import Model, ModelParameterNode, System
from dxtb._src.components.custom import CustomClassicalCoreSetup
from dxtb.components.base import (
    Classical,
    ClassicalCache,
    Interaction,
    InteractionCache,
)
from dxtb.config import Config
from dxtb.typing import DD, Tensor, override

from ..utils import scf_strategy

DTYPE = torch.float64
DEVICE = torch.device("cpu")
DD64: DD = {"device": DEVICE, "dtype": DTYPE}


class HarmonicRestraintCache(ClassicalCache):
    """Composition-dependent setup data for the harmonic restraint."""

    force_constant: Tensor
    atom_weights: Tensor
    __slots__ = ["force_constant", "atom_weights"]

    def __init__(
        self,
        force_constant: Tensor,
        atom_weights: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=force_constant.device if device is None else device,
            dtype=force_constant.dtype if dtype is None else dtype,
        )
        self.force_constant = force_constant
        self.atom_weights = atom_weights


class HarmonicRestraint(Classical):
    """Weighted harmonic restraint around the origin."""

    force_constant: Tensor
    __slots__ = ["force_constant"]

    def __init__(
        self,
        force_constant: Tensor,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(
            device=force_constant.device if device is None else device,
            dtype=force_constant.dtype if dtype is None else dtype,
        )
        self.force_constant = force_constant

    @override
    def get_cache(
        self,
        numbers: Tensor,
        ihelp: IndexHelper | None = None,
    ) -> HarmonicRestraintCache:
        """Build per-atom active weights from ``numbers``."""
        _ = ihelp
        weights = (numbers > 0).to(
            device=self.force_constant.device, dtype=self.force_constant.dtype
        )
        return HarmonicRestraintCache(self.force_constant, weights)

    @override
    def get_energy(
        self,
        positions: Tensor,
        cache: HarmonicRestraintCache,
        **_: Any,
    ) -> Tensor:
        """Return atom-resolved harmonic restraint energies."""
        return (
            0.5
            * cache.force_constant
            * cache.atom_weights
            * positions.square().sum(dim=-1)
        )


class OnsiteShiftCache(InteractionCache):
    """Per-geometry interaction cache for :class:`OnsiteShift`."""

    shift: Tensor
    hardness: Tensor
    __slots__ = ["shift", "hardness"]

    def __init__(self, shift: Tensor, hardness: Tensor) -> None:
        super().__init__(device=shift.device, dtype=shift.dtype)
        self.shift = shift
        self.hardness = hardness


class OnsiteShift(Interaction):
    """Quadratic atomic charge penalty coupled self-consistently."""

    scale: Tensor
    hardness: Tensor
    __slots__ = ["scale", "hardness"]

    def __init__(self, scale: Tensor, hardness: Tensor) -> None:
        super().__init__(device=scale.device, dtype=scale.dtype)
        self.scale = scale
        self.hardness = hardness

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> OnsiteShiftCache:
        """Compute geometry-dependent onsite shifts."""
        _ = (numbers, ihelp)
        assert positions is not None
        radial = self.scale * positions.square().sum(dim=-1)
        return OnsiteShiftCache(radial, self.hardness)

    @override
    def get_monopole_atom_potential(
        self,
        cache: OnsiteShiftCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """Return ``dE / dqat``."""
        _ = (qdp, qqp)
        return cache.shift + cache.hardness * qat

    @override
    def get_monopole_atom_energy(
        self,
        cache: OnsiteShiftCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """Return atom-resolved onsite shift energy."""
        _ = (qdp, qqp)
        return cache.shift * qat + 0.5 * cache.hardness * qat.square()


class MultipolePenaltyCache(InteractionCache):
    """Cache holding scaled radial prefactors for shell/atom multipoles."""

    shell_k: Tensor
    mono_k: Tensor
    dip_k: Tensor
    quad_k: Tensor
    geom_factor: Tensor
    __slots__ = ["shell_k", "mono_k", "dip_k", "quad_k", "geom_factor"]

    def __init__(
        self,
        shell_k: Tensor,
        mono_k: Tensor,
        dip_k: Tensor,
        quad_k: Tensor,
        geom_factor: Tensor,
    ) -> None:
        super().__init__(device=geom_factor.device, dtype=geom_factor.dtype)
        self.shell_k = shell_k
        self.mono_k = mono_k
        self.dip_k = dip_k
        self.quad_k = quad_k
        self.geom_factor = geom_factor


class FullMultipolePenalty(Interaction):
    """Self-consistent penalty exercising shell monopole and atom multipole hooks."""

    def __init__(
        self,
        shell_k: Tensor,
        mono_k: Tensor,
        dip_k: Tensor,
        quad_k: Tensor,
    ) -> None:
        super().__init__(device=shell_k.device, dtype=shell_k.dtype)
        self.shell_k = shell_k
        self.mono_k = mono_k
        self.dip_k = dip_k
        self.quad_k = quad_k

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> MultipolePenaltyCache:
        """Build geometry factor ``1 + 0.05 * r^2`` per atom."""
        _ = (numbers, ihelp)
        assert positions is not None
        geom_factor = 1.0 + 0.05 * positions.square().sum(dim=-1)
        return MultipolePenaltyCache(
            self.shell_k, self.mono_k, self.dip_k, self.quad_k, geom_factor
        )

    @override
    def get_monopole_shell_potential(
        self, cache: MultipolePenaltyCache, qsh: Tensor
    ) -> Tensor:
        """Return ``dE_sh / dqsh`` of shape ``(..., nsh)``."""
        return cache.shell_k * qsh

    @override
    def get_monopole_shell_energy(
        self, cache: MultipolePenaltyCache, qsh: Tensor, **_: Any
    ) -> Tensor:
        """Return shell-resolved energy of shape ``(..., nsh)``."""
        return 0.5 * cache.shell_k * qsh.square()

    @override
    def get_monopole_atom_potential(
        self,
        cache: MultipolePenaltyCache,
        qat: Tensor,
        qdp: Tensor | None = None,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """Return ``dE_at / dqat`` of shape ``(..., nat)``."""
        _ = (qdp, qqp)
        return cache.mono_k * cache.geom_factor * qat

    @override
    def get_monopole_atom_energy(
        self, cache: MultipolePenaltyCache, qat: Tensor, **_: Any
    ) -> Tensor:
        """Return atom-resolved monopole energy of shape ``(..., nat)``."""
        return 0.5 * cache.mono_k * cache.geom_factor * qat.square()

    @override
    def get_dipole_atom_potential(
        self,
        cache: MultipolePenaltyCache,
        qat: Tensor,
        qdp: Tensor,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """Return ``dE_dip / dqdp`` of shape ``(..., nat, 3)``."""
        _ = (qat, qqp)
        return cache.dip_k * cache.geom_factor.unsqueeze(-1) * qdp

    @override
    def get_dipole_atom_energy(
        self,
        cache: MultipolePenaltyCache,
        qat: Tensor,
        qdp: Tensor,
        qqp: Tensor | None = None,
    ) -> Tensor:
        """Return atom-resolved dipole energy of shape ``(..., nat)``."""
        _ = (qat, qqp)
        return 0.5 * cache.dip_k * cache.geom_factor * qdp.square().sum(dim=-1)

    @override
    def get_quadrupole_atom_potential(
        self,
        cache: MultipolePenaltyCache,
        qat: Tensor,
        qdp: Tensor | None,
        qqp: Tensor,
    ) -> Tensor:
        """Return ``dE_quad / dqqp`` of shape ``(..., nat, 6)``."""
        _ = (qat, qdp)
        return cache.quad_k * cache.geom_factor.unsqueeze(-1) * qqp

    @override
    def get_quadrupole_atom_energy(
        self,
        cache: MultipolePenaltyCache,
        qat: Tensor,
        qdp: Tensor | None,
        qqp: Tensor,
    ) -> Tensor:
        """Return atom-resolved quadrupole energy of shape ``(..., nat)``."""
        _ = (qat, qdp)
        return 0.5 * cache.quad_k * cache.geom_factor * qqp.square().sum(dim=-1)


class QuadrupoleOnlyInteraction(Interaction):
    """Custom Interaction defining quadrupole hooks without dipole hooks."""

    def __init__(self, quad_k: Tensor) -> None:
        super().__init__(device=quad_k.device, dtype=quad_k.dtype)
        self.quad_k = quad_k

    @override
    def get_cache(
        self,
        *,
        numbers: Tensor | None = None,
        positions: Tensor | None = None,
        ihelp: IndexHelper | None = None,
    ) -> InteractionCache:
        """Return an empty cache so ``self.quad_k`` is read directly off ``self``."""
        _ = (numbers, positions, ihelp)
        return InteractionCache(device=self.device, dtype=self.dtype)

    @override
    def get_quadrupole_atom_potential(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None,
        qqp: Tensor,
    ) -> Tensor:
        """Return ``dE / dqqp`` of shape ``(..., nat, 6)``."""
        _ = (cache, qat, qdp)
        return self.quad_k * qqp

    @override
    def get_quadrupole_atom_energy(
        self,
        cache: InteractionCache,
        qat: Tensor,
        qdp: Tensor | None,
        qqp: Tensor,
    ) -> Tensor:
        """Return atom-resolved quadrupole energy of shape ``(..., nat)``."""
        _ = (cache, qat, qdp)
        return 0.5 * self.quad_k * qqp.square().sum(dim=-1)


def _lih_inputs() -> tuple[Tensor, Tensor]:
    """Return LiH atomic numbers and reference coordinates."""
    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], **DD64)
    return numbers, positions


def test_docs_harmonic_restraint_and_onsite_shift_across_calculator_modes() -> (
    None
):
    """Documentation custom Classical and Interaction examples run across 1D and 2D batches."""
    restraint = HarmonicRestraint(torch.tensor(0.05, **DD64), **DD64)
    interaction = OnsiteShift(
        torch.tensor(0.02, **DD64), torch.tensor(0.10, **DD64)
    )
    opts = {"verbosity": 0, "int_driver": "pytorch"}

    num_lih, pos_lih = _lih_inputs()
    calc_1d = Calculator(
        num_lih,
        GFN1_XTB,
        classical=(restraint,),
        interaction=(interaction,),
        opts=opts,
        **DD64,
    )
    res_1d = calc_1d.singlepoint(pos_lih)
    assert restraint.label in dict(res_1d.classical)
    assert torch.isfinite(res_1d.energy).all()

    pos_lih_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], **DD64)
    calc_1d_b = Calculator(
        num_lih,
        GFN1_XTB,
        classical=(restraint,),
        interaction=(interaction,),
        opts=opts,
        **DD64,
    )
    res_1d_b = calc_1d_b.singlepoint(pos_lih_b)

    # Mode 2: conformer batch
    nums_conf = torch.stack([num_lih, num_lih], dim=0)
    pos_conf = torch.stack([pos_lih, pos_lih_b], dim=0)
    calc_conf = Calculator(
        nums_conf,
        GFN1_XTB,
        classical=(restraint,),
        interaction=(interaction,),
        opts={**opts, "batch_mode": 2},
        **DD64,
    )
    e_conf = calc_conf.energy(pos_conf)
    torch.testing.assert_close(
        e_conf,
        torch.stack([res_1d.energy.sum(-1), res_1d_b.energy.sum(-1)]),
    )

    # Mode 1: padded heterogeneous batch (H2O + LiH)
    num_h2o = torch.tensor([8, 1, 1], device=DEVICE)
    pos_h2o = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]], **DD64
    )
    e_h2o = Calculator(
        num_h2o,
        GFN1_XTB,
        classical=(restraint,),
        interaction=(interaction,),
        opts=opts,
        **DD64,
    ).energy(pos_h2o)
    nums_pad = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    pos_pad = torch.stack(
        [pos_h2o, torch.cat([pos_lih, torch.zeros((1, 3), **DD64)], dim=0)],
        dim=0,
    )
    calc_pad = Calculator(
        nums_pad,
        GFN1_XTB,
        classical=(restraint,),
        interaction=(interaction,),
        opts={**opts, "batch_mode": 1},
        **DD64,
    )
    torch.testing.assert_close(
        calc_pad.energy(pos_pad),
        torch.stack([e_h2o, res_1d.energy.sum(-1)]),
    )


@pytest.mark.parametrize("variable", ["charge", "potential", "fock"])
def test_custom_multipole_hooks_implicit_matches_unrolled_and_fd(
    variable: str,
) -> None:
    """Shell/atom multipole and quadrupole-only custom Interactions match across unrolled, implicit, and FD."""
    numbers, base_pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)

    def run_mode(
        differentiation: str,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        pos = base_pos.clone().requires_grad_(True)
        cls_k = torch.tensor(0.04, **DD64, requires_grad=True)
        sh_k = torch.tensor(0.06, **DD64, requires_grad=True)
        mono_k = torch.tensor(0.08, **DD64, requires_grad=True)
        dip_k = torch.tensor(0.02, **DD64, requires_grad=True)
        quad_k = torch.tensor(0.002, **DD64, requires_grad=True)
        quad_only_k = torch.tensor(0.001, **DD64, requires_grad=True)

        restraint = HarmonicRestraint(cls_k, **DD64)
        multipole = FullMultipolePenalty(sh_k, mono_k, dip_k, quad_k)
        quad_only = QuadrupoleOnlyInteraction(quad_only_k)
        cfg = Config.create(
            int_driver="pytorch",
            differentiation=differentiation,
            strategy=scf_strategy("anderson", variable),
            x_atol=1.0e-9,
            x_atol_max=1.0e-9,
            maxiter=60,
        )
        system = Model(
            par=par,
            config=cfg,
            classical=(restraint,),
            interaction=(multipole, quad_only),
            auto_int_level=True,
        ).setup(numbers)
        assert system.config.ints.level >= labels.INTLEVEL_QUADRUPOLE
        total = system.singlepoint(pos).energy.sum()
        grads = torch.autograd.grad(
            total, (pos, cls_k, sh_k, mono_k, dip_k, quad_k, quad_only_k)
        )
        return total.detach(), grads

    e_unrolled, g_unrolled = run_mode("unrolled")
    e_implicit, g_implicit = run_mode("implicit")
    torch.testing.assert_close(e_implicit, e_unrolled, rtol=1e-9, atol=1e-10)
    for gi, gu in zip(g_implicit, g_unrolled):
        assert torch.isfinite(gi).all() and (gi.abs() > 0).any()
        torch.testing.assert_close(gi, gu, rtol=1e-6, atol=1e-7)

    # Central finite-difference check on quad_only_k (direct-self parameter)
    if variable == "potential":
        eps = 1.0e-6
        vals: list[Tensor] = []
        for delta in (-eps, eps):
            restraint = HarmonicRestraint(torch.tensor(0.04, **DD64), **DD64)
            multipole = FullMultipolePenalty(
                torch.tensor(0.06, **DD64),
                torch.tensor(0.08, **DD64),
                torch.tensor(0.02, **DD64),
                torch.tensor(0.002, **DD64),
            )
            quad_only = QuadrupoleOnlyInteraction(
                torch.tensor(0.001 + delta, **DD64)
            )
            sys_fd = Model(
                par=par,
                config=Config.create(
                    int_driver="pytorch",
                    strategy=scf_strategy("anderson", variable),
                    x_atol=1.0e-10,
                    x_atol_max=1.0e-10,
                    maxiter=60,
                ),
                classical=(restraint,),
                interaction=(multipole, quad_only),
            ).setup(numbers)
            vals.append(sys_fd.singlepoint(base_pos).energy.sum())
        fd_grad = (vals[1] - vals[0]) / (2.0 * eps)
        torch.testing.assert_close(
            g_implicit[-1], fd_grad, rtol=1e-5, atol=1e-7
        )


def test_functional_setup_and_batch_implicit_with_custom_components() -> None:
    """setup_functional and batch_implicit preserve exact parameter and geometry gradients."""
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.25]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)
    lgam_key = "parameter_tree.element.H.lgam.param"

    def eval_functional(differentiation: str) -> tuple[Tensor, ...]:
        pos = pos_a.clone().requires_grad_(True)
        cls_k = torch.tensor(0.05, **DD64, requires_grad=True)
        shift_s = torch.tensor(0.02, **DD64, requires_grad=True)
        hard_h = torch.tensor(0.12, **DD64, requires_grad=True)
        model = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=differentiation,
                x_atol=1.0e-10,
                x_atol_max=1.0e-10,
                maxiter=80,
            ),
            classical=(HarmonicRestraint(cls_k, **DD64),),
            interaction=(OnsiteShift(shift_s, hard_h),),
        )
        base_tree = model.parameter_tree
        lgam_leaf = (
            base_tree.parameters.params[lgam_key]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        new_params = dict(base_tree.parameters.params)
        new_params[lgam_key] = lgam_leaf
        tree: ModelParameterNode = replace(
            base_tree,
            parameters=ModuleNode(
                module=base_tree.parameters.module, params=new_params
            ),
        )
        system = model.setup_functional(numbers, tree)
        loss = system.singlepoint(pos).energy.sum()
        return torch.autograd.grad(
            loss, (pos, lgam_leaf, cls_k, shift_s, hard_h)
        )

    for gi, gu in zip(eval_functional("implicit"), eval_functional("unrolled")):
        torch.testing.assert_close(gi, gu, rtol=1e-6, atol=1e-7)

    # Batch implicit conformer evaluation vs unrolled reference
    def eval_conformer_batch(
        differentiation: str,
    ) -> tuple[Tensor, tuple[Tensor, ...]]:
        pos_batch = torch.stack([pos_a, pos_b], dim=0).requires_grad_(True)
        cls_k = torch.tensor(0.05, **DD64, requires_grad=True)
        shift_s = torch.tensor(0.02, **DD64, requires_grad=True)
        hard_h = torch.tensor(0.12, **DD64, requires_grad=True)
        cfg = Config.create(
            int_driver="pytorch",
            differentiation=differentiation,
            x_atol=1.0e-10,
            x_atol_max=1.0e-10,
            maxiter=80,
        )
        restraint = HarmonicRestraint(cls_k, **DD64)
        shift = OnsiteShift(shift_s, hard_h)
        sys_0 = Model(
            par=par, config=cfg, classical=(restraint,), interaction=(shift,)
        ).setup(numbers)
        if differentiation == "implicit":
            prepared = prepare_converged_root_batch(sys_0, pos_batch)
            result = evaluate_converged_root_batch(prepared)
            energies = result.energy.sum(-1)
        else:
            energies = torch.stack(
                [
                    sys_0.singlepoint(pos_batch[0]).energy.sum(-1),
                    sys_0.singlepoint(pos_batch[1]).energy.sum(-1),
                ]
            )
        weighted = (energies * torch.tensor([1.0, 1.7], **DD64)).sum()
        grads = torch.autograd.grad(
            weighted, (pos_batch, cls_k, shift_s, hard_h)
        )
        return energies.detach(), grads

    e_bi, g_bi = eval_conformer_batch("implicit")
    e_bu, g_bu = eval_conformer_batch("unrolled")
    torch.testing.assert_close(e_bi, e_bu, rtol=1e-9, atol=1e-10)
    for gi, gu in zip(g_bi, g_bu):
        torch.testing.assert_close(gi, gu, rtol=1e-6, atol=1e-7)


def _system_total_energy(system: System, positions: Tensor) -> Tensor:
    """Return scalar total energy of ``system`` at ``positions``."""
    return system.energy(positions)


def _system_field_energy(
    system: System, positions: Tensor, field: Tensor
) -> Tensor:
    """Return scalar total energy of ``system`` at ``positions`` and ``field``."""
    return system.energy(positions, field=field)


def test_system_vmap_forces_and_hessian_with_custom_components() -> None:
    """Tensor-only custom Classical and Interaction components compose with vmap, forces, and hessian."""
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], **DD64)
    restraint = HarmonicRestraint(torch.tensor(0.04, **DD64), **DD64)
    shift = OnsiteShift(torch.tensor(0.015, **DD64), torch.tensor(0.08, **DD64))
    system = Model(
        par=ParamModule(GFN1_XTB, **DD64),
        config=Config.create(
            int_driver="pytorch",
            x_atol=1.0e-10,
            x_atol_max=1.0e-10,
            maxiter=14,
        ),
        classical=(restraint,),
        interaction=(shift,),
    ).setup(numbers)
    assert can_vmap_system(system)

    def energy_of_pos(pos: Tensor) -> Tensor:
        """Total energy as a pure function of positions."""
        return _system_total_energy(system, pos)

    batch_pos = torch.stack([pos_a, pos_b], dim=0)
    mapped_energies = vmap(energy_of_pos)(batch_pos)  # (2,)
    expected_energies = torch.stack(
        [energy_of_pos(pos_a), energy_of_pos(pos_b)]
    )
    torch.testing.assert_close(mapped_energies, expected_energies)

    forces = system.forces(pos_a)
    hess = system.hessian(pos_a)
    torch.testing.assert_close(forces, -jacrev(energy_of_pos)(pos_a))
    torch.testing.assert_close(
        hess, hessian(energy_of_pos)(pos_a), rtol=1e-5, atol=1e-6
    )


def test_system_electric_and_vibrational_transforms_with_custom_components() -> (
    None
):
    """System dipole, polarizability, vibration, IR, and Raman work with custom components."""
    numbers = torch.tensor([1, 1], device=DEVICE)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], **DD64)
    restraint = HarmonicRestraint(torch.tensor(0.04, **DD64), **DD64)
    shift = OnsiteShift(torch.tensor(0.02, **DD64), torch.tensor(0.08, **DD64))
    system = Model(
        par=ParamModule(GFN1_XTB, **DD64),
        config=Config.create(
            int_driver="pytorch",
            int_level=labels.INTLEVEL_DIPOLE,
            x_atol=1.0e-10,
            x_atol_max=1.0e-10,
            maxiter=6,
        ),
        classical=(restraint,),
        interaction=(shift,),
    ).setup(numbers)

    zero_field = torch.zeros(3, **DD64)

    def energy_of_field(f: Tensor) -> Tensor:
        """Total energy at fixed geometry as a function of external field."""
        return _system_field_energy(system, pos, f)

    torch.testing.assert_close(
        system.dipole(pos), -jacrev(energy_of_field)(zero_field)
    )
    torch.testing.assert_close(
        system.polarizability(pos),
        -hessian(energy_of_field)(zero_field),
        rtol=1e-5,
        atol=1e-6,
    )
    vib = system.vibration(pos)
    ir_res = system.ir(pos)
    raman_res = system.raman(pos)
    assert torch.isfinite(vib.freqs).all()
    assert torch.isfinite(ir_res.ints).all()
    assert torch.isfinite(raman_res.ints).all()


def test_vmap_runtime_fallback_for_data_dependent_custom_component() -> None:
    """evaluate_lane_group falls back cleanly when a tensor-only custom hook uses .item()."""

    class DataDependentClassical(Classical):
        def __init__(self, scale: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.scale = scale

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return (numbers > 0).to(dtype=self.scale.dtype)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            # Data-dependent host conversion fails under torch.func.vmap
            sign = 1.0 if float(positions.sum().item()) >= -1.0e9 else -1.0
            return sign * self.scale * cache * positions.square().sum(dim=-1)

    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], **DD64)
    scale = torch.tensor(0.03, **DD64, requires_grad=True)
    comp = DataDependentClassical(scale)
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")
    sys_a = Model(par=par, config=cfg, classical=(comp,)).setup(numbers)
    assert can_vmap_system(sys_a)

    pa = pos_a.clone().requires_grad_(True)
    pb = pos_b.clone().requires_grad_(True)
    z0 = torch.tensor(0.0, **DD64)
    group = LaneGroup(system=sys_a, source_ids=(0, 1), use_vmap=True)
    results = evaluate_lane_group(
        group,
        (pa, pb),
        (z0, z0),
        (None, None),
        (None, None),
        (None, None),
    )
    loss = results[0].energy.sum() + results[1].energy.sum()
    g_pa, g_scale = torch.autograd.grad(loss, (pa, scale))

    ref_scale = torch.tensor(0.03, **DD64, requires_grad=True)
    ref_sys = Model(
        par=par, config=cfg, classical=(DataDependentClassical(ref_scale),)
    ).setup(numbers)
    ref_pa = pos_a.clone().requires_grad_(True)
    ref_loss = (
        ref_sys.singlepoint(ref_pa).energy.sum()
        + ref_sys.singlepoint(pos_b).energy.sum()
    )
    ref_g_pa, ref_g_scale = torch.autograd.grad(ref_loss, (ref_pa, ref_scale))
    torch.testing.assert_close(g_pa, ref_g_pa)
    torch.testing.assert_close(g_scale, ref_g_scale)


def test_dtype_conversion_and_in_place_mutation_slots_and_dict() -> None:
    """Custom components with or without __slots__ convert dtypes cleanly and detect mutations."""

    class DictClassical(Classical):
        """Subclass without __slots__ holding both int64 indices and a float parameter."""

        def __init__(self, coeff: Tensor, atom_idx: Tensor) -> None:
            super().__init__(device=coeff.device, dtype=coeff.dtype)
            self.coeff = coeff
            self.atom_idx = atom_idx

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> dict[str, Tensor]:
            _ = (numbers, ihelp)
            return {"coeff": self.coeff, "atom_idx": self.atom_idx}

        @override
        def get_energy(
            self, positions: Tensor, cache: dict[str, Tensor], **_: Any
        ) -> Tensor:
            selected = positions.index_select(-2, cache["atom_idx"])
            contrib = cache["coeff"] * selected.square().sum()
            return (
                positions.new_zeros(positions.shape[:-1])
                + contrib / positions.shape[-2]
            )

    numbers, pos64 = _lih_inputs()
    cls_comp = DictClassical(
        torch.tensor(0.05, **DD64),
        torch.tensor([0, 1], device=DEVICE, dtype=torch.int64),
    )
    int_comp = OnsiteShift(
        torch.tensor(0.02, **DD64), torch.tensor(0.10, **DD64)
    )
    calc64 = Calculator(
        numbers,
        GFN1_XTB,
        classical=(cls_comp,),
        interaction=(int_comp,),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    e64 = calc64.energy(pos64)

    calc32 = calc64.type(torch.float32)
    e32 = calc32.energy(pos64.to(dtype=torch.float32))
    assert e32.dtype == torch.float32
    conv_cls = calc32.classicals.components[-1]
    assert isinstance(conv_cls, DictClassical)
    assert conv_cls.coeff.dtype == torch.float32
    assert conv_cls.atom_idx.dtype == torch.int64
    torch.testing.assert_close(
        e32.to(dtype=torch.float64), e64, rtol=1e-4, atol=1e-4
    )

    # Direct in-place mutation on System invalidates captured CustomClassicalCoreSetup
    sys64 = Model(
        par=ParamModule(GFN1_XTB, **DD64),
        config=Config.create(int_driver="pytorch", exclude="scf"),
        classical=(cls_comp,),
    ).setup(numbers)
    before = dict(sys64.singlepoint(pos64).classical)[cls_comp.label].sum()
    with torch.no_grad():
        cls_comp.coeff.mul_(2.0)
    after = dict(sys64.singlepoint(pos64).classical)[cls_comp.label].sum()
    torch.testing.assert_close(after, 2.0 * before)


def _extract_rst_python_blocks(rst_path: Path) -> list[str]:
    """Extract all ``.. code-block:: python`` blocks from ``rst_path``."""
    lines = rst_path.read_text(encoding="utf-8").splitlines()
    blocks: list[str] = []
    idx = 0
    while idx < len(lines):
        if lines[idx].strip() == ".. code-block:: python":
            idx += 1
            while idx < len(lines) and not lines[idx].strip():
                idx += 1
            block_lines: list[str] = []
            while idx < len(lines):
                raw = lines[idx]
                if raw.strip() and not raw.startswith(" "):
                    break
                block_lines.append(raw)
                idx += 1
            blocks.append(textwrap.dedent("\n".join(block_lines)))
        else:
            idx += 1
    return blocks


def test_components_new_tutorial_snippets_execute() -> None:
    """Every Python code block in docs/source/02_indepth/components_new.rst executes cleanly."""
    rst_path = (
        Path(__file__).parents[2]
        / "docs"
        / "source"
        / "02_indepth"
        / "components_new.rst"
    )
    blocks = _extract_rst_python_blocks(rst_path)
    assert len(blocks) >= 2
    for block in blocks:
        ns: dict[str, Any] = {"__builtins__": __builtins__}
        exec(compile(block, str(rst_path), "exec"), ns)
        assert "result" in ns
        assert torch.isfinite(ns["result"].energy).all()


def test_private_and_dunder_slots_implicit_gradients_and_dtype() -> None:
    """Private (__x) and mid-dunder (r0__bohr) slots survive .type(), .reset(), and implicit SCF."""

    class PrivateSlotInteraction(Interaction):
        __slots__ = ["__scale", "r0__bohr"]

        def __init__(self, scale: Tensor, r0: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.__scale = scale
            self.r0__bohr = r0

        def scale_tensor(self) -> Tensor:
            return self.__scale

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> InteractionCache:
            _ = (numbers, positions, ihelp)
            return InteractionCache(device=self.device, dtype=self.dtype)

        @override
        def get_monopole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            weights = torch.arange(
                1, qat.shape[-1] + 1, device=qat.device, dtype=qat.dtype
            )
            return 2.0 * (self.__scale + self.r0__bohr) * qat * weights

        @override
        def get_monopole_atom_energy(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            weights = torch.arange(
                1, qat.shape[-1] + 1, device=qat.device, dtype=qat.dtype
            )
            return (self.__scale + self.r0__bohr) * qat.square() * weights

    s0 = torch.tensor(0.04, **DD64, requires_grad=True)
    r0 = torch.tensor(0.06, **DD64, requires_grad=True)
    inter32 = PrivateSlotInteraction(s0, r0).type(torch.float32)
    assert inter32.scale_tensor().dtype == torch.float32
    assert inter32.r0__bohr.dtype == torch.float32
    inter32.reset()
    assert inter32.scale_tensor().requires_grad

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]], **DD64
    )
    par = ParamModule(GFN1_XTB, **DD64)

    def run_mode(diff_mode: str) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        s_param = torch.tensor(0.04, **DD64, requires_grad=True)
        r_param = torch.tensor(0.06, **DD64, requires_grad=True)
        cfg = Config.create(
            int_driver="pytorch",
            differentiation=diff_mode,
            x_atol=1.0e-11,
            x_atol_max=1.0e-11,
            maxiter=80,
        )
        sys_m = Model(
            par=par,
            config=cfg,
            interaction=(PrivateSlotInteraction(s_param, r_param),),
        ).setup(numbers)
        res = sys_m.singlepoint(pos)
        orb_w = torch.linspace(0.2, 1.0, res.charges.mono.shape[-1], **DD64)
        q_loss = (res.charges.mono * orb_w).sum()
        gs_q, gr_q = torch.autograd.grad(
            q_loss, (s_param, r_param), retain_graph=True
        )
        gs_e, gr_e = torch.autograd.grad(res.energy.sum(), (s_param, r_param))
        return gs_q, gr_q, gs_e, gr_e

    u_gs_q, u_gr_q, u_gs_e, u_gr_e = run_mode("unrolled")
    i_gs_q, i_gr_q, i_gs_e, i_gr_e = run_mode("implicit")
    torch.testing.assert_close(i_gs_q, u_gs_q, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(i_gr_q, u_gr_q, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(i_gs_e, u_gs_e, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(i_gr_e, u_gr_e, rtol=1e-6, atol=1e-7)


def test_monopole_multipole_cross_hooks_potential_and_energy_consistency() -> (
    None
):
    """Monopole-dipole coupling reading qdp in get_monopole_atom_energy preserves variational forces."""

    class MonopoleDipoleCoupling(Interaction):
        __slots__ = ["coupling"]

        def __init__(self, coupling: Tensor) -> None:
            super().__init__(device=coupling.device, dtype=coupling.dtype)
            self.coupling = coupling

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> InteractionCache:
            _ = (numbers, positions, ihelp)
            return InteractionCache(device=self.device, dtype=self.dtype)

        @override
        def get_monopole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qqp)
            assert qdp is not None
            return self.coupling * qdp[..., 2]

        @override
        def get_dipole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qqp)
            assert qdp is not None
            out = torch.zeros_like(qdp)
            return torch.stack(
                [out[..., 0], out[..., 1], self.coupling * qat], dim=-1
            )

        @override
        def get_monopole_atom_energy(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (cache, qqp)
            assert qdp is not None
            return self.coupling * qat * qdp[..., 2]

    numbers, pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)

    def eval_coupling(diff_mode: str) -> tuple[Tensor, Tensor, Tensor]:
        k = torch.tensor(0.03, **DD64, requires_grad=True)
        p = pos.clone().requires_grad_(True)
        sys_c = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=diff_mode,
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=60,
            ),
            interaction=(MonopoleDipoleCoupling(k),),
        ).setup(numbers)
        e_tot = sys_c.singlepoint(p).energy.sum()
        gp, gk = torch.autograd.grad(e_tot, (p, k))
        return e_tot.detach(), gp, gk

    eu, gpu, gku = eval_coupling("unrolled")
    ei, gpi, gki = eval_coupling("implicit")
    torch.testing.assert_close(ei, eu, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(gpi, gpu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gki, gku, rtol=1e-6, atol=1e-7)


@dataclass(frozen=True)
class _FrozenSpringConfig:
    """Frozen dataclass container with an init=False metadata attribute."""

    k: Tensor
    tag: str = field(init=False, default="harmonic")


class _NamedPair(NamedTuple):
    """NamedTuple holding floating and integer tensors."""

    scale: Tensor
    indices: Tensor


def test_dataclass_and_namedtuple_containers_dtype_and_shared_aliases() -> None:
    """Dataclass and NamedTuple containers convert on Calculator.type() while preserving shared aliases."""

    class ContainerClassical(Classical):
        def __init__(self, cfg: _FrozenSpringConfig, pair: _NamedPair) -> None:
            super().__init__(device=cfg.k.device, dtype=cfg.k.dtype)
            self.cfg = cfg
            self.pair = pair

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> tuple[_FrozenSpringConfig, _NamedPair]:
            _ = (numbers, ihelp)
            return (self.cfg, self.pair)

        @override
        def get_energy(
            self,
            positions: Tensor,
            cache: tuple[_FrozenSpringConfig, _NamedPair],
            **_: Any,
        ) -> Tensor:
            cfg_c, pair_c = cache
            sel = positions.index_select(-2, pair_c.indices)
            val = (cfg_c.k + pair_c.scale) * sel.square().sum()
            return (
                positions.new_zeros(positions.shape[:-1])
                + val / positions.shape[-2]
            )

    class ContainerInteraction(Interaction):
        def __init__(self, cfg: _FrozenSpringConfig) -> None:
            super().__init__(device=cfg.k.device, dtype=cfg.k.dtype)
            self.cfg = cfg

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> _FrozenSpringConfig:
            _ = (numbers, positions, ihelp)
            return self.cfg

        @override
        def get_monopole_atom_potential(
            self,
            cache: _FrozenSpringConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (qdp, qqp)
            return 2.0 * cache.k * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: _FrozenSpringConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (qdp, qqp)
            return cache.k * qat.square()

    numbers, pos64 = _lih_inputs()
    cls64 = ContainerClassical(
        _FrozenSpringConfig(torch.tensor(0.03, **DD64)),
        _NamedPair(
            torch.tensor(0.02, **DD64),
            torch.tensor([0, 1], device=DEVICE, dtype=torch.int64),
        ),
    )
    int64 = ContainerInteraction(
        _FrozenSpringConfig(torch.tensor(0.04, **DD64))
    )
    calc = Calculator(
        numbers,
        GFN1_XTB,
        classical=(cls64,),
        interaction=(int64,),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    e64 = calc.energy(pos64)
    calc32 = calc.type(torch.float32)
    e32 = calc32.energy(pos64.to(dtype=torch.float32))
    assert e32.dtype == torch.float32
    torch.testing.assert_close(
        e32.to(dtype=torch.float64), e64, rtol=1e-4, atol=1e-4
    )

    # Shared tensor aliases between component and cache stay deduplicated (2 float leaves, not 4)
    core32 = calc32.system.exact_core
    assert core32 is not None
    custom_term = core32.classical_terms[-1]
    assert isinstance(custom_term.setup, CustomClassicalCoreSetup)
    assert len(custom_term.setup.floating_tensors) == 2
    comp_view, cache_view = custom_term.setup.reconstruct()
    assert cache_view[0].k is comp_view.cfg.k


def test_non_tensor_scalar_mutation_and_string_dd_device() -> None:
    """String dd['device'] builds CustomClassicalCoreSetup, and direct scalar mutation invalidates it."""

    class ScalarAttrClassical(Classical):
        def __init__(self, scale: float) -> None:
            super().__init__(device=DEVICE, dtype=DTYPE)
            self.scale = scale

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return torch.full(numbers.shape, self.scale, **DD64)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return cache * positions.square().sum(dim=-1)

    numbers, pos = _lih_inputs()
    comp = ScalarAttrClassical(1.0)
    sys_s = Model(
        par=ParamModule(GFN1_XTB, **DD64),
        config=Config.create(int_driver="pytorch"),
        classical=(comp,),
    ).setup(numbers, dd={"device": "cpu", "dtype": DTYPE})
    assert can_vmap_system(sys_s)

    e_before = dict(sys_s.singlepoint(pos).classical)[comp.label].sum()
    comp.scale = 10.0
    e_after = dict(sys_s.singlepoint(pos).classical)[comp.label].sum()
    torch.testing.assert_close(e_after, 10.0 * e_before)


def test_exact_core_tree_stack_and_duplicate_label_non_tensor_classicals() -> (
    None
):
    """ExactSystemCore stacks across parameter values, and duplicate-label non-tensor Classicals keep setups."""

    class SliceTaggedHarmonic(Classical):
        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k
            self.atom_slice = slice(0, None)

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return self.k * (numbers[self.atom_slice] > 0).to(
                dtype=self.k.dtype
            )

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return 0.5 * cache * positions.square().sum(dim=-1)

    numbers, pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")
    sys_1 = Model(
        par=par,
        config=cfg,
        classical=(SliceTaggedHarmonic(torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    sys_2 = Model(
        par=par,
        config=cfg,
        classical=(SliceTaggedHarmonic(torch.tensor(0.12, **DD64)),),
    ).setup(numbers)
    assert sys_1.exact_core is not None and sys_2.exact_core is not None
    stacked_core = tree_stack((sys_1.exact_core, sys_2.exact_core))
    stacked_custom = stacked_core.classical_terms[-1].setup
    assert isinstance(stacked_custom, CustomClassicalCoreSetup)
    assert stacked_custom.floating_tensors[0].shape[0] == 2

    # Two non-tensor-only custom Classicals sharing a label keep independent id-keyed setups
    class OpaqueTag:
        def __init__(self, factor: float) -> None:
            self.factor = factor

    class DuplicateLabelOpaqueClassical(Classical):
        def __init__(self, factor: float) -> None:
            super().__init__(device=DEVICE, dtype=DTYPE)
            self.opaque = OpaqueTag(factor)
            self.cache_calls = 0
            self.label = "SharedCustomClassical"

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.cache_calls += 1
            return torch.full(numbers.shape, self.opaque.factor, **DD64)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return cache * positions.square().sum(dim=-1)

    c_a = DuplicateLabelOpaqueClassical(0.05)
    c_b = DuplicateLabelOpaqueClassical(0.20)
    c_b.label = "InitialB"
    calc_dup = Calculator(
        numbers,
        GFN1_XTB,
        classical=(c_a, c_b),
        opts={"verbosity": 0, "exclude": ["all"]},
        **DD64,
    )
    c_b.label = "SharedCustomClassical"
    eval_sys = calc_dup._prepare_eval_system()
    calls_after_setup = (c_a.cache_calls, c_b.cache_calls)
    res_1 = eval_sys.singlepoint(pos)
    res_2 = eval_sys.singlepoint(pos)
    assert (c_a.cache_calls, c_b.cache_calls) == calls_after_setup
    expected = (0.05 + 0.20) * pos.square().sum()
    torch.testing.assert_close(res_1.energy.sum(), expected)
    torch.testing.assert_close(res_2.energy.sum(), expected)

    c_a_2d = DuplicateLabelOpaqueClassical(0.05)
    c_b_2d = DuplicateLabelOpaqueClassical(0.20)
    calc_dup_2d = Calculator(
        numbers.unsqueeze(0),
        GFN1_XTB,
        classical=(c_a_2d, c_b_2d),
        opts={"verbosity": 0, "exclude": ["all"], "batch_mode": 2},
        **DD64,
    )
    res_2d = calc_dup_2d.singlepoint(pos.unsqueeze(0))
    torch.testing.assert_close(res_2d.energy.sum(), expected)


def test_batch_implicit_second_order_derivatives_with_custom_interaction() -> (
    None
):
    """evaluate_converged_root_batch supports create_graph=True 2nd-order derivatives with custom Interaction."""
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)

    def second_deriv(diff_mode: str) -> tuple[Tensor, Tensor]:
        s_param = torch.tensor(0.025, **DD64, requires_grad=True)
        h_param = torch.tensor(0.09, **DD64, requires_grad=True)
        pos_batch = torch.stack([pos_a, pos_b], dim=0)
        cfg = Config.create(
            int_driver="pytorch",
            differentiation=diff_mode,
            x_atol=1.0e-11,
            x_atol_max=1.0e-11,
            maxiter=80,
        )
        sys_b = Model(
            par=par,
            config=cfg,
            interaction=(OnsiteShift(s_param, h_param),),
        ).setup(numbers)
        if diff_mode == "implicit":
            prepared = prepare_converged_root_batch(sys_b, pos_batch)
            loss = evaluate_converged_root_batch(prepared).energy.sum()
        else:
            loss = (
                sys_b.singlepoint(pos_batch[0]).energy.sum()
                + sys_b.singlepoint(pos_batch[1]).energy.sum()
            )
        (g1,) = torch.autograd.grad(loss, s_param, create_graph=True)
        (g2,) = torch.autograd.grad(g1, s_param)
        return g1.detach(), g2.detach()

    g1_i, g2_i = second_deriv("implicit")
    g1_u, g2_u = second_deriv("unrolled")
    torch.testing.assert_close(g1_i, g1_u, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(g2_i, g2_u, rtol=1e-4, atol=1e-5)
