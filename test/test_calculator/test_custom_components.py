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
from tad_mctc.tree import ModuleNode, Node, child, context
from tad_mctc.tree import stack as tree_stack
from torch.func import grad as func_grad
from torch.func import hessian, jacrev, vmap

from dxtb import (
    GFN1_XTB,
    Calculator,
    IndexHelper,
    OutputHandler,
    ParamModule,
    labels,
)
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
from dxtb._src.calculators.packed_result import (
    _sync_lane_system_labels,
    build_calculator_lane_systems,
    evaluate_packed_systems,
)
from dxtb._src.components.custom import (
    CustomClassicalCoreSetup,
    prepare_custom_interaction_entry,
)
from dxtb._src.components.interactions import (
    InteractionList,
    InteractionListCache,
)
from dxtb._src.scf.data import SCFCharges
from dxtb._src.scf.interactions import (
    interaction_energy_from_charges,
    potential_from_charges,
    prepare_scf_interactions,
)
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


class _SharedPairNode(Node):
    """Frozen Node container with two child tensor slots for alias testing."""

    a: Tensor = child()
    b: Tensor = child()


class _CustomConvertChildNode(Node):
    """Node overriding _convert_child on a keep_dtype=True optional floating tensor."""

    primary: Tensor = child()
    optional_float: Tensor | None = child(keep_dtype=True, default=None)

    @override
    def _convert_child(
        self,
        name: str,
        value: Any,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> Any:
        if name == "optional_float" and isinstance(value, Tensor):
            return value.to(device=device, dtype=dtype)
        return super()._convert_child(name, value, device, dtype)


class _KeepDtypeFloatNode(Node):
    """Default Node keeping a reference float64 tensor on keep_dtype=True."""

    primary: Tensor = child()
    ref_fp64: Tensor = child(keep_dtype=True)


class _DualBoxNode(Node):
    """Default Node sharing a container across keep_dtype=True and normal child fields."""

    kept_box: list[Tensor] = child(keep_dtype=True)
    normal_box: list[Tensor] = child()


class _ContextMetaNode(Node):
    """Default Node storing arbitrary metadata in a context() field."""

    primary: Tensor = child()
    meta: Any = context()
    items: Any = child(keep_dtype=True, default=None)


@dataclass(frozen=True)
class _PostInitFrozenTensorConfig:
    """Frozen dataclass initializing an init=False Tensor in __post_init__."""

    base: Tensor
    derived: Tensor = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "derived", 2.0 * self.base)


@dataclass
class _PostInitMutableTensorConfig:
    """Mutable dataclass initializing an init=False Tensor in __post_init__."""

    base: Tensor
    derived: Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.derived = 2.0 * self.base


def test_duplicate_interaction_labels_rejected_across_setup_and_eval_modes() -> (
    None
):
    """Distinct custom interactions with colliding labels raise ValueError across all modes."""
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.25]], **DD64)
    nums_bm2 = torch.stack([numbers, numbers], dim=0)
    pos_bm2 = torch.stack([pos_a, pos_b], dim=0)
    nums_bm1 = torch.tensor(
        [[8, 1, 1], [3, 1, 0]], device=DEVICE, dtype=torch.int64
    )
    pos_bm1 = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
        ],
        **DD64,
    )
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")

    def make_pair() -> tuple[OnsiteShift, OnsiteShift]:
        inter_1 = OnsiteShift(
            torch.tensor(0.015, **DD64), torch.tensor(0.08, **DD64)
        )
        inter_2 = OnsiteShift(
            torch.tensor(0.045, **DD64), torch.tensor(0.22, **DD64)
        )
        return inter_1, inter_2

    # 1. Initial Model.setup rejects duplicate labels in batch_mode 0, 1, and 2
    for bm, nums in ((0, numbers), (1, nums_bm1), (2, nums_bm2)):
        i1, i2 = make_pair()
        with pytest.raises(
            ValueError,
            match="Distinct interactions in a single System share the label 'OnsiteShift'",
        ):
            Model(par=par, config=cfg, interaction=(i1, i2)).setup(
                nums, batch_mode=bm
            )

    # 2. Batched Calculator construction rejects duplicate labels in batch_mode 1 and 2
    for bm, nums in ((1, nums_bm1), (2, nums_bm2)):
        i1, i2 = make_pair()
        with pytest.raises(
            ValueError,
            match="Distinct interactions in a single System share the label 'OnsiteShift'",
        ):
            Calculator(
                nums,
                GFN1_XTB,
                interaction=(i1, i2),
                opts={
                    "verbosity": 0,
                    "int_driver": "pytorch",
                    "batch_mode": bm,
                },
                **DD64,
            )

    # 3. Post-construction label collision on 1D, batch_mode=2, and batch_mode=1 Calculators,
    #    followed by clean evaluation once unique labels are restored
    s1 = torch.tensor(0.015, **DD64, requires_grad=True)
    h1 = torch.tensor(0.08, **DD64, requires_grad=True)
    s2 = torch.tensor(0.045, **DD64, requires_grad=True)
    h2 = torch.tensor(0.22, **DD64, requires_grad=True)

    for bm, nums, eval_pos in (
        (0, numbers, pos_a),
        (2, nums_bm2, pos_bm2),
        (1, nums_bm1, pos_bm1),
    ):
        inter_a = OnsiteShift(s1, h1)
        inter_b = OnsiteShift(s2, h2)
        inter_b.label = "OnsiteShiftSecond"
        opts: dict[str, Any] = {"verbosity": 0, "int_driver": "pytorch"}
        if bm != 0:
            opts["batch_mode"] = bm
        calc = Calculator(
            nums, GFN1_XTB, interaction=(inter_a, inter_b), opts=opts, **DD64
        )
        inter_b.label = "OnsiteShift"
        with pytest.raises(
            ValueError,
            match="Distinct interactions in a single System share the label 'OnsiteShift'",
        ):
            calc.singlepoint(eval_pos)
        # Caller-owned labels are never silently renamed
        assert inter_a.label == "OnsiteShift"
        assert inter_b.label == "OnsiteShift"

        # Restoring unique labels succeeds and matches combined single-interaction physics
        inter_b.label = "OnsiteShiftRestored"
        res_pair = calc.singlepoint(eval_pos)
        combined = OnsiteShift(s1 + s2, h1 + h2)
        calc_ref = Calculator(
            nums, GFN1_XTB, interaction=(combined,), opts=opts, **DD64
        )
        res_ref = calc_ref.singlepoint(eval_pos)
        torch.testing.assert_close(
            res_pair.energy, res_ref.energy, rtol=1e-11, atol=1e-12
        )
        if bm == 0:
            grads_pair = torch.autograd.grad(
                res_pair.energy.sum(), (s1, h1, s2, h2)
            )
            grads_ref = torch.autograd.grad(
                res_ref.energy.sum(), (s1, h1, s2, h2)
            )
            for gp, gr in zip(grads_pair, grads_ref):
                torch.testing.assert_close(gp, gr, rtol=1e-10, atol=1e-11)

    # 4. Direct post-setup label collision on System is also rejected before SCF
    ia_sys, ib_sys = make_pair()
    ib_sys.label = "OnsiteShiftDistinct"
    sys_direct = Model(par=par, config=cfg, interaction=(ia_sys, ib_sys)).setup(
        numbers
    )
    ib_sys.label = "OnsiteShift"
    with pytest.raises(
        ValueError,
        match="Distinct interactions in a single System share the label 'OnsiteShift'",
    ):
        sys_direct.singlepoint(pos_a)
    with pytest.raises(
        ValueError,
        match="Distinct interactions in a single System share the label 'OnsiteShift'",
    ):
        evaluate_packed_systems((sys_direct, sys_direct), (pos_a, pos_b))

    ia_excl, ib_excl = make_pair()
    ib_excl.label = "OnsiteShiftDistinct"
    sys_excl = Model(
        par=par,
        config=Config.create(int_driver="pytorch", exclude="scf"),
        interaction=(ia_excl, ib_excl),
    ).setup(numbers)
    ib_excl.label = "OnsiteShift"
    with pytest.raises(
        ValueError,
        match="Distinct interactions in a single System share the label 'OnsiteShift'",
    ):
        sys_excl.singlepoint(pos_a)


def test_post_eval_and_external_component_mutations_invalidate_setup() -> None:
    """Scalar/tensor mutations inside get_energy or between calls invalidate captured setups."""

    class _OpaqueHandle:
        def __init__(self, tag: str) -> None:
            self.tag = tag

    class SelfMutatingTensorClassical(Classical):
        """Tensor-only Classical whose get_cache depends on self.scale and get_energy mutates it."""

        def __init__(self, scale: float, weight: Tensor) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.scale = scale
            self.weight = weight
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.cache_calls += 1
            return (
                self.scale * self.weight * numbers.to(dtype=self.weight.dtype)
            )

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            out = 0.5 * cache * positions.square().sum(dim=-1)
            self.scale = 2.5
            return out

    class SelfMutatingOpaqueClassical(Classical):
        """Opaque Classical using fallback snapshots whose get_energy mutates self.scale."""

        def __init__(self, scale: float, weight: Tensor) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.scale = scale
            self.weight = weight
            self.handle = _OpaqueHandle("opaque")
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.cache_calls += 1
            return (
                self.scale * self.weight * numbers.to(dtype=self.weight.dtype)
            )

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            out = 0.5 * cache * positions.square().sum(dim=-1)
            self.scale = 2.5
            return out

    numbers, pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")
    w = torch.tensor(0.03, **DD64)

    for cls_type, expects_core in (
        (SelfMutatingTensorClassical, True),
        (SelfMutatingOpaqueClassical, False),
    ):
        comp = cls_type(1.0, w.clone())
        sys_mut = Model(par=par, config=cfg, classical=(comp,)).setup(numbers)
        assert can_vmap_system(sys_mut) is expects_core
        assert comp.cache_calls == 1

        ref_1 = (
            Model(par=par, config=cfg, classical=(cls_type(1.0, w.clone()),))
            .setup(numbers)
            .singlepoint(pos)
        )
        res_1 = sys_mut.singlepoint(pos)
        torch.testing.assert_close(
            res_1.energy, ref_1.energy, rtol=1e-12, atol=1e-12
        )
        assert comp.scale == 2.5
        assert comp.cache_calls == 1

        # Second evaluation must NOT reuse the stale scale=1.0 setup
        ref_2 = (
            Model(par=par, config=cfg, classical=(cls_type(2.5, w.clone()),))
            .setup(numbers)
            .singlepoint(pos)
        )
        res_2 = sys_mut.singlepoint(pos)
        assert comp.cache_calls == 2
        torch.testing.assert_close(
            res_2.energy, ref_2.energy, rtol=1e-12, atol=1e-12
        )

        # External tensor replacement and in-place modification also invalidate setup
        comp.weight = torch.tensor(0.09, **DD64)
        res_3 = sys_mut.singlepoint(pos)
        ref_3 = (
            Model(
                par=par,
                config=cfg,
                classical=(cls_type(2.5, torch.tensor(0.09, **DD64)),),
            )
            .setup(numbers)
            .singlepoint(pos)
        )
        torch.testing.assert_close(
            res_3.energy, ref_3.energy, rtol=1e-12, atol=1e-12
        )

        with torch.no_grad():
            comp.weight.mul_(0.5)
        res_4 = sys_mut.singlepoint(pos)
        ref_4 = (
            Model(
                par=par,
                config=cfg,
                classical=(cls_type(2.5, torch.tensor(0.045, **DD64)),),
            )
            .setup(numbers)
            .singlepoint(pos)
        )
        torch.testing.assert_close(
            res_4.energy, ref_4.energy, rtol=1e-12, atol=1e-12
        )


def test_shared_tensor_aliases_preserved_across_type_to_and_transforms() -> (
    None
):
    """Shared tensor references stay identical across .type(), .to(), ComponentCache, and Calculator.type()."""

    class AliasCache(ClassicalCache):
        __slots__ = ["a", "b", "idx_x", "idx_y"]

        def __init__(
            self, a: Tensor, b: Tensor, idx_x: Tensor, idx_y: Tensor
        ) -> None:
            super().__init__(device=a.device, dtype=a.dtype)
            self.a = a
            self.b = b
            self.idx_x = idx_x
            self.idx_y = idx_y

    class RichAliasClassical(Classical):
        __slots__ = [
            "left",
            "right",
            "nested_list",
            "nested_dict",
            "named_pair",
            "dc_cfg",
            "node_box",
            "idx_a",
            "idx_b",
        ]

        def __init__(self, shared: Tensor, idx: Tensor) -> None:
            super().__init__(device=shared.device, dtype=shared.dtype)
            self.left = shared
            self.right = shared
            self.extra_shared = shared
            self.nested_list = [shared, (shared,)]
            self.nested_dict = {"primary": shared}
            self.named_pair = _NamedPair(shared, idx)
            self.dc_cfg = _FrozenSpringConfig(shared)
            self.node_box = _SharedPairNode(a=shared, b=shared)
            self.idx_a = idx
            self.idx_b = idx

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> AliasCache:
            _ = (numbers, ihelp)
            return AliasCache(self.left, self.right, self.idx_a, self.idx_b)

        @override
        def get_energy(
            self, positions: Tensor, cache: AliasCache, **_: Any
        ) -> Tensor:
            sel = positions.index_select(-2, cache.idx_x)
            total_k = (
                cache.a
                + cache.b
                + self.extra_shared
                + self.nested_list[0]
                + self.nested_dict["primary"]
                + self.named_pair.scale
                + self.dc_cfg.k
                + self.node_box.a
            )
            contrib = 0.125 * total_k * sel.square().sum()
            return (
                positions.new_zeros(positions.shape[:-1])
                + contrib / positions.shape[-2]
            )

    numbers, pos64 = _lih_inputs()
    shared64 = torch.tensor(0.04, **DD64, requires_grad=True)
    idx64 = torch.tensor([0, 1], device=DEVICE, dtype=torch.int64)
    comp64 = RichAliasClassical(shared64, idx64)

    # 1. Component.type(float32) preserves all aliases and leaves original untouched
    comp32 = comp64.type(torch.float32)
    assert comp32 is not comp64
    assert comp64.left is shared64 and comp64.left.dtype == torch.float64
    assert comp32.left.dtype == torch.float32
    assert comp32.left is comp32.right
    assert comp32.left is comp32.extra_shared
    assert comp32.left is comp32.nested_list[0]
    assert comp32.left is comp32.nested_list[1][0]
    assert comp32.left is comp32.nested_dict["primary"]
    assert comp32.left is comp32.named_pair.scale
    assert comp32.left is comp32.dc_cfg.k
    assert comp32.left is comp32.node_box.a
    assert comp32.node_box.a is comp32.node_box.b
    assert comp32.idx_a is comp32.idx_b
    assert comp32.idx_a.dtype == torch.int64

    # 2. ComponentCache conversion and cross-device .to(device="meta") preserve aliases
    cache64 = comp64.get_cache(numbers)
    cache32 = cache64.type(torch.float32)
    assert cache32.a is cache32.b and cache32.a.dtype == torch.float32
    assert cache32.idx_x is cache32.idx_y and cache32.idx_x.dtype == torch.int64

    comp_meta = comp64.to(device=torch.device("meta"))
    assert comp_meta.left.device.type == "meta"
    assert comp_meta.left is comp_meta.right
    assert comp_meta.left is comp_meta.extra_shared
    assert comp_meta.left is comp_meta.node_box.a
    assert comp_meta.idx_a is comp_meta.idx_b
    assert comp_meta.idx_a.device.type == "meta"
    assert comp_meta.idx_a.dtype == torch.int64

    # Node subclasses overriding _convert_child convert keep_dtype=True floating fields properly
    class CustomConvertNodeClassical(Classical):
        __slots__ = ["custom_node", "shared_list"]

        def __init__(
            self, cnode: _CustomConvertChildNode, slist: list[Tensor]
        ) -> None:
            super().__init__(
                device=cnode.primary.device, dtype=cnode.primary.dtype
            )
            self.custom_node = cnode
            self.shared_list = slist

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> list[Tensor]:
            _ = (numbers, ihelp)
            return self.shared_list

        @override
        def get_energy(
            self, positions: Tensor, cache: list[Tensor], **_: Any
        ) -> Tensor:
            opt = self.custom_node.optional_float
            assert opt is not None
            return 0.5 * (cache[0] + opt) * positions.square().sum(dim=-1)

    cnode64 = _CustomConvertChildNode(
        primary=torch.tensor(0.02, **DD64),
        optional_float=torch.tensor(0.03, **DD64),
    )
    slist64 = [shared64]
    calc_cnode = Calculator(
        numbers,
        GFN1_XTB,
        classical=(CustomConvertNodeClassical(cnode64, slist64),),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    ).type(torch.float32)
    conv_cnode_cls = calc_cnode.classicals.components[-1]
    assert isinstance(conv_cnode_cls, CustomConvertNodeClassical)
    assert conv_cnode_cls.custom_node.primary.dtype == torch.float32
    assert conv_cnode_cls.custom_node.optional_float is not None
    assert conv_cnode_cls.custom_node.optional_float.dtype == torch.float32
    _, _, snap_setup32 = calc_cnode.system._custom_classical_snapshots[-1]
    assert snap_setup32 is conv_cnode_cls.shared_list

    if torch.cuda.is_available():
        comp_cuda = comp64.to(device=torch.device("cuda:0"))
        assert comp_cuda.left is comp_cuda.right
        assert comp_cuda.left.device.type == "cuda"

    # 3. Calculator.type(float32) preserves aliases across classical, interaction, and setup,
    #    and backpropagates cleanly to the single original shared64 parameter
    inter64 = OnsiteShift(shared64, shared64)
    calc = Calculator(
        numbers,
        GFN1_XTB,
        classical=(RichAliasClassical(shared64, idx64),),
        interaction=(inter64,),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    calc32 = calc.type(torch.float32)
    cls_in_calc = calc32.classicals.components[-1]
    int_in_calc = calc32.interactions.components[-1]
    assert isinstance(cls_in_calc, RichAliasClassical)
    assert isinstance(int_in_calc, OnsiteShift)
    assert cls_in_calc.left is cls_in_calc.right
    assert int_in_calc.scale is int_in_calc.hardness
    assert cls_in_calc.left is int_in_calc.scale

    pos32 = pos64.to(dtype=torch.float32)
    e32 = calc32.energy(pos32)
    (grad_shared_from_32,) = torch.autograd.grad(e32.sum(), shared64)
    assert torch.isfinite(grad_shared_from_32)

    # Compare against float64 reference gradient and torch.func.grad
    par64 = ParamModule(GFN1_XTB, **DD64)
    cfg64 = Config.create(int_driver="pytorch")

    def energy_for_shared(s_val: Tensor) -> Tensor:
        """Total System energy when one shared tensor drives both Classical and Interaction."""
        sys_shared = Model(
            par=par64,
            config=cfg64,
            classical=(RichAliasClassical(s_val, idx64),),
            interaction=(OnsiteShift(s_val, s_val),),
        ).setup(numbers)
        return sys_shared.singlepoint(pos64).energy.sum()

    func_g = func_grad(energy_for_shared)(shared64.detach())
    e64_ref = energy_for_shared(shared64)
    (auto_g,) = torch.autograd.grad(e64_ref, shared64)
    torch.testing.assert_close(func_g, auto_g, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(
        grad_shared_from_32, auto_g, rtol=1e-3, atol=1e-4
    )


def test_frozen_dataclass_init_false_tensor_explicitly_rejected() -> None:
    """Frozen dataclasses with init=False tensors raise TypeError; mutable equivalents work."""

    class BadFrozenClassical(Classical):
        def __init__(self, cfg: _PostInitFrozenTensorConfig) -> None:
            super().__init__(device=cfg.base.device, dtype=cfg.base.dtype)
            self.cfg = cfg

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> _PostInitFrozenTensorConfig:
            _ = (numbers, ihelp)
            return self.cfg

        @override
        def get_energy(
            self,
            positions: Tensor,
            cache: _PostInitFrozenTensorConfig,
            **_: Any,
        ) -> Tensor:
            return 0.5 * cache.derived * positions.square().sum(dim=-1)

    class BadFrozenInteraction(Interaction):
        def __init__(self, cfg: _PostInitFrozenTensorConfig) -> None:
            super().__init__(device=cfg.base.device, dtype=cfg.base.dtype)
            self.cfg = cfg

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> _PostInitFrozenTensorConfig:
            _ = (numbers, positions, ihelp)
            return self.cfg

        @override
        def get_monopole_atom_potential(
            self,
            cache: _PostInitFrozenTensorConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (qdp, qqp)
            return cache.derived * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: _PostInitFrozenTensorConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (qdp, qqp)
            return 0.5 * cache.derived * qat.square()

    numbers, pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)
    bad_cfg = _PostInitFrozenTensorConfig(torch.tensor(0.05, **DD64))
    bad_cls = BadFrozenClassical(bad_cfg)
    bad_int = BadFrozenInteraction(bad_cfg)

    with pytest.raises(TypeError, match="init=False"):
        bad_cls.type(torch.float32)
    with pytest.raises(TypeError, match="init=False"):
        bad_cls.to(dtype=torch.float64)
    with pytest.raises(TypeError, match="init=False"):
        Model(par=par, classical=(bad_cls,)).setup(numbers)
    with pytest.raises(TypeError, match="init=False"):
        Model(par=par, interaction=(bad_int,)).setup(numbers)
    with pytest.raises(TypeError, match="init=False"):
        prepare_custom_interaction_entry(bad_int, bad_cfg)

    # Mutable dataclass with __post_init__ derived tensor supports .type(), unrolled, and implicit
    class MutableDerivedClassical(Classical):
        def __init__(self, cfg: _PostInitMutableTensorConfig) -> None:
            super().__init__(device=cfg.base.device, dtype=cfg.base.dtype)
            self.cfg = cfg

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> _PostInitMutableTensorConfig:
            _ = (numbers, ihelp)
            return self.cfg

        @override
        def get_energy(
            self,
            positions: Tensor,
            cache: _PostInitMutableTensorConfig,
            **_: Any,
        ) -> Tensor:
            return (
                0.5
                * (cache.base + cache.derived)
                * positions.square().sum(dim=-1)
            )

    class MutableDerivedInteraction(Interaction):
        def __init__(self, cfg: _PostInitMutableTensorConfig) -> None:
            super().__init__(device=cfg.base.device, dtype=cfg.base.dtype)
            self.cfg = cfg

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> _PostInitMutableTensorConfig:
            _ = (numbers, positions, ihelp)
            return self.cfg

        @override
        def get_monopole_atom_potential(
            self,
            cache: _PostInitMutableTensorConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (qdp, qqp)
            return (cache.base + cache.derived) * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: _PostInitMutableTensorConfig,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (qdp, qqp)
            return 0.5 * (cache.base + cache.derived) * qat.square()

    mut_cls32 = MutableDerivedClassical(
        _PostInitMutableTensorConfig(torch.tensor(0.05, **DD64))
    ).type(torch.float32)
    assert mut_cls32.cfg.base.dtype == torch.float32
    assert mut_cls32.cfg.derived.dtype == torch.float32

    def eval_mutable_dc(diff_mode: str) -> tuple[Tensor, Tensor]:
        b_cls = torch.tensor(0.03, **DD64, requires_grad=True)
        b_int = torch.tensor(0.04, **DD64, requires_grad=True)
        sys_m = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=diff_mode,
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=70,
            ),
            classical=(
                MutableDerivedClassical(_PostInitMutableTensorConfig(b_cls)),
            ),
            interaction=(
                MutableDerivedInteraction(_PostInitMutableTensorConfig(b_int)),
            ),
        ).setup(numbers)
        loss = sys_m.singlepoint(pos).energy.sum()
        g_cls, g_int = torch.autograd.grad(loss, (b_cls, b_int))
        return g_cls, g_int

    gc_u, gi_u = eval_mutable_dc("unrolled")
    gc_i, gi_i = eval_mutable_dc("implicit")
    # Direct reference: base + derived = 3 * base (0.5 * 3 * b_cls * sum(r^2) for classical)
    b_cls_ref = torch.tensor(0.03, **DD64, requires_grad=True)
    b_int_ref = torch.tensor(0.04, **DD64, requires_grad=True)
    sys_ref = Model(
        par=par,
        config=Config.create(
            int_driver="pytorch",
            differentiation="unrolled",
            x_atol=1.0e-11,
            x_atol_max=1.0e-11,
            maxiter=70,
        ),
        interaction=(OnsiteShift(torch.tensor(0.0, **DD64), 3.0 * b_int_ref),),
    ).setup(numbers)
    ref_total = (
        sys_ref.singlepoint(pos).energy.sum()
        + 0.5 * (3.0 * b_cls_ref) * pos.square().sum()
    )
    gc_ref, gi_ref = torch.autograd.grad(ref_total, (b_cls_ref, b_int_ref))
    torch.testing.assert_close(gc_i, gc_u, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gc_i, gc_ref, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(gi_i, gi_u, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gi_i, gi_ref, rtol=1e-6, atol=1e-7)


def test_vmap_fallback_safety_user_errors_mutation_and_source_id() -> None:
    """Automatic vmap fallback rejects stateful mutations and preserves unrelated user exceptions."""
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.3]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")

    # 1. Pure data-dependent component using .item() falls back cleanly; if one lane then
    #    raises a data-dependent ValueError, source_id identifies that exact source lane
    class ThresholdItemClassical(Classical):
        __slots__ = ["k"]

        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = (numbers, ihelp)
            return self.k

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            dist = torch.linalg.norm(
                positions[..., 1, :] - positions[..., 0, :]
            )
            if dist.item() > 3.2:
                raise ValueError("bond stretched past cutoff")
            return 0.5 * cache * positions.square().sum(dim=-1)

    sys_thresh = Model(
        par=par,
        config=cfg,
        classical=(ThresholdItemClassical(torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    assert can_vmap_system(sys_thresh)
    group_thresh = LaneGroup(
        system=sys_thresh, source_ids=(4, 19), use_vmap=True
    )
    pos_map = {4: pos_a, 19: pos_b}
    chrg_map = {
        4: torch.tensor(0.0, **DD64),
        19: torch.tensor(0.0, **DD64),
    }
    none_map: dict[int, Tensor | None] = {4: None, 19: None}
    with pytest.raises(
        ValueError, match="bond stretched past cutoff"
    ) as exc_lane:
        evaluate_lane_group(
            group_thresh, pos_map, chrg_map, none_map, none_map, none_map
        )
    assert getattr(exc_lane.value, "source_id", None) == 19

    # 2. Component modifying Python state before hitting a vmap limitation is NOT retried
    class ImpureItemClassical(Classical):
        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k
            self.calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = (numbers, ihelp)
            return self.k

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            self.calls += 1
            scale = 1.0 if positions.sum().item() > 0.0 else 0.5
            return scale * cache * positions.square().sum(dim=-1)

    impure = ImpureItemClassical(torch.tensor(0.04, **DD64))
    sys_impure = Model(par=par, config=cfg, classical=(impure,)).setup(numbers)
    assert can_vmap_system(sys_impure)
    group_impure = LaneGroup(
        system=sys_impure, source_ids=(7, 11), use_vmap=True
    )
    pos_impure = {7: pos_a, 11: pos_b}
    chrg_impure = {
        7: torch.tensor(0.0, **DD64),
        11: torch.tensor(0.0, **DD64),
    }
    none_impure: dict[int, Tensor | None] = {7: None, 11: None}
    with pytest.raises(
        RuntimeError, match="mutated state during the mapped attempt"
    ) as exc_mut:
        evaluate_lane_group(
            group_impure,
            pos_impure,
            chrg_impure,
            none_impure,
            none_impure,
            none_impure,
        )
    assert impure.calls == 1
    assert getattr(exc_mut.value, "source_id", None) == 7

    # 3. Custom hook raising an unrelated RuntimeError mentioning "vmap" is never swallowed or retried
    user_calls = 0

    class UserErrorInteraction(Interaction):
        __slots__ = ["k"]

        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k

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
            nonlocal user_calls
            _ = (cache, qat, qdp, qqp)
            user_calls += 1
            raise RuntimeError(
                "vmap: custom domain validation failed inside user hook"
            )

    sys_uerr = Model(
        par=par,
        config=cfg,
        interaction=(UserErrorInteraction(torch.tensor(0.02, **DD64)),),
    ).setup(numbers)
    assert can_vmap_system(sys_uerr)
    group_uerr = LaneGroup(system=sys_uerr, source_ids=(3, 8), use_vmap=True)
    pos_uerr = {3: pos_a, 8: pos_b}
    chrg_uerr = {
        3: torch.tensor(0.0, **DD64),
        8: torch.tensor(0.0, **DD64),
    }
    none_uerr: dict[int, Tensor | None] = {3: None, 8: None}
    with pytest.raises(
        RuntimeError, match="custom domain validation failed inside user hook"
    ) as exc_uerr:
        evaluate_lane_group(
            group_uerr, pos_uerr, chrg_uerr, none_uerr, none_uerr, none_uerr
        )
    assert user_calls == 1
    assert getattr(exc_uerr.value, "source_id", None) == 3

    # 4. Cache-only mutation before hitting a vmap limitation is ALSO caught and not retried
    class CacheMutatingCache(ClassicalCache):
        __slots__ = ["k", "calls"]

        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k
            self.calls = 0

    class CacheMutatingClassical(Classical):
        __slots__ = ["k"]

        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> CacheMutatingCache:
            _ = (numbers, ihelp)
            return CacheMutatingCache(self.k)

        @override
        def get_energy(
            self, positions: Tensor, cache: CacheMutatingCache, **_: Any
        ) -> Tensor:
            cache.calls += 1
            _ = positions.sum().item()
            return 0.5 * cache.k * positions.square().sum(dim=-1)

    sys_cmut = Model(
        par=par,
        config=cfg,
        classical=(CacheMutatingClassical(torch.tensor(0.04, **DD64)),),
    ).setup(numbers)
    assert can_vmap_system(sys_cmut)
    group_cmut = LaneGroup(system=sys_cmut, source_ids=(2, 5), use_vmap=True)
    with pytest.raises(
        RuntimeError, match="mutated state during the mapped attempt"
    ) as exc_cmut:
        evaluate_lane_group(
            group_cmut,
            {2: pos_a, 5: pos_b},
            {2: torch.tensor(0.0, **DD64), 5: torch.tensor(0.0, **DD64)},
            {2: None, 5: None},
            {2: None, 5: None},
            {2: None, 5: None},
        )
    assert getattr(exc_cmut.value, "source_id", None) == 2

    # 5. Generator-expression .item() falls back cleanly without duplicating SCFConvergenceWarning
    class GenexprItemClassical(Classical):
        __slots__ = ["k"]

        def __init__(self, k: Tensor) -> None:
            super().__init__(device=k.device, dtype=k.dtype)
            self.k = k

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = (numbers, ihelp)
            return self.k

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            coord_sum = sum(v.sum().item() for v in positions.unbind(-1))
            factor = 1.0 if coord_sum >= 0.0 else 0.5
            return 0.5 * factor * cache * positions.square().sum(dim=-1)

    sys_gen_warn = Model(
        par=par,
        config=Config.create(
            int_driver="pytorch", maxiter=1, force_convergence=False
        ),
        classical=(GenexprItemClassical(torch.tensor(0.03, **DD64)),),
    ).setup(numbers)
    OutputHandler.clear_warnings()
    packed_gen = evaluate_packed_systems(
        (sys_gen_warn, sys_gen_warn),
        (pos_a, pos_b),
        eager_status_policy=True,
    )
    assert torch.isfinite(packed_gen.energy).all()
    assert len(OutputHandler.warnings) == 2
    OutputHandler.clear_warnings()


def test_nonlinear_multipole_interaction_variational_and_second_derivatives() -> (
    None
):
    """Nonlinear shell/atom/dipole/quadrupole interaction satisfies variational potential = dE/dq and 2nd-order implicit parity."""

    class NonlinearMultipolarInteraction(Interaction):
        """Anharmonic charge + saturating dipole/quadrupole polarization + cross-multipole couplings."""

        __slots__ = [
            "c_sh3",
            "u_at4",
            "alpha_dp",
            "alpha_qp",
            "lam_md",
            "lam_dq",
        ]

        def __init__(
            self,
            c_sh3: Tensor,
            u_at4: Tensor,
            alpha_dp: Tensor,
            alpha_qp: Tensor,
            lam_md: Tensor,
            lam_dq: Tensor,
        ) -> None:
            super().__init__(device=c_sh3.device, dtype=c_sh3.dtype)
            self.c_sh3 = c_sh3
            self.u_at4 = u_at4
            self.alpha_dp = alpha_dp
            self.alpha_qp = alpha_qp
            self.lam_md = lam_md
            self.lam_dq = lam_dq

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
        def get_monopole_shell_potential(
            self,
            cache: InteractionCache,
            qsh: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            return self.c_sh3 * qsh.square()

        @override
        def get_monopole_shell_energy(
            self,
            cache: InteractionCache,
            qsh: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            return (self.c_sh3 / 3.0) * qsh.pow(3)

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
            return self.u_at4 * qat.pow(3) + self.lam_md * qdp[..., 2]

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
            return (
                0.25 * self.u_at4 * qat.pow(4) + self.lam_md * qat * qdp[..., 2]
            )

        @override
        def get_dipole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = cache
            assert qdp is not None and qqp is not None
            denom = torch.sqrt(1.0 + qdp.square().sum(dim=-1, keepdim=True))
            pot = self.alpha_dp * qdp / denom
            z_zeros = torch.zeros_like(qat)
            cross_md = torch.stack(
                [z_zeros, z_zeros, self.lam_md * qat], dim=-1
            )
            cross_dq = torch.stack(
                [self.lam_dq * qqp[..., 0], z_zeros, z_zeros], dim=-1
            )
            return pot + cross_md + cross_dq

        @override
        def get_dipole_atom_energy(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qat, qqp)
            assert qdp is not None
            return self.alpha_dp * (
                torch.sqrt(1.0 + qdp.square().sum(dim=-1)) - 1.0
            )

        @override
        def get_quadrupole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qat)
            assert qdp is not None and qqp is not None
            q_scale = qqp.new_tensor([1.0, 2.0, 1.0, 2.0, 2.0, 1.0])
            q_norm2 = (qqp.square() * q_scale).sum(dim=-1, keepdim=True)
            pot = 2.0 * self.alpha_qp * q_norm2 * qqp * q_scale
            zeros = torch.zeros_like(qqp[..., 0])
            cross = torch.stack(
                [self.lam_dq * qdp[..., 0], zeros, zeros, zeros, zeros, zeros],
                dim=-1,
            )
            return pot + cross

        @override
        def get_quadrupole_atom_energy(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qat)
            assert qdp is not None and qqp is not None
            q_scale = qqp.new_tensor([1.0, 2.0, 1.0, 2.0, 2.0, 1.0])
            q_norm2 = (qqp.square() * q_scale).sum(dim=-1)
            return (
                0.5 * self.alpha_qp * q_norm2.square()
                + self.lam_dq * qdp[..., 0] * qqp[..., 0]
            )

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    pos_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]], **DD64
    )
    pos_b = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.88], [0.0, 1.65, -0.45]], **DD64
    )
    par = ParamModule(GFN1_XTB, **DD64)

    # 1. Exact variational check: analytical potential == d(E_int)/d(charges)
    base_inter = NonlinearMultipolarInteraction(
        torch.tensor(0.02, **DD64),
        torch.tensor(0.03, **DD64),
        torch.tensor(0.04, **DD64),
        torch.tensor(0.015, **DD64),
        torch.tensor(0.025, **DD64),
        torch.tensor(0.01, **DD64),
    )
    sys_var = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        interaction=(base_inter,),
    ).setup(numbers)
    ihelp = sys_var.ihelp
    norb = int(ihelp.orbitals_to_shell.shape[0])
    nat = int(numbers.shape[0])
    q_mono = torch.linspace(-0.18, 0.22, norb, **DD64).requires_grad_(True)
    q_dip = (
        torch.linspace(-0.12, 0.15, nat * 3, **DD64)
        .reshape(nat, 3)
        .requires_grad_(True)
    )
    q_quad = (
        torch.linspace(-0.08, 0.09, nat * 6, **DD64)
        .reshape(nat, 6)
        .requires_grad_(True)
    )
    idata = InteractionListCache()
    idata[base_inter.label] = base_inter.get_cache(
        numbers=numbers, positions=pos_a, ihelp=ihelp
    )
    ilist = InteractionList(base_inter, **DD64)
    scf_iset = prepare_scf_interactions(ilist, idata, numbers)
    charges = SCFCharges(mono=q_mono, dipole=q_dip, quad=q_quad)
    pot = potential_from_charges(charges, scf_iset, ihelp)
    e_int = interaction_energy_from_charges(charges, scf_iset, ihelp).sum()
    d_mono, d_dip, d_quad = torch.autograd.grad(e_int, (q_mono, q_dip, q_quad))
    assert (
        pot.mono is not None and pot.dipole is not None and pot.quad is not None
    )
    torch.testing.assert_close(pot.mono, d_mono, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(pot.dipole, d_dip, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(pot.quad, d_quad, rtol=1e-12, atol=1e-12)

    # 2. Unrolled vs implicit 1st and 2nd derivatives + single-system vs batched implicit
    def run_scf_mode(
        diff_mode: str, *, batched: bool
    ) -> tuple[Tensor, tuple[Tensor, ...], Tensor]:
        p_in = (
            torch.stack([pos_a, pos_b], dim=0).requires_grad_(True)
            if batched
            else pos_a.clone().requires_grad_(True)
        )
        c_sh3 = torch.tensor(0.02, **DD64, requires_grad=True)
        u_at4 = torch.tensor(0.03, **DD64, requires_grad=True)
        alpha_dp = torch.tensor(0.04, **DD64, requires_grad=True)
        alpha_qp = torch.tensor(0.015, **DD64, requires_grad=True)
        lam_md = torch.tensor(0.025, **DD64, requires_grad=True)
        lam_dq = torch.tensor(0.01, **DD64, requires_grad=True)
        inter = NonlinearMultipolarInteraction(
            c_sh3, u_at4, alpha_dp, alpha_qp, lam_md, lam_dq
        )
        sys_nl = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=diff_mode,
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=80,
            ),
            interaction=(inter,),
        ).setup(numbers)
        if batched and diff_mode == "implicit":
            prep = prepare_converged_root_batch(sys_nl, p_in)
            total_e = evaluate_converged_root_batch(prep).energy.sum()
        elif batched:
            total_e = (
                sys_nl.singlepoint(p_in[0]).energy.sum()
                + sys_nl.singlepoint(p_in[1]).energy.sum()
            )
        else:
            total_e = sys_nl.singlepoint(p_in).energy.sum()
        first_grads = torch.autograd.grad(
            total_e,
            (p_in, c_sh3, u_at4, alpha_dp, alpha_qp, lam_md, lam_dq),
            create_graph=True,
        )
        (d2_lam_md,) = torch.autograd.grad(first_grads[5], lam_md)
        return (
            total_e.detach(),
            tuple(g.detach() for g in first_grads),
            d2_lam_md.detach(),
        )

    e_u1, g_u1, d2_u1 = run_scf_mode("unrolled", batched=False)
    e_i1, g_i1, d2_i1 = run_scf_mode("implicit", batched=False)
    torch.testing.assert_close(e_i1, e_u1, rtol=1e-9, atol=1e-10)
    for gi, gu in zip(g_i1, g_u1):
        torch.testing.assert_close(gi, gu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(d2_i1, d2_u1, rtol=1e-4, atol=1e-5)

    # Central finite-difference check on lam_md in float64
    eps = 1.0e-6
    fd_energies: list[Tensor] = []
    for delta in (-eps, eps):
        inter_fd = NonlinearMultipolarInteraction(
            torch.tensor(0.02, **DD64),
            torch.tensor(0.03, **DD64),
            torch.tensor(0.04, **DD64),
            torch.tensor(0.015, **DD64),
            torch.tensor(0.025 + delta, **DD64),
            torch.tensor(0.01, **DD64),
        )
        sys_fd = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=80,
            ),
            interaction=(inter_fd,),
        ).setup(numbers)
        fd_energies.append(sys_fd.singlepoint(pos_a).energy.sum())
    fd_grad_lam_md = (fd_energies[1] - fd_energies[0]) / (2.0 * eps)
    torch.testing.assert_close(g_i1[5], fd_grad_lam_md, rtol=1e-5, atol=1e-7)

    # Batched implicit vs 2-lane unrolled reference
    e_ub, g_ub, d2_ub = run_scf_mode("unrolled", batched=True)
    e_ib, g_ib, d2_ib = run_scf_mode("implicit", batched=True)
    torch.testing.assert_close(e_ib, e_ub, rtol=1e-9, atol=1e-10)
    for gib, gub in zip(g_ib, g_ub):
        torch.testing.assert_close(gib, gub, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(d2_ib, d2_ub, rtol=1e-4, atol=1e-5)


def test_cyclic_component_and_cache_graphs_rejected_while_acyclic_dags_work() -> (
    None
):
    """Cyclic graphs raise ValueError across conversion and setup while acyclic DAG aliases work."""
    numbers, pos = _lih_inputs()
    par = ParamModule(GFN1_XTB, **DD64)
    cfg_noscf = Config.create(int_driver="pytorch", exclude="scf")
    cfg_scf = Config.create(int_driver="pytorch")

    class PayloadClassical(Classical):
        def __init__(self, payload: Any, param: Tensor) -> None:
            super().__init__(device=param.device, dtype=param.dtype)
            self.payload = payload
            self.param = param

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Any:
            _ = (numbers, ihelp)
            return self.payload

        @override
        def get_energy(self, positions: Tensor, cache: Any, **_: Any) -> Tensor:
            _ = cache
            return 0.5 * self.param * positions.square().sum(dim=-1)

    class PayloadInteraction(Interaction):
        def __init__(self, payload: Any, param: Tensor) -> None:
            super().__init__(device=param.device, dtype=param.dtype)
            self.payload = payload
            self.param = param

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> Any:
            _ = (numbers, positions, ihelp)
            return self.payload

        @override
        def get_monopole_atom_potential(
            self,
            cache: Any,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            return 2.0 * self.param * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: Any,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (cache, qdp, qqp)
            return self.param * qat.square()

    p0 = torch.tensor(0.05, **DD64)

    # 1. Self-referential list
    self_list: list[Any] = [p0]
    self_list.append(self_list)

    # 2. Self-referential dict
    self_dict: dict[str, Any] = {"p": p0}
    self_dict["self"] = self_dict

    # 3. Mutually referencing containers
    mut_a: list[Any] = [p0]
    mut_b: dict[str, Any] = {"a": mut_a}
    mut_a.append(mut_b)

    for cyclic_payload in (self_list, self_dict, mut_a):
        cls_cyc = PayloadClassical(cyclic_payload, p0)
        int_cyc = PayloadInteraction(cyclic_payload, p0)
        for comp in (cls_cyc, int_cyc):
            with pytest.raises(ValueError, match="Circular reference"):
                comp.type(torch.float32)
            with pytest.raises(ValueError, match="Circular reference"):
                comp.type(torch.float64)
            with pytest.raises(ValueError, match="Circular reference"):
                comp.to(device=torch.device("meta"))
        with pytest.raises(ValueError, match="Circular reference"):
            Model(par=par, config=cfg_noscf, classical=(cls_cyc,)).setup(
                numbers
            )
        with pytest.raises(ValueError, match="Circular reference"):
            Model(par=par, config=cfg_scf, interaction=(int_cyc,)).setup(
                numbers
            )

    # 4. Component referencing itself (Classical and Interaction), plus post-setup cycle injection
    cls_self = PayloadClassical(p0, p0)
    cls_self.self_ref = cls_self
    int_self = PayloadInteraction(p0, p0)
    int_self.self_ref = int_self
    for comp in (cls_self, int_self):
        with pytest.raises(ValueError, match="Circular reference"):
            comp.type(torch.float32)
    with pytest.raises(ValueError, match="Circular reference"):
        Model(par=par, config=cfg_noscf, classical=(cls_self,)).setup(numbers)
    with pytest.raises(ValueError, match="Circular reference"):
        Model(par=par, config=cfg_scf, interaction=(int_self,)).setup(numbers)

    valid_cls = PayloadClassical([p0], p0)
    sys_post = Model(par=par, config=cfg_noscf, classical=(valid_cls,)).setup(
        numbers
    )
    valid_cls.payload.append(valid_cls.payload)
    with pytest.raises(ValueError, match="Circular reference"):
        sys_post.singlepoint(pos)

    # 5a. Mutual component <-> cache cycle is rejected
    class MutualCycleCache(ClassicalCache):
        __slots__ = ["owner", "scale"]

        def __init__(self, owner: Any, scale: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.owner = owner
            self.scale = scale

    class MutualCycleClassical(Classical):
        def __init__(self, scale: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.scale = scale
            self.cached_ref: Any = None

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> MutualCycleCache:
            _ = (numbers, ihelp)
            c = MutualCycleCache(self, self.scale)
            self.cached_ref = c
            return c

        @override
        def get_energy(
            self, positions: Tensor, cache: MutualCycleCache, **_: Any
        ) -> Tensor:
            return 0.5 * cache.scale * positions.square().sum(dim=-1)

    with pytest.raises(ValueError, match="Circular reference"):
        Model(
            par=par,
            config=cfg_noscf,
            classical=(MutualCycleClassical(p0),),
        ).setup(numbers)

    # 5b. One-way acyclic cache -> component DAG reference is accepted and preserves view identity
    class DagOwnerCache(ClassicalCache):
        __slots__ = ["owner", "scale"]

        def __init__(self, owner: Any, scale: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.owner = owner
            self.scale = scale

    class DagOwnerClassical(Classical):
        __slots__ = ["scale"]

        def __init__(self, scale: Tensor) -> None:
            super().__init__(device=scale.device, dtype=scale.dtype)
            self.scale = scale

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> DagOwnerCache:
            _ = (numbers, ihelp)
            return DagOwnerCache(self, self.scale)

        @override
        def get_energy(
            self, positions: Tensor, cache: DagOwnerCache, **_: Any
        ) -> Tensor:
            return 0.5 * cache.owner.scale * positions.square().sum(dim=-1)

    dag_cls = DagOwnerClassical(torch.tensor(0.05, **DD64))
    sys_dag = Model(par=par, config=cfg_noscf, classical=(dag_cls,)).setup(
        numbers
    )
    assert sys_dag.exact_core is not None
    core_dag32 = sys_dag.exact_core.to(dtype=torch.float32)
    dag_term = core_dag32.classical_terms[-1].setup
    assert isinstance(dag_term, CustomClassicalCoreSetup)
    comp_v, cache_v = dag_term.reconstruct()
    assert cache_v.owner is comp_v
    assert cache_v.owner.scale.dtype == torch.float32
    assert dag_cls.scale.dtype == torch.float64

    # 6 & 7. Acyclic shared tensor AND shared nested container aliases across conversion and func_grad
    class DagSharedInteraction(Interaction):
        def __init__(self, shared_box: list[Tensor], shared_t: Tensor) -> None:
            super().__init__(device=shared_t.device, dtype=shared_t.dtype)
            self.left_box = shared_box
            self.right_map = {"box": shared_box, "leaf": shared_t}

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> dict[str, Any]:
            _ = (numbers, positions, ihelp)
            return {"box": self.left_box, "map": self.right_map}

        @override
        def get_monopole_atom_potential(
            self,
            cache: dict[str, Any],
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (qdp, qqp)
            eff = (
                cache["box"][0] + cache["map"]["box"][0] + cache["map"]["leaf"]
            )
            return 2.0 * eff * qat

        @override
        def get_monopole_atom_energy(
            self,
            cache: dict[str, Any],
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
            **_: Any,
        ) -> Tensor:
            _ = (qdp, qqp)
            eff = (
                cache["box"][0] + cache["map"]["box"][0] + cache["map"]["leaf"]
            )
            return eff * qat.square()

    shared_leaf = torch.tensor(0.03, **DD64, requires_grad=True)
    shared_box = [shared_leaf]
    dag_int = DagSharedInteraction(shared_box, shared_leaf)
    dag_int32 = dag_int.type(torch.float32)
    assert dag_int32.left_box is dag_int32.right_map["box"]
    assert dag_int32.left_box[0] is dag_int32.right_map["leaf"]
    assert dag_int32.left_box[0].dtype == torch.float32

    def eval_dag_energy(theta: Tensor, diff_mode: str) -> Tensor:
        """Total SCF energy where one leaf appears 3 times via shared tensor and container DAG."""
        box = [theta]
        sys_theta = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=diff_mode,
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=80,
            ),
            interaction=(DagSharedInteraction(box, theta),),
        ).setup(numbers)
        return sys_theta.singlepoint(pos).energy.sum()

    def eval_dag_unrolled(theta: Tensor) -> Tensor:
        """Unrolled energy wrapper for torch.func.grad."""
        return eval_dag_energy(theta, "unrolled")

    def eval_dag_implicit(theta: Tensor) -> Tensor:
        """Implicit energy wrapper for torch.func.grad."""
        return eval_dag_energy(theta, "implicit")

    theta0 = torch.tensor(0.03, **DD64)
    g_unrolled = func_grad(eval_dag_unrolled)(theta0)
    g_implicit = func_grad(eval_dag_implicit)(theta0)
    torch.testing.assert_close(g_implicit, g_unrolled, rtol=1e-6, atol=1e-7)


def test_opaque_external_object_tensor_rejected_and_nontensor_consistent() -> (
    None
):
    """Opaque objects holding Tensors raise TypeError while non-tensor opaque objects stay consistent."""
    from functools import partial

    from dxtb import GFN2_XTB
    from dxtb._src.components.interactions.dispersion.d4sc import (
        DispersionD4SC,
        new_d4sc,
    )

    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.25]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)

    class ExternalState:
        def __init__(self, parameter: Any) -> None:
            self.parameter = parameter

    class ExtClassical(Classical):
        def __init__(self, state: Any, weight: Tensor) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.state = state
            self.weight = weight

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return self.weight * numbers.to(dtype=self.weight.dtype)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return (
                0.5
                * self.state.parameter
                * cache
                * positions.square().sum(dim=-1)
            )

    class ExtInteraction(Interaction):
        def __init__(self, state: Any, weight: Tensor) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.state = state
            self.weight = weight

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
            return 2.0 * self.state.parameter * self.weight * qat

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
            return self.state.parameter * self.weight * qat.square()

    # 1. Opaque object holding a Tensor (or partial holding a Tensor) is rejected across all entry points
    t_hidden = torch.tensor(2.5, **DD64, requires_grad=True)
    w0 = torch.tensor(0.04, **DD64, requires_grad=True)
    for bad_holder in (
        ExternalState(t_hidden),
        partial(int, t_hidden),
    ):
        bad_cls = ExtClassical(bad_holder, w0)
        bad_int = ExtInteraction(bad_holder, w0)
        for bad_comp in (bad_cls, bad_int):
            with pytest.raises(TypeError, match="Opaque object"):
                bad_comp.type(torch.float32)
            with pytest.raises(TypeError, match="Opaque object"):
                bad_comp.type(torch.float64)
            with pytest.raises(TypeError, match="Opaque object"):
                bad_comp.to(device=torch.device("meta"))
        for bm, nums in (
            (0, numbers),
            (1, torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)),
            (2, torch.stack([numbers, numbers], dim=0)),
        ):
            with pytest.raises(TypeError, match="Opaque object"):
                Model(
                    par=par,
                    config=Config.create(int_driver="pytorch"),
                    classical=(bad_cls,),
                ).setup(nums, batch_mode=bm)
            with pytest.raises(TypeError, match="Opaque object"):
                Model(
                    par=par,
                    config=Config.create(int_driver="pytorch"),
                    interaction=(bad_int,),
                ).setup(nums, batch_mode=bm)

    # Post-setup injection of an opaque Tensor holder onto an Interaction fails before SCF
    post_int = ExtInteraction(ExternalState(2.5), w0)
    sys_post_int = Model(
        par=par,
        config=Config.create(int_driver="pytorch", differentiation="implicit"),
        interaction=(post_int,),
    ).setup(numbers)
    post_int.state = ExternalState(t_hidden)
    with pytest.raises(TypeError, match="Opaque object"):
        sys_post_int.singlepoint(pos_a)

    # 2. Opaque object holding ONLY non-tensor state works across conversion, SCF, func_grad, and batches
    ok_state = ExternalState(2.5)
    ok_cls = ExtClassical(ok_state, torch.tensor(0.02, **DD64))
    ok_int = ExtInteraction(ok_state, torch.tensor(0.04, **DD64))
    calc64 = Calculator(
        numbers,
        GFN1_XTB,
        classical=(ok_cls,),
        interaction=(ok_int,),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    e64 = calc64.energy(pos_a)
    calc32 = calc64.type(torch.float32)
    e32 = calc32.energy(pos_a.to(dtype=torch.float32))
    assert e32.dtype == torch.float32
    assert calc32.interactions.components[-1].state is ok_state
    torch.testing.assert_close(
        e32.to(dtype=torch.float64), e64, rtol=1e-4, atol=1e-4
    )

    def eval_nontensor_opaque(weight: Tensor, diff_mode: str) -> Tensor:
        """Total SCF energy with non-tensor ExternalState and differentiable Interaction weight."""
        sys_m = Model(
            par=par,
            config=Config.create(
                int_driver="pytorch",
                differentiation=diff_mode,
                x_atol=1.0e-11,
                x_atol_max=1.0e-11,
                maxiter=80,
            ),
            classical=(ExtClassical(ok_state, torch.tensor(0.02, **DD64)),),
            interaction=(ExtInteraction(ok_state, weight),),
        ).setup(numbers)
        assert not can_vmap_system(sys_m)
        return sys_m.singlepoint(pos_a).energy.sum()

    def eval_opaque_unrolled(weight: Tensor) -> Tensor:
        """Unrolled wrapper for torch.func.grad."""
        return eval_nontensor_opaque(weight, "unrolled")

    def eval_opaque_implicit(weight: Tensor) -> Tensor:
        """Implicit wrapper for torch.func.grad."""
        return eval_nontensor_opaque(weight, "implicit")

    w_test = torch.tensor(0.04, **DD64, requires_grad=True)
    e_u = eval_nontensor_opaque(w_test, "unrolled")
    (g_auto_u,) = torch.autograd.grad(e_u, (w_test,))
    g_func_u = func_grad(eval_opaque_unrolled)(w_test.detach())
    g_func_i = func_grad(eval_opaque_implicit)(w_test.detach())
    torch.testing.assert_close(g_auto_u, g_func_u, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(g_func_i, g_func_u, rtol=1e-6, atol=1e-7)

    # Independent batch evaluation via evaluate_packed_systems and Calculator (batch_mode=1 and 2)
    sys_ind = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        classical=(ok_cls,),
        interaction=(ok_int,),
    ).setup(numbers)
    packed_res = evaluate_packed_systems(
        (sys_ind, sys_ind), torch.stack([pos_a, pos_b], dim=0)
    )
    torch.testing.assert_close(
        packed_res.energy.sum(-1),
        torch.stack([e64, sys_ind.singlepoint(pos_b).energy.sum(-1)]),
    )
    calc_bm2 = Calculator(
        torch.stack([numbers, numbers], dim=0),
        GFN1_XTB,
        classical=(ok_cls,),
        interaction=(ok_int,),
        opts={"verbosity": 0, "int_driver": "pytorch", "batch_mode": 2},
        **DD64,
    )
    torch.testing.assert_close(
        calc_bm2.energy(torch.stack([pos_a, pos_b], dim=0)),
        packed_res.energy.sum(-1),
    )

    # 3. Custom DispersionD4SC subclass preserves TensorLike (D4Model/Cutoff) + RationalDamping
    class CustomD4SC(DispersionD4SC):
        pass

    nums_h2o = torch.tensor([8, 1, 1], device=DEVICE)
    pos_h2o = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]], **DD64
    )
    par2 = ParamModule(GFN2_XTB, **DD64)
    base_d4sc = new_d4sc(nums_h2o, par2, **DD64)
    assert base_d4sc is not None
    sub_d4sc = CustomD4SC(
        base_d4sc.param,
        base_d4sc.model,
        base_d4sc.rcov,
        base_d4sc.r4r2,
        base_d4sc.cutoff,
        base_d4sc.counting_function,
        base_d4sc.damping_function,
        **DD64,
    )
    sub_d4sc.label = "CustomD4SC"
    sub_d4sc_32 = sub_d4sc.type(torch.float32)
    assert sub_d4sc_32.model.wf.dtype == torch.float32
    assert sub_d4sc_32.model.numbers.dtype == nums_h2o.dtype

    wf_grads: list[Tensor] = []
    for diff_mode in ("unrolled", "implicit"):
        wf_p = torch.tensor(6.0, **DD64, requires_grad=True)
        model_p = type(base_d4sc.model)(nums_h2o, wf=wf_p, **DD64)
        sub_p = CustomD4SC(
            base_d4sc.param,
            model_p,
            base_d4sc.rcov,
            base_d4sc.r4r2,
            base_d4sc.cutoff,
            base_d4sc.counting_function,
            base_d4sc.damping_function,
            **DD64,
        )
        sub_p.label = "CustomD4SC"
        sys_d4sc = Model(
            par=par2,
            config=Config.create(
                int_driver="pytorch",
                exclude="d4sc",
                differentiation=diff_mode,
                x_atol=1.0e-10,
                x_atol_max=1.0e-10,
                maxiter=80,
            ),
            interaction=(sub_p,),
        ).setup(nums_h2o)
        loss_d4sc = sys_d4sc.singlepoint(pos_h2o).energy.sum()
        wf_grads.append(torch.autograd.grad(loss_d4sc, wf_p)[0])
    torch.testing.assert_close(wf_grads[1], wf_grads[0], rtol=1e-6, atol=1e-8)


def test_mutation_and_reconstruction_boundaries_preserve_unrelated_builtins() -> (
    None
):
    """Component and converted-cache mutations invalidate only the affected setup without mutating caller state."""
    numbers = torch.tensor([8, 1, 1, 17], device=DEVICE)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [0.0, 1.7, -0.4],
            [2.5, 0.0, 0.0],
        ],
        **DD64,
    )
    par = ParamModule(GFN1_XTB, **DD64)

    class SharedAliasCache(ClassicalCache):
        __slots__ = ["shared", "buf"]

        def __init__(self, shared: Tensor, buf: Tensor) -> None:
            super().__init__(device=shared.device, dtype=shared.dtype)
            self.shared = shared
            self.buf = buf
            self.note = "clean"

    class TrackedClassical(Classical):
        def __init__(self, label: str, shared: Tensor, factor: float) -> None:
            super().__init__(device=shared.device, dtype=shared.dtype)
            self.label = label
            self.left = shared
            self.right = shared
            self.factor = factor
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> SharedAliasCache:
            _ = ihelp
            self.cache_calls += 1
            buf = torch.full(
                numbers.shape, self.factor, device=self.device, dtype=self.dtype
            )
            return SharedAliasCache(self.left, buf)

        @override
        def get_energy(
            self, positions: Tensor, cache: SharedAliasCache, **_: Any
        ) -> Tensor:
            return (
                0.25
                * (self.left + self.right + cache.shared)
                * cache.buf
                * positions.square().sum(dim=-1)
            )

    p_pure = torch.tensor(0.02, **DD64, requires_grad=True)
    p_mut = torch.tensor(0.03, **DD64, requires_grad=True)
    pure_comp = TrackedClassical("PureCustom", p_pure, 1.0)
    mut_comp = TrackedClassical("MutCustom", p_mut, 1.5)

    sys64 = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        classical=(pure_comp, mut_comp),
    ).setup(numbers)

    orig_rep = sys64.repulsion_setup
    orig_hal = sys64.halogen_setup
    orig_d3 = sys64.d3_setups
    orig_es2 = sys64.es2_setup
    orig_es3 = sys64.es3_setup
    assert orig_rep is not None and orig_hal is not None
    assert orig_es2 is not None and orig_es3 is not None
    assert (pure_comp.cache_calls, mut_comp.cache_calls) == (1, 1)

    # 1. Repeated evaluations of pure custom components reuse captured setups
    r1 = sys64.singlepoint(pos)
    r2 = sys64.singlepoint(pos)
    assert (pure_comp.cache_calls, mut_comp.cache_calls) == (1, 1)
    torch.testing.assert_close(r1.energy, r2.energy)

    # 2. Reconstruction preserves tensor aliases and never mutates caller-owned components
    assert sys64.exact_core is not None
    core32 = sys64.exact_core.to(dtype=torch.float32)
    pure_term32 = core32.classical_terms[-2].setup
    assert isinstance(pure_term32, CustomClassicalCoreSetup)
    comp_v32, cache_v32 = pure_term32.reconstruct()
    assert comp_v32 is not pure_comp
    assert comp_v32.left is comp_v32.right
    assert comp_v32.left is cache_v32.shared
    assert comp_v32.left.dtype == torch.float32
    assert pure_comp.dtype == torch.float64
    assert pure_comp.left is p_pure

    # 3. Mutating mut_comp invalidates only mut_comp; pure_comp and all built-in setups are reused
    mut_comp.factor = 2.75
    assert sys64._active_exact_core() is None
    r3 = sys64.singlepoint(pos)
    assert pure_comp.cache_calls == 1
    assert mut_comp.cache_calls == 2
    assert sys64.repulsion_setup is orig_rep
    assert sys64.halogen_setup is orig_hal
    assert sys64.d3_setups == orig_d3
    assert sys64.es2_setup is orig_es2
    assert sys64.es3_setup is orig_es3

    ref_sys = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        classical=(
            TrackedClassical("PureCustom", p_pure.detach(), 1.0),
            TrackedClassical("MutCustom", p_mut.detach(), 2.75),
        ),
    ).setup(numbers)
    torch.testing.assert_close(r3.energy, ref_sys.singlepoint(pos).energy)

    # 4. Converted Calculator cache shares identity with CustomClassicalCoreSetup and invalidates on cache mutation
    calc_c = TrackedClassical("CalcCustom", torch.tensor(0.04, **DD64), 1.0)
    calc32 = Calculator(
        numbers,
        GFN1_XTB,
        classical=(calc_c,),
        opts={"verbosity": 0, "int_driver": "pytorch", "exclude": ["scf"]},
        **DD64,
    ).type(torch.float32)
    eval32 = calc32._prepare_eval_system()
    assert eval32._active_exact_core() is not None
    pub_cache32 = eval32.classical_cache[calc_c.label]
    assert eval32.exact_core is not None
    core_cache32 = eval32.exact_core.classical_terms[-1].setup.template.cache  # type: ignore[union-attr]
    assert pub_cache32 is core_cache32
    pub_cache32.note = "dirty"
    assert eval32._active_exact_core() is None
    conv_comp32 = eval32.classicals.components[-1]
    assert isinstance(conv_comp32, TrackedClassical)
    before_calls = conv_comp32.cache_calls
    eval32.singlepoint(pos.to(dtype=torch.float32))
    assert conv_comp32.cache_calls == before_calls + 1

    # 5. In-place mutations of non-tensor set/deque attributes invalidate snapshots,
    #    and Node keep_dtype=True floating fields stay float64 across core conversion.
    from collections import deque
    from types import MappingProxyType

    class KeepDtypeNodeClassical(Classical):
        __slots__ = ["knode"]

        def __init__(self, knode: _KeepDtypeFloatNode) -> None:
            super().__init__(
                device=knode.primary.device, dtype=knode.primary.dtype
            )
            self.knode = knode

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> _KeepDtypeFloatNode:
            _ = (numbers, ihelp)
            return self.knode

        @override
        def get_energy(
            self, positions: Tensor, cache: _KeepDtypeFloatNode, **_: Any
        ) -> Tensor:
            return (
                0.01
                * (cache.primary + cache.ref_fp64.to(cache.primary.dtype))
                * positions.square().sum(dim=-1)
            )

    class SetDequeClassical(Classical):
        def __init__(self, weight: Tensor) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.weight = weight
            self.active_z: set[int] = {1, 8}
            self.weights_dq: deque[float] = deque([1.0])
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.cache_calls += 1
            factor = sum(self.weights_dq) * len(self.active_z)
            return factor * self.weight * numbers.to(dtype=self.weight.dtype)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return 0.01 * cache * positions.square().sum(dim=-1)

    knode64 = _KeepDtypeFloatNode(
        primary=torch.tensor(0.2, **DD64),
        ref_fp64=torch.tensor(0.3, **DD64),
    )
    sys_kn = Model(
        par=par,
        config=Config.create(int_driver="pytorch", exclude="scf"),
        classical=(KeepDtypeNodeClassical(knode64),),
    ).setup(numbers)
    assert sys_kn.exact_core is not None
    kn_core32 = sys_kn.exact_core.to(dtype=torch.float32)
    kn_term32 = kn_core32.classical_terms[-1].setup
    assert isinstance(kn_term32, CustomClassicalCoreSetup)
    _, kn_cache32 = kn_term32.reconstruct()
    assert kn_cache32.primary.dtype == torch.float32
    assert kn_cache32.ref_fp64.dtype == torch.float64

    sd_comp = SetDequeClassical(torch.tensor(0.2, **DD64))
    sys_sd = Model(
        par=par,
        config=Config.create(int_driver="pytorch", exclude="scf"),
        classical=(sd_comp,),
    ).setup(numbers)
    sys_sd.singlepoint(pos)
    assert sd_comp.cache_calls == 1
    sd_comp.active_z.add(17)
    sys_sd.singlepoint(pos)
    assert sd_comp.cache_calls == 2
    sd_comp.weights_dq.append(0.5)
    sys_sd.singlepoint(pos)
    assert sd_comp.cache_calls == 3

    # 6. Shared-DAG opaque/frozen tensor holders, stdlib C containers, Node context(),
    #    and custom .to/.type overrides agree across all conversion and setup entry points.
    class _ExtHolder:
        def __init__(self, inner: Any) -> None:
            self.inner = inner

    class _CustomTypeOnlyCache(ClassicalCache):
        __slots__ = ["val", "type_calls"]

        def __init__(self, val: Tensor, type_calls: int = 0) -> None:
            super().__init__(device=val.device, dtype=val.dtype)
            self.val = val
            self.type_calls = type_calls

        @override
        def type(self, dtype: torch.dtype) -> _CustomTypeOnlyCache:
            return _CustomTypeOnlyCache(
                self.val.to(dtype=dtype), self.type_calls + 1
            )

    class _CustomToTypeClassical(Classical):
        def __init__(self, payload: Any, p: Tensor) -> None:
            super().__init__(device=p.device, dtype=p.dtype)
            self.box = [p]
            self.payload = payload
            self.p = p

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Any:
            _ = (numbers, ihelp)
            return self.payload

        @override
        def get_energy(self, positions: Tensor, cache: Any, **_: Any) -> Tensor:
            _ = cache
            return 0.5 * self.p * positions.square().sum(dim=-1)

        @override
        def to(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> _CustomToTypeClassical:
            return super().to(device=device, dtype=dtype)

        @override
        def type(self, dtype: torch.dtype) -> _CustomToTypeClassical:
            return super().type(dtype)

    class _CustomToOnlyCache(ClassicalCache):
        __slots__ = ["val", "to_calls", "last_device"]

        def __init__(
            self,
            val: Tensor,
            to_calls: int = 0,
            last_device: torch.device | None = None,
        ) -> None:
            super().__init__(device=val.device, dtype=val.dtype)
            self.val = val
            self.to_calls = to_calls
            self.last_device = last_device

        @override
        def to(
            self,
            device: torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> _CustomToOnlyCache:
            return _CustomToOnlyCache(
                self.val.to(device=device, dtype=dtype),
                self.to_calls + 1,
                device,
            )

    t_leaf = torch.tensor(0.05, **DD64)
    shared_list = [t_leaf]

    # Nested TensorLike overriding .type() (without .to()) has .type() invoked during parent .type()
    type_only_owner = _CustomToTypeClassical(
        _CustomTypeOnlyCache(t_leaf), t_leaf
    )
    conv_owner32 = type_only_owner.type(torch.float32)
    assert conv_owner32.payload.type_calls == 1
    assert conv_owner32.payload.val.dtype == torch.float32

    # Nested TensorLike overriding .to() (without .type()) has .to() invoked during parent .to(dtype=...)
    to_only_owner = _CustomToTypeClassical(_CustomToOnlyCache(t_leaf), t_leaf)
    conv_to_only32 = to_only_owner.to(dtype=torch.float32)
    assert conv_to_only32.payload.to_calls == 1
    assert conv_to_only32.payload.last_device is None
    assert conv_to_only32.payload.val.dtype == torch.float32

    conv_to_dev32 = to_only_owner.to(device=DEVICE, dtype=torch.float32)
    assert conv_to_dev32.payload.to_calls == 1
    assert conv_to_dev32.payload.last_device == DEVICE
    assert conv_to_dev32.payload.val.dtype == torch.float32

    # Shared container visited under keep_dtype=True first still converts on sibling child field
    p_fp32 = torch.tensor(0.05, device=DEVICE, dtype=torch.float32)
    dual_shared_box = [torch.tensor(0.07, **DD64)]
    dual_owner = _CustomToTypeClassical(
        _DualBoxNode(kept_box=dual_shared_box, normal_box=dual_shared_box),
        p_fp32,
    )
    assert dual_owner.dtype == torch.float32
    dual_type32 = dual_owner.type(torch.float32)
    assert dual_type32.payload.kept_box[0].dtype == torch.float64
    assert dual_type32.payload.normal_box[0].dtype == torch.float32
    dual_to32 = dual_owner.to(dtype=torch.float32)
    assert dual_to32.payload.kept_box[0].dtype == torch.float64
    assert dual_to32.payload.normal_box[0].dtype == torch.float32

    # Shared-box opaque tensor holder + set/frozenset/deque/MappingProxyType + Node context opaque tensor
    comp_shared_dag = _CustomToTypeClassical(_ExtHolder(shared_list), t_leaf)
    comp_shared_dag.box = shared_list
    for bad_obj in (
        comp_shared_dag,
        _CustomToTypeClassical({t_leaf}, t_leaf),
        _CustomToTypeClassical(frozenset([t_leaf]), t_leaf),
        _CustomToTypeClassical(deque([t_leaf]), t_leaf),
        _CustomToTypeClassical(MappingProxyType({"t": t_leaf}), t_leaf),
        _CustomToTypeClassical(
            _ContextMetaNode(primary=t_leaf, meta=_ExtHolder(t_leaf)), t_leaf
        ),
    ):
        with pytest.raises(TypeError):
            bad_obj.type(torch.float64)
        with pytest.raises(TypeError):
            bad_obj.type(torch.float32)
        with pytest.raises(TypeError):
            Model(
                par=par,
                config=Config.create(int_driver="pytorch", exclude="scf"),
                classical=(bad_obj,),
            ).setup(numbers)

    # Mutual cycle across custom .to/.type subclasses, cyclic deque, cyclic MappingProxyType, and cyclic Node context
    peer_a = _CustomToTypeClassical(1, t_leaf)
    peer_b = _CustomToTypeClassical(2, t_leaf)
    peer_a.payload = peer_b
    peer_b.payload = peer_a

    cyc_dq: deque[Any] = deque()
    cyc_dq.append(cyc_dq)

    cyc_dict: dict[str, Any] = {}
    cyc_dict["self"] = cyc_dict
    cyc_mp = MappingProxyType(cyc_dict)

    cyc_ext = _ExtHolder(None)
    cyc_ext.inner = cyc_ext

    for cyc_comp in (
        peer_a,
        _CustomToTypeClassical(cyc_dq, t_leaf),
        _CustomToTypeClassical(cyc_mp, t_leaf),
        _CustomToTypeClassical(
            _ContextMetaNode(primary=t_leaf, meta=cyc_ext), t_leaf
        ),
    ):
        with pytest.raises(ValueError, match="Circular reference"):
            cyc_comp.type(torch.float64)
        with pytest.raises(ValueError, match="Circular reference"):
            cyc_comp.type(torch.float32)
        with pytest.raises(ValueError, match="Circular reference"):
            Model(
                par=par,
                config=Config.create(int_driver="pytorch", exclude="scf"),
                classical=(cyc_comp,),
            ).setup(numbers)


@dataclass(frozen=True)
class _FrozenTensorBox:
    """Hashable frozen dataclass carrying a Tensor leaf."""

    weight: Tensor


class _KeepPairNode(Node):
    """Diamond-testable Node with two keep_dtype=True children."""

    a: Any = child(keep_dtype=True)
    b: Any = child(keep_dtype=True)


class _TypeOnlyNode(Node):
    """Node overriding only ``type(dtype)`` without ``to`` or ``_convert_child``."""

    val: Tensor = child()
    calls: int = context(default=0)

    @override
    def type(self, dtype: torch.dtype) -> _TypeOnlyNode:
        return _TypeOnlyNode(val=self.val.to(dtype=dtype), calls=self.calls + 1)


def test_h2_review_hardening_and_custom_scf_rebase_contracts() -> None:
    """Cover Node context tensors, diamond DAGs, dict-key mutations, module refs, and custom SCF."""
    import time
    import warnings

    from dxtb._src.components.interactions import InteractionList

    numbers = torch.tensor([3, 1], dtype=torch.int64, device=DEVICE)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)
    t_leaf = torch.tensor(0.05, **DD64)

    # 1. Node context() holding Tensor via frozen dataclass or ClassicalCache is rejected
    class _SimpleCache(ClassicalCache):
        __slots__ = ["w"]

        def __init__(self, w: Tensor) -> None:
            super().__init__(device=w.device, dtype=w.dtype)
            self.w = w

    class _PayloadClassical(Classical):
        def __init__(self, payload: Any) -> None:
            super().__init__(**DD64)
            self.payload = payload

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Any:
            _ = (numbers, ihelp)
            return self.payload

        @override
        def get_energy(self, positions: Tensor, cache: Any, **_: Any) -> Tensor:
            _ = cache
            return positions.sum(-1)

    for ctx_holder in (_FrozenTensorBox(t_leaf), _SimpleCache(t_leaf)):
        bad_ctx = _PayloadClassical(
            _ContextMetaNode(primary=t_leaf, meta=ctx_holder)
        )
        with pytest.raises(TypeError, match=r"context\(\)"):
            bad_ctx.to(device=torch.device("meta"))
        with pytest.raises(TypeError, match=r"context\(\)"):
            bad_ctx.type(torch.float32)
        with pytest.raises(TypeError, match=r"context\(\)"):
            Model(
                par=par,
                config=Config.create(int_driver="pytorch", exclude="scf"),
                classical=(bad_ctx,),
            ).setup(numbers)

    # 2. O(V + E) traversal, template hashing, and spec equality on deep diamond DAGs
    dag: Any = [torch.tensor(1.0, **DD64)]
    dag_other: Any = [torch.tensor(1.0, **DD64)]
    for _ in range(25):
        dag = [dag, dag]
        dag_other = [dag_other, dag_other]
    dag_comp = _PayloadClassical(dag)
    t0 = time.perf_counter()
    dag_f32 = dag_comp.type(torch.float32)
    assert (time.perf_counter() - t0) < 1.0
    assert dag_f32.payload[0] is dag_f32.payload[1]

    from tad_mctc.exceptions import DtypeError

    from dxtb._src.calculators.types.base import _convert_eval_system_dtype
    from dxtb._src.components.custom import (
        is_tensor_only_object,
        try_build_custom_classical_core_setup,
    )

    t1 = time.perf_counter()
    s_dag1 = try_build_custom_classical_core_setup(dag_comp, dag, DD64)
    s_dag2 = try_build_custom_classical_core_setup(
        _PayloadClassical(dag_other), dag_other, DD64
    )
    assert s_dag1 is not None and s_dag2 is not None
    assert hash(s_dag1.template) == hash(s_dag2.template)
    assert s_dag1.template == s_dag2.template
    assert (time.perf_counter() - t1) < 1.0

    keep_dag: Any = torch.tensor(1.0, **DD64)
    for _ in range(22):
        keep_dag = _KeepPairNode(a=keep_dag, b=keep_dag)

    class _KeepDiamondClassical(Classical):
        def __init__(self, payload: Any, trigger: Tensor) -> None:
            super().__init__(device=trigger.device, dtype=trigger.dtype)
            self.payload = payload
            self.trigger = trigger

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Any:
            _ = (numbers, ihelp)
            return self.payload

        @override
        def get_energy(self, positions: Tensor, cache: Any, **_: Any) -> Tensor:
            _ = cache
            return positions.sum(-1)

    t2 = time.perf_counter()
    kd_conv = _KeepDiamondClassical(keep_dag, torch.tensor(2.0, **DD64)).type(
        torch.float32
    )
    assert (time.perf_counter() - t2) < 1.0
    assert kd_conv.trigger.dtype == torch.float32
    assert kd_conv.payload is keep_dag

    # Node overriding only .type(dtype) is respected and excluded from tensor-only fast path
    tn = _TypeOnlyNode(val=torch.tensor(1.0, **DD64))
    assert not is_tensor_only_object(tn)
    tn_owner32 = _PayloadClassical(tn).type(torch.float32)
    assert tn_owner32.payload.calls == 1
    assert tn_owner32.payload.val.dtype == torch.float32

    # 3. In-place mutation of large Tensor dict keys invalidates captured setup
    class _DictKeyClassical(Classical):
        def __init__(self, kt: Tensor) -> None:
            super().__init__(**DD64)
            self.table = {kt: 2.0}
            self.calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.calls += 1
            k, v = next(iter(self.table.items()))
            return k[500] * v * numbers.to(k.dtype)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return 0.5 * cache * positions.square().sum(dim=-1)

    big_key = torch.zeros(1005, **DD64)
    dk_comp = _DictKeyClassical(big_key)
    sys_dk = Model(
        par=par,
        config=Config.create(int_driver="pytorch", exclude="all"),
        classical=(dk_comp,),
    ).setup(numbers)
    assert float(sys_dk.singlepoint(pos).energy.sum()) == 0.0
    assert dk_comp.calls == 1
    big_key[500] = 1.0
    assert sys_dk._active_exact_core() is None
    assert float(sys_dk.singlepoint(pos).energy.sum()) > 0.0
    assert dk_comp.calls == 2

    # 4. ModuleType reference on custom component is not crawled in batch_implicit
    class _ModuleClassical(Classical):
        def __init__(self, w: Tensor) -> None:
            super().__init__(**DD64)
            self.w = w
            self.mod = torch

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return self.w * numbers.to(self.w.dtype)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return 0.5 * cache * self.mod.sum(positions.square(), dim=-1)

    nums_w = torch.tensor([8, 1, 1], dtype=torch.int64, device=DEVICE)
    pos_w = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [0.0, 1.7, -0.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.85], [0.0, 1.75, -0.42]],
        ],
        **DD64,
    )
    sys_plain = Model(
        par=par,
        config=Config.create(int_driver="pytorch", differentiation="implicit"),
    ).setup(nums_w)
    sys_mod = Model(
        par=par,
        config=Config.create(int_driver="pytorch", differentiation="implicit"),
        classical=(_ModuleClassical(torch.tensor(0.01, **DD64)),),
    ).setup(nums_w)
    prep_plain = prepare_converged_root_batch(sys_plain, pos_w)
    with warnings.catch_warnings(record=True) as w_rec:
        warnings.simplefilter("always")
        prep_mod = prepare_converged_root_batch(sys_mod, pos_w)
    assert (
        len(prep_mod.system_tensor_refs)
        <= len(prep_plain.system_tensor_refs) + 4
    )
    assert not any("reduce_op" in str(w.message) for w in w_rec)

    # 5. Custom Interaction with keep_dtype=True + int64/bool leaves & duplicate-label / ihelp guards
    class _KeptSCFInteraction(Interaction):
        def __init__(self, kn: _KeepDtypeFloatNode) -> None:
            super().__init__(device=DEVICE, dtype=kn.primary.dtype)
            self.kn = kn
            self.mask = torch.tensor([True, True], device=DEVICE)
            self.idx = torch.tensor([0, 1], dtype=torch.int64, device=DEVICE)

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> tuple[_KeepDtypeFloatNode, Tensor, Tensor]:
            _ = (numbers, positions, ihelp)
            return (self.kn, self.mask, self.idx)

        @override
        def get_monopole_atom_potential(
            self,
            cache: tuple[_KeepDtypeFloatNode, Tensor, Tensor],
            qat: Tensor,
            **_: Any,
        ) -> Tensor:
            node_c, mask_c, idx_c = cache
            scale = node_c.primary + node_c.ref_fp64.to(qat.dtype)
            return torch.where(
                mask_c, scale * qat[idx_c], torch.zeros_like(qat)
            )

        @override
        def get_monopole_atom_energy(
            self,
            cache: tuple[_KeepDtypeFloatNode, Tensor, Tensor],
            qat: Tensor,
            **_: Any,
        ) -> Tensor:
            return 0.5 * qat * self.get_monopole_atom_potential(cache, qat)

    kn64 = _KeepDtypeFloatNode(
        primary=torch.tensor(0.04, **DD64),
        ref_fp64=torch.tensor(0.02, **DD64),
    )
    with pytest.raises(ValueError, match="Distinct interactions"):
        InteractionList(_KeptSCFInteraction(kn64), _KeptSCFInteraction(kn64))

    calc_ki = Calculator(
        numbers,
        GFN1_XTB,
        interaction=(_KeptSCFInteraction(kn64),),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    e64 = calc_ki.singlepoint(pos).energy.sum()
    calc_ki32 = calc_ki.type(torch.float32)
    conv_inter = calc_ki32.interactions.components[-1]
    assert isinstance(conv_inter, _KeptSCFInteraction)
    assert conv_inter.kn.ref_fp64.dtype == torch.float64
    e32 = calc_ki32.singlepoint(pos.to(torch.float32)).energy.sum()
    assert e32.dtype == torch.float32
    torch.testing.assert_close(e32.to(torch.float64), e64, rtol=1e-4, atol=1e-4)

    class _TamperIhelpInteraction(Interaction):
        def __init__(self) -> None:
            super().__init__(**DD64)

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> Tensor:
            _ = (numbers, positions)
            assert ihelp is not None
            ihelp.shell_index[0] = 99
            return torch.zeros(1, **DD64)

    sys_tamper = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        interaction=(_TamperIhelpInteraction(),),
    ).setup(numbers)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        sys_tamper.singlepoint(pos)

    # 6. Custom Classical.get_cache mutating ihelp in Model.setup or fallback is rejected
    class _TamperSetupClassical(Classical):
        def __init__(self) -> None:
            super().__init__(**DD64)

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = numbers
            assert ihelp is not None
            ihelp.shells_to_atom[0] = 1
            return torch.zeros(1, **DD64)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            _ = cache
            return positions.new_zeros(positions.shape[:-1])

    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        Model(
            par=par,
            config=Config.create(int_driver="pytorch"),
            classical=(_TamperSetupClassical(),),
        ).setup(numbers.clone())

    class _TamperFallbackClassical(Classical):
        def __init__(self) -> None:
            super().__init__(**DD64)
            self.arm = False

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = numbers
            if self.arm:
                assert ihelp is not None
                ihelp.shells_to_atom[0] = 1
            return torch.zeros(1, **DD64)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            _ = cache
            return positions.new_zeros(positions.shape[:-1])

    tf_cls = _TamperFallbackClassical()
    sys_tf = Model(
        par=par,
        config=Config.create(int_driver="pytorch"),
        classical=(tf_cls,),
    ).setup(numbers.clone())
    tf_cls.arm = True
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        sys_tf.singlepoint(pos)

    # 7. Wrong-dtype non-keep_dtype floating leaf on custom Interaction raises DtypeError
    class _WrongDtypeInteraction(Interaction):
        def __init__(self) -> None:
            super().__init__(**DD64)
            self.w = torch.tensor([0.05, 0.05], **DD64)

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> Tensor:
            _ = (numbers, positions, ihelp)
            return self.w

        @override
        def get_monopole_atom_potential(
            self, cache: Tensor, qat: Tensor, **_: Any
        ) -> Tensor:
            return cache * qat

        @override
        def get_monopole_atom_energy(
            self, cache: Tensor, qat: Tensor, **_: Any
        ) -> Tensor:
            return 0.5 * cache * qat.square()

    sys_wrong_dt = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float32),
        config=Config.create(int_driver="pytorch"),
        interaction=(_WrongDtypeInteraction(),),
    ).setup(numbers.clone())
    with pytest.raises(DtypeError):
        sys_wrong_dt.singlepoint(pos.to(torch.float32))

    # 8. _convert_eval_system_dtype with exact_core=None remaps owners without mutating input
    nocore64 = replace(sys_plain, exact_core=None)
    nocore32 = _convert_eval_system_dtype(nocore64, torch.float32)
    assert nocore64.classicals.dtype == torch.float64
    assert nocore32.repulsion_classical in nocore32.classicals.components
    res_nocore32 = nocore32.singlepoint(pos_w[0].to(torch.float32))
    assert res_nocore32.energy.dtype == torch.float32


def test_custom_classical_cache_invalidation_preserved_across_dtype_conversion() -> (
    None
):
    """Mutated custom Classical state stays invalidated across Calculator.type and _convert_eval_system_dtype."""
    from dxtb._src.calculators.types.base import (
        _convert_eval_system_dtype,
        _convert_system_dtype,
    )

    class _OpaqueTag:
        def __init__(self, name: str) -> None:
            self.name = name

    class _ScaledAliasCache(ClassicalCache):
        __slots__ = ["shared", "buf"]

        def __init__(self, shared: Tensor, buf: Tensor) -> None:
            super().__init__(device=shared.device, dtype=shared.dtype)
            self.shared = shared
            self.buf = buf
            self.note = "clean"

    class ScaledClassical(Classical):
        """Tensor-only Classical matching the WP-CC-H3 reproduction."""

        def __init__(
            self,
            factor: float,
            weight: Tensor,
            *,
            label: str = "ScaledClassical",
        ) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.label = label
            self.factor = factor
            self.left = weight
            self.right = weight
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> _ScaledAliasCache:
            _ = ihelp
            self.cache_calls += 1
            active = (numbers > 0).to(device=self.device, dtype=self.dtype)
            buf = active * torch.full(
                numbers.shape,
                self.factor,
                dtype=self.dtype,
                device=self.device,
            )
            return _ScaledAliasCache(self.left, buf)

        @override
        def get_energy(
            self, positions: Tensor, cache: _ScaledAliasCache, **_: Any
        ) -> Tensor:
            return (
                0.25
                * (self.left + self.right + cache.shared)
                * cache.buf
                * positions.square().sum(dim=-1)
            )

    class OpaqueScaledClassical(Classical):
        """Opaque Classical always evaluated through _evaluate_custom_classical_fallback."""

        def __init__(
            self,
            factor: float,
            weight: Tensor,
            *,
            label: str = "OpaqueScaled",
        ) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.label = label
            self.factor = factor
            self.weight = weight
            self.tag = _OpaqueTag(label)
            self.cache_calls = 0

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            self.cache_calls += 1
            active = (numbers > 0).to(device=self.device, dtype=self.dtype)
            return (
                active
                * self.weight
                * torch.full(
                    numbers.shape,
                    self.factor,
                    dtype=self.dtype,
                    device=self.device,
                )
            )

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return 0.75 * cache * positions.square().sum(dim=-1)

    numbers = torch.tensor([8, 1, 17], dtype=torch.int64, device=DEVICE)
    pos64 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [2.4, 0.3, -0.2]], **DD64
    )
    pos32 = pos64.to(dtype=torch.float32)
    par64 = ParamModule(GFN1_XTB, **DD64)
    par32 = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float32)

    # 1. Exact WP-CC-H3 reproduction across SCF-enabled and exclude={"scf"},
    #    covering scalar mutation, tensor replacement, and in-place tensor mutation
    #    for both tensor-only and opaque custom classicals, with energy + first derivatives.
    for exclude_scf in (False, True):
        cfg = (
            Config.create(int_driver="pytorch", exclude="scf")
            if exclude_scf
            else Config.create(int_driver="pytorch")
        )
        opts: dict[str, Any] = {"verbosity": 0, "int_driver": "pytorch"}
        if exclude_scf:
            opts["exclude"] = ["scf"]

        for cls_factory, is_core in (
            (ScaledClassical, True),
            (OpaqueScaledClassical, False),
        ):
            for mut_kind in ("scalar", "replace_tensor", "inplace_tensor"):
                w64 = torch.tensor(0.06, **DD64, requires_grad=True)
                comp = cls_factory(1.0, w64)
                sys64 = Model(par=par64, config=cfg, classical=(comp,)).setup(
                    numbers
                )
                assert sys64._active_exact_core() is not None

                expected_factor = 1.0
                expected_w_val = 0.06
                if mut_kind == "scalar":
                    comp.factor = 2.0
                    expected_factor = 2.0
                    grad_source = w64
                elif mut_kind == "replace_tensor":
                    w64_new = torch.tensor(0.11, **DD64, requires_grad=True)
                    if isinstance(comp, ScaledClassical):
                        comp.left = w64_new
                        comp.right = w64_new
                    else:
                        comp.weight = w64_new
                    expected_w_val = 0.11
                    grad_source = w64_new
                else:
                    w64_mut = w64.detach().clone()
                    with torch.no_grad():
                        w64_mut.mul_(2.25)
                    w64_mut.requires_grad_(True)
                    if isinstance(comp, ScaledClassical):
                        comp.left = w64_mut
                        comp.right = w64_mut
                    else:
                        comp.weight = w64_mut
                    # Also verify genuine in-place _version bump on an existing leaf:
                    sys64_ip = Model(
                        par=par64, config=cfg, classical=(comp,)
                    ).setup(numbers)
                    with torch.no_grad():
                        w64_mut.add_(0.03)
                    expected_w_val = 0.06 * 2.25 + 0.03
                    sys64 = sys64_ip
                    grad_source = w64_mut

                if is_core:
                    assert sys64._active_exact_core() is None

                sys32 = _convert_eval_system_dtype(sys64, torch.float32)
                if is_core:
                    assert sys32._active_exact_core() is None
                    assert sys32.exact_core is not None
                    inv_term = sys32.exact_core.classical_terms[-1]
                    assert isinstance(inv_term.setup, CustomClassicalCoreSetup)
                    assert inv_term.setup.template.cache is None
                    with pytest.raises(RuntimeError, match="invalidated"):
                        inv_term.setup.reconstruct()
                assert comp.label not in sys32._packed_classical_setups
                assert comp.label not in sys32.classical_cache
                assert sys32._custom_classical_snapshots == ()

                comp32 = sys32.classicals.components[-1]
                param32 = (
                    comp32.left
                    if isinstance(comp32, ScaledClassical)
                    else comp32.weight
                )
                p_eval = pos32.clone().requires_grad_(True)
                res_conv = sys32.singlepoint(p_eval)

                w_ref32 = torch.tensor(
                    expected_w_val,
                    device=DEVICE,
                    dtype=torch.float32,
                    requires_grad=True,
                )
                ref_comp32 = cls_factory(expected_factor, w_ref32)
                p_ref = pos32.clone().requires_grad_(True)
                res_ref = (
                    Model(par=par32, config=cfg, classical=(ref_comp32,))
                    .setup(numbers)
                    .singlepoint(p_ref)
                )

                rtol = 1e-5 if not exclude_scf else 1e-6
                atol = 1e-5 if not exclude_scf else 1e-6
                torch.testing.assert_close(
                    res_conv.energy, res_ref.energy, rtol=rtol, atol=atol
                )
                g_pos, g_p32, g_src64 = torch.autograd.grad(
                    res_conv.energy.sum(),
                    (p_eval, param32, grad_source),
                )
                g_pos_ref, g_w_ref = torch.autograd.grad(
                    res_ref.energy.sum(), (p_ref, w_ref32)
                )
                torch.testing.assert_close(
                    g_pos, g_pos_ref, rtol=rtol, atol=atol
                )
                torch.testing.assert_close(g_p32, g_w_ref, rtol=rtol, atol=atol)
                torch.testing.assert_close(
                    g_src64.to(dtype=torch.float32),
                    g_w_ref,
                    rtol=rtol,
                    atol=atol,
                )

        # Calculator.type(torch.float32) after pre-conversion mutation (1D)
        w_calc64 = torch.tensor(0.08, **DD64, requires_grad=True)
        c_calc = ScaledClassical(1.0, w_calc64)
        calc64 = Calculator(
            numbers,
            GFN1_XTB,
            classical=(c_calc,),
            opts=dict(opts),
            **DD64,
        )
        c_calc.factor = 2.25
        assert calc64.system._active_exact_core() is None
        calc32 = calc64.type(torch.float32)
        assert calc32.system._active_exact_core() is None
        assert c_calc.label not in calc32.system._packed_classical_setups
        assert c_calc.label not in calc32.system.classical_cache
        assert calc32.system._custom_classical_snapshots == ()
        assert calc32.system.exact_core is not None
        assert (
            calc32.system.exact_core.classical_terms[-1].setup.template.cache  # type: ignore[union-attr]
            is None
        )

        p_calc = pos32.clone().requires_grad_(True)
        e_calc = calc32.singlepoint(p_calc).energy
        w_calc_ref = torch.tensor(
            0.08, device=DEVICE, dtype=torch.float32, requires_grad=True
        )
        p_calc_ref = pos32.clone().requires_grad_(True)
        calc_ref32 = Calculator(
            numbers,
            par32,
            classical=(ScaledClassical(2.25, w_calc_ref),),
            opts=dict(opts),
            device=DEVICE,
            dtype=torch.float32,
        )
        e_calc_ref = calc_ref32.singlepoint(p_calc_ref).energy
        torch.testing.assert_close(e_calc, e_calc_ref, rtol=1e-5, atol=1e-5)
        c_calc32 = calc32.classicals.components[-1]
        assert isinstance(c_calc32, ScaledClassical)
        gc_pos, gc_p32, gc_src64 = torch.autograd.grad(
            e_calc.sum(), (p_calc, c_calc32.left, w_calc64)
        )
        gc_pos_ref, gc_w_ref = torch.autograd.grad(
            e_calc_ref.sum(), (p_calc_ref, w_calc_ref)
        )
        torch.testing.assert_close(gc_pos, gc_pos_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(gc_p32, gc_w_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(
            gc_src64.to(torch.float32), gc_w_ref, rtol=1e-5, atol=1e-5
        )

    # 2. Direct calc32.system evaluation when exclude="all" AND direct _convert_system_dtype
    c_excl = ScaledClassical(1.0, torch.tensor(0.5, **DD64))
    calc_excl = Calculator(
        numbers,
        par64,
        classical=(c_excl,),
        opts={"verbosity": 0, "int_driver": "pytorch", "exclude": ["all"]},
        **DD64,
    )
    c_excl.factor = 2.0
    assert calc_excl.system._active_exact_core() is None
    calc_excl32 = calc_excl.type(torch.float32)
    assert calc_excl32.system._active_exact_core() is None
    torch.testing.assert_close(
        calc_excl32.system.singlepoint(pos32).energy,
        calc_excl32.singlepoint(pos32).energy,
    )
    sys_direct_conv = _convert_system_dtype(calc_excl.system, torch.float32)
    assert sys_direct_conv._active_exact_core() is None
    assert c_excl.label not in sys_direct_conv.classical_cache

    # 3. Batched Calculator.type(torch.float32) (batch_mode=1 padded and batch_mode=2 conformer)
    nums_pad = torch.tensor([[3, 1, 0], [8, 1, 17]], device=DEVICE)
    pos_pad32 = torch.stack(
        [
            torch.tensor(
                [[0.0, 0.0, 0.0], [0.0, 0.0, 3.0], [0.0, 0.0, 0.0]],
                device=DEVICE,
                dtype=torch.float32,
            ),
            pos32,
        ]
    )
    nums_conf = torch.stack([numbers, numbers])
    pos_conf32 = torch.stack([pos32, pos32 * 1.03])
    for batch_mode, b_nums, b_pos in (
        (1, nums_pad, pos_pad32),
        (2, nums_conf, pos_conf32),
    ):
        cb = ScaledClassical(1.0, torch.tensor(0.05, **DD64))
        bcalc = Calculator(
            b_nums,
            par64,
            classical=(cb,),
            opts={
                "verbosity": 0,
                "int_driver": "pytorch",
                "batch_mode": batch_mode,
            },
            **DD64,
        )
        cb.factor = 1.75
        bcalc32 = bcalc.type(torch.float32)
        assert cb.label not in bcalc32.system._packed_classical_setups
        assert cb.label not in bcalc32.system.classical_cache
        assert bcalc32.system._custom_classical_snapshots == ()
        bref32 = Calculator(
            b_nums,
            par32,
            classical=(
                ScaledClassical(
                    1.75,
                    torch.tensor(0.05, device=DEVICE, dtype=torch.float32),
                ),
            ),
            opts={
                "verbosity": 0,
                "int_driver": "pytorch",
                "batch_mode": batch_mode,
            },
            device=DEVICE,
            dtype=torch.float32,
        )
        torch.testing.assert_close(
            bcalc32.singlepoint(b_pos).energy,
            bref32.singlepoint(b_pos).energy,
            rtol=1e-5,
            atol=1e-5,
        )

    # 4. Multiple custom classicals when only one was invalidated, plus pre-conversion
    #    cache mutation and preservation of unrelated built-ins & valid H2 shared-cache identity.
    pure_w = torch.tensor(0.04, **DD64, requires_grad=True)
    mut_w = torch.tensor(0.07, **DD64, requires_grad=True)
    pure_cls = ScaledClassical(1.25, pure_w, label="PureCustom")
    mut_cls = ScaledClassical(1.0, mut_w, label="MutCustom")
    cfg_full = Config.create(int_driver="pytorch")
    sys_multi64 = Model(
        par=par64, config=cfg_full, classical=(pure_cls, mut_cls)
    ).setup(numbers)
    assert (pure_cls.cache_calls, mut_cls.cache_calls) == (1, 1)

    mut_cls.factor = 2.5
    assert sys_multi64._active_exact_core() is None
    sys_multi32 = _convert_eval_system_dtype(sys_multi64, torch.float32)
    assert sys_multi32._active_exact_core() is None

    # Built-in setups were converted without discarding any built-in term
    assert sys_multi32.repulsion_setup is not None
    assert sys_multi32.halogen_setup is not None
    assert len(sys_multi32.d3_setups) == 1
    assert sys_multi32.es2_setup is not None
    assert sys_multi32.es3_setup is not None
    assert sys_multi32.h0_setup is not None
    assert sys_multi32.integral_setup is not None

    # Valid PureCustom keeps H2 shared-cache & alias identity; MutCustom is purged
    assert "PureCustom" in sys_multi32.classical_cache
    assert "MutCustom" not in sys_multi32.classical_cache
    assert "MutCustom" not in sys_multi32._packed_classical_setups
    assert len(sys_multi32._custom_classical_snapshots) == 1
    pure_cls32 = sys_multi32.classicals.components[-2]
    mut_cls32 = sys_multi32.classicals.components[-1]
    assert isinstance(pure_cls32, ScaledClassical)
    assert isinstance(mut_cls32, ScaledClassical)
    assert sys_multi32._custom_classical_snapshots[0][0] is pure_cls32
    pure_cache32 = sys_multi32.classical_cache["PureCustom"]
    assert pure_cache32 is sys_multi32._packed_classical_setups["PureCustom"]
    assert pure_cache32 is sys_multi32._custom_classical_snapshots[0][2]
    assert sys_multi32.exact_core is not None
    pure_core_term = sys_multi32.exact_core.classical_terms[-2]
    mut_core_term = sys_multi32.exact_core.classical_terms[-1]
    assert isinstance(pure_core_term.setup, CustomClassicalCoreSetup)
    assert isinstance(mut_core_term.setup, CustomClassicalCoreSetup)
    assert pure_core_term.setup.template.cache is pure_cache32
    assert mut_core_term.setup.template.cache is None
    assert pure_cls32.left is pure_cls32.right
    assert pure_cls32.left is pure_cache32.shared

    # Evaluate twice: PureCustom reuses its converted setup (cache_calls stays 1),
    # while MutCustom recomputes fresh on each fallback call.
    out_m1 = sys_multi32.singlepoint(pos32)
    out_m2 = sys_multi32.singlepoint(pos32)
    assert pure_cls32.cache_calls == 1
    assert mut_cls32.cache_calls == 3
    torch.testing.assert_close(out_m1.energy, out_m2.energy)
    ref_multi32 = (
        Model(
            par=par32,
            config=cfg_full,
            classical=(
                ScaledClassical(
                    1.25,
                    torch.tensor(0.04, device=DEVICE, dtype=torch.float32),
                    label="PureCustom",
                ),
                ScaledClassical(
                    2.5,
                    torch.tensor(0.07, device=DEVICE, dtype=torch.float32),
                    label="MutCustom",
                ),
            ),
        )
        .setup(numbers)
        .singlepoint(pos32)
    )
    torch.testing.assert_close(
        out_m1.energy, ref_multi32.energy, rtol=1e-5, atol=1e-5
    )

    # 5. Originally valid snapshot preserves active exact core and H2 shared-cache invalidation,
    #    and mutating the captured cache BEFORE conversion also preserves invalidation.
    valid_comp = ScaledClassical(
        1.5, torch.tensor(0.05, **DD64), label="ValidCustom"
    )
    valid_sys64 = Model(
        par=par64,
        config=Config.create(int_driver="pytorch", exclude="scf"),
        classical=(valid_comp,),
    ).setup(numbers)
    valid_sys32 = _convert_eval_system_dtype(valid_sys64, torch.float32)
    assert valid_sys32._active_exact_core() is valid_sys32.exact_core
    v_cache32 = valid_sys32.classical_cache["ValidCustom"]
    assert valid_sys32.exact_core is not None
    assert (
        valid_sys32.exact_core.classical_terms[-1].setup.template.cache  # type: ignore[union-attr]
        is v_cache32
    )
    v_cache32.note = "post_conv_dirty"
    assert valid_sys32._active_exact_core() is None

    pre_dirty_comp = ScaledClassical(
        1.5, torch.tensor(0.05, **DD64), label="PreDirtyCache"
    )
    pre_dirty_sys64 = Model(
        par=par64,
        config=Config.create(int_driver="pytorch", exclude="scf"),
        classical=(pre_dirty_comp,),
    ).setup(numbers)
    pre_dirty_sys64.classical_cache["PreDirtyCache"].note = "pre_conv_dirty"
    assert pre_dirty_sys64._active_exact_core() is None
    pre_dirty_sys32 = _convert_eval_system_dtype(pre_dirty_sys64, torch.float32)
    assert pre_dirty_sys32._active_exact_core() is None
    assert "PreDirtyCache" not in pre_dirty_sys32.classical_cache
    assert pre_dirty_sys32._custom_classical_snapshots == ()


def test_duplicate_custom_classical_labels_preserve_label_in_get_cache() -> (
    None
):
    """Same-label custom Classicals see their real self.label in get_cache across Model.setup and Calculator."""
    seen_labels: list[str] = []
    call_counts: dict[int, int] = {}
    label_factors = {
        "SharedCustomLabel": 1.5,
        "InitialB": 0.25,
        "TempSynthetic": 100.0,
    }

    class LabelDependentClassical(Classical):
        __slots__ = ["weight"]

        def __init__(
            self,
            weight: Tensor,
            label: str = "SharedCustomLabel",
        ) -> None:
            super().__init__(device=weight.device, dtype=weight.dtype)
            self.weight = weight
            self.label = label

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            seen_labels.append(self.label)
            call_counts[id(self)] = call_counts.get(id(self), 0) + 1
            factor = label_factors[self.label]
            mask = (numbers > 0).to(dtype=self.dtype)
            return factor * self.weight * mask

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return cache * positions.square().sum(dim=-1)

    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.2]], **DD64)
    nums_bm2 = torch.stack([numbers, numbers], dim=0)
    pos_bm2 = torch.stack([pos_a, pos_b], dim=0).requires_grad_(True)
    par64 = ParamModule(GFN1_XTB, **DD64)
    par32 = ParamModule(GFN1_XTB, device=DEVICE, dtype=torch.float32)
    cfg_excl = Config.create(exclude="all")

    # 1. Direct single-System Model.setup(batch_mode=0) still rejects duplicate labels,
    #    while Model.setup(batch_mode=2) preserves real labels for each instance.
    m_a = LabelDependentClassical(torch.tensor(0.05, **DD64))
    m_b = LabelDependentClassical(torch.tensor(0.12, **DD64))
    with pytest.raises(ValueError, match="SharedCustomLabel"):
        Model(par=par64, config=cfg_excl, classical=(m_a, m_b)).setup(numbers)
    sys_bm2 = Model(
        par=par64,
        config=cfg_excl,
        classical=(m_a, m_b),
    ).setup(nums_bm2, batch_mode=2)
    assert len(sys_bm2._custom_classical_snapshots) == 2
    model_bm2_slots = [
        owner.get_energy(pos_bm2.detach(), captured_setup)
        for owner, _, captured_setup in sys_bm2._custom_classical_snapshots
    ]
    model_bm2_energy = model_bm2_slots[0] + model_bm2_slots[1]

    ref_sys_a = Model(
        par=par64,
        config=cfg_excl,
        classical=(LabelDependentClassical(torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    ref_sys_b = Model(
        par=par64,
        config=cfg_excl,
        classical=(LabelDependentClassical(torch.tensor(0.12, **DD64)),),
    ).setup(numbers)
    ref_bm2_energy = torch.stack(
        [
            ref_sys_a.singlepoint(pos_a).energy
            + ref_sys_b.singlepoint(pos_a).energy,
            ref_sys_a.singlepoint(pos_b).energy
            + ref_sys_b.singlepoint(pos_b).energy,
        ],
        dim=0,
    )
    torch.testing.assert_close(model_bm2_energy, ref_bm2_energy)

    # 2. Calculator.singlepoint() in batch_mode=2 preserves real labels, vmap core, and gradients
    seen_labels.clear()
    w_a = torch.tensor(0.05, **DD64, requires_grad=True)
    w_b = torch.tensor(0.12, **DD64, requires_grad=True)
    c_a = LabelDependentClassical(w_a)
    c_b = LabelDependentClassical(w_b)
    calc_bm2 = Calculator(
        nums_bm2,
        par64,
        classical=(c_a, c_b),
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "exclude": ["all"],
            "batch_mode": 2,
        },
        **DD64,
    )
    lanes_bm2 = build_calculator_lane_systems(calc_bm2)
    assert lanes_bm2[0] is lanes_bm2[1]
    assert lanes_bm2[0]._active_exact_core() is lanes_bm2[0].exact_core
    assert len(lanes_bm2[0].exact_core.classical_terms) == 2  # type: ignore[union-attr]

    res_bm2 = calc_bm2.singlepoint(pos_bm2)
    assert set(seen_labels) == {"SharedCustomLabel"}
    assert [lbl for lbl, _ in res_bm2.classical] == [
        "SharedCustomLabel",
        "SharedCustomLabel",
    ]
    torch.testing.assert_close(res_bm2.classical[0][1], model_bm2_slots[0])
    torch.testing.assert_close(res_bm2.classical[1][1], model_bm2_slots[1])
    torch.testing.assert_close(res_bm2.energy, model_bm2_energy)

    g_pos, g_wa, g_wb = torch.autograd.grad(
        res_bm2.energy.sum(), (pos_bm2, w_a, w_b)
    )
    expected_pos_grad = 2.0 * 1.5 * (0.05 + 0.12) * pos_bm2.detach()
    expected_w_grad = 1.5 * pos_bm2.detach().square().sum()
    torch.testing.assert_close(g_pos, expected_pos_grad)
    torch.testing.assert_close(g_wa, expected_w_grad)
    torch.testing.assert_close(g_wb, expected_w_grad)

    # 3. Calculator.type(torch.float32) on batch_mode=2 preserves label-dependent behavior
    seen_labels.clear()
    calc_bm2_32 = calc_bm2.type(torch.float32)
    res_bm2_32 = calc_bm2_32.singlepoint(pos_bm2.detach().to(torch.float32))
    assert set(seen_labels) == {"SharedCustomLabel"}
    torch.testing.assert_close(
        res_bm2_32.energy,
        model_bm2_energy.to(torch.float32),
        rtol=1e-5,
        atol=1e-6,
    )

    # 4. Padded batch (batch_mode=1) and 1D post-init collision with SCF active
    nums_w = torch.tensor([8, 1, 1], device=DEVICE)
    pos_w = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.9], [-1.4, 0.0, 0.9]], **DD64
    )
    padded_nums = torch.tensor([[8, 1, 1], [3, 1, 0]], device=DEVICE)
    padded_pos = torch.stack(
        [pos_w, torch.cat([pos_a, torch.zeros((1, 3), **DD64)], dim=0)],
        dim=0,
    )
    seen_labels.clear()
    calc_bm1 = Calculator(
        padded_nums,
        par64,
        classical=(
            LabelDependentClassical(torch.tensor(0.03, **DD64)),
            LabelDependentClassical(torch.tensor(0.07, **DD64)),
        ),
        opts={"verbosity": 0, "int_driver": "pytorch", "batch_mode": 1},
        **DD64,
    )
    lanes_bm1 = build_calculator_lane_systems(calc_bm1)
    assert all(can_vmap_system(lane) for lane in lanes_bm1)
    res_bm1 = calc_bm1.singlepoint(padded_pos)
    assert set(seen_labels) == {"SharedCustomLabel"}
    ref_bm1_w = (
        Model(
            par=par64,
            config=Config.create(int_driver="pytorch"),
            classical=(LabelDependentClassical(torch.tensor(0.10, **DD64)),),
        )
        .setup(nums_w)
        .singlepoint(pos_w)
    )
    ref_bm1_lih = (
        Model(
            par=par64,
            config=Config.create(int_driver="pytorch"),
            classical=(LabelDependentClassical(torch.tensor(0.10, **DD64)),),
        )
        .setup(numbers)
        .singlepoint(pos_a)
    )
    torch.testing.assert_close(res_bm1.energy[0], ref_bm1_w.energy)
    torch.testing.assert_close(res_bm1.energy[1, :2], ref_bm1_lih.energy)

    d1_a = LabelDependentClassical(torch.tensor(0.03, **DD64))
    d1_b = LabelDependentClassical(torch.tensor(0.07, **DD64), label="InitialB")
    calc_1d = Calculator(
        numbers,
        par64,
        classical=(d1_a, d1_b),
        opts={"verbosity": 0, "int_driver": "pytorch"},
        **DD64,
    )
    seen_labels.clear()
    d1_b.label = "SharedCustomLabel"
    res_1d = calc_1d.singlepoint(pos_a)
    assert seen_labels == ["SharedCustomLabel", "SharedCustomLabel"]
    torch.testing.assert_close(res_1d.energy, ref_bm1_lih.energy)

    # Also verify Calculator.type(torch.float32) after 1D post-init label collision
    calc_1d_32 = calc_1d.type(torch.float32)
    ref_1d_32 = (
        Model(
            par=par32,
            config=Config.create(int_driver="pytorch"),
            classical=(
                LabelDependentClassical(
                    torch.tensor(0.10, device=DEVICE, dtype=torch.float32)
                ),
            ),
        )
        .setup(numbers)
        .singlepoint(pos_a.to(torch.float32))
    )
    torch.testing.assert_close(
        calc_1d_32.singlepoint(pos_a.to(torch.float32)).energy,
        ref_1d_32.energy,
        rtol=1e-5,
        atol=1e-5,
    )

    # 5. Direct _sync_lane_system_labels invalidates a cache built under a different label
    temp_comp = LabelDependentClassical(
        torch.tensor(0.05, **DD64), label="TempSynthetic"
    )
    lane_sys = Model(par=par64, config=cfg_excl, classical=(temp_comp,)).setup(
        numbers
    )
    assert lane_sys._active_exact_core() is lane_sys.exact_core
    restored_source = LabelDependentClassical(
        torch.tensor(0.05, **DD64), label="SharedCustomLabel"
    )
    synced_sys = _sync_lane_system_labels(
        lane_sys,
        (restored_source,),
        (),
        {id(temp_comp)},
    )
    assert synced_sys._active_exact_core() is None
    assert synced_sys._custom_classical_snapshots == ()
    assert "TempSynthetic" not in synced_sys._packed_classical_setups
    assert "SharedCustomLabel" not in synced_sys._packed_classical_setups
    torch.testing.assert_close(
        synced_sys.singlepoint(pos_a).energy,
        ref_sys_a.singlepoint(pos_a).energy,
    )

    # 6. Built-in + invalidated custom Classical sharing a non-reserved label does not
    #    leak the invalidated custom cache into _packed_classical_setups across .type()
    from dxtb.components.dispersion import new_dispersion

    d3_shared = new_dispersion(nums_bm2, par64, **DD64)
    assert d3_shared is not None
    d3_shared.label = "SharedCustomLabel"
    c_shared = LabelDependentClassical(torch.tensor(0.05, **DD64))
    calc_mixed = Calculator(
        nums_bm2,
        par64,
        classical=(d3_shared, c_shared),
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "exclude": ["disp", "scf"],
            "batch_mode": 2,
        },
        **DD64,
    )
    c_shared.weight = torch.tensor(0.20, **DD64)
    calc_mixed32 = calc_mixed.type(torch.float32)
    assert (
        "SharedCustomLabel" not in calc_mixed32.system._packed_classical_setups
    )
    ref_mixed32 = Calculator(
        nums_bm2,
        par32,
        classical=(
            d3_shared.type(torch.float32),
            LabelDependentClassical(
                torch.tensor(0.20, device=DEVICE, dtype=torch.float32)
            ),
        ),
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "exclude": ["disp", "scf"],
            "batch_mode": 2,
        },
        device=DEVICE,
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        calc_mixed32.singlepoint(pos_bm2.detach().to(torch.float32)).energy,
        ref_mixed32.singlepoint(pos_bm2.detach().to(torch.float32)).energy,
        rtol=1e-5,
        atol=1e-5,
    )
