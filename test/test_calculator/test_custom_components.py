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

    force_constant: Tensor = child()
    atom_weights: Tensor = child()


class HarmonicRestraint(Classical):
    """Weighted harmonic restraint around the origin."""

    force_constant: Tensor = child()

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
        return HarmonicRestraintCache(
            force_constant=self.force_constant, atom_weights=weights
        )

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

    shift: Tensor = child()
    hardness: Tensor = child()


class OnsiteShift(Interaction):
    """Quadratic atomic charge penalty coupled self-consistently."""

    scale: Tensor = child()
    hardness: Tensor = child()

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
        return OnsiteShiftCache(shift=radial, hardness=self.hardness)

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

    shell_k: Tensor = child()
    mono_k: Tensor = child()
    dip_k: Tensor = child()
    quad_k: Tensor = child()
    geom_factor: Tensor = child()


class FullMultipolePenalty(Interaction):
    """Self-consistent penalty exercising shell monopole and atom multipole hooks."""

    shell_k: Tensor = child()
    mono_k: Tensor = child()
    dip_k: Tensor = child()
    quad_k: Tensor = child()

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
            shell_k=self.shell_k,
            mono_k=self.mono_k,
            dip_k=self.dip_k,
            quad_k=self.quad_k,
            geom_factor=geom_factor,
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

    quad_k: Tensor = child()

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
        return InteractionCache()

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
    restraint = HarmonicRestraint(force_constant=torch.tensor(0.05, **DD64)).to(
        **DD64
    )
    interaction = OnsiteShift(
        scale=torch.tensor(0.02, **DD64), hardness=torch.tensor(0.1, **DD64)
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

        restraint = HarmonicRestraint(force_constant=cls_k).to(**DD64)
        multipole = FullMultipolePenalty(
            shell_k=sh_k, mono_k=mono_k, dip_k=dip_k, quad_k=quad_k
        )
        quad_only = QuadrupoleOnlyInteraction(quad_k=quad_only_k)
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
            restraint = HarmonicRestraint(
                force_constant=torch.tensor(0.04, **DD64)
            ).to(**DD64)
            multipole = FullMultipolePenalty(
                shell_k=torch.tensor(0.06, **DD64),
                mono_k=torch.tensor(0.08, **DD64),
                dip_k=torch.tensor(0.02, **DD64),
                quad_k=torch.tensor(0.002, **DD64),
            )
            quad_only = QuadrupoleOnlyInteraction(
                quad_k=torch.tensor(0.001 + delta, **DD64)
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
            classical=(HarmonicRestraint(force_constant=cls_k).to(**DD64),),
            interaction=(OnsiteShift(scale=shift_s, hardness=hard_h),),
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
        restraint = HarmonicRestraint(force_constant=cls_k).to(**DD64)
        shift = OnsiteShift(scale=shift_s, hardness=hard_h)
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
    restraint = HarmonicRestraint(force_constant=torch.tensor(0.04, **DD64)).to(
        **DD64
    )
    shift = OnsiteShift(
        scale=torch.tensor(0.015, **DD64), hardness=torch.tensor(0.08, **DD64)
    )
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
    restraint = HarmonicRestraint(force_constant=torch.tensor(0.04, **DD64)).to(
        **DD64
    )
    shift = OnsiteShift(
        scale=torch.tensor(0.02, **DD64), hardness=torch.tensor(0.08, **DD64)
    )
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
        scale: Tensor = child()

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
    comp = DataDependentClassical(scale=scale)
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
        par=par,
        config=cfg,
        classical=(DataDependentClassical(scale=ref_scale),),
    ).setup(numbers)
    ref_pa = pos_a.clone().requires_grad_(True)
    ref_loss = (
        ref_sys.singlepoint(ref_pa).energy.sum()
        + ref_sys.singlepoint(pos_b).energy.sum()
    )
    ref_g_pa, ref_g_scale = torch.autograd.grad(ref_loss, (ref_pa, ref_scale))
    torch.testing.assert_close(g_pa, ref_g_pa)
    torch.testing.assert_close(g_scale, ref_g_scale)


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


def test_monopole_multipole_cross_hooks_potential_and_energy_consistency() -> (
    None
):
    """Monopole-dipole coupling reading qdp in get_monopole_atom_energy preserves variational forces."""

    class MonopoleDipoleCoupling(Interaction):
        coupling: Tensor = child()

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> InteractionCache:
            _ = (numbers, positions, ihelp)
            return InteractionCache()

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
            interaction=(MonopoleDipoleCoupling(coupling=k),),
        ).setup(numbers)
        e_tot = sys_c.singlepoint(p).energy.sum()
        gp, gk = torch.autograd.grad(e_tot, (p, k))
        return e_tot.detach(), gp, gk

    eu, gpu, gku = eval_coupling("unrolled")
    ei, gpi, gki = eval_coupling("implicit")
    torch.testing.assert_close(ei, eu, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(gpi, gpu, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(gki, gku, rtol=1e-6, atol=1e-7)


def test_exact_core_tree_stack_and_duplicate_label_non_tensor_classicals() -> (
    None
):
    """ExactSystemCore stacks across parameter values, and duplicate-label non-tensor Classicals keep setups."""

    class SliceTaggedHarmonic(Classical):
        k: Tensor = child()
        atom_start: int = context(default=0)

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return self.k * (numbers[self.atom_start :] > 0).to(
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
        classical=(SliceTaggedHarmonic(k=torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    sys_2 = Model(
        par=par,
        config=cfg,
        classical=(SliceTaggedHarmonic(k=torch.tensor(0.12, **DD64)),),
    ).setup(numbers)
    assert sys_1.exact_core is not None and sys_2.exact_core is not None
    stacked_core = tree_stack((sys_1.exact_core, sys_2.exact_core))
    stacked_custom = stacked_core.classical_terms[-1].setup
    assert isinstance(stacked_custom, CustomClassicalCoreSetup)
    assert stacked_custom.component.k.shape[0] == 2
    assert stacked_custom.cache.shape[0] == 2

    # Two tensor-free custom Classicals sharing a label keep their own setups
    class SharedLabelClassical(Classical):
        factor: float = context()
        label: str = context(default="SharedCustomClassical")

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            return torch.full(numbers.shape, self.factor, **DD64)

        @override
        def get_energy(
            self, positions: Tensor, cache: Tensor, **_: Any
        ) -> Tensor:
            return cache * positions.square().sum(dim=-1)

    expected = (0.05 + 0.20) * pos.square().sum()
    c_a_2d = SharedLabelClassical(factor=0.05)
    c_b_2d = SharedLabelClassical(factor=0.20)
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
            interaction=(OnsiteShift(scale=s_param, hardness=h_param),),
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
            scale=torch.tensor(0.015, **DD64),
            hardness=torch.tensor(0.08, **DD64),
        )
        inter_2 = OnsiteShift(
            scale=torch.tensor(0.045, **DD64),
            hardness=torch.tensor(0.22, **DD64),
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

    # 3. Distinct labels on 1D, batch_mode=2, and batch_mode=1 Calculators match
    #    the combined single-interaction physics
    s1 = torch.tensor(0.015, **DD64, requires_grad=True)
    h1 = torch.tensor(0.08, **DD64, requires_grad=True)
    s2 = torch.tensor(0.045, **DD64, requires_grad=True)
    h2 = torch.tensor(0.22, **DD64, requires_grad=True)

    for bm, nums, eval_pos in (
        (0, numbers, pos_a),
        (2, nums_bm2, pos_bm2),
        (1, nums_bm1, pos_bm1),
    ):
        inter_a = OnsiteShift(scale=s1, hardness=h1)
        inter_b = OnsiteShift(scale=s2, hardness=h2, label="OnsiteShiftSecond")
        opts: dict[str, Any] = {"verbosity": 0, "int_driver": "pytorch"}
        if bm != 0:
            opts["batch_mode"] = bm
        calc = Calculator(
            nums, GFN1_XTB, interaction=(inter_a, inter_b), opts=opts, **DD64
        )
        res_pair = calc.singlepoint(eval_pos)
        combined = OnsiteShift(scale=s1 + s2, hardness=h1 + h2)
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


def test_nonlinear_multipole_interaction_variational_and_second_derivatives() -> (
    None
):
    """Nonlinear shell/atom/dipole/quadrupole interaction satisfies variational potential = dE/dq and 2nd-order implicit parity."""

    class NonlinearMultipolarInteraction(Interaction):
        """Anharmonic charge + saturating dipole/quadrupole polarization + cross-multipole couplings."""

        c_sh3: Tensor = child()
        u_at4: Tensor = child()
        alpha_dp: Tensor = child()
        alpha_qp: Tensor = child()
        lam_md: Tensor = child()
        lam_dq: Tensor = child()

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> InteractionCache:
            _ = (numbers, positions, ihelp)
            return InteractionCache()

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
        c_sh3=torch.tensor(0.02, **DD64),
        u_at4=torch.tensor(0.03, **DD64),
        alpha_dp=torch.tensor(0.04, **DD64),
        alpha_qp=torch.tensor(0.015, **DD64),
        lam_md=torch.tensor(0.025, **DD64),
        lam_dq=torch.tensor(0.01, **DD64),
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
    ilist = InteractionList(base_inter)
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
            c_sh3=c_sh3,
            u_at4=u_at4,
            alpha_dp=alpha_dp,
            alpha_qp=alpha_qp,
            lam_md=lam_md,
            lam_dq=lam_dq,
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
            c_sh3=torch.tensor(0.02, **DD64),
            u_at4=torch.tensor(0.03, **DD64),
            alpha_dp=torch.tensor(0.04, **DD64),
            alpha_qp=torch.tensor(0.015, **DD64),
            lam_md=torch.tensor(0.025 + delta, **DD64),
            lam_dq=torch.tensor(0.01, **DD64),
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
        weight: Tensor = child()
        label: str = context(default="SharedCustomLabel")

        @override
        def get_cache(
            self, numbers: Tensor, ihelp: IndexHelper | None = None
        ) -> Tensor:
            _ = ihelp
            seen_labels.append(self.label)
            call_counts[id(self)] = call_counts.get(id(self), 0) + 1
            factor = label_factors[self.label]
            mask = (numbers > 0).to(dtype=self.weight.dtype)
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
    m_a = LabelDependentClassical(weight=torch.tensor(0.05, **DD64))
    m_b = LabelDependentClassical(weight=torch.tensor(0.12, **DD64))
    with pytest.raises(ValueError, match="SharedCustomLabel"):
        Model(par=par64, config=cfg_excl, classical=(m_a, m_b)).setup(numbers)
    sys_bm2 = Model(
        par=par64,
        config=cfg_excl,
        classical=(m_a, m_b),
    ).setup(nums_bm2, batch_mode=2)
    assert len(sys_bm2._custom_classical_setups) == 2
    model_bm2_slots = [
        owner.get_energy(pos_bm2.detach(), captured_setup)
        for owner, captured_setup in sys_bm2._custom_classical_setups
    ]
    model_bm2_energy = model_bm2_slots[0] + model_bm2_slots[1]

    ref_sys_a = Model(
        par=par64,
        config=cfg_excl,
        classical=(LabelDependentClassical(weight=torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    ref_sys_b = Model(
        par=par64,
        config=cfg_excl,
        classical=(LabelDependentClassical(weight=torch.tensor(0.12, **DD64)),),
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
    c_a = LabelDependentClassical(weight=w_a)
    c_b = LabelDependentClassical(weight=w_b)
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

    # 4. Padded batch (batch_mode=1) with SCF active
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
            LabelDependentClassical(weight=torch.tensor(0.03, **DD64)),
            LabelDependentClassical(weight=torch.tensor(0.07, **DD64)),
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
            classical=(
                LabelDependentClassical(weight=torch.tensor(0.1, **DD64)),
            ),
        )
        .setup(nums_w)
        .singlepoint(pos_w)
    )
    ref_bm1_lih = (
        Model(
            par=par64,
            config=Config.create(int_driver="pytorch"),
            classical=(
                LabelDependentClassical(weight=torch.tensor(0.1, **DD64)),
            ),
        )
        .setup(numbers)
        .singlepoint(pos_a)
    )
    torch.testing.assert_close(res_bm1.energy[0], ref_bm1_w.energy)
    torch.testing.assert_close(res_bm1.energy[1, :2], ref_bm1_lih.energy)


def test_vmap_fallback_source_id_and_user_errors() -> None:
    """
    A data-dependent custom component falls back from vmap and reports the
    failing source lane; an unrelated user error mentioning vmap propagates.
    """
    numbers, pos_a = _lih_inputs()
    pos_b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 3.3]], **DD64)
    par = ParamModule(GFN1_XTB, **DD64)
    cfg = Config.create(int_driver="pytorch")

    class ThresholdItemClassical(Classical):
        k: Tensor = child()

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
        classical=(ThresholdItemClassical(k=torch.tensor(0.05, **DD64)),),
    ).setup(numbers)
    assert can_vmap_system(sys_thresh)
    group = LaneGroup(system=sys_thresh, source_ids=(4, 19), use_vmap=True)
    charges = {4: torch.tensor(0.0, **DD64), 19: torch.tensor(0.0, **DD64)}
    none: dict[int, Tensor | None] = {4: None, 19: None}
    with pytest.raises(ValueError, match="bond stretched") as exc_lane:
        evaluate_lane_group(
            group, {4: pos_a, 19: pos_b}, charges, none, none, none
        )
    assert getattr(exc_lane.value, "source_id", None) == 19

    user_calls: list[str] = []

    class UserErrorInteraction(Interaction):
        k: Tensor = child()

        @override
        def get_cache(
            self,
            *,
            numbers: Tensor | None = None,
            positions: Tensor | None = None,
            ihelp: IndexHelper | None = None,
        ) -> InteractionCache:
            _ = (numbers, positions, ihelp)
            return InteractionCache()

        @override
        def get_monopole_atom_potential(
            self,
            cache: InteractionCache,
            qat: Tensor,
            qdp: Tensor | None = None,
            qqp: Tensor | None = None,
        ) -> Tensor:
            _ = (cache, qat, qdp, qqp)
            user_calls.append(self.label)
            raise RuntimeError(
                "vmap: custom domain validation failed inside user hook"
            )

    sys_uerr = Model(
        par=par,
        config=cfg,
        interaction=(UserErrorInteraction(k=torch.tensor(0.02, **DD64)),),
    ).setup(numbers)
    assert can_vmap_system(sys_uerr)
    group = LaneGroup(system=sys_uerr, source_ids=(3, 8), use_vmap=True)
    charges = {3: torch.tensor(0.0, **DD64), 8: torch.tensor(0.0, **DD64)}
    none = {3: None, 8: None}
    with pytest.raises(RuntimeError, match="custom domain validation") as exc:
        evaluate_lane_group(
            group, {3: pos_a, 8: pos_b}, charges, none, none, none
        )
    assert len(user_calls) == 1
    assert getattr(exc.value, "source_id", None) == 3


class _TupleCache(NamedTuple):
    k: Tensor
    weights: Tensor


class _SetupDataClassical(Classical):
    """Custom Classical returning configurable setup data."""

    k: Tensor = child()
    kind: str = context(default="node")

    @override
    def get_cache(self, numbers: Tensor, ihelp: IndexHelper | None = None):
        _ = ihelp
        weights = torch.ones(numbers.shape, device=numbers.device)  # float32
        if self.kind == "namedtuple":
            return _TupleCache(k=self.k, weights=weights.to(self.k.dtype))
        if self.kind == "dtype":
            return (weights,)
        if self.kind == "dataclass":
            return _PlainData(weights)
        return HarmonicRestraintCache(
            force_constant=self.k, atom_weights=weights.to(self.k.dtype)
        )

    @override
    def get_energy(self, positions: Tensor, cache: Any, **_: Any) -> Tensor:
        weights = cache[-1] if isinstance(cache, tuple) else cache.atom_weights
        return 0.5 * self.k * weights * positions.square().sum(dim=-1)


@dataclass
class _PlainData:
    weights: Tensor


@pytest.mark.parametrize("kind", ["node", "namedtuple", "dtype"])
def test_custom_setup_data_kinds_and_dtype_conversion(kind: str) -> None:
    """
    Node, NamedTuple and differently typed setup data evaluate and survive
    ``Calculator.type``.
    """
    numbers, pos = _lih_inputs()
    k = torch.tensor(0.05, **DD64)
    expected = (0.5 * k * pos.square().sum(dim=-1)).sum()

    calc = Calculator(
        numbers,
        GFN1_XTB,
        classical=(_SetupDataClassical(k=k, kind=kind),),
        opts={"verbosity": 0, "exclude": ["all"]},
        **DD64,
    )
    torch.testing.assert_close(calc.energy(pos), expected)

    calc32 = calc.type(torch.float32)
    energy32 = calc32.energy(pos.to(torch.float32))
    assert energy32.dtype == torch.float32
    torch.testing.assert_close(energy32, expected.to(torch.float32))


def test_custom_setup_data_of_unsupported_type_is_rejected() -> None:
    """Setup data that hides its tensors from transforms raises at setup."""
    numbers, _ = _lih_inputs()
    component = _SetupDataClassical(
        k=torch.tensor(0.05, **DD64), kind="dataclass", label="Plain"
    )
    with pytest.raises(TypeError, match="'Plain.get_cache' must return"):
        Calculator(
            numbers,
            GFN1_XTB,
            classical=(component,),
            opts={"verbosity": 0, "exclude": ["all"]},
            **DD64,
        )
