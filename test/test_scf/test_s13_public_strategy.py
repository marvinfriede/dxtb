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
"""Public strategy configuration and compatibility-boundary tests."""

from __future__ import annotations

from test.test_baseline.molecules import SYSTEMS

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels, scf
from dxtb._src.calculators.model import Model
from dxtb.config import Config, ConfigSCF


def test_default_strategy_is_public_anderson_potential() -> None:
    """Default Config and Config.create resolve the same public preset."""
    for config in (ConfigSCF(), Config.create().scf):
        stage = config.strategy.stages[0]
        assert isinstance(stage.method, scf.AndersonMethod)
        assert stage.representation is scf.SCFRepresentation.POTENTIAL
        assert stage.method.settings.generations == 5
        assert stage.method.settings.diagonal_offset == 0.01
        assert config.differentiation is scf.SCFDifferentiation.UNROLLED
        assert config.response.representation is scf.SCFRepresentation.POTENTIAL
        assert config.response.rtol == 1.0e-10
        assert config.response.atol == 1.0e-12
        assert config.response.maxiter is None


def test_named_presets_have_stable_method_and_representation() -> None:
    """Named presets expose methods and coordinate choices without labels."""
    expected = {
        "linear": (scf.LinearMethod, scf.SCFRepresentation.POTENTIAL),
        "anderson": (scf.AndersonMethod, scf.SCFRepresentation.POTENTIAL),
        "broyden": (scf.BroydenMethod, scf.SCFRepresentation.POTENTIAL),
        "cdiis": (scf.CDIISMethod, scf.SCFRepresentation.FOCK),
    }
    for name, (method_type, representation) in expected.items():
        config = ConfigSCF.create(strategy=name)
        stage = config.strategy.stages[0]
        assert isinstance(stage.method, method_type)
        assert stage.representation is representation
        assert name in str(config.info()).casefold()


def test_legacy_and_new_scf_authorities_cannot_be_mixed() -> None:
    """Compatibility arguments translate only when new authorities are omitted."""
    with pytest.raises(ValueError, match="differentiation or legacy scf_mode"):
        Config.create(differentiation="implicit", scf_mode="full")
    with pytest.raises(ValueError, match="differentiation or legacy scf_mode"):
        Config.create(differentiation="unrolled", scf_mode="implicit")
    with pytest.raises(ValueError, match="strategy or mixer"):
        Config.create(strategy="anderson", mixer="broyden")
    with pytest.raises(ValueError, match="strategy or legacy scp_mode"):
        Config.create(strategy="anderson", scp_mode="potential")
    with pytest.raises(ValueError, match="strategy or legacy scp_mode"):
        Config.create(
            strategy=scf.strategy_preset("anderson"), scp_mode="potential"
        )
    with pytest.raises(ValueError, match="Legacy Anderson options"):
        Config.create(strategy="anderson", damp_generations=8)
    with pytest.raises(ValueError, match="differentiation or legacy scf_mode"):
        Config.from_dict({"differentiation": "implicit", "scf_mode": "full"})

    assert (
        Config.create().scf.differentiation is scf.SCFDifferentiation.UNROLLED
    )
    assert (
        Config.create(scf_mode="implicit").scf.differentiation
        is scf.SCFDifferentiation.IMPLICIT
    )
    legacy_broyden = Config.create(mixer="broyden", scp_mode="charge")
    assert isinstance(
        legacy_broyden.scf.strategy.stages[0].method, scf.BroydenMethod
    )
    assert (
        legacy_broyden.scf.strategy.stages[0].representation
        is scf.SCFRepresentation.CHARGE
    )
    legacy_anderson = Config.create(damp_generations=8)
    assert (
        legacy_anderson.scf.strategy.stages[0].method.settings.generations == 8
    )


def test_soscf_internal_safety_settings_are_not_public() -> None:
    """SOSCF exposes its validated method, not epsilon-scale guard knobs."""
    assert "SOSCFMethod" in scf.__all__
    assert "SOSCFMethodSettings" not in scf.__all__
    assert isinstance(
        scf.strategy_preset("cdiis-soscf").stages[-1].method,
        scf.SOSCFMethod,
    )


def test_dynamic_and_soscf_presets_require_implicit_eager_policy() -> None:
    """Eager-only stages reject unrolled configuration before SCF setup."""
    with pytest.raises(ValueError, match="requires eager/implicit"):
        ConfigSCF.create(strategy="broyden-cdiis")
    with pytest.raises(ValueError, match="requires eager/implicit"):
        ConfigSCF.create(strategy="cdiis-soscf", fermi_etemp=0.0)

    strategy = ConfigSCF.create(
        strategy="cdiis-soscf",
        differentiation="implicit",
        fermi_etemp=0.0,
    )
    assert isinstance(strategy.strategy.stages[-1].method, scf.SOSCFMethod)


def test_soscf_preset_rejects_active_fermi_smearing() -> None:
    """SOSCF eligibility rejects active smearing rather than freezing it."""
    with pytest.raises(ValueError, match="active Fermi smearing"):
        ConfigSCF.create(
            strategy="cdiis-soscf",
            differentiation="implicit",
            fermi_etemp=300.0,
        )


def test_retired_f_atol_is_not_accepted_as_convergence_configuration() -> None:
    """The old rootfinder tolerance is explicitly retired at the parser edge."""
    with pytest.raises(ValueError, match="f_atol is retired"):
        Config.create(f_atol=1.0e-8)


def test_default_system_singlepoint_is_vmap_and_reverse_mode_compatible() -> (
    None
):
    """The public default solves, maps, and differentiates without SCF options."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    ).setup(torch.tensor([1, 1]))

    result = system.singlepoint(positions)
    assert bool(result.converged)
    gradient = torch.autograd.grad(result.energy.sum(), positions)[0]
    assert torch.isfinite(gradient).all()

    geometries = torch.stack((positions.detach(), 1.02 * positions.detach()))
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(geometries)
    serial = [system.singlepoint(geometry) for geometry in geometries]
    torch.testing.assert_close(
        mapped.energy, torch.stack([item.energy for item in serial])
    )
    torch.testing.assert_close(
        mapped.charges.mono,
        torch.stack([item.charges.mono for item in serial]),
    )


def test_default_and_broyden_reach_the_same_tight_physical_root() -> None:
    """The new default changes the trajectory, not the certified root."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    results = []
    for strategy in ("anderson", "broyden"):
        config = Config.create(
            method="GFN1-xTB",
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_HCORE,
            strategy=strategy,
            convergence="tight",
            fermi_etemp=0.0,
        )
        system = Model(
            par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
        ).setup(numbers)
        result = system.singlepoint(positions)
        assert bool(result.converged)
        results.append(result)

    for name in ("energy", "fenergy", "density", "hamiltonian", "occupation"):
        torch.testing.assert_close(
            getattr(results[0], name),
            getattr(results[1], name),
            atol=2.0e-9,
            rtol=2.0e-9,
        )
    torch.testing.assert_close(
        results[0].charges.mono,
        results[1].charges.mono,
        atol=2.0e-9,
        rtol=2.0e-9,
    )


def test_default_float32_h2o_root_and_gradient_are_finite() -> None:
    """The standard default resolves SCF tolerances safely in float32."""
    from test.test_baseline.molecules import SYSTEMS

    source = SYSTEMS["H2O"]
    numbers = source.numbers.detach().cpu()
    positions = (
        source.positions.detach().cpu().to(torch.float32).requires_grad_()
    )
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float32), config=config
    ).setup(numbers)
    result = system.singlepoint(positions)
    gradient = torch.autograd.grad(result.energy.sum(), positions)[0]
    assert bool(result.converged)
    assert torch.isfinite(result.energy).all()
    assert torch.isfinite(gradient).all()


def test_default_gfn2_h2o_root_and_gradient_are_finite() -> None:
    """The no-option GFN2 public route uses the same strategy contract."""
    source = SYSTEMS["H2O"]
    numbers = source.numbers.detach().cpu()
    positions = (
        source.positions.detach().cpu().to(torch.float64).requires_grad_()
    )
    config = Config.create(
        method="GFN2-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_DIPOLE,
    )
    system = Model(
        par=ParamModule(GFN2_XTB, dtype=positions.dtype), config=config
    ).setup(numbers)
    result = system.singlepoint(positions)
    gradient = torch.autograd.grad(result.energy.sum(), positions)[0]
    assert bool(result.converged)
    assert torch.isfinite(result.energy).all()
    assert torch.isfinite(gradient).all()
