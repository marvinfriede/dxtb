# This file is part of dxtb.
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
"""
Model and system (B3): `Model.setup` builds everything that depends on the
atomic numbers, and gradients flow through it to the parameters.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from dxtb import GFN1_XTB, GFN2_XTB, Calculator, OutputHandler, ParamModule
from dxtb._src.calculators.model import Model, System
from dxtb._src.components.classicals import (
    ClassicalList,
    DispersionD3,
    Repulsion,
)
from dxtb._src.components.classicals.base import Classical
from dxtb._src.components.classicals.dispersion.factory import new_dispersion
from dxtb._src.components.classicals.repulsion.factory import new_repulsion
from dxtb._src.components.interactions.coulomb.secondorder import ES2
from dxtb._src.components.interactions.coulomb.thirdorder import ES3, ES3Cache
from dxtb._src.constants import labels
from dxtb._src.exlibs.available import has_libcint
from dxtb.components.coulomb import new_es2, new_es3
from dxtb.config import Config

from ..conftest import DEVICE

DD = {"device": DEVICE, "dtype": torch.double}


def test_setup() -> None:
    OutputHandler.verbosity = 0
    numbers = torch.tensor([3, 1], device=DEVICE)
    model = Model(par=ParamModule(GFN2_XTB, **DD))
    system = model.setup(numbers)

    assert isinstance(system, System)
    assert not hasattr(system, "model")
    assert not hasattr(system, "integrals")
    assert system.batch_mode == 0
    # the integral level follows the parametrization
    assert system.config.ints.level == labels.INTLEVEL_QUADRUPOLE
    assert model.config.ints.level != labels.INTLEVEL_QUADRUPOLE
    # the data of the classical terms is part of the system
    assert set(system.classical_cache) == set(
        c.label for c in system.classicals.components
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        system.batch_mode = 1  # type: ignore[misc]

    batch = model.setup(torch.tensor([[3, 1], [1, 1]], device=DEVICE))
    assert batch.batch_mode == 1


def test_single_system_setup_does_not_use_generic_classical_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build exact family setups without calling the generic list builder."""
    numbers = torch.tensor([3, 1], device=DEVICE)

    def fail_get_cache(*args, **kwargs):
        raise AssertionError("ClassicalList.get_cache must not run")

    monkeypatch.setattr(ClassicalList, "get_cache", fail_get_cache)
    system = Model(par=ParamModule(GFN1_XTB, **DD)).setup(numbers)

    assert type(system.repulsion_setup).__name__ == "RepulsionCoreSetup"
    assert len(system.d3_setups) == 1
    assert type(system.d3_setups[0][1]).__name__ == "DispersionD3CoreSetup"
    assert not system._classical_extension_cache
    assert system._packed_classical_cache is None
    assert set(system.classical_cache) == {
        component.label for component in system.classicals.components
    }
    assert system.classical_cache is not system.classical_cache
    assert system.classical_cache["Repulsion"] is system.repulsion_setup
    assert list(system.classical_cache) == [
        component.label for component in system.classicals.components
    ]
    first_view = system.classical_cache
    second_view = system.classical_cache
    first_view["Repulsion"] = object()
    assert first_view is not second_view
    assert second_view["Repulsion"] is system.repulsion_setup
    assert system.classical_cache["Repulsion"] is system.repulsion_setup


def test_empty_single_system_has_no_classical_cache_entries() -> None:
    """The compatibility view is empty when every classical is excluded."""
    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.zeros((2, 3), **DD)
    system = Model(
        par=ParamModule(GFN1_XTB, **DD),
        config=Config.create(exclude="all"),
    ).setup(numbers)

    assert system.classicals.components == []
    assert not system._classical_extension_cache
    assert not system.classical_cache
    result = system.singlepoint(positions)
    assert result.classical == ()
    torch.testing.assert_close(result.energy, torch.zeros_like(result.energy))


def test_classical_cache_view_includes_legacy_extension_data() -> None:
    """The view combines family Nodes with setup-time extension values."""

    class ClassicalExtension(Classical):
        __slots__ = ["scale"]

        def __init__(self) -> None:
            super().__init__(device=DEVICE, dtype=torch.double)
            self.scale = torch.tensor(0.125, device=DEVICE, dtype=torch.double)
            self.label = "ClassicalExtension"

        def get_cache(self, numbers, ihelp=None, **kwargs):
            return self.scale

        def get_energy(self, positions, cache, **kwargs):
            return torch.ones(positions.shape[-2], **DD) * cache

    numbers = torch.tensor([3, 1], device=DEVICE)
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], **DD)
    extension = ClassicalExtension()
    system = Model(
        par=ParamModule(GFN1_XTB, **DD),
        config=Config.create(exclude="scf"),
        classical=(extension,),
    ).setup(numbers)

    first = system.classical_cache
    second = system.classical_cache
    assert first is not second
    assert first[extension.label] is extension.scale
    assert second[extension.label] is extension.scale
    first[extension.label] = object()
    assert system.classical_cache[extension.label] is extension.scale

    charge = torch.tensor(0.0, **DD)
    compatibility = system.classicals.get_energy(
        positions, second, charge=charge
    )
    expected = dict(system.singlepoint(positions, chrg=charge).classical)
    assert compatibility.keys() == expected.keys()
    for label, energy in compatibility.items():
        torch.testing.assert_close(energy, expected[label])


def test_single_system_rejects_ambiguous_extension_labels() -> None:
    """Reject labels that would collapse distinct System contributions."""
    numbers = torch.tensor([3, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)

    exact_d3 = new_dispersion(
        numbers, par, charge=torch.tensor(0.0, **DD), **DD
    )
    assert type(exact_d3) is DispersionD3

    class D3Extension(DispersionD3):
        pass

    extension = D3Extension(
        exact_d3.numbers, exact_d3.param, dtype=torch.double, device=DEVICE
    )
    exact_d3.label = extension.label = "D3:shared"

    with pytest.raises(ValueError, match="D3:shared"):
        Model(
            par=par,
            config=Config.create(exclude="disp"),
            classical=(exact_d3, extension),
        ).setup(numbers)

    repulsion_a = new_repulsion(numbers.unique(), par, **DD)
    repulsion_b = new_repulsion(numbers.unique(), par, **DD)
    assert repulsion_a is not None and repulsion_b is not None
    repulsion_a.label = repulsion_b.label = "Repulsion:extension"
    with pytest.raises(ValueError, match="Repulsion:extension"):
        Model(
            par=par,
            config=Config.create(exclude="rep"),
            classical=(repulsion_a, repulsion_b),
        ).setup(numbers)

    class RepulsionExtension(Repulsion):
        pass

    repulsion_extension_a = RepulsionExtension(
        repulsion_a.arep,
        repulsion_a.zeff,
        repulsion_a.kexp,
        repulsion_a.klight,
        repulsion_a.cutoff,
        device=DEVICE,
        dtype=torch.double,
        en=repulsion_a.en,
        enscale=repulsion_a.enscale,
    )
    repulsion_extension_b = RepulsionExtension(
        repulsion_b.arep,
        repulsion_b.zeff,
        repulsion_b.kexp,
        repulsion_b.klight,
        repulsion_b.cutoff,
        device=DEVICE,
        dtype=torch.double,
        en=repulsion_b.en,
        enscale=repulsion_b.enscale,
    )
    repulsion_extension_a.label = repulsion_extension_b.label = (
        "Repulsion:two-subclasses"
    )
    with pytest.raises(ValueError, match="Repulsion:two-subclasses"):
        Model(
            par=par,
            config=Config.create(exclude="rep,disp"),
            classical=(repulsion_extension_a, repulsion_extension_b),
        ).setup(numbers)

    exact_d3_extension = D3Extension(
        exact_d3.numbers, exact_d3.param, dtype=torch.double, device=DEVICE
    )
    repulsion_extension_a.label = exact_d3_extension.label = (
        "Two-unrelated-extensions"
    )
    with pytest.raises(ValueError, match="Two-unrelated-extensions"):
        Model(
            par=par,
            config=Config.create(exclude="rep,disp"),
            classical=(repulsion_extension_a, exact_d3_extension),
        ).setup(numbers)

    d3_repulsion = new_repulsion(numbers.unique(), par, **DD)
    assert d3_repulsion is not None
    exact_d3.label = d3_repulsion.label = "same-custom-label"
    with pytest.raises(ValueError, match="same-custom-label"):
        Model(
            par=par,
            config=Config.create(exclude="rep,disp"),
            classical=(exact_d3, d3_repulsion),
        ).setup(numbers)


def test_single_system_d3_labels_preserve_order_and_identity() -> None:
    """Distinct D3 labels map to their captured Nodes in list order."""
    numbers = torch.tensor([3, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    first = new_dispersion(numbers, par, charge=torch.tensor(0.0, **DD), **DD)
    second = new_dispersion(numbers, par, charge=torch.tensor(0.0, **DD), **DD)
    assert type(first) is DispersionD3 and type(second) is DispersionD3
    first.label = "D3:first"
    second.label = "D3:second"

    system = Model(
        par=par,
        config=Config.create(exclude="disp"),
        classical=(first, second, first),
    ).setup(numbers)

    selected_d3 = [
        component
        for component in system.classicals.components
        if type(component) is DispersionD3
    ]
    assert selected_d3 == [first, second]
    assert list(system.classical_cache) == [
        component.label for component in system.classicals.components
    ]
    assert len(system.d3_setups) == 2
    view = system.classical_cache
    assert view["D3:first"] is system.d3_setups[0][1]
    assert view["D3:second"] is system.d3_setups[1][1]


def test_single_system_rejects_multiple_exact_single_setup_families() -> None:
    """One family-owned setup field cannot represent two exact instances."""
    numbers = torch.tensor([3, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    first = new_repulsion(numbers.unique(), par, **DD)
    second = new_repulsion(numbers.unique(), par, **DD)
    assert type(first) is Repulsion and type(second) is Repulsion
    first.label = "Repulsion:first"
    second.label = "Repulsion:second"

    with pytest.raises(ValueError, match="at most one exact Repulsion"):
        Model(
            par=par,
            config=Config.create(exclude="all"),
            classical=(first, second),
        ).setup(numbers)


def test_custom_ordinary_es2_replaces_excluded_builtin() -> None:
    """The explicit ES2 setup comes from the active custom ordinary term."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]],
        **DD,
    )
    par = ParamModule(GFN1_XTB, **DD)
    default = new_es2(torch.unique(numbers), par, dtype=torch.double)
    assert default is not None
    custom = ES2(
        default.hubbard,
        default.lhubbard,
        average=default.average,
        gexp=torch.tensor(1.4, **DD),
        shell_resolved=default.shell_resolved,
        dtype=torch.double,
    )
    model = Model(
        par=par,
        config=Config.create(exclude="es2"),
        interaction=(custom,),
    )
    system = model.setup(numbers)

    assert system.es2_interaction is custom
    assert system.es2_setup is not None
    torch.testing.assert_close(system.es2_setup.gexp, custom.gexp)
    assert torch.isfinite(system.singlepoint(positions).energy).all()


def test_es2_subclass_uses_legacy_extension_path() -> None:
    """A custom ES2 subclass is not intercepted by the built-in setup path."""

    class ES2Extension(ES2):
        __slots__ = ("cache_calls",)

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.cache_calls = 0

        def get_cache(self, **kwargs):
            self.cache_calls += 1
            return super().get_cache(**kwargs)

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]],
        **DD,
    )
    par = ParamModule(GFN1_XTB, **DD)
    original = new_es2(torch.unique(numbers), par, dtype=torch.double)
    assert original is not None
    custom = ES2Extension(
        original.hubbard,
        original.lhubbard,
        average=original.average,
        gexp=original.gexp,
        shell_resolved=original.shell_resolved,
        dtype=torch.double,
    )
    custom.label = "ES2"
    model = Model(
        par=par,
        config=Config.create(exclude="es2"),
        interaction=(custom,),
    )
    system = model.setup(numbers)

    assert system.es2_setup is None
    assert system.es2_interaction is None
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
    assert custom.cache_calls == 1


def test_es2_subclass_keeps_legacy_update_and_reset() -> None:
    """ES2 subclasses retain the generic Component mutation behavior."""

    class ES2Extension(ES2):
        __slots__ = ("extension_value",)

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.extension_value = torch.tensor(1.0, **DD)

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    original = new_es2(torch.unique(numbers), par, dtype=torch.double)
    assert original is not None
    custom = ES2Extension(
        original.hubbard,
        original.lhubbard,
        average=original.average,
        gexp=original.gexp,
        shell_resolved=original.shell_resolved,
        dtype=torch.double,
    )
    custom.label = "ES2"
    system = Model(
        par=par,
        config=Config.create(exclude="es2"),
        interaction=(custom,),
    ).setup(numbers)

    system.interactions.update("ES2", extension_value=torch.tensor(2.0, **DD))
    assert custom.extension_value.item() == 2.0
    system.interactions.reset("ES2")
    assert custom.extension_value.item() == 2.0
    system.interactions.reset_all()
    assert custom.extension_value.item() == 2.0


def test_es3_custom_exact_replaces_excluded_builtin() -> None:
    """An active exact custom ES3 supplies the System setup values."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]],
        **DD,
    )
    par = ParamModule(GFN1_XTB, **DD)
    reference_system = Model(par=par).setup(numbers)
    reference_result = reference_system.singlepoint(positions)
    custom = new_es3(torch.unique(numbers), par, dtype=torch.double)
    assert custom is not None
    custom.hubbard_derivs = custom.hubbard_derivs * 2.0
    system = Model(
        par=par,
        config=Config.create(exclude="es3"),
        interaction=(custom,),
    ).setup(numbers)

    assert system.es3_interaction is custom
    assert system.es3_setup is not None
    expected = system.ihelp.spread_uspecies_to_atom(custom.hubbard_derivs)
    torch.testing.assert_close(system.es3_setup.hubbard_derivs, expected)
    result = system.singlepoint(positions)
    assert torch.isfinite(result.energy).all()
    assert not torch.isclose(result.energy.sum(), reference_result.energy.sum())


def test_es3_subclass_keeps_legacy_extension_and_mutation() -> None:
    """An ES3 subclass uses its cache hook and legacy mutation methods."""

    class ES3Extension(ES3):
        __slots__ = ("extension_scale", "cache_calls", "reset_calls")

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.extension_scale = torch.tensor(1.0, **DD)
            self.cache_calls = 0
            self.reset_calls = 0

        def get_cache(self, **kwargs):
            self.cache_calls += 1
            cache = super().get_cache(**kwargs)
            return ES3Cache(
                cache.hd * self.extension_scale,
                shell_resolved=cache.shell_resolved,
            )

        def reset(self) -> None:
            self.reset_calls += 1
            super().reset()

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]],
        **DD,
    )
    par = ParamModule(GFN1_XTB, **DD)
    original = new_es3(torch.unique(numbers), par, dtype=torch.double)
    assert original is not None
    custom = ES3Extension(
        original.hubbard_derivs,
        shell_scale=original.shell_scale,
        dtype=torch.double,
    )
    custom.label = "ES3"
    calc = Calculator(
        numbers,
        par,
        interaction=(custom,),
        opts={"verbosity": 0, "exclude": "es3"},
        dtype=torch.double,
    )

    assert calc.system.es3_setup is None
    first = calc.singlepoint(positions)
    assert first.energy.isfinite().all()
    assert custom.cache_calls == 1
    calc.interactions.update("ES3", extension_scale=torch.tensor(2.0, **DD))
    assert custom.extension_scale.item() == 2.0
    second = calc.singlepoint(positions.clone())
    assert custom.cache_calls == 2
    assert not torch.isclose(first.energy.sum(), second.energy.sum())
    calc.interactions.reset("ES3")
    assert custom.reset_calls == 1
    calc.reset()
    assert custom.reset_calls == 2
    assert custom.extension_scale.item() == 2.0


def test_exact_es2_es3_mutations_are_rejected() -> None:
    """Only exact migrated ES2/ES3 objects reject update and reset."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    calc = Calculator(numbers, par, dtype=torch.double, opts={"verbosity": 0})
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.interactions.update("ES2", gexp=torch.tensor(3.0, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.interactions.reset("ES2")
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.interactions.update("ES3", hubbard_derivs=torch.ones(1, **DD))
    with pytest.raises(RuntimeError, match="setup-derived"):
        calc.interactions.reset("ES3")


def test_es2_setup_does_not_alias_legacy_gexp() -> None:
    """In-place writes through the exact legacy ES2 object cannot alter setup."""
    from dxtb._src.components.interactions.coulomb.secondorder import (
        build_es2_coulomb,
    )

    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]], **DD
    )
    system = Model(par=ParamModule(GFN1_XTB, **DD)).setup(numbers)
    assert system.es2_setup is not None
    assert system.es2_interaction is not None
    setup_gexp = system.es2_setup.gexp.clone()
    matrix = build_es2_coulomb(system.es2_setup, positions)

    with torch.no_grad():
        system.es2_interaction.gexp.add_(0.5)

    torch.testing.assert_close(system.es2_setup.gexp, setup_gexp)
    torch.testing.assert_close(
        build_es2_coulomb(system.es2_setup, positions.clone()), matrix
    )


@pytest.mark.parametrize("label", ["ES2", "ES3"])
def test_duplicate_migrated_interaction_labels_rejected(label: str) -> None:
    """Built-in plus additional ES2/ES3 labels are rejected at setup."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    if label == "ES2":
        interaction1 = new_es2(torch.unique(numbers), par, dtype=torch.double)
        interaction2 = new_es2(torch.unique(numbers), par, dtype=torch.double)
    else:
        interaction1 = new_es3(torch.unique(numbers), par, dtype=torch.double)
        interaction2 = new_es3(torch.unique(numbers), par, dtype=torch.double)
    assert interaction1 is not None and interaction2 is not None
    with pytest.raises(
        ValueError, match="Duplicate migrated interaction labels"
    ):
        Model(par=par, interaction=(interaction1,)).setup(numbers)


def test_calculator_reset_preserves_es2_es3_setup() -> None:
    """Legacy reset leaves migrated interaction setup and its values intact."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-1.4, 0.0, 0.0]],
        **DD,
    )
    par = ParamModule(GFN1_XTB, **DD)
    params = dict(par.named_parameters())
    gam = params["parameter_tree.element.O.gam.param"]
    gam3 = params["parameter_tree.element.O.gam3.param"]
    gam.requires_grad_(True)
    gam3.requires_grad_(True)
    calc = Calculator(numbers, par, dtype=torch.double, opts={"verbosity": 0})
    es2_setup = calc.system.es2_setup
    es3_setup = calc.system.es3_setup
    assert es2_setup is not None and es3_setup is not None

    first = calc.singlepoint(positions)
    es2_parameters = (
        es2_setup.atom_hubbard,
        es2_setup.shell_hubbard,
        es2_setup.gexp,
    )
    es3_parameters = es3_setup.hubbard_derivs
    calc.reset()
    second = calc.singlepoint(positions.clone())

    assert calc.system.es2_setup is es2_setup
    assert calc.system.es3_setup is es3_setup
    assert es2_setup.atom_hubbard is es2_parameters[0]
    assert es2_setup.shell_hubbard is es2_parameters[1]
    assert es2_setup.gexp is es2_parameters[2]
    assert es3_setup.hubbard_derivs is es3_parameters
    torch.testing.assert_close(first.energy, second.energy)
    es2_hubbard = (
        es2_setup.atom_hubbard
        if es2_setup.atom_hubbard is not None
        else es2_setup.shell_hubbard
    )
    assert es2_hubbard is not None
    setup_scalar = (
        es2_hubbard.sum() + es2_setup.gexp + es3_setup.hubbard_derivs.sum()
    )
    grad_gam, grad_gam3 = torch.autograd.grad(setup_scalar, (gam, gam3))
    assert torch.isfinite(grad_gam).all()
    assert torch.isfinite(grad_gam3).all()
    assert torch.count_nonzero(grad_gam) > 0
    assert torch.count_nonzero(grad_gam3) > 0


def test_calculator_type_does_not_convert_es2_es3_setup() -> None:
    """Post-type evaluation is unsupported for setup-derived ES2/ES3 data."""
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    calc = Calculator(
        numbers, GFN1_XTB, dtype=torch.double, opts={"verbosity": 0}
    )
    assert calc.system.es2_setup is not None
    assert calc.system.es3_setup is not None

    calc.type(torch.float32)

    assert calc.system.es2_setup.gexp.dtype == torch.float64
    assert calc.system.es3_setup.hubbard_derivs.dtype == torch.float64


def test_system_does_not_retain_mutable_integral_layer() -> None:
    """System retains setup values, not mutable integral builders."""
    from dxtb._src.integral.base import BaseIntegral
    from dxtb._src.integral.container import Integrals
    from dxtb._src.integral.driver import DriverManager
    from dxtb._src.xtb.base import BaseHamiltonian

    system = Model(par=ParamModule(GFN2_XTB, **DD)).setup(
        torch.tensor([3, 1], device=DEVICE)
    )
    forbidden = (Integrals, DriverManager, BaseHamiltonian, BaseIntegral)
    for field in dataclasses.fields(system):
        assert not isinstance(getattr(system, field.name), forbidden)


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param("pytorch", id="pytorch"),
        pytest.param(
            "libcint",
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_system_integral_setup_is_narrow(driver: str) -> None:
    """System's integral setup retains values, not model or adapter objects."""
    from dxtb._src.basis.bas import BasisSetup
    from dxtb._src.integral.base import BaseIntegral
    from dxtb._src.integral.container import Integrals
    from dxtb._src.integral.driver import DriverManager
    from dxtb._src.integral.evaluation import IntegralSetup
    from dxtb._src.param import ParamModule
    from dxtb._src.xtb.base import BaseHamiltonian

    calc = Calculator(
        torch.tensor([1, 1], device=DEVICE),
        GFN2_XTB,
        dtype=torch.double,
        opts={"verbosity": 0, "int_driver": driver},
    )
    system = calc.system
    assert not hasattr(system, "model")
    assert not hasattr(system, "integrals")
    setup = system.integral_setup
    assert isinstance(setup, IntegralSetup)
    assert not isinstance(setup, (ParamModule, DriverManager))
    assert not hasattr(setup, "par")
    legacy_types = (Integrals, DriverManager, BaseHamiltonian, BaseIntegral)
    assert not isinstance(setup.pytorch, legacy_types)
    assert not isinstance(setup.libcint, legacy_types)
    if setup.libcint is not None:
        for basis_setup in setup.libcint.basis_setups:
            assert isinstance(basis_setup, BasisSetup)
            assert not hasattr(basis_setup, "par")
            assert not hasattr(basis_setup, "basis")
            assert not hasattr(basis_setup, "driver")
            assert {
                field.name for field in dataclasses.fields(basis_setup)
            } == {
                "numbers",
                "unique",
                "ihelp",
                "ngauss",
                "pqn",
                "slater",
                "valence",
            }
    if setup.pytorch is not None:
        assert not hasattr(setup.pytorch, "par")
        assert not hasattr(setup.pytorch, "driver")


def test_setup_is_differentiable() -> None:
    """The element parameters are gathered in `setup`."""
    OutputHandler.verbosity = 0
    numbers = torch.tensor([3, 1], device=DEVICE)
    par = ParamModule(GFN1_XTB, **DD)
    arep = dict(par.named_parameters())["parameter_tree.element.H.arep.param"]
    arep.requires_grad_(True)

    system = Model(par=par).setup(numbers)
    (grad,) = torch.autograd.grad(
        system.classical_cache["Repulsion"].arep.sum(), arep
    )
    assert grad.abs().sum() > 0


def test_calculator_holds_the_system() -> None:
    calc = Calculator(
        torch.tensor([3, 1], device=DEVICE), GFN1_XTB, opts={"verbosity": 0}
    )
    assert calc.ihelp is calc.system.ihelp
    assert calc.classicals is calc.system.classicals
    assert calc.opts is calc.system.config
