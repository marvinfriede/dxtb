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
"""B3b final-layout ownership and transform boundary probes."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
from tad_mctc.tree import child
from tad_mctc.exceptions import DeviceError, DtypeError
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils._python_dispatch import TorchDispatchMode

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, labels
from dxtb._src.basis.indexhelper import IndexHelper
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data, singlepoint
from dxtb._src.integral.driver.pytorch import build_overlap
from dxtb._src.param import ParamModule
from dxtb.config import Config

from ..molecules import mols
from ..utils import scf_strategy


class _MapCloneCounter(TorchDispatchMode):
    def __init__(self, maps: tuple[torch.Tensor, ...]) -> None:
        self.maps = maps
        self.count = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if (
            func == torch.ops.aten.clone.default
            and args
            and any(args[0] is value for value in self.maps)
        ):
            self.count += 1
        return func(*args, **(kwargs or {}))


def _model(dtype: torch.dtype = torch.float64) -> Model:
    config = Config.create(
        int_driver=labels.INTDRIVER_PYTORCH,
        maxiter=6,
        fermi_etemp=0.0,
    )
    return Model(par=ParamModule(GFN1_XTB, dtype=dtype), config=config)


def test_single_and_packed_modes_keep_separate_ownership() -> None:
    """Only exact single Systems own core setup Nodes and a derived view."""
    numbers = torch.tensor([1, 1])
    model = _model()

    single = model.setup(numbers, batch_mode=0)
    assert single.batch_mode == 0
    assert not hasattr(single, "_packed_classical_cache")
    assert not hasattr(single, "_exact_compatibility")
    assert single.repulsion_setup is not None
    assert type(single.repulsion_setup).__name__ == "RepulsionSetup"
    assert single.classical_cache is not single.classical_cache
    assert single.classical_cache["Repulsion"] is single.repulsion_setup
    assert single.exact_core is not None
    assert single.exact_core.numbers is single.numbers
    assert single.exact_core.config is single.config
    assert single.exact_core.effective_dtype is torch.float64
    assert single.exact_core.integral is not None
    assert single.exact_core.integral.backend is (
        single.integral_setup.pytorch
        if single.integral_setup.pytorch is not None
        else single.integral_setup.libcint
    )
    assert single.exact_core.h0_setup is single.h0_setup
    assert single.exact_core.index is single.ihelp
    assert not hasattr(single.exact_core.integral, "index")
    assert not hasattr(single.exact_core.integral.backend, "index")
    assert "index" in single.exact_core._child_names
    assert not hasattr(single.exact_core, "batch_mode")
    assert "dd" not in single.exact_core._child_names
    assert "dd" not in single.exact_core._context_names
    assert not hasattr(single.exact_core, "ihelp")
    assert not hasattr(single.exact_core, "classicals")
    assert not hasattr(single.exact_core, "interactions")
    assert [term.label for term in single.exact_core.classical_terms] == [
        component.label
        for component in single.classicals.components
        if type(component).__name__
        in {
            "DispersionD3",
            "Repulsion",
            "ShortRangeBond",
            "DispersionD4",
            "IES",
            "Halogen",
        }
    ]
    for component in single.classicals.components:
        setup = single.classical_setup(component)
        if setup is not None:
            assert any(
                term.setup is setup
                for term in single.exact_core.classical_terms
            )
    assert [term.label for term in single.exact_core.interaction_terms] == [
        component.label
        for component in single.interactions.components
        if single.interaction_setup(component) is not None
    ]
    assert single.interaction_setup(single.es2_interaction) is single.es2_setup
    assert single.interaction_setup(single.es3_interaction) is single.es3_setup
    assert (
        single.interaction_setup(single.d4sc_interaction) is single.d4sc_setup
    )

    packed_numbers = torch.tensor([[1, 1], [1, 1]])
    for mode in (1, 2):
        packed = model.setup(packed_numbers, batch_mode=mode)
        assert packed.batch_mode == mode
        assert packed.exact_core is None
        assert not hasattr(packed, "_packed_classical_cache")
        assert packed.classical_cache is not packed.classical_cache
        assert type(packed.repulsion_setup).__name__ == "RepulsionSetup"
        assert packed.integral_setup is None


@pytest.mark.parametrize(
    ("method", "method_label"),
    [
        (GFN0_XTB, labels.GFN0_XTB),
        (GFN1_XTB, labels.GFN1_XTB),
        (GFN2_XTB, labels.GFN2_XTB),
    ],
    ids=["gfn0", "gfn1", "gfn2"],
)
def test_classical_descriptors_drive_exact_evaluation(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    method_label: int,
) -> None:
    """Captured ordered descriptors evaluate built-ins without component hooks."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    model = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(
            method=method_label,
            int_driver=labels.INTDRIVER_PYTORCH,
            exclude="scf",
        ),
    )
    system = model.setup(numbers)
    core = system.exact_core
    assert core is not None
    assert system._active_exact_core() is core
    assert [term.position for term in core.classical_terms] == sorted(
        term.position for term in core.classical_terms
    )
    assert [term.label for term in core.classical_terms] == [
        component.label
        for component in system.classicals.components
        if system.classical_setup(component) is not None
    ]
    assert all(
        term.setup is system.classical_setup(component)
        for term, component in zip(
            core.classical_terms,
            (
                c
                for c in system.classicals.components
                if system.classical_setup(c) is not None
            ),
        )
    )
    assert all(not hasattr(term, "component") for term in core.classical_terms)

    charge = torch.zeros(1, dtype=positions.dtype, device=positions.device)
    # Establish the compatibility result through the legacy component APIs,
    # independently of the descriptor-driven System evaluator below.
    legacy = system.classicals.get_energy(
        positions, system.classical_cache, charge=charge
    )
    for position, component in enumerate(system.classicals.components):
        if core.classical_term_at(position) is not None:
            monkeypatch.setattr(
                type(component),
                "get_energy",
                lambda *args, **kwargs: pytest.fail(
                    "exact classical descriptor called component.get_energy"
                ),
            )
    actual = system.singlepoint(positions, chrg=charge)
    assert [label for label, _ in actual.classical] == [
        label for label in legacy
    ]
    for label, energy in actual.classical:
        torch.testing.assert_close(energy, legacy[label])


@pytest.mark.parametrize(
    ("method", "method_label"),
    [(GFN1_XTB, labels.GFN1_XTB), (GFN2_XTB, labels.GFN2_XTB)],
    ids=["gfn1", "gfn2"],
)
def test_interaction_descriptors_feed_functional_preparation(
    method: str, method_label: int
) -> None:
    """Exact setup descriptors supply the existing functional entry schema."""
    from dxtb._src.scf.interactions import (
        prepare_scf_interactions,
    )

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(
            method=method_label,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    ).setup(numbers)
    core = system.exact_core
    assert core is not None
    assert system._active_exact_core() is core
    assert [term.position for term in core.interaction_terms] == sorted(
        term.position for term in core.interaction_terms
    )
    assert [term.label for term in core.interaction_terms] == [
        component.label
        for component in system.interactions.components
        if system.interaction_setup(component) is not None
    ]
    assert all(
        term.setup is system.interaction_setup(component)
        for term, component in zip(
            core.interaction_terms,
            (
                c
                for c in system.interactions.components
                if system.interaction_setup(c) is not None
            ),
        )
    )
    data = _interaction_data(system, positions)
    assert data.scf_kinds == {
        term.label: term.kind for term in core.interaction_terms
    }
    prepared = prepare_scf_interactions(
        system.interactions,
        data,
        numbers,
        kind_by_label=data.scf_kinds,
    )
    assert tuple(entry.kind for entry in prepared.entries) == tuple(
        term.kind for term in core.interaction_terms
    )


def test_additional_classical_keeps_global_result_order() -> None:
    """Descriptor-built terms retain the Classical list order."""
    from dxtb._src.components.classicals import DispersionD3
    from dxtb._src.components.classicals.dispersion import new_dispersion

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    extra = new_dispersion(numbers, par, dtype=torch.float64)
    assert type(extra) is DispersionD3
    extra = extra.replace(label="D3:extra")
    system = Model(
        par=par,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            exclude="scf",
        ),
        classical=(extra,),
    ).setup(numbers)
    assert system.classicals.components[-1] is extra
    assert system.exact_core is not None
    assert system._active_exact_core() is system.exact_core
    actual = system.singlepoint(positions)
    expected = replace(system, exact_core=None).singlepoint(positions)
    assert [label for label, _ in actual.classical] == [
        component.label for component in system.classicals.components
    ]
    assert [label for label, _ in actual.classical] == [
        label for label, _ in expected.classical
    ]
    for (label, energy), (expected_label, expected_energy) in zip(
        actual.classical, expected.classical
    ):
        assert label == expected_label
        torch.testing.assert_close(energy, expected_energy)


def test_mixed_classical_extension_keeps_global_result_order() -> None:
    """Descriptor-built terms and custom Classical subclasses retain list ordering."""
    from dxtb._src.components.classicals import Classical

    class ClassicalExtension(Classical):
        scale: torch.Tensor = child(
            default_factory=lambda: torch.tensor(0.125, dtype=torch.float64)
        )

        def get_cache(self, numbers, ihelp=None, **kwargs):
            _ = (numbers, ihelp, kwargs)
            return self.scale

        def get_energy(self, positions, cache, **kwargs):
            _ = kwargs
            return (
                torch.ones(positions.shape[-2], dtype=positions.dtype) * cache
            )

    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    extension = ClassicalExtension()
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            exclude="scf",
        ),
        classical=(extension,),
    ).setup(numbers)
    assert system.exact_core is not None
    assert system._active_exact_core() is system.exact_core
    actual = system.singlepoint(positions)
    expected = replace(system, exact_core=None).singlepoint(positions)
    assert [label for label, _ in actual.classical] == [
        component.label for component in system.classicals.components
    ]
    for (label, energy), (expected_label, expected_energy) in zip(
        actual.classical, expected.classical
    ):
        assert label == expected_label
        torch.testing.assert_close(energy, expected_energy)


@pytest.mark.parametrize(
    "method,method_label",
    [
        (GFN0_XTB, labels.GFN0_XTB),
        (GFN1_XTB, labels.GFN1_XTB),
        (GFN2_XTB, labels.GFN2_XTB),
    ],
    ids=["gfn0", "gfn1", "gfn2"],
)
def test_effective_dtype_fallback_preserves_explicit_setup_override(
    method: str, method_label: int
) -> None:
    """No-integral setups still validate against the effective setup dtype."""
    config = Config.create(
        method=method_label,
        exclude=("all",),
        int_level=0,
    )
    system = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=config,
        auto_int_level=False,
    ).setup(
        torch.tensor([1, 1]),
        dd={"device": torch.device("cpu"), "dtype": torch.float32},
    )
    assert system.h0_setup is None
    assert system.integral_setup is None
    assert system.exact_core is not None
    assert system.exact_core.effective_dtype is torch.float32
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float32
    )
    result = system.singlepoint(positions)
    assert result.energy.dtype == torch.float32
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        system.singlepoint(positions.to(torch.float64))
    original_dtype = system.dd["dtype"]
    try:
        system.dd["dtype"] = torch.float64
        assert system.singlepoint(positions).energy.dtype is torch.float32
        with pytest.raises(DtypeError, match="Dtype mismatch"):
            system.singlepoint(positions.to(torch.float64))
    finally:
        system.dd["dtype"] = original_dtype


def test_exact_core_deactivates_when_family_setup_is_replaced() -> None:
    """A façade setup replacement cannot leave the old core authoritative."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = _model().setup(torch.tensor([1, 1]))
    assert system.exact_core is not None
    assert system.repulsion_setup is not None
    original = system.repulsion_setup
    replacement = replace(original, arep=original.arep * 1.7)
    changed = replace(system, repulsion_setup=replacement)

    assert system._active_exact_core() is system.exact_core
    assert replace(system, repulsion_setup=None)._active_exact_core() is None
    assert changed._active_exact_core() is None

    baseline = system.singlepoint(positions)
    changed_result = changed.singlepoint(positions)
    expected = replace(changed, exact_core=None).singlepoint(positions)
    torch.testing.assert_close(changed_result.energy, expected.energy)
    torch.testing.assert_close(
        dict(changed_result.classical)["Repulsion"],
        dict(expected.classical)["Repulsion"],
    )
    assert not torch.allclose(
        dict(baseline.classical)["Repulsion"],
        dict(changed_result.classical)["Repulsion"],
    )


@pytest.mark.parametrize(
    ("method", "method_label", "field"),
    [
        (GFN1_XTB, labels.GFN1_XTB, "repulsion_setup"),
        (GFN0_XTB, labels.GFN0_XTB, "srb_setup"),
        (GFN2_XTB, labels.GFN2_XTB, "d4_setup"),
        (GFN0_XTB, labels.GFN0_XTB, "ies_setup"),
        (GFN1_XTB, labels.GFN1_XTB, "halogen_setup"),
    ],
    ids=["repulsion", "srb", "d4", "ies", "halogen"],
)
def test_exact_core_checks_classical_setup_coherency(
    method: str, method_label: int, field: str
) -> None:
    """Every migrated classical facade setup is checked by identity."""
    system = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(method=method_label),
    ).setup(torch.tensor([1, 1]))
    assert system.config.method == method_label
    assert system.exact_core is not None
    setup = getattr(system, field)
    if setup is None:
        pytest.fail(f"{method_label} fixture has no {field}")
    assert system._active_exact_core() is system.exact_core
    assert replace(system, **{field: None})._active_exact_core() is None


@pytest.mark.parametrize(
    "field", ["es2_setup", "es3_setup", "aes2_setup", "d4sc_setup"]
)
def test_exact_core_checks_interaction_setup_coherency(field: str) -> None:
    """Each captured interaction setup remains coherent with its façade."""
    method = GFN1_XTB if field in ("es2_setup", "es3_setup") else GFN2_XTB
    method_label = labels.GFN1_XTB if method is GFN1_XTB else labels.GFN2_XTB
    system = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(method=method_label),
    ).setup(torch.tensor([1, 1]))
    assert system.config.method == method_label
    assert system.exact_core is not None
    setup = getattr(system, field)
    assert setup is not None, f"default configuration has no {field}"
    assert system._active_exact_core() is system.exact_core
    assert replace(system, **{field: None})._active_exact_core() is None


def test_exact_core_checks_generalized_born_setup_coherency() -> None:
    """GeneralizedBorn setup remains tied to its exact interaction object."""
    from dxtb._src.components.interactions.solvation.alpb import GeneralizedBorn

    numbers = torch.tensor([1, 1])
    gb = GeneralizedBorn.create(
        numbers, 78.9, device="cpu", dtype=torch.float64
    )
    system = Model(
        par=ParamModule(GFN2_XTB, dtype=torch.float64),
        config=Config.create(method=labels.GFN2_XTB),
        interaction=(gb,),
    ).setup(numbers)
    assert system.generalized_born_setup is not None
    assert system._active_exact_core() is system.exact_core
    assert (
        replace(system, generalized_born_setup=None)._active_exact_core()
        is None
    )


def test_exact_core_checks_ordered_d3_setup_coherency() -> None:
    """D3 order and interaction ownership remain tied to the façade."""
    from dxtb._src.components.classicals.dispersion.d3 import DispersionD3

    numbers = torch.tensor([1, 1])
    parameters = {
        "s6": torch.tensor(1.0, dtype=torch.float64),
        "a1": torch.tensor(0.5, dtype=torch.float64),
        "a2": torch.tensor(5.7, dtype=torch.float64),
    }
    first = DispersionD3(numbers=numbers, param=parameters).to(
        dtype=torch.float64
    )
    first = first.replace(label="D3:first")
    second = DispersionD3(numbers=numbers, param=parameters).to(
        dtype=torch.float64
    )
    second = second.replace(label="D3:second")
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(method=labels.GFN1_XTB, exclude=("disp",)),
        classical=(first, second),
    )
    system = model.setup(numbers)
    assert system.exact_core is not None
    assert tuple(owner for owner, _ in system.d3_setups) == (first, second)
    d3_terms = [
        term for term in system.exact_core.classical_terms if term.kind == "d3"
    ]
    assert [term.label for term in d3_terms] == ["D3:first", "D3:second"]
    assert [term.position for term in d3_terms] == sorted(
        term.position for term in d3_terms
    )
    assert all(
        term.setup is setup
        for term, (_, setup) in zip(d3_terms, system.d3_setups)
    )
    assert (
        replace(
            system, d3_setups=tuple(reversed(system.d3_setups))
        )._active_exact_core()
        is None
    )


def test_exact_core_checks_d4sc_component_identity() -> None:
    """A same-label interaction cannot claim another component's setup."""
    system = Model(
        par=ParamModule(GFN2_XTB, dtype=torch.float64),
        config=Config.create(method=labels.GFN2_XTB),
    ).setup(torch.tensor([1, 1]))
    assert system.d4sc_interaction is not None
    assert system.es2_interaction is not None
    assert system.exact_core is not None
    assert (
        system.interaction_setup(system.d4sc_interaction) is system.d4sc_setup
    )
    assert (
        replace(
            system,
            d4sc_interaction=system.es2_interaction,
        )._active_exact_core()
        is None
    )


def test_exact_core_floating_setup_device_and_dtype_are_prevalidated() -> None:
    """Coherent exact cores reject mismatched setup device and dtype early."""
    system = _model().setup(torch.tensor([1, 1]))
    assert system.exact_core is not None
    assert system.h0_setup is not None
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )

    meta_h0 = system.h0_setup.to(device="meta")
    # An inconsistent mixed-device core is rejected during Node construction;
    # the established facade validation still reports the early device error.
    meta_system = replace(system, h0_setup=meta_h0, exact_core=None)
    with pytest.raises(DeviceError, match="System floating setup"):
        meta_system.singlepoint(positions)

    mismatched_core = system.exact_core.replace(effective_dtype=torch.float32)
    mismatched = replace(system, exact_core=mismatched_core)
    with pytest.raises(DtypeError, match="exact core declares"):
        mismatched.singlepoint(positions)


def test_device_mismatched_captured_index_uses_legacy_preparation() -> None:
    """A structural device mismatch declines trusted map reuse safely."""
    from dxtb._src.calculators.singlepoint import singlepoint

    system = _model().setup(torch.tensor([1, 1]))
    assert system.exact_core is not None
    assert system.exact_core.index is not None
    mismatched_backend = system.exact_core.integral.backend.to(device="meta")
    mismatched_integral = system.exact_core.integral.replace(
        backend=mismatched_backend
    )
    mismatched_core = system.exact_core.replace(integral=mismatched_integral)
    fallback_system = replace(system, exact_core=mismatched_core)
    assert fallback_system._active_exact_core() is None
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    result = singlepoint(fallback_system, positions)
    assert torch.isfinite(result.energy).all()


def test_active_core_index_device_mismatch_uses_public_solver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Device-mismatched backends fail validation; mismatched core.index falls back to scf.solve."""
    import copy

    from dxtb._src.calculators import singlepoint as singlepoint_module

    system = _model().setup(torch.tensor([1, 1]))
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    assert system.exact_core is not None
    assert system.exact_core.integral is not None

    # 1. Replacing the backend with a meta-device backend is rejected by structural validation.
    backend = system.integral_setup.pytorch
    meta_backend = backend.to(device="meta")
    meta_integrals = replace(system.integral_setup, backend=meta_backend)
    meta_integral = system.exact_core.integral.replace(backend=meta_backend)
    meta_core = system.exact_core.replace(integral=meta_integral)
    mismatched = replace(
        system, integral_setup=meta_integrals, exact_core=meta_core
    )
    assert mismatched._ihelp_compatible() is False
    assert mismatched._active_exact_core() is None
    dtype = torch.float64
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=dtype)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        singlepoint(mismatched, positions)

    # 2. Singlepoint always passes canonical system.ihelp to integral building and
    # routes directly through scf.solve without dual-index handoff.
    assert not hasattr(singlepoint_module.scf, "solve_with_core_index")
    corrupted_core = copy.copy(system.exact_core)
    object.__setattr__(corrupted_core, "index", system.ihelp.to(device="meta"))
    monkeypatch.setattr(
        type(system), "_active_exact_core", lambda self: corrupted_core
    )

    observed_ihelp: list[IndexHelper] = []
    orig_build = singlepoint_module.build_integral_matrices

    def record_build(setup, h0, ihelp, pos, charge=0.0):
        observed_ihelp.append(ihelp)
        return orig_build(setup, h0, ihelp, pos, charge=charge)

    monkeypatch.setattr(
        singlepoint_module, "build_integral_matrices", record_build
    )

    class AuthoritativeSolverReached(RuntimeError):
        pass

    def solver_reached(*args, **kwargs):
        assert kwargs == {}
        assert args[6] is system.ihelp
        raise AuthoritativeSolverReached

    monkeypatch.setattr(singlepoint_module.scf, "solve", solver_reached)
    with pytest.raises(AuthoritativeSolverReached):
        singlepoint(system, positions)
    assert observed_ihelp == [system.ihelp]


def test_calculator_type_updates_no_setup_exact_core_dtype() -> None:
    """Calculator.type keeps effective dtype when no float setup exists."""
    from dxtb import Calculator

    calculator = Calculator(
        torch.tensor([1, 1]),
        GFN1_XTB,
        opts={"exclude": "all", "int_level": 0, "verbosity": 0},
        dtype=torch.float64,
        auto_int_level=False,
    )
    assert calculator.system.exact_core is not None
    assert calculator.system.integral_setup is None
    assert calculator.system.h0_setup is None
    assert calculator.system.exact_core.effective_dtype is torch.float64

    calculator.type(torch.float32)
    assert calculator.system.exact_core is not None
    assert calculator.system.exact_core.effective_dtype is torch.float32
    positions32 = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float32
    )
    assert (
        calculator.system.singlepoint(positions32).energy.dtype is torch.float32
    )

    calculator.type(torch.float64)
    calculator.type(torch.float32)
    assert calculator.system.exact_core.effective_dtype is torch.float32


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_no_setup_dtype_checks_include_device(dtype: torch.dtype) -> None:
    """No-integral Systems retain dtype and device validation."""
    config = Config.create(
        method=labels.GFN1_XTB,
        exclude=("all",),
        int_level=0,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=dtype),
        config=config,
        auto_int_level=False,
    ).setup(torch.tensor([1, 1]), dd={"dtype": dtype, "device": "cpu"})
    positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=dtype)
    assert system.singlepoint(positions).energy.dtype == dtype
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        other_dtype = torch.float64 if dtype == torch.float32 else torch.float32
        system.singlepoint(positions.to(other_dtype))
    with pytest.raises(DeviceError, match="Device mismatch"):
        system.singlepoint(positions.to(device="meta"))


@pytest.mark.parametrize(
    "method,method_label,molecule",
    [
        (GFN0_XTB, labels.GFN0_XTB, "SiH4"),
        (GFN1_XTB, labels.GFN1_XTB, "H2"),
        (GFN2_XTB, labels.GFN2_XTB, "H2O"),
    ],
    ids=["gfn0-SiH4", "gfn1-H2", "gfn2-H2O"],
)
def test_captured_index_maps_match_h0_and_legacy(
    method: str, method_label: int, molecule: str
) -> None:
    """Single canonical IndexHelper is owned by System and ExactSystemCore, not integral backends."""
    numbers = mols[molecule]["numbers"].to(device="cpu")
    config = Config.create(
        method=method_label, int_driver=labels.INTDRIVER_PYTORCH
    )
    system = Model(
        par=ParamModule(method, dtype=torch.float64), config=config
    ).setup(numbers)
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    assert system.h0_setup is not None
    assert system.exact_core is not None
    assert isinstance(system.exact_core.index, IndexHelper)
    assert system.exact_core.index is system.ihelp
    assert not hasattr(system.integral_setup, "index")
    assert not hasattr(system.integral_setup.pytorch, "index")
    assert torch.equal(
        system.integral_setup.pytorch.pair_plan.atom,
        system.ihelp.shells_to_atom,
    )
    assert not hasattr(system.h0_setup, "index")
    assert not hasattr(system.h0_setup, "ihelp")


def test_captured_libcint_index_maps_match_h0_and_legacy() -> None:
    """Libcint System keeps canonical IndexHelper on System and ExactSystemCore only."""
    pytest.importorskip("tad_libcint")
    numbers = mols["H2O"]["numbers"].to(device="cpu")
    config = Config.create(
        method=labels.GFN1_XTB, int_driver=labels.INTDRIVER_LIBCINT
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64), config=config
    ).setup(numbers)
    assert system.integral_setup is not None
    assert system.integral_setup.libcint is not None
    assert system.h0_setup is not None
    assert system.exact_core is not None
    assert isinstance(system.exact_core.index, IndexHelper)
    assert system.exact_core.index is system.ihelp
    assert not hasattr(system.integral_setup, "index")
    assert not hasattr(system.integral_setup.libcint, "index")
    assert not hasattr(system.h0_setup, "index")
    assert not hasattr(system.h0_setup, "ihelp")


def test_singlepoint_reuses_captured_scf_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible System SCF inputs reuse the active core map Node."""
    from dxtb._src.scf import eager

    system = _model().setup(torch.tensor([1, 1]))
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    assert system.exact_core is not None
    expected_index = system.exact_core.index
    assert expected_index is system.ihelp
    original = eager.setup_scf_inputs
    index_type = IndexHelper
    constructions = 0
    prepared = []
    original_init = index_type.__init__

    def count_init(self, *args, **kwargs):
        nonlocal constructions
        constructions += 1
        original_init(self, *args, **kwargs)

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(eager, "setup_scf_inputs", record)
    monkeypatch.setattr(index_type, "__init__", count_init)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    clone_counter = _MapCloneCounter(
        (system.ihelp.shells_to_atom, system.ihelp.orbitals_to_shell)
    )
    with clone_counter:
        system.singlepoint(positions)
    assert len(prepared) == 1
    assert prepared[0].index is expected_index
    assert constructions == 0
    assert clone_counter.count == 0


def test_implicit_singlepoint_reuses_captured_scf_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The implicit physical-root route receives the same captured map."""
    from dataclasses import replace

    from dxtb._src.scf import eager
    from dxtb.config import ConfigSCF

    system = _model().setup(torch.tensor([1, 1]))
    scf = ConfigSCF.create(
        method=system.config.scf.method,
        differentiation="implicit",
        strategy=scf_strategy("linear"),
        maxiter=100,
        x_atol=1.0e-12,
        x_atol_max=1.0e-12,
    )
    assert system.exact_core is not None
    config = replace(system.config, scf=scf)
    exact_core = replace(system.exact_core, config=config)
    system = replace(
        system,
        config=config,
        exact_core=exact_core,
    )
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    assert system.exact_core is not None
    expected_index = system.exact_core.index
    assert expected_index is system.ihelp
    original = eager.setup_scf_inputs
    prepared = []

    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        prepared.append(result)
        return result

    monkeypatch.setattr(eager, "setup_scf_inputs", record)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    result = singlepoint(system, positions)
    assert result.converged is not None and bool(result.converged)
    assert result.residual is not None
    assert result.status is not None
    assert len(prepared) == 1
    assert prepared[0].index is expected_index


def test_direct_scf_input_preparation_keeps_isolated_map_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct setup reuses the frozen IndexHelper without cloning."""
    from test.test_scf.test_map import _prepare_case

    from dxtb._src.scf.data import setup_scf_inputs

    case = _prepare_case(GFN1_XTB, setup_functional=False)
    legacy_maps = (
        case.system.ihelp.shells_to_atom,
        case.system.ihelp.orbitals_to_shell,
    )
    index_type = IndexHelper
    constructions = 0
    original_init = index_type.__init__

    def count_init(self, *args, **kwargs):
        nonlocal constructions
        constructions += 1
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(index_type, "__init__", count_init)
    clone_counter = _MapCloneCounter(legacy_maps)
    with clone_counter:
        inputs = setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            case.system.config.scf,
        )
    assert isinstance(inputs.index, IndexHelper)
    assert inputs.index is case.system.ihelp
    assert clone_counter.count == 0
    assert constructions == 0

    assert case.system.exact_core is not None
    unrelated_index = IndexHelper(
        **{
            name: getattr(case.system.ihelp, name).clone()
            for name in case.system.ihelp._child_names
        },
        batch_mode=0,
    )
    with pytest.raises(TypeError, match="core_index"):
        setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            case.system.ihelp,
            case.system.config.scf,
            core_index=unrelated_index,  # type: ignore[call-arg]
        )
    other_ihelp = IndexHelper.from_numbers(
        torch.tensor([1, 1]), ParamModule(GFN1_XTB, dtype=torch.float64)
    )
    with pytest.raises(ValueError, match="atom shape"):
        setup_scf_inputs(
            case.system.numbers,
            case.integrals,
            case.n0,
            case.occupation,
            case.nel,
            case.interactions,
            case.interaction_data,
            other_ihelp,
            case.system.config.scf,
        )


def test_scf_orbital_mask_and_conditioning_keep_indexhelper_padding() -> None:
    """The captured reduction maps do not replace padding-aware mask setup."""
    from test.test_scf.test_map import _prepare_case

    from dxtb._src.scf.data import _symmetric_inverse_sqrt

    case = _prepare_case(GFN1_XTB, sample_name="SiH4")
    expected_mask = case.system.ihelp.spread_shell_to_orbital(
        case.system.ihelp.orbitals_per_shell
    ).to(dtype=torch.bool)
    torch.testing.assert_close(case.inputs.orbital_mask, expected_mask)
    padding_diagonal = (~expected_mask).to(dtype=case.integrals.overlap.dtype)
    conditioned_overlap = case.integrals.overlap + torch.diag_embed(
        padding_diagonal
    )
    expected_inverse = _symmetric_inverse_sqrt(conditioned_overlap)
    torch.testing.assert_close(case.inputs.inverse_cholesky, expected_inverse)


def test_singlepoint_does_not_regather_param_module_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Captured exact System evaluation does not query parameter tables."""
    system = _model().setup(torch.tensor([1, 1]))
    positions_a = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    positions_b = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64
    )
    calls = []
    original_get = ParamModule.get
    original_get_elem_param = ParamModule.get_elem_param

    def record_get(self, *args, **kwargs):
        calls.append(("get", args))
        return original_get(self, *args, **kwargs)

    def record_get_elem_param(self, *args, **kwargs):
        calls.append(("get_elem_param", args))
        return original_get_elem_param(self, *args, **kwargs)

    monkeypatch.setattr(ParamModule, "get", record_get)
    monkeypatch.setattr(ParamModule, "get_elem_param", record_get_elem_param)
    system.singlepoint(positions_a)
    system.singlepoint(positions_b)
    assert calls == []


def test_captured_integral_maps_are_not_rebuilt_during_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pure single-System path consumes its captured integral setup."""
    numbers = torch.tensor([1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    system = _model().setup(numbers)
    setup = system.integral_setup
    assert setup is not None and setup.pytorch is not None
    assert type(setup.pytorch).__name__ == "PytorchIntegralSetup"
    assert not hasattr(setup, "index")
    assert not hasattr(setup.pytorch, "index")
    assert not hasattr(setup.pytorch, "ihelp")
    assert not hasattr(system.h0_setup, "ihelp")
    assert system.exact_core is not None
    assert isinstance(system.exact_core.index, IndexHelper)
    assert system.exact_core.index is system.ihelp

    def fail_rebuild(*args, **kwargs):
        raise AssertionError("IndexHelper construction ran during evaluation")

    monkeypatch.setattr(IndexHelper, "from_numbers", fail_rebuild)
    result = singlepoint(system, positions)
    assert torch.isfinite(result.energy).all()


def test_overlap_history_transforms_and_parameter_graph() -> None:
    """Captured setup supports A/B/A reuse and the established pure transforms."""
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    par.set_differentiable("element.H.slater")
    numbers = torch.tensor([1, 1])
    system = Model(
        par=par,
        config=Config.create(int_driver=labels.INTDRIVER_PYTORCH),
    ).setup(numbers)
    assert system.integral_setup is not None
    setup = system.integral_setup.pytorch
    assert setup is not None
    parameter = par.get("element.H.slater")

    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)
    evaluate = lambda geometry: build_overlap(setup, geometry)
    overlap_a = evaluate(a)
    evaluate(b)
    overlap_a_after = evaluate(a)
    torch.testing.assert_close(overlap_a_after, overlap_a)

    direction = torch.ones_like(a) * 0.1
    tangent = jvp(evaluate, (a,), (direction,))[1]
    forward = jacfwd(evaluate)(a)
    reverse = jacrev(evaluate)(a)
    torch.testing.assert_close(forward, reverse)
    torch.testing.assert_close(
        tangent, torch.einsum("ijab,ab->ij", forward, direction)
    )
    batched = vmap(evaluate)(torch.stack((a, b)))
    torch.testing.assert_close(batched, torch.stack((evaluate(a), evaluate(b))))

    value = evaluate(a).square().sum()
    (parameter_gradient,) = torch.autograd.grad(value, parameter)
    assert torch.isfinite(parameter_gradient).all()
    assert torch.count_nonzero(parameter_gradient) > 0


def test_singlepoint_geometry_history_preserves_core_outputs_and_gradients() -> (
    None
):
    """System evaluation remains independent of an intervening geometry."""
    numbers = torch.tensor([1, 1])
    system = _model().setup(numbers)
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)

    first = system.singlepoint(a)
    system.singlepoint(b)
    last = system.singlepoint(a)
    torch.testing.assert_close(first.energy, last.energy)
    torch.testing.assert_close(first.overlap, last.overlap)
    torch.testing.assert_close(first.hcore, last.hcore)
    assert tuple(name for name, _ in first.classical) == tuple(
        name for name, _ in last.classical
    )
    for (_, left), (_, right) in zip(first.classical, last.classical):
        torch.testing.assert_close(left, right)
    assert first.charges is not None and last.charges is not None
    assert first.potential is not None and last.potential is not None
    for name in ("mono", "dipole", "quadrupole"):
        left = getattr(first.charges, name)
        right = getattr(last.charges, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right)
        left = getattr(first.potential, name)
        right = getattr(last.potential, name)
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right)

    positions_first = a.clone().requires_grad_(True)
    energy_first = system.singlepoint(positions_first).energy
    (gradient_first,) = torch.autograd.grad(energy_first.sum(), positions_first)
    system.singlepoint(b)
    positions_last = a.clone().requires_grad_(True)
    energy_last = system.singlepoint(positions_last).energy
    (gradient_last,) = torch.autograd.grad(energy_last.sum(), positions_last)
    torch.testing.assert_close(gradient_first, gradient_last)


def test_call_local_interaction_data_is_geometry_history_independent() -> None:
    """The ES2 matrix cache is rebuilt from setup and current geometry."""
    config = Config.create(
        int_driver=labels.INTDRIVER_PYTORCH,
        exclude=("d4sc",),
    )
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64), config=config
    ).setup(torch.tensor([1, 1]))
    a = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
    b = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.6]], dtype=torch.float64)
    first = _interaction_data(system, a)
    _interaction_data(system, b)
    last = _interaction_data(system, a)
    assert tuple(first) == tuple(last)
    for label in first:
        fields = tuple(
            name for name in ("mat", "hd") if hasattr(first[label], name)
        )
        assert fields
        for name in fields:
            torch.testing.assert_close(
                getattr(first[label], name), getattr(last[label], name)
            )


def test_exact_core_optional_none_fields_and_stacking_compatibility() -> None:
    """Optional None setup fields round-trip and stack checks structural compatibility."""
    from tad_mctc.tree import stack
    from torch.utils import _pytree

    model = _model()
    no_int = Model(
        par=model.par,
        config=Config.create(
            method=labels.GFN1_XTB, exclude=("all",), int_level=0
        ),
        auto_int_level=False,
    ).setup(torch.tensor([1, 1]))
    assert no_int.exact_core is not None
    assert no_int.exact_core.integral is None
    assert no_int.exact_core.h0_setup is None
    leaves, spec = _pytree.tree_flatten(no_int.exact_core)
    rebuilt = _pytree.tree_unflatten(leaves, spec)
    assert rebuilt.integral is None
    assert rebuilt.h0_setup is None

    c2 = model.setup(torch.tensor([6, 6])).exact_core
    n2 = model.setup(torch.tensor([7, 7])).exact_core
    h2 = model.setup(torch.tensor([1, 1])).exact_core
    assert c2 is not None and n2 is not None and h2 is not None
    stacked = stack((c2, n2))
    assert stacked.numbers.shape == (2, 2)
    with pytest.raises(ValueError, match="different structure"):
        stack((h2, c2))


@pytest.mark.parametrize(
    ("method", "method_label", "kind"),
    [
        (GFN1_XTB, labels.GFN1_XTB, "d3"),
        (GFN2_XTB, labels.GFN2_XTB, "aes2"),
    ],
    ids=["d3", "aes2"],
)
def test_descriptor_conversion_preserves_metadata_dtypes_and_pytree(
    method: object,
    method_label: int,
    kind: str,
) -> None:
    """D3 and AES2 descriptor conversion preserves metadata, structural dtypes, and PyTree structure."""
    from torch.utils import _pytree

    system = Model(
        par=ParamModule(method, dtype=torch.float64),
        config=Config.create(
            method=method_label,
            int_driver=labels.INTDRIVER_PYTORCH,
        ),
    ).setup(torch.tensor([8, 1, 1]))
    assert system.exact_core is not None
    descriptor = next(
        term
        for term in (
            *system.exact_core.classical_terms,
            *system.exact_core.interaction_terms,
        )
        if term.kind == kind
    )
    leaves, spec = _pytree.tree_flatten(descriptor)

    converted = descriptor.to(dtype=torch.float32)
    converted_leaves, converted_spec = _pytree.tree_flatten(converted)
    rebuilt = _pytree.tree_unflatten(converted_leaves, converted_spec)

    assert converted.position == descriptor.position
    assert converted.kind == descriptor.kind
    assert converted.label == descriptor.label
    assert converted_spec == spec
    assert type(rebuilt) is type(descriptor)
    assert type(rebuilt.setup) is type(descriptor.setup)
    for before, after in zip(leaves, converted_leaves, strict=True):
        if before.is_floating_point():
            assert after.dtype == torch.float32
        elif before.dtype is torch.bool:
            assert after.dtype is torch.bool
        else:
            assert after.dtype == before.dtype


def test_system_ihelp_replacement_and_structural_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Equivalent IndexHelper replacement preserves core reuse; incompatible replacements fail early."""
    from dxtb._src.calculators.lane_evaluation import can_vmap_system
    from dxtb._src.calculators.packed_result import evaluate_packed_systems
    from dxtb._src.scf import eager as scf_eager

    model = _model()
    numbers = mols["H2O"]["numbers"].to(device="cpu")
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)
    system = model.setup(numbers)
    assert system.exact_core is not None

    # 1. Original IndexHelper & 2. Equivalent separately constructed IndexHelper
    equiv_ihelp = IndexHelper.from_numbers(numbers, model.par)
    assert equiv_ihelp is not system.ihelp
    equiv_system = replace(system, ihelp=equiv_ihelp)

    assert system._active_exact_core() is system.exact_core
    assert equiv_system._active_exact_core() is system.exact_core
    assert can_vmap_system(system) is True
    assert can_vmap_system(equiv_system) is True

    prepared_inputs = []
    orig_setup_scf_inputs = scf_eager.setup_scf_inputs

    def record_scf_inputs(*args, **kwargs):
        res = orig_setup_scf_inputs(*args, **kwargs)
        prepared_inputs.append(res)
        return res

    monkeypatch.setattr(scf_eager, "setup_scf_inputs", record_scf_inputs)

    pos_orig = positions.clone().requires_grad_(True)
    pos_equiv = positions.clone().requires_grad_(True)
    res_orig = system.singlepoint(pos_orig)
    res_equiv = equiv_system.singlepoint(pos_equiv)
    (grad_orig,) = torch.autograd.grad(res_orig.energy.sum(), pos_orig)
    (grad_equiv,) = torch.autograd.grad(res_equiv.energy.sum(), pos_equiv)

    torch.testing.assert_close(res_equiv.energy, res_orig.energy)
    assert res_equiv.charges is not None and res_orig.charges is not None
    torch.testing.assert_close(res_equiv.charges.mono, res_orig.charges.mono)
    torch.testing.assert_close(grad_equiv, grad_orig)

    assert len(prepared_inputs) == 2
    assert prepared_inputs[0].index is system.ihelp
    assert prepared_inputs[0].index is system.exact_core.index
    assert prepared_inputs[1].index is equiv_system.ihelp
    torch.testing.assert_close(
        prepared_inputs[1].orbital_mask, prepared_inputs[0].orbital_mask
    )
    torch.testing.assert_close(prepared_inputs[1].n0, prepared_inputs[0].n0)

    packed = evaluate_packed_systems(
        (system, equiv_system), (positions, positions)
    )
    torch.testing.assert_close(packed.energy[0], res_orig.energy)
    torch.testing.assert_close(packed.energy[1], res_orig.energy)

    # 3. IndexHelper with incompatible shell/orbital mappings (same shapes)
    bad_shells = equiv_ihelp.shells_to_atom.clone()
    bad_shells[0], bad_shells[-1] = (
        bad_shells[-1].clone(),
        bad_shells[0].clone(),
    )
    incompat_shell_ihelp = IndexHelper(
        unique_angular=equiv_ihelp.unique_angular.clone(),
        angular=equiv_ihelp.angular.clone(),
        atom_to_unique=equiv_ihelp.atom_to_unique.clone(),
        ushells_to_unique=equiv_ihelp.ushells_to_unique.clone(),
        ushells_per_unique=equiv_ihelp.ushells_per_unique.clone(),
        shells_to_ushell=equiv_ihelp.shells_to_ushell.clone(),
        shells_per_atom=equiv_ihelp.shells_per_atom.clone(),
        shell_index=equiv_ihelp.shell_index.clone(),
        shells_to_atom=bad_shells,
        orbitals_per_shell=equiv_ihelp.orbitals_per_shell.clone(),
        orbital_index=equiv_ihelp.orbital_index.clone(),
        orbitals_to_shell=equiv_ihelp.orbitals_to_shell.clone(),
        batch_mode=0,
    )
    bad_orbs = equiv_ihelp.orbitals_to_shell.clone()
    bad_orbs[0] = bad_orbs[-1]
    incompat_orb_ihelp = IndexHelper(
        unique_angular=equiv_ihelp.unique_angular.clone(),
        angular=equiv_ihelp.angular.clone(),
        atom_to_unique=equiv_ihelp.atom_to_unique.clone(),
        ushells_to_unique=equiv_ihelp.ushells_to_unique.clone(),
        ushells_per_unique=equiv_ihelp.ushells_per_unique.clone(),
        shells_to_ushell=equiv_ihelp.shells_to_ushell.clone(),
        shells_per_atom=equiv_ihelp.shells_per_atom.clone(),
        shell_index=equiv_ihelp.shell_index.clone(),
        shells_to_atom=equiv_ihelp.shells_to_atom.clone(),
        orbitals_per_shell=equiv_ihelp.orbitals_per_shell.clone(),
        orbital_index=equiv_ihelp.orbital_index.clone(),
        orbitals_to_shell=bad_orbs,
        batch_mode=0,
    )

    # 4. IndexHelper from a different chemical composition on the same device
    h2_ihelp = IndexHelper.from_numbers(torch.tensor([1, 1]), model.par)
    permuted_h2o_ihelp = IndexHelper.from_numbers(
        torch.tensor([1, 8, 1]), model.par
    )

    for bad_ihelp in (
        incompat_shell_ihelp,
        incompat_orb_ihelp,
        h2_ihelp,
        permuted_h2o_ihelp,
    ):
        bad_system = replace(system, ihelp=bad_ihelp)
        assert bad_system._active_exact_core() is None
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            can_vmap_system(bad_system)
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            bad_system.singlepoint(positions)
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            bad_system.energy(positions)
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            evaluate_packed_systems((bad_system,), (positions,))


def test_system_ihelp_in_place_mutation_detected() -> None:
    """In-place mutation of setup IndexHelper tensors invalidates token and fails validation."""
    from dxtb._src.calculators.lane_evaluation import can_vmap_system

    model = _model()
    numbers = mols["H2O"]["numbers"].to(device="cpu")
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)

    for attr, bad_val in (
        ("shells_to_atom", 2),
        ("orbitals_to_shell", 3),
        ("unique_angular", 2),
    ):
        system = model.setup(numbers.clone())
        assert system._active_exact_core() is system.exact_core
        target = getattr(system.ihelp, attr)
        orig_val = int(target[0].item())
        target[0] = bad_val
        assert system._active_exact_core() is None
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            can_vmap_system(system)
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            system.singlepoint(positions)
        # Restoring the original value makes the structural check pass again.
        target[0] = orig_val
        assert system._active_exact_core() is system.exact_core


def test_system_numbers_replacement_and_in_place_mutation_detected() -> None:
    """Replacing or mutating System.numbers with same-dimension N/O species fails validation."""
    from dxtb._src.calculators.lane_evaluation import can_vmap_system
    from dxtb._src.calculators.packed_result import evaluate_packed_systems

    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(int_driver=labels.INTDRIVER_PYTORCH, maxiter=6),
    )
    n2_numbers = torch.tensor([7, 7])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 2.1]], dtype=torch.float64
    )
    system = model.setup(n2_numbers.clone())
    assert system._active_exact_core() is system.exact_core
    baseline = system.singlepoint(positions)

    # 1. Fresh tensor with identical composition [7, 7] remains structurally compatible
    same_comp = replace(system, numbers=torch.tensor([7, 7]))
    assert same_comp._ihelp_compatible() is True
    torch.testing.assert_close(
        same_comp.singlepoint(positions).energy, baseline.energy
    )

    # 2. Replacing System.numbers with [8, 8] or [7, 8] (same atom/shell/orbital counts in GFN1)
    for bad_numbers in (torch.tensor([8, 8]), torch.tensor([7, 8])):
        for candidate in (
            replace(system, numbers=bad_numbers),
            replace(system, numbers=bad_numbers, _setup_index_token=None),
        ):
            assert candidate._active_exact_core() is None
            assert candidate._ihelp_compatible() is False
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                can_vmap_system(candidate)
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                candidate.singlepoint(positions)
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                candidate.energy(positions)
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                evaluate_packed_systems((candidate,), (positions,))

    # 3. In-place mutation of System.numbers (N -> O preserves GFN1 basis dimensions)
    system.numbers[0] = 8
    assert system._active_exact_core() is None
    assert system._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        can_vmap_system(system)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        system.singlepoint(positions)
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        system.energy(positions)

    # Restoring the original atomic number restores exact-core eligibility
    system.numbers[0] = 7
    assert system._active_exact_core() is system.exact_core
    assert system._ihelp_compatible() is True
    torch.testing.assert_close(
        system.singlepoint(positions).energy, baseline.energy
    )


def test_meta_device_leaf_mismatch_is_metadata_rejected_only() -> None:
    """Meta tensors probe structural validation, never numerical device support."""
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN1_XTB, int_driver=labels.INTDRIVER_PYTORCH
        ),
    )
    core = model.setup(torch.tensor([1, 1])).exact_core
    assert core is not None
    assert core.integral is not None and core.integral.backend is not None
    backend = core.integral.backend
    meta_alpha = backend.alphas[0].to(device="meta")
    with pytest.raises(RuntimeError, match="different devices"):
        backend.replace(alphas=(meta_alpha, *backend.alphas[1:]))


def test_pair_plan_and_foreign_pytorch_backend_mutation_guards() -> None:
    """In-place PairPlan mutations and foreign same-shaped PyTorch backends fail validation."""
    numbers = torch.tensor([8, 1, 1])
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)
    model = Model(
        par=ParamModule(GFN2_XTB, dtype=torch.float64),
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            maxiter=6,
        ),
    )

    # 1. In-place mutation of every compiled PairPlan structural tensor.
    system = model.setup(numbers.clone())
    assert system.integral_setup is not None
    pt_setup = system.integral_setup.pytorch
    assert pt_setup is not None
    assert system._has_intact_setup_token() is True
    assert system._active_exact_core() is system.exact_core

    cls0 = pt_setup.pair_plan.classes[0]
    for tensor, delta in (
        (pt_setup.pair_plan.orbitals_to_atom, 1),
        (cls0.ib, 1),
        (cls0.jk, 1),
        (cls0.idx, 1),
        (cls0.weight, 0.25),
    ):
        flat = tensor.view(-1)
        original = flat[0].clone()
        with torch.no_grad():
            flat[0] = original + delta
        assert system._has_intact_setup_token() is False
        assert system._ihelp_compatible() is False
        assert system._active_exact_core() is None
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            system.singlepoint(positions)
        with torch.no_grad():
            flat[0] = original
        # Version counter stays bumped, so full value check runs and succeeds!
        assert system._has_intact_setup_token() is False
        assert system._ihelp_compatible() is True
        assert system._active_exact_core() is system.exact_core

    # Detached PairPlan.atom mismatch (without touching System.ihelp.shells_to_atom).
    bad_atom = pt_setup.pair_plan.atom.clone()
    bad_atom[0] = 2
    bad_plan = pt_setup.pair_plan.replace(atom=bad_atom)
    bad_pt = pt_setup.replace(pair_plan=bad_plan)
    bad_is = system.integral_setup.replace(backend=bad_pt)
    assert system.exact_core is not None
    bad_sys = replace(
        system,
        integral_setup=bad_is,
        exact_core=system.exact_core.replace(integral=bad_is),
    )
    assert bad_sys._ihelp_compatible() is False
    assert bad_sys._active_exact_core() is None
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        bad_sys.singlepoint(positions)

    # 2. Foreign same-shaped PyTorch backend (permuted H2O [1, 8, 1] on [8, 1, 1]).
    foreign_sys = model.setup(torch.tensor([1, 8, 1]))
    assert foreign_sys.integral_setup is not None
    for clear_token in (False, True):
        swapped = replace(
            system,
            integral_setup=foreign_sys.integral_setup,
            exact_core=system.exact_core.replace(
                integral=foreign_sys.integral_setup
            ),
            _setup_index_token=(
                None if clear_token else system._setup_index_token
            ),
        )
        assert swapped._ihelp_compatible() is False
        assert swapped._active_exact_core() is None
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            swapped.singlepoint(positions)


def test_libcint_structural_safeguards_mutations_and_padded_unsorted() -> None:
    """Libcint Systems validate canonical IndexHelper, mutations, shell_layout, and Z=0 padding."""
    pytest.importorskip("tad_libcint")
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    model_lc = Model(
        par=par,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_LIBCINT,
            maxiter=8,
        ),
    )
    numbers = torch.tensor([8, 1, 1])
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)
    system = model_lc.setup(numbers.clone())
    assert system.exact_core is not None
    assert system.integral_setup is not None
    assert system.integral_setup.libcint is not None
    baseline = system.singlepoint(positions)

    # 1. Equivalent independently allocated int64 replacement reuses core;
    # non-int64 IndexHelper is rejected before torch.gather/scatter_reduce.
    equiv_i64 = IndexHelper.from_numbers(numbers, par)
    equiv_sys = replace(system, ihelp=equiv_i64)
    assert equiv_sys._ihelp_compatible() is True
    assert equiv_sys._active_exact_core() is system.exact_core
    torch.testing.assert_close(
        equiv_sys.singlepoint(positions).energy, baseline.energy
    )

    equiv_i32 = IndexHelper(
        **{
            name: getattr(equiv_i64, name).to(torch.int32)
            for name in equiv_i64._child_names
        },
        batch_mode=0,
    )
    equiv_i32_sys = replace(system, ihelp=equiv_i32)
    assert equiv_i32_sys._ihelp_compatible() is False
    assert equiv_i32_sys._active_exact_core() is None
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        equiv_i32_sys.singlepoint(positions)

    # 2. Same-shaped incompatible IndexHelper on libcint System & mismatched core.index.
    permuted_ihelp = IndexHelper.from_numbers(torch.tensor([1, 8, 1]), par)
    assert (
        replace(
            system, exact_core=system.exact_core.replace(index=permuted_ihelp)
        )._active_exact_core()
        is None
    )
    bad_ihelp_sys = replace(system, ihelp=permuted_ihelp)
    assert bad_ihelp_sys._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        bad_ihelp_sys.singlepoint(positions)

    # 3. In-place mutation of canonical IndexHelper on libcint System & no-integral System.
    system.ihelp.shells_to_atom[0] = 2
    assert system._has_intact_setup_token() is False
    assert system._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        system.singlepoint(positions)
    system.ihelp.shells_to_atom[0] = 0
    assert system._ihelp_compatible() is True

    no_int = Model(
        par=par,
        config=Config.create(
            method=labels.GFN2_XTB, exclude=("all",), int_level=0
        ),
        auto_int_level=False,
    ).setup(numbers.clone())
    assert no_int.exact_core is not None and no_int.exact_core.integral is None
    assert no_int.exact_core.index is no_int.ihelp
    no_int.ihelp.orbitals_to_shell[0] = 2
    assert no_int._ihelp_compatible() is False
    no_int.ihelp.orbitals_to_shell[0] = 0
    assert no_int._ihelp_compatible() is True

    # 4. In-place mutation of LibcintIntegralSetup.numbers and corrupted shell_layout.
    lc_backend = system.integral_setup.libcint
    lc_backend.numbers[0] = 7
    assert system._has_intact_setup_token() is False
    assert system._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        system.singlepoint(positions)
    lc_backend.numbers[0] = 8
    assert system._ihelp_compatible() is True

    orig_layout = lc_backend.shell_layout
    first_z, first_shells = orig_layout[0]
    corrupted_layouts = (
        ((7, first_shells), *orig_layout[1:]),
        (
            (
                first_z,
                (
                    (first_shells[0][0] + 1, first_shells[0][1]),
                    *first_shells[1:],
                ),
            ),
            *orig_layout[1:],
        ),
        (
            (
                first_z,
                (
                    (first_shells[0][0], orig_layout[1][1][0][1]),
                    *first_shells[1:],
                ),
            ),
            *orig_layout[1:],
        ),
        orig_layout[1:],
    )
    for bad_layout in corrupted_layouts:
        bad_lc = lc_backend.replace(shell_layout=bad_layout)
        bad_is = system.integral_setup.replace(backend=bad_lc)
        bad_sys = replace(
            system,
            integral_setup=bad_is,
            exact_core=system.exact_core.replace(integral=bad_is),
            _setup_index_token=None,
        )
        assert bad_sys._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            bad_sys.singlepoint(positions)

    # 5. Unsorted & noncontiguous Z=0 padded System: active_atoms mutation & Z=0 shell guard.
    nums_pad = torch.tensor([1, 0, 14, 0, 8, 1])
    pad_sys = model_lc.setup(nums_pad)
    assert pad_sys._ihelp_compatible() is True
    assert pad_sys._active_exact_core() is pad_sys.exact_core
    assert pad_sys.integral_setup is not None
    pad_lc = pad_sys.integral_setup.libcint
    assert pad_lc is not None and pad_lc.active_atoms is not None
    orig_active0 = int(pad_lc.active_atoms[0].item())
    pad_lc.active_atoms[0] = 1  # points to Z=0 slot!
    assert pad_sys._has_intact_setup_token() is False
    assert pad_sys._ihelp_compatible() is False
    pad_lc.active_atoms[0] = orig_active0
    assert pad_sys._ihelp_compatible() is True

    # Dummy Z=0 species cannot claim a real angular momentum shell (unique_angular >= 0).
    bad_ua = pad_sys.ihelp.unique_angular.clone()
    bad_ua[0] = 0
    bad_pad_ihelp = pad_sys.ihelp.replace(unique_angular=bad_ua)
    assert (
        replace(
            pad_sys, ihelp=bad_pad_ihelp, _setup_index_token=None
        )._ihelp_compatible()
        is False
    )


def test_exact_core_conversions_stacking_and_zero_sync_fast_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dtype/device conversions preserve canonical index consistency and zero-sync fast path."""
    from tad_mctc.tree import stack
    from torch.utils._pytree import tree_flatten, tree_unflatten

    from dxtb import Calculator
    from dxtb._src.calculators.singlepoint import build_integral_matrices
    from dxtb._src.calculators.types.base import (
        _convert_eval_system_dtype,
        _convert_system_dtype,
    )

    numbers = torch.tensor([8, 1, 1])
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)

    drivers = [labels.INTDRIVER_PYTORCH]
    if pytest.importorskip("dxtb._src.exlibs.available").has_libcint:
        drivers.append(labels.INTDRIVER_LIBCINT)

    for driver in drivers:
        calc = Calculator(
            numbers.clone(),
            GFN2_XTB,
            dtype=torch.float64,
            opts={"verbosity": 0, "int_driver": driver, "maxiter": 8},
        )
        system = calc.system
        assert system.exact_core is not None
        assert system.integral_setup is not None
        assert system.h0_setup is not None

        # ExactSystemCore.to(dtype=...) preserves canonical IndexHelper identity!
        core_f32 = system.exact_core.to(dtype=torch.float32)
        assert core_f32.index is system.ihelp
        assert core_f32.effective_dtype is torch.float32

        # _convert_system_dtype and _convert_eval_system_dtype refresh token cleanly.
        sys_f32 = _convert_system_dtype(system, torch.float32)
        eval_f32 = _convert_eval_system_dtype(system, torch.float32)
        for converted in (sys_f32, eval_f32):
            assert converted.exact_core is not None
            assert converted.exact_core.index is converted.ihelp
            assert converted._has_intact_setup_token() is True
            assert converted._ihelp_compatible() is True
        assert eval_f32._active_exact_core() is eval_f32.exact_core

        # Zero-GPU-sync fast path: normal evaluation on original and converted Systems
        # never calls torch.unique.
        def forbid_unique(*args, **kwargs):
            raise AssertionError(
                "torch.unique fallback ran on unmodified System"
            )

        monkeypatch.setattr(torch, "unique", forbid_unique)
        res64 = system.singlepoint(positions)
        res32 = eval_f32.singlepoint(positions.to(dtype=torch.float32))
        assert torch.isfinite(res64.energy).all()
        assert torch.isfinite(res32.energy).all()
        monkeypatch.undo()

        # Adversarial 1: replacing System.ihelp before dtype conversion stays rejected!
        permuted_ihelp = IndexHelper.from_numbers(
            torch.tensor([1, 8, 1]), calc.model.par
        )
        corrupted = replace(system, ihelp=permuted_ihelp)
        for bad_conv in (
            _convert_system_dtype(corrupted, torch.float32),
            _convert_eval_system_dtype(corrupted, torch.float32),
        ):
            assert bad_conv._ihelp_compatible() is False
            assert bad_conv._active_exact_core() is None

        # Adversarial 2: in-place mutation before dtype conversion drops the token
        # instead of re-blessing mutated tensors!
        system.numbers[0] = 7
        try:
            for mut_conv in (
                _convert_system_dtype(system, torch.float32),
                _convert_eval_system_dtype(system, torch.float32),
            ):
                assert mut_conv._setup_index_token is None
                assert mut_conv._has_intact_setup_token() is False
                assert mut_conv._ihelp_compatible() is False
                assert mut_conv._active_exact_core() is None
        finally:
            system.numbers[0] = 8

        # ExactSystemCore.to(device="meta") moves canonical index & logical_device to meta
        # while keeping libcint backend tensors CPU-resident.
        core_meta = system.exact_core.to(device="meta")
        assert core_meta.index.device.type == "meta"
        assert core_meta.numbers.device.type == "meta"
        assert core_meta.integral is not None
        assert core_meta.integral.logical_device == torch.device("meta")
        if driver == labels.INTDRIVER_LIBCINT:
            assert core_meta.integral.libcint is not None
            assert core_meta.integral.libcint.device.type == "cpu"
        else:
            assert core_meta.integral.pytorch is not None
            assert core_meta.integral.pytorch.device.type == "meta"

        # build_integral_matrices rejects mismatched canonical index device directly.
        with pytest.raises(
            DeviceError, match="Device mismatch: canonical IndexHelper"
        ):
            build_integral_matrices(
                system.integral_setup,
                system.h0_setup,
                system.ihelp.to(device="meta"),
                positions,
                0.0,
            )

        # Stack and unstack ExactSystemCore.
        model = Model(
            par=ParamModule(GFN1_XTB, dtype=torch.float64),
            config=Config.create(
                method=labels.GFN1_XTB, int_driver=driver, maxiter=6
            ),
        )
        c2_core = model.setup(torch.tensor([6, 6])).exact_core
        n2_core = model.setup(torch.tensor([7, 7])).exact_core
        assert c2_core is not None and n2_core is not None
        if driver == labels.INTDRIVER_LIBCINT:
            # Libcint shell_layout stores element Z in static context, so C2 vs N2
            # rejects stacking while same-composition parameter variants stack cleanly.
            with pytest.raises(ValueError, match="different structure"):
                stack((c2_core, n2_core))
            alt_par = ParamModule(GFN1_XTB, dtype=torch.float64)
            with torch.no_grad():
                alt_par.get("element.N.slater").mul_(1.03)
            n2_alt_core = (
                Model(par=alt_par, config=model.config)
                .setup(torch.tensor([7, 7]))
                .exact_core
            )
            assert n2_alt_core is not None
            first_core, second_core = n2_core, n2_alt_core
        else:
            first_core, second_core = c2_core, n2_core
        stacked_core = stack((first_core, second_core))
        assert stacked_core.index.shells_to_atom.shape == (2, 4)
        leaves, _ = tree_flatten(stacked_core)
        _, spec_single = tree_flatten(first_core)
        unstacked_second = tree_unflatten(
            [leaf[1] for leaf in leaves], spec_single
        )
        assert type(unstacked_second) is type(second_core)
        assert torch.equal(unstacked_second.numbers, second_core.numbers)
        assert torch.equal(
            unstacked_second.index.shells_to_atom,
            second_core.index.shells_to_atom,
        )


def test_reviewer_adversarial_tampering_compile_and_dtype_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify tampering guards across eval dtype conversion, H0, PairPlan dtypes, and Dynamo."""
    from torch.utils._pytree import tree_flatten, tree_unflatten

    from dxtb._src.calculators.batch_implicit import (
        _validate_system_and_positions,
    )
    from dxtb._src.calculators.types.base import (
        _convert_eval_system_dtype,
        _convert_system_dtype,
    )
    from dxtb._src.integral.driver.pytorch.impls.pairs import PairClass

    numbers = torch.tensor([8, 1, 1])
    positions = mols["H2O"]["positions"].to(device="cpu", dtype=torch.float64)
    par2 = ParamModule(GFN2_XTB, dtype=torch.float64)
    model2 = Model(
        par=par2,
        config=Config.create(
            method=labels.GFN2_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            maxiter=8,
        ),
    )
    sys = model2.setup(numbers.clone())
    ref_e64 = sys.singlepoint(positions).energy
    ref_eval_f32 = _convert_eval_system_dtype(sys, torch.float32)
    ref_e32 = ref_eval_f32.singlepoint(positions.to(dtype=torch.float32)).energy

    # 1. [BLOCKER-1] _convert_eval_system_dtype must never promote setups from an
    # inactive/tampered ExactSystemCore or re-bless _setup_index_token over bad ES2.
    assert sys.exact_core is not None and sys.es2_setup is not None
    assert sys.es2_setup.shells_to_atom is not None
    bad_es2 = sys.es2_setup.replace(
        shells_to_atom=torch.full_like(sys.es2_setup.shells_to_atom, 2)
    )
    tampered_terms = tuple(
        term.replace(setup=bad_es2) if term.kind == "es2" else term
        for term in sys.exact_core.interaction_terms
    )
    tampered_core_sys = replace(
        sys,
        exact_core=sys.exact_core.replace(interaction_terms=tampered_terms),
    )
    assert tampered_core_sys._active_exact_core() is None
    conv_tampered = _convert_eval_system_dtype(tampered_core_sys, torch.float32)
    assert conv_tampered._active_exact_core() is None
    assert conv_tampered.es2_setup is not None
    assert conv_tampered.es2_setup.shells_to_atom is not None
    assert torch.equal(
        conv_tampered.es2_setup.shells_to_atom, sys.ihelp.shells_to_atom
    )
    torch.testing.assert_close(
        conv_tampered.singlepoint(positions.to(dtype=torch.float32)).energy,
        ref_e32,
        atol=1e-6,
        rtol=1e-6,
    )

    bad_sys_es2 = replace(sys, es2_setup=bad_es2)
    conv_bad_es2 = _convert_eval_system_dtype(bad_sys_es2, torch.float32)
    assert conv_bad_es2._setup_index_token is None
    assert conv_bad_es2._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        conv_bad_es2.singlepoint(positions.to(dtype=torch.float32))

    # 2. [BLOCKER-2] H0SetupNode structural tampering (in-place numbers, foreign swap,
    # and valence mutation with or without _setup_index_token).
    assert sys.h0_setup is not None
    sys.h0_setup.numbers[0] = 1
    sys.h0_setup.numbers[1] = 8
    try:
        assert sys._has_intact_setup_token() is False
        assert sys._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            sys.singlepoint(positions)
    finally:
        sys.h0_setup.numbers[0] = 8
        sys.h0_setup.numbers[1] = 1
    assert sys._ihelp_compatible() is True

    foreign = model2.setup(torch.tensor([1, 8, 1]))
    swapped_h0 = replace(
        sys,
        h0_setup=foreign.h0_setup,
        exact_core=sys.exact_core.replace(h0_setup=foreign.h0_setup),
    )
    assert swapped_h0._has_intact_setup_token() is False
    assert swapped_h0._ihelp_compatible() is False
    with pytest.raises(ValueError, match="IndexHelper is incompatible"):
        swapped_h0.singlepoint(positions)

    par1 = ParamModule(GFN1_XTB, dtype=torch.float64)
    model1 = Model(
        par=par1,
        config=Config.create(
            method=labels.GFN1_XTB,
            int_driver=labels.INTDRIVER_PYTORCH,
            maxiter=8,
        ),
    )
    sys_gfn1 = model1.setup(numbers.clone())
    assert sys_gfn1.h0_setup is not None
    orig_val = sys_gfn1.h0_setup.valence.clone()
    sys_gfn1.h0_setup.valence.zero_()
    try:
        assert sys_gfn1._has_intact_setup_token() is False
        assert sys_gfn1._ihelp_compatible() is False
        no_tok_gfn1 = replace(sys_gfn1, _setup_index_token=None)
        assert no_tok_gfn1._ihelp_compatible() is False
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            no_tok_gfn1.singlepoint(positions)
    finally:
        sys_gfn1.h0_setup.valence.copy_(orig_val)
    assert sys_gfn1._ihelp_compatible() is True

    # 3. [MAJOR-1] Strict int64 dtypes in PairClass/PairPlan and mixed-dtype IndexHelper.
    assert sys.integral_setup is not None
    pt_setup = sys.integral_setup.pytorch
    assert pt_setup is not None
    cls0 = pt_setup.pair_plan.classes[0]
    with pytest.raises(TypeError, match="torch.int64"):
        PairClass(
            ub=cls0.ub,
            uk=cls0.uk,
            lb=cls0.lb,
            lk=cls0.lk,
            ib=cls0.ib.to(torch.int32),
            jk=cls0.jk.to(torch.int32),
            idx=cls0.idx.to(torch.int32),
            weight=cls0.weight,
        )

    plan_leaves, plan_spec = tree_flatten(pt_setup.pair_plan)
    int32_plan_leaves = [
        leaf.to(torch.int32) if leaf.dtype is torch.int64 else leaf
        for leaf in plan_leaves
    ]
    unflat_int32_plan = tree_unflatten(int32_plan_leaves, plan_spec)
    bad_pt_is = sys.integral_setup.replace(
        backend=pt_setup.replace(pair_plan=unflat_int32_plan)
    )
    bad_plan_sys = replace(
        sys,
        integral_setup=bad_pt_is,
        exact_core=sys.exact_core.replace(integral=bad_pt_is),
        _setup_index_token=None,
    )
    assert bad_plan_sys._ihelp_compatible() is False
    assert bad_plan_sys._active_exact_core() is None

    ih_leaves, ih_spec = tree_flatten(sys.ihelp)
    mixed_ih_leaves = [
        ih_leaves[0],
        *(leaf.to(torch.int32) for leaf in ih_leaves[1:]),
    ]
    mixed_ihelp = tree_unflatten(mixed_ih_leaves, ih_spec)
    assert replace(sys, ihelp=mixed_ihelp)._ihelp_compatible() is False

    # 4. [MAJOR-2] Token drops on replaced ihelp; equivalent ihelp aligns on conversion
    # and runs exactly one _index_helpers_equal (12 field checks) inside _active_exact_core().
    import dxtb._src.calculators.model as model_mod

    fresh_sys = model2.setup(numbers.clone())
    assert fresh_sys._has_intact_setup_token() is True
    assert fresh_sys.exact_core is not None

    permuted_ihelp = IndexHelper.from_numbers(torch.tensor([1, 8, 1]), par2)
    corrupted_ihelp = replace(fresh_sys, ihelp=permuted_ihelp)
    assert corrupted_ihelp._has_intact_setup_token() is False
    assert (
        _convert_system_dtype(corrupted_ihelp, torch.float32)._setup_index_token
        is None
    )

    equiv_ihelp = IndexHelper.from_numbers(numbers.clone(), par2)
    equiv_sys = replace(fresh_sys, ihelp=equiv_ihelp)
    assert equiv_sys._has_intact_setup_token() is False
    assert equiv_sys._setup_token_without_ihelp_intact() is True
    conv_equiv = _convert_system_dtype(equiv_sys, torch.float32)
    assert conv_equiv.exact_core is not None
    assert conv_equiv.exact_core.index is equiv_ihelp

    ih_eq_calls = 0
    tensor_eq_calls = 0
    real_ih_equal = model_mod._index_helpers_equal
    real_equal = torch.equal

    def count_ih_equal(a: IndexHelper, b: IndexHelper) -> bool:
        nonlocal ih_eq_calls
        ih_eq_calls += 1
        return real_ih_equal(a, b)

    def count_equal(a: Tensor, b: Tensor) -> bool:
        nonlocal tensor_eq_calls
        tensor_eq_calls += 1
        return real_equal(a, b)

    monkeypatch.setattr(model_mod, "_index_helpers_equal", count_ih_equal)
    monkeypatch.setattr(torch, "equal", count_equal)
    assert equiv_sys._active_exact_core() is equiv_sys.exact_core
    monkeypatch.undo()
    assert ih_eq_calls == 1
    assert tensor_eq_calls == 1

    # 5. [MAJOR-3] batch_implicit checks positions device before fallback ihelp ValueError.
    with pytest.raises(DeviceError, match="Device mismatch"):
        _validate_system_and_positions(
            corrupted_ihelp,
            positions.unsqueeze(0).to(device="meta"),
            None,
            None,
        )

    # 6. [BLOCKER-3] torch.compile preserves structural-tampering rejection & core deactivation.
    if torch._dynamo.is_dynamo_supported():
        from dxtb._src.calculators.singlepoint import build_integral_matrices

        torch._dynamo.reset()

        compile_sys = model2.setup(numbers.clone())
        assert compile_sys.h0_setup is not None
        assert compile_sys.integral_setup is not None
        assert compile_sys.exact_core is not None
        chrg0 = torch.tensor(
            0.0, dtype=positions.dtype, device=positions.device
        )
        ref_ovlp = build_integral_matrices(
            compile_sys.integral_setup,
            compile_sys.h0_setup,
            compile_sys.ihelp,
            positions,
            chrg0,
        )[0].overlap

        def eval_intact_compiled(p: Tensor) -> Tensor:
            """Validate System and evaluate overlap under torch.compile."""
            _validate_system_and_positions(
                compile_sys, p.unsqueeze(0), None, None
            )
            assert compile_sys._active_exact_core() is compile_sys.exact_core
            assert compile_sys.integral_setup is not None
            assert compile_sys.h0_setup is not None
            return build_integral_matrices(
                compile_sys.integral_setup,
                compile_sys.h0_setup,
                compile_sys.ihelp,
                p,
                chrg0,
            )[0].overlap

        compiled_intact = torch.compile(
            eval_intact_compiled, backend="aot_eager"
        )
        torch.testing.assert_close(
            compiled_intact(positions),
            ref_ovlp,
            atol=1e-12,
            rtol=1e-12,
        )
        # In-place structural mutation AFTER compilation is still caught via real _version!
        compile_sys.h0_setup.numbers[0] = 1
        compile_sys.h0_setup.numbers[1] = 8
        try:
            with pytest.raises(ValueError, match="IndexHelper is incompatible"):
                compiled_intact(positions)
        finally:
            compile_sys.h0_setup.numbers[0] = 8
            compile_sys.h0_setup.numbers[1] = 1

        corrupted_no_core = replace(
            fresh_sys, ihelp=permuted_ihelp, exact_core=None
        )

        def eval_corrupted_ihelp(p: Tensor) -> Tensor:
            """Validate tampered System under torch.compile."""
            _validate_system_and_positions(
                corrupted_no_core, p.unsqueeze(0), None, None
            )
            return p * 2.0

        compiled_bad = torch.compile(eval_corrupted_ihelp, backend="aot_eager")
        with pytest.raises(ValueError, match="IndexHelper is incompatible"):
            compiled_bad(positions)

        corrupted_core_idx = replace(
            fresh_sys,
            exact_core=fresh_sys.exact_core.replace(index=permuted_ihelp),
        )
        assert corrupted_core_idx._has_intact_setup_token() is True

        def eval_corrupted_core(p: Tensor) -> Tensor:
            """Verify tampered ExactSystemCore.index deactivates core under torch.compile."""
            assert corrupted_core_idx._active_exact_core() is None
            assert corrupted_core_idx.integral_setup is not None
            assert corrupted_core_idx.h0_setup is not None
            return build_integral_matrices(
                corrupted_core_idx.integral_setup,
                corrupted_core_idx.h0_setup,
                corrupted_core_idx.ihelp,
                p,
                chrg0,
            )[0].overlap

        compiled_core_fallback = torch.compile(
            eval_corrupted_core, backend="aot_eager"
        )
        torch.testing.assert_close(
            compiled_core_fallback(positions),
            ref_ovlp,
            atol=1e-12,
            rtol=1e-12,
        )
