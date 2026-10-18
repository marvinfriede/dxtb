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

import pytest
import torch
from tad_mctc.exceptions import DeviceError, DtypeError
from torch.func import jacfwd, jacrev, jvp, vmap
from torch.utils._python_dispatch import TorchDispatchMode

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, labels
from dxtb._src.basis.indexhelper import IndexHelper
from dxtb._src.calculators.model import Model
from dxtb._src.calculators.singlepoint import _interaction_data, singlepoint
from dxtb._src.integral.driver.pytorch.overlap import build_overlap
from dxtb._src.param import ParamModule
from dxtb._src.scf.functional.core import CoreIndexData
from dxtb._src.xtb.h0 import H0IndexNode
from dxtb.config import Config

from ..molecules import mols


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
    config = Config.create(int_driver=labels.INTDRIVER_PYTORCH)
    return Model(par=ParamModule(GFN1_XTB, dtype=dtype), config=config)


def test_single_and_packed_modes_keep_separate_ownership() -> None:
    """Only exact single Systems own core setup Nodes and a derived view."""
    numbers = torch.tensor([1, 1])
    model = _model()

    single = model.setup(numbers, batch_mode=0)
    assert single.batch_mode == 0
    assert single._packed_classical_cache is None
    assert single.repulsion_setup is not None
    assert type(single.repulsion_setup).__name__ == "RepulsionCoreSetup"
    assert single.classical_cache is not single.classical_cache
    assert single.classical_cache["Repulsion"] is single.repulsion_setup
    assert single.exact_core is not None
    assert single.exact_core.numbers is single.numbers
    assert single.exact_core.config is single.config
    assert single.exact_core.effective_dtype is torch.float64
    assert single.exact_core.integral_setup is single.integral_setup
    assert single.exact_core.h0_setup is single.h0_setup
    assert single.exact_core.index is single.integral_setup.pytorch.index
    assert not hasattr(single.exact_core, "batch_mode")
    assert not hasattr(single.exact_core, "dd")
    assert not hasattr(single.exact_core, "ihelp")
    assert not hasattr(single.exact_core, "classicals")
    assert not hasattr(single.exact_core, "interactions")
    assert [entry.label for entry in single.exact_core.classical_setups] == [
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
        setup = single.exact_core.classical_setup(component)
        if setup is not None:
            assert any(
                entry.component_identity == id(component)
                and entry.setup is setup
                for entry in single.exact_core.classical_setups
            )
    assert [entry.label for entry in single.exact_core.interaction_setups] == [
        component.label
        for component in single.interactions.components
        if single.exact_core.interaction_setup(component) is not None
    ]
    assert (
        single.exact_core.interaction_setup(single.es2_interaction)
        is single.es2_setup
    )
    assert (
        single.exact_core.interaction_setup(single.es3_interaction)
        is single.es3_setup
    )
    assert (
        single.exact_core.interaction_setup(single.d4sc_interaction)
        is single.d4sc_setup
    )

    packed_numbers = torch.tensor([[1, 1], [1, 1]])
    for mode in (1, 2):
        packed = model.setup(packed_numbers, batch_mode=mode)
        assert packed.batch_mode == mode
        assert packed.exact_core is None
        assert packed._packed_classical_cache is not None
        assert packed.classical_cache is packed._packed_classical_cache
        assert type(packed.repulsion_setup).__name__ == "RepulsionSetup"
        assert packed.integral_setup is None


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
    "method,method_label",
    [
        (GFN0_XTB, labels.GFN0_XTB),
        (GFN1_XTB, labels.GFN1_XTB),
        (GFN2_XTB, labels.GFN2_XTB),
    ],
    ids=["gfn0", "gfn1", "gfn2"],
)
@pytest.mark.parametrize("molecule", ["H2", "LiH", "H2O", "SiH4"])
def test_captured_index_maps_match_h0_and_legacy(
    method: str, method_label: int, molecule: str
) -> None:
    """Captured PyTorch and H0 maps preserve legacy index ordering."""
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
    pytorch_index = system.integral_setup.pytorch.index
    h0_index = system.h0_setup.index
    legacy = system.ihelp
    for name in ("shells_to_atom", "orbitals_to_shell"):
        expected = getattr(legacy, name)
        actual = getattr(pytorch_index, name)
        h0_actual = getattr(h0_index, name)
        assert actual.shape == expected.shape == h0_actual.shape
        assert actual.dtype == expected.dtype == h0_actual.dtype
        assert actual.device == expected.device == h0_actual.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(h0_actual, expected, rtol=0, atol=0)
        assert actual is not expected
        assert h0_actual is not expected
        assert (
            actual.untyped_storage().data_ptr()
            != expected.untyped_storage().data_ptr()
        )
        assert (
            h0_actual.untyped_storage().data_ptr()
            != expected.untyped_storage().data_ptr()
        )


def test_captured_libcint_index_maps_match_h0_and_legacy() -> None:
    """Libcint captures equivalent SCF maps on its configured backend."""
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
    libcint_index = system.integral_setup.libcint.index
    for name in ("shells_to_atom", "orbitals_to_shell"):
        actual = getattr(libcint_index, name)
        expected = getattr(system.ihelp, name)
        h0_actual = getattr(system.h0_setup.index, name)
        assert actual.shape == expected.shape == h0_actual.shape
        assert actual.dtype == expected.dtype == h0_actual.dtype
        assert actual.device == expected.device == h0_actual.device
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(h0_actual, expected, rtol=0, atol=0)
        assert (
            actual.untyped_storage().data_ptr()
            != expected.untyped_storage().data_ptr()
        )
        assert (
            h0_actual.untyped_storage().data_ptr()
            != expected.untyped_storage().data_ptr()
        )


def test_functional_singlepoint_reuses_captured_scf_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Eligible System SCF inputs reuse the active backend map Node."""
    from dxtb._src.scf.functional import data as functional_data
    from dxtb._src.scf.functional import eager

    system = _model().setup(torch.tensor([1, 1]))
    assert system.integral_setup is not None
    assert system.integral_setup.pytorch is not None
    expected_index = system.integral_setup.pytorch.index
    original = eager.setup_scf_inputs
    index_type = functional_data.CoreIndexData
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

    from dxtb._src.scf.functional import eager

    system = _model().setup(torch.tensor([1, 1]))
    scf = replace(
        system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        mixer=labels.MIXER_LINEAR,
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
    expected_index = system.integral_setup.pytorch.index
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
    result = system.singlepoint(positions)
    assert result.converged is not None and bool(result.converged)
    assert len(prepared) == 1
    assert prepared[0].index is expected_index


def test_direct_scf_input_preparation_keeps_isolated_map_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct functional setup remains independent of captured System maps."""
    from test.test_scf.test_functional_map import _prepare_case

    from dxtb._src.scf.functional.data import setup_scf_inputs

    case = _prepare_case(GFN1_XTB, setup_functional=False)
    legacy_maps = (
        case.system.ihelp.shells_to_atom,
        case.system.ihelp.orbitals_to_shell,
    )
    from dxtb._src.scf.functional import data as functional_data

    index_type = functional_data.CoreIndexData
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
    assert isinstance(inputs.index, CoreIndexData)
    assert inputs.index is not case.system.integral_setup.pytorch.index
    assert inputs.index.shells_to_atom is not legacy_maps[0]
    assert inputs.index.orbitals_to_shell is not legacy_maps[1]
    assert clone_counter.count == 2
    assert constructions == 1

    assert case.system.exact_core is not None
    unrelated_index = CoreIndexData(
        shells_to_atom=legacy_maps[0].clone(),
        orbitals_to_shell=legacy_maps[1].clone(),
    )
    with pytest.raises(ValueError, match="does not belong to this System"):
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
            core_index=unrelated_index,
            core_index_owner=case.system.exact_core,
        )


def test_scf_orbital_mask_and_conditioning_keep_indexhelper_padding() -> None:
    """The captured reduction maps do not replace padding-aware mask setup."""
    from test.test_scf.test_functional_map import _prepare_case

    from tad_mctc import storch

    case = _prepare_case(GFN1_XTB, sample_name="SiH4")
    expected_mask = case.system.ihelp.spread_shell_to_orbital(
        case.system.ihelp.orbitals_per_shell
    ).to(dtype=torch.bool)
    torch.testing.assert_close(case.inputs.orbital_mask, expected_mask)
    padding_diagonal = (~expected_mask).to(dtype=case.integrals.overlap.dtype)
    conditioned_overlap = case.integrals.overlap + torch.diag_embed(
        padding_diagonal
    )
    expected_inverse = storch.inv_cholesky_factor(
        conditioned_overlap, is_posdef=True
    )
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
    assert type(setup.pytorch).__name__ == "PytorchIntegralSetupNode"
    assert not hasattr(setup.pytorch, "ihelp")
    assert isinstance(system.h0_setup.ihelp, H0IndexNode)
    assert not isinstance(system.h0_setup.ihelp, IndexHelper)
    captured_shell_map = setup.pytorch.index.shells_to_atom
    legacy_shell_map = system.ihelp.shells_to_atom
    assert captured_shell_map is not legacy_shell_map
    captured_snapshot = captured_shell_map.clone()
    legacy_snapshot = legacy_shell_map.clone()
    try:
        with torch.no_grad():
            legacy_shell_map[0] = legacy_shell_map[0] + 1
        assert torch.equal(captured_shell_map, captured_snapshot)
    finally:
        with torch.no_grad():
            legacy_shell_map.copy_(legacy_snapshot)

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
