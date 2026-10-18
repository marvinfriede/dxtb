"""Core setup and ownership tests for the exact single-System D3 path."""

from __future__ import annotations

import pytest
import tad_dftd3 as d3
import torch
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN1_XTB, Calculator
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    ClassicalList,
    DispersionD3CoreSetup,
    new_repulsion,
)
from dxtb._src.components.classicals.dispersion import (
    DispersionD3Cache,
    core_dispersion_d3_setup,
    dispersion_d3_energy,
    new_dispersion,
)
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.ncoord import exp_count
from dxtb._src.param import ParamModule

from .samples import samples


def _component(numbers: torch.Tensor):
    component = new_dispersion(numbers, GFN1_XTB, dtype=torch.double)
    assert component is not None
    return component


def _core(numbers: torch.Tensor, setup_numbers: torch.Tensor | None = None):
    component = _component(numbers)
    gathered_numbers = numbers if setup_numbers is None else setup_numbers
    cache = component.get_cache(gathered_numbers)
    assert isinstance(cache, DispersionD3Cache)
    return (
        component,
        cache,
        core_dispersion_d3_setup(
            cache, gathered_numbers, component.numbers, component.param
        ),
    )


def test_duplicate_canonical_d3_labels_are_rejected() -> None:
    numbers = torch.tensor([6, 1], dtype=torch.int64)
    built_in = _component(numbers)
    replacement = _component(numbers)
    with pytest.raises(ValueError, match="Duplicate DispersionD3 labels"):
        ClassicalList(built_in, replacement)

    from dxtb.config import Config

    with pytest.raises(ValueError, match="Duplicate DispersionD3 labels"):
        Model(
            par=ParamModule(GFN1_XTB),
            config=Config.create(exclude=("scf",)),
            classical=(replacement,),
        ).setup(numbers)

    first = _component(numbers)
    second = _component(numbers)
    first.param["s6"] = torch.tensor(0.1, dtype=torch.double)
    second.param["s6"] = torch.tensor(0.9, dtype=torch.double)
    with pytest.raises(ValueError, match="Duplicate DispersionD3 labels"):
        ClassicalList(first, second)

    first.label = "D3:custom"
    second.label = "D3:custom"
    with pytest.raises(ValueError, match="D3:custom"):
        ClassicalList(first, second)

    repeated = ClassicalList(first, first)
    assert repeated.components == [first]

    first.label = "D3:first"
    second.label = "D3:second"
    distinct_labels = ClassicalList(first, second)
    assert distinct_labels.components == [first, second]


def test_core_schema_and_pytree_roundtrip() -> None:
    numbers = samples["C4H5NCS"]["numbers"]
    component, cache, core = _core(numbers)
    assert type(core) is DispersionD3CoreSetup
    assert core.setup_numbers is numbers
    assert core.component_numbers is component.numbers
    assert core.reference_cn is cache.ref.cn
    assert core.reference_c6 is cache.ref.c6
    for name in ("rcov", "rvdw", "r4r2", "cutoff"):
        assert getattr(core, name) is getattr(cache, name)
    assert tuple(name for name, _ in core.parameters) == tuple(component.param)
    for (name, value), (_, original) in zip(
        core.parameters, component.param.items(), strict=True
    ):
        assert name in component.param
        if isinstance(value, torch.Tensor):
            assert value is original
        else:
            assert value == original

    leaves, spec = tree_flatten(core)
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    assert any(leaf is core.setup_numbers for leaf in leaves)
    assert any(leaf is core.component_numbers for leaf in leaves)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is DispersionD3CoreSetup
    assert rebuilt.counting_function is core.counting_function
    assert rebuilt.weighting_function is core.weighting_function
    assert rebuilt.damping_function is core.damping_function
    for name in (
        "setup_numbers",
        "component_numbers",
        "reference_cn",
        "reference_c6",
        "rcov",
        "rvdw",
        "r4r2",
        "cutoff",
    ):
        assert getattr(rebuilt, name) is getattr(core, name)


def test_core_conversion_and_energy_preserve_legacy_numbers_contract() -> None:
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    alternate_setup_numbers = numbers.roll(1)
    component, legacy, core = _core(numbers, alternate_setup_numbers)

    assert not torch.equal(core.setup_numbers, core.component_numbers)
    expected = component.get_energy(positions, legacy)
    actual = dispersion_d3_energy(core, positions)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        dispersion_d3_energy(core, positions, chunk_size=1),
        component.get_energy(positions, legacy, chunk_size=1),
    )


def test_core_dtype_conversion_and_transform_probe() -> None:
    numbers = samples["LiH"]["numbers"]
    _, _, core = _core(numbers)
    converted = core.to(dtype=torch.float32)
    assert converted.component_numbers.dtype == numbers.dtype
    assert converted.reference_cn.dtype == torch.float32
    assert converted.reference_c6.dtype == torch.float32
    assert converted.rcov.dtype == torch.float32
    assert converted.rvdw.dtype == torch.float32
    assert converted.r4r2.dtype == torch.float32
    assert converted.cutoff.dtype == torch.float32
    assert core.rcov.dtype == torch.double

    stacked = stack((core, core))
    result = torch.func.vmap(lambda item: item.rcov.sum())(stacked)
    expected = torch.stack((core.rcov.sum(), core.rcov.sum()))
    torch.testing.assert_close(result, expected)

    meta = core.to(device="meta")
    meta_leaves, _ = tree_flatten(meta)
    assert all(leaf.device.type == "meta" for leaf in meta_leaves)
    assert meta.component_numbers.dtype == numbers.dtype


def test_parameter_value_types_and_sparse_mapping_are_preserved() -> None:
    numbers = samples["LiH"]["numbers"]
    component = _component(numbers)
    sparse = {"s6": torch.tensor(1.0, dtype=torch.double)}
    cache = component.get_cache(numbers)
    core = core_dispersion_d3_setup(cache, numbers, numbers, sparse)
    assert tuple(name for name, _ in core.parameters) == ("s6",)
    assert core.parameters[0][1] is sparse["s6"]

    mixed = core_dispersion_d3_setup(
        cache,
        numbers,
        numbers,
        {
            "tensor": torch.tensor(1, dtype=torch.int64),
            "bool_tensor": torch.tensor(True),
            "integer": 3,
            "boolean": True,
            "floating": 2.5,
            "string": "kept",
        },
    )
    moved = mixed.to(dtype=torch.float32)
    values = dict(moved.parameters)
    assert values["tensor"].dtype == torch.int64
    assert values["bool_tensor"].dtype == torch.bool
    assert type(values["integer"]) is int
    assert type(values["boolean"]) is bool
    assert type(values["floating"]) is float
    assert type(values["string"]) is str


def test_sparse_d3_parameters_preserve_energy() -> None:
    numbers = samples["LiH"]["numbers"]
    positions = samples["LiH"]["positions"].double()
    sparse = {
        "s6": torch.tensor(1.0, dtype=torch.double),
        "a1": torch.tensor(0.49484001, dtype=torch.double),
        "a2": torch.tensor(5.73083694, dtype=torch.double),
    }
    component = DispersionD3(numbers, sparse, dtype=torch.double)
    legacy = component.get_cache(numbers)
    core = core_dispersion_d3_setup(
        legacy, numbers, component.numbers, component.param
    )
    assert tuple(name for name, _ in core.parameters) == tuple(sparse)
    assert core.as_param().keys() == sparse.keys()
    torch.testing.assert_close(
        dispersion_d3_energy(core, positions),
        component.get_energy(positions, legacy),
    )


def test_custom_cache_callables_remain_legacy_compatible() -> None:
    numbers = samples["LiH"]["numbers"]
    positions = samples["LiH"]["positions"].double()

    def count(*args, **kwargs):
        return exp_count(*args, **kwargs)

    def weighting(*args, **kwargs):
        return d3.model.gaussian_weight(*args, **kwargs)

    def damping(*args, **kwargs):
        return d3.damping.rational_damping(*args, **kwargs)

    component = _component(numbers)
    cache = component.get_cache(
        numbers,
        counting_function=count,
        weighting_function=weighting,
        damping_function=damping,
    )
    assert isinstance(cache, DispersionD3Cache)
    assert cache.counting_function is count
    assert cache.weighting_function is weighting
    assert cache.damping_function is damping
    result = component.get_energy(positions, cache)
    assert torch.isfinite(result).all()
    with pytest.raises(TypeError, match="built-in stateless exp_count"):
        core_dispersion_d3_setup(
            cache, numbers, component.numbers, component.param
        )


def test_legacy_and_core_energy_position_and_parameter_gradients_match() -> (
    None
):
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    from dxtb._src.param import ParamModule

    par = ParamModule(GFN1_XTB, dtype=torch.double)
    par.set_differentiable("dispersion", "d3", "s6")
    component = new_dispersion(numbers, par, dtype=torch.double)
    assert component is not None
    legacy = component.get_cache(numbers)
    core = core_dispersion_d3_setup(
        legacy, numbers, component.numbers, component.param
    )
    parameter = par.get("dispersion.d3.s6")
    p_legacy = sample["positions"].double().clone().requires_grad_(True)
    e_legacy = component.get_energy(p_legacy, legacy).sum()
    grad_pos_legacy, grad_par_legacy = torch.autograd.grad(
        e_legacy, (p_legacy, parameter)
    )

    p_core = sample["positions"].double().clone().requires_grad_(True)
    e_core = component.get_energy(p_core, core).sum()
    grad_pos_core, grad_par_core = torch.autograd.grad(
        e_core, (p_core, parameter)
    )
    torch.testing.assert_close(e_core, e_legacy)
    torch.testing.assert_close(grad_pos_core, grad_pos_legacy)
    torch.testing.assert_close(grad_par_core, grad_par_legacy)

    p_legacy_mixed = sample["positions"].double().clone().requires_grad_(True)
    e_legacy_mixed = component.get_energy(p_legacy_mixed, legacy).sum()
    pos_grad_legacy = torch.autograd.grad(
        e_legacy_mixed, p_legacy_mixed, create_graph=True
    )[0]
    mixed_legacy = torch.autograd.grad(pos_grad_legacy[0, 0], parameter)[0]

    p_core_mixed = sample["positions"].double().clone().requires_grad_(True)
    e_core_mixed = component.get_energy(p_core_mixed, core).sum()
    pos_grad_core = torch.autograd.grad(
        e_core_mixed, p_core_mixed, create_graph=True
    )[0]
    mixed_core = torch.autograd.grad(pos_grad_core[0, 0], parameter)[0]
    assert torch.isfinite(mixed_core)
    torch.testing.assert_close(mixed_core, mixed_legacy)

    for chunk_size in (None, 1):
        torch.testing.assert_close(
            component.get_energy(
                sample["positions"].double(), core, chunk_size=chunk_size
            ),
            component.get_energy(
                sample["positions"].double(), legacy, chunk_size=chunk_size
            ),
        )
    with pytest.raises(TypeError, match="DispersionD3Cache"):
        component.get_energy(sample["positions"].double(), object())


def test_model_setup_owns_core_and_singlepoint_uses_it() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude="scf"),
    )
    system = model.setup(numbers)
    d3_components = [
        c
        for c in system.classicals.components
        if type(c).__name__ == "DispersionD3"
    ]
    assert len(d3_components) == 1
    component = d3_components[0]
    setups = [
        setup for selected, setup in system.d3_setups if selected is component
    ]
    assert len(setups) == 1
    setup = setups[0]
    assert type(setup) is DispersionD3CoreSetup
    assert system.d3_setup is setup
    assert system.classical_cache[component.label] is setup
    result = system.singlepoint(positions)
    contributions = dict(result.classical)
    assert component.label in contributions
    torch.testing.assert_close(
        contributions[component.label], dispersion_d3_energy(setup, positions)
    )


def test_system_d3_setup_preserves_parameter_graph_and_snapshot() -> None:
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    par = ParamModule(GFN1_XTB, dtype=torch.double)
    par.set_differentiable("dispersion", "d3", "s6")
    system = Model(par=par, config=Config.create(exclude="scf")).setup(numbers)
    component, setup = system.d3_setups[0]
    assert type(component) is DispersionD3
    before = dispersion_d3_energy(setup, positions).sum()
    leaf = par.get("dispersion.d3.s6")
    (gradient,) = torch.autograd.grad(before, leaf, retain_graph=True)
    assert torch.isfinite(gradient)

    original = setup.as_param()["s6"]
    component.param["s6"] = torch.full_like(original, 0.25)
    component.numbers = torch.ones_like(component.numbers)
    after = dispersion_d3_energy(setup, positions).sum()
    torch.testing.assert_close(after, before)
    assert setup.as_param()["s6"] is not component.param["s6"]


def test_system_d3_geometry_history_is_independent() -> None:
    """Verify A/B/A evaluations preserve setup and position gradients."""
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions_a = sample["positions"].double()
    positions_b = positions_a.clone()
    positions_b[1, 2] += 0.2
    from dxtb.config import Config

    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude="scf"),
    ).setup(numbers)
    setup = system.d3_setup
    assert setup is not None
    position_a_first = positions_a.clone().requires_grad_(True)
    first = dict(system.singlepoint(position_a_first).classical)["DispersionD3"]
    first_gradient = torch.autograd.grad(first.sum(), position_a_first)[0]
    position_b = positions_b.clone().requires_grad_(True)
    middle = dict(system.singlepoint(position_b).classical)["DispersionD3"]
    torch.autograd.grad(middle.sum(), position_b)
    position_a_last = positions_a.clone().requires_grad_(True)
    last = dict(system.singlepoint(position_a_last).classical)["DispersionD3"]
    last_gradient = torch.autograd.grad(last.sum(), position_a_last)[0]
    assert system.d3_setup is setup
    torch.testing.assert_close(first, last)
    torch.testing.assert_close(first_gradient, last_gradient)


def test_classical_list_evaluates_system_owned_d3_core() -> None:
    """Verify the classical-list compatibility API accepts the System Node."""
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude="scf"),
    ).setup(numbers)
    assert len(system.classicals.components) > 1
    d3_component = next(
        component
        for component in system.classicals.components
        if type(component) is DispersionD3
    )
    core = system.classical_cache[d3_component.label]
    assert type(core) is DispersionD3CoreSetup

    charge = torch.tensor(0.0, dtype=torch.double)
    contributions = system.classicals.get_energy(
        positions, system.classical_cache, charge=charge
    )
    result = system.singlepoint(positions, chrg=charge)
    result_contributions = dict(result.classical)
    assert contributions.keys() == result_contributions.keys()
    for label, energy in contributions.items():
        torch.testing.assert_close(energy, result_contributions[label])
    torch.testing.assert_close(
        contributions[d3_component.label], dispersion_d3_energy(core, positions)
    )
    for chunk_size in (None, 1):
        torch.testing.assert_close(
            d3_component.get_energy(positions, core, chunk_size=chunk_size),
            dispersion_d3_energy(core, positions, chunk_size=chunk_size),
        )


def test_core_cache_snapshot_and_legacy_component_ownership_diverge() -> None:
    """Verify core cache authority survives component mutation."""
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude="scf"),
    ).setup(numbers)
    component, core = system.d3_setups[0]
    legacy = component.get_cache(numbers)
    cache = system.classical_cache
    before_core = system.classicals.get_energy(positions, cache)[
        component.label
    ]
    before_legacy = component.get_energy(positions, legacy)

    component.param["s6"] = torch.full_like(component.param["s6"], 0.25)
    component.numbers = numbers.flip(0)

    after_core = system.classicals.get_energy(positions, cache)[component.label]
    after_legacy = component.get_energy(positions, legacy)
    torch.testing.assert_close(after_core, before_core)
    assert not torch.allclose(after_legacy, before_legacy)
    torch.testing.assert_close(
        after_core, dispersion_d3_energy(core, positions)
    )


def test_d3_custom_label_validation_preserves_extension_collisions() -> None:
    """Keep non-exact D3 and cross-family labels on existing policy."""
    numbers = samples["LiH"]["numbers"]

    class D3Extension(DispersionD3):
        pass

    exact = _component(numbers)
    exact.label = "D3:shared"
    extension = D3Extension(numbers, exact.param, dtype=torch.double)
    extension.label = exact.label
    assert ClassicalList(exact, extension).components == [exact, extension]

    other = _component(numbers)
    other.label = "D3:extension-only"
    extension2 = D3Extension(numbers, other.param, dtype=torch.double)
    extension2.label = other.label
    assert ClassicalList(other, extension2).components == [other, extension2]

    first_extension = D3Extension(numbers, exact.param, dtype=torch.double)
    second_extension = D3Extension(numbers, exact.param, dtype=torch.double)
    first_extension.label = second_extension.label = "D3:two-extensions"
    assert ClassicalList(first_extension, second_extension).components == [
        first_extension,
        second_extension,
    ]

    repulsion = new_repulsion(
        numbers.unique(),
        ParamModule(GFN1_XTB, dtype=torch.double),
        dtype=torch.double,
    )
    assert repulsion is not None
    repulsion.label = "D3:cross-family"
    cross_family = _component(numbers)
    cross_family.label = repulsion.label
    assert ClassicalList(cross_family, repulsion).components == [
        cross_family,
        repulsion,
    ]


def test_legacy_cache_api_and_distinct_custom_labels() -> None:
    numbers = samples["LiH"]["numbers"]
    first = _component(numbers)
    second = _component(numbers)
    first.label = "D3:first"
    second.label = "D3:second"
    classicals = ClassicalList(first, second)
    caches = classicals.get_cache(numbers, None)  # type: ignore[arg-type]
    assert isinstance(caches[first.label], DispersionD3Cache)
    assert isinstance(caches[second.label], DispersionD3Cache)


def test_exact_custom_d3_replacement_is_captured() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    replacement = _component(numbers)
    replacement.label = "D3:replacement"
    replacement.param["s6"] = torch.tensor(0.3, dtype=torch.double)
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude=("disp", "scf")),
        classical=(replacement,),
    )
    system = model.setup(numbers)
    assert len(system.d3_setups) == 1
    selected, setup = system.d3_setups[0]
    assert selected is replacement
    assert setup.as_param()["s6"].item() == pytest.approx(0.3)
    result = system.singlepoint(positions)
    contribution = dict(result.classical)[replacement.label]
    torch.testing.assert_close(
        contribution, dispersion_d3_energy(setup, positions)
    )


def test_multiple_distinct_custom_d3_labels_keep_order_and_setups() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    first = _component(numbers)
    first.label = "D3:first"
    second = _component(numbers)
    second.label = "D3:second"
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude=("disp", "scf")),
        classical=(first, second),
    ).setup(numbers)
    selected = tuple(
        c.label
        for c in system.classicals.components
        if c is first or c is second
    )
    assert selected == (first.label, second.label)
    assert len(system.d3_setups) == 2
    contributions = dict(system.singlepoint(positions).classical)
    for component, setup in system.d3_setups:
        torch.testing.assert_close(
            contributions[component.label],
            dispersion_d3_energy(setup, positions),
        )


def test_d3_reset_preserves_single_system_setup_and_legacy_batch_energy() -> (
    None
):
    """D3 reset clones component numbers but preserves captured System data."""
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb._src.components.classicals import ClassicalList

    par = ParamModule(GFN1_XTB, dtype=torch.double)
    par.set_differentiable("dispersion", "d3", "s6")
    calc = Calculator(
        numbers,
        par,
        dtype=torch.double,
        opts={"verbosity": 0, "exclude": ["scf", "rep", "hal", "ies", "srb"]},
    )
    system = calc.system
    component, setup = system.d3_setups[0]
    assert type(component) is DispersionD3
    original_numbers = component.numbers
    original_setup = setup
    parameter = par.get("dispersion.d3.s6")

    def single_system_observables():
        current_positions = positions.clone().requires_grad_(True)
        result = system.singlepoint(current_positions)
        energy = dict(result.classical)[component.label].sum()
        position_gradient, parameter_gradient = torch.autograd.grad(
            energy, (current_positions, parameter)
        )
        return (
            energy.detach(),
            position_gradient.detach(),
            parameter_gradient.detach(),
        )

    before = single_system_observables()
    calc.reset()
    after = single_system_observables()

    assert component.numbers is not original_numbers
    assert torch.equal(component.numbers, original_numbers)
    assert system.d3_setups[0][1] is original_setup
    torch.testing.assert_close(after[0], before[0])
    torch.testing.assert_close(after[1], before[1])
    torch.testing.assert_close(after[2], before[2])
    assert torch.isfinite(after[2]).all()
    assert torch.count_nonzero(after[2]) > 0

    packed_numbers = torch.stack((numbers, numbers))
    packed_positions = torch.stack((positions, positions))
    packed_par = ParamModule(GFN1_XTB, dtype=torch.double)
    packed_component = new_dispersion(
        packed_numbers, packed_par, dtype=torch.double
    )
    assert type(packed_component) is DispersionD3
    classicals = ClassicalList(packed_component)
    from dxtb._src.basis.indexhelper import IndexHelper

    ihelp = IndexHelper.from_numbers(packed_numbers, packed_par, batch_mode=1)
    packed_cache = classicals.get_cache(packed_numbers, ihelp)
    packed_before = classicals.get_energy(packed_positions, packed_cache)[
        packed_component.label
    ]
    packed_original_numbers = packed_component.numbers
    classicals.reset_all()
    packed_after = classicals.get_energy(packed_positions, packed_cache)[
        packed_component.label
    ]
    assert packed_component.numbers is not packed_original_numbers
    assert torch.equal(packed_component.numbers, packed_original_numbers)
    torch.testing.assert_close(packed_after, packed_before)


def test_normal_singlepoint_uses_captured_d3_core() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude=("rep", "hal", "ies", "srb")),
    ).setup(numbers)
    setup = system.d3_setup
    assert setup is not None
    result = system.singlepoint(positions)
    assert result.scf is not None
    assert "DispersionD3" in dict(result.classical)
    torch.testing.assert_close(
        dict(result.classical)["DispersionD3"],
        dispersion_d3_energy(setup, positions),
    )


def test_d3_subclass_remains_on_legacy_cache_route() -> None:
    class D3Extension(DispersionD3):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.energy_calls = 0

        def get_energy(self, positions, cache, **kwargs):
            self.energy_calls += 1
            return super().get_energy(positions, cache, **kwargs)

    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    base = _component(numbers)
    extension = D3Extension(numbers, base.param, dtype=torch.double)
    extension.label = "D3:extension"
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude=("disp", "scf")),
        classical=(extension,),
    ).setup(numbers)
    assert system.d3_setups == ()
    assert isinstance(
        system.classical_cache[extension.label], DispersionD3Cache
    )
    system.singlepoint(positions)
    assert extension.energy_calls == 1
