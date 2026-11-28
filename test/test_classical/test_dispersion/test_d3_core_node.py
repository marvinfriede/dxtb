"""Core setup and ownership tests for the exact single-System D3 path."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.data import radii
from tad_mctc.ncoord import exp_count
from tad_mctc.tree import stack
from torch.utils._pytree import tree_flatten, tree_unflatten

import tad_dftd as d3
from dxtb import GFN1_XTB, Calculator
from dxtb._src.calculators.model import Model
from dxtb._src.components.classicals import (
    ClassicalList,
    DispersionD3Setup,
    new_repulsion,
)
from dxtb._src.components.classicals.dispersion import (
    DispersionD3Setup,
    dispersion_d3_energy,
    new_dispersion,
)
from dxtb._src.components.classicals.dispersion.d3 import DispersionD3
from dxtb._src.param import ParamModule

from .samples import samples


def test_local_d3_cutoffs_match_supported_legacy_pair_and_atm_limits() -> None:
    """The local D3 adapter keeps its shared 50-bohr pre-migration limit."""
    numbers = torch.tensor([8, 1, 1], dtype=torch.int64)
    component = _component(numbers)
    component = component.replace(
        param={**component.param, "s9": torch.tensor(1.0, dtype=torch.double)}
    )
    cache = core = component.get_cache(numbers)
    parameters = core.param
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [45.0, 0.0, 0.0], [22.5, 38.97114317, 0.0]],
        dtype=torch.double,
    ).requires_grad_(True)

    cn = d3.ncoord.coordination_number(
        numbers,
        positions,
        counting_function=cache.counting_function,
        rcov=cache.rcov,
        cutoff=cache.cutoff,
    )
    weights = d3.model.weight_references(numbers, cn, cache.ref)
    c6 = d3.model.atomic_c6(numbers, weights, cache.ref)
    cutoff_50 = d3.Cutoff(disp2=50.0, disp3=50.0)
    expected = d3.disp.dispersion(
        numbers,
        positions,
        parameters,
        c6,
        cache.rvdw,
        cache.r4r2,
        cache.damping_function,
        cutoff=cutoff_50,
    )
    actual = dispersion_d3_energy(core, positions)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-24)
    actual_gradient = torch.autograd.grad(actual.sum(), positions)[0]
    expected_gradient = torch.autograd.grad(expected.sum(), positions)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-24
    )

    narrower_positions = positions.detach().clone().requires_grad_(True)
    cutoff_40 = d3.Cutoff(disp2=40.0, disp3=40.0)
    narrower = d3.disp.dispersion(
        numbers,
        narrower_positions,
        parameters,
        c6,
        cache.rvdw,
        cache.r4r2,
        cache.damping_function,
        cutoff=cutoff_40,
    )
    assert not torch.equal(actual, narrower)
    narrower_gradient = torch.autograd.grad(narrower.sum(), narrower_positions)[
        0
    ]
    assert not torch.equal(actual_gradient, narrower_gradient)

    custom_positions = positions.detach().clone().requires_grad_(True)
    custom_cutoff = d3.Cutoff(disp2=42.0, disp3=48.0)
    custom_combined = d3.disp.dispersion(
        numbers,
        custom_positions,
        parameters,
        c6,
        cache.rvdw,
        cache.r4r2,
        cache.damping_function,
        cutoff=custom_cutoff,
    )
    custom_split = d3.disp.dispersion2(
        numbers,
        custom_positions,
        parameters,
        c6,
        cache.r4r2,
        cache.damping_function,
        custom_cutoff.disp2,
    ) + d3.disp.dispersion3(
        numbers,
        custom_positions,
        parameters,
        c6,
        cache.rvdw,
        custom_cutoff.disp3,
    )
    torch.testing.assert_close(custom_combined, custom_split)
    custom_combined_gradient = torch.autograd.grad(
        custom_combined.sum(), custom_positions, retain_graph=True
    )[0]
    custom_split_gradient = torch.autograd.grad(
        custom_split.sum(), custom_positions
    )[0]
    torch.testing.assert_close(custom_combined_gradient, custom_split_gradient)


def _component(numbers: torch.Tensor):
    component = new_dispersion(numbers, GFN1_XTB, dtype=torch.double)
    assert component is not None
    return component


def _setup(numbers: torch.Tensor, setup_numbers: torch.Tensor | None = None):
    component = _component(numbers)
    gathered_numbers = numbers if setup_numbers is None else setup_numbers
    setup = component.get_cache(gathered_numbers)
    assert isinstance(setup, DispersionD3Setup)
    return component, setup


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

    first = first.replace(label="D3:custom")
    second = second.replace(label="D3:custom")
    with pytest.raises(ValueError, match="D3:custom"):
        ClassicalList(first, second)

    repeated = ClassicalList(first, first)
    assert repeated.components == [first]

    first = first.replace(label="D3:first")
    second = second.replace(label="D3:second")
    distinct_labels = ClassicalList(first, second)
    assert distinct_labels.components == [first, second]


def test_setup_schema_and_pytree_roundtrip() -> None:
    """The D3 setup holds the component numbers and parameters as leaves."""
    numbers = samples["C4H5NCS"]["numbers"]
    component, setup = _setup(numbers)
    assert torch.equal(setup.numbers, component.numbers)
    assert tuple(setup.param) == tuple(component.param)
    for name, value in setup.param.items():
        assert value is not component.param[name]
        assert torch.equal(value, component.param[name])

    leaves, spec = tree_flatten(setup)
    assert all(isinstance(leaf, torch.Tensor) for leaf in leaves)
    assert any(leaf is setup.numbers for leaf in leaves)
    rebuilt = tree_unflatten(leaves, spec)
    assert type(rebuilt) is DispersionD3Setup
    assert rebuilt.counting_function is setup.counting_function
    assert rebuilt.weighting_function is setup.weighting_function
    assert rebuilt.damping_function is setup.damping_function
    for name in ("numbers", "rcov", "rvdw", "r4r2", "cutoff"):
        assert getattr(rebuilt, name) is getattr(setup, name)
    assert rebuilt.ref.cn is setup.ref.cn


def test_setup_gathers_tables_from_setup_numbers() -> None:
    """Tables follow the numbers given to ``get_cache``; kernels the component's."""
    sample = samples["C4H5NCS"]
    numbers = sample["numbers"]
    alternate_setup_numbers = numbers.roll(1)
    component, setup = _setup(numbers, alternate_setup_numbers)

    assert torch.equal(setup.numbers, component.numbers)
    torch.testing.assert_close(
        setup.rcov, radii.COV_D3(dtype=torch.double)[alternate_setup_numbers]
    )
    energy = dispersion_d3_energy(setup, sample["positions"].double())
    assert torch.isfinite(energy).all()


def test_setup_dtype_conversion_and_transform_probe() -> None:
    """Conversion casts floating leaves only; stacked setups can be vmapped."""
    numbers = samples["LiH"]["numbers"]
    _, core = _setup(numbers)
    converted = core.to(dtype=torch.float32)
    assert converted.numbers.dtype == numbers.dtype
    assert converted.ref.cn.dtype == torch.float32
    assert converted.ref.c6.dtype == torch.float32
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
    assert meta.numbers.dtype == numbers.dtype


def test_sparse_d3_parameters_are_kept() -> None:
    """A component with a sparse parameter mapping keeps exactly its keys."""
    numbers = samples["LiH"]["numbers"]
    positions = samples["LiH"]["positions"].double()
    sparse = {
        "s6": torch.tensor(1.0, dtype=torch.double),
        "a1": torch.tensor(0.49484001, dtype=torch.double),
        "a2": torch.tensor(5.73083694, dtype=torch.double),
    }
    component = DispersionD3(numbers=numbers, param=sparse).to(
        dtype=torch.double
    )
    setup = component.get_cache(numbers)
    assert setup.param.keys() == sparse.keys()
    assert torch.isfinite(component.get_energy(positions, setup)).all()


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
    assert isinstance(cache, DispersionD3Setup)
    assert cache.counting_function is count
    assert cache.weighting_function is weighting
    assert cache.damping_function is damping
    result = component.get_energy(positions, cache)
    assert torch.isfinite(result).all()


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
    legacy = core = component.get_cache(numbers)
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

    torch.testing.assert_close(
        component.get_energy(sample["positions"].double(), core),
        component.get_energy(sample["positions"].double(), legacy),
    )
    with pytest.raises(TypeError, match="DispersionD3Setup"):
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
    assert type(setup) is DispersionD3Setup
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

    original = setup.param["s6"]
    component.param["s6"] = torch.full_like(original, 0.25)
    component = component.replace(numbers=torch.ones_like(component.numbers))
    after = dispersion_d3_energy(setup, positions).sum()
    torch.testing.assert_close(after, before)
    assert setup.param["s6"] is not component.param["s6"]


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
    assert type(core) is DispersionD3Setup

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
    torch.testing.assert_close(
        d3_component.get_energy(positions, core),
        dispersion_d3_energy(core, positions),
    )


def test_d3_custom_label_validation_preserves_extension_collisions() -> None:
    """Keep non-exact D3 and cross-family labels on existing policy."""
    numbers = samples["LiH"]["numbers"]

    class D3Extension(DispersionD3):
        pass

    exact = _component(numbers)
    exact = exact.replace(label="D3:shared")
    extension = D3Extension(numbers=numbers, param=exact.param).to(
        dtype=torch.double
    )
    extension = extension.replace(label=exact.label)
    assert ClassicalList(exact, extension).components == [exact, extension]

    other = _component(numbers)
    other = other.replace(label="D3:extension-only")
    extension2 = D3Extension(numbers=numbers, param=other.param).to(
        dtype=torch.double
    )
    extension2 = extension2.replace(label=other.label)
    assert ClassicalList(other, extension2).components == [other, extension2]

    first_extension = D3Extension(numbers=numbers, param=exact.param).to(
        dtype=torch.double
    )
    second_extension = D3Extension(numbers=numbers, param=exact.param).to(
        dtype=torch.double
    )
    first_extension = first_extension.replace(label="D3:two-extensions")
    second_extension = second_extension.replace(label="D3:two-extensions")
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
    repulsion = repulsion.replace(label="D3:cross-family")
    cross_family = _component(numbers)
    cross_family = cross_family.replace(label=repulsion.label)
    assert ClassicalList(cross_family, repulsion).components == [
        cross_family,
        repulsion,
    ]


def test_legacy_cache_api_and_distinct_custom_labels() -> None:
    numbers = samples["LiH"]["numbers"]
    first = _component(numbers)
    second = _component(numbers)
    first = first.replace(label="D3:first")
    second = second.replace(label="D3:second")
    classicals = ClassicalList(first, second)
    caches = classicals.get_cache(numbers, None)  # type: ignore[arg-type]
    assert isinstance(caches[first.label], DispersionD3Setup)
    assert isinstance(caches[second.label], DispersionD3Setup)


def test_exact_custom_d3_replacement_is_captured() -> None:
    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    from dxtb.config import Config

    replacement = _component(numbers)
    replacement = replacement.replace(label="D3:replacement")
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
    assert setup.param["s6"].item() == pytest.approx(0.3)
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
    first = first.replace(label="D3:first")
    second = _component(numbers)
    second = second.replace(label="D3:second")
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


def test_d3_subclass_evaluates_via_custom_classical_path() -> None:
    """D3 subclasses evaluate through the custom Classical hook path."""
    from dxtb.config import Config

    energy_calls = []

    class D3Extension(DispersionD3):
        def get_energy(self, positions, cache, **kwargs):
            energy_calls.append(self.label)
            return super().get_energy(positions, cache, **kwargs)

    sample = samples["LiH"]
    numbers = sample["numbers"]
    positions = sample["positions"].double()
    base = _component(numbers)
    extension = D3Extension(numbers=numbers, param=base.param).to(
        dtype=torch.double
    )
    extension = extension.replace(label="D3:extension")
    system = Model(
        par=ParamModule(GFN1_XTB, dtype=torch.double),
        config=Config.create(exclude=("disp", "scf")),
        classical=(extension,),
    ).setup(numbers)
    assert system.d3_setups == ()
    assert isinstance(
        system.classical_cache[extension.label], DispersionD3Setup
    )
    system.singlepoint(positions)
    assert len(energy_calls) == 1
