"""Private retained legacy composition acceptance controls."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.legacy_composition import (
    LegacySCFOptions,
    LegacySystemRoute,
    classify_legacy_system,
    compose_legacy_systems,
)
from dxtb._src.calculators.model import Model, System
from dxtb._src.components.classicals.base import Classical
from dxtb._src.components.interactions.base import (
    Interaction,
    InteractionCache,
)
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.field.efield import LABEL_EFIELD
from dxtb.config import Config

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _system(
    numbers: list[int],
    *,
    driver: int = labels.INTDRIVER_PYTORCH,
    classical: tuple[Classical, ...] = (),
    interaction: tuple[Interaction, ...] = (),
) -> System:
    config = Config.create(
        method="GFN1-xTB",
        int_driver=driver,
        int_level=labels.INTLEVEL_QUADRUPOLE,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE),
        config=config,
        classical=classical,
        interaction=interaction,
    )
    return model.setup(torch.tensor(numbers, device=DEVICE))


def _positions(numbers: list[int], shift: float = 0.0) -> torch.Tensor:
    base = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=DTYPE)
    if len(numbers) == 2:
        return base + shift
    return (
        torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.4, 1.0],
                [0.0, -1.4, 1.0],
            ],
            dtype=DTYPE,
        )
        + shift
    )


class _GeometryInteractionCache(InteractionCache):
    __slots__ = ["geometry_scale"]

    def __init__(self, geometry_scale: torch.Tensor) -> None:
        self.geometry_scale = geometry_scale


class _NonzeroInteraction(Interaction):
    label = "E6bLegacyNonzeroInteraction"

    def __init__(self) -> None:
        super().__init__(device=DEVICE, dtype=DTYPE)
        self.calls = 0

    def get_cache(self, *, numbers=None, positions=None, ihelp=None):
        return _GeometryInteractionCache(positions.square().sum() * 1.0e-1)

    def get_monopole_atom_potential(
        self,
        cache: InteractionCache,
        qat: torch.Tensor,
        qdp: torch.Tensor | None = None,
        qqp: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.calls += 1
        atom_shape = qat.shape[-1]
        pattern = torch.arange(atom_shape, dtype=qat.dtype, device=qat.device)
        return cache.geometry_scale * pattern

    def get_monopole_atom_energy(
        self, cache: InteractionCache, qat: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        self.calls += 1
        return 0.5 * cache.geometry_scale * qat.square()


class _CacheClassical(Classical):
    def __init__(self) -> None:
        super().__init__(device=DEVICE, dtype=DTYPE)
        self.label = "E6bCacheClassical"
        self.cache_calls = 0

    def get_cache(self, numbers, ihelp=None, **kwargs):
        self.cache_calls += 1
        return numbers.to(dtype=DTYPE).sum()

    def get_energy(self, positions, cache, **kwargs):
        atom_scale = cache / positions.shape[-2]
        return atom_scale * positions.square().sum(dim=-1) * 1.0e-4


def test_libcint_composition_matches_independent_serial_systems() -> None:
    pytest.importorskip("tad_libcint")
    na, nb = [1, 1], [8, 1, 1]
    a = _system(na, driver=labels.INTDRIVER_LIBCINT)
    b = _system(nb, driver=labels.INTDRIVER_LIBCINT)
    pa, pb = _positions(na), _positions(nb)
    assert classify_legacy_system(a) is LegacySystemRoute.LEGACY_COMPOSED
    assert classify_legacy_system(b) is LegacySystemRoute.LEGACY_COMPOSED

    composed = compose_legacy_systems((a, b), (pa, pb), source_ids=(4, 19))
    reversed_result = compose_legacy_systems(
        (b, a), (pb, pa), source_ids=(19, 4)
    )
    assert composed.source_ids == (4, 19)
    assert reversed_result.source_ids == (19, 4)
    for actual, system, pos in zip(composed.results, (a, b), (pa, pb)):
        expected = system.singlepoint(pos)
        torch.testing.assert_close(actual.energy, expected.energy)
        torch.testing.assert_close(actual.hcore, expected.hcore)
        torch.testing.assert_close(actual.overlap, expected.overlap)
    for actual, expected in zip(
        composed.results, reversed_result.results[::-1]
    ):
        torch.testing.assert_close(actual.energy, expected.energy)


def test_libcint_composition_preserves_shared_fields_and_responses() -> None:
    pytest.importorskip("tad_libcint")
    numbers = [1, 1]
    system = _system(numbers, driver=labels.INTDRIVER_LIBCINT)
    field = torch.tensor([0.01, -0.02, 0.015], dtype=DTYPE, requires_grad=True)
    field_grad = torch.eye(3, dtype=DTYPE) * 0.002
    pos_a = _positions(numbers).requires_grad_()
    pos_b = _positions(numbers).requires_grad_()
    result = compose_legacy_systems(
        (system, system),
        (pos_a, pos_b),
        field=field,
        field_grad=field_grad,
    )
    serial = system.singlepoint(pos_a, field=field, field_grad=field_grad)
    torch.testing.assert_close(result.results[0].energy, serial.energy)
    (composed_field_grad,) = torch.autograd.grad(
        result.results[0].energy.sum(), field, retain_graph=True
    )
    (serial_field_grad,) = torch.autograd.grad(serial.energy.sum(), field)
    torch.testing.assert_close(composed_field_grad, serial_field_grad)
    (physical_pos_grad,) = torch.autograd.grad(
        result.results[0].energy.sum(), pos_a
    )
    assert torch.count_nonzero(physical_pos_grad) > 0


def test_libcint_composition_keeps_per_system_charge_and_spin() -> None:
    pytest.importorskip("tad_libcint")
    na, nb = [1, 1], [8, 1, 1]
    a = _system(na, driver=labels.INTDRIVER_LIBCINT)
    b = _system(nb, driver=labels.INTDRIVER_LIBCINT)
    pa, pb = _positions(na), _positions(nb)
    composed = compose_legacy_systems(
        (a, b), (pa, pb), charges=(1, 0), spins=(1, 0)
    )
    for result, system, pos, charge, spin in zip(
        composed.results, (a, b), (pa, pb), (1, 0), (1, 0)
    ):
        serial = system.singlepoint(pos, chrg=charge, spin=spin)
        torch.testing.assert_close(result.energy, serial.energy)
        torch.testing.assert_close(result.scf, serial.scf)
        assert result.charges is not None and serial.charges is not None
        torch.testing.assert_close(result.charges.mono, serial.charges.mono)


def test_libcint_composition_preserves_model_parameter_gradient() -> None:
    pytest.importorskip("tad_libcint")
    par = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    kpol = par.parameter_tree.hamiltonian.xtb.kpol.param
    kpol.requires_grad_(True)
    numbers = torch.tensor([8, 1, 1], device=DEVICE)
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_LIBCINT,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        fermi_etemp=0.0,
    )
    system = Model(par=par, config=config).setup(numbers)
    positions = _positions(numbers.tolist()).requires_grad_()
    composed = compose_legacy_systems((system,), (positions,)).results[0]
    (composed_grad,) = torch.autograd.grad(
        composed.energy.sum(), kpol, retain_graph=True
    )

    serial_positions = positions.detach().clone().requires_grad_()
    serial = system.singlepoint(serial_positions)
    (serial_grad,) = torch.autograd.grad(serial.energy.sum(), kpol)
    torch.testing.assert_close(composed_grad, serial_grad)
    assert torch.isfinite(composed_grad).all()
    assert torch.count_nonzero(composed_grad) > 0
    assert par.parameter_tree.hamiltonian.xtb.kpol.param is kpol


def test_legacy_field_label_conflict_is_not_silently_replaced() -> None:
    pytest.importorskip("tad_libcint")
    custom = _NonzeroInteraction()
    custom.label = LABEL_EFIELD
    numbers = [1, 1]
    system = _system(
        numbers,
        driver=labels.INTDRIVER_LIBCINT,
        interaction=(custom,),
    )
    with pytest.raises(ValueError, match="Explicit field"):
        compose_legacy_systems(
            (system,),
            (_positions(numbers),),
            field=torch.ones(3, dtype=DTYPE),
        )


def test_custom_interaction_is_retained_and_order_independent() -> None:
    interaction = _NonzeroInteraction()
    na, nb = [1, 1], [3, 1]
    a = _system(na, interaction=(interaction,))
    b = _system(nb, interaction=(interaction,))
    pa, pb = _positions(na), _positions(nb)
    assert classify_legacy_system(a) is LegacySystemRoute.LEGACY_COMPOSED
    ab = compose_legacy_systems((a, b), (pa, pb))
    ba = compose_legacy_systems((b, a), (pb, pa))
    assert interaction.calls > 0
    for result, system, pos in zip(ab.results, (a, b), (pa, pb)):
        serial = system.singlepoint(pos)
        torch.testing.assert_close(result.energy, serial.energy)
        torch.testing.assert_close(result.scf, serial.scf)
    for lhs, rhs in zip(ab.results, ba.results[::-1]):
        torch.testing.assert_close(lhs.energy, rhs.energy)
    assert interaction.calls > 0
    baseline = _system(na).singlepoint(pa)
    assert ab.results[0].potential is not None
    assert baseline.potential is not None
    assert not torch.allclose(
        ab.results[0].potential.mono, baseline.potential.mono
    )

    mapped_pos = pa.clone().requires_grad_()
    serial_pos = pa.clone().requires_grad_()
    mapped = compose_legacy_systems((a,), (mapped_pos,)).results[0]
    serial = a.singlepoint(serial_pos)
    (mapped_grad,) = torch.autograd.grad(mapped.energy.sum(), mapped_pos)
    (serial_grad,) = torch.autograd.grad(serial.energy.sum(), serial_pos)
    torch.testing.assert_close(mapped_grad, serial_grad)
    assert torch.count_nonzero(mapped_grad) > 0


def test_custom_classical_cache_is_refreshed_per_composition_call() -> None:
    component = _CacheClassical()
    numbers = [1, 1]
    system = _system(numbers, classical=(component,))
    positions = _positions(numbers)
    assert classify_legacy_system(system) is LegacySystemRoute.LEGACY_COMPOSED
    before = component.cache_calls
    first = compose_legacy_systems((system,), (positions,)).results[0]
    second = compose_legacy_systems((system,), (positions,)).results[0]
    assert component.cache_calls >= before + 2
    first_classicals = dict(first.classical)
    second_classicals = dict(second.classical)
    assert component.label in first_classicals
    assert torch.count_nonzero(first_classicals[component.label]) > 0
    torch.testing.assert_close(
        first_classicals[component.label], second_classicals[component.label]
    )
    torch.testing.assert_close(first.energy, second.energy)


def test_legacy_composition_preserves_custom_classical_position_gradient() -> (
    None
):
    component = _CacheClassical()
    numbers = [1, 1]
    system = _system(numbers, classical=(component,))
    mapped_pos = _positions(numbers).requires_grad_()
    serial_pos = _positions(numbers).requires_grad_()
    result = compose_legacy_systems((system,), (mapped_pos,)).results[0]
    serial = system.singlepoint(serial_pos)
    (mapped_grad,) = torch.autograd.grad(result.energy.sum(), mapped_pos)
    (serial_grad,) = torch.autograd.grad(serial.energy.sum(), serial_pos)
    torch.testing.assert_close(mapped_grad, serial_grad)
    assert torch.count_nonzero(mapped_grad) > 0


def test_mapped_eligible_system_is_not_accepted_by_legacy_composer() -> None:
    numbers = [1, 1]
    system = _system(numbers)
    assert classify_legacy_system(system) is LegacySystemRoute.MAPPED
    with pytest.raises(ValueError, match="Mapped-eligible"):
        compose_legacy_systems((system,), (_positions(numbers),))


@pytest.mark.parametrize(
    "options",
    [
        {"fermi_etemp": -1.0},
        {"fermi_maxiter": -1},
    ],
    ids=["negative-temperature", "negative-fermi-maxiter"],
)
def test_legacy_owned_fermi_edges_select_legacy_route(
    options: dict[str, float | int],
) -> None:
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        **options,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE), config=config
    ).setup(torch.tensor([1, 1], device=DEVICE))
    assert classify_legacy_system(system) is LegacySystemRoute.LEGACY_COMPOSED


def test_negative_outer_maxiter_keeps_functional_error_policy() -> None:
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        maxiter=-1,
    )
    system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE), config=config
    ).setup(torch.tensor([1, 1], device=DEVICE))
    with pytest.raises(ValueError, match="maxiter >= 0"):
        classify_legacy_system(system)


def test_classical_only_calculation_retains_separate_route() -> None:
    numbers = [1, 1]
    config = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        exclude="scf",
    )
    system = Model(
        par=ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE), config=config
    ).setup(torch.tensor(numbers, device=DEVICE))
    assert classify_legacy_system(system) is LegacySystemRoute.SEPARATE_ROUTE
    result = compose_legacy_systems((system,), (_positions(numbers),)).results[
        0
    ]
    assert result.scf is not None
    assert torch.count_nonzero(result.scf) == 0
    assert result.charges is None


def test_legacy_engine_keywords_are_not_filtered() -> None:
    numbers = [1, 1]
    system = _system(numbers, classical=(_CacheClassical(),))
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        compose_legacy_systems(
            (system,),
            (_positions(numbers),),
            legacy_options=LegacySCFOptions(kwargs={"unexpected_option": 1}),
        )


def test_legacy_engine_positional_extras_are_not_filtered() -> None:
    numbers = [1, 1]
    system = _system(numbers, classical=(_CacheClassical(),))
    with pytest.raises(TypeError):
        compose_legacy_systems(
            (system,),
            (_positions(numbers),),
            legacy_options=LegacySCFOptions(args=(object(),)),
        )


def test_legacy_engine_constructor_options_are_consumed_by_legacy_owner() -> (
    None
):
    numbers = [1, 1]
    system = _system(numbers, classical=(_CacheClassical(),))
    options = LegacySCFOptions(
        kwargs={
            "fwd_options": {"damp": 0.25},
            "bck_options": {"posdef": False},
            "eigen_options": {"method": "exacteig"},
        }
    )
    result = compose_legacy_systems(
        (system,), (_positions(numbers),), legacy_options=options
    ).results[0]
    assert torch.isfinite(result.energy).all()


def test_legacy_composition_preserves_option_values_and_constructor_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward option identities and retain legacy constructor defaults."""
    from dxtb._src.scf import unrolling

    original = unrolling.SelfConsistentFieldFull
    captured: dict[str, object] = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        engine = original(*args, **kwargs)
        captured["engine"] = engine
        return engine

    monkeypatch.setattr(unrolling, "SelfConsistentFieldFull", capture)
    numbers = [1, 1]
    system = _system(numbers, classical=(_CacheClassical(),))
    fwd_options = {"damp": 0.25}
    bck_options = {"posdef": False}
    eigen_options = {"method": "exacteig"}
    options = LegacySCFOptions(
        kwargs={
            "fwd_options": fwd_options,
            "bck_options": bck_options,
            "eigen_options": eigen_options,
        }
    )

    compose_legacy_systems(
        (system,), (_positions(numbers),), legacy_options=options
    )

    assert captured["fwd_options"] is fwd_options
    assert captured["bck_options"] is bck_options
    assert captured["eigen_options"] is eigen_options
    assert captured["config"] is system.config.scf
    engine = captured["engine"]
    assert engine.fwd_options["damp"] == 0.25
    assert engine.fwd_options["maxiter"] == system.config.scf.maxiter
    assert engine.bck_options == {"posdef": False}
    assert engine.eigen_options == {"method": "exacteig"}


@pytest.mark.parametrize(
    "keyword", ["fwd_options", "bck_options", "eigen_options"]
)
def test_legacy_composition_preserves_malformed_option_errors(
    keyword: str,
) -> None:
    """Malformed legacy option mappings fail at the legacy owner."""
    numbers = [1, 1]
    system = _system(numbers, classical=(_CacheClassical(),))
    with pytest.raises(TypeError, match="mapping"):
        compose_legacy_systems(
            (system,),
            (_positions(numbers),),
            legacy_options=LegacySCFOptions(kwargs={keyword: None}),
        )


def test_heterogeneous_classical_results_keep_per_system_schema() -> None:
    """Each heterogeneous lane retains its own configured classical labels."""
    numbers = [1, 1]
    first_component = _CacheClassical()
    second_component = _CacheClassical()
    second_component.label = "E6bOtherCacheClassical"
    first_system = _system(numbers, classical=(first_component,))
    second_system = _system(numbers, classical=(second_component,))
    results = compose_legacy_systems(
        (first_system, second_system),
        (_positions(numbers), _positions(numbers, shift=0.1)),
    ).results

    assert first_component.label in dict(results[0].classical)
    assert second_component.label not in dict(results[0].classical)
    assert second_component.label in dict(results[1].classical)
    assert first_component.label not in dict(results[1].classical)
    labels_0 = tuple(label for label, _ in results[0].classical)
    labels_1 = tuple(label for label, _ in results[1].classical)
    common_0 = tuple(
        label for label in labels_0 if label != first_component.label
    )
    common_1 = tuple(
        label for label in labels_1 if label != second_component.label
    )
    assert common_0 == common_1


def test_legacy_composition_preserves_original_exception_and_source_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retained evaluation re-raises the original exception with lane context."""
    from dxtb._src.calculators import legacy_composition

    system = _system([1, 1], driver=labels.INTDRIVER_LIBCINT)
    original_error = ValueError("invalid geometry")

    def fail(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(legacy_composition, "_singlepoint_impl", fail)
    with pytest.raises(ValueError, match="invalid geometry") as err:
        compose_legacy_systems(
            (system,), (_positions([1, 1]),), source_ids=(91,)
        )

    assert err.value is original_error
    assert err.value.source_id == 91


def test_invalid_geometry_error_matches_independent_serial_call() -> None:
    """Invalid geometry errors retain the single-System owner and category."""
    pytest.importorskip("tad_libcint")
    system = _system([1, 1], driver=labels.INTDRIVER_LIBCINT)
    invalid_positions = torch.zeros(3, dtype=DTYPE)

    with pytest.raises(Exception) as serial_error:
        system.singlepoint(invalid_positions)
    with pytest.raises(type(serial_error.value)) as composed_error:
        compose_legacy_systems((system,), (invalid_positions,))

    assert str(composed_error.value) == str(serial_error.value)
    assert composed_error.value.source_id == 0


def test_legacy_composer_propagates_numerical_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dxtb._src.calculators import legacy_composition

    numbers = [1, 1]
    system = _system(numbers, driver=labels.INTDRIVER_LIBCINT)

    def fail(*args, **kwargs):
        raise RuntimeError("injected evaluation failure")

    monkeypatch.setattr(legacy_composition, "_singlepoint_impl", fail)
    with pytest.raises(
        RuntimeError, match="injected evaluation failure"
    ) as err:
        compose_legacy_systems(
            (system,), (_positions(numbers),), source_ids=(37,)
        )
    assert err.value.source_id == 37
