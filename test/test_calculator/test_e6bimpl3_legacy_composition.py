"""Private retained legacy composition acceptance controls."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.legacy_composition import (
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
    assert not classify_legacy_system(a)
    assert not classify_legacy_system(b)

    composed = compose_legacy_systems((a, b), (pa, pb), source_ids=(4, 19))
    reversed_result = compose_legacy_systems(
        (b, a), (pb, pa), source_ids=(19, 4)
    )
    for actual, system, pos in zip(composed, (a, b), (pa, pb)):
        expected = system.singlepoint(pos)
        torch.testing.assert_close(actual.energy, expected.energy)
        torch.testing.assert_close(actual.hcore, expected.hcore)
        torch.testing.assert_close(actual.overlap, expected.overlap)
    for actual, expected in zip(composed, reversed_result[::-1]):
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
    torch.testing.assert_close(result[0].energy, serial.energy)
    (composed_field_grad,) = torch.autograd.grad(
        result[0].energy.sum(), field, retain_graph=True
    )
    (serial_field_grad,) = torch.autograd.grad(serial.energy.sum(), field)
    torch.testing.assert_close(composed_field_grad, serial_field_grad)
    (physical_pos_grad,) = torch.autograd.grad(result[0].energy.sum(), pos_a)
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
        composed, (a, b), (pa, pb), (1, 0), (1, 0)
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
    composed = compose_legacy_systems((system,), (positions,))[0]
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
    with pytest.raises(NotImplementedError, match="interaction extension"):
        compose_legacy_systems(
            (system,),
            (_positions(numbers),),
            field=torch.ones(3, dtype=DTYPE),
        )


def test_custom_interaction_is_explicitly_unsupported() -> None:
    interaction = _NonzeroInteraction()
    na, nb = [1, 1], [3, 1]
    a = _system(na, interaction=(interaction,))
    b = _system(nb, interaction=(interaction,))
    pa, pb = _positions(na), _positions(nb)
    with pytest.raises(NotImplementedError, match="interaction extension"):
        classify_legacy_system(a)
    with pytest.raises(NotImplementedError, match="interaction extension"):
        compose_legacy_systems((a, b), (pa, pb))


def test_custom_classical_cache_is_refreshed_per_composition_call() -> None:
    component = _CacheClassical()
    numbers = [1, 1]
    system = _system(numbers, classical=(component,))
    positions = _positions(numbers)
    assert not classify_legacy_system(system)
    before = component.cache_calls
    first = compose_legacy_systems((system,), (positions,))[0]
    second = compose_legacy_systems((system,), (positions,))[0]
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
    result = compose_legacy_systems((system,), (mapped_pos,))[0]
    serial = system.singlepoint(serial_pos)
    (mapped_grad,) = torch.autograd.grad(result.energy.sum(), mapped_pos)
    (serial_grad,) = torch.autograd.grad(serial.energy.sum(), serial_pos)
    torch.testing.assert_close(mapped_grad, serial_grad)
    assert torch.count_nonzero(mapped_grad) > 0


def test_mapped_eligible_system_is_not_accepted_by_legacy_composer() -> None:
    numbers = [1, 1]
    system = _system(numbers)
    assert classify_legacy_system(system)
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
def test_invalid_fermi_edges_are_rejected(
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
    with pytest.raises(
        ValueError, match="Electronic Temperature|Fermi maxiter"
    ):
        classify_legacy_system(system)


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
    assert not classify_legacy_system(system)
    result = compose_legacy_systems((system,), (_positions(numbers),))[0]
    assert result.scf is not None
    assert torch.count_nonzero(result.scf) == 0
    assert result.charges is None


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
    )

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


def test_s13_strategy_method_and_differentiation_eligibility() -> None:
    """Verify Section 7/8 strategy, method, representation, and response routing."""
    from dataclasses import replace

    from dxtb import GFN2_XTB
    from dxtb._src.calculators.packed_result import evaluate_packed_systems
    from dxtb._src.scf.functional.padded import evaluate_padded_fixed_trip
    from dxtb.scf import (
        AndersonMethod,
        AndersonMethodSettings,
        BroydenMethod,
        BroydenMethodSettings,
        CDIISMethod,
        CDIISMethodSettings,
        LinearMethod,
        SCFDifferentiation,
        SCFRepresentation,
        SCFResponseSettings,
        SCFStage,
        SCFStrategy,
        SOSCFMethod,
        strategy_preset,
    )

    par_gfn1 = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    par_gfn2 = ParamModule(GFN2_XTB, device=DEVICE, dtype=DTYPE)
    nums_h2 = torch.tensor([1, 1], device=DEVICE)
    nums_h2o = torch.tensor([8, 1, 1], device=DEVICE)
    pos_h2 = _positions([1, 1])
    pos_h2o = _positions([8, 1, 1])

    anderson = AndersonMethod(AndersonMethodSettings())
    broyden = BroydenMethod(BroydenMethodSettings())
    cdiis = CDIISMethod(CDIISMethodSettings())
    linear = LinearMethod()

    # 1. Single-stage fixed-trip methods in unrolled mode -> MAPPED
    mapped_strategies = [
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(linear, SCFRepresentation.CHARGE),)),
        ),
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(linear, SCFRepresentation.POTENTIAL),)),
        ),
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(anderson, SCFRepresentation.POTENTIAL),)),
        ),
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(anderson, SCFRepresentation.CHARGE),)),
        ),
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(broyden, SCFRepresentation.POTENTIAL),)),
        ),
        (
            par_gfn1,
            "GFN1-xTB",
            SCFStrategy((SCFStage(cdiis, SCFRepresentation.FOCK),)),
        ),
        (
            par_gfn2,
            "GFN2-xTB",
            SCFStrategy((SCFStage(anderson, SCFRepresentation.POTENTIAL),)),
        ),
        (
            par_gfn2,
            "GFN2-xTB",
            SCFStrategy((SCFStage(anderson, SCFRepresentation.CHARGE),)),
        ),
        (
            par_gfn2,
            "GFN2-xTB",
            SCFStrategy((SCFStage(cdiis, SCFRepresentation.FOCK),)),
        ),
    ]
    for par, method_name, strat in mapped_strategies:
        cfg = Config.create(
            method=method_name,
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_QUADRUPOLE,
            strategy=strat,
            differentiation=SCFDifferentiation.UNROLLED,
            convergence="tight",
            maxiter=15,
        )
        model = Model(par=par, config=cfg)
        s_a = model.setup(nums_h2)
        s_b = model.setup(nums_h2o)
        assert classify_legacy_system(s_a)
        assert classify_legacy_system(s_b)
        packed = evaluate_packed_systems((s_a, s_b), (pos_h2, pos_h2o))
        ref_a = s_a.singlepoint(pos_h2)
        ref_b = s_b.singlepoint(pos_h2o)
        torch.testing.assert_close(
            packed.energy[0, :2], ref_a.energy, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            packed.energy[1, :3], ref_b.energy, atol=1e-11, rtol=1e-11
        )

    # 2. Multi-stage and SOSCF strategies:
    #    - rejected under UNROLLED differentiation at both ConfigSCF.__post_init__ and classify_legacy_system
    #    - SOSCF rejected with active Fermi smearing (fermi_etemp > 0)
    #    - routed to LEGACY_COMPOSED under IMPLICIT differentiation (with etemp=0 for SOSCF)
    soscf_strat = SCFStrategy(
        (SCFStage(SOSCFMethod(), SCFRepresentation.FOCK),)
    )
    base_cfg = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        fermi_etemp=0.0,
    )
    base_sys = Model(par=par_gfn1, config=base_cfg).setup(nums_h2)
    for bad_unrolled_strat in (
        strategy_preset("broyden-cdiis"),
        strategy_preset("cdiis-soscf"),
        soscf_strat,
    ):
        with pytest.raises(ValueError, match="requires eager/implicit"):
            Config.create(
                method="GFN1-xTB",
                int_driver=labels.INTDRIVER_PYTORCH,
                int_level=labels.INTLEVEL_QUADRUPOLE,
                strategy=bad_unrolled_strat,
                differentiation=SCFDifferentiation.UNROLLED,
                fermi_etemp=0.0,
            )
        bypassed_scf = replace(base_cfg.scf)
        object.__setattr__(bypassed_scf, "strategy", bad_unrolled_strat)
        bypassed_cfg = replace(base_cfg)
        object.__setattr__(bypassed_cfg, "scf", bypassed_scf)
        mutated_sys = replace(base_sys, config=bypassed_cfg)
        with pytest.raises(
            ValueError, match="requires eager/implicit|single-stage fixed-trip"
        ):
            classify_legacy_system(mutated_sys)

    # Direct evaluate_padded_fixed_trip also rejects multi-stage or eager-only strategies
    from types import SimpleNamespace

    dummy_padded = SimpleNamespace(
        core=SimpleNamespace(config=bypassed_cfg),
    )
    with pytest.raises(ValueError, match="single-stage"):
        evaluate_padded_fixed_trip(
            dummy_padded,  # type: ignore[arg-type]
            pos_h2,
            torch.tensor(0.0, dtype=DTYPE),
            torch.tensor(0.0, dtype=DTYPE),
        )

    with pytest.raises(ValueError, match="active Fermi smearing"):
        Config.create(
            method="GFN1-xTB",
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_QUADRUPOLE,
            strategy=soscf_strat,
            differentiation=SCFDifferentiation.IMPLICIT,
            fermi_etemp=300.0,
        )
    base_smear_cfg = Config.create(
        method="GFN1-xTB",
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_QUADRUPOLE,
        differentiation=SCFDifferentiation.IMPLICIT,
        fermi_etemp=300.0,
    )
    bypassed_smear_scf = replace(base_smear_cfg.scf)
    object.__setattr__(bypassed_smear_scf, "strategy", soscf_strat)
    bypassed_smear_cfg = replace(base_smear_cfg)
    object.__setattr__(bypassed_smear_cfg, "scf", bypassed_smear_scf)
    sys_soscf_smear = replace(
        Model(par=par_gfn1, config=base_smear_cfg).setup(nums_h2),
        config=bypassed_smear_cfg,
    )
    with pytest.raises(
        ValueError, match="active Fermi smearing|fixed-occupation"
    ):
        classify_legacy_system(sys_soscf_smear)

    # Multi-stage and SOSCF (at etemp=0) + alternate response coordinates under IMPLICIT -> LEGACY_COMPOSED
    implicit_cases = [
        ("broyden-cdiis", SCFResponseSettings(representation=None)),
        (
            "cdiis-soscf",
            SCFResponseSettings(representation=SCFRepresentation.POTENTIAL),
        ),
        (
            SCFStrategy((SCFStage(cdiis, SCFRepresentation.FOCK),)),
            SCFResponseSettings(representation=SCFRepresentation.POTENTIAL),
        ),
        (
            SCFStrategy((SCFStage(anderson, SCFRepresentation.POTENTIAL),)),
            SCFResponseSettings(representation=SCFRepresentation.CHARGE),
        ),
    ]
    for strat, resp_settings in implicit_cases:
        cfg_imp = Config.create(
            method="GFN1-xTB",
            int_driver=labels.INTDRIVER_PYTORCH,
            int_level=labels.INTLEVEL_QUADRUPOLE,
            strategy=strat,
            differentiation=SCFDifferentiation.IMPLICIT,
            response=resp_settings,
            fermi_etemp=0.0,
            convergence="tight",
        )
        model_imp = Model(par=par_gfn1, config=cfg_imp)
        sys_imp_a = model_imp.setup(nums_h2)
        sys_imp_b = model_imp.setup(nums_h2o)
        assert not classify_legacy_system(sys_imp_a)
        p_a = pos_h2.detach().clone().requires_grad_(True)
        p_b = pos_h2o.detach().clone().requires_grad_(True)
        packed_imp = evaluate_packed_systems(
            (sys_imp_a, sys_imp_b), (p_a, p_b)
        )
        (g_a, g_b) = torch.autograd.grad(
            packed_imp.energy.sum(), (p_a, p_b)
        )
        p_b_ref = pos_h2o.detach().clone().requires_grad_(True)
        ref_b = sys_imp_b.singlepoint(p_b_ref)
        (g_b_ref,) = torch.autograd.grad(ref_b.energy.sum(), p_b_ref)
        torch.testing.assert_close(
            packed_imp.energy[1, :3], ref_b.energy, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(g_b, g_b_ref, atol=1e-9, rtol=1e-9)
