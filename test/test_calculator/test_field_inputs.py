"""Explicit field inputs at the single-System and Calculator boundaries."""

from __future__ import annotations

import pytest
import torch
from tad_mctc.exceptions import DtypeError
from torch import nn

from dxtb import GFN1_XTB, Calculator, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb._src.components.interactions.container import Charges
from dxtb._src.components.interactions.field.efield import (
    ElectricField,
    build_electric_field_data,
    new_efield,
)
from dxtb._src.components.interactions.field.efieldgrad import (
    ElectricFieldGrad,
    build_electric_field_grad_data,
    new_efield_grad,
)
from dxtb._src.components.interactions.list import (
    InteractionList,
    InteractionListCache,
)
from dxtb.config import Config


def _calc(numbers: torch.Tensor, *, interaction=None) -> Calculator:
    return Calculator(
        numbers,
        GFN1_XTB,
        interaction=interaction,
        opts={
            "verbosity": 0,
            "int_driver": "pytorch",
            "int_level": labels.INTLEVEL_QUADRUPOLE,
        },
        dtype=torch.double,
    )


def test_system_field_inputs_are_fresh_and_history_independent() -> None:
    """Explicit fields leave the System's interaction list unchanged."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    system = _calc(numbers).system
    before = tuple(system.interactions.components)
    f1 = torch.tensor([0.01, -0.02, 0.03], dtype=torch.double)
    f2 = torch.tensor([-0.03, 0.01, 0.02], dtype=torch.double)
    g1 = torch.tensor(
        [[0.01, 0.02, 0.0], [0.03, -0.01, 0.01], [0.0, 0.02, 0.0]],
        dtype=torch.double,
    )
    g2 = -0.7 * g1

    a = system.singlepoint(positions, field=f1, field_grad=g1).energy
    _ = system.singlepoint(positions, field=f2).energy
    _ = system.singlepoint(positions, field_grad=g2).energy
    b = system.singlepoint(positions, field=f1, field_grad=g1).energy
    fresh = (
        _calc(numbers)
        .system.singlepoint(positions, field=f1, field_grad=g1)
        .energy
    )
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, fresh)
    assert tuple(system.interactions.components) == before
    assert not any(type(x) is ElectricField for x in before)
    assert not any(type(x) is ElectricFieldGrad for x in before)
    assert not hasattr(system, "field")
    assert not hasattr(system, "field_grad")


def test_field_call_data_vmap_matches_loop() -> None:
    """Pure field and field-gradient builders compose with vmap."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    system = _calc(numbers).system
    fields = torch.tensor(
        [[0.01, 0.0, -0.02], [-0.02, 0.03, 0.01]], dtype=torch.double
    )
    grads = torch.stack((torch.eye(3), -0.4 * torch.eye(3))).to(torch.double)

    mapped_field = torch.func.vmap(
        lambda f: build_electric_field_data(positions, f).vat
    )(fields)
    loop_field = torch.stack(
        [build_electric_field_data(positions, f).vat for f in fields]
    )
    torch.testing.assert_close(mapped_field, loop_field)

    mapped_grad = torch.func.vmap(
        lambda g: build_electric_field_grad_data(positions, g).vqp
    )(grads)
    loop_grad = torch.stack(
        [build_electric_field_grad_data(positions, g).vqp for g in grads]
    )
    torch.testing.assert_close(mapped_grad, loop_grad)

    ihelp = system.ihelp
    qatom = torch.tensor([0.2, -0.1, -0.1], dtype=torch.double)
    qorb = ihelp.spread_atom_to_orbital(qatom)
    atom_dipole = torch.arange(9, dtype=torch.double).reshape(3, 3) / 50
    atom_quad = torch.arange(18, dtype=torch.double).reshape(3, 6) / 100
    charges = Charges(
        mono=qorb,
        dipole=ihelp.spread_atom_to_orbital(atom_dipole, dim=-2, extra=True),
        quad=ihelp.spread_atom_to_orbital(atom_quad, dim=-2, extra=True),
    )
    efield = new_efield(torch.zeros(3, dtype=torch.double))
    efield_grad = new_efield_grad(torch.zeros((3, 3), dtype=torch.double))
    interactions = InteractionList(efield, efield_grad)

    def field_energy(f: torch.Tensor) -> torch.Tensor:
        cache = build_electric_field_data(positions, f)
        return (
            efield.get_monopole_atom_energy(cache, qatom)
            + efield.get_dipole_atom_energy(cache, qatom, atom_dipole)
        ).sum()

    def field_grad_energy(g: torch.Tensor) -> torch.Tensor:
        cache = build_electric_field_grad_data(positions, g)
        return (
            efield_grad.get_monopole_atom_energy(cache, qatom)
            + efield_grad.get_dipole_atom_energy(cache, qatom, atom_dipole)
            + efield_grad.get_quadrupole_atom_energy(
                cache, qatom, atom_dipole, atom_quad
            )
        ).sum()

    mapped_energy = torch.func.vmap(field_energy)(fields)
    loop_energy = torch.stack([field_energy(f) for f in fields])
    torch.testing.assert_close(mapped_energy, loop_energy)

    mapped_grad_energy = torch.func.vmap(field_grad_energy)(grads)
    loop_grad_energy = torch.stack([field_grad_energy(g) for g in grads])
    torch.testing.assert_close(mapped_grad_energy, loop_grad_energy)

    def potential(f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        call_data = InteractionListCache()
        call_data[efield.label] = build_electric_field_data(positions, f)
        call_data[efield_grad.label] = build_electric_field_grad_data(
            positions, g
        )
        return interactions.get_potential(call_data, charges, ihelp).as_tensor()

    mapped_potential = torch.func.vmap(potential)(fields, grads)
    loop_potential = torch.stack(
        [potential(f, g) for f, g in zip(fields, grads, strict=True)]
    )
    torch.testing.assert_close(mapped_potential, loop_potential)


def test_system_field_jvp_and_jacfwd_match_reverse() -> None:
    """Core field tangents agree with reverse and jacfwd derivatives."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    system = _calc(numbers).system
    field = torch.tensor([0.01, 0.02, -0.01], dtype=torch.double)
    direction = torch.tensor([0.2, -0.1, 0.3], dtype=torch.double)

    def field_energy(value: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(positions, field=value).energy.sum()

    field_leaf = field.clone().requires_grad_(True)
    (reverse,) = torch.autograd.grad(field_energy(field_leaf), field_leaf)
    _, tangent = torch.func.jvp(field_energy, (field,), (direction,))
    jacfwd = torch.func.jacfwd(field_energy)(field)
    torch.testing.assert_close(tangent, torch.dot(reverse, direction))
    torch.testing.assert_close(jacfwd, reverse)

    grad = torch.tensor(
        [[0.01, 0.0, 0.02], [0.0, -0.01, 0.01], [0.0, 0.0, 0.0]],
        dtype=torch.double,
    )
    grad_direction = torch.arange(1, 10, dtype=torch.double).reshape(3, 3) / 17

    def grad_energy(value: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(positions, field_grad=value).energy.sum()

    grad_leaf = grad.clone().requires_grad_(True)
    (grad_reverse,) = torch.autograd.grad(grad_energy(grad_leaf), grad_leaf)
    _, grad_tangent = torch.func.jvp(grad_energy, (grad,), (grad_direction,))
    grad_jacfwd = torch.func.jacfwd(grad_energy)(grad)
    torch.testing.assert_close(
        grad_tangent, torch.sum(grad_reverse * grad_direction)
    )
    torch.testing.assert_close(grad_jacfwd, grad_reverse)


def test_same_field_leaves_can_be_differentiated_repeatedly() -> None:
    """Consumed perturbation graphs are not retained by a System."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    system = _calc(numbers).system
    field = torch.tensor(
        [0.01, -0.02, 0.03], dtype=torch.double, requires_grad=True
    )
    grad = torch.tensor(
        [[0.01, 0.02, 0.0], [0.03, -0.01, 0.01], [0.0, 0.02, 0.0]],
        dtype=torch.double,
        requires_grad=True,
    )

    first_field = system.singlepoint(positions, field=field)
    (first_field_grad,) = torch.autograd.grad(first_field.energy.sum(), field)
    second_field = system.singlepoint(positions, field=field)
    (second_field_grad,) = torch.autograd.grad(second_field.energy.sum(), field)
    torch.testing.assert_close(first_field.energy, second_field.energy)
    torch.testing.assert_close(first_field_grad, second_field_grad)
    torch.testing.assert_close(
        first_field.charges.mono, second_field.charges.mono
    )
    torch.testing.assert_close(first_field.density, second_field.density)
    assert (
        first_field.potential is not None and second_field.potential is not None
    )
    torch.testing.assert_close(
        first_field.potential.mono, second_field.potential.mono
    )

    first_grad = system.singlepoint(positions, field_grad=grad)
    (first_grad_grad,) = torch.autograd.grad(first_grad.energy.sum(), grad)
    second_grad = system.singlepoint(positions, field_grad=grad)
    (second_grad_grad,) = torch.autograd.grad(second_grad.energy.sum(), grad)
    torch.testing.assert_close(first_grad.energy, second_grad.energy)
    torch.testing.assert_close(first_grad_grad, second_grad_grad)
    torch.testing.assert_close(
        first_grad.charges.mono, second_grad.charges.mono
    )
    torch.testing.assert_close(first_grad.density, second_grad.density)
    assert (
        first_grad.potential is not None and second_grad.potential is not None
    )
    torch.testing.assert_close(
        first_grad.potential.mono, second_grad.potential.mono
    )


def test_field_gradient_geometry_mixed_derivative() -> None:
    """The field-gradient response remains connected to positions."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    ).requires_grad_(True)
    system = _calc(numbers).system
    field_grad = torch.tensor(
        [[0.01, 0.02, 0.0], [0.03, -0.01, 0.01], [0.0, 0.02, 0.0]],
        dtype=torch.double,
        requires_grad=True,
    )
    direction = torch.arange(1, 10, dtype=torch.double).reshape(3, 3) / 17

    energy = system.singlepoint(positions, field_grad=field_grad).energy.sum()
    (response,) = torch.autograd.grad(energy, field_grad, create_graph=True)
    projected = torch.sum(response * direction)
    (mixed,) = torch.autograd.grad(projected, positions)

    assert torch.isfinite(mixed).all()
    assert torch.count_nonzero(mixed) > 0


def test_numerical_responses_leave_calculator_defaults_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Numerical property perturbations are local values, never state writes."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    field = torch.tensor([0.01, -0.02, 0.03], dtype=torch.double)
    field_grad = torch.tensor(
        [[0.01, 0.02, 0.0], [0.03, -0.01, 0.01], [0.0, 0.02, 0.0]],
        dtype=torch.double,
    )
    calc = _calc(
        numbers,
        interaction=[new_efield(field), new_efield_grad(field_grad)],
    )

    def energy_stub(
        self,
        positions: torch.Tensor,
        chrg=0,
        spin=None,
        field=None,
        field_grad=None,
        **kwargs,
    ) -> torch.Tensor:
        value = positions.new_zeros(())
        if field is not None:
            value = value + torch.pow(field, 3).sum()
        if field_grad is not None:
            value = value + torch.pow(field_grad, 3).sum()
        return value

    monkeypatch.setattr(type(calc), "energy", energy_stub)
    for property_function in (
        calc.dipole_numerical,
        calc.quadrupole_numerical,
        calc.polarizability_numerical,
        calc.hyperpolarizability_numerical,
    ):
        property_function(positions)
        assert calc._field_default is field
        assert calc._field_grad_default is field_grad
        torch.testing.assert_close(calc._field_default, field)
        torch.testing.assert_close(calc._field_grad_default, field_grad)


def test_parammodule_field_mixed_derivative() -> None:
    """A real GFN1 leaf differentiates through field response and setup."""

    class FieldEnergy(nn.Module):
        def __init__(self, par: ParamModule, numbers: torch.Tensor) -> None:
            super().__init__()
            self.par = par
            self.register_buffer("numbers", numbers.clone())

        def forward(
            self, positions: torch.Tensor, field: torch.Tensor
        ) -> torch.Tensor:
            model = Model(
                par=self.par,
                config=Config.create(
                    method="gfn1-xtb",
                    int_driver="pytorch",
                    int_level=labels.INTLEVEL_DIPOLE,
                    maxiter=150,
                    f_atol=1.0e-10,
                    x_atol=1.0e-10,
                ),
            )
            system = model.setup(self.numbers)
            return system.singlepoint(positions, field=field).energy.sum()

    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    field = torch.tensor([0.01, -0.02, 0.03], dtype=torch.double)
    direction = torch.tensor([0.3, -0.2, 0.4], dtype=torch.double)
    module = FieldEnergy(ParamModule(GFN1_XTB, dtype=torch.double), numbers)
    matches = [
        name
        for name, _ in module.named_parameters()
        if name.endswith("parameter_tree.element.O.gam.param")
    ]
    assert len(matches) == 1
    name = matches[0]
    original = dict(module.named_parameters())[name]
    selected = {name: original.detach().clone().requires_grad_(True)}

    def projected_field_gradient(
        state: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        def energy_of_field(value: torch.Tensor) -> torch.Tensor:
            return torch.func.functional_call(
                module, state, (positions, value), strict=False
            )

        response = torch.func.grad(energy_of_field)(field)
        return torch.dot(response, direction)

    mixed = torch.func.grad(projected_field_gradient)(selected)[name]
    assert torch.isfinite(mixed)
    assert torch.count_nonzero(mixed) > 0


def test_explicit_field_validation_and_integral_level() -> None:
    """Explicit perturbations are shape-checked and require matching integrals."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.zeros((3, 3), dtype=torch.double)
    system = _calc(numbers).system
    with pytest.raises(ValueError, match=r"shape \(3,\)"):
        system.singlepoint(positions, field=torch.zeros(2, dtype=torch.double))
    with pytest.raises(ValueError, match=r"shape \(3, 3\)"):
        system.singlepoint(
            positions, field_grad=torch.zeros((3, 2), dtype=torch.double)
        )
    with pytest.raises(DtypeError, match="Dtype mismatch"):
        system.singlepoint(positions, field=torch.zeros(3, dtype=torch.float32))
    with pytest.raises(RuntimeError, match="Device mismatch"):
        system.singlepoint(
            positions, field=torch.empty(3, device="meta", dtype=torch.double)
        )
    with pytest.raises(RuntimeError, match="requires dipole integrals"):
        Calculator(
            numbers,
            GFN1_XTB,
            opts={"verbosity": 0, "int_driver": "pytorch"},
            dtype=torch.double,
        ).system.singlepoint(
            positions, field=torch.zeros(3, dtype=torch.double)
        )
    with pytest.raises(RuntimeError, match="requires quadrupole integrals"):
        Calculator(
            numbers,
            GFN1_XTB,
            opts={"verbosity": 0, "int_driver": "pytorch"},
            dtype=torch.double,
        ).system.singlepoint(
            positions, field_grad=torch.zeros((3, 3), dtype=torch.double)
        )


def test_direct_model_rejects_exact_field_components() -> None:
    """Only Calculator can translate legacy field components to defaults."""
    numbers = torch.tensor([8, 1, 1])
    field = torch.ones(3, dtype=torch.double)
    field_grad = torch.eye(3, dtype=torch.double)
    par = ParamModule(GFN1_XTB, dtype=torch.double)

    with pytest.raises(ValueError, match="Pass field= or field_grad="):
        Model(
            par=par,
            interaction=(new_efield(field),),
            auto_int_level=False,
        ).setup(numbers)

    with pytest.raises(ValueError, match="Pass field= or field_grad="):
        Model(
            par=par,
            interaction=(new_efield_grad(field_grad),),
            auto_int_level=False,
        ).setup(numbers)


def test_calculator_legacy_promotion_ignores_auto_int_level() -> None:
    """Constructor-field compatibility promotes explicit low levels."""
    numbers = torch.tensor([8, 1, 1])
    field = torch.ones(3, dtype=torch.double)
    field_grad = torch.eye(3, dtype=torch.double)
    options = {
        "verbosity": 0,
        "int_driver": "pytorch",
        "int_level": labels.INTLEVEL_HCORE,
    }

    dipole_calc = Calculator(
        numbers,
        GFN1_XTB,
        interaction=new_efield(field),
        opts=options,
        dtype=torch.double,
        auto_int_level=False,
    )
    assert dipole_calc.opts.ints.level >= labels.INTLEVEL_DIPOLE
    assert not any(
        type(x) is ElectricField for x in dipole_calc.model.interaction
    )
    assert not any(
        type(x) is ElectricField
        for x in dipole_calc.system.interactions.components
    )

    quadrupole_calc = Calculator(
        numbers,
        GFN1_XTB,
        interaction=new_efield_grad(field_grad),
        opts=options,
        dtype=torch.double,
        auto_int_level=False,
    )
    assert quadrupole_calc.opts.ints.level >= labels.INTLEVEL_QUADRUPOLE
    assert not any(
        type(x) is ElectricFieldGrad for x in quadrupole_calc.model.interaction
    )
    assert not any(
        type(x) is ElectricFieldGrad
        for x in quadrupole_calc.system.interactions.components
    )


def test_field_subclasses_are_not_translated_as_defaults() -> None:
    """Only exact field types are intercepted by the Calculator adapter."""

    class FieldExtension(ElectricField):
        pass

    numbers = torch.tensor([8, 1, 1])
    extension = FieldExtension(torch.ones(3, dtype=torch.double))
    calc = _calc(numbers, interaction=extension)

    assert calc._field_default is None
    assert calc.model.interaction == (extension,)
    assert any(
        component is extension
        for component in calc.system.interactions.components
    )


def test_calculate_forwards_explicit_field_and_grad() -> None:
    """The property dispatcher forwards explicit perturbations once."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    calc = _calc(numbers)
    field = torch.tensor([0.01, -0.03, 0.02], dtype=torch.double)
    field_grad = torch.eye(3, dtype=torch.double) * 0.02

    direct = calc.energy(positions, field=field, field_grad=field_grad)
    dispatched = calc.calculate(
        ["energy"], positions, field=field, field_grad=field_grad
    )["energy"]
    torch.testing.assert_close(dispatched, direct)


def test_legacy_calculator_defaults_overrides_and_reset_graph() -> None:
    """Legacy field defaults resolve per call and survive Calculator.reset."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        dtype=torch.double,
    )
    default = torch.tensor([0.01, 0.02, -0.01], dtype=torch.double)
    override = torch.tensor([-0.02, 0.01, 0.03], dtype=torch.double)
    calc = _calc(numbers, interaction=new_efield(default))
    assert calc._field_default is default
    assert calc.opts.ints.level >= labels.INTLEVEL_DIPOLE
    assert not any(
        type(x) is ElectricField for x in calc.system.interactions.components
    )
    e0 = calc.energy(positions)
    e_override = calc.energy(positions, field=override)
    e_none = calc.energy(positions, field=None)
    e_again = calc.energy(positions)
    torch.testing.assert_close(e0, e_again)
    assert not torch.allclose(e0, e_override)
    assert not torch.allclose(e0, e_none)

    default_grad = torch.tensor(
        [[0.01, 0.0, 0.02], [0.0, -0.01, 0.01], [0.0, 0.0, 0.0]],
        dtype=torch.double,
    )
    calc_grad = _calc(
        numbers,
        interaction=[new_efield(default), new_efield_grad(default_grad)],
    )
    assert calc_grad._field_grad_default is default_grad
    assert calc_grad.opts.ints.level >= labels.INTLEVEL_QUADRUPOLE
    assert not any(
        type(x) is ElectricFieldGrad
        for x in calc_grad.system.interactions.components
    )

    grad_override = torch.triu(default_grad) * -0.5
    grad_default_energy = calc_grad.energy(positions)
    grad_override_energy = calc_grad.energy(positions, field_grad=grad_override)
    grad_none_energy = calc_grad.energy(positions, field_grad=None)
    grad_default_again = calc_grad.energy(positions)
    torch.testing.assert_close(grad_default_energy, grad_default_again)
    assert not torch.allclose(grad_default_energy, grad_override_energy)
    assert not torch.allclose(grad_default_energy, grad_none_energy)

    leaf = torch.tensor(
        [0.02, -0.01, 0.03], dtype=torch.double, requires_grad=True
    )
    calc_graph = _calc(numbers, interaction=new_efield(leaf))
    energy = calc_graph.energy(positions)
    calc_graph.reset()
    repeated = calc_graph.energy(positions)
    (gradient,) = torch.autograd.grad(repeated, leaf)
    torch.testing.assert_close(energy, repeated)
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0

    grad_leaf = torch.eye(3, dtype=torch.double, requires_grad=True)
    calc_grad_graph = _calc(numbers, interaction=new_efield_grad(grad_leaf))
    first = calc_grad_graph.energy(positions)
    (first_grad,) = torch.autograd.grad(first, grad_leaf)
    calc_grad_graph.reset()
    second = calc_grad_graph.energy(positions)
    (second_grad,) = torch.autograd.grad(second, grad_leaf)
    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first_grad, second_grad)


def test_calculator_type_keeps_legacy_field_default_dtype() -> None:
    """Field defaults follow the documented incomplete ``Calculator.type``."""
    numbers = torch.tensor([8, 1, 1])
    field = torch.ones(3, dtype=torch.double)
    calc = _calc(numbers, interaction=new_efield(field))

    calc.type(torch.float32)

    assert calc.dtype == torch.float32
    assert calc._field_default is field
    assert calc._field_default.dtype == torch.double


def test_field_get_cache_is_fresh_and_update_reset_are_rejected() -> None:
    """Field compatibility caches are fresh and exact fields are immutable."""
    numbers = torch.tensor([8, 1, 1])
    positions = torch.randn((3, 3), dtype=torch.double)
    field = torch.ones(3, dtype=torch.double)
    ef = new_efield(field)
    first = ef.get_cache(numbers=numbers, positions=positions)
    second = ef.get_cache(numbers=numbers, positions=positions)
    assert first is not second
    assert ef.cache is None and ef._cachevars is None and ef._cachegrad is None
    with pytest.raises(RuntimeError, match="per-evaluation"):
        ef.update(field=field)
    with pytest.raises(RuntimeError, match="per-evaluation"):
        ef.reset()
    field_list = InteractionList(ef)
    with pytest.raises(RuntimeError, match="per-evaluation"):
        field_list.update("ElectricField", field=field)
    with pytest.raises(RuntimeError, match="per-evaluation"):
        field_list.reset("ElectricField")

    grad = new_efield_grad(torch.ones((3, 3), dtype=torch.double))
    first_grad = grad.get_cache(numbers=numbers, positions=positions)
    second_grad = grad.get_cache(numbers=numbers, positions=positions)
    assert first_grad is not second_grad
    assert (
        grad.cache is None
        and grad._cachevars is None
        and grad._cachegrad is None
    )


def test_duplicate_legacy_field_defaults_are_rejected() -> None:
    """Duplicate exact field defaults fail at the Calculator boundary."""
    numbers = torch.tensor([8, 1, 1])
    with pytest.raises(ValueError, match="Only one exact ElectricField"):
        _calc(
            numbers,
            interaction=[
                new_efield(torch.zeros(3, dtype=torch.double)),
                new_efield(torch.ones(3, dtype=torch.double)),
            ],
        )
    with pytest.raises(ValueError, match="Only one exact ElectricFieldGrad"):
        _calc(
            numbers,
            interaction=[
                new_efield_grad(torch.zeros((3, 3), dtype=torch.double)),
                new_efield_grad(torch.ones((3, 3), dtype=torch.double)),
            ],
        )
