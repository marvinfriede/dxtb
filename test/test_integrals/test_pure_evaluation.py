# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group

from __future__ import annotations

import subprocess
import sys

import pytest
import torch
from torch.func import jacfwd, jvp, vmap

from dxtb import Calculator, GFN0_XTB, GFN1_XTB, GFN2_XTB, labels
from dxtb._src.components.interactions import new_efield
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.evaluation import build_integral_matrices
from dxtb._src.typing import Tensor

from .samples import samples


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_integral_matrices_are_history_independent(driver: int) -> None:
    """One System setup produces current-geometry matrices on every call."""
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and h0_setup is not None
    if setup.libcint is not None:
        assert not hasattr(setup.libcint, "par")
        assert all(
            not hasattr(basis, "par") for basis in setup.libcint.basis_setups
        )

    moved = positions.clone()
    moved[1, 0] += 0.08

    first, refocc1, norm1 = build_integral_matrices(
        setup, h0_setup, positions, charge=0.0
    )
    assert first.overlap.device == positions.device
    assert first.hcore.device == positions.device
    assert refocc1.device == positions.device
    assert norm1.device == positions.device
    second, refocc2, norm2 = build_integral_matrices(
        setup, h0_setup, moved, charge=0.0
    )
    # Mutating compatibility builders in a different order must not affect
    # the next pure evaluation from this System setup.
    calc.integrals.build_quadrupole(moved)
    calc.integrals.build_overlap(positions)
    calc.integrals.build_dipole(positions)
    repeated, refocc3, norm3 = build_integral_matrices(
        setup, h0_setup, positions.clone(), charge=0.0
    )

    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        first_matrix = getattr(first, name)
        second_matrix = getattr(second, name)
        repeated_matrix = getattr(repeated, name)
        assert first_matrix is not None
        assert second_matrix is not None
        assert repeated_matrix is not None
        torch.testing.assert_close(repeated_matrix, first_matrix)
        assert not torch.equal(second_matrix, first_matrix)
    torch.testing.assert_close(refocc1, refocc2)
    torch.testing.assert_close(refocc1, refocc3)
    torch.testing.assert_close(norm1, norm3)
    assert torch.isfinite(norm1).all()
    assert torch.isfinite(norm2).all()

    def loss(geometry: Tensor) -> Tensor:
        matrices, refocc, _ = build_integral_matrices(
            setup, h0_setup, geometry, charge=0.0
        )
        assert matrices.dipole is not None
        assert matrices.quadrupole is not None
        return (
            matrices.overlap.sum()
            + matrices.dipole.sum()
            + matrices.quadrupole.sum()
            + matrices.hcore.sum()
            + refocc.sum()
        )

    leaf1 = positions.clone().requires_grad_()
    grad1 = torch.autograd.grad(loss(leaf1), leaf1)[0]
    leaf2 = positions.clone().requires_grad_()
    grad2 = torch.autograd.grad(loss(leaf2), leaf2)[0]
    assert torch.isfinite(grad1).all()
    assert torch.isfinite(grad2).all()
    torch.testing.assert_close(grad1, grad2)


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_pure_integral_matrices_match_legacy_adapter(driver: int) -> None:
    """The legacy adapter agrees with the pure evaluation outputs.

    This is an adapter-consistency check. Independent physics values are
    covered by the existing end-to-end reference baseline.
    """
    sample = samples["H2O"]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": driver,
            "int_level": labels.INTLEVEL_QUADRUPOLE,
        },
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and h0_setup is not None

    legacy_overlap = calc.integrals.build_overlap(positions)
    assert calc.integrals.overlap is not None
    legacy_norm = calc.integrals.overlap.norm.clone()
    calc.integrals.build_dipole(positions)
    legacy_quadrupole = calc.integrals.build_quadrupole(positions)
    # At quadrupole level the legacy adapter intentionally returns the
    # origin-centered dipole from build_dipole and shifts its stored value
    # only while building the quadrupole.
    assert calc.integrals.dipole is not None
    legacy_dipole = calc.integrals.dipole.matrix
    legacy_hcore = calc.integrals.build_hcore(
        positions, with_overlap=True, charge=0.0
    )
    matrices, refocc, overlap_norm = build_integral_matrices(
        setup, h0_setup, positions, charge=0.0
    )

    torch.testing.assert_close(matrices.overlap, legacy_overlap)
    torch.testing.assert_close(overlap_norm, legacy_norm)
    assert matrices.dipole is not None and calc.integrals.dipole is not None
    torch.testing.assert_close(matrices.dipole, legacy_dipole)
    assert (
        matrices.quadrupole is not None
        and calc.integrals.quadrupole is not None
    )
    torch.testing.assert_close(matrices.quadrupole, legacy_quadrupole)
    torch.testing.assert_close(matrices.hcore, legacy_hcore)
    torch.testing.assert_close(refocc, calc.integrals.hcore.refocc)


@pytest.mark.parametrize(
    "method,level",
    [
        ("gfn0", labels.INTLEVEL_HCORE),
        ("gfn1", labels.INTLEVEL_HCORE),
        ("gfn1", labels.INTLEVEL_DIPOLE),
        ("gfn2", labels.INTLEVEL_HCORE),
        ("gfn2", labels.INTLEVEL_DIPOLE),
        ("gfn2", labels.INTLEVEL_QUADRUPOLE),
    ],
)
@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_pure_integral_levels_match_legacy(
    method: str, level: int, driver: int
) -> None:
    """Each requested level returns exactly the matrix set it promises."""
    methods = {"gfn0": GFN0_XTB, "gfn1": GFN1_XTB, "gfn2": GFN2_XTB}
    numbers = samples["H2"]["numbers"]
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        methods[method],
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": driver,
            "int_level": level,
        },
        auto_int_level=False,
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and h0_setup is not None

    legacy_overlap = calc.integrals.build_overlap(positions)
    legacy_dipole = None
    legacy_quadrupole = None
    if level >= labels.INTLEVEL_DIPOLE:
        legacy_dipole = calc.integrals.build_dipole(positions)
    if level >= labels.INTLEVEL_QUADRUPOLE:
        legacy_quadrupole = calc.integrals.build_quadrupole(positions)
        assert calc.integrals.dipole is not None
        legacy_dipole = calc.integrals.dipole.matrix
    legacy_hcore = calc.integrals.build_hcore(
        positions, with_overlap=True, charge=0.0
    )
    assert calc.integrals.hcore is not None
    legacy_refocc = calc.integrals.hcore.refocc

    matrices, refocc, norm = build_integral_matrices(
        setup, h0_setup, positions, charge=0.0
    )
    assert matrices.overlap.device == positions.device
    assert matrices.hcore.device == positions.device
    assert refocc.device == positions.device
    assert norm.device == positions.device
    torch.testing.assert_close(matrices.overlap, legacy_overlap)
    torch.testing.assert_close(matrices.hcore, legacy_hcore)
    torch.testing.assert_close(refocc, legacy_refocc)
    if level >= labels.INTLEVEL_DIPOLE:
        assert matrices.dipole is not None and legacy_dipole is not None
        torch.testing.assert_close(matrices.dipole, legacy_dipole)
    else:
        assert matrices.dipole is None
    if level >= labels.INTLEVEL_QUADRUPOLE:
        assert matrices.quadrupole is not None
        assert legacy_quadrupole is not None
        torch.testing.assert_close(matrices.quadrupole, legacy_quadrupole)
    else:
        assert matrices.quadrupole is None


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_singlepoint_mirrors_latest_matrices_to_legacy_adapter(
    driver: int,
) -> None:
    """Legacy matrix access reflects each pure singlepoint result."""
    sample = samples["H2"]
    positions = sample["positions"].to(dtype=torch.float64)
    calc = Calculator(
        sample["numbers"],
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    moved = positions.clone()
    moved[1, 0] += 0.12

    first_matrices = None
    for call_index, geometry in enumerate((positions, moved, positions.clone())):
        result = calc.singlepoint(
            geometry,
            chrg=0.0,
            store_overlap=True,
            store_dipole=True,
            store_quadrupole=True,
        )
        assert result.integrals is not None
        adapter_pairs = (
            ("overlap", calc.integrals.overlap),
            ("hcore", calc.integrals.hcore),
            ("dipole", calc.integrals.dipole),
            ("quadrupole", calc.integrals.quadrupole),
        )
        for name, adapter in adapter_pairs:
            expected = getattr(result.integrals, name)
            if expected is None:
                assert adapter is None
                continue
            assert adapter is not None and adapter.matrix is not None
            torch.testing.assert_close(adapter.matrix, expected)
        if call_index == 0:
            first_matrices = result.integrals
        elif call_index == 1:
            assert first_matrices is not None
            assert not torch.equal(
                first_matrices.overlap, result.integrals.overlap
            )
        else:
            assert first_matrices is not None
            for name in ("overlap", "hcore", "dipole", "quadrupole"):
                torch.testing.assert_close(
                    getattr(first_matrices, name),
                    getattr(result.integrals, name),
                )

    # H0 remains on the legacy wrapper even when the default cache policy
    # clears non-gradient overlap/multipole matrices after the call.
    result = calc.singlepoint(moved, chrg=0.0)
    assert result.integrals is not None
    assert calc.integrals.hcore is not None
    torch.testing.assert_close(calc.integrals.hcore.matrix, result.integrals.hcore)


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_hcore_adapter_does_not_retain_consumed_position_graph(
    driver: int,
) -> None:
    """Two differentiated calls leave H0 mirroring on the latest geometry."""
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        samples["H2"]["numbers"],
        GFN1_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    first_positions = positions.clone().requires_grad_(True)
    first = calc.singlepoint(first_positions, chrg=0.0)
    first_gradient = torch.autograd.grad(first.total.sum(), first_positions)[0]

    second_positions = positions.clone()
    second_positions[1, 0] += 0.09
    second_positions.requires_grad_()
    second = calc.singlepoint(second_positions, chrg=0.0)
    second_gradient = torch.autograd.grad(second.total.sum(), second_positions)[
        0
    ]

    assert torch.isfinite(first_gradient).all()
    assert torch.isfinite(second_gradient).all()
    assert second.integrals is not None
    assert calc.integrals.hcore is not None
    torch.testing.assert_close(
        calc.integrals.hcore.matrix, second.integrals.hcore
    )


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
def test_fresh_system_rebuild_preserves_basis_parameter_gradients(
    driver: int,
) -> None:
    """Basis-derived graphs are rebuilt for each parameter-training loss."""
    from dxtb import ParamModule

    numbers = samples["H2"]["numbers"]
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    parameter = dict(calc.model.par.named_parameters())[
        "parameter_tree.element.H.slater.param"
    ]
    parameter.requires_grad_(True)

    gradients = []
    for _ in range(2):
        system = calc.model.setup(numbers, dd=calc.dd)
        assert system.integral_setup is not None
        assert system.h0_setup is not None
        matrices, _, _ = build_integral_matrices(
            system.integral_setup,
            system.h0_setup,
            positions,
            charge=0.0,
        )
        loss = matrices.overlap.sum() + matrices.hcore.sum()
        (gradient,) = torch.autograd.grad(loss, parameter)
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
        gradients.append(gradient)
    torch.testing.assert_close(gradients[0], gradients[1])


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(labels.INTDRIVER_PYTORCH, id="pytorch"),
        pytest.param(
            labels.INTDRIVER_LIBCINT,
            id="libcint",
            marks=pytest.mark.skipif(
                not has_libcint, reason="libcint not available"
            ),
        ),
    ],
)
@pytest.mark.parametrize("property_name", ["forces", "dipole", "quadrupole"])
def test_analytical_properties_after_pure_singlepoint(
    driver: int, property_name: str
) -> None:
    """Analytical compatibility APIs remain usable after pure evaluation."""
    if (
        property_name == "forces"
        and driver == labels.INTDRIVER_PYTORCH
    ):
        pytest.skip(
            "PyTorch overlap has no analytical gradient; use its autograd path"
        )
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    interactions = None
    if property_name == "dipole":
        interactions = [new_efield(torch.zeros(3, dtype=positions.dtype))]
    calc = Calculator(
        samples["H2"]["numbers"],
        GFN1_XTB if property_name == "forces" else GFN2_XTB,
        interaction=interactions,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": driver},
    )
    calc.singlepoint(
        positions,
        chrg=0.0,
        store_overlap=True,
        store_dipole=True,
        store_quadrupole=True,
    )

    if property_name == "forces":
        value = calc.forces_analytical(positions.clone().requires_grad_())
    elif property_name == "dipole":
        value = calc.dipole_analytical(positions)
    else:
        value = calc.quadrupole_analytical(positions)
    assert torch.isfinite(value).all()


def test_dxtb_and_pytorch_integrals_do_not_require_libcint_import() -> None:
    """The PyTorch core remains usable when tad_libcint is unimportable."""
    script = r'''
import importlib.abc
import sys

class BlockLibcint(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tad_libcint" or fullname.startswith("tad_libcint."):
            raise ModuleNotFoundError("blocked optional tad_libcint")
        return None

sys.meta_path.insert(0, BlockLibcint())
import torch
import dxtb
from dxtb._src.integral.evaluation import build_integral_matrices

numbers = torch.tensor([1, 1])
positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64)
calc = dxtb.Calculator(
    numbers,
    dxtb.GFN1_XTB,
    dtype=positions.dtype,
    opts={"verbosity": 0, "int_driver": dxtb.labels.INTDRIVER_PYTORCH},
)
assert calc.system.integral_setup is not None
assert calc.system.integral_setup.pytorch is not None
assert calc.system.integral_setup.libcint is None
matrices, _, _ = build_integral_matrices(
    calc.system.integral_setup, calc.system.h0_setup, positions, charge=0.0
)
assert torch.isfinite(matrices.overlap).all()
assert "tad_libcint" not in sys.modules
'''
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("batch_mode", [1, 2])
def test_batched_calculator_keeps_legacy_integral_adapter(
    batch_mode: int,
) -> None:
    """Existing batch modes remain on Calculator-local legacy builders."""
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.55]],
        ],
        dtype=torch.float64,
    )
    numbers = torch.tensor([[1, 1], [1, 1]])
    calc = Calculator(
        numbers,
        GFN1_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
        batch_mode=batch_mode,
    )
    assert calc.system.integral_setup is None
    assert calc.integrals is not None

    charges = torch.tensor([0.0, 1.0], dtype=positions.dtype)
    result = calc.singlepoint(positions, chrg=charges)
    assert torch.isfinite(result.total).all()

    separate = []
    for geometry, charge in zip(positions, charges):
        single = Calculator(
            torch.tensor([1, 1]),
            GFN1_XTB,
            dtype=positions.dtype,
            opts={"verbosity": 0, "int_driver": labels.INTDRIVER_PYTORCH},
        )
        separate.append(
            single.singlepoint(geometry, chrg=float(charge)).total.sum()
        )
    torch.testing.assert_close(result.total.sum(-1), torch.stack(separate))
    assert not torch.isclose(result.total[0].sum(), result.total[1].sum())


def test_pure_integral_evaluation_supports_torch_transforms() -> None:
    """The full single-system transform pipeline supports torch.func."""
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        samples["H2"]["numbers"],
        GFN2_XTB,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_PYTORCH,
            "int_level": labels.INTLEVEL_QUADRUPOLE,
        },
    )
    setup = calc.system.integral_setup
    h0_setup = calc.system.h0_setup
    assert setup is not None and setup.pytorch is not None
    assert h0_setup is not None

    def evaluate(geometry: Tensor) -> tuple[Tensor, ...]:
        matrices, refocc, norm = build_integral_matrices(
            setup, h0_setup, geometry, charge=0.0
        )
        assert matrices.dipole is not None
        assert matrices.quadrupole is not None
        return (
            matrices.overlap,
            matrices.hcore,
            matrices.dipole,
            matrices.quadrupole,
            refocc,
            norm,
        )

    def scalar(geometry: Tensor) -> Tensor:
        return sum(
            (index + 1) * matrix.square().sum()
            for index, matrix in enumerate(evaluate(geometry))
        )

    leaf = positions.clone().requires_grad_(True)
    first = torch.autograd.grad(scalar(leaf), leaf, create_graph=True)[0]
    second = torch.autograd.grad(first.sum(), leaf, create_graph=True)[0]
    third = torch.autograd.grad(second.sum(), leaf)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()

    primal, tangent = jvp(evaluate, (positions,), (torch.ones_like(positions),))
    assert all(torch.isfinite(value).all() for value in primal)
    assert all(torch.isfinite(value).all() for value in tangent)

    jacobian = jacfwd(scalar)(positions)
    assert torch.isfinite(jacobian).all()

    geometries = torch.stack((positions, positions.clone()))
    mapped = vmap(evaluate)(geometries)
    loop_outputs = [evaluate(geometry) for geometry in geometries]
    looped = tuple(
        torch.stack([output[index] for output in loop_outputs])
        for index in range(len(mapped))
    )
    for mapped_value, looped_value in zip(mapped, looped):
        torch.testing.assert_close(mapped_value, looped_value)
