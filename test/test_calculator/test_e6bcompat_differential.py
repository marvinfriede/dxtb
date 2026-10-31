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
"""Differential controls for public packed and private mixed-System paths."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

import pytest
import torch

from dxtb import (
    GFN1_XTB,
    GFN2_XTB,
    Calculator,
    OutputHandler,
    ParamModule,
    labels,
)
from dxtb._src.calculators.canonical_padding import (
    bucket_canonical_exact_cores,
    canonicalize_positions,
    pad_canonical_exact_core,
)
from dxtb._src.calculators.config.main import Config
from dxtb._src.calculators.model import Model
from dxtb._src.scf.functional.padded import evaluate_padded_systems

DTYPE = torch.float64
DEVICE = torch.device("cpu")

_SETTINGS = {
    "int_driver": labels.INTDRIVER_PYTORCH,
    "int_level": labels.INTLEVEL_QUADRUPOLE,
    "scf_mode": labels.SCF_MODE_FULL,
    "scp_mode": labels.SCP_MODE_CHARGE,
    "mixer": labels.MIXER_LINEAR,
    "maxiter": 12,
    "damp": 0.3,
    "fermi_etemp": 0.0,
}


def _geometry(numbers: tuple[int, ...], variant: int = 0) -> torch.Tensor:
    """Return stable float64 fixtures for the public and private paths."""
    coordinates = {
        (1, 1): [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.42]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.68]],
        ],
        (1, 1, 1): [
            [[0.0, 0.0, 0.0], [0.0, 1.4, 1.0], [0.0, -1.4, 1.0]],
        ],
        (8, 1, 1): [
            [[0.0, 0.0, 0.0], [0.82, 0.0, 0.61], [-0.82, 0.0, 0.61]],
        ],
        (6, 1, 1, 1, 1): [
            [
                [0.0, 0.0, 0.0],
                [0.629, 0.629, 0.629],
                [0.629, -0.629, -0.629],
                [-0.629, 0.629, -0.629],
                [-0.629, -0.629, 0.629],
            ],
        ],
    }
    return torch.tensor(coordinates[numbers][variant], dtype=DTYPE)


def _pad_public_inputs(
    compositions: Sequence[tuple[int, ...]],
    source_positions: Sequence[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    max_atoms = max(map(len, compositions))
    numbers = torch.zeros((len(compositions), max_atoms), dtype=torch.int64)
    positions = torch.zeros((len(compositions), max_atoms, 3), dtype=DTYPE)
    for lane, (composition, geometry) in enumerate(
        zip(compositions, source_positions)
    ):
        count = len(composition)
        numbers[lane, :count] = torch.tensor(composition, dtype=torch.int64)
        positions[lane, :count] = geometry
    return numbers, positions


def _private_mapped(
    method: int,
    compositions: Sequence[tuple[int, ...]],
    positions: Sequence[torch.Tensor],
    charges: Sequence[float],
    spins: Sequence[float],
    *,
    field: torch.Tensor | None = None,
    field_grad: torch.Tensor | None = None,
    settings: dict | None = None,
):
    param = ParamModule(
        GFN1_XTB if method == labels.GFN1_XTB else GFN2_XTB,
        device=DEVICE,
        dtype=DTYPE,
    )
    config = Config.create(method=method, **(_SETTINGS | (settings or {})))
    model = Model(par=param, config=config, auto_int_level=False)
    systems = tuple(
        model.setup(torch.tensor(composition, dtype=torch.int64))
        for composition in compositions
    )
    cores = tuple(system.exact_core for system in systems)
    assert all(core is not None for core in cores)
    bucket = bucket_canonical_exact_cores(cores)
    padded = tuple(
        pad_canonical_exact_core(core, bucket, lane)
        for lane, core in enumerate(cores)
    )
    canonical_positions = tuple(
        canonicalize_positions(position, bucket, lane)
        for lane, position in enumerate(positions)
    )
    charge_tensors = tuple(
        torch.tensor(value, dtype=DTYPE) for value in charges
    )
    spin_tensors = tuple(torch.tensor(value, dtype=DTYPE) for value in spins)
    mapped, decoded = evaluate_padded_systems(
        padded,
        bucket,
        canonical_positions,
        charge_tensors,
        spin_tensors,
        field=field,
        field_grad=field_grad,
    )
    serial = tuple(
        system.singlepoint(
            position,
            charge,
            spin,
            field=field,
            field_grad=field_grad,
        )
        for system, position, charge, spin in zip(
            systems, positions, charge_tensors, spin_tensors
        )
    )
    return param, systems, bucket, mapped, decoded, serial


def _assert_vector_row(
    public: torch.Tensor | None,
    private: torch.Tensor | None,
    lane: int,
    extent: int,
) -> None:
    if private is None:
        assert public is None
        return
    assert public is not None
    torch.testing.assert_close(
        public.select(0, lane)[..., :extent], private, atol=2.0e-9, rtol=2.0e-9
    )


def _assert_matrix_row(
    public: torch.Tensor | None,
    private: torch.Tensor | None,
    lane: int,
    extent: int,
) -> None:
    if private is None:
        assert public is None
        return
    assert public is not None
    row = public.select(0, lane)
    torch.testing.assert_close(
        row[..., :extent, :extent], private, atol=2.0e-9, rtol=2.0e-9
    )


def _assert_packed_decoded_parity(
    public, decoded, systems, compositions: Sequence[tuple[int, ...]]
) -> None:
    count = len(compositions)
    max_atoms = max(map(len, compositions))
    assert public.energy.shape == (count, max_atoms)
    assert public.scf.shape == (count, max_atoms)
    assert tuple(name for name, _ in public.classical) == tuple(
        name for name, _ in decoded[0].classical
    )
    assert public.status is None  # Legacy packed status is not SCFStatus.
    assert public.converged is None and public.residual is None
    public_iterations = torch.as_tensor(public.iterations)
    assert public_iterations.ndim == 0
    for lane, (physical, system, composition) in enumerate(
        zip(decoded, systems, compositions)
    ):
        atoms = len(composition)
        orbitals = system.ihelp.nao
        _assert_vector_row(public.energy, physical.energy, lane, atoms)
        _assert_vector_row(public.scf, physical.scf, lane, atoms)
        _assert_vector_row(public.fenergy, physical.fenergy, lane, atoms)
        if atoms < max_atoms:
            assert torch.count_nonzero(public.energy[lane, atoms:]) == 0
            assert torch.count_nonzero(public.scf[lane, atoms:]) == 0
            for _, values in public.classical:
                assert torch.count_nonzero(values[lane, atoms:]) == 0
        torch.testing.assert_close(
            public_iterations,
            torch.as_tensor(physical.iterations),
        )
        for name, private_value in physical.classical:
            public_value = dict(public.classical)[name]
            _assert_vector_row(public_value, private_value, lane, atoms)
        for field_name in ("density", "hamiltonian", "overlap", "hcore"):
            _assert_matrix_row(
                getattr(public, field_name),
                getattr(physical, field_name),
                lane,
                orbitals,
            )
        for field_name in (
            "dipole_integrals",
            "quadrupole_integrals",
        ):
            _assert_matrix_row(
                getattr(public, field_name),
                getattr(physical, field_name),
                lane,
                orbitals,
            )
        for field_name in ("emo", "occupation", "overlap_norm"):
            _assert_vector_row(
                getattr(public, field_name),
                getattr(physical, field_name),
                lane,
                orbitals,
            )
        assert public.charges is not None and physical.charges is not None
        _assert_vector_row(
            public.charges.mono, physical.charges.mono, lane, orbitals
        )
        assert public.potential is not None and physical.potential is not None
        _assert_vector_row(
            public.potential.mono, physical.potential.mono, lane, orbitals
        )
        torch.testing.assert_close(
            torch.as_tensor(public.energy[lane, :atoms].sum()),
            physical.energy.sum(),
            atol=2.0e-9,
            rtol=2.0e-9,
        )
        assert physical.status is not None


@pytest.mark.parametrize(
    "compositions",
    [
        ((1, 1), (1, 1)),
        ((1, 1), (1, 1, 1)),
        ((1, 1), (8, 1, 1)),
        ((8, 1, 1), (1, 1)),
        ((8, 1, 1), (6, 1, 1, 1, 1)),
    ],
    ids=["h2-h2", "h2-h3plus", "h2-water", "water-h2", "water-ch4"],
)
def test_public_gfn1_mode1_matches_serial_and_private_mapped(
    compositions: tuple[tuple[int, ...], tuple[int, ...]],
) -> None:
    """Compare packed mode 1 with independent serial and mapped Results."""
    source_positions = tuple(_geometry(numbers) for numbers in compositions)
    charges = (0.0, 1.0) if compositions[1] == (1, 1, 1) else (0.0, 0.0)
    spins = (0.0, 0.0) if charges[1] == 1.0 else (0.0, 0.0)
    param, systems, _, _, decoded, serial = _private_mapped(
        labels.GFN1_XTB,
        compositions,
        source_positions,
        charges,
        spins,
    )
    public_numbers, public_positions = _pad_public_inputs(
        compositions, source_positions
    )
    opts = dict(_SETTINGS, method=labels.GFN1_XTB, verbosity=0)
    calculator = Calculator(
        public_numbers,
        param,
        opts=opts,
        batch_mode=1,
        dtype=DTYPE,
        auto_int_level=False,
    )
    public = calculator.singlepoint(
        public_positions,
        torch.tensor(charges, dtype=DTYPE),
        torch.tensor(spins, dtype=DTYPE),
    )
    _assert_packed_decoded_parity(public, decoded, systems, compositions)
    for actual, reference in zip(decoded, serial):
        torch.testing.assert_close(
            actual.energy, reference.energy, atol=2.0e-9, rtol=2.0e-9
        )
        torch.testing.assert_close(actual.scf, reference.scf)

    batch_charge = torch.tensor(charges, dtype=DTYPE)
    batch_spin = torch.tensor(spins, dtype=DTYPE)
    energy = calculator.energy(public_positions, batch_charge, batch_spin)
    energy_keepdim = calculator.energy(
        public_positions, batch_charge, batch_spin, keepdim=True
    )
    assert energy.shape == (2,)
    assert energy_keepdim.shape == (2, 1)
    torch.testing.assert_close(energy, public.energy.sum(dim=-1))
    torch.testing.assert_close(energy_keepdim[:, 0], energy)


def test_public_gfn1_mode2_matches_serial_and_private_mapped() -> None:
    """Compare same-composition packed conformers and force derivatives."""
    compositions = ((1, 1), (1, 1))
    source_positions = (
        _geometry((1, 1), 0).requires_grad_(),
        _geometry((1, 1), 1).requires_grad_(),
    )
    charges = (0.0, 0.0)
    spins = (0.0, 0.0)
    param, systems, _, _, decoded, serial = _private_mapped(
        labels.GFN1_XTB,
        compositions,
        source_positions,
        charges,
        spins,
    )
    public_numbers, public_positions = _pad_public_inputs(
        compositions, source_positions
    )
    public_positions.requires_grad_()
    opts = dict(_SETTINGS, method=labels.GFN1_XTB, verbosity=0)
    calculator = Calculator(
        public_numbers,
        param,
        opts=opts,
        batch_mode=2,
        dtype=DTYPE,
        auto_int_level=False,
    )
    public = calculator.singlepoint(
        public_positions,
        torch.tensor(charges, dtype=DTYPE),
        torch.tensor(spins, dtype=DTYPE),
    )
    _assert_packed_decoded_parity(public, decoded, systems, compositions)
    public_total = public.energy.sum(dim=-1)
    (public_gradient,) = torch.autograd.grad(
        public_total.sum(), public_positions
    )
    mapped_gradients = torch.autograd.grad(
        torch.stack(tuple(result.energy.sum() for result in decoded)).sum(),
        tuple(source_positions),
    )
    torch.testing.assert_close(
        public_gradient,
        torch.stack(mapped_gradients),
        atol=2.0e-8,
        rtol=2.0e-8,
    )
    assert torch.count_nonzero(public_gradient) > 0
    for actual, reference in zip(decoded, serial):
        torch.testing.assert_close(actual.energy, reference.energy)


def test_public_gfn2_packed_fock_matches_private_and_serial() -> None:
    """The legacy GFN2 Fock policy preserves physical packed values."""
    compositions = ((1, 1), (8, 1, 1))
    source_positions = tuple(_geometry(numbers) for numbers in compositions)
    settings = _SETTINGS | {"scp_mode": labels.SCP_MODE_FOCK}
    param, systems, _, _, decoded, serial = _private_mapped(
        labels.GFN2_XTB,
        compositions,
        source_positions,
        (0.0, 0.0),
        (0.0, 0.0),
        settings=settings,
    )
    public_numbers, public_positions = _pad_public_inputs(
        compositions, source_positions
    )
    opts = dict(
        settings,
        method=labels.GFN2_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        verbosity=0,
    )
    calculator = Calculator(
        public_numbers,
        param,
        opts=opts,
        batch_mode=1,
        dtype=DTYPE,
        auto_int_level=False,
    )
    OutputHandler.clear_warnings()
    public = calculator.singlepoint(
        public_positions,
        torch.zeros(2, dtype=DTYPE),
        torch.zeros(2, dtype=DTYPE),
    )
    _assert_packed_decoded_parity(public, decoded, systems, compositions)
    assert not any(
        "Changing to Fock matrix automatically" in message
        for message, _ in OutputHandler.warnings
    )
    for actual, reference in zip(decoded, serial):
        torch.testing.assert_close(
            actual.energy, reference.energy, atol=5.0e-8, rtol=5.0e-8
        )
        torch.testing.assert_close(actual.density, reference.density)


def test_public_shared_fields_match_serial_and_private_mode2() -> None:
    """Shared field perturbations preserve values and response gradients."""
    compositions = ((1, 1), (1, 1))
    source_positions = (_geometry((1, 1), 0), _geometry((1, 1), 1))
    field = torch.tensor(
        [0.004, -0.003, 0.002], dtype=DTYPE, requires_grad=True
    )
    field_grad = (torch.eye(3, dtype=DTYPE) * 0.001).requires_grad_()
    charges = (0.0, 0.0)
    spins = (0.0, 0.0)
    param, systems, _, _, decoded, serial = _private_mapped(
        labels.GFN1_XTB,
        compositions,
        source_positions,
        charges,
        spins,
        field=field,
        field_grad=field_grad,
    )
    numbers, positions = _pad_public_inputs(compositions, source_positions)
    opts = dict(_SETTINGS, method=labels.GFN1_XTB, verbosity=0)
    calculator = Calculator(
        numbers,
        param,
        opts=opts,
        batch_mode=2,
        dtype=DTYPE,
        auto_int_level=False,
    )
    public = calculator.singlepoint(
        positions,
        torch.tensor(charges, dtype=DTYPE),
        torch.tensor(spins, dtype=DTYPE),
        field=field,
        field_grad=field_grad,
    )
    _assert_packed_decoded_parity(public, decoded, systems, compositions)
    for actual, reference in zip(decoded, serial):
        torch.testing.assert_close(actual.energy, reference.energy)
    public_field_grad = torch.autograd.grad(
        public.energy.sum(), (field, field_grad), retain_graph=True
    )
    mapped_field_grad = torch.autograd.grad(
        torch.stack(tuple(result.energy.sum() for result in decoded)).sum(),
        (field, field_grad),
        retain_graph=True,
    )
    for actual, reference in zip(public_field_grad, mapped_field_grad):
        torch.testing.assert_close(actual, reference, atol=2.0e-8, rtol=2.0e-8)
        assert torch.count_nonzero(actual) > 0


@pytest.mark.parametrize(
    "scp_mode", [labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL]
)
@pytest.mark.parametrize("strict", [False, True], ids=["warn", "strict"])
@pytest.mark.parametrize(
    "mixer",
    [labels.MIXER_LINEAR, labels.MIXER_ANDERSON],
    ids=["linear", "anderson"],
)
def test_public_packed_gfn2_nonfock_policy_is_preserved(
    scp_mode: int,
    strict: bool,
    mixer: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Packed GFN2 retains legacy warning/switch and strict rejection."""
    numbers = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    positions = torch.stack((_geometry((1, 1), 0), _geometry((1, 1), 1)))
    opts = {
        "method": labels.GFN2_XTB,
        "int_driver": labels.INTDRIVER_LIBCINT,
        "int_level": labels.INTLEVEL_QUADRUPOLE,
        "scf_mode": labels.SCF_MODE_FULL,
        "scp_mode": scp_mode,
        "mixer": mixer,
        "maxiter": 2,
        "strict": strict,
        "verbosity": 0,
    }
    calculator = Calculator(
        numbers,
        GFN2_XTB,
        opts=opts,
        batch_mode=2,
        dtype=DTYPE,
    )
    OutputHandler.clear_warnings()
    if strict:
        with pytest.raises(NotImplementedError, match="only Fock matrix"):
            calculator.singlepoint(positions)
        assert not OutputHandler.warnings
        return

    from dxtb._src.scf import unrolling

    original = unrolling.SelfConsistentFieldFull
    seen: list[int] = []

    def inspect_after_init(*args, **kwargs):
        engine = original(*args, **kwargs)
        seen.append(engine.config.scp_mode)
        return engine

    monkeypatch.setattr(
        unrolling, "SelfConsistentFieldFull", inspect_after_init
    )
    result = calculator.singlepoint(positions)
    assert result.energy.shape == (2, 2)
    assert seen == [labels.SCP_MODE_FOCK]
    messages = tuple(message for message, _ in OutputHandler.warnings)
    assert any("Changing to Fock matrix automatically" in m for m in messages)


def test_public_packed_gfn2_fock_route_has_no_nonfock_warning() -> None:
    """The supported packed GFN2 Fock configuration has no switch warning."""
    numbers = torch.tensor([[1, 1], [1, 1]], dtype=torch.int64)
    positions = torch.stack((_geometry((1, 1), 0), _geometry((1, 1), 1)))
    calculator = Calculator(
        numbers,
        GFN2_XTB,
        opts={
            "method": labels.GFN2_XTB,
            "int_driver": labels.INTDRIVER_LIBCINT,
            "int_level": labels.INTLEVEL_QUADRUPOLE,
            "scf_mode": labels.SCF_MODE_FULL,
            "scp_mode": labels.SCP_MODE_FOCK,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "verbosity": 0,
        },
        batch_mode=2,
        dtype=DTYPE,
    )
    OutputHandler.clear_warnings()
    result = calculator.singlepoint(positions)
    assert result.energy.shape == (2, 2)
    assert not any(
        "Changing to Fock matrix automatically" in message
        for message, _ in OutputHandler.warnings
    )


def test_missing_libcint_dependency_is_not_replaced_by_pytorch() -> None:
    """An unavailable explicit libcint request propagates its import error."""
    script = r"""
import builtins
original_import = builtins.__import__
def without_tad_libcint(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "tad_libcint" or name.startswith("tad_libcint."):
        raise ModuleNotFoundError("No module named 'tad_libcint'", name="tad_libcint")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = without_tad_libcint
import torch
from dxtb import Calculator, GFN1_XTB, OutputHandler, labels
from dxtb._src.calculators.config.main import Config
try:
    Calculator(
        torch.tensor([1, 1], dtype=torch.int64),
        GFN1_XTB,
        opts={
            "method": labels.GFN1_XTB,
            "int_driver": "libcint",
            "int_level": labels.INTLEVEL_QUADRUPOLE,
            "scf_mode": labels.SCF_MODE_FULL,
            "scp_mode": labels.SCP_MODE_CHARGE,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 2,
            "verbosity": 0,
        },
        dtype=torch.float64,
    )
except ValueError as exc:
    assert "interface is not installed" in str(exc)
else:
    raise AssertionError("explicit libcint request was not rejected")
OutputHandler.clear_warnings()
default_config = Config.create(
    method=labels.GFN1_XTB,
    int_driver=labels.INTDRIVER_LIBCINT,
)
assert default_config.ints.driver == labels.INTDRIVER_PYTORCH
assert any("Falling back to the PyTorch driver" in message for message, _ in OutputHandler.warnings)
print("EXPLICIT_REJECT_AND_DEFAULT_WARNED_FALLBACK")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "EXPLICIT_REJECT_AND_DEFAULT_WARNED_FALLBACK" in completed.stdout


@pytest.mark.parametrize("method", [labels.GFN1_XTB, labels.GFN2_XTB])
def test_pack_system_results_and_evaluate_packed_systems_parity(
    method: int,
) -> None:
    """Candidate E6broute packed adapter matches public Calculator and serial Results."""
    from dxtb._src.calculators.packed_result import (
        evaluate_packed_systems,
        pack_system_results,
    )

    compositions = ((1, 1), (8, 1, 1), (6, 1, 1, 1, 1))
    source_positions = tuple(
        _geometry(comp, 0).clone().requires_grad_(True) for comp in compositions
    )
    charges = (0.0, 0.0, 0.0)
    spins = (0.0, 0.0, 0.0)
    field = torch.tensor(
        [0.0015, -0.001, 0.002], dtype=DTYPE, requires_grad=True
    )
    field_grad = torch.tensor(
        [
            [0.0005, 0.0002, -0.0001],
            [0.0002, -0.0003, 0.0001],
            [-0.0001, 0.0001, -0.0002],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    extra_settings = (
        {"scp_mode": labels.SCP_MODE_FOCK} if method == labels.GFN2_XTB else {}
    )
    param, systems, _, _, decoded, serial = _private_mapped(
        method,
        compositions,
        source_positions,
        charges,
        spins,
        field=field,
        field_grad=field_grad,
        settings=extra_settings,
    )
    numbers, padded_positions = _pad_public_inputs(
        compositions, [p.detach() for p in source_positions]
    )
    opts = dict(_SETTINGS, **extra_settings, method=method, verbosity=0)
    calculator = Calculator(
        numbers,
        param,
        opts=opts,
        batch_mode=1,
        dtype=DTYPE,
        auto_int_level=False,
    )
    public = calculator.singlepoint(
        padded_positions,
        torch.tensor(charges, dtype=DTYPE),
        torch.tensor(spins, dtype=DTYPE),
        field=field,
        field_grad=field_grad,
    )

    packed_compat = evaluate_packed_systems(
        systems,
        source_positions,
        charges,
        spins,
        field=field,
        field_grad=field_grad,
        legacy_compat=True,
    )
    _assert_packed_decoded_parity(packed_compat, decoded, systems, compositions)
    torch.testing.assert_close(
        packed_compat.energy, public.energy, atol=2.0e-9, rtol=2.0e-9
    )

    packed_functional = evaluate_packed_systems(
        systems,
        source_positions,
        charges,
        spins,
        field=field,
        field_grad=field_grad,
        legacy_compat=False,
    )
    assert packed_functional.iterations.shape == (3,)
    assert packed_functional.converged is not None
    assert packed_functional.converged.shape == (3,)
    assert packed_functional.residual is not None
    assert packed_functional.residual.shape[0] == 3
    assert packed_functional.status is not None
    assert packed_functional.status.trajectory_failure.shape == (3,)
    assert packed_functional.status.final_failure.shape == (3,)

    # Direct pack_system_results on serial results matches packed_functional
    repacked_serial = pack_system_results(serial, legacy_compat=False)
    torch.testing.assert_close(
        packed_functional.energy,
        repacked_serial.energy,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert packed_functional.multipoles is not None

    # Cross-lane derivative isolation and field/field_grad derivative parity
    for lane_idx in range(len(compositions)):
        grads = torch.autograd.grad(
            packed_functional.energy[lane_idx].sum(),
            source_positions,
            retain_graph=True,
        )
        for other_idx, g in enumerate(grads):
            if other_idx == lane_idx:
                ref_g = torch.autograd.grad(
                    serial[lane_idx].energy.sum(),
                    source_positions[lane_idx],
                    retain_graph=True,
                )[0]
                torch.testing.assert_close(g, ref_g, atol=1.0e-10, rtol=1.0e-10)
            else:
                assert float(g.abs().max()) == 0.0

    packed_fg = torch.autograd.grad(
        packed_functional.energy.sum(), (field, field_grad), retain_graph=True
    )
    public_fg = torch.autograd.grad(
        public.energy.sum(), (field, field_grad), retain_graph=True
    )
    for actual, expected in zip(packed_fg, public_fg):
        torch.testing.assert_close(actual, expected, atol=2.0e-8, rtol=2.0e-8)


def test_evaluate_packed_systems_mixed_buckets_and_retained_routes() -> None:
    """evaluate_packed_systems coordinates multi-bucket MAPPED and retained lanes."""
    from dxtb._src.calculators.legacy_composition import (
        LegacySCFOptions,
        LegacySystemRoute,
        classify_legacy_system,
    )
    from dxtb._src.calculators.packed_result import evaluate_packed_systems

    param_gfn1 = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    param_gfn2 = ParamModule(GFN2_XTB, device=DEVICE, dtype=DTYPE)

    cfg_gfn1_mapped = Config.create(method=labels.GFN1_XTB, **_SETTINGS)
    cfg_gfn2_mapped = Config.create(
        method=labels.GFN2_XTB,
        **(_SETTINGS | {"scp_mode": labels.SCP_MODE_FOCK}),
    )
    cfg_gfn1_legacy = Config.create(
        method=labels.GFN1_XTB,
        **(_SETTINGS | {"scf_mode": labels.SCF_MODE_IMPLICIT}),
    )

    sys0 = Model(
        par=param_gfn1, config=cfg_gfn1_mapped, auto_int_level=False
    ).setup(torch.tensor([1, 1], dtype=torch.int64))
    sys1 = Model(
        par=param_gfn1, config=cfg_gfn1_legacy, auto_int_level=False
    ).setup(torch.tensor([8, 1, 1], dtype=torch.int64))
    sys2 = Model(
        par=param_gfn2, config=cfg_gfn2_mapped, auto_int_level=False
    ).setup(torch.tensor([1, 1, 1], dtype=torch.int64))

    assert classify_legacy_system(sys0) is LegacySystemRoute.MAPPED
    assert classify_legacy_system(sys1) is LegacySystemRoute.LEGACY_COMPOSED
    assert classify_legacy_system(sys2) is LegacySystemRoute.MAPPED

    systems = (sys0, sys1, sys2)
    positions = (
        _geometry((1, 1), 0),
        _geometry((8, 1, 1), 0),
        _geometry((1, 1, 1), 0),
    )
    charges = (0.0, 0.0, 1.0)
    spins = (0.0, 0.0, 0.0)

    packed = evaluate_packed_systems(systems, positions, charges, spins)
    serial = tuple(
        s.singlepoint(p, c, sp)
        for s, p, c, sp in zip(systems, positions, charges, spins)
    )
    assert packed.energy.shape == (3, 3)
    for idx, ref in enumerate(serial):
        nat = ref.energy.numel()
        torch.testing.assert_close(
            packed.energy[idx, :nat], ref.energy, atol=1.0e-10, rtol=1.0e-10
        )

    # E6bcompat strict vs warned GFN2 non-Fock and unknown kwargs
    cfg_gfn2_charge = Config.create(
        method=labels.GFN2_XTB,
        **(_SETTINGS | {"scp_mode": labels.SCP_MODE_CHARGE}),
    )
    sys_gfn2_charge = Model(
        par=param_gfn2, config=cfg_gfn2_charge, auto_int_level=False
    ).setup(torch.tensor([1, 1], dtype=torch.int64))

    with pytest.raises(NotImplementedError, match="Fock-mode SCF"):
        evaluate_packed_systems(
            (sys_gfn2_charge,),
            (_geometry((1, 1), 0),),
            strict_e6bcompat=True,
        )

    OutputHandler.clear_warnings()
    warned_res = evaluate_packed_systems(
        (sys_gfn2_charge,),
        (_geometry((1, 1), 0),),
        strict_e6bcompat=False,
    )
    assert warned_res.energy.shape == (1, 2)
    assert any(
        "Resetting 'scp_mode' to 'fock'" in msg
        for msg, _ in OutputHandler.warnings
    )

    with pytest.raises(TypeError, match="unexpected keyword arguments"):
        evaluate_packed_systems(
            (sys0,),
            (_geometry((1, 1), 0),),
            legacy_options=LegacySCFOptions(kwargs={"unknown_flag": True}),
            strict_e6bcompat=True,
        )

    # Open-shell (odd-electron OH.) with default spin=None and mixed SCF/non-SCF occupation packing
    cfg_gfn1_open = Config.create(
        method=labels.GFN1_XTB,
        **(_SETTINGS | {"fermi_etemp": 300.0}),
    )
    sys_oh = Model(
        par=param_gfn1, config=cfg_gfn1_open, auto_int_level=False
    ).setup(torch.tensor([8, 1], dtype=torch.int64))
    cfg_non_scf = Config.create(
        method=labels.GFN1_XTB,
        **(_SETTINGS | {"exclude": ["scf"]}),
    )
    sys_non_scf = Model(
        par=param_gfn1, config=cfg_non_scf, auto_int_level=False
    ).setup(torch.tensor([1, 1], dtype=torch.int64))
    oh_pos = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.83]], dtype=DTYPE)
    h2_pos = _geometry((1, 1), 0)
    packed_open_and_nonscf = evaluate_packed_systems(
        (sys_oh, sys_non_scf),
        (oh_pos, h2_pos),
        legacy_compat=False,
    )
    ref_oh = sys_oh.singlepoint(oh_pos)
    ref_non_scf = sys_non_scf.singlepoint(h2_pos)
    torch.testing.assert_close(
        packed_open_and_nonscf.energy[0],
        ref_oh.energy,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    torch.testing.assert_close(
        packed_open_and_nonscf.energy[1],
        ref_non_scf.energy,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert packed_open_and_nonscf.occupation is not None
    assert packed_open_and_nonscf.occupation.shape[0] == 2
    assert packed_open_and_nonscf.occupation.shape[1] == 2
    assert int(torch.count_nonzero(packed_open_and_nonscf.occupation[1])) == 0


def test_packed_classical_multiplicity_and_gradients() -> None:
    """Repeated classical labels preserve multiplicity, order, and individual gradients."""
    from torch.utils import _pytree as pytree

    from dxtb._src.calculators.packed_result import (
        evaluate_packed_systems,
        pack_system_results,
    )
    from dxtb._src.components.classicals.base import Classical
    from dxtb._src.components.classicals.dispersion import new_dispersion

    param = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    numbers_h2 = torch.tensor([1, 1], dtype=torch.int64)
    numbers_h2o = torch.tensor([8, 1, 1], dtype=torch.int64)

    # 1. Multiple identity-associated DispersionD3 setups + repeated label
    d3_h2_a = new_dispersion(numbers_h2, param, device=DEVICE, dtype=DTYPE)
    d3_h2_b = new_dispersion(numbers_h2, param, device=DEVICE, dtype=DTYPE)
    d3_h2o_a = new_dispersion(numbers_h2o, param, device=DEVICE, dtype=DTYPE)
    d3_h2o_b = new_dispersion(numbers_h2o, param, device=DEVICE, dtype=DTYPE)
    assert (
        d3_h2_a is not None
        and d3_h2_b is not None
        and d3_h2o_a is not None
        and d3_h2o_b is not None
    )
    d3_h2_a.label = "D3:first"
    d3_h2_b.label = "D3:second"
    d3_h2o_a.label = "D3:first"
    d3_h2o_b.label = "D3:second"

    cfg_no_disp = Config.create(
        method=labels.GFN1_XTB,
        **(_SETTINGS | {"exclude": ["disp"]}),
    )
    sys_h2 = Model(
        par=param,
        config=cfg_no_disp,
        classical=(d3_h2_a, d3_h2_b),
        auto_int_level=False,
    ).setup(numbers_h2)
    sys_h2o = Model(
        par=param,
        config=cfg_no_disp,
        classical=(d3_h2o_a, d3_h2o_b),
        auto_int_level=False,
    ).setup(numbers_h2o)
    assert len(sys_h2.d3_setups) == 2 and len(sys_h2o.d3_setups) == 2
    # Give both identity-associated D3 instances the same label to exercise repeated-label packing
    d3_h2_a.label = "DispersionD3"
    d3_h2_b.label = "DispersionD3"
    d3_h2o_a.label = "DispersionD3"
    d3_h2o_b.label = "DispersionD3"

    pos_h2 = _geometry((1, 1), 0).clone().requires_grad_(True)
    pos_h2o = _geometry((8, 1, 1), 0).clone().requires_grad_(True)

    packed = evaluate_packed_systems(
        (sys_h2, sys_h2o),
        (pos_h2, pos_h2o),
        legacy_compat=False,
    )
    serial_h2 = sys_h2.singlepoint(pos_h2)
    serial_h2o = sys_h2o.singlepoint(pos_h2o)

    d3_slots = [
        val for label, val in packed.classical if label == "DispersionD3"
    ]
    serial_h2_d3 = [
        val for label, val in serial_h2.classical if label == "DispersionD3"
    ]
    serial_h2o_d3 = [
        val for label, val in serial_h2o.classical if label == "DispersionD3"
    ]
    assert len(d3_slots) == 2
    assert len(serial_h2_d3) == 2 and len(serial_h2o_d3) == 2
    torch.testing.assert_close(
        d3_slots[0][0, :2], serial_h2_d3[0], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        d3_slots[1][0, :2], serial_h2_d3[1], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        d3_slots[0][1, :3], serial_h2o_d3[0], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        d3_slots[1][1, :3], serial_h2o_d3[1], atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(
        packed.energy[0, :2], serial_h2.energy, atol=1e-11, rtol=1e-11
    )
    torch.testing.assert_close(
        packed.energy[1, :3], serial_h2o.energy, atol=1e-11, rtol=1e-11
    )

    # Check Result.get_energies() and PyTree round-trip with repeated labels
    energies_summary = packed.get_energies()
    expected_ctotal = float(
        sum(val.sum().item() for _, val in packed.classical)
    )
    assert abs(energies_summary["Classical"]["value"] - expected_ctotal) < 1e-12
    assert "DispersionD3 (1)" in energies_summary["Classical"]["sub"]
    assert "DispersionD3 (2)" in energies_summary["Classical"]["sub"]

    leaves, spec = pytree.tree_flatten(packed)
    unflattened = pytree.tree_unflatten(leaves, spec)
    assert len(unflattened.classical) == len(packed.classical)
    for (l1, v1), (l2, v2) in zip(packed.classical, unflattened.classical):
        assert l1 == l2
        assert v1 is v2

    # 2. Distinct custom classical terms sharing a label with distinct gradients
    class _ScaledHarmonicClassical(Classical):
        def __init__(self, scale: torch.Tensor) -> None:
            super().__init__(device=DEVICE, dtype=DTYPE)
            self.label = "CustomHarmonic"
            self.scale = scale

        def get_cache(self, numbers: torch.Tensor, ihelp=None):
            return numbers

        def get_energy(
            self, positions: torch.Tensor, cache, **kwargs
        ) -> torch.Tensor:
            return self.scale * (positions**2).sum(dim=-1)

    scale_a = torch.tensor(0.15, dtype=DTYPE, requires_grad=True)
    scale_b = torch.tensor(0.35, dtype=DTYPE, requires_grad=True)
    c_a = _ScaledHarmonicClassical(scale_a)
    c_b = _ScaledHarmonicClassical(scale_b)

    cfg_classical_only = Config.create(
        method=labels.GFN1_XTB,
        **(_SETTINGS | {"exclude": ["scf"]}),
    )
    sys_custom = Model(
        par=param, config=cfg_classical_only, auto_int_level=False
    ).setup(numbers_h2)
    sys_custom.classicals.components = [
        *sys_custom.classicals.components,
        c_a,
        c_b,
    ]

    pos_c = _geometry((1, 1), 0).clone().requires_grad_(True)
    packed_custom = evaluate_packed_systems(
        (sys_custom, sys_h2),
        (pos_c, pos_h2),
        legacy_compat=False,
    )
    harmonic_slots = [
        val
        for label, val in packed_custom.classical
        if label == "CustomHarmonic"
    ]
    assert len(harmonic_slots) == 2
    g_a = torch.autograd.grad(
        harmonic_slots[0].sum(),
        (pos_c, scale_a, scale_b),
        retain_graph=True,
        allow_unused=True,
    )
    g_b = torch.autograd.grad(
        harmonic_slots[1].sum(),
        (pos_c, scale_a, scale_b),
        retain_graph=True,
        allow_unused=True,
    )
    torch.testing.assert_close(g_a[0], 2.0 * scale_a * pos_c)
    torch.testing.assert_close(g_b[0], 2.0 * scale_b * pos_c)
    assert g_a[1] is not None and float(g_a[1].abs()) > 0.0 and g_a[2] is None
    assert g_b[1] is None and g_b[2] is not None and float(g_b[2].abs()) > 0.0


def test_packed_options_charge_spin_and_unknown_kwargs_contract() -> None:
    """Verify unknown kwargs, charge/spin normalization, parity, and capacity rules."""
    from dxtb._src.calculators.legacy_composition import LegacySCFOptions
    from dxtb._src.calculators.packed_result import evaluate_packed_systems
    from dxtb._src.scf.status import SCFFailureCode

    param_gfn1 = ParamModule(GFN1_XTB, device=DEVICE, dtype=DTYPE)
    param_gfn2 = ParamModule(GFN2_XTB, device=DEVICE, dtype=DTYPE)
    cfg1 = Config.create(method=labels.GFN1_XTB, **_SETTINGS)
    cfg2 = Config.create(
        method=labels.GFN2_XTB,
        **(_SETTINGS | {"scp_mode": labels.SCP_MODE_FOCK}),
    )
    sys_h2 = Model(par=param_gfn1, config=cfg1, auto_int_level=False).setup(
        torch.tensor([1, 1], dtype=torch.int64)
    )
    sys_h3p = Model(par=param_gfn2, config=cfg2, auto_int_level=False).setup(
        torch.tensor([1, 1, 1], dtype=torch.int64)
    )
    pos_h2 = _geometry((1, 1), 0)
    pos_h3p = _geometry((1, 1, 1), 0)

    # 1. Unknown kwargs rejected by default (both public and legacy_options)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        evaluate_packed_systems(
            (sys_h2,), (pos_h2,), unknown_public_option=True
        )
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        evaluate_packed_systems(
            (sys_h2,),
            (pos_h2,),
            legacy_options=LegacySCFOptions(kwargs={"unknown_low_level": 1}),
            strict_e6bcompat=False,
        )
    # Removed store_* always rejected even with ignore_unknown_kwargs=True
    with pytest.raises(TypeError, match="store_"):
        evaluate_packed_systems(
            (sys_h2,),
            (pos_h2,),
            ignore_unknown_kwargs=True,
            store_overlap=True,
        )
    # Explicit compatibility flag ignores unknown kwargs while forwarding valid low-level options
    res_ignored = evaluate_packed_systems(
        (sys_h2,),
        (pos_h2,),
        legacy_options=LegacySCFOptions(
            kwargs={"fwd_options": {"damp": 0.25}, "ignored_extra": 123}
        ),
        ignore_unknown_kwargs=True,
    )
    assert res_ignored.energy.shape == (1, 2)

    # 2. Charge rules in legacy_compat=True vs legacy_compat=False
    with pytest.raises(ValueError, match="single float or integer"):
        evaluate_packed_systems(
            (sys_h3p, sys_h3p),
            (pos_h3p, pos_h3p),
            chrg=1.0,
            legacy_compat=True,
        )
    with pytest.raises(ValueError, match="0D scalar tensor"):
        evaluate_packed_systems(
            (sys_h3p, sys_h3p),
            (pos_h3p, pos_h3p),
            chrg=torch.tensor(1.0, dtype=DTYPE),
            legacy_compat=True,
        )
    # Default scalar charge 0 is accepted in legacy_compat=True
    res_default_chrg = evaluate_packed_systems(
        (sys_h2, sys_h2),
        (pos_h2, pos_h2),
        chrg=0,
        legacy_compat=True,
    )
    assert res_default_chrg.energy.shape == (2, 2)

    # Nonzero scalar charge is broadcast in legacy_compat=False
    res_broadcast_chrg = evaluate_packed_systems(
        (sys_h3p, sys_h3p),
        (pos_h3p, pos_h3p),
        chrg=1.0,
        spin=0,
        legacy_compat=False,
    )
    assert res_broadcast_chrg.energy.shape == (2, 3)

    # Per-lane charge tensor (1D and 2D) and sequence, mixed GFN1/GFN2
    for chrg_input in (
        torch.tensor([0.0, 1.0], dtype=DTYPE),
        torch.tensor([[0.0], [1.0]], dtype=DTYPE),
        (0.0, 1.0),
    ):
        res_mixed = evaluate_packed_systems(
            (sys_h2, sys_h3p),
            (pos_h2, pos_h3p),
            chrg=chrg_input,
            spin=None,
            legacy_compat=True,
        )
        assert res_mixed.energy.shape == (2, 3)

    # Invalid charge/spin shapes and parity
    with pytest.raises(ValueError, match="length 2"):
        evaluate_packed_systems(
            (sys_h2, sys_h3p),
            (pos_h2, pos_h3p),
            chrg=torch.zeros(3, dtype=DTYPE),
        )
    with pytest.raises(ValueError, match="Spin sequence length"):
        evaluate_packed_systems(
            (sys_h2, sys_h3p),
            (pos_h2, pos_h3p),
            spin=(0.0, 0.0, 0.0),
        )
    with pytest.raises(ValueError, match="unpaired electrons"):
        evaluate_packed_systems(
            (sys_h2, sys_h2),
            (pos_h2, pos_h2),
            chrg=(0.0, 0.0),
            spin=(0.0, 1.0),
        )

    # Too many electrons: status in legacy_compat=False vs ValueError in legacy_compat=True
    cfg_etemp = Config.create(
        method=labels.GFN1_XTB,
        **(
            _SETTINGS
            | {
                "fermi_etemp": 300.0,
                "mixer": labels.MIXER_ANDERSON,
                "maxiter": 30,
            }
        ),
    )
    sys_h2_etemp = Model(
        par=param_gfn1, config=cfg_etemp, auto_int_level=False
    ).setup(torch.tensor([1, 1], dtype=torch.int64))

    res_too_many = evaluate_packed_systems(
        (sys_h2_etemp, sys_h2_etemp),
        (pos_h2, pos_h2),
        chrg=(0.0, -7.0),
        spin=(0.0, 1.0),
        legacy_compat=False,
    )
    assert res_too_many.status is not None
    assert int(res_too_many.status.trajectory_failure[0].item()) == int(
        SCFFailureCode.NONE
    )
    assert int(res_too_many.status.final_failure[0].item()) == int(
        SCFFailureCode.NONE
    )
    assert int(res_too_many.status.trajectory_failure[1].item()) == int(
        SCFFailureCode.TOO_MANY_ELECTRONS
    )
    assert int(res_too_many.status.final_failure[1].item()) == int(
        SCFFailureCode.TOO_MANY_ELECTRONS
    )
    assert res_too_many.converged is not None
    assert bool(res_too_many.converged[0])

    with pytest.raises(ValueError, match="exceeds the number of orbitals"):
        evaluate_packed_systems(
            (sys_h2_etemp, sys_h2_etemp),
            (pos_h2, pos_h2),
            chrg=torch.tensor([0.0, -7.0], dtype=DTYPE),
            spin=torch.tensor([0.0, 1.0], dtype=DTYPE),
            legacy_compat=True,
        )


def test_complete_packed_result_adapter_all_fields_and_model_param_gradients() -> (
    None
):
    """Audit all 20 Result fields + multipoles and shared Model parameter gradients."""
    from dataclasses import fields

    from dxtb._src.calculators.packed_result import evaluate_packed_systems
    from dxtb._src.calculators.result import Result

    param_gfn2 = ParamModule(GFN2_XTB, device=DEVICE, dtype=DTYPE)
    param_gfn2.set_differentiable("element", "C", "gam")
    param_gfn2.set_differentiable("element", "O", "gam")
    cfg_gfn2 = Config.create(
        method=labels.GFN2_XTB,
        **(
            _SETTINGS
            | {
                "scp_mode": labels.SCP_MODE_FOCK,
                "fermi_etemp": 300.0,
            }
        ),
    )
    model_gfn2 = Model(par=param_gfn2, config=cfg_gfn2, auto_int_level=False)

    compositions = (
        torch.tensor([1, 1], dtype=torch.int64),
        torch.tensor([8, 1, 1], dtype=torch.int64),
        torch.tensor([6, 1, 1, 1, 1], dtype=torch.int64),
    )
    positions = (
        _geometry((1, 1), 0).clone().requires_grad_(True),
        _geometry((8, 1, 1), 0).clone().requires_grad_(True),
        _geometry((6, 1, 1, 1, 1), 0).clone().requires_grad_(True),
    )
    per_lane_fields = (
        torch.tensor([0.002, -0.001, 0.0015], dtype=DTYPE, requires_grad=True),
        None,
        torch.tensor([-0.001, 0.002, -0.0005], dtype=DTYPE, requires_grad=True),
    )
    per_lane_fgs = (
        (torch.eye(3, dtype=DTYPE) * 0.0004).requires_grad_(True),
        None,
        (torch.eye(3, dtype=DTYPE) * -0.0003).requires_grad_(True),
    )

    ptree = model_gfn2.parameter_tree
    systems = tuple(
        model_gfn2.setup_functional(nums, ptree) for nums in compositions
    )
    packed = evaluate_packed_systems(
        systems,
        positions,
        chrg=(0.0, 0.0, 0.0),
        spin=(None, None, None),
        field=per_lane_fields,
        field_grad=per_lane_fgs,
        legacy_compat=False,
    )
    serial = tuple(
        sys.singlepoint(pos, 0.0, None, field=f, field_grad=fg)
        for sys, pos, f, fg in zip(
            systems, positions, per_lane_fields, per_lane_fgs
        )
    )

    expected_fields = {
        "energy",
        "scf",
        "classical",
        "fenergy",
        "iterations",
        "charges",
        "density",
        "coefficients",
        "emo",
        "occupation",
        "potential",
        "hamiltonian",
        "overlap",
        "hcore",
        "dipole_integrals",
        "quadrupole_integrals",
        "overlap_norm",
        "converged",
        "residual",
        "status",
    }
    assert {f.name for f in fields(Result)} == expected_fields
    for fname in expected_fields:
        assert getattr(packed, fname) is not None, f"Field {fname} was None"
    assert packed.multipoles is not None
    assert packed.multipoles.dipole is not None
    assert packed.multipoles.quadrupole is not None
    assert packed.charges is not None
    assert packed.charges.dipole is not None
    assert packed.charges.quadrupole is not None
    assert packed.potential is not None
    assert packed.potential.mono is not None
    assert packed.potential.dipole is not None
    assert packed.potential.quadrupole is not None

    max_atoms = 5
    max_nao = max(s.ihelp.nao for s in systems)
    for lane, (ref, sys) in enumerate(zip(serial, systems)):
        nat = int(sys.numbers.numel())
        nao = int(sys.ihelp.nao)
        torch.testing.assert_close(
            packed.energy[lane, :nat], ref.energy, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            packed.scf[lane, :nat], ref.scf, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            packed.fenergy[lane, :nat], ref.fenergy, atol=1e-11, rtol=1e-11
        )
        torch.testing.assert_close(
            packed.charges.mono[lane, :nao],
            ref.charges.mono,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.charges.dipole[lane, :nat],
            ref.charges.dipole,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.charges.quadrupole[lane, :nat],
            ref.charges.quadrupole,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.potential.mono[lane, :nao],
            ref.potential.mono,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.potential.dipole[lane, :nat],
            ref.potential.dipole,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.potential.quadrupole[lane, :nat],
            ref.potential.quadrupole,
            atol=1e-11,
            rtol=1e-11,
        )
        torch.testing.assert_close(
            packed.coefficients[lane, :nao, :nao].abs(),
            ref.coefficients.abs(),
            atol=1e-9,
            rtol=1e-9,
        )
        if nat < max_atoms:
            assert (
                int(torch.count_nonzero(packed.charges.dipole[lane, nat:])) == 0
            )
            assert (
                int(torch.count_nonzero(packed.charges.quadrupole[lane, nat:]))
                == 0
            )
        if nao < max_nao:
            assert (
                int(torch.count_nonzero(packed.coefficients[lane, nao:, :]))
                == 0
            )
            assert (
                int(torch.count_nonzero(packed.coefficients[lane, :, nao:]))
                == 0
            )

    # Verify per-lane field gradients and shared Model parameter gradients
    grad_inputs = (
        positions[0],
        positions[1],
        positions[2],
        per_lane_fields[0],
        per_lane_fields[2],
        per_lane_fgs[0],
        per_lane_fgs[2],
        *tuple(t for t in ptree.parameters.params.values() if t.requires_grad),
    )
    packed_grads = torch.autograd.grad(
        packed.energy.sum(),
        grad_inputs,
        retain_graph=True,
        allow_unused=True,
    )
    serial_grads = torch.autograd.grad(
        sum(r.energy.sum() for r in serial),
        grad_inputs,
        retain_graph=True,
        allow_unused=True,
    )
    non_zero_param_grads = 0
    for idx, (pg, sg) in enumerate(zip(packed_grads, serial_grads)):
        if sg is None:
            assert pg is None
        else:
            assert pg is not None
            torch.testing.assert_close(pg, sg, atol=1e-9, rtol=1e-9)
            if idx >= 7 and float(pg.abs().max()) > 0.0:
                non_zero_param_grads += 1
    assert non_zero_param_grads > 0
