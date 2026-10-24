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
