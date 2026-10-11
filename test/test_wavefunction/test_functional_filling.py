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
"""Transform-compatible Fermi occupation core tests."""

from __future__ import annotations

import torch
import pytest
from torch.func import jacfwd, jacrev, jvp, vmap

from dxtb._src.wavefunction import filling


def _case() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = torch.float64
    spectra = torch.tensor(
        [
            [[-0.8, 0.1, 0.9, 0.0], [-0.6, 0.4, 1.2, 0.0]],
            [[-1.1, 0.3, 1.4, 0.0], [-0.2, 0.6, 1.8, 0.0]],
            [[-0.9, 0.2, 1.1, 0.0], [-0.5, 0.8, 1.5, 0.0]],
        ],
        dtype=dtype,
    )
    electrons = torch.tensor(
        [[1.0, 0.5], [1.5, 0.75], [2.0, 1.25]], dtype=dtype
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    kt = torch.tensor(0.025, dtype=dtype)
    return spectra, electrons, mask, kt


def test_core_matches_eager_and_independent_vmap_loops() -> None:
    """The fixed-trip core agrees with eager search and maps spectra/counts."""
    spectra, electrons, mask, kt = _case()
    actual = vmap(
        lambda e, n: filling.get_fermi_occupation_core(n, e, kt, mask=mask)
    )(spectra, electrons)
    loop = [
        filling.get_fermi_occupation_core(n, e, kt, mask=mask)
        for e, n in zip(spectra, electrons)
    ]
    for index, result in enumerate(loop):
        torch.testing.assert_close(actual.occupation[index], result.occupation)
        assert torch.equal(
            actual.status.converged[index], result.status.converged
        )
        torch.testing.assert_close(
            actual.occupation[index],
            filling.get_fermi_occupation(
                electrons[index], spectra[index], kt, mask=mask
            ),
        )
    assert torch.all(actual.occupation[..., -1] == 0)


def test_core_vmap_over_fractional_electron_counts() -> None:
    """Vmap over fractional electron targets matches individual searches."""
    spectra, electrons, mask, kt = _case()
    targets = torch.tensor(
        [[1.0, 0.5], [1.5, 0.75], [2.75, 1.25]], dtype=torch.float64
    )
    fixed_spectrum = spectra[0].expand(targets.shape[0], -1, -1)
    actual = vmap(
        lambda n, e: filling.get_fermi_occupation_core(n, e, kt, mask=mask)
    )(targets, fixed_spectrum)
    expected = torch.stack(
        [
            filling.get_fermi_occupation_core(
                n, spectra[0], kt, mask=mask
            ).occupation
            for n in targets
        ]
    )
    torch.testing.assert_close(actual.occupation, expected)
    assert bool(actual.status.converged.all())

    def occupied_orbitals(emo: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation_core(
            electrons[0], emo, kt, mask=mask
        ).occupation

    energy_jacobian = jacrev(occupied_orbitals)(spectra[0])
    assert torch.count_nonzero(energy_jacobian[..., -1]) == 0
    assert torch.count_nonzero(energy_jacobian[:, -1, :, :]) == 0


def test_core_electron_count_gradient_is_channel_local() -> None:
    """Active occupation derivatives conserve electron number by channel."""
    spectra, electrons, mask, kt = _case()

    def occupation(nel: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation_core(
            nel, spectra[1], kt, mask=mask
        ).occupation

    jacobian = jacrev(occupation)(electrons[1])
    response = jacobian.sum(dim=-2)
    torch.testing.assert_close(response, torch.eye(2, dtype=torch.float64))
    assert torch.isfinite(jacobian).all()


def test_core_empty_full_channels_keep_exact_occupations() -> None:
    """Empty and full channels retain zero-response occupation semantics."""
    spectra = torch.tensor(
        [[-0.8, 0.1, 0.9, 0.0], [-0.8, 0.1, 0.9, 0.0]],
        dtype=torch.float64,
    )
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]], dtype=torch.bool)
    electrons = torch.tensor([0.0, 3.0], dtype=torch.float64)
    result = filling.get_fermi_occupation_core(
        electrons, spectra, torch.tensor(0.025, dtype=torch.float64), mask=mask
    )
    torch.testing.assert_close(
        result.occupation,
        torch.tensor([[0, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.float64),
    )
    assert bool(result.status.converged.all())
    assert bool(result.status.electron_count_ok.all())


def test_core_jvp_jacfwd_and_reverse_agree() -> None:
    """The direct occupation core supports JVP, jacfwd and reverse mode."""
    spectra, electrons, mask, kt = _case()
    spectrum = spectra[1].clone().requires_grad_()
    direction = torch.full_like(spectrum, 0.2)

    def scalar(emo: torch.Tensor) -> torch.Tensor:
        occupation = filling.get_fermi_occupation_core(
            electrons[1], emo, kt, mask=mask
        ).occupation
        weights = torch.arange(
            1, occupation.numel() + 1, dtype=occupation.dtype
        )
        return torch.sum(occupation.reshape(-1) * weights)

    gradient = torch.autograd.grad(scalar(spectrum), spectrum)[0]
    tangent = jvp(scalar, (spectrum,), (direction,))[1]
    expected = (gradient * direction).sum()
    torch.testing.assert_close(tangent, expected, rtol=1e-9, atol=1e-10)
    forward = jacfwd(scalar)(spectrum)
    torch.testing.assert_close(forward, gradient, rtol=1e-9, atol=1e-10)


def test_core_reverse_derivatives_through_order_three_are_finite() -> None:
    """The fixed search preserves the established third-order attachment."""
    spectra, electrons, mask, kt = _case()
    energies = spectra[1].clone().requires_grad_()
    direction = torch.arange(
        1, energies.numel() + 1, dtype=energies.dtype
    ).reshape_as(energies)

    def scalar(value: torch.Tensor) -> torch.Tensor:
        return filling.get_fermi_occupation_core(
            electrons[1], value, kt, mask=mask
        ).occupation[0, 1]

    value = scalar(energies)
    first = torch.autograd.grad(value, energies, create_graph=True)[0]
    second = torch.autograd.grad(
        (first * direction).sum(), energies, create_graph=True
    )[0]
    third = torch.autograd.grad((second * direction).sum(), energies)[0]
    assert torch.isfinite(first).all()
    assert torch.isfinite(second).all()
    assert torch.isfinite(third).all()


def test_core_temperature_gradient_is_transform_compatible() -> None:
    """Positive scalar temperature derivatives agree across AD modes."""
    spectra, electrons, mask, kt = _case()
    kt = kt.clone().requires_grad_()

    def scalar(temp: torch.Tensor) -> torch.Tensor:
        return (
            filling.get_fermi_occupation_core(
                electrons[1], spectra[1], temp, mask=mask
            )
            .occupation.square()
            .sum()
        )

    reverse = torch.autograd.grad(scalar(kt), kt)[0]
    tangent = jvp(scalar, (kt,), (torch.ones_like(kt),))[1]
    forward = jacfwd(scalar)(kt)
    assert torch.isfinite(reverse).all()
    torch.testing.assert_close(tangent, reverse, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(forward, reverse, rtol=1e-9, atol=1e-10)


def test_core_returns_finite_status_for_bad_or_unconverged_inputs() -> None:
    """Invalid and under-iterated searches return finite values plus status."""
    spectra, electrons, mask, kt = _case()
    invalid = filling.get_fermi_occupation_core(
        electrons[0], spectra[0], -kt, mask=mask
    )
    assert invalid.status.invalid_temperature
    assert torch.isfinite(invalid.occupation).all()

    too_many = filling.get_fermi_occupation_core(
        torch.tensor([4.0, 0.5], dtype=torch.float64),
        spectra[0],
        kt,
        mask=mask,
    )
    assert too_many.status.too_many_electrons[0]
    assert torch.isfinite(too_many.occupation).all()

    unfinished = filling.get_fermi_occupation_core(
        electrons[1], spectra[1], kt, mask=mask, maxiter=0
    )
    assert not bool(unfinished.status.converged.all())
    assert torch.isfinite(unfinished.status.residual).all()
    assert torch.isfinite(unfinished.occupation).all()


def test_converged_fermi_channels_freeze_the_search_bracket() -> None:
    """Fixed trips leave a converged channel's root-search state unchanged."""
    delta = torch.tensor([[0.5]], dtype=torch.float64)
    below = torch.tensor([[False, False]])
    above = torch.tensor([[True, True]])
    energies = torch.tensor([[-0.5, 0.5]], dtype=torch.float64)
    kt = torch.tensor(0.025, dtype=torch.float64)
    start = torch.zeros((1, 1), dtype=torch.float64)
    active = torch.ones((1, 1), dtype=torch.bool)
    threshold = torch.tensor(1.0e5, dtype=torch.float64)

    one_trip = filling._fermi_energy_search_fixed(
        delta,
        below,
        above,
        energies,
        kt,
        start,
        active,
        None,
        threshold,
        0,
    )
    fixed_trips = filling._fermi_energy_search_fixed(
        delta,
        below,
        above,
        energies,
        kt,
        start,
        active,
        None,
        threshold,
        5,
    )
    assert bool(one_trip[2].all())
    assert bool(fixed_trips[2].all())
    torch.testing.assert_close(fixed_trips[3], one_trip[3])
    torch.testing.assert_close(fixed_trips[4], one_trip[4])


def test_core_bypasses_host_control_search_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed-trip occupation does not call either eager host-control helper."""
    spectra, electrons, mask, kt = _case()

    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("the functional Fermi core called eager control")

    monkeypatch.setattr(filling, "_read_flags", fail)
    monkeypatch.setattr(filling, "_fermi_energy_search", fail)
    result = filling.get_fermi_occupation_core(
        electrons[0], spectra[0], kt, mask=mask
    )
    assert bool(result.status.converged.all())


def test_transform_critical_fermi_sources_have_no_host_control() -> None:
    """Fixed-trip Fermi and entropy sources contain no host synchronization."""
    import inspect

    sources = (
        inspect.getsource(filling.get_fermi_occupation_core),
        inspect.getsource(filling._fermi_energy_search_fixed),
        inspect.getsource(filling._fermi_search_step),
        inspect.getsource(filling.get_fermi_entropy_finite),
    )
    for source in sources:
        for forbidden in (".item(", ".tolist(", "_read_flags(", "bool("):
            assert forbidden not in source


def test_search_status_is_separate_from_scf_electron_count_guard() -> None:
    """A loose search threshold can pass while legacy electron safety fails."""
    energies = torch.tensor(
        [
            [-547.5523, 591.2507, -1815.7304, -1180.9888],
            [-5879.5701, 1027.9796, -4776.4331, -3708.9585],
        ],
        dtype=torch.float64,
    )
    electrons = torch.tensor([0.5, 0.5], dtype=torch.float64)
    result = filling.get_fermi_occupation_core(
        electrons,
        energies,
        torch.tensor(2.8405667729892673e-05, dtype=torch.float64),
        thr=100.0,
        maxiter=0,
        diff_order=0,
    )
    assert bool(result.status.converged.all())
    assert not bool(result.status.electron_count_ok.all())
    assert torch.isfinite(result.occupation).all()
