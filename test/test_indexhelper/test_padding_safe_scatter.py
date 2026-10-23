# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2024 Grimme Group
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
"""Padding safety for structural scatter reductions."""

from __future__ import annotations

import torch

from dxtb._src.scf.functional.padded import PaddedCoreIndexData
from dxtb._src.utils.scattergather import wrap_scatter_reduce


def test_sum_excludes_invalid_orbital_to_shell_sources() -> None:
    index = PaddedCoreIndexData(
        shells_to_atom=torch.tensor([0, 1]),
        orbitals_to_shell=torch.tensor([0, -1, 1]),
        atom_count=2,
        shell_count=2,
        valid_shells=torch.tensor([True, True]),
        valid_orbitals=torch.tensor([True, False, True]),
    )
    values = torch.tensor([3.0, 999.0, 4.0])
    torch.testing.assert_close(
        index.reduce_orbital_to_shell(values), torch.tensor([3.0, 4.0])
    )


def test_sum_excludes_invalid_shell_to_atom_sources() -> None:
    index = PaddedCoreIndexData(
        shells_to_atom=torch.tensor([0, -1, 1]),
        orbitals_to_shell=torch.tensor([0, 1, 2]),
        atom_count=2,
        shell_count=3,
        valid_shells=torch.tensor([True, False, True]),
        valid_orbitals=torch.tensor([True, True, True]),
    )
    values = torch.tensor([3.0, 999.0, 4.0])
    torch.testing.assert_close(
        index.reduce_shell_to_atom(values), torch.tensor([3.0, 4.0])
    )


def test_sum_excludes_invalid_sources_in_two_axis_reduction() -> None:
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor(
        [[3.0, 999.0, 5.0], [999.0, 999.0, 999.0], [7.0, 999.0, 11.0]]
    )

    result = wrap_scatter_reduce(values, (0, 1), idx, "sum", output_size=(2, 2))

    torch.testing.assert_close(result, torch.tensor([[3.0, 5.0], [7.0, 11.0]]))


def test_sum_all_valid_and_multiple_invalid_sources() -> None:
    all_valid = wrap_scatter_reduce(
        torch.tensor([1.0, 2.0, 3.0]),
        0,
        torch.tensor([0, 1, 0]),
        "sum",
        output_size=2,
    )
    torch.testing.assert_close(all_valid, torch.tensor([4.0, 2.0]))

    multiple_invalid = wrap_scatter_reduce(
        torch.tensor([3.0, 999.0, 888.0, 4.0]),
        0,
        torch.tensor([0, -1, -1, 1]),
        "sum",
        output_size=2,
    )
    torch.testing.assert_close(multiple_invalid, torch.tensor([3.0, 4.0]))

    all_invalid = wrap_scatter_reduce(
        torch.tensor([999.0, 888.0]),
        0,
        torch.tensor([-1, -1]),
        "sum",
        output_size=2,
    )
    torch.testing.assert_close(all_invalid, torch.zeros(2))


def test_batched_indices_map_with_explicit_output_shape() -> None:
    values = torch.tensor([[3.0, 999.0, 4.0], [5.0, 6.0, 999.0]])
    indices = torch.tensor([[0, -1, 1], [0, 1, -1]])

    result = torch.func.vmap(
        lambda x, idx: wrap_scatter_reduce(x, 0, idx, "sum", output_size=2)
    )(values, indices)

    torch.testing.assert_close(result, torch.tensor([[3.0, 4.0], [5.0, 6.0]]))


def test_two_axis_reduction_with_batched_indices_excludes_invalid_sources() -> (
    None
):
    values = torch.tensor(
        [
            [
                [3.0, 999.0, 5.0],
                [999.0, 999.0, 999.0],
                [7.0, 999.0, 11.0],
            ],
            [
                [13.0, 999.0, 17.0],
                [999.0, 999.0, 999.0],
                [19.0, 999.0, 23.0],
            ],
        ]
    )
    indices = torch.tensor([[0, -1, 1], [1, -1, 0]])
    result = torch.func.vmap(
        lambda x, idx: wrap_scatter_reduce(
            x, (0, 1), idx, "sum", output_size=(2, 2)
        )
    )(values, indices)
    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [[3.0, 5.0], [7.0, 11.0]],
                [[23.0, 19.0], [17.0, 13.0]],
            ]
        ),
    )


def test_two_axis_reduction_preserves_leading_channels() -> None:
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor(
        [
            [
                [3.0, 999.0, 5.0],
                [999.0, 999.0, 999.0],
                [7.0, 999.0, 11.0],
            ],
            [
                [13.0, 999.0, 17.0],
                [999.0, 999.0, 999.0],
                [19.0, 999.0, 23.0],
            ],
        ]
    )
    result = wrap_scatter_reduce(
        values, (-2, -1), idx, "sum", output_size=(2, 2)
    )
    torch.testing.assert_close(
        result,
        torch.tensor(
            [
                [[3.0, 5.0], [7.0, 11.0]],
                [[13.0, 17.0], [19.0, 23.0]],
            ]
        ),
    )


def test_invalid_sources_have_zero_reverse_and_forward_derivatives() -> None:
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor([3.0, 999.0, 4.0], requires_grad=True)

    total = wrap_scatter_reduce(values, 0, idx, "sum", output_size=2).sum()
    (gradient,) = torch.autograd.grad(total, values)
    torch.testing.assert_close(gradient, torch.tensor([1.0, 0.0, 1.0]))

    _, tangent = torch.func.jvp(
        lambda x: wrap_scatter_reduce(x, 0, idx, "sum", output_size=2),
        (values.detach(),),
        (torch.ones_like(values),),
    )
    torch.testing.assert_close(tangent, torch.tensor([1.0, 1.0]))


def test_two_axis_invalid_sources_have_no_gradient() -> None:
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor(
        [[3.0, 999.0, 5.0], [999.0, 999.0, 999.0], [7.0, 999.0, 11.0]],
        requires_grad=True,
    )

    total = wrap_scatter_reduce(
        values, (0, 1), idx, "sum", output_size=(2, 2)
    ).sum()
    (gradient,) = torch.autograd.grad(total, values)
    torch.testing.assert_close(
        gradient,
        torch.tensor([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
    )

    _, tangent = torch.func.jvp(
        lambda value: wrap_scatter_reduce(
            value, (0, 1), idx, "sum", output_size=(2, 2)
        ),
        (values.detach(),),
        (torch.ones_like(values),),
    )
    torch.testing.assert_close(
        tangent,
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )


def test_non_sum_reduction_behavior_is_unchanged() -> None:
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor([3.0, 999.0, 4.0])
    result = wrap_scatter_reduce(values, 0, idx, "amax", output_size=2)
    torch.testing.assert_close(result, torch.tensor([999.0, 4.0]))
