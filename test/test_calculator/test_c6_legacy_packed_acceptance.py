# This file is part of dxtb.
#
# SPDX-Identifier: Apache-2.0
# Copyright (C) 2026 Grimme Group
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""C6 legacy packed Calculator compatibility controls."""

from __future__ import annotations

import torch

from dxtb import GFN1_XTB, Calculator


def test_mode_two_position_gradient_matches_serial_evaluation() -> None:
    """Conformer-batch forces equal independent single-System forces."""
    dtype = torch.float64
    numbers = torch.tensor([[1, 1], [1, 1]])
    positions = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.7]],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    packed = Calculator(
        numbers,
        GFN1_XTB,
        dtype=dtype,
        opts={"verbosity": 0},
        batch_mode=2,
    )
    assert packed.system.batch_mode == 2
    assert packed.system.exact_core is None

    packed_energy = packed.singlepoint(positions).total
    (packed_gradient,) = torch.autograd.grad(packed_energy.sum(), positions)

    serial_energies = []
    serial_gradients = []
    for geometry in positions.detach():
        serial_positions = geometry.clone().requires_grad_()
        serial = Calculator(
            torch.tensor([1, 1]),
            GFN1_XTB,
            dtype=dtype,
            opts={"verbosity": 0},
        )
        energy = serial.singlepoint(serial_positions).total
        (gradient,) = torch.autograd.grad(energy.sum(), serial_positions)
        serial_energies.append(energy.sum())
        serial_gradients.append(gradient)

    torch.testing.assert_close(
        packed_energy.sum(-1), torch.stack(serial_energies)
    )
    torch.testing.assert_close(packed_gradient, torch.stack(serial_gradients))
