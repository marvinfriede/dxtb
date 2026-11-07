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

import pytest
import torch
from tad_mctc.tree import stack
from torch.func import jacfwd, jacrev, vmap

from dxtb import GFN1_XTB, IndexHelper
from dxtb._src.calculators.model import Model
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.param import ParamModule
from dxtb._src.utils.scattergather import wrap_scatter_reduce
from dxtb.config import Config
from dxtb._src.constants import labels


def test_sum_excludes_invalid_orbital_to_shell_sources() -> None:
    index = IndexHelper(
        unique_angular=torch.tensor([0]),
        atom_to_unique=torch.tensor([0, 0]),
        ushells_to_unique=torch.tensor([0]),
        ushells_per_unique=torch.tensor([1]),
        shells_per_atom=torch.tensor([1, 1]),
        shell_index=torch.tensor([0, 1]),
        angular=torch.tensor([0, 0]),
        shells_to_ushell=torch.tensor([0, 0]),
        shells_to_atom=torch.tensor([0, 1]),
        orbitals_per_shell=torch.tensor([1, 1]),
        orbital_index=torch.tensor([0, 2]),
        orbitals_to_shell=torch.tensor([0, -1, 1]),
        batch_mode=0,
    )
    values = torch.tensor([3.0, 999.0, 4.0])
    torch.testing.assert_close(
        index.reduce_orbital_to_shell(values), torch.tensor([3.0, 4.0])
    )


def test_sum_excludes_invalid_shell_to_atom_sources() -> None:
    index = IndexHelper(
        unique_angular=torch.tensor([0]),
        atom_to_unique=torch.tensor([0, 0]),
        ushells_to_unique=torch.tensor([0]),
        ushells_per_unique=torch.tensor([1]),
        shells_per_atom=torch.tensor([1, 1]),
        shell_index=torch.tensor([0, 2]),
        angular=torch.tensor([0, -1, 0]),
        shells_to_ushell=torch.tensor([0, -1, 0]),
        shells_to_atom=torch.tensor([0, -1, 1]),
        orbitals_per_shell=torch.tensor([1, 0, 1]),
        orbital_index=torch.tensor([0, -1, 2]),
        orbitals_to_shell=torch.tensor([0, 1, 2]),
        batch_mode=0,
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


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("sum", [1.0, 4.0, 0.0]),
        ("prod", [-6.0, 4.0, 0.0]),
        ("mean", [0.5, 4.0, 0.0]),
        ("amax", [3.0, 4.0, 0.0]),
        ("amin", [-2.0, 4.0, 0.0]),
    ],
)
def test_non_sum_reductions_exclude_negative_indices_and_zero_empty_buckets(
    reduce: str, expected: list[float]
) -> None:
    """All supported reduction modes exclude negative indices and zero empty buckets."""
    idx = torch.tensor([0, -1, 0, -999, 1])
    values = torch.tensor([3.0, 999.0, -2.0, -999.0, 4.0], requires_grad=True)
    result = wrap_scatter_reduce(values, 0, idx, reduce, output_size=3)
    torch.testing.assert_close(result, torch.tensor(expected))

    (grad,) = torch.autograd.grad(result.sum(), values)
    assert grad[1].item() == 0.0
    assert grad[3].item() == 0.0


def test_amax_and_amin_preserve_sign_for_all_negative_or_all_positive_buckets() -> (
    None
):
    """Initial fill values do not clamp all-negative amax or all-positive amin to zero."""
    idx = torch.tensor([0, -1, 0, 1])
    neg_values = torch.tensor([-5.0, 999.0, -2.0, -8.0])
    pos_values = torch.tensor([5.0, -999.0, 2.0, 8.0])

    torch.testing.assert_close(
        wrap_scatter_reduce(neg_values, 0, idx, "amax", output_size=3),
        torch.tensor([-2.0, -8.0, 0.0]),
    )
    torch.testing.assert_close(
        wrap_scatter_reduce(pos_values, 0, idx, "amin", output_size=3),
        torch.tensor([2.0, 8.0, 0.0]),
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_amax_and_amin_preserve_floating_infinities_and_integer_limits(
    dtype: torch.dtype,
) -> None:
    """Floating-point amax/amin preserve -inf/+inf valid values while excluding padded entries."""
    idx = torch.tensor([0, -1, -999, 1])
    amax_vals = torch.tensor([-torch.inf, 999.0, 888.0, -4.0], dtype=dtype)
    amin_vals = torch.tensor([torch.inf, -999.0, -888.0, 4.0], dtype=dtype)

    amax_out = wrap_scatter_reduce(amax_vals, 0, idx, "amax", output_size=3)
    amin_out = wrap_scatter_reduce(amin_vals, 0, idx, "amin", output_size=3)
    assert torch.equal(
        amax_out, torch.tensor([-torch.inf, -4.0, 0.0], dtype=dtype)
    )
    assert torch.equal(
        amin_out, torch.tensor([torch.inf, 4.0, 0.0], dtype=dtype)
    )

    for int_dtype in (torch.int32, torch.int64):
        int_neg = torch.tensor([-5, 999, -2, -8], dtype=int_dtype)
        int_pos = torch.tensor([5, -999, 2, 8], dtype=int_dtype)
        idx_int = torch.tensor([0, -1, 0, 1])
        assert torch.equal(
            wrap_scatter_reduce(int_neg, 0, idx_int, "amax", output_size=3),
            torch.tensor([-2, -8, 0], dtype=int_dtype),
        )
        assert torch.equal(
            wrap_scatter_reduce(int_pos, 0, idx_int, "amin", output_size=3),
            torch.tensor([2, 8, 0], dtype=int_dtype),
        )


@pytest.mark.parametrize(
    ("reduce", "expected"),
    [
        ("sum", [[8.0, 5.0, 0.0], [7.0, 11.0, 0.0], [0.0, 0.0, 0.0]]),
        ("prod", [[15.0, 5.0, 0.0], [7.0, 11.0, 0.0], [0.0, 0.0, 0.0]]),
        ("mean", [[4.0, 5.0, 0.0], [7.0, 11.0, 0.0], [0.0, 0.0, 0.0]]),
        ("amax", [[5.0, 5.0, 0.0], [7.0, 11.0, 0.0], [0.0, 0.0, 0.0]]),
        ("amin", [[3.0, 5.0, 0.0], [7.0, 11.0, 0.0], [0.0, 0.0, 0.0]]),
    ],
)
def test_two_axis_non_sum_reductions_exclude_negative_indices(
    reduce: str, expected: list[list[float]]
) -> None:
    """Two-axis reductions exclude ghost rows/columns for all supported modes."""
    idx = torch.tensor([0, -1, 0, 1])
    values = torch.tensor(
        [
            [1.0, 999.0, 2.0, 2.0],
            [999.0, 999.0, 999.0, 999.0],
            [2.0, 999.0, 3.0, 3.0],
            [3.0, 999.0, 4.0, 11.0],
        ],
        requires_grad=True,
    )
    if reduce in ("prod", "mean", "amax", "amin"):
        # Use simpler 1-element-per-bucket matrix for exact comparison
        idx_simple = torch.tensor([0, -1, 1])
        val_simple = torch.tensor(
            [[3.0, 999.0, 5.0], [999.0, 999.0, 999.0], [7.0, 999.0, 11.0]],
            requires_grad=True,
        )
        res_simple = wrap_scatter_reduce(
            val_simple, (0, 1), idx_simple, reduce, output_size=(3, 3)
        )
        torch.testing.assert_close(
            res_simple[:2, :2], torch.tensor([[3.0, 5.0], [7.0, 11.0]])
        )
        torch.testing.assert_close(res_simple[2, :], torch.zeros(3))
        torch.testing.assert_close(res_simple[:, 2], torch.zeros(3))
        (grad_simple,) = torch.autograd.grad(res_simple.sum(), val_simple)
        torch.testing.assert_close(grad_simple[1, :], torch.zeros(3))
        torch.testing.assert_close(grad_simple[:, 1], torch.zeros(3))
    else:
        result = wrap_scatter_reduce(
            values, (0, 1), idx, reduce, output_size=(3, 3)
        )
        torch.testing.assert_close(result, torch.tensor(expected))


def test_unsupported_reduction_mode_raises_value_error() -> None:
    """Unknown reduction modes are rejected explicitly."""
    idx = torch.tensor([0, -1, 1])
    values = torch.tensor([3.0, 999.0, 4.0])
    with pytest.raises(ValueError, match="Unsupported reduction mode"):
        wrap_scatter_reduce(values, 0, idx, "invalid_mode", output_size=2)


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda", torch.cuda.current_device()))
    return devices


@pytest.mark.parametrize("device", _devices(), ids=lambda d: d.type)
def test_indexhelper_node_transforms_vmap_jac_and_compile(
    device: torch.device,
) -> None:
    """Unified IndexHelper(Node) works inside vmap, jacrev, jacfwd, and torch.compile."""
    dtype = torch.float64
    par = ParamModule(GFN1_XTB, dtype=dtype, device=device)
    batched_legacy = IndexHelper.from_numbers(
        torch.tensor([[8, 1, 1], [1, 1, 0]], device=device), par
    )
    shared_names = ("unique_angular", "ushells_to_unique", "ushells_per_unique")
    ihelp_h2o = IndexHelper(
        **{name: getattr(batched_legacy, name) for name in shared_names},
        **{
            name: getattr(batched_legacy, name)[0]
            for name in batched_legacy._child_names
            if name not in shared_names
        },
        batch_mode=0,
    )
    ihelp_h2 = IndexHelper(
        **{name: getattr(batched_legacy, name) for name in shared_names},
        **{
            name: getattr(batched_legacy, name)[1]
            for name in batched_legacy._child_names
            if name not in shared_names
        },
        batch_mode=0,
    )
    assert bool((ihelp_h2.shells_to_ushell == -1).any())
    assert bool((ihelp_h2.shells_to_atom == -999).any())
    assert bool((ihelp_h2.orbitals_to_shell == -999).any())
    assert bool((ihelp_h2.shell_index == -999).any())
    assert bool((ihelp_h2.orbital_index == -999).any())
    # Also exercise -1 sentinel alongside -999 in shells_to_atom and orbitals_to_shell
    ihelp_h2.shells_to_atom[-1] = -1
    ihelp_h2.orbitals_to_shell[-1] = -1

    max_orbitals = int(ihelp_h2o.orbitals_to_shell.shape[0])
    batched_ihelp = stack((ihelp_h2o, ihelp_h2))

    def kernel(ihelp: IndexHelper, orb_vals: torch.Tensor) -> torch.Tensor:
        sh_sum = ihelp.reduce_orbital_to_shell(orb_vals, reduce="sum")
        sh_mean = ihelp.reduce_orbital_to_shell(orb_vals, reduce="mean")
        sh_amax = ihelp.reduce_orbital_to_shell(orb_vals, reduce="amax")
        at_sum = ihelp.reduce_shell_to_atom(
            sh_sum + 0.5 * sh_mean + 0.25 * sh_amax, reduce="sum"
        )
        at_direct = ihelp.reduce_orbital_to_atom(orb_vals, reduce="sum")
        spread_orb = ihelp.spread_atom_to_orbital(at_sum + at_direct)
        valid_orb = ihelp.spread_shell_to_orbital(ihelp.orbitals_per_shell) > 0
        masked_orb = torch.where(
            valid_orb, spread_orb * orb_vals, torch.zeros_like(orb_vals)
        )
        return ihelp.reduce_orbital_to_atom(masked_orb, reduce="sum")

    valid_orb_mask_1 = ihelp_h2.orbitals_to_shell >= 0
    invalid_orb_mask_1 = ~valid_orb_mask_1
    invalid_atom_mask_1 = ihelp_h2.shells_per_atom == 0
    assert bool(invalid_orb_mask_1.any())
    assert bool(invalid_atom_mask_1.any())

    orb_1 = torch.where(
        valid_orb_mask_1,
        torch.arange(1.0, max_orbitals + 1.0, device=device, dtype=dtype),
        torch.full((max_orbitals,), 999.0, device=device, dtype=dtype),
    )
    orb_batch = torch.stack(
        [
            torch.arange(1.0, max_orbitals + 1.0, device=device, dtype=dtype),
            orb_1,
        ]
    )

    # 1. vmap over mapped IndexHelper(Node) leaves and orbital inputs
    vmapped_out = vmap(kernel)(batched_ihelp, orb_batch)
    expected_0 = kernel(ihelp_h2o, orb_batch[0])
    expected_1 = kernel(ihelp_h2, orb_batch[1])
    torch.testing.assert_close(vmapped_out[0], expected_0)
    torch.testing.assert_close(vmapped_out[1], expected_1)
    # Padded atoms in H2 must be strictly zero
    torch.testing.assert_close(
        vmapped_out[1, invalid_atom_mask_1],
        torch.zeros_like(vmapped_out[1, invalid_atom_mask_1]),
    )

    # 2. First-order derivatives via jacrev, jacfwd, and vmap(jacrev/jacfwd)
    j_rev_1 = jacrev(lambda x: kernel(ihelp_h2, x))(orb_batch[1])
    j_fwd_1 = jacfwd(lambda x: kernel(ihelp_h2, x))(orb_batch[1])
    torch.testing.assert_close(j_rev_1, j_fwd_1)
    # Padded orbitals and padded atoms in H2 must have zero derivatives
    torch.testing.assert_close(
        j_rev_1[:, invalid_orb_mask_1],
        torch.zeros_like(j_rev_1[:, invalid_orb_mask_1]),
    )
    torch.testing.assert_close(
        j_rev_1[invalid_atom_mask_1, :],
        torch.zeros_like(j_rev_1[invalid_atom_mask_1, :]),
    )

    vmap_j_rev = vmap(jacrev(kernel, argnums=1))(batched_ihelp, orb_batch)
    vmap_j_fwd = vmap(jacfwd(kernel, argnums=1))(batched_ihelp, orb_batch)
    torch.testing.assert_close(vmap_j_rev, vmap_j_fwd)
    torch.testing.assert_close(vmap_j_rev[1], j_rev_1)

    # 3. torch.compile over unified IndexHelper(Node) (both aot_eager and default Inductor)
    for compiled_kernel in (
        torch.compile(kernel, backend="aot_eager"),
        torch.compile(kernel),
    ):
        compiled_0 = compiled_kernel(ihelp_h2o, orb_batch[0])
        compiled_1 = compiled_kernel(ihelp_h2, orb_batch[1])
        torch.testing.assert_close(compiled_0, expected_0)
        torch.testing.assert_close(compiled_1, expected_1)


def test_two_axis_density_reduction_is_vmap_safe():
    index = IndexHelper(
        unique_angular=torch.tensor([0]),
        atom_to_unique=torch.tensor([0, 0]),
        ushells_to_unique=torch.tensor([0]),
        ushells_per_unique=torch.tensor([1]),
        shells_per_atom=torch.tensor([1, 1]),
        shell_index=torch.tensor([0, 1]),
        angular=torch.tensor([0, 0]),
        shells_to_ushell=torch.tensor([0, 0]),
        shells_to_atom=torch.tensor([0, 1]),
        orbitals_per_shell=torch.tensor([1, 1]),
        orbital_index=torch.tensor([0, 2]),
        orbitals_to_shell=torch.tensor([0, -1, 1]),
        batch_mode=0,
    )
    matrices = torch.tensor(
        [
            [[3.0, 999.0, 5.0], [999.0, 999.0, 999.0], [7.0, 999.0, 11.0]],
            [[13.0, 999.0, 17.0], [999.0, 999.0, 999.0], [19.0, 999.0, 23.0]],
        ]
    )
    mapped = torch.func.vmap(
        lambda value: index.reduce_orbital_to_shell(value, dim=(0, 1))
    )(matrices)
    expected = torch.tensor(
        [
            [[3.0, 5.0], [7.0, 11.0]],
            [[13.0, 17.0], [19.0, 23.0]],
        ]
    )
    torch.testing.assert_close(mapped, expected)
