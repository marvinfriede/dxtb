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
"""
Test the gradient-tracking cache key (stale-graph stopgap, T0.5).
"""

from __future__ import annotations

import torch

from dxtb._src.utils.tensors import grad_key, grad_key_matches


def test_no_key() -> None:
    assert grad_key_matches(None, torch.zeros(3)) is False


def test_no_grad_values_suffice() -> None:
    x = torch.zeros(3)
    key = grad_key(x)
    assert grad_key_matches(key, x) is True
    assert grad_key_matches(key, torch.zeros(3)) is True


def test_flag_change() -> None:
    x = torch.zeros(3)
    key = grad_key(x)
    x.requires_grad_(True)
    assert grad_key_matches(key, x) is False


def test_grad_requires_same_object() -> None:
    x = torch.zeros(3, requires_grad=True)
    key = grad_key(x)
    assert grad_key_matches(key, x) is True
    assert grad_key_matches(key, x.detach().clone().requires_grad_()) is False
    assert grad_key_matches(key, x * 1.0) is False


def test_several_tensors() -> None:
    x = torch.zeros(3, requires_grad=True)
    y = torch.zeros(3)
    key = grad_key(x, y)
    assert grad_key_matches(key, x, y) is True
    assert grad_key_matches(key, x) is False
    assert grad_key_matches(key, x, y.requires_grad_(True)) is False


def test_normalize_device() -> None:
    # pylint: disable=import-outside-toplevel
    from dxtb._src.utils.tensors import normalize_device

    assert normalize_device(None) is None
    assert normalize_device("cpu") == torch.device("cpu")
    assert normalize_device(torch.device("cpu")) == torch.zeros(1).device
    assert normalize_device("cuda:1") == torch.device("cuda", 1)
    assert normalize_device("cuda").index is not None
