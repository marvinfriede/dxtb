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
"""Public tensor-valued status for functional SCF evaluations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from dxtb._src.typing import Tensor

__all__ = ["SCFFailureCode", "SCFStatus"]


class SCFFailureCode(IntEnum):
    """Stable public categories for functional Fermi-subsolve failures."""

    NONE = 0
    INVALID_TEMPERATURE = 1
    TOO_MANY_ELECTRONS = 2
    FERMI_NONCONVERGENCE = 3
    ELECTRON_COUNT = 4


@dataclass(frozen=True, eq=False)
class SCFStatus:
    """Immutable failure codes for the physical trajectory and final spectrum."""

    trajectory_failure: Tensor
    final_failure: Tensor
