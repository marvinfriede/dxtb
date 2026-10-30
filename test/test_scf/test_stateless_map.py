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
"""Physical agreement between FULL and functional implicit SCF routes."""

from __future__ import annotations

import pytest
import torch

from dxtb import GFN1_XTB, Calculator

from ..conftest import DEVICE
from ..molecules import mols

DD = {"device": DEVICE, "dtype": torch.double}


@pytest.mark.parametrize("scp_mode", ["charge", "potential", "fock"])
def test_result_hamiltonian_matches_full(scp_mode: str) -> None:
    """Implicit and FULL routes agree on the converged physical observables."""
    molecule = mols["H2O"]
    results = {}
    for mode in ("implicit", "full"):
        tolerance = 1e-12
        options = {
            "verbosity": 0,
            "scf_mode": mode,
            "scp_mode": scp_mode,
            "f_atol": tolerance,
            "x_atol": tolerance,
            "x_atol_max": tolerance,
        }
        calculator = Calculator(
            molecule["numbers"].to(DEVICE), GFN1_XTB, opts=options, **DD
        )
        results[mode] = calculator.singlepoint(
            molecule["positions"].to(**DD), torch.tensor(0.0, **DD)
        )

    for name in ("hamiltonian", "density", "emo"):
        implicit = getattr(results["implicit"], name)
        full = getattr(results["full"], name)
        assert torch.allclose(implicit, full, atol=1e-8, rtol=0), name
