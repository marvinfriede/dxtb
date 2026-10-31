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
"""Compare NO2 finite-temperature implicit geometry responses by root policy."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from test.test_scf.test_functional_root_derivatives import _case

import torch

from benchmarks.prototypes.scf_newton_krylov import NewtonKrylovMethod
from dxtb import GFN2_XTB, labels
from dxtb._src.scf.functional.broyden import BroydenMethod
from dxtb._src.scf.functional.cdiis import CDIISMethod
from dxtb._src.scf.functional.eager import _prepare_solver
from dxtb._src.scf.functional.problem import SCFRepresentation
from dxtb._src.scf.functional.strategy import (
    SCFStage,
    SCFStrategy,
    SCFSwitchPolicy,
)


def main() -> None:
    """Run two implicit-root policies and print geometry derivatives."""
    case = _case(
        "NO2",
        method=GFN2_XTB,
        representation=labels.SCP_MODE_FOCK,
        tolerance=1.0e-8,
        maxiter=120,
        temperature=2000.0,
    )
    scf = replace(
        case.system.config.scf,
        scf_mode=labels.SCF_MODE_IMPLICIT,
        scp_mode=labels.SCP_MODE_FOCK,
        mixer=labels.MIXER_BROYDEN,
        maxiter=120,
        x_atol=1.0e-8,
        x_atol_max=1.0e-8,
    )
    system = replace(case.system, config=replace(case.system.config, scf=scf))
    stages_by_policy = (
        (
            "CDIIS",
            (SCFStage(CDIISMethod(), representation=SCFRepresentation.FOCK),),
        ),
        (
            "Broyden->NewtonKrylov",
            (
                SCFStage(
                    BroydenMethod(),
                    representation=SCFRepresentation.FOCK,
                    switch_policy=SCFSwitchPolicy(
                        min_stage_iterations=2,
                        fixed_point_l2=1.0e-2,
                    ),
                ),
                SCFStage(
                    NewtonKrylovMethod(),
                    representation=SCFRepresentation.FOCK,
                    initial_damp=1.0,
                ),
            ),
        ),
    )

    records = []
    for name, stages in stages_by_policy:

        def prepare(config, dtype):
            settings, _ = _prepare_solver(config, dtype)
            return settings, SCFStrategy(stages)

        import dxtb._src.scf.functional.eager as eager

        previous = eager._prepare_solver
        eager._prepare_solver = prepare
        try:
            positions = case.positions.clone().requires_grad_(True)
            field = torch.zeros(3, dtype=positions.dtype).requires_grad_(True)
            result = system.singlepoint(positions, field=field)
            gradient, field_gradient = torch.autograd.grad(
                result.energy.sum(),
                (positions, field),
                create_graph=True,
                retain_graph=True,
            )
            second = torch.autograd.grad(gradient[0, 0], positions)[0][0, 0]
        finally:
            eager._prepare_solver = previous

        records.append(
            {
                "strategy": name,
                "certified": bool(result.converged),
                "maps": int(result.iterations),
                "energy": float(result.energy.sum().detach()),
                "fixed_point_l2": float(result.residual[0].detach()),
                "dE_dx00": float(gradient[0, 0].detach()),
                "dE_dfield_y": float(field_gradient[1].detach()),
                "d2E_dx00": float(second.detach()),
            }
        )

    print(
        json.dumps(
            {
                "case": "NO2/GFN2/2000K",
                "coordinate": "atom 0, x",
                "records": records,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
