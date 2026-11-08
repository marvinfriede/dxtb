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
"""Characterize IFT response coordinates at physical SCF roots."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path
from test.test_scf import test_functional_root_derivatives as root_reference

import torch

from dxtb import GFN1_XTB, GFN2_XTB
from dxtb._src.scf.functional.broyden import BroydenConverger
from dxtb._src.scf.functional.energy import (
    electronic_energy_from_spectral,
    free_energy_from_spectral,
)
from dxtb._src.scf.functional.implicit import solve_linear
from dxtb._src.scf.functional.problem import SCFVariable
from dxtb._src.scf.functional.response import (
    CPSCFSettings,
    prepare_response_root,
)
from dxtb._src.scf.functional.solver import setup_solver_settings


def _git_value(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], text=True).strip()


def _case_record(name: str, method: int, temperature: float):
    if name == "C4H5NCS":
        # This retained molecule is in the general test molecule table, while
        # the root-derivative helper keeps a deliberately smaller local set.
        from test.molecules import mols

        root_reference.samples = {
            **root_reference.samples,
            "C4H5NCS": mols["C4H5NCS"],
        }

    case = root_reference._case(
        name,
        method=method,
        variable=SCFVariable.POTENTIAL,
        converger=BroydenConverger,
        tolerance=1.0e-12,
        maxiter=300,
        temperature=temperature,
    )
    record = root_reference._run_root(
        case,
        case.inputs,
        tolerance=1.0e-13,
        maxiter=300,
        eager=True,
        converger=BroydenConverger,
    )
    if not record.converged:
        raise RuntimeError(f"{name} failed to certify at {temperature:g} K.")
    return case, record


def _measure_response(
    case,
    record,
    variable: SCFVariable,
    *,
    dense_reference: bool,
    response_rtol: float,
) -> dict[str, object]:
    settings = setup_solver_settings(case.system.config.scf, record.root.dtype)
    candidate_started = time.perf_counter()
    prepared = prepare_response_root(
        record.outcome,
        settings,
        CPSCFSettings(
            variable=variable,
            rtol=response_rtol,
            atol=response_rtol * 1.0e-2,
        ),
    )
    candidate_seconds = time.perf_counter() - candidate_started
    shape = prepared.root_x.shape
    root = prepared.root_x.reshape(-1)

    def raw_map(value: torch.Tensor) -> torch.Tensor:
        return prepared.adapter(
            value.reshape(shape), *prepared.parameters
        ).reshape(-1)

    def operator(value: torch.Tensor) -> torch.Tensor:
        return value - torch.func.jvp(raw_map, (root,), (value,))[1]

    started = time.perf_counter()
    _, pullback = torch.func.vjp(raw_map, root)
    vjp_setup_seconds = time.perf_counter() - started

    def transpose_operator(value: torch.Tensor) -> torch.Tensor:
        return value - pullback(value)[0]

    hcore_index = prepared.adapter.parameter_paths.index("integrals.hcore")
    parameters = prepared.parameters
    hcore = parameters[hcore_index]
    direction = torch.zeros_like(hcore)
    direction[0, 0] = 1.0
    zero, one = hcore.new_zeros(()), hcore.new_ones(())

    def parameter_map(increment: torch.Tensor) -> torch.Tensor:
        updated = (
            parameters[:hcore_index]
            + (hcore + increment * direction,)
            + parameters[hcore_index + 1 :]
        )
        return prepared.adapter(prepared.root_x, *updated).reshape(-1)

    tangent_rhs = torch.func.jvp(parameter_map, (zero,), (one,))[1]

    def energy(value: torch.Tensor) -> torch.Tensor:
        mapped = prepared.adapter.evaluate(
            value.reshape(shape), *prepared.parameters
        )
        spectral = mapped.spectral
        return (
            electronic_energy_from_spectral(
                spectral, prepared.problem.inputs
            ).sum()
            + free_energy_from_spectral(spectral, prepared.problem.inputs).sum()
        )

    energy_adjoint_rhs = torch.func.grad(energy)(root)

    def weighted_charge(value: torch.Tensor) -> torch.Tensor:
        spectral = prepared.adapter.evaluate(
            value.reshape(shape), *prepared.parameters
        ).spectral
        weights = torch.arange(
            1,
            spectral.charges.mono.numel() + 1,
            device=value.device,
            dtype=value.dtype,
        )
        return torch.sum(weights * spectral.charges.mono.reshape(-1))

    adjoint_rhs = torch.func.grad(weighted_charge)(root)
    started = time.perf_counter()
    tangent = solve_linear(
        operator,
        tangent_rhs,
        rtol=response_rtol,
        atol=response_rtol * 1.0e-2,
    )
    tangent_seconds = time.perf_counter() - started
    started = time.perf_counter()
    adjoint = solve_linear(
        transpose_operator,
        adjoint_rhs,
        rtol=response_rtol,
        atol=response_rtol * 1.0e-2,
    )
    adjoint_seconds = time.perf_counter() - started
    energy_adjoint = solve_linear(
        transpose_operator,
        energy_adjoint_rhs,
        rtol=response_rtol,
        atol=response_rtol * 1.0e-2,
    )

    result: dict[str, object] = {
        "variable": variable.value,
        "dimension": int(root.numel()),
        "linear_rtol": response_rtol,
        "linear_atol": response_rtol * 1.0e-2,
        "response_candidate_seconds": candidate_seconds,
        "response_candidate_residual_l2": float(
            prepared.certification.metrics.fixed_point.l2
        ),
        "tangent_iterations": int(tangent.iterations),
        "tangent_residual": float(tangent.residual),
        "weighted_charge_adjoint_iterations": int(adjoint.iterations),
        "weighted_charge_adjoint_residual": float(adjoint.residual),
        "vjp_setup_seconds": vjp_setup_seconds,
        "tangent_seconds": tangent_seconds,
        "adjoint_seconds": adjoint_seconds,
        "tangent_solution_norm": float(
            torch.linalg.vector_norm(tangent.solution)
        ),
        "weighted_charge_adjoint_solution_norm": float(
            torch.linalg.vector_norm(adjoint.solution)
        ),
        "adjoint_observable": "weighted Mulliken monopole charge",
        "energy_adjoint_rhs_norm": float(
            torch.linalg.vector_norm(energy_adjoint_rhs)
        ),
        "energy_adjoint_observable": "total electronic plus free energy",
        "energy_adjoint_iterations": int(energy_adjoint.iterations),
        "energy_adjoint_residual": float(energy_adjoint.residual),
    }

    if dense_reference:
        started = time.perf_counter()
        jacobian = torch.func.jacfwd(raw_map)(root)
        dense_operator = (
            torch.eye(
                root.numel(),
                **{
                    "device": root.device,
                    "dtype": root.dtype,
                },
            )
            - jacobian
        )
        result["dense_jacobian_seconds"] = time.perf_counter() - started
        result["condition_number"] = float(torch.linalg.cond(dense_operator))
        result["dense_tangent_error"] = float(
            torch.linalg.vector_norm(
                tangent.solution
                - torch.linalg.solve(dense_operator, tangent_rhs)
            )
        )
        result["dense_adjoint_error"] = float(
            torch.linalg.vector_norm(
                adjoint.solution
                - torch.linalg.solve(dense_operator.mT, adjoint_rhs)
            )
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("H2", "H2O", "NO2", "C4H5NCS"),
        default="H2",
    )
    parser.add_argument("--model", choices=("gfn1", "gfn2"), default="gfn1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--response-rtol", type=float, default=1.0e-10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    method = GFN1_XTB if args.model == "gfn1" else GFN2_XTB
    case, record = _case_record(args.case, method, args.temperature)
    spectral = record.outcome.map_result.spectral
    fractionality_cutoff = 1.0e-6
    fractional = (spectral.occupation > fractionality_cutoff) & (
        spectral.occupation < 1.0 - fractionality_cutoff
    )
    fractional_values = spectral.occupation[fractional]
    if fractional_values.numel() == 0:
        minimum_nontrivial_occupation = 0.0
        maximum_fractional_hole = 0.0
    else:
        minimum_nontrivial_occupation = float(fractional_values.min())
        maximum_fractional_hole = float((1.0 - fractional_values).max())
        maximum_fractionality = float(
            torch.minimum(fractional_values, 1.0 - fractional_values).max()
        )
    if fractional_values.numel() == 0:
        maximum_fractionality = 0.0
    electron_count = torch.einsum(
        "ij,ji->", spectral.density, case.inputs.integrals.overlap
    )
    variables = (
        SCFVariable.CHARGE,
        SCFVariable.POTENTIAL,
        SCFVariable.FOCK,
    )
    measurements = [
        _measure_response(
            case,
            record,
            variable,
            dense_reference=(args.case in ("H2", "H2O")),
            response_rtol=args.response_rtol,
        )
        for variable in variables
    ]
    document = {
        "source_sha": _git_value("rev-parse", "HEAD"),
        "dirty": bool(_git_value("status", "--porcelain")),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "dtype": "float64",
        "threads": torch.get_num_threads(),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "case": args.case,
        "model": args.model,
        "temperature_kelvin": args.temperature,
        "nao": int(spectral.density.shape[-1]),
        "primal_certified": record.converged,
        "primal_maps": record.map_count,
        "minimum_nontrivial_occupation": minimum_nontrivial_occupation,
        "maximum_fractional_hole": maximum_fractional_hole,
        "maximum_fractionality": maximum_fractionality,
        "fractional_occupation_count": int(fractional_values.numel()),
        "fractionality_cutoff": fractionality_cutoff,
        "electron_count_error": float(
            (electron_count - case.inputs.nel.sum()).abs()
        ),
        "primal_solve_tolerance": 1.0e-13,
        "response_requested_tolerance": 1.0e-12,
        "response_rtol": args.response_rtol,
        "measurements": measurements,
    }
    rendered = json.dumps(document, indent=2, sort_keys=True)
    if args.output is None:
        print(rendered)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
        print(args.output)


if __name__ == "__main__":
    main()
