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
"""Small, guarded utility for SCF certification characterization rows."""

from __future__ import annotations

import hashlib

import torch

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.scf.functional import (
    SCFPhysicalState,
    SCFProblem,
    SCFRepresentation,
    certification_metrics,
    electronic_energy_from_spectral,
    fock_initial_guess,
    potential_initial_guess,
    stationarity_matrix,
    unpack_charges,
)
from dxtb._src.scf.functional.data import PotentialLayout
from dxtb._src.scf.functional.method import LinearMethod
from dxtb._src.scf.functional.solver import (
    SCFSolverSettings,
    _resolve_tolerance,
    _solve_map,
)
from dxtb._src.scf.functional.strategy import SCFStage, SCFStrategy

from .test_functional_map import _prepare_case

_STRATEGY = SCFStrategy((SCFStage(LinearMethod()),))


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a stable digest for one detached CPU characterization tensor."""
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def characterize_case(
    method: int,
    *,
    sample_name: str,
    representation: SCFRepresentation = SCFRepresentation.CHARGE,
    charge: float = 0.0,
    spin: float = 0.0,
    temperature: float = 0.0,
    dtype: torch.dtype,
    tolerance: float,
) -> dict[str, object]:
    """Build, method-check, solve, and summarize one independent case."""
    if method is GFN1_XTB:
        expected_config, expected_h0, expected_name = (
            labels.GFN1_XTB,
            "gfn1",
            "GFN1-xTB",
        )
    elif method is GFN2_XTB:
        expected_config, expected_h0, expected_name = (
            labels.GFN2_XTB,
            "gfn2",
            "GFN2-xTB",
        )
    else:
        raise ValueError(f"Unsupported characterization method {method!r}.")
    case = _prepare_case(
        method,
        sample_name=sample_name,
        charge=charge,
        spin=spin,
        dtype=dtype,
        scf_options={"fermi_etemp": temperature},
    )

    if (
        case.calc.model.config.method != expected_config
        or case.system.config.method != expected_config
        or case.system.h0_setup.method != expected_h0
        or case.par.forward().meta.name.value != expected_name
    ):
        raise AssertionError(
            "Requested model does not match resolved Calculator/Model/H0/"
            "parameter identity: "
            f"requested={expected_name}, config={case.system.config.method}, "
            f"h0={case.system.h0_setup.method}, "
            f"parameter={case.par.forward().meta.name.value}."
        )

    layout: PotentialLayout | None = None
    if representation is SCFRepresentation.CHARGE:
        x = case.x
    else:
        charges = unpack_charges(case.x, case.inputs.charge_layout)
        if representation is SCFRepresentation.POTENTIAL:
            x, layout = potential_initial_guess(charges, case.inputs)
        elif representation is SCFRepresentation.FOCK:
            x = fock_initial_guess(charges, case.inputs)
        else:
            raise ValueError(f"Unknown representation {representation!r}.")

    problem = SCFProblem(case.inputs, representation, layout)
    effective_tolerance = _resolve_tolerance(tolerance, x.dtype)
    settings = SCFSolverSettings(
        maxiter=180,
        mix_guess=False,
        damp=0.2,
        x_atol=effective_tolerance,
        x_atol_max=effective_tolerance,
    )
    outcome = _solve_map(
        x,
        settings,
        problem.map,
        problem=problem,
        strategy=_STRATEGY,
    )
    # Keep the map/result pair from the solver: no additional physical map is
    # evaluated, and the tested root input remains paired with its map result.
    state = SCFPhysicalState(outcome.root_x, outcome.map_result)
    metrics = certification_metrics(problem, state)
    spectral = state.map_result.spectral
    orthogonalized = (
        case.inputs.inverse_cholesky.transpose(-2, -1)
        @ stationarity_matrix(problem, state)
        @ case.inputs.inverse_cholesky
    )
    fermi = state.map_result.fermi_status
    return {
        "model": expected_name,
        "model_config": int(case.system.config.method),
        "h0_method": case.system.h0_setup.method,
        "parameter_model": case.par.forward().meta.name.value,
        "molecule": sample_name,
        "numbers": case.inputs.numbers.tolist(),
        "charge": charge,
        "spin": spin,
        "integral_level": case.int_level,
        "representation": representation.value,
        "temperature_kelvin": temperature,
        "dtype": str(dtype),
        "requested_root_tolerance": tolerance,
        "effective_root_tolerance": effective_tolerance,
        "fixed_point_l2": metrics.fixed_point.l2.item(),
        "fixed_point_max": metrics.fixed_point.max_abs.item(),
        "stationarity_ao_l2": metrics.stationarity_l2.item(),
        "stationarity_ao_max": metrics.stationarity_max.item(),
        "stationarity_orth_l2": float(
            orthogonalized.square().sum().sqrt().item()
        ),
        "stationarity_orth_max": float(orthogonalized.abs().amax().item()),
        "electron_count": metrics.electron_count.item(),
        "electron_count_error": metrics.electron_count_error.item(),
        "fermi_residual": (
            None if fermi is None else fermi.residual.detach().tolist()
        ),
        "fermi_converged": bool(metrics.fermi_converged),
        "fermi_electron_count_ok": bool(metrics.fermi_electron_count_ok),
        "finite_ok": bool(metrics.finite_ok),
        "fermi_ok": bool(metrics.fermi_ok),
        "converged": bool(outcome.converged),
        "electronic_energy": electronic_energy_from_spectral(
            spectral, case.inputs
        )
        .sum()
        .item(),
        "electronic_free_energy": metrics.electronic_free_energy.item(),
        "physical_map_evaluations": int(outcome.iterations),
        "iterations": int(outcome.iterations),
        "nao": int(case.inputs.integrals.overlap.shape[-1]),
        "active_nao": int(case.inputs.orbital_mask.sum()),
        "root_x_shape": list(outcome.root_x.shape),
        "initial_x_shape": list(x.shape),
        "initial_x_sha256": _tensor_sha256(x),
        "n0": case.inputs.n0.detach().tolist(),
        "occupation": case.inputs.occupation.detach().tolist(),
        "nel": case.inputs.nel.detach().tolist(),
        "hcore_norm": torch.linalg.vector_norm(
            case.inputs.integrals.hcore.detach()
        ).item(),
        "hcore_sum": case.inputs.integrals.hcore.detach().sum().item(),
        "overlap_norm": torch.linalg.vector_norm(
            case.inputs.integrals.overlap.detach()
        ).item(),
        "overlap_sum": case.inputs.integrals.overlap.detach().sum().item(),
    }
