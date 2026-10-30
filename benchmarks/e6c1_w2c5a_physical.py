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
"""Measure accepted physical SCF chemical-potential corrections.

The harness wraps existing SCF functions and records detached diagnostics. It
does not replace a physical result or alter the accepted correction
calculation. Run one System per fresh process for bounded memory, for example::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      python -u benchmarks/e6c1_w2c5a_physical.py \
      --case tmpda0 --output benchmarks/results/e6c1-w2c5a/tmpda0.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "test"), str(ROOT / "src")]

from test.molecules import mols  # noqa: E402

import benchmarks.e6c1_w2c2_implicit as w2c2  # noqa: E402
import dxtb  # noqa: E402
from dxtb import GFN1_XTB, GFN2_XTB, ParamModule, labels  # noqa: E402
from dxtb._src.calculators.model import Model  # noqa: E402
from dxtb._src.scf.functional import eager as eager_module  # noqa: E402
from dxtb._src.scf.functional import map as map_module  # noqa: E402
from dxtb._src.scf.functional import solver as solver_module  # noqa: E402
from dxtb._src.scf.functional import spectral as spectral_module  # noqa: E402
from dxtb.config import Config  # noqa: E402

DTYPE = torch.float64
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
GUARD_RATIO = -0.25 * torch.log(torch.tensor(torch.finfo(DTYPE).eps))


def _cpu_name() -> str:
    """Return the host CPU name where procfs provides it."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _emit(stream: Any, event: str, **payload: Any) -> None:
    """Write a flushed machine-readable event."""
    stream.write(
        json.dumps(
            {"event": event, "time": time.time(), **payload},
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    stream.flush()


def _scalar(value: torch.Tensor) -> Any:
    """Detach an ordinary diagnostic tensor into JSON-compatible values."""
    return value.detach().cpu().tolist()


def _safe_shift_update(
    density: torch.Tensor,
    delta: torch.Tensor,
    kt: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnostic guarded shift with all dangerous operands sanitized first."""
    from benchmarks.e6c1_w2c5_fermi_shift import _shift_density

    updated = _shift_density(density, delta, kt, active)
    valid = torch.isfinite(updated).all(dim=(-1, -2))
    return updated, valid


def _check_invalid_shift_safety(stream: Any) -> None:
    """Assert invalid input lanes are isolated before division and solves."""
    from torch.func import vmap

    densities = torch.stack(
        (
            torch.tensor([[0.7, 0.0], [0.0, 0.2]], dtype=DTYPE),
            torch.tensor([[0.8, 0.0], [0.0, float("nan")]], dtype=DTYPE),
            torch.tensor([[0.8, 0.0], [0.0, 0.2]], dtype=DTYPE),
            torch.tensor([[0.8, 0.0], [0.0, 0.2]], dtype=DTYPE),
            torch.tensor([[0.8, 0.0], [0.0, float("nan")]], dtype=DTYPE),
        )
    )
    delta = torch.tensor([0.01, 0.01, float("nan"), 0.01, 0.01], dtype=DTYPE)
    kt = torch.tensor([0.2, 0.2, 0.2, 0.0, 0.2], dtype=DTYPE)
    active = torch.tensor(
        [
            [True, True],
            [True, True],
            [True, True],
            [True, True],
            [True, False],
        ],
        dtype=torch.bool,
    )
    updated, valid = vmap(_safe_shift_update)(densities, delta, kt, active)
    assert valid.tolist() == [True, False, False, False, True]
    assert torch.isfinite(updated[0]).all()
    assert torch.isnan(updated[1:4]).all()
    assert torch.isfinite(updated[4]).all()
    assert torch.count_nonzero(updated[4, 1]) == 0
    batch_gradient = torch.func.grad(
        lambda value: vmap(_safe_shift_update)(value, delta, kt, active)[0][
            [0, 4]
        ].sum()
    )(densities)
    for lane_index in (0, 4):
        lane_gradient = torch.func.grad(
            lambda value: _safe_shift_update(
                value,
                delta[lane_index],
                kt[lane_index],
                active[lane_index],
            )[0].sum()
        )(densities[lane_index])
        torch.testing.assert_close(
            batch_gradient[lane_index],
            lane_gradient,
            atol=0.0,
            rtol=0.0,
        )
    assert torch.isfinite(batch_gradient[[0, 4]]).all()
    assert torch.count_nonzero(batch_gradient[[1, 2, 3]]) == 0
    reference, reference_valid = _safe_shift_update(
        densities[0], delta[0], kt[0], active[0]
    )
    torch.testing.assert_close(updated[0], reference, atol=0.0, rtol=0.0)
    assert bool(reference_valid)
    _emit(
        stream,
        "invalid_shift_operand_isolation",
        valid_lanes=valid.tolist(),
        output_lane_finite=torch.isfinite(updated).all(dim=(-1, -2)).tolist(),
        valid_lane_gradients_match_serial=True,
        assertions_passed=True,
    )


def _model_and_case(
    case_id: str,
) -> tuple[Any, Any, torch.Tensor, torch.Tensor, str]:
    """Construct a frozen workload or a documented small physical control."""
    if case_id.startswith("tmpda"):
        index = int(case_id[-1])
        numbers, positions = w2c2.construct_workload(index)
        method, method_label, scp_mode, temperature = (
            GFN1_XTB,
            labels.GFN1_XTB,
            labels.SCP_MODE_CHARGE,
            300.0,
        )
        system_id = f"tmpda-conformer-{index}-frozen-W2"
    elif case_id in (
        "h2o-gfn1",
        "h2o-gfn1-10000K",
        "h2o-gfn2-10000K",
    ):
        numbers = mols["H2O"]["numbers"].to(torch.int64)
        positions = mols["H2O"]["positions"].to(DTYPE)
        gfn2_case = case_id == "h2o-gfn2-10000K"
        high_temperature = case_id.endswith("10000K")
        method = GFN2_XTB if gfn2_case else GFN1_XTB
        method_label = labels.GFN2_XTB if gfn2_case else labels.GFN1_XTB
        scp_mode = labels.SCP_MODE_FOCK if gfn2_case else labels.SCP_MODE_CHARGE
        temperature = 10_000.0 if high_temperature else 300.0
        system_id = case_id
    else:
        raise ValueError(f"Unknown physical case: {case_id}")

    parameters = ParamModule(method, dtype=DTYPE)
    config = Config.create(
        method=method_label,
        int_driver=labels.INTDRIVER_PYTORCH,
        mixer=labels.MIXER_ANDERSON,
        scf_mode=labels.SCF_MODE_FULL,
        scp_mode=scp_mode,
        maxiter=300,
        f_atol=1.0e-8,
        x_atol=1.0e-8,
        x_atol_max=1.0e-5,
        fermi_etemp=temperature,
        force_convergence=False,
    )
    model = Model(par=parameters, config=config, auto_int_level=True)
    system = model.setup(numbers, batch_mode=0)
    return model, system, numbers, positions, system_id


class CorrectionRecorder:
    """Scoped wrappers that observe, but do not replace, Fermi calculations."""

    def __init__(self, system_id: str, stream: Any) -> None:
        self.system_id = system_id
        self.stream = stream
        self.map_count = 0
        self.stage = "primal-scf"
        self.current_density: dict[str, Any] | None = None
        self.corrections: list[dict[str, Any]] = []
        self.snapshots: list[dict[str, Any]] = []
        self.performance_input: dict[str, Any] | None = None
        self._restore: list[tuple[Any, str, Any]] = []

    def _replace(self, module: Any, name: str, function: Any) -> None:
        self._restore.append((module, name, getattr(module, name)))
        setattr(module, name, function)

    def __enter__(self) -> CorrectionRecorder:
        original_matrix_fermi = spectral_module._matrix_fermi
        original_density = spectral_module.fermi_density_matrix
        original_free_energy_setup = spectral_module._fermi_free_energy_setup
        original_trace = solver_module.fermi_free_energy_trace

        def observe_matrix_fermi(
            matrix: torch.Tensor,
            shift: torch.Tensor,
            kt: torch.Tensor,
            active: torch.Tensor,
        ) -> torch.Tensor:
            value = original_matrix_fermi(matrix, shift, kt, active)
            record = self.current_density
            if record is not None:
                record["calls"].append(
                    {
                        "matrix": matrix.detach().clone(),
                        "shift": shift.detach().clone(),
                        "kt": kt.detach().clone(),
                        "active": active.detach().clone(),
                        "density": value.detach().clone(),
                    }
                )
            return value

        def observe_density(
            hamiltonian: torch.Tensor,
            nel: torch.Tensor,
            kt: torch.Tensor,
            state: Any,
            **kwargs: Any,
        ) -> torch.Tensor:
            if self.stage.startswith("implicit-response"):
                return original_density(hamiltonian, nel, kt, state, **kwargs)
            record: dict[str, Any] = {
                "system_id": self.system_id,
                "observable_kind": "density",
                "stage": self.stage,
                "map_evaluation": self.map_count,
                "nel": nel.detach().clone(),
                "kt": kt.detach().clone(),
                "state": state,
                "calls": [],
            }
            self.current_density = record
            succeeded = False
            try:
                result = original_density(hamiltonian, nel, kt, state, **kwargs)
                succeeded = True
                return result
            finally:
                self.current_density = None
                self._finish_density(record, succeeded)

        def observe_free_energy_setup(
            hamiltonian: torch.Tensor,
            nel: torch.Tensor,
            kt: torch.Tensor,
            state: Any,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if self.stage.startswith("implicit-response"):
                return original_free_energy_setup(
                    hamiltonian, nel, kt, state, *args, **kwargs
                )
            record: dict[str, Any] = {
                "system_id": self.system_id,
                "observable_kind": "free-energy",
                "stage": self.stage,
                "map_evaluation": self.map_count,
                "nel": nel.detach().clone(),
                "kt": kt.detach().clone(),
                "state": state,
                "calls": [],
            }
            self.current_density = record
            succeeded = False
            try:
                result = original_free_energy_setup(
                    hamiltonian, nel, kt, state, *args, **kwargs
                )
                succeeded = True
                return result
            finally:
                self.current_density = None
                self._finish_density(record, succeeded)

        def observe_free_energy_trace(*args: Any, **kwargs: Any) -> Any:
            previous_stage = self.stage
            self.stage = f"{previous_stage}-free-energy"
            try:
                return original_trace(*args, **kwargs)
            finally:
                self.stage = previous_stage

        def observe_map(function: Any):
            def mapped(*args: Any, **kwargs: Any) -> Any:
                self.map_count += 1
                return function(*args, **kwargs)

            return mapped

        self._replace(spectral_module, "_matrix_fermi", observe_matrix_fermi)
        self._replace(
            spectral_module,
            "_fermi_free_energy_setup",
            observe_free_energy_setup,
        )
        self._replace(map_module, "fermi_density_matrix", observe_density)
        self._replace(
            solver_module, "fermi_free_energy_trace", observe_free_energy_trace
        )
        for module, names in (
            (map_module, ("scf_map", "fock_map", "potential_map")),
            (eager_module, ("scf_map", "fock_map", "potential_map")),
            (w2c2, ("scf_map",)),
        ):
            for name in names:
                if hasattr(module, name):
                    self._replace(
                        module, name, observe_map(getattr(module, name))
                    )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for module, name, original in reversed(self._restore):
            setattr(module, name, original)

    def _finish_density(self, record: dict[str, Any], succeeded: bool) -> None:
        """Derive corrections from the recorded unmodified function calls."""
        state = record.pop("state")
        calls = record.pop("calls")
        target = record.pop("nel")
        kt = record.pop("kt")
        if len(calls) < 3:
            _emit(
                self.stream,
                "correction_call_incomplete",
                **record,
                matrix_fermi_call_count=len(calls),
                original_completed=succeeded,
            )
            return

        ref = state.chemical_potential_ref.squeeze(-1)
        offset = state.chemical_potential_offset.squeeze(-1)
        active_channel = state.active
        diff_mask = (
            active_channel
            & ~state.gap.to(torch.bool)
            & ~state.empty
            & ~state.full
            & ~state.too_many
        )

        for step in range(2):
            before = calls[step]
            delta = calls[step + 1]["shift"] - before["shift"]
            density = before["density"]
            count = density.diagonal(dim1=-2, dim2=-1).sum(-1)
            slope = (density - density @ density).diagonal(
                dim1=-2, dim2=-1
            ).sum(-1) / kt
            correction_ratio = delta / kt
            admitted = correction_ratio.abs() <= GUARD_RATIO.to(
                correction_ratio.device
            )
            channel_count = before["active"].sum(-1)
            details = []
            for channel in range(target.numel()):
                details.append(
                    {
                        "channel": channel,
                        "electron_target": float(target[channel]),
                        "temperature_kt": float(kt.expand_as(target)[channel]),
                        "active_orbitals": int(channel_count[channel]),
                        "chemical_potential_reference": float(ref[channel]),
                        "chemical_potential_offset": float(offset[channel]),
                        "electron_count_before": float(count[channel]),
                        "electron_count_residual": float(
                            target[channel] - count[channel]
                        ),
                        "response_slope": float(slope[channel]),
                        "delta": float(delta[channel]),
                        "delta_over_kt": float(correction_ratio[channel]),
                        "diff_mask": bool(diff_mask[channel]),
                        "guard_admits": bool(admitted[channel]),
                    }
                )
            self.corrections.append(
                {
                    "system_id": record["system_id"],
                    "observable_kind": record["observable_kind"],
                    "stage": record["stage"],
                    "map_evaluation": record["map_evaluation"],
                    "correction_step": step + 1,
                    "channels": details,
                }
            )
            _emit(
                self.stream,
                "physical_mu_correction",
                system_id=record["system_id"],
                observable_kind=record["observable_kind"],
                stage=record["stage"],
                map_evaluation=record["map_evaluation"],
                correction_step=step + 1,
                original_completed=succeeded,
                channels=details,
            )

            if (
                record["map_evaluation"] in (1, 2, 10, 30)
                or record["stage"] != "primal-scf"
            ):
                self.snapshots.append(
                    {
                        **record,
                        "step": step + 1,
                        "matrix": before["matrix"],
                        "shift": before["shift"],
                        "delta": delta,
                        "kt": before["kt"],
                        "active": before["active"],
                        "density": density,
                    }
                )

        if (
            self.performance_input is None
            and record["stage"] == "primal-scf"
            and record["observable_kind"] == "density"
        ):
            self.performance_input = {
                "system_id": record["system_id"],
                "stage": record["stage"],
                "map_evaluation": record["map_evaluation"],
                "matrix": calls[0]["matrix"].cpu(),
                "shifts": torch.stack([call["shift"] for call in calls]).cpu(),
                "kt": calls[0]["kt"].cpu(),
                "active": calls[0]["active"].cpu(),
                "target": target.detach().cpu(),
                "diff_mask": diff_mask.detach().cpu(),
                "reference_densities": torch.stack(
                    [call["density"] for call in calls]
                ).cpu(),
            }


def _conditioning_diagnostics(recorder: CorrectionRecorder) -> None:
    """Inspect selected physical density/update matrices outside autograd."""
    from tad_mctc.storch.linsolve import dense_solve

    original = spectral_module._matrix_fermi
    samples: list[dict[str, Any]] = []
    with torch.no_grad():
        for sample in recorder.snapshots:
            density = sample["density"]
            matrix = sample["matrix"]
            shift = sample["shift"]
            delta = sample["delta"]
            kt = sample["kt"].expand_as(delta)
            active = sample["active"]
            ratio = -delta / kt
            coefficient = torch.expm1(ratio)
            eye = torch.eye(
                density.shape[-1], dtype=density.dtype, device=density.device
            )
            denominator = eye + coefficient[:, None, None] * (eye - density)
            updated = dense_solve(denominator, density)
            direct = original(matrix, shift + delta, kt, active)

            channels = []
            for channel in range(delta.numel()):
                indices = torch.nonzero(active[channel]).flatten()
                d_active = (
                    density[channel]
                    .index_select(0, indices)
                    .index_select(1, indices)
                )
                b_active = (
                    denominator[channel]
                    .index_select(0, indices)
                    .index_select(1, indices)
                )
                d_eigenvalues = torch.linalg.eigvalsh(d_active)
                b_eigenvalues = torch.linalg.eigvalsh(b_active)
                minimum = float(b_eigenvalues.abs().min())
                maximum = float(b_eigenvalues.abs().max())
                channels.append(
                    {
                        "channel": channel,
                        "symmetry_max_abs": float(
                            (d_active - d_active.T).abs().max()
                        ),
                        "density_eigen_min": float(d_eigenvalues.min()),
                        "density_eigen_max": float(d_eigenvalues.max()),
                        "density_distance_below_zero": max(
                            0.0, -float(d_eigenvalues.min())
                        ),
                        "density_distance_above_one": max(
                            0.0, float(d_eigenvalues.max()) - 1.0
                        ),
                        "nearly_empty_count": int(
                            (d_eigenvalues < 1.0e-12).sum()
                        ),
                        "nearly_full_count": int(
                            (d_eigenvalues > 1.0 - 1.0e-12).sum()
                        ),
                        "shift_over_kt": float(ratio[channel]),
                        "denominator_eigen_min": float(b_eigenvalues.min()),
                        "denominator_eigen_max": float(b_eigenvalues.max()),
                        "denominator_condition": maximum / minimum,
                        "updated_vs_direct_max_abs": float(
                            (updated[channel] - direct[channel]).abs().max()
                        ),
                    }
                )
            samples.append(
                {
                    "system_id": sample["system_id"],
                    "stage": sample["stage"],
                    "map_evaluation": sample["map_evaluation"],
                    "correction_step": sample["step"],
                    "channels": channels,
                }
            )
    if samples:
        _emit(recorder.stream, "selected_density_conditioning", samples=samples)


def run_case(
    case_id: str, output: Path, corrections_only: bool = False
) -> None:
    """Run an accepted physical System under a correction recorder."""
    mismatches = {
        key: (os.environ.get(key), value)
        for key, value in THREAD_ENV.items()
        if os.environ.get(key) != value
    }
    if mismatches:
        raise RuntimeError(
            f"Thread environment must be configured: {mismatches}"
        )
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (torch.get_num_threads(), torch.get_num_interop_threads()) != (1, 1):
        raise RuntimeError("Expected one intra-op and inter-op thread")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)
    model, system, numbers, positions, system_id = _model_and_case(case_id)
    recorder = CorrectionRecorder(system_id, output.open("a", encoding="utf-8"))
    _emit(
        recorder.stream,
        "environment",
        system_id=system_id,
        python=platform.python_version(),
        torch=torch.__version__,
        cpu=_cpu_name(),
        source=str(Path(dxtb.__file__).resolve()),
        threads={
            "torch_intra": torch.get_num_threads(),
            "torch_inter": torch.get_num_interop_threads(),
            **THREAD_ENV,
        },
        dependencies={
            name: importlib.metadata.version(name)
            for name in (
                "tad-mctc",
                "tad-multicharge",
                "tad-libcint",
                "tad-dftd3",
                "tad-dftd4",
            )
        },
        atoms=int(numbers.numel()),
        temperature_kelvin=system.config.scf.fermi.etemp,
        method=system.config.method,
        scp_mode=system.config.scf.scp_mode,
        maxiter=system.config.scf.maxiter,
        x_atol=system.config.scf.x_atol,
        x_atol_max=system.config.scf.x_atol_max,
        f_atol=system.config.scf.f_atol,
    )
    _check_invalid_shift_safety(recorder.stream)

    try:
        if case_id.startswith("tmpda") or case_id == "h2o-gfn1":
            positions = positions.detach().clone().requires_grad_(True)
            case = w2c2.PhysicalCase(
                model=model,
                system=system,
                numbers=numbers,
                positions=positions,
                charge=torch.zeros(1, dtype=DTYPE),
                spin=None,
            )
            prepared = w2c2.prepare_call_inputs(case)
            with recorder:
                outcome, count, duration = w2c2.prepare_primal_root(prepared)
                recorder.stage = "final-root-validation"
                validated = w2c2.validate_physical_root(prepared, outcome)
                adapter, tensor_parameters, independent_residual, root = (
                    validated
                )
                # The implicit response linearization differentiates this
                # same physical map at the validated root. Probe its primal
                # correction values outside torch.func so the recorder sees
                # ordinary tensors; the probe is diagnostic and has no graph.
                recorder.stage = "response-primal-base-probe"
                with torch.no_grad():
                    adapter.evaluate(outcome.root_x, *tensor_parameters)
                recorder.stage = "primal-finalization"
                with torch.no_grad():
                    primal_map = adapter.evaluate(
                        outcome.root_x, *tensor_parameters
                    )
                    primal_result, _ = w2c2.finalize_physical_result(
                        prepared, outcome, primal_map
                    )
                if corrections_only:
                    differentiable_result = primal_result
                else:
                    recorder.stage = "differentiable-finalization"
                    implicit_value = w2c2.attach_implicit_root(root)
                    implicit_map = adapter.evaluate(
                        implicit_value, *tensor_parameters
                    )
                    differentiable_result, _ = w2c2.finalize_physical_result(
                        prepared, outcome, implicit_map
                    )
                    recorder.stage = "implicit-response"
                    response_started = time.perf_counter()
                    _, gradients = w2c2.differentiate_positions(
                        differentiable_result,
                        positions,
                        (),
                        observable="energy",
                    )
                    response_seconds = time.perf_counter() - response_started
                    if (
                        gradients[0] is None
                        or not torch.isfinite(gradients[0]).all()
                    ):
                        raise RuntimeError(
                            "The physical position response failed."
                        )
                    _emit(
                        recorder.stream,
                        "implicit_response_completed",
                        duration_s=response_seconds,
                        gradient_norm=float(
                            torch.linalg.vector_norm(gradients[0])
                        ),
                        gradient_max_abs=float(gradients[0].abs().max()),
                        position_gradient=gradients[0].detach().cpu().tolist(),
                        response_primal_correction_values_reused_from=(
                            "independently validated root map; no value extraction "
                            "was attempted inside torch.func transforms"
                        ),
                    )
            physical_result = differentiable_result
            solver = outcome
            converged = bool(outcome.converged)
            fermi_status = outcome.map_result.fermi_status
            _emit(
                recorder.stream,
                "physical_root_status",
                converged=converged,
                iterations=int(outcome.iterations.detach()),
                fermi_converged=(
                    None
                    if fermi_status is None
                    else _scalar(fermi_status.converged)
                ),
                electron_count_ok=(
                    None
                    if fermi_status is None
                    else _scalar(fermi_status.electron_count_ok)
                ),
                fermi_residual=(
                    None
                    if fermi_status is None
                    else _scalar(fermi_status.residual)
                ),
                physical_residual=_scalar(outcome.residual),
            )
            iterations = int(outcome.iterations.detach())
            map_count = count
        else:
            calculation_started = time.perf_counter()
            with recorder:
                result = system.singlepoint(positions)
            duration = time.perf_counter() - calculation_started
            solver = result
            converged = bool(result.converged)
            independent_residual = None
            iterations = int(result.iterations)
            map_count = iterations
            physical_result = result

        _conditioning_diagnostics(recorder)
        if recorder.performance_input is not None:
            torch.save(
                recorder.performance_input,
                output.with_suffix(".matrix-fermi.pt"),
            )
        guard_ratios = [
            abs(channel["delta_over_kt"])
            for event in _read_corrections(output)
            for channel in event["channels"]
            if channel["diff_mask"]
        ]
        max_ratio = max(guard_ratios, default=0.0)
        _emit(
            recorder.stream,
            "physical_case_complete",
            system_id=system_id,
            iterations=iterations,
            map_evaluations=map_count,
            converged=converged,
            duration_s=duration,
            max_abs_active_delta_over_kt=max_ratio,
            active_correction_count=len(guard_ratios),
            active_guard_admission_fraction=(
                sum(value <= float(GUARD_RATIO) for value in guard_ratios)
                / len(guard_ratios)
                if guard_ratios
                else 1.0
            ),
            independent_residual_norm=(
                None
                if independent_residual is None
                else float(torch.linalg.vector_norm(independent_residual))
            ),
            energy_finite=bool(torch.isfinite(physical_result.energy).all()),
            energy_sum=float(physical_result.energy.detach().sum()),
            free_energy_sum=(
                None
                if physical_result.fenergy is None
                else float(physical_result.fenergy.detach().sum())
            ),
            free_energy_finite=bool(
                torch.isfinite(physical_result.fenergy).all()
            ),
            density_finite=bool(torch.isfinite(physical_result.density).all()),
            fermi_status=(
                None
                if physical_result.status is None
                else {
                    "trajectory_failure": _scalar(
                        physical_result.status.trajectory_failure
                    ),
                    "final_failure": _scalar(
                        physical_result.status.final_failure
                    ),
                }
            ),
        )
    finally:
        recorder.stream.close()


def _read_corrections(path: Path) -> list[dict[str, Any]]:
    """Read streamed correction events for the summary event."""
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line)).get("event") == "physical_mu_correction"
    ]


def main() -> None:
    """Run one bounded diagnostic System."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "tmpda0",
            "tmpda1",
            "h2o-gfn1",
            "h2o-gfn1-10000K",
            "h2o-gfn2-10000K",
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--corrections-only",
        action="store_true",
        help="Stop after root and response-base correction characterization.",
    )
    args = parser.parse_args()
    run_case(args.case, args.output, corrections_only=args.corrections_only)


if __name__ == "__main__":
    main()
