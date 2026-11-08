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
"""Compare serial and vmapped fixed-trip System calculations for H2 conformers."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import time
from pathlib import Path

import torch

from dxtb import GFN1_XTB, ParamModule, labels
from dxtb._src.calculators.model import Model
from dxtb.config import Config


def _system(converger: str, maxiter: int) -> object:
    """Create one fixed-trip H2 System with static structure."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=torch.float64
    )
    config = Config.create(
        method=labels.GFN1_XTB,
        int_driver=labels.INTDRIVER_PYTORCH,
        int_level=labels.INTLEVEL_HCORE,
        strategy=converger,
        differentiation="unrolled",
        maxiter=maxiter,
        force_convergence=False,
        fermi_etemp=0.0,
        x_atol=0.0,
        x_atol_max=0.0,
    )
    model = Model(
        par=ParamModule(GFN1_XTB, dtype=positions.dtype), config=config
    )
    return model.setup(torch.tensor([1, 1]))


def _positions(count: int) -> torch.Tensor:
    distances = torch.linspace(1.25, 1.70, count, dtype=torch.float64)
    origins = torch.zeros((count, 2, 3), dtype=torch.float64)
    origins[:, 1, 2] = distances
    return origins


def _max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float((actual - expected).abs().max()) if actual.numel() else 0.0


def _run(name: str, converger: str, count: int, maxiter: int) -> dict:
    system = _system(converger, maxiter)
    positions = _positions(count)

    started = time.perf_counter()
    mapped = torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)
    mapped_seconds = time.perf_counter() - started

    started = time.perf_counter()
    serial = [system.singlepoint(value) for value in positions]
    serial_seconds = time.perf_counter() - started

    energy = torch.stack([item.energy for item in serial])
    density = torch.stack([item.density for item in serial])
    charges = torch.stack([item.charges.mono for item in serial])
    residual = torch.stack([item.residual for item in serial])
    iterations = torch.stack([item.iterations for item in serial])
    converged = torch.stack([item.converged for item in serial])

    def lane_energy(position: torch.Tensor) -> torch.Tensor:
        return system.singlepoint(position).energy.sum()

    started = time.perf_counter()
    mapped_gradient = torch.func.vmap(torch.func.grad(lane_energy))(positions)
    mapped_gradient_seconds = time.perf_counter() - started
    started = time.perf_counter()
    serial_gradient = torch.stack(
        [torch.func.grad(lane_energy)(position) for position in positions]
    )
    serial_gradient_seconds = time.perf_counter() - started

    torch.testing.assert_close(mapped.energy, energy)
    torch.testing.assert_close(mapped.density, density)
    torch.testing.assert_close(mapped.charges.mono, charges)
    torch.testing.assert_close(mapped.residual, residual)
    torch.testing.assert_close(mapped.iterations, iterations)
    torch.testing.assert_close(mapped.converged, converged)
    torch.testing.assert_close(mapped_gradient, serial_gradient)

    return {
        "method": name,
        "conformers": count,
        "maxiter": maxiter,
        "certified_lanes": int(converged.sum()),
        "physical_maps_per_lane": maxiter + 1,
        "serial_seconds": serial_seconds,
        "vmap_seconds": mapped_seconds,
        "serial_gradient_seconds": serial_gradient_seconds,
        "vmap_gradient_seconds": mapped_gradient_seconds,
        "max_energy_error": _max_error(mapped.energy, energy),
        "max_density_error": _max_error(mapped.density, density),
        "max_charge_error": _max_error(mapped.charges.mono, charges),
        "max_residual_error": _max_error(mapped.residual, residual),
        "max_gradient_error": _max_error(mapped_gradient, serial_gradient),
        "process_peak_rss_kib": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    methods = {
        "Linear": "linear",
        "Anderson": "anderson",
        "Broyden": "broyden",
    }
    runs = [
        _run(name, converger, count, args.maxiter)
        for name, converger in methods.items()
        for count in args.counts
    ]
    result = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True
            ).strip()
        ),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "OMP_NUM_THREADS": "set by caller; expected 1",
        "num_workers": 1,
        "intraop_threads": torch.get_num_threads(),
        "interoperability_threads": torch.get_num_interop_threads(),
        "runs": runs,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
