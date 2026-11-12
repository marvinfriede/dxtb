#!/usr/bin/env python3
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
WP1 stage and operation hotspot profiler for LiH higher-order derivatives.

Measures per-stage and per-operation wall-clock time and memory under 0th, 1st,
2nd, and 3rd-order differentiation (both unrolled and implicit) on bounded SCF
budgets.
"""

from __future__ import annotations

import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch

torch.set_num_threads(1)

import dxtb
import dxtb._src.scf.anderson as scf_anderson_mod
import dxtb._src.scf.energy as scf_energy_mod
import dxtb._src.scf.iterator as scf_iterator_mod
import dxtb._src.scf.map as scf_map_mod
import dxtb._src.scf.solver as scf_solver_mod
import dxtb._src.scf.spectral as scf_spectral_mod
import dxtb._src.wavefunction.filling as wf_filling_mod
from dxtb import Calculator, GFN1_XTB


def _read_rss_hwm_mb() -> tuple[float, float]:
    rss_kb = 0.0
    hwm_kb = 0.0
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                rss_kb = float(line.split()[1])
            elif line.startswith("VmHWM:"):
                hwm_kb = float(line.split()[1])
    return rss_kb / 1024.0, hwm_kb / 1024.0


def profile_operations(
    maxiter: int = 10,
    driver: str = "pytorch",
    differentiation: str = "unrolled",
) -> dict[str, Any]:
    """Profile internal stages during energy, dipole, polarizability, and hyperpol."""
    timings: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    def wrap_fn(mod: Any, attr: str, label: str) -> Any:
        orig = getattr(mod, attr)

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            timings[label] += time.perf_counter() - t0
            counts[label] += 1
            return out

        setattr(mod, attr, wrapped)
        return orig

    # Hook stages
    origs = [
        (
            scf_solver_mod,
            "solve_scf_problem",
            wrap_fn(scf_solver_mod, "solve_scf_problem", "2_solve_scf_problem_total"),
        ),
        (
            scf_map_mod,
            "potential_to_hamiltonian",
            wrap_fn(scf_map_mod, "potential_to_hamiltonian", "3a1_potential_to_hamiltonian"),
        ),
        (
            scf_map_mod,
            "potential_from_charges",
            wrap_fn(scf_map_mod, "potential_from_charges", "3a2_potential_from_charges"),
        ),
        (
            scf_anderson_mod,
            "anderson_mix",
            wrap_fn(scf_anderson_mod, "anderson_mix", "3b_anderson_mixer"),
        ),
        (
            scf_map_mod,
            "_transitional_diagonalize",
            wrap_fn(scf_map_mod, "_transitional_diagonalize", "4_spectral_eighb"),
        ),
        (
            scf_spectral_mod,
            "_matrix_fermi_impl",
            wrap_fn(scf_spectral_mod, "_matrix_fermi_impl", "5a_matrix_fermi_density"),
        ),
        (
            scf_spectral_mod,
            "_matrix_expm1",
            wrap_fn(scf_spectral_mod, "_matrix_expm1", "5a1_matrix_expm1"),
        ),
        (
            scf_spectral_mod,
            "dense_solve",
            wrap_fn(scf_spectral_mod, "dense_solve", "5a2_matrix_fermi_dense_solve"),
        ),
        (
            wf_filling_mod,
            "_get_fermi_occupation_core_state",
            wrap_fn(
                wf_filling_mod,
                "_get_fermi_occupation_core_state",
                "5b_fermi_occupation_search",
            ),
        ),
        (
            scf_solver_mod,
            "_finalize_problem_outcome",
            wrap_fn(
                scf_solver_mod, "_finalize_problem_outcome", "6_finalize_scf_energy"
            ),
        ),
    ]
    scf_solver_mod.anderson_mix = scf_anderson_mod.anderson_mix
    scf_iterator_mod.anderson_mix = scf_anderson_mod.anderson_mix

    dtype = torch.float64
    numbers = torch.tensor([3, 1], dtype=torch.long)
    positions = torch.tensor(
        [
            [0.0, 0.0, -1.50796743897235],
            [0.0, 0.0, +1.50796743897235],
        ],
        dtype=dtype,
    )
    charge = torch.tensor(0.0, dtype=dtype)
    field = torch.zeros((3,), dtype=dtype)
    opts = {
        "int_level": dxtb.labels.INTLEVEL_DIPOLE,
        "int_driver": driver,
        "maxiter": maxiter,
        "strategy": "anderson",
        "differentiation": differentiation,
        "verbosity": 0,
        "x_atol": 1e-10,
        "fermi_diff_order": 0,
    }

    t_setup0 = time.perf_counter()
    calc = Calculator(numbers, GFN1_XTB, opts=opts, dtype=dtype)
    sys_obj = calc.system
    setup_s = time.perf_counter() - t_setup0

    results_by_order: dict[str, Any] = {"1_setup_s": round(setup_s, 6)}

    orders = [
        ("0_system_energy", lambda: sys_obj.energy(positions, chrg=charge, field=field)),
        ("1_system_dipole", lambda: sys_obj.dipole(positions, chrg=charge, field=field)),
        (
            "2_system_polarizability",
            lambda: sys_obj.polarizability(positions, chrg=charge, field=field),
        ),
        (
            "3_system_hyperpol",
            lambda: sys_obj.hyperpolarizability(positions, chrg=charge, field=field),
        ),
        (
            "3_calc_hyperpol_functorch_energy",
            lambda: calc.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                use_functorch=True,
                derived_quantity="energy",
                field=field.clone(),
            ),
        ),
        (
            "3_calc_hyperpol_autograd",
            lambda: calc.hyperpolarizability(
                positions.clone(),
                chrg=charge,
                use_functorch=False,
                field=field.clone().requires_grad_(True),
            ),
        ),
    ]

    for order_name, fn in orders:
        timings.clear()
        counts.clear()
        rss_before, hwm_before = _read_rss_hwm_mb()
        t0 = time.perf_counter()
        out = fn()
        total_s = time.perf_counter() - t0
        rss_after, hwm_after = _read_rss_hwm_mb()

        results_by_order[order_name] = {
            "total_s": round(total_s, 6),
            "rss_before_mb": round(rss_before, 2),
            "rss_after_mb": round(rss_after, 2),
            "hwm_after_mb": round(hwm_after, 2),
            "stage_timings_s": {k: round(v, 6) for k, v in sorted(timings.items())},
            "stage_shares_pct": {
                k: round(100.0 * v / total_s, 2) for k, v in sorted(timings.items())
            },
            "stage_counts": dict(sorted(counts.items())),
            "out_norm": float(torch.linalg.norm(out.detach()).item()),
        }

    for mod, attr, orig in origs:
        setattr(mod, attr, orig)

    return results_by_order


if __name__ == "__main__":
    out_data = {
        "pytorch_unrolled_maxiter10": profile_operations(
            maxiter=10, driver="pytorch", differentiation="unrolled"
        ),
        "libcint_unrolled_maxiter10": profile_operations(
            maxiter=10, driver="libcint", differentiation="unrolled"
        ),
        "pytorch_unrolled_maxiter20": profile_operations(
            maxiter=20, driver="pytorch", differentiation="unrolled"
        ),
    }
    out_path = Path(__file__).resolve().parent / "results" / "hod_p1_wp1_stage_profile.json"
    out_path.write_text(json.dumps(out_data, indent=2))
    print(json.dumps(out_data, indent=2))
