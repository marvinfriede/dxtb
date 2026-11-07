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
"""Benchmark heterogeneous Calculator batching (canonical vs. grouped vmap).

The script only uses the public batched ``Calculator`` API, so the same file
runs against the canonical-padding baseline and the grouped-vmap route. Each
case runs in its own subprocess so that peak RSS is attributable to that case.

Usage::

    OMP_NUM_THREADS=1 python benchmarks/grouped_vmap_batching.py \
        --output benchmarks/results/grouped-vmap-<label>.json

Only CPU is measured. The canonical atom/AO extent is reported when the
running revision still provides ``canonical_padding``. A watchdog kills a
case whose resident memory exceeds ``--mem-gib``; that case is recorded as
failed instead of exhausting host memory.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))

DTYPE = torch.float64

# Each case maps to an ordered list of (molecule name, lane count). Lanes are
# shuffled with a fixed seed so that source order is heterogeneous. Lane
# counts are sized for a CPU host: every mapped lane records all fixed SCF
# trips on the autograd tape, which dominates memory for energy + forces.
CASES: dict[str, list[tuple[str, int]]] = {
    "A_conformers": [("H2O", 32)],
    "B_repeated": [("H2O", 8), ("CH4", 8), ("NH3", 4)],
    "C_heterogeneous": [
        ("H2", 1),
        ("H2O", 1),
        ("NH3", 1),
        ("CH4", 1),
        ("SiH4", 1),
        ("C4H5NCS", 1),
    ],
    "D_mixed_sizes": [("H2O", 7), ("C4H5NCS", 1)],
}


def _build_inputs(case: str, seed: int = 7):
    from molecules import mols
    from tad_mctc.batch import pack

    generator = torch.Generator().manual_seed(seed)
    lanes: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    for name, count in CASES[case]:
        numbers = mols[name]["numbers"]
        positions = mols[name]["positions"].to(DTYPE)
        for _ in range(count):
            noise = 0.05 * torch.randn(
                positions.shape, generator=generator, dtype=DTYPE
            )
            lanes.append((name, numbers, positions + noise))
    order = torch.randperm(len(lanes), generator=generator).tolist()
    lanes = [lanes[i] for i in order]
    names = [name for name, _, _ in lanes]
    numbers = pack([n for _, n, _ in lanes])
    positions = pack([p for _, _, p in lanes])
    return names, numbers, positions


def _structure_counts(calc, numbers: torch.Tensor) -> dict[str, object]:
    """Real atoms/AOs per lane and the lane-System grouping."""
    from dxtb._src.calculators.packed_result import (
        build_calculator_lane_systems,
    )

    systems = build_calculator_lane_systems(calc)
    groups: dict[int, int] = {}
    for system in systems:
        groups[id(system)] = groups.get(id(system), 0) + 1
    real_atoms = [int((row > 0).sum()) for row in numbers]
    real_aos = [int(s.ihelp.nao) for s in systems]
    info: dict[str, object] = {
        "lanes": len(systems),
        "distinct_systems": len(groups),
        "group_sizes": sorted(groups.values(), reverse=True),
        "real_atoms_max": max(real_atoms),
        "real_aos_max": max(real_aos),
        "real_aos_sum": sum(real_aos),
    }
    try:
        from dxtb._src.calculators.canonical_padding import (
            bucket_canonical_exact_cores,
        )
    except ImportError:
        info["canonical_extent"] = None
        return info

    bucket = bucket_canonical_exact_cores(
        tuple(system.exact_core for system in systems)
    )
    info["canonical_extent"] = {
        "atoms": bucket.atoms,
        "aos": bucket.orbitals,
        "padded_aos_sum": bucket.orbitals * len(systems),
    }
    return info


MODES = ("energy", "forces")


def run_case(case: str, mode: str, repeats: int) -> dict[str, object]:
    """Time one case; ``energy`` runs without autograd, ``forces`` with."""
    from dxtb import GFN1_XTB, Calculator, labels

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    names, numbers, positions = _build_inputs(case)
    opts = {
        "method": labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "verbosity": 0,
    }
    calc = Calculator(numbers, GFN1_XTB, opts=opts, dtype=DTYPE)
    info = _structure_counts(calc, numbers)

    timings = []
    force_norm = None
    for _ in range(repeats):
        start = time.perf_counter()
        if mode == "energy":
            with torch.no_grad():
                result = calc.singlepoint(positions)
            energy = result.energy.sum()
        else:
            pos = positions.clone().requires_grad_(True)
            result = calc.singlepoint(pos)
            energy = result.energy.sum()
            (forces,) = torch.autograd.grad(energy, pos)
            force_norm = float(forces.norm())
        timings.append(time.perf_counter() - start)

    info.update(
        {
            "case": case,
            "mode": mode,
            "composition": CASES[case],
            "energy_sum": float(energy.detach()),
            "force_norm": force_norm,
            "iterations": result.iterations.tolist(),
            "wall_s": timings,
            "wall_s_min": min(timings),
            "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024.0,
        }
    )
    return info


def _rss_bytes(pid: int) -> int:
    """Resident set size of a live process, read from /proc."""
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def _run_with_rss_limit(
    command: list[str], limit: int
) -> tuple[int, str, str, bool]:
    """Run a case and kill it if its resident memory exceeds ``limit``."""
    with tempfile.TemporaryFile("w+") as out, tempfile.TemporaryFile(
        "w+"
    ) as err:
        process = subprocess.Popen(command, stdout=out, stderr=err, text=True)
        killed = False
        while process.poll() is None:
            if _rss_bytes(process.pid) > limit:
                process.kill()
                killed = True
                break
            time.sleep(0.2)
        returncode = process.wait()
        out.seek(0)
        err.seek(0)
        return returncode, out.read(), err.read(), killed


def _run_case_subprocess(
    case: str, mode: str, args: argparse.Namespace
) -> dict[str, object]:
    """Run one case/mode in a memory-capped subprocess and summarize it."""
    returncode, stdout, stderr, killed = _run_with_rss_limit(
        [
            sys.executable,
            __file__,
            "--case",
            case,
            "--mode",
            mode,
            "--repeats",
            str(args.repeats),
        ],
        int(args.mem_gib * 1024**3),
    )
    if returncode != 0:
        error = (
            f"killed: resident memory above {args.mem_gib} GiB"
            if killed
            else (stderr.strip().splitlines() or ["?"])[-1]
        )
        print(f"{case}/{mode}: FAILED ({error})", flush=True)
        return {
            "case": case,
            "mode": mode,
            "status": "failed",
            "returncode": returncode,
            "error": error,
        }

    result = json.loads(stdout.strip().splitlines()[-1])
    result["status"] = "ok"
    print(
        f"{case}/{mode}: {result['wall_s_min']:.2f} s, "
        f"{result['peak_rss_mib']:.0f} MiB, groups {result['group_sizes']}",
        flush=True,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--mode", choices=MODES, default="forces")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mem-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.case is not None:
        print(json.dumps(run_case(args.case, args.mode, args.repeats)))
        return

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    report: dict[str, object] = {
        "commit": commit,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mem_limit_gib": args.mem_gib,
        "cases": {},
    }
    for case in CASES:
        for mode in MODES:
            key = f"{case}/{mode}"
            report["cases"][key] = _run_case_subprocess(case, mode, args)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
