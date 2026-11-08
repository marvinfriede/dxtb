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
"""Phase breakdown of grouped-vmap evaluation (benchmark case B).

Wall time of one energy evaluation (no autograd) is attributed to

- ``integrals``:  ``build_integral_matrices``;
- ``scf_prep``:   ``setup_scf_inputs`` (orthonormalizer, interaction setup);
- ``scf_solve``:  ``_solve_variable`` (all map trips, final spectral
  evaluation and status);
- ``eigensolver``: every symmetric eigensolver call (inside ``scf_solve``),
  with its call count = number of spectral evaluations;
- ``split``:      ``_select_lane`` (mapped Result -> lane Results);
- ``pack``:       ``_pack_lane_results``;
- ``other``:      the remainder (classicals, interaction data, Result
  assembly, dispatch).

Cases: one H2O group (8 lanes), one CH4 group (8), one NH3 group (4), and the
public heterogeneous case B (all three groups in one batched Calculator call),
each with the grouped-vmap route and with every lane evaluated eagerly.

Usage::

    OMP_NUM_THREADS=1 python benchmarks/grouped_vmap_phase_profile.py \
        --output benchmarks/results/grouped-vmap-phase-profile.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "test"))
sys.path.insert(0, str(ROOT / "benchmarks"))

DTYPE = torch.float64
GROUPS = {"H2O": 8, "CH4": 8, "NH3": 4}

_TIMES: dict[str, float] = defaultdict(float)
_CALLS: dict[str, int] = defaultdict(int)


def _timed(name: str, function):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _TIMES[name] += time.perf_counter() - start
            _CALLS[name] += 1

    return wrapper


def _instrument() -> None:
    from tad_mctc import storch

    from dxtb._src.calculators import lane_evaluation, packed_result
    from dxtb._src.calculators import singlepoint as sp_mod
    from dxtb._src.scf.functional import eager as eager_mod

    sp_mod.build_integral_matrices = _timed(
        "integrals", sp_mod.build_integral_matrices
    )
    eager_mod.setup_scf_inputs = _timed("scf_prep", eager_mod.setup_scf_inputs)
    eager_mod._solve_variable = _timed(
        "scf_solve", eager_mod._solve_variable
    )
    torch.linalg.eigh = _timed("eigensolver", torch.linalg.eigh)
    storch.eighb = _timed("eigensolver", storch.eighb)
    lane_evaluation._select_lane = _timed("split", lane_evaluation._select_lane)
    packed_result._pack_lane_results = _timed(
        "pack", packed_result._pack_lane_results
    )


def _lane_inputs(case: str):
    from molecules import mols

    names = (
        [name for name, count in GROUPS.items() for _ in range(count)]
        if case == "B"
        else [case] * GROUPS[case]
    )
    order = torch.randperm(
        len(names), generator=torch.Generator().manual_seed(7)
    ).tolist()
    names = [names[i] for i in order]
    generator = torch.Generator().manual_seed(11)
    lanes = []
    for name in names:
        base = mols[name]["positions"].to(DTYPE)
        noise = 0.03 * torch.randn(base.shape, generator=generator, dtype=DTYPE)
        lanes.append((name, mols[name]["numbers"], base + noise))
    return lanes


def run(case: str, strategy: str, repeats: int) -> dict[str, object]:
    from tad_mctc.batch import pack

    from dxtb import GFN1_XTB, Calculator, labels
    from dxtb._src.calculators import lane_evaluation

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    _instrument()
    lanes = _lane_inputs(case)
    numbers = pack([numbers for _, numbers, _ in lanes])
    positions = pack([positions for _, _, positions in lanes])
    opts = {
        "method": labels.GFN1_XTB,
        "int_driver": labels.INTDRIVER_PYTORCH,
        "verbosity": 0,
    }
    calc = Calculator(numbers, GFN1_XTB, opts=opts, dtype=DTYPE)

    if strategy == "eager":
        # Same public call, but every group is evaluated lane by lane.
        original = lane_evaluation.evaluate_lane_group

        def eager_group(group, *args):
            return original(
                lane_evaluation.LaneGroup(
                    group.system, group.source_ids, False
                ),
                *args,
            )

        from dxtb._src.calculators import packed_result

        packed_result.evaluate_lane_group = eager_group

    runs = []
    for _ in range(repeats):
        _TIMES.clear()
        _CALLS.clear()
        start = time.perf_counter()
        with torch.no_grad():
            result = calc.singlepoint(positions)
        total = time.perf_counter() - start
        phases = dict(_TIMES)
        top = ("integrals", "scf_prep", "scf_solve", "split", "pack")
        phases["other"] = total - sum(phases.get(name, 0.0) for name in top)
        runs.append(
            {
                "total_s": total,
                "phases_s": phases,
                "calls": dict(_CALLS),
                "iterations": result.iterations.tolist(),
            }
        )
    best = min(runs, key=lambda entry: entry["total_s"])
    return {
        "case": case,
        "strategy": strategy,
        "lanes": len(lanes),
        "energy_sum": float(result.energy.sum()),
        **best,
    }


def main() -> None:
    from grouped_vmap_batching import _run_with_rss_limit

    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=[*GROUPS, "B"])
    parser.add_argument("--strategy", choices=["vmap", "eager"])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--mem-gib", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.case is not None:
        print(json.dumps(run(args.case, args.strategy, args.repeats)))
        return

    report: dict[str, object] = {
        "commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": "cpu",
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "cases": [],
    }
    for case in [*GROUPS, "B"]:
        for strategy in ("vmap", "eager"):
            code, out, err, killed = _run_with_rss_limit(
                [
                    sys.executable,
                    __file__,
                    "--case",
                    case,
                    "--strategy",
                    strategy,
                    "--repeats",
                    str(args.repeats),
                ],
                int(args.mem_gib * 1024**3),
            )
            if code != 0:
                reason = "killed" if killed else err.strip().splitlines()[-1]
                print(f"{case}/{strategy}: FAILED ({reason})", flush=True)
                report["cases"].append(
                    {"case": case, "strategy": strategy, "error": reason}
                )
                continue
            entry = json.loads(out.strip().splitlines()[-1])
            phases = ", ".join(
                f"{name} {value:.2f}"
                for name, value in sorted(entry["phases_s"].items())
            )
            print(
                f"{case}/{strategy}: {entry['total_s']:.2f} s ({phases}); "
                f"eigensolver calls {entry['calls'].get('eigensolver', 0)}",
                flush=True,
            )
            report["cases"].append(entry)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
