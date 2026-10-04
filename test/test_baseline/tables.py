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
Render the recorded status as Markdown tables for the baseline report.

Run from the repository root::

    python -m test.test_baseline.tables > docs/plan/T0-tables.md
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from . import matrix as mx
from .components import CHECKS, TARGETS
from .status import STATUS_DIR

SYMBOL = {"pass": "✓", "fail": "✗", "error": "E", "nonfinite": "NaN"}


def _load(name: str) -> dict:
    path = STATUS_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def _summary(results: list[dict]) -> str:
    if not results:
        return "–"
    c = Counter(r["status"] for r in results)
    if len(c) == 1:
        (status,) = c
        return SYMBOL[status]
    return " ".join(f"{SYMBOL[s]}{n}" for s, n in sorted(c.items()))


def matrix_tables() -> str:
    data = _load("matrix.json")
    cells = {c.id: c for c in mx.CELLS}
    by = defaultdict(list)
    for cid, res in data.items():
        c = cells.get(cid)
        if c is None:
            continue
        by[(c.quantity, c.path, c.input)].append((c, res))

    paths = ("analytical", "autograd", "functorch", "forward", "numerical")
    lines = []
    for inp in mx.INPUTS:
        lines += [
            f"#### Input: {inp}",
            "",
            "Each entry aggregates method × driver × SCF mode "
            "(✓ pass, ✗ wrong value, E exception, NaN not finite; counts if "
            "mixed).",
            "",
            "| quantity | order | " + " | ".join(paths) + " |",
            "| --- | --- | " + " | ".join("---" for _ in paths) + " |",
        ]
        for q, (order, qpaths, inputs) in mx.QUANTITIES.items():
            if inp not in inputs:
                continue
            row = [q, str(order)]
            for p in paths:
                if p not in qpaths:
                    row.append("n/a")
                    continue
                row.append(_summary([r for _, r in by[(q, p, inp)]]))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
    return "\n".join(lines)


def matrix_by_setting() -> str:
    """Pass counts per method, driver and SCF mode (single system)."""
    data = _load("matrix.json")
    cells = {c.id: c for c in mx.CELLS}
    count = defaultdict(Counter)
    for cid, res in data.items():
        c = cells.get(cid)
        if c is None or c.input != "single":
            continue
        count[(c.method, c.driver, c.scf)][res["status"]] += 1
    lines = [
        "| method | driver | SCF mode | ✓ | ✗ | E | NaN |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for (m, d, s), c in sorted(count.items()):
        lines.append(
            f"| {m} | {d} | {s} | {c['pass']} | {c['fail']} | {c['error']} "
            f"| {c['nonfinite']} |"
        )
    return "\n".join(lines)


def matrix_failures() -> str:
    """Distinct failure messages with the number of cells."""
    data = _load("matrix.json")
    msgs = Counter()
    example = {}
    for cid, res in data.items():
        if res["status"] == "pass":
            continue
        msg = res["message"] or (
            f"wrong value (max_abs={res['max_abs']:.1e})"
            if res.get("max_abs") is not None
            else res["status"]
        )
        key = (res["status"], msg.split(" [")[0][:110])
        msgs[key] += 1
        example.setdefault(key, cid)
    lines = [
        "| cells | status | message | example cell |",
        "| --- | --- | --- | --- |",
    ]
    for (status, msg), n in msgs.most_common(40):
        lines.append(
            f"| {n} | {status} | {msg.replace('|', '/')} | `{example[(status, msg)]}` |"
        )
    return "\n".join(lines)


def matrix_cost() -> str:
    data = _load("matrix.json")
    rows = []
    for q in mx.QUANTITIES:
        for p in mx.QUANTITIES[q][1]:
            cid = f"{q}-{p}-gfn2-pytorch-full-single"
            r = data.get(cid)
            if r is None:
                continue
            rows.append(
                f"| {q} | {p} | {SYMBOL[r['status']]} | "
                + (
                    f"{r['max_abs']:.1e}"
                    if r.get("max_abs") not in (None, float("inf"))
                    else "–"
                )
                + f" | {r['seconds']:.1f} | {r.get('peak_rss_mb') or 0:.0f} |"
            )
    head = [
        "GFN2, PyTorch driver, unrolled SCF, water:",
        "",
        "| quantity | path | status | max abs. dev. | wall time (s) | peak RSS (MB) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    return "\n".join(head + rows)


def component_table() -> str:
    data = _load("components.json")
    lines = [
        "| target | " + " | ".join(CHECKS) + " |",
        "| --- | " + " | ".join("---" for _ in CHECKS) + " |",
    ]
    for t in TARGETS:
        row = [f"`{t}`"]
        for c in CHECKS:
            r = data.get(f"{t}:{c}")
            row.append(SYMBOL[r["status"]] if r else "–")
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "Messages of the checks that do not pass:", ""]
    msgs = defaultdict(list)
    for key, r in data.items():
        if r["status"] != "pass":
            msgs[r["message"].split(" [")[0][:150]].append(key)
    for msg, keys in sorted(msgs.items(), key=lambda x: -len(x[1])):
        lines.append(
            f"- {msg.replace('|', '/')}: "
            + ", ".join(f"`{k}`" for k in sorted(keys))
        )
    return "\n".join(lines)


def transform_table() -> str:
    data = _load("transforms.json")
    if not data:
        return "–"
    names = list(next(iter(data.values())))
    lines = [
        "| check | " + " | ".join(data) + " |",
        "| --- | " + " | ".join("---" for _ in data) + " |",
    ]
    for n in names:
        lines.append(
            f"| {n} | " + " | ".join(data[k][n]["status"] for k in data) + " |"
        )
    lines += ["", "Messages:", ""]
    msgs = defaultdict(list)
    for combo, checks in data.items():
        for n, r in checks.items():
            if r["status"] != "works":
                msgs[r["message"][:180]].append(f"{combo}: {n}")
    for msg, where in msgs.items():
        lines.append(f"- {msg.replace('|', '/')} ({'; '.join(where)})")
    return "\n".join(lines)


def param_table() -> str:
    data = _load("param_coverage.json")
    if not data:
        return "–"
    lines = []
    for method, leaves in data.items():
        skipped = leaves.pop("__not_checked__", {}).get("count", 0)
        c = Counter(r["verdict"] for r in leaves.values())
        c["not checked (element not in set)"] = skipped
        lines += [
            f"**{method}**: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(c.items())),
            "",
        ]
        bad = {
            k: r
            for k, r in leaves.items()
            if r["verdict"] in ("gradient path cut", "wrong gradient")
        }
        if bad:
            lines += [
                "| leaf | verdict | energy grad | forces grad |",
                "| --- | --- | --- | --- |",
            ]
            for k, r in sorted(bad.items()):
                lines.append(
                    f"| `{k}` | {r['verdict']} | {r['energy_grad']} | {r['forces_grad']} |"
                )
            lines.append("")
        nf = sorted(
            k
            for k, r in leaves.items()
            if r["verdict"] == "no effect on this set"
        )
        if nf:
            lines += [
                "No effect on the set (autograd and finite difference zero): "
                + ", ".join(f"`{k}`" for k in nf),
                "",
            ]
    return "\n".join(lines)


def main() -> None:
    print("<!-- generated by `python -m test.test_baseline.tables` -->\n")
    print("### Derivative status matrix (T0.3)\n")
    print(matrix_tables())
    print("\n#### Single system by setting\n")
    print(matrix_by_setting())
    print("\n#### Cost of the cells\n")
    print(matrix_cost())
    print("\n#### Distinct outcomes of the cells that do not pass\n")
    print(matrix_failures())
    print("\n### Component-level checks (T0.4)\n")
    print(component_table())
    print("\n### Transform status (T0.7)\n")
    print(transform_table())
    print("\n### Parameter-gradient coverage (T0.9)\n")
    print(param_table())


if __name__ == "__main__":
    main()
