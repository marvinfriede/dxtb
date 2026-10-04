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
Test-suite inventory (T0.10).

Run from the repository root after a full run with a JUnit report::

    pytest test -m "not large" --junitxml=junit.xml ...
    python -m test.test_baseline.inventory junit.xml

Prints the number of tests and the runtime per test file, the layer
markers of each file (see ``LAYER_MARKERS`` in ``test/conftest.py``) and
the tests that failed or errored (flaky candidates when they pass on rerun).
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).parents[2]


def main(path: str) -> None:
    sys.path.insert(0, str(ROOT))
    # pylint: disable=import-outside-toplevel
    from test.conftest import KEYWORD_MARKERS, LAYER_MARKERS

    tree = ET.parse(path)
    per_file: dict[str, list[float]] = defaultdict(list)
    bad = []
    for case in tree.iter("testcase"):
        cls = case.get("classname", "")
        parts = cls.split(".")
        if parts and parts[-1][:1].isupper():  # test class inside the module
            parts = parts[:-1]
        file = "/".join(parts) + ".py"
        if not file.startswith("test/"):
            file = "test/" + file
        per_file[file].append(float(case.get("time", 0.0)))
        for tag in ("failure", "error"):
            if case.find(tag) is not None:
                bad.append(f"{cls}::{case.get('name')} ({tag})")

    def markers(file: str) -> str:
        rel = file[len("test/") :] if file.startswith("test/") else file
        best = ""
        for prefix in LAYER_MARKERS:
            if (rel == prefix or rel.startswith(prefix + "/")) and len(
                prefix
            ) > len(best):
                best = prefix
        names = set(LAYER_MARKERS.get(best, ()))
        names |= {m for k, m in KEYWORD_MARKERS.items() if k in rel.lower()}
        return ", ".join(sorted(names)) or "-"

    total = sum(sum(v) for v in per_file.values())
    print(f"{sum(len(v) for v in per_file.values())} tests, {total:.0f} s\n")
    print("| file | tests | time (s) | markers |")
    print("| --- | --- | --- | --- |")
    for file, times in sorted(per_file.items(), key=lambda x: -sum(x[1])):
        print(
            f"| `{file}` | {len(times)} | {sum(times):.1f} | {markers(file)} |"
        )
    if bad:
        print("\nFailed or errored:\n")
        for b in bad:
            print(f"- `{b}`")


if __name__ == "__main__":
    main(sys.argv[1])
