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
Component-level derivative checks (T0.4).

One test per target and check (see ``components.py``). Checks that do not
pass today are ``xfail(strict=True)`` with the recorded outcome
(``status/components.json``). Third-order checks are ``slow``. Regenerate
the status with ``python -m test.test_baseline.status components``.
"""

from __future__ import annotations

import json

import pytest

from .components import CHECKS, TARGETS, run_check
from .status import STATUS_DIR, run_isolated

_STATUS_FILE = STATUS_DIR / "components.json"
STATUS: dict[str, dict] = (
    json.loads(_STATUS_FILE.read_text()) if _STATUS_FILE.exists() else {}
)


def _param(target: str, check: str):
    key = f"{target}:{check}"
    marks = []
    if check == "order3" or STATUS.get(key, {}).get("seconds", 0.0) > 30.0:
        marks.append(pytest.mark.slow)
    if any(k in target for k in ("overlap", "dipint", "quadint", "hcore")):
        marks.append(pytest.mark.integrals)

    recorded = STATUS.get(key)
    if recorded is None:
        marks.append(pytest.mark.skip(reason="no recorded status"))
    elif recorded["status"] != "pass":
        reason = f"{recorded['status']}: {recorded['message']}"
        marks.append(pytest.mark.xfail(strict=True, reason=reason))
    return pytest.param(target, check, marks=marks, id=key)


@pytest.mark.parametrize(
    "target,check", [_param(t, c) for t in TARGETS for c in CHECKS]
)
def test_component(target: str, check: str) -> None:
    key = f"{target}:{check}"
    if STATUS.get(key, {}).get("status", "pass") != "pass":
        # may crash or exhaust memory: run with limits in a fresh process
        result = run_isolated("components", key)
        assert result["status"] == "pass", result["message"]
        return

    res = run_check(target, check)
    assert res.status == "pass", f"{res.status}: {res.message}"
