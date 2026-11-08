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
Labels: SCF
===========

Labels for SCF-related options.
"""

__all__ = [
    "GUESS_EEQ",
    "GUESS_EEQ_STRS",
    "GUESS_SAD",
    "GUESS_SAD_STRS",
    "GUESS_MAP",
    "FERMI_PARTITION_EQUAL",
    "FERMI_PARTITION_EQUAL_STRS",
    "FERMI_PARTITION_ATOMIC",
    "FERMI_PARTITION_ATOMIC_STRS",
    "FERMI_PARTITION_MAP",
]

# guess
GUESS_EEQ = 0
"""Integer code for EEQ guess."""
GUESS_EEQ_STRS = ("eeq", "equilibration")
"""String codes for EEQ guess."""

GUESS_SAD = 1
"""Integer code for SAD (superposition of atomic density; zero) guess."""
GUESS_SAD_STRS = ("sad", "zero")
"""String codes for SAD (superposition of atomic density; zero) guess."""

GUESS_MAP = ["EEQ", "SAD"]
"""String map (for printing) of SCF guess methods."""

# fermi partition
FERMI_PARTITION_EQUAL = 0
"""Integer code for equal Fermi partition."""

FERMI_PARTITION_EQUAL_STRS = ("equal", "same")
"""String codes for equal Fermi partition."""

FERMI_PARTITION_ATOMIC = 1
"""Integer code for atomic Fermi partition."""

FERMI_PARTITION_ATOMIC_STRS = ("atom", "atomic")
"""String codes for atomic Fermi partition."""

FERMI_PARTITION_MAP = ["equal", "atom"]
"""String map (for printing) of Fermi partitioning methods."""
