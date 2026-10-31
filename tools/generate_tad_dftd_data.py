#!/usr/bin/env python3
# This file is part of tad-dftd.
# SPDX-Identifier: Apache-2.0
"""
Development-time data extraction script for `tad_dftd`.

Extracts pairwise weighting factor tables (`wfpair-d3s.pt`, `wfpair-d4s.pt`)
from the authoritative Grimme-lab Fortran `dftd` repository:
- Repository: https://github.com/grimme-lab/dftd
- Commit: d2740777049b2a77fd5acc86c4e1abe75d5374ee
- Upstream license: LGPL-3.0-or-later
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

DEFAULT_DFTD_ROOT = Path("/tmp1/friede/work/dftd-grimme-lab/dftd")
REPO_ROOT = Path(__file__).resolve().parent.parent


def _parse_fortran_column_data(inc_text: str, max_elem: int) -> torch.Tensor:
    """
    Parse Fortran `data weight_factors(:, col) / ... /` statements into a
    `(max_elem + 1, max_elem + 1)` float64 tensor (0-padded at index 0 for
    dummy/padding atoms).
    """
    table = torch.zeros((max_elem + 1, max_elem + 1), dtype=torch.float64)
    pattern = re.compile(
        r"data\s+weight_factors\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/(.*?)/",
        re.DOTALL,
    )
    found_cols = set()
    for match in pattern.finditer(inc_text):
        col = int(match.group(1))
        body = match.group(2)
        vals = [
            float(v)
            for v in re.findall(r"([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp", body)
        ]
        if len(vals) != max_elem:
            raise ValueError(
                f"Expected {max_elem} values in column {col}, got {len(vals)}"
            )
        table[1 : max_elem + 1, col] = torch.tensor(vals, dtype=torch.float64)
        found_cols.add(col)

    if len(found_cols) != max_elem:
        raise ValueError(
            f"Expected {max_elem} columns, found {len(found_cols)}"
        )
    return table


def generate_wfpair_tables(dftd_root: Path, data_dir: Path) -> None:
    """Generate `wfpair-d3s.pt` and `wfpair-d4s.pt`."""
    d3s_inc = (dftd_root / "src/dftd/data/wfpair/d3s.inc").read_text()
    d3s_table = _parse_fortran_column_data(d3s_inc, max_elem=94)
    torch.save(d3s_table, data_dir / "wfpair-d3s.pt")

    d4s_inc = (dftd_root / "src/dftd/data/wfpair/d4s.inc").read_text()
    d4s_table = _parse_fortran_column_data(d4s_inc, max_elem=118)
    torch.save(d4s_table, data_dir / "wfpair-d4s.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dftd-root",
        type=Path,
        default=DEFAULT_DFTD_ROOT,
        help="Path to the authoritative Fortran dftd checkout.",
    )
    args = parser.parse_args()

    data_dir = REPO_ROOT / "src/tad_dftd/data"
    data_dir.mkdir(parents=True, exist_ok=True)
    generate_wfpair_tables(args.dftd_root, data_dir)
    print(f"Generated wfpair tables in {data_dir}")


if __name__ == "__main__":
    main()
