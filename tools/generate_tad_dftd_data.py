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


def generate_d3s_legacy_c6(dftd_root: Path, ref_dir: Path) -> None:
    """
    Generate `d3s-legacy-c6.pt` of shape `(8, 95, 3, 7)` for Z=87..94 paired
    with Z=0..94 from `dftd/src/dftd/model/d3/reference/smooth.inc`.
    """
    smooth_inc = (
        dftd_root / "src/dftd/model/d3/reference/smooth.inc"
    ).read_text()

    # Number of references for Z=1..94 in D3S
    nref_1_to_86 = [
        2, 1,
        2, 3, 5, 5, 4, 3, 2, 1,
        2, 3, 4, 5, 4, 3, 2, 1,
        2, 3,
        3, 3, 3, 3, 3, 3, 4, 4, 2, 2,
        4, 5, 4, 3, 2, 1,
        2, 3,
        3, 3, 3, 3, 3, 3, 3, 3, 2, 2,
        4, 5, 4, 3, 2, 1,
        2, 3,
        3, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 3, 3, 3, 3, 3, 3, 3, 2, 2,
        4, 5, 4, 3, 2, 1,
    ]
    nref = [0] + nref_1_to_86 + [2, 3, 2, 2, 2, 2, 2, 2]
    assert len(nref) == 95

    legacy_c6 = torch.zeros((8, 95, 3, 7), dtype=torch.float64)
    for idx_z, iz in enumerate(range(87, 95)):
        pattern = re.compile(
            rf"legacy_c6_0{iz}\s*\(\s*\d+\s*\)\s*=\s*\[(.*?)\]",
            re.DOTALL,
        )
        match = pattern.search(smooth_inc)
        if match is None:
            raise ValueError(f"Could not find legacy_c6_0{iz} in smooth.inc")
        vals = [
            float(v)
            for v in re.findall(
                r"([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp", match.group(1)
            )
        ]
        expected_len = sum(nref[1 : iz + 1]) * nref[iz]
        if len(vals) != expected_len:
            raise ValueError(
                f"Expected {expected_len} values for Z={iz}, got {len(vals)}"
            )
        prefix_sum = 0
        for jz in range(1, iz + 1):
            for ir in range(nref[iz]):
                for jr in range(nref[jz]):
                    flat_idx = prefix_sum * nref[iz] + ir * nref[jz] + jr
                    legacy_c6[idx_z, jz, ir, jr] = vals[flat_idx]
            prefix_sum += nref[jz]

    torch.save(legacy_c6, ref_dir / "d3s-legacy-c6.pt")


def _fortran_nint(x: float) -> int:
    """Fortran `nint`: round half away from zero."""
    import math

    return int(math.copysign(math.floor(abs(x) + 0.5), x))


def _parse_d4_inc(inc_text: str, *, include_gxtb: bool) -> dict[str, torch.Tensor]:
    """
    Parse `dftd/src/dftd/model/d4/reference/{orig,srev}.inc` into a dictionary
    of tensors for `d4-ref.pt` / `d4srev-ref.pt`.
    """
    max_elem = 118
    mref = 7
    ngrid = 23
    nsec = 17
    max_cn = 19

    sscale = torch.zeros((nsec + 1,), dtype=torch.float64)
    secaiw = torch.zeros((nsec + 1, ngrid), dtype=torch.float64)

    for m in re.finditer(
        r"data\s+sscale\s*\(\s*(\d+)\s*\)\s*/\s*([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp\s*/",
        inc_text,
    ):
        sscale[int(m.group(1))] = float(m.group(2))

    for m in re.finditer(
        r"data\s+secaiw\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/(.*?)/",
        inc_text,
        re.DOTALL,
    ):
        isec = int(m.group(1))
        vals = [
            float(v)
            for v in re.findall(
                r"([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp", m.group(2)
            )
        ]
        if len(vals) != ngrid:
            raise ValueError(f"Expected {ngrid} secaiw values for {isec}")
        secaiw[isec] = torch.tensor(vals, dtype=torch.float64)

    refn = torch.zeros((max_elem + 1,), dtype=torch.long)
    for m in re.finditer(
        r"data\s+refn\s*\(\s*(\d+)\s*\)\s*/\s*(\d+)\s*/",
        inc_text,
    ):
        refn[int(m.group(1))] = int(m.group(2))

    refsys = torch.zeros((max_elem + 1, mref), dtype=torch.long)
    for m in re.finditer(
        r"data\s+refsys\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*(\d+)\s*/",
        inc_text,
    ):
        ir, iz = int(m.group(1)) - 1, int(m.group(2))
        refsys[iz, ir] = int(m.group(3))

    scalar_2d_names = [
        "refcn",
        "refcovcn",
        "ascale",
        "hcount",
        "clsq",
        "clsh",
        "eeqbcq",
        "eeqbch",
        "refq",
        "refh",
    ]
    if include_gxtb:
        scalar_2d_names.extend(["gxtbq", "gxtbh"])

    arrays_2d: dict[str, torch.Tensor] = {
        name: torch.zeros((max_elem + 1, mref), dtype=torch.float64)
        for name in scalar_2d_names
    }
    for name in scalar_2d_names:
        pattern = re.compile(
            rf"data\s+{name}\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp\s*/"
        )
        for m in pattern.finditer(inc_text):
            ir, iz = int(m.group(1)) - 1, int(m.group(2))
            arrays_2d[name][iz, ir] = float(m.group(3))

    alphaiw = torch.zeros((max_elem + 1, mref, ngrid), dtype=torch.float64)
    for m in re.finditer(
        r"data\s+alphaiw\s*\(\s*:\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*/(.*?)/",
        inc_text,
        re.DOTALL,
    ):
        ir, iz = int(m.group(1)) - 1, int(m.group(2))
        vals = [
            float(v)
            for v in re.findall(
                r"([+-]?\d+(?:\.\d+)?(?:[eEdD][+-]?\d+)?)_wp", m.group(3)
            )
        ]
        if len(vals) != ngrid:
            raise ValueError(
                f"Expected {ngrid} alphaiw values for Z={iz}, ir={ir + 1}"
            )
        alphaiw[iz, ir] = torch.tensor(vals, dtype=torch.float64)

    # Precompute `ngw` using Fortran `set_refgw` logic
    ngw = torch.zeros((max_elem + 1, mref), dtype=torch.long)
    refcn = arrays_2d["refcn"]
    for iz in range(1, max_elem + 1):
        n = int(refn[iz].item())
        cnc = [1] + [0] * max_cn
        for ir in range(n):
            icn = min(_fortran_nint(float(refcn[iz, ir].item())), max_cn)
            cnc[icn] += 1
        for ir in range(n):
            icn = min(_fortran_nint(float(refcn[iz, ir].item())), max_cn)
            ngw[iz, ir] = cnc[icn] * (cnc[icn] + 1) // 2

    out: dict[str, torch.Tensor] = {
        "refn": refn,
        "ngw": ngw,
        "refsys": refsys,
        "sscale": sscale,
        "secaiw": secaiw,
        "alphaiw": alphaiw,
        **arrays_2d,
    }
    return out


def generate_d4_references(dftd_root: Path, ref_dir: Path) -> None:
    """Generate `d4-ref.pt` and `d4srev-ref.pt`."""
    orig_inc = (dftd_root / "src/dftd/model/d4/reference/orig.inc").read_text()
    d4_data = _parse_d4_inc(orig_inc, include_gxtb=False)
    torch.save(d4_data, ref_dir / "d4-ref.pt")

    srev_inc = (dftd_root / "src/dftd/model/d4/reference/srev.inc").read_text()
    d4srev_data = _parse_d4_inc(srev_inc, include_gxtb=True)
    torch.save(d4srev_data, ref_dir / "d4srev-ref.pt")


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

    ref_dir = REPO_ROOT / "src/tad_dftd/reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    generate_d3s_legacy_c6(args.dftd_root, ref_dir)
    generate_d4_references(args.dftd_root, ref_dir)
    print(f"Generated wfpair and reference tables in {data_dir} and {ref_dir}")


if __name__ == "__main__":
    main()
