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


def compute_fortran_periodic_fixtures(dftd_root: Path) -> dict[str, dict[str, dict[str, object]]]:
    """
    Compute the 21 periodic reference fixtures (`1d`, `2d`, `3d` x `d1`, `d2`,
    `d3`, `d3s`, `d4`, `d4s`, `d4srev`) directly from the authoritative Fortran
    `libdftd.so` C-API (`d2740777049b2a77fd5acc86c4e1abe75d5374ee`).

    Specification
    -------------
    - Repository & Commit: https://github.com/grimme-lab/dftd @ d2740777049b2a77fd5acc86c4e1abe75d5374ee
    - Shared Library & C-API (`src/dftd/api.f90`): `<dftd_root>/_build/libdftd.so`
      (`dftd_new_error`, `dftd_new_structure`, `dftd_new_model`,
      `dftd_set_model_realspace_cutoff`, `dftd_new_damping`,
      `dftd_new_empty_param`, `dftd_set_param`, `dftd_get_dispersion`)
    - Units: Atomic units (positions/lattice/cutoffs in Bohr, energy/sigma in Eh,
      gradient in Eh/Bohr)
    - Precision: IEEE-754 binary64 (`ctypes.c_double`, 17 decimal digits)
    - Geometry: `natoms = 2`, `numbers = [6, 6]`,
      `positions = [[0.15, 0.25, 0.35], [1.95, 2.15, 2.35]]` Bohr
    - Lattice (row vectors `a_1, a_2, a_3` in Bohr):
      `[[6.2, 0.3, 0.1], [0.3, 6.4, 0.2], [0.1, 0.2, 6.6]]`
      (fractional coordinates lie in `(0.01, 0.35) subset [0, 1)`, so
      `wrap_to_central_cell` in `new_structure_api` is an identity map)
    - Periodicity flags (`c_bool[3]`):
      `1d = [True, False, False]`, `2d = [True, True, False]`,
      `3d = [True, True, True]`
    - Charges: `charge = 0.0`, `qmodel = 1` (EEQ, `ga=3.0, gc=2.0, wf=6.0`);
      by inversion symmetry between the two identical carbon atoms,
      `q = [0.0, 0.0]` identically in 1D, 2D, and 3D
    - Damping & ATM (ATM enabled with `s9 = 1.0` for all 7 models):
      - `d1`: Fermi 2B (`damping_2b_id=8`) + `zero_avg` 3B (`damping_3b_id=4`),
        `s6=0.75, rs6=1.0, alp=23.0, s9=1.0, rs9=1.0, alp3=16.0, a1=0.4, a2=4.0, a13=0.4, a23=4.0`
      - `d2`: Fermi 2B (`damping_2b_id=8`) + `zero_avg` 3B (`damping_3b_id=4`),
        `s6=0.75, rs6=1.0, alp=20.0, s9=1.0, rs9=1.0, alp3=16.0, a1=0.4, a2=4.0, a13=0.4, a23=4.0`
      - `d3`, `d3s`: Rational (BJ) 2B (`damping_2b_id=1`) + `zero` 3B (`damping_3b_id=3`),
        PBE parameters `s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407, s9=1.0, rs9=4/3, alp=14.0, alp3=14.0`
      - `d4`, `d4s`, `d4srev`: Rational (BJ) 2B (`damping_2b_id=1`) + `zero_avg` 3B (`damping_3b_id=4`),
        PBE parameters `s6=1.0, s8=0.95948085, a1=0.38574991, a2=4.80688534, s9=1.0, rs9=1.0, alp=16.0, alp3=16.0`
    - Cutoffs (`dftd_set_model_realspace_cutoff`, hard cutoffs `width2=width3=0.0`):
      - `d1`, `d2`, `d3`, `d3s`: `disp2 = 14.0, disp3 = 10.0, cn = 40.0` Bohr
      - `d4`, `d4s`, `d4srev`: `disp2 = 14.0, disp3 = 10.0, cn = 30.0` Bohr (`cn_eeq = 25.0` Bohr)
    - Strain tensor definition:
      `sigma_{alpha,beta} = dE/d(eps_{alpha,beta}) = sum_i r_{i,alpha} dE/dr_{i,beta} + sum_k L_{k,alpha} dE/dL_{k,beta}`
    """
    import ctypes

    lib_path = dftd_root / "_build/libdftd.so"
    if not lib_path.exists():
        raise FileNotFoundError(f"Fortran shared library not found at {lib_path}")
    lib = ctypes.CDLL(str(lib_path))

    c_vp = ctypes.c_void_p
    c_int = ctypes.c_int
    c_bool = ctypes.c_bool
    c_dbl = ctypes.c_double
    c_char_p = ctypes.c_char_p
    c_int_p = ctypes.POINTER(c_int)
    c_bool_p = ctypes.POINTER(c_bool)
    c_dbl_p = ctypes.POINTER(c_dbl)

    lib.dftd_new_error.restype = c_vp
    lib.dftd_delete_error.argtypes = [ctypes.POINTER(c_vp)]
    lib.dftd_new_structure.argtypes = [c_vp, c_int, c_int_p, c_dbl_p, c_dbl_p, c_dbl_p, c_bool_p]
    lib.dftd_new_structure.restype = c_vp
    lib.dftd_delete_structure.argtypes = [ctypes.POINTER(c_vp)]

    lib.dftd_new_model.argtypes = [c_vp, c_vp, c_char_p]
    lib.dftd_new_model.restype = c_vp
    lib.dftd_set_model_realspace_cutoff.argtypes = [c_vp, c_vp, c_dbl, c_dbl, c_dbl]
    lib.dftd_delete_model.argtypes = [ctypes.POINTER(c_vp)]

    lib.dftd_new_damping.argtypes = [c_vp, c_int, c_int]
    lib.dftd_new_damping.restype = c_vp
    lib.dftd_delete_damping.argtypes = [ctypes.POINTER(c_vp)]

    lib.dftd_new_empty_param.argtypes = []
    lib.dftd_new_empty_param.restype = c_vp
    lib.dftd_set_param.argtypes = [c_vp, c_vp, c_char_p, c_dbl]
    lib.dftd_delete_param.argtypes = [ctypes.POINTER(c_vp)]

    lib.dftd_get_dispersion.argtypes = [c_vp, c_vp, c_vp, c_vp, c_vp, c_dbl_p, c_dbl_p, c_dbl_p]

    nums = (c_int * 2)(6, 6)
    pos = (c_dbl * 6)(0.15, 0.25, 0.35, 1.95, 2.15, 2.35)
    lat = (c_dbl * 9)(6.2, 0.3, 0.1, 0.3, 6.4, 0.2, 0.1, 0.2, 6.6)
    chrg = c_dbl(0.0)

    pbc_map = {
        "1d": (c_bool * 3)(True, False, False),
        "2d": (c_bool * 3)(True, True, False),
        "3d": (c_bool * 3)(True, True, True),
    }

    model_specs: dict[str, tuple[int, int, float, dict[str, float]]] = {
        "d1": (
            8,
            4,
            40.0,
            {
                "s6": 0.75,
                "rs6": 1.0,
                "alp": 23.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp3": 16.0,
                "a1": 0.4,
                "a2": 4.0,
                "a13": 0.4,
                "a23": 4.0,
            },
        ),
        "d2": (
            8,
            4,
            40.0,
            {
                "s6": 0.75,
                "rs6": 1.0,
                "alp": 20.0,
                "s9": 1.0,
                "rs9": 1.0,
                "alp3": 16.0,
                "a1": 0.4,
                "a2": 4.0,
                "a13": 0.4,
                "a23": 4.0,
            },
        ),
        "d3": (
            1,
            4,
            40.0,
            {
                "s6": 1.0,
                "s8": 0.7875,
                "a1": 0.4289,
                "a2": 4.4407,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "alp3": 14.0,
                "a13": 1.0,
                "a23": 0.0,
            },
        ),
        "d3s": (
            1,
            4,
            40.0,
            {
                "s6": 1.0,
                "s8": 0.7875,
                "a1": 0.4289,
                "a2": 4.4407,
                "s9": 1.0,
                "rs9": 4.0 / 3.0,
                "alp": 14.0,
                "alp3": 14.0,
                "a13": 1.0,
                "a23": 0.0,
            },
        ),
        "d4": (
            1,
            4,
            30.0,
            {
                "s6": 1.0,
                "s8": 0.95948085,
                "a1": 0.38574991,
                "a2": 4.80688534,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "alp3": 16.0,
                "a13": 0.38574991,
                "a23": 4.80688534,
            },
        ),
        "d4s": (
            1,
            4,
            30.0,
            {
                "s6": 1.0,
                "s8": 0.95948085,
                "a1": 0.38574991,
                "a2": 4.80688534,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "alp3": 16.0,
                "a13": 0.38574991,
                "a23": 4.80688534,
            },
        ),
        "d4srev": (
            1,
            4,
            30.0,
            {
                "s6": 1.0,
                "s8": 0.95948085,
                "a1": 0.38574991,
                "a2": 4.80688534,
                "s9": 1.0,
                "rs9": 1.0,
                "alp": 16.0,
                "alp3": 16.0,
                "a13": 0.38574991,
                "a23": 4.80688534,
            },
        ),
    }

    out: dict[str, dict[str, dict[str, object]]] = {}
    for pbc_key, pbc_arr in pbc_map.items():
        out[pbc_key] = {}
        err = c_vp(lib.dftd_new_error())
        mol = c_vp(
            lib.dftd_new_structure(
                err, 2, nums, pos, ctypes.byref(chrg), lat, pbc_arr
            )
        )
        for m, (damp_2b, damp_3b, cn_cut, p_dict) in model_specs.items():
            mod = c_vp(lib.dftd_new_model(err, mol, m.encode("ascii")))
            lib.dftd_set_model_realspace_cutoff(err, mod, 14.0, 10.0, cn_cut)
            damp = c_vp(lib.dftd_new_damping(err, damp_2b, damp_3b))
            par = c_vp(lib.dftd_new_empty_param())
            for k, v in p_dict.items():
                lib.dftd_set_param(err, par, k.encode("ascii"), float(v))

            energy = c_dbl(0.0)
            grad = (c_dbl * 6)()
            sigma = (c_dbl * 9)()
            lib.dftd_get_dispersion(
                err, mol, mod, damp, par, ctypes.byref(energy), grad, sigma
            )
            out[pbc_key][m] = {
                "energy": float(energy.value),
                "grad0": [float(grad[0]), float(grad[1]), float(grad[2])],
                "sigma": [
                    [float(sigma[0]), float(sigma[1]), float(sigma[2])],
                    [float(sigma[3]), float(sigma[4]), float(sigma[5])],
                    [float(sigma[6]), float(sigma[7]), float(sigma[8])],
                ],
            }
            lib.dftd_delete_param(ctypes.byref(par))
            lib.dftd_delete_damping(ctypes.byref(damp))
            lib.dftd_delete_model(ctypes.byref(mod))
        lib.dftd_delete_structure(ctypes.byref(mol))
        lib.dftd_delete_error(ctypes.byref(err))

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dftd-root",
        type=Path,
        default=DEFAULT_DFTD_ROOT,
        help="Path to the authoritative Fortran dftd checkout.",
    )
    parser.add_argument(
        "--verify-periodic-fixtures",
        action="store_true",
        help="Verify _FORTRAN_PERIODIC_REFS against Fortran libdftd.so without overwriting assets.",
    )
    args = parser.parse_args()

    if args.verify_periodic_fixtures:
        import sys

        sys.path.insert(0, str(REPO_ROOT))
        from test.test_tad_dftd.test_periodic_and_transforms import (
            _FORTRAN_PERIODIC_REFS,
        )

        computed = compute_fortran_periodic_fixtures(args.dftd_root)
        max_diff = 0.0
        for pbc_key, models_dict in _FORTRAN_PERIODIC_REFS.items():
            for m, ref in models_dict.items():
                got = computed[pbc_key][m]
                e_diff = abs(float(ref["energy"]) - float(got["energy"]))  # type: ignore[arg-type]
                g_diff = max(
                    abs(a - b)
                    for a, b in zip(ref["grad0"], got["grad0"])  # type: ignore[arg-type]
                )
                s_diff = max(
                    abs(a - b)
                    for row_r, row_g in zip(ref["sigma"], got["sigma"])  # type: ignore[arg-type]
                    for a, b in zip(row_r, row_g)
                )
                max_diff = max(max_diff, e_diff, g_diff, s_diff)
        print(
            f"Verified all 21 _FORTRAN_PERIODIC_REFS fixtures against "
            f"{args.dftd_root / '_build/libdftd.so'}: max abs diff = {max_diff:.3e}"
        )
        return

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
