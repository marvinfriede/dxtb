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
"""Compare experimental reuse paths with the invariant Fermi density.

This bounded harness exercises synthetic symmetric Hamiltonians. It leaves
the production SCF and spectral implementation unchanged.

Example
-------
Run with the controlled CPU thread settings::

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
        /path/to/python benchmarks/e6c1_w2c4_spectral_reuse.py \
        --output benchmarks/results/e6c1-w2c4/spectral-py311.jsonl
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from tad_mctc import storch

import dxtb  # noqa: E402
from dxtb._src.scf.functional.spectral import (  # noqa: E402
    _matrix_fermi,
    _prepare_matrix,
    fermi_density_matrix,
)
from dxtb._src.wavefunction import filling  # noqa: E402

REQUIRED_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
DTYPE = torch.float64


def _cpu_model() -> str:
    """Return the first available CPU model description."""
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _version(name: str) -> str:
    """Read an installed dependency version without importing its API."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def emit(output: Path, event: str, **values: Any) -> None:
    """Append one structured diagnostic event and print it immediately."""
    record = {"event": event, "time": time.time(), **values}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    print(json.dumps(record, sort_keys=True, default=str), flush=True)


def _validate_environment(output: Path) -> None:
    """Set and assert the single-thread CPU diagnostic environment."""
    wrong = {
        name: (os.environ.get(name), expected)
        for name, expected in REQUIRED_THREAD_ENV.items()
        if os.environ.get(name) != expected
    }
    if wrong:
        raise RuntimeError(
            f"Thread environment must be explicitly set: {wrong}"
        )

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError(
            "PyTorch thread settings are not the required value 1"
        )

    packages = {
        name: _version(name)
        for name in (
            "tad-mctc",
            "tad-multicharge",
            "tad-libcint",
            "tad-dftd3",
            "tad-dftd4",
        )
    }
    source_path = str(Path(dxtb.__file__).resolve())
    expected_source = str((ROOT / "src" / "dxtb" / "__init__.py").resolve())
    if source_path != expected_source:
        raise RuntimeError(
            f"Imported dxtb source {source_path} != expected {expected_source}"
        )
    emit(
        output,
        "environment",
        python=sys.version.split()[0],
        torch=torch.__version__,
        cpu=_cpu_model(),
        dxtb_source=source_path,
        pytorch_intra_threads=torch.get_num_threads(),
        pytorch_inter_threads=torch.get_num_interop_threads(),
        thread_environment={
            name: os.environ[name] for name in REQUIRED_THREAD_ENV
        },
        dependencies=packages,
    )


def _state_from_energies(
    energies: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    mask: torch.Tensor | None = None,
):
    """Build the shared differentiable scalar filling state for a matrix."""
    return filling._get_fermi_occupation_core_state(
        electrons,
        energies,
        kt,
        thr=1.0e-14,
        maxiter=300,
        diff_order=3,
        mask=mask,
    )


def _accepted_density(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    state: Any,
    l_inv: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate the accepted invariant matrix-Fermi density."""
    return fermi_density_matrix(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=l_inv,
        mask=mask,
        diff_steps=filling._diff_steps(3),
    )


def _eigenbasis_density(coefficients: torch.Tensor, state: Any) -> torch.Tensor:
    """Experimental Candidate A using a retained eigensystem and occupations."""
    occupations = state.result.occupation
    total_occupation = occupations.sum(dim=0)
    return (coefficients * total_occupation.unsqueeze(0)) @ coefficients.T


def _single_scalar_mu_density(
    hamiltonian: torch.Tensor,
    electrons: torch.Tensor,
    kt: torch.Tensor,
    state: Any,
    l_inv: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Experimental Candidate B with one matrix function at scalar mu."""
    _, spectral, active = _prepare_matrix(hamiltonian, l_inv, mask)
    channels = electrons.shape[0]
    spectral = spectral.unsqueeze(0).expand(channels, -1, -1)
    active = active.unsqueeze(0).expand(channels, -1)
    identity = torch.eye(
        hamiltonian.shape[-1],
        dtype=hamiltonian.dtype,
        device=hamiltonian.device,
    )
    mu = (
        state.chemical_potential_ref + state.chemical_potential_offset
    ).squeeze(-1)
    relative = spectral - mu[:, None, None] * identity
    density = _matrix_fermi(relative, torch.zeros_like(electrons), kt, active)
    empty = state.empty
    full = state.full
    density = torch.where(empty[:, None, None], 0.0, density)
    density = torch.where(
        full[:, None, None], torch.diag_embed(active.to(density.dtype)), density
    )
    density = density.sum(dim=0)
    if l_inv is not None:
        density = l_inv.T @ density @ l_inv
    pair = active[0].unsqueeze(-1) & active[0].unsqueeze(-2)
    return torch.where(pair, density, torch.zeros_like(density))


def _rotate(diagonal: torch.Tensor, angle: float) -> torch.Tensor:
    """Rotate a three-level diagonal spectrum in a fixed plane."""
    cosine = diagonal.new_tensor(angle).cos()
    sine = diagonal.new_tensor(angle).sin()
    rotation = torch.stack(
        (
            torch.stack((cosine, -sine, cosine.new_zeros(()))),
            torch.stack((sine, cosine, cosine.new_zeros(()))),
            torch.stack(
                (
                    cosine.new_zeros(()),
                    cosine.new_zeros(()),
                    cosine.new_ones(()),
                )
            ),
        )
    )
    return rotation @ torch.diag(diagonal) @ rotation.T


def _directional_third(
    function: Any, point: torch.Tensor, direction: torch.Tensor
):
    """Compute a scalar third directional derivative by repeated JVP."""
    from torch.func import jvp

    def first(value: torch.Tensor) -> torch.Tensor:
        return jvp(function, (value,), (direction,))[1]

    def second(value: torch.Tensor) -> torch.Tensor:
        return jvp(first, (value,), (direction,))[1]

    return jvp(second, (point,), (direction,))[1]


def _scalar_occupation_oracle(
    energies: torch.Tensor, electrons: torch.Tensor, kt: torch.Tensor
) -> torch.Tensor:
    """Solve each scalar Fermi count by bisection for an independent value oracle."""
    occupations = []
    for channel in range(electrons.numel()):
        spectrum = energies.detach()
        target = float(electrons[channel])
        thermal = float(kt)
        lower = float(spectrum.min()) - 100.0 * thermal
        upper = float(spectrum.max()) + 100.0 * thermal
        for _ in range(160):
            middle = 0.5 * (lower + upper)
            count = torch.sigmoid((middle - spectrum) / thermal).sum().item()
            if count < target:
                lower = middle
            else:
                upper = middle
        chemical_potential = 0.5 * (lower + upper)
        occupations.append(
            torch.sigmoid((chemical_potential - spectrum) / thermal)
        )
    return torch.stack(occupations)


def _run_case(output: Path, name: str, diagonal: list[float]) -> None:
    """Compare accepted, eigensystem and scalar-mu density routes."""
    values = torch.tensor(diagonal, dtype=DTYPE)
    hamiltonian = _rotate(values, 0.37).requires_grad_(True)
    electrons = torch.tensor([1.3, 0.8], dtype=DTYPE)
    kt = torch.tensor(0.2, dtype=DTYPE)
    probe = torch.tensor(
        [[0.2, 0.4, -0.3], [0.4, -0.1, 0.25], [-0.3, 0.25, 0.7]],
        dtype=DTYPE,
    )

    def accepted(value: torch.Tensor) -> torch.Tensor:
        energies = torch.linalg.eigvalsh(value).expand(2, -1)
        current = _state_from_energies(energies, electrons, kt)
        return (_accepted_density(value, electrons, kt, current) * probe).sum()

    def eigenbasis(value: torch.Tensor) -> torch.Tensor:
        energies, coefficients = storch.eighb(
            value,
            broadening_method=None,
            factor=torch.finfo(value.dtype).eps ** 0.5,
            is_posdef=True,
        )
        current = _state_from_energies(energies.expand(2, -1), electrons, kt)
        return (_eigenbasis_density(coefficients, current) * probe).sum()

    def scalar_mu(value: torch.Tensor) -> torch.Tensor:
        energies = torch.linalg.eigvalsh(value).expand(2, -1)
        current = _state_from_energies(energies, electrons, kt)
        return (
            _single_scalar_mu_density(value, electrons, kt, current) * probe
        ).sum()

    started = time.perf_counter()
    reference_energies = torch.linalg.eigvalsh(hamiltonian).expand(2, -1)
    reference_state = _state_from_energies(reference_energies, electrons, kt)
    density_ref = _accepted_density(hamiltonian, electrons, kt, reference_state)
    try:
        energies_a, coefficients_a = storch.eighb(
            hamiltonian,
            broadening_method=None,
            factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
            is_posdef=True,
        )
        state_a = _state_from_energies(energies_a.expand(2, -1), electrons, kt)
        density_a = _eigenbasis_density(coefficients_a, state_a)
        value_a_error = float((density_a - density_ref).abs().max().detach())
    except Exception as exc:  # Diagnostic records candidate breakdowns.
        value_a_error = None
        emit(
            output, "candidate_error", case=name, candidate="A", error=repr(exc)
        )

    try:
        density_b = _single_scalar_mu_density(
            hamiltonian, electrons, kt, reference_state
        )
        value_b_error = float((density_b - density_ref).abs().max().detach())
    except Exception as exc:  # Diagnostic records candidate breakdowns.
        value_b_error = None
        emit(
            output, "candidate_error", case=name, candidate="B", error=repr(exc)
        )

    direction = torch.tensor(
        [[0.2, -0.1, 0.05], [-0.1, 0.15, 0.08], [0.05, 0.08, -0.12]],
        dtype=DTYPE,
    )
    results: dict[str, Any] = {
        "case": name,
        "spectrum": diagonal,
        "kt": float(kt),
        "electrons": electrons.tolist(),
        "accepted_density_trace": float(density_ref.trace().detach()),
        "candidate_a_density_max_abs": value_a_error,
        "candidate_b_density_max_abs": value_b_error,
        "candidate_a_electron_count": (
            float(density_a.trace().detach())
            if value_a_error is not None
            else None
        ),
        "candidate_b_electron_count": (
            float(density_b.trace().detach())
            if value_b_error is not None
            else None
        ),
    }
    eigenvalues = torch.linalg.eigvalsh(hamiltonian.detach())
    scalar_oracle = _scalar_occupation_oracle(eigenvalues, electrons, kt)
    scalar_density = (
        coefficients_a.detach() * scalar_oracle.sum(dim=0).unsqueeze(0)
    ) @ coefficients_a.detach().T
    results["scalar_oracle_density_max_abs"] = float(
        (scalar_density - density_ref.detach()).abs().max()
    )
    results["scalar_oracle_electron_count"] = float(scalar_density.trace())

    def accepted_count_response(counts: torch.Tensor) -> torch.Tensor:
        current = _state_from_energies(
            reference_energies.expand(2, -1), counts, kt
        )
        return _accepted_density(hamiltonian, counts, kt, current).trace()

    def scalar_count_response(counts: torch.Tensor) -> torch.Tensor:
        current = _state_from_energies(
            reference_energies.expand(2, -1), counts, kt
        )
        return _single_scalar_mu_density(
            hamiltonian, counts, kt, current
        ).trace()

    results["accepted_electron_count_jacobian"] = (
        torch.func.jacrev(accepted_count_response)(electrons).detach().tolist()
    )
    results["candidate_b_electron_count_jacobian"] = (
        torch.func.jacrev(scalar_count_response)(electrons).detach().tolist()
    )

    for label, function in (
        ("accepted", accepted),
        ("candidate_a", eigenbasis),
        ("candidate_b", scalar_mu),
    ):
        try:
            grad = torch.autograd.grad(
                function(hamiltonian), hamiltonian, create_graph=True
            )[0]
            second = torch.autograd.grad(
                (grad * direction).sum(), hamiltonian, create_graph=True
            )[0]
            third = torch.autograd.grad(
                (second * direction).sum(), hamiltonian
            )[0]
            first_directional = torch.func.jvp(
                function, (hamiltonian.detach(),), (direction,)
            )[1]
            second_directional = torch.func.jvp(
                lambda value: torch.func.jvp(function, (value,), (direction,))[
                    1
                ],
                (hamiltonian.detach(),),
                (direction,),
            )[1]
            third_directional = _directional_third(
                function, hamiltonian.detach(), direction
            )
            results[f"{label}_grad_finite"] = bool(torch.isfinite(grad).all())
            results[f"{label}_second_finite"] = bool(
                torch.isfinite(second).all()
            )
            results[f"{label}_third_finite"] = bool(torch.isfinite(third).all())
            results[f"{label}_grad_norm"] = float(grad.norm().detach())
            results[f"{label}_third_norm"] = float(third.norm().detach())
            results[f"{label}_first_directional"] = float(
                first_directional.detach()
            )
            results[f"{label}_second_directional"] = float(
                second_directional.detach()
            )
            results[f"{label}_third_directional"] = float(
                third_directional.detach()
            )
            if label != "accepted":
                results[f"{label}_grad_max_abs_diff"] = float(
                    (grad - accepted_grad).abs().max().detach()
                )
                results[f"{label}_second_max_abs_diff"] = float(
                    (second - accepted_second).abs().max().detach()
                )
                results[f"{label}_third_max_abs_diff"] = float(
                    (third - accepted_third).abs().max().detach()
                )
        except Exception as exc:  # Degenerate eigensolvers may reject backward.
            results[f"{label}_derivative_error"] = repr(exc)
        if label == "accepted":
            accepted_grad = grad.detach()
            accepted_second = second.detach()
            accepted_third = third.detach()

    results["elapsed_s"] = time.perf_counter() - started
    emit(output, "spectral_candidate", **results)


def _run_value_fixture(
    output: Path,
    name: str,
    hamiltonian: torch.Tensor,
    overlap: torch.Tensor | None = None,
    structural_mask: torch.Tensor | None = None,
) -> None:
    """Check density values for generalized or structurally padded inputs."""
    l_inv = (
        None
        if overlap is None
        else storch.inv_cholesky_factor(overlap, is_posdef=True)
    )
    energies, coefficients = storch.eighb(
        hamiltonian,
        l_inv=l_inv,
        broadening_method=None,
        factor=torch.finfo(hamiltonian.dtype).eps ** 0.5,
        is_posdef=True,
    )
    active_count = (
        hamiltonian.shape[-1]
        if structural_mask is None
        else int(structural_mask.sum())
    )
    spectral_mask = torch.arange(energies.numel()) < active_count
    channel_mask = spectral_mask.unsqueeze(0).expand(2, -1)
    electrons = hamiltonian.new_tensor([1.2, 0.7])
    kt = hamiltonian.new_tensor(0.25)
    state = _state_from_energies(
        energies.expand(2, -1), electrons, kt, channel_mask
    )
    accepted = _accepted_density(
        hamiltonian,
        electrons,
        kt,
        state,
        l_inv=l_inv,
        mask=structural_mask,
    )
    eigenbasis = _eigenbasis_density(coefficients, state)
    scalar_occupation = _scalar_occupation_oracle(energies, electrons, kt)
    scalar_density = (
        coefficients * scalar_occupation.sum(0).unsqueeze(0)
    ) @ coefficients.T
    if structural_mask is not None:
        pair = structural_mask.unsqueeze(-1) & structural_mask.unsqueeze(-2)
        scalar_density = torch.where(pair, scalar_density, 0.0)

    emit(
        output,
        "value_fixture",
        case=name,
        energies=energies.detach().tolist(),
        active_count=active_count,
        accepted_trace=float(accepted.trace().detach()),
        accepted_electron_count=float(
            torch.trace(
                accepted if overlap is None else accepted @ overlap
            ).detach()
        ),
        eigenbasis_trace=float(eigenbasis.trace().detach()),
        scalar_oracle_trace=float(scalar_density.trace().detach()),
        eigenbasis_max_abs=float((eigenbasis - accepted).abs().max().detach()),
        scalar_oracle_max_abs=float(
            (scalar_density - accepted).abs().max().detach()
        ),
        density_symmetry_error=float(
            (eigenbasis - eigenbasis.T).abs().max().detach()
        ),
        overlap=overlap is not None,
        noncontiguous_mask=(
            None if structural_mask is None else structural_mask.tolist()
        ),
    )


def main() -> None:
    """Run a bounded synthetic candidate comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.unlink()
    _validate_environment(args.output)
    _run_case(args.output, "rotated_non_degenerate", [-0.5, 0.6, 4.0])
    _run_case(
        args.output, "rotated_near_degenerate", [-0.3, -0.3 + 1.0e-10, 0.5]
    )
    _run_case(args.output, "rotated_exact_degenerate", [-0.3, -0.3, 0.5])
    _run_value_fixture(
        args.output,
        "generalized_nonidentity_overlap",
        torch.tensor(
            [[-0.4, 0.12, 0.03], [0.12, 0.2, -0.07], [0.03, -0.07, 0.8]],
            dtype=DTYPE,
        ),
        overlap=torch.tensor(
            [[1.2, 0.08, 0.03], [0.08, 0.9, 0.06], [0.03, 0.06, 1.1]],
            dtype=DTYPE,
        ),
    )
    _run_value_fixture(
        args.output,
        "noncontiguous_ao_padding",
        torch.tensor(
            [[-0.4, 0.0, 0.12], [0.0, 0.0, 0.0], [0.12, 0.0, 0.8]],
            dtype=DTYPE,
        ),
        overlap=torch.tensor(
            [[1.2, 0.0, 0.08], [0.0, 1.0, 0.0], [0.08, 0.0, 1.1]],
            dtype=DTYPE,
        ),
        structural_mask=torch.tensor([True, False, True]),
    )


if __name__ == "__main__":
    main()
