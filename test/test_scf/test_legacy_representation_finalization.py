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
"""Legacy convergence-representation finalization characterization."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from tad_mctc.batch import real_atoms

from dxtb import GFN1_XTB, GFN2_XTB, labels
from dxtb._src.components.interactions.container import Potential
from dxtb._src.scf.pure import conversions
from dxtb._src.scf.utils import get_density
from dxtb._src.wavefunction import filling

from .test_functional_map import _prepare_case


@pytest.mark.parametrize(
    "scp_mode",
    [labels.SCP_MODE_CHARGE, labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK],
    ids=["charge", "potential", "fock"],
)
@pytest.mark.parametrize("maxiter", [0, 1, 2])
@pytest.mark.parametrize("mix_guess", [False, True])
def test_legacy_representation_map_and_finalization_counts(
    scp_mode: int, maxiter: int, mix_guess: bool
) -> None:
    """Record raw maps, transitions and final-conversion eigensolves."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2O",
        scf_options={
            "scp_mode": scp_mode,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": maxiter,
            "mix_guess": mix_guess,
            "x_atol": 0.0,
            "x_atol_max": 0.0,
        },
        setup_functional=False,
    )
    engine = case.engine
    method_name = {
        labels.SCP_MODE_CHARGE: "iterate_charges",
        labels.SCP_MODE_POTENTIAL: "iterate_potential",
        labels.SCP_MODE_FOCK: "iterate_fockian",
    }[scp_mode]
    raw_calls: list[dict[str, torch.Tensor]] = []
    diagonalized: list[torch.Tensor] = []
    spectra: list[tuple[torch.Tensor, torch.Tensor]] = []
    transitions: list[torch.Tensor] = []
    raw_map = getattr(engine, method_name)
    original_diagonalize = engine.diagonalize
    original_mix = engine.mixer.iter

    def record_map(value: torch.Tensor) -> torch.Tensor:
        mapped = raw_map(value)
        raw_charges = conversions.density_to_charges(
            engine._data.density, engine._data, engine.config
        )
        raw_calls.append(
            {
                "input": value.clone(),
                "mapped": mapped.clone(),
                "charges": raw_charges.mono.clone(),
                "density": engine._data.density.clone(),
                "hamiltonian": engine._data.hamiltonian.clone(),
                "emo": engine._data.evals.clone(),
                "coefficients": engine._data.evecs.clone(),
                "occupation": engine._data.occupation.clone(),
                "orbital_energy": engine._data.energy.clone(),
            }
        )
        return mapped

    def record_diagonalize(hamiltonian: torch.Tensor) -> Any:
        diagonalized.append(hamiltonian.clone())
        result = original_diagonalize(hamiltonian)
        spectra.append((result[0].clone(), result[1].clone()))
        return result

    def record_mix(mapped: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
        result = original_mix(mapped, old)
        transitions.append(engine.mixer._delta.clone())
        return result

    setattr(engine, method_name, record_map)
    engine.diagonalize = record_diagonalize
    engine.mixer.iter = record_mix
    engine._guess = lambda _charges: case.x.clone()
    result = engine(case.x.clone())
    final_charges = result["charges"]
    final_potential = result["potential"]

    expected_maps = maxiter + 1
    expected_transitions = 0 if maxiter == 0 else maxiter + int(mix_guess)
    expected_final_eigensolves = int(scp_mode != labels.SCP_MODE_CHARGE)
    assert len(raw_calls) == expected_maps
    assert engine._data.iter == expected_maps
    assert len(transitions) == expected_transitions
    assert len(diagonalized) == expected_maps + expected_final_eigensolves
    assert result["iterations"] == expected_maps

    last = raw_calls[-1]
    charges_from_final_density = conversions.density_to_charges(
        result["density"], engine._data, engine.config
    )
    torch.testing.assert_close(
        final_charges.mono, charges_from_final_density.mono
    )
    torch.testing.assert_close(
        result["density"],
        get_density(result["coefficients"], result["occupation"].sum(-2)),
    )
    torch.testing.assert_close(result["occupation"], case.occupation)
    torch.testing.assert_close(
        result["fenergy"], engine.get_electronic_free_energy()
    )
    torch.testing.assert_close(
        result["potential"].mono,
        engine.charges_to_potential(final_charges).mono,
    )
    if scp_mode == labels.SCP_MODE_CHARGE:
        torch.testing.assert_close(final_charges.mono, last["charges"])
        torch.testing.assert_close(result["hamiltonian"], last["hamiltonian"])
        torch.testing.assert_close(result["density"], last["density"])
        torch.testing.assert_close(result["emo"], last["emo"])
        torch.testing.assert_close(result["coefficients"], last["coefficients"])
        torch.testing.assert_close(result["occupation"], last["occupation"])
        torch.testing.assert_close(engine._data.energy, last["orbital_energy"])
    else:
        # Potential/Fock final conversion evaluates one additional spectrum.
        if scp_mode == labels.SCP_MODE_POTENTIAL:
            selected_potential = Potential.from_tensor(
                last["mapped"], engine._data.potential
            )
            expected_final_hamiltonian = engine.potential_to_hamiltonian(
                selected_potential
            )
        else:
            expected_final_hamiltonian = last["mapped"]

        torch.testing.assert_close(diagonalized[-1], expected_final_hamiltonian)
        torch.testing.assert_close(
            result["hamiltonian"], expected_final_hamiltonian
        )
        torch.testing.assert_close(result["emo"], spectra[-1][0])
        torch.testing.assert_close(result["coefficients"], spectra[-1][1])
        final_orbital_energy = torch.einsum(
            "ik,ki->i", result["density"], engine._data.ints.hcore
        )
        expected_energy = engine._data.ihelp.reduce_orbital_to_atom(
            final_orbital_energy
        ) + engine.interactions.get_energy(
            final_charges, engine._data.cache, engine._data.ihelp
        )
        torch.testing.assert_close(result["energy"], expected_energy)
        assert torch.max(torch.abs(final_charges.mono - last["charges"])) > 1e-9
        expected_potential = engine.charges_to_potential(final_charges)
        torch.testing.assert_close(
            final_potential.mono, expected_potential.mono
        )
        if scp_mode == labels.SCP_MODE_FOCK:
            torch.testing.assert_close(result["hamiltonian"], last["mapped"])
    assert torch.isfinite(result["energy"]).all()
    assert torch.isfinite(result["fenergy"]).all()
    assert torch.isfinite(result["density"]).all()
    assert torch.isfinite(result["emo"]).all()
    assert torch.isfinite(result["coefficients"]).all()
    assert torch.isfinite(result["occupation"]).all()


@pytest.mark.parametrize(
    "scp_mode", [labels.SCP_MODE_POTENTIAL, labels.SCP_MODE_FOCK]
)
def test_legacy_representation_final_conversion_at_finite_temperature(
    scp_mode: int,
) -> None:
    """Finite-T representation finalization performs a second spectrum."""
    case = _prepare_case(
        GFN1_XTB,
        sample_name="H2",
        scf_options={
            "scp_mode": scp_mode,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
            "fermi_etemp": 300.0,
        },
        setup_functional=False,
    )
    engine = case.engine
    diagonalized: list[torch.Tensor] = []
    spectra: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = engine.diagonalize

    def record(hamiltonian: torch.Tensor) -> Any:
        diagonalized.append(hamiltonian.clone())
        result = original(hamiltonian)
        spectra.append((result[0].clone(), result[1].clone()))
        return result

    engine.diagonalize = record
    engine._guess = lambda _charges: case.x.clone()
    result = engine(case.x.clone())
    assert len(diagonalized) == 2
    torch.testing.assert_close(result["emo"], spectra[-1][0])
    torch.testing.assert_close(result["coefficients"], spectra[-1][1])

    # The public occupation and free energy come from the final, extra
    # representation-to-density conversion, not the preceding raw map.
    energies = spectra[-1][0].unsqueeze(-2).expand(*engine._data.nel.shape, -1)
    orbital_mask = engine._data.ihelp.spread_shell_to_orbital(
        engine._data.ihelp.orbitals_per_shell
    )
    mask = orbital_mask.unsqueeze(-2).expand_as(energies)
    expected_occupation = filling.get_fermi_occupation(
        engine._data.nel,
        energies,
        engine.kt,
        mask=mask,
        maxiter=engine.config.fermi.maxiter,
        thr=engine.config.fermi.thresh,
        diff_order=engine.config.fermi.diff_order,
    )
    torch.testing.assert_close(result["occupation"], expected_occupation)
    expected_density = get_density(spectra[-1][1], expected_occupation.sum(-2))
    torch.testing.assert_close(result["density"], expected_density)
    assert torch.isfinite(result["occupation"]).all()
    torch.testing.assert_close(
        result["occupation"].sum(-1), engine._data.nel, atol=1e-4, rtol=0.0
    )
    charges_from_final_density = conversions.density_to_charges(
        result["density"], engine._data, engine.config
    )
    torch.testing.assert_close(
        result["charges"].mono, charges_from_final_density.mono
    )
    expected_entropy = filling.get_fermi_entropy(
        expected_occupation, spectra[-1][0], engine.kt, orbital_mask
    )
    real = real_atoms(engine._data.numbers)
    expected_free_energy = torch.where(
        real,
        (expected_entropy.sum() / real.count_nonzero()).expand_as(real),
        expected_entropy.new_tensor(0.0),
    )
    torch.testing.assert_close(result["fenergy"], expected_free_energy)
    assert torch.isfinite(result["fenergy"]).all()
    assert torch.isfinite(result["energy"]).all()


def test_legacy_gfn2_potential_finalization_preserves_multipoles() -> None:
    """Potential conversion retains GFN2 monopole/dipole/quadrupole rows."""
    case = _prepare_case(
        GFN2_XTB,
        sample_name="H2O",
        int_level=labels.INTLEVEL_QUADRUPOLE,
        scf_options={
            "scp_mode": labels.SCP_MODE_POTENTIAL,
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 0,
            "fermi_etemp": 300.0,
        },
        setup_functional=False,
    )
    charges = case.engine.scf(case.x, return_charges=True)
    potential = case.engine.charges_to_potential(charges)
    assert charges.dipole is not None
    assert charges.quad is not None
    assert potential.dipole is not None
    assert potential.quad is not None
    assert charges.dipole.abs().max() > 0.0
    assert charges.quad.abs().max() > 0.0


def test_legacy_full_mixes_before_reading_convergence() -> None:
    """Legacy full mode calls mixer.iter before observing convergence."""
    case = _prepare_case(
        GFN1_XTB,
        scf_options={
            "mixer": labels.MIXER_LINEAR,
            "maxiter": 1,
            "mix_guess": False,
        },
    )
    events: list[str] = []

    class ConvergedMixer:
        def iter(self, mapped: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
            events.append("iter")
            return mapped

        @property
        def converged(self) -> bool:
            events.append("converged")
            return True

    case.engine.mixer = ConvergedMixer()
    case.engine.scf(case.x, return_charges=False)
    assert events == ["iter", "converged"]
