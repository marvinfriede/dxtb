# This file is part of dxtb.
#
# SPDX-License-Identifier: Apache-2.0
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
"""Production schema and parity tests for pure libcint setup Nodes."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from tad_mctc.tree import Node, stack
from torch.utils._pytree import tree_flatten, tree_unflatten

from dxtb import GFN0_XTB, GFN1_XTB, GFN2_XTB, Calculator, IndexHelper, labels
from dxtb._src.basis.bas import Basis
from dxtb._src.exlibs import libcint
from dxtb._src.exlibs.available import has_libcint
from dxtb._src.integral.driver.libcint import (
    LibcintIntegralSetup,
    LibcintIntegralSetupNode,
    build_overlap_libcint,
    core_libcint_setup,
    setup_libcint,
)
from dxtb._src.integral.evaluation import (
    IntegralSetup,
    build_integral_matrices,
    setup_integral_evaluation,
)
from dxtb._src.param import ParamModule

from .samples import samples

pytestmark = pytest.mark.skipif(not has_libcint, reason="libcint not available")


def _legacy_setup(numbers, parametrization, force_cpu: bool):
    device = torch.device("cpu") if force_cpu else numbers.device
    backend_numbers = numbers.to(device=device)
    ihelp = IndexHelper.from_numbers(backend_numbers, parametrization).to(
        device=device
    )
    basis = Basis(
        backend_numbers,
        parametrization,
        ihelp,
        device=device,
        dtype=parametrization.dtype,
    )
    return LibcintIntegralSetup(
        numbers=backend_numbers,
        ihelp=ihelp,
        basis_setups=(basis.setup_data(),),
        force_cpu=force_cpu,
    )


def _leaves(value) -> tuple[torch.Tensor, ...]:
    leaves, _ = tree_flatten(value)
    return tuple(leaf for leaf in leaves if isinstance(leaf, torch.Tensor))


@pytest.mark.parametrize("method", ["gfn1", "gfn2"])
@pytest.mark.parametrize("force_cpu", [True, False])
def test_libcint_setup_node_schema_snapshot_and_conversion(
    method: str, force_cpu: bool
) -> None:
    """All captured libcint tensors are visible and preserve source dtypes."""
    numbers = samples["H2O"]["numbers"]
    par = ParamModule(
        GFN1_XTB if method == "gfn1" else GFN2_XTB, dtype=torch.float64
    )
    legacy = _legacy_setup(numbers, par, force_cpu=force_cpu)
    core = core_libcint_setup(legacy)
    assert type(core) is LibcintIntegralSetupNode
    assert isinstance(core, Node)
    assert not hasattr(core, "ihelp")
    assert not hasattr(core, "par")
    assert not hasattr(core.basis_setups[0], "ihelp")
    assert len(core.basis_setups) == 1

    leaves = _leaves(core)
    assert len(leaves) == 14
    assert all(leaf.device.type == "cpu" for leaf in leaves)
    basis = core.basis_setups[0]
    assert basis.slater.dtype == torch.float64
    for name in (
        "numbers",
        "unique",
        "ngauss",
        "pqn",
        "valence",
        "shells_per_atom",
        "shells_to_ushell",
        "angular",
        "ushells_per_unique",
        "unique_angular",
    ):
        assert getattr(basis, name).dtype in (
            torch.int32,
            torch.int64,
            torch.uint8,
            torch.bool,
        )

    flat, spec = tree_flatten(core)
    rebuilt = tree_unflatten(flat, spec)
    assert type(rebuilt) is LibcintIntegralSetupNode
    assert rebuilt.force_cpu is force_cpu
    assert torch.equal(rebuilt.index.shells_to_atom, core.index.shells_to_atom)
    assert torch.equal(
        rebuilt.index.orbitals_to_shell, core.index.orbitals_to_shell
    )

    f32 = core.to(dtype=torch.float32)
    f64 = f32.to(dtype=torch.float64)
    meta = core.to(device="meta")
    for converted in (f32, f64, meta):
        assert type(converted) is LibcintIntegralSetupNode
        assert converted.force_cpu is force_cpu
    target_device = "cpu" if force_cpu else "meta"
    assert all(leaf.device.type == target_device for leaf in _leaves(meta))
    assert f32.basis_setups[0].slater.dtype == torch.float32
    assert f64.basis_setups[0].slater.dtype == torch.float64
    assert f32.numbers.dtype == numbers.dtype
    assert f32.basis_setups[0].angular.dtype == basis.angular.dtype

    stacked = stack((core, core))
    assert type(stacked) is LibcintIntegralSetupNode
    assert stacked.basis_setups[0].slater.shape[0] == 2
    assert stacked.index.shells_to_atom.shape[0] == 2

    # The Node is an owned setup snapshot, including the differentiable leaf.
    old_slater = core.basis_setups[0].slater.clone()
    old_shell_map = core.basis_setups[0].shells_to_ushell.clone()
    with torch.no_grad():
        legacy.basis_setups[0].slater.add_(0.5)
        legacy.basis_setups[0].ihelp.shells_to_ushell.add_(1)
    assert torch.equal(core.basis_setups[0].slater, old_slater)
    assert torch.equal(core.basis_setups[0].shells_to_ushell, old_shell_map)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
@pytest.mark.parametrize("force_cpu", [True, False])
def test_libcint_system_node_matches_legacy_matrices_and_history(
    method: str,
    force_cpu: bool,
) -> None:
    """Node-backed and legacy wrappers preserve matrices and call history."""
    numbers = samples["H2O"]["numbers"]
    positions = samples["H2O"]["positions"].to(dtype=torch.float64)
    parametrization = {
        "gfn0": GFN0_XTB,
        "gfn1": GFN1_XTB,
        "gfn2": GFN2_XTB,
    }[method]
    calc = Calculator(
        numbers,
        parametrization,
        dtype=positions.dtype,
        opts={
            "verbosity": 0,
            "int_driver": labels.INTDRIVER_LIBCINT,
            "int_level": labels.INTLEVEL_QUADRUPOLE,
        },
    )
    param_module = ParamModule(parametrization, dtype=positions.dtype)
    legacy_backend = _legacy_setup(numbers, param_module, force_cpu)
    legacy_outer = IntegralSetup(
        driver_type=labels.INTDRIVER_LIBCINT,
        intlevel=labels.INTLEVEL_QUADRUPOLE,
        libcint=legacy_backend,
    )
    system_outer = calc.system.integral_setup
    assert system_outer is not None and system_outer.libcint is not None
    assert type(system_outer.libcint) is LibcintIntegralSetupNode
    node_ihelp = IndexHelper.from_numbers(numbers, param_module).to(
        device=legacy_backend.numbers.device
    )
    core_outer = setup_integral_evaluation(
        numbers,
        param_module,
        node_ihelp,
        driver_type=labels.INTDRIVER_LIBCINT,
        intlevel=labels.INTLEVEL_QUADRUPOLE,
        force_cpu_for_libcint=force_cpu,
    )
    assert core_outer is not None and core_outer.libcint is not None
    assert type(core_outer.libcint) is LibcintIntegralSetupNode
    assert core_outer.libcint.force_cpu is force_cpu
    core = core_outer.libcint

    a = positions.clone().requires_grad_()
    b = positions.clone()
    b[1, 0] += 0.07
    core_a, refocc_a, norm_a = build_integral_matrices(
        core_outer, calc.system.h0_setup, a, 0.0
    )
    legacy_a, legacy_refocc, legacy_norm = build_integral_matrices(
        legacy_outer, calc.system.h0_setup, a, 0.0
    )
    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        torch.testing.assert_close(
            getattr(core_a, name), getattr(legacy_a, name)
        )
    torch.testing.assert_close(refocc_a, legacy_refocc)
    torch.testing.assert_close(norm_a, legacy_norm)

    build_integral_matrices(core_outer, calc.system.h0_setup, b, 0.0)
    repeated, refocc_repeated, norm_repeated = build_integral_matrices(
        core_outer, calc.system.h0_setup, a, 0.0
    )
    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        torch.testing.assert_close(
            getattr(repeated, name), getattr(core_a, name)
        )
    torch.testing.assert_close(refocc_repeated, refocc_a)
    torch.testing.assert_close(norm_repeated, norm_a)

    core_loss = core_a.overlap.square().sum() + core_a.dipole.square().sum()
    legacy_loss = (
        legacy_a.overlap.square().sum() + legacy_a.dipole.square().sum()
    )
    core_gradient = torch.autograd.grad(core_loss, a, retain_graph=True)[0]
    legacy_gradient = torch.autograd.grad(legacy_loss, a)[0]
    torch.testing.assert_close(core_gradient, legacy_gradient)
    assert core_outer.libcint is core


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
def test_libcint_core_basis_keeps_slater_parameter_graph(method: str) -> None:
    """Captured Slater tensors remain differentiable through CGTO creation."""
    numbers = samples["H2O"]["numbers"]
    parametrization = {
        "gfn0": GFN0_XTB,
        "gfn1": GFN1_XTB,
        "gfn2": GFN2_XTB,
    }[method]
    par = ParamModule(parametrization, dtype=torch.float64)
    parameter_name = "parameter_tree.element.H.slater.param"
    parameter = dict(par.named_parameters())[parameter_name]
    parameter.requires_grad_(True)
    legacy = _legacy_setup(numbers, par, force_cpu=True)
    setup = core_libcint_setup(legacy)
    alphas, coeffs = setup.basis_setups[0].create_libcint_cgtos()
    objective = sum(alpha.square().sum() for alpha in alphas) + sum(
        coeff.square().sum() for coeff in coeffs
    )
    gradient = torch.autograd.grad(objective, parameter)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0

    positions = samples["H2O"]["positions"].to(dtype=torch.float64)
    core_overlap = build_overlap_libcint(setup_libcint(setup, positions))
    legacy_overlap = build_overlap_libcint(setup_libcint(legacy, positions))
    core_gradient = torch.autograd.grad(
        core_overlap.square().sum(), parameter, retain_graph=True
    )[0]
    legacy_gradient = torch.autograd.grad(
        legacy_overlap.square().sum(), parameter
    )[0]
    assert torch.isfinite(core_gradient).all()
    assert core_gradient.abs().sum() > 0
    torch.testing.assert_close(core_gradient, legacy_gradient)


def test_libcint_node_preserves_higher_angular_shell_order() -> None:
    """The wrapper's derived AO order matches explicit IndexHelper maps."""
    numbers = samples["SiH4"]["numbers"]
    positions = samples["SiH4"]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    legacy = _legacy_setup(numbers, par, force_cpu=True)
    core = core_libcint_setup(legacy)
    core_overlap = build_overlap_libcint(setup_libcint(core, positions))
    legacy_overlap = build_overlap_libcint(setup_libcint(legacy, positions))
    assert core_overlap.shape == legacy_overlap.shape
    torch.testing.assert_close(core_overlap, legacy_overlap)


@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "SiH4", "CH4"])
def test_libcint_node_wrapper_maps_match_legacy(name: str) -> None:
    """The wrapper-derived contracted and backward maps retain exact order."""
    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    legacy = _legacy_setup(numbers, par, force_cpu=True)
    atom_basis = legacy.basis_setups[0].create_libcint(positions)
    legacy_wrapper = libcint.LibcintWrapper(atom_basis, legacy.ihelp)
    core_wrapper = libcint.LibcintWrapper(atom_basis)

    for name in ("full_ao_to_atom", "full_ao_to_shell"):
        torch.testing.assert_close(
            getattr(core_wrapper, name), getattr(legacy_wrapper, name)
        )
    np.testing.assert_array_equal(
        core_wrapper.full_shell_to_aoloc,
        legacy_wrapper.full_shell_to_aoloc,
    )
    assert core_wrapper.nao() == legacy_wrapper.nao()
    assert core_wrapper.shell_idxs == legacy_wrapper.shell_idxs
    assert core_wrapper.ao_idxs() == legacy_wrapper.ao_idxs()
    assert core_wrapper.ngauss_at_shell == legacy_wrapper.ngauss_at_shell
    torch.testing.assert_close(
        core_wrapper.ao_to_atom(), legacy_wrapper.ao_to_atom()
    )
    torch.testing.assert_close(
        core_wrapper.ao_to_shell(), legacy_wrapper.ao_to_shell()
    )

    core_uncontracted, core_map = core_wrapper.get_uncontracted_wrapper()
    legacy_uncontracted, legacy_map = legacy_wrapper.get_uncontracted_wrapper()
    torch.testing.assert_close(core_map, legacy_map)
    for name in ("full_ao_to_atom", "full_ao_to_shell"):
        torch.testing.assert_close(
            getattr(core_uncontracted, name),
            getattr(legacy_uncontracted, name),
        )
    np.testing.assert_array_equal(
        core_uncontracted.full_shell_to_aoloc,
        legacy_uncontracted.full_shell_to_aoloc,
    )


def test_core_libcint_setup_rejects_non_cpu_force_cpu_source() -> None:
    """Direct conversion requires normalized CPU sources when force_cpu is set."""
    numbers = samples["H2"]["numbers"]
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    legacy = _legacy_setup(numbers, par, force_cpu=True)
    inconsistent = replace(legacy, numbers=legacy.numbers.to(device="meta"))
    with pytest.raises(ValueError, match="CPU-resident libcint setup tensors"):
        core_libcint_setup(inconsistent)


def test_libcint_setup_node_is_used_by_system_and_call_data_is_local() -> None:
    """Single-System setup owns a Node while wrappers remain call-local."""
    numbers = samples["H2"]["numbers"]
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_LIBCINT},
    )
    setup = calc.system.integral_setup
    assert setup is not None
    assert type(setup.libcint) is LibcintIntegralSetupNode
    call1 = setup_libcint(setup.libcint, positions)
    call2 = setup_libcint(setup.libcint, positions)
    assert call1 is not call2
    assert call1.drivers[0] is not call2.drivers[0]
    assert call1.ihelp is setup.libcint.index
    assert call1.batch_mode == 0

    converted = setup.to(device="meta", dtype=torch.float32)
    assert converted.libcint is not None
    assert type(converted.libcint) is LibcintIntegralSetupNode
    assert all(leaf.device.type == "cpu" for leaf in _leaves(converted.libcint))
    assert converted.libcint.basis_setups[0].slater.dtype == torch.float32


def test_libcint_singlepoint_matches_legacy_setup() -> None:
    """The pure System singlepoint preserves the legacy libcint result."""
    numbers = samples["H2"]["numbers"]
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_LIBCINT},
    )
    system = calc.system
    assert system.integral_setup is not None
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    legacy_backend = _legacy_setup(numbers, par, force_cpu=True)
    legacy_setup = replace(system.integral_setup, libcint=legacy_backend)
    legacy_system = replace(system, integral_setup=legacy_setup)

    core_result = system.singlepoint(positions)
    legacy_result = legacy_system.singlepoint(positions)
    torch.testing.assert_close(core_result.energy, legacy_result.energy)
    for name in ("hcore", "overlap", "dipole_integrals", "overlap_norm"):
        torch.testing.assert_close(
            getattr(core_result, name), getattr(legacy_result, name)
        )
    torch.testing.assert_close(core_result.occupation, legacy_result.occupation)
