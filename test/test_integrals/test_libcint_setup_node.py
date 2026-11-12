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

from typing import Any

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
    build_overlap_libcint,
    setup_libcint,
)
from dxtb._src.integral.evaluation import (
    build_integral_matrices,
    setup_integral_evaluation,
)
from dxtb._src.param import ParamModule

from .samples import samples

pytestmark = pytest.mark.skipif(not has_libcint, reason="libcint not available")


def _basis_and_setup(
    numbers: torch.Tensor,
    parametrization: ParamModule,
    force_cpu: bool,
) -> tuple[Basis, LibcintIntegralSetup]:
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
    return basis, setup_libcint(basis, force_cpu=force_cpu)


def _leaves(value: Any) -> tuple[torch.Tensor, ...]:
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
    basis, core = _basis_and_setup(numbers, par, force_cpu=force_cpu)
    assert type(core) is LibcintIntegralSetup
    assert isinstance(core, Node)
    assert not hasattr(core, "ihelp")
    assert not hasattr(core, "par")
    assert not hasattr(core, "basis_setups")
    assert core.index is basis.ihelp

    leaves = _leaves(core)
    expected_n_shells = 4 if method == "gfn1" else 3
    assert len(core.alphas) == expected_n_shells
    assert len(core.coeffs) == expected_n_shells
    assert len(leaves) == 13 + 2 * expected_n_shells
    assert all(leaf.device.type == "cpu" for leaf in leaves)
    assert all(a.dtype == torch.float64 for a in core.alphas)
    assert all(c.dtype == torch.float64 for c in core.coeffs)
    assert core.numbers.dtype in (torch.int32, torch.int64)
    assert core.active_atoms is None

    flat, spec = tree_flatten(core)
    rebuilt = tree_unflatten(flat, spec)
    assert type(rebuilt) is LibcintIntegralSetup
    assert rebuilt.force_cpu is force_cpu
    assert rebuilt.shell_layout == core.shell_layout
    assert torch.equal(rebuilt.index.shells_to_atom, core.index.shells_to_atom)
    assert torch.equal(
        rebuilt.index.orbitals_to_shell, core.index.orbitals_to_shell
    )

    f32 = core.to(dtype=torch.float32)
    f64 = f32.to(dtype=torch.float64)
    meta = core.to(device="meta")
    for converted in (f32, f64, meta):
        assert type(converted) is LibcintIntegralSetup
        assert converted.force_cpu is force_cpu
    target_device = "cpu" if force_cpu else "meta"
    assert all(leaf.device.type == target_device for leaf in _leaves(meta))
    assert all(a.dtype == torch.float32 for a in f32.alphas)
    assert all(c.dtype == torch.float32 for c in f32.coeffs)
    assert all(a.dtype == torch.float64 for a in f64.alphas)
    assert all(c.dtype == torch.float64 for c in f64.coeffs)
    assert f32.numbers.dtype == numbers.dtype
    assert f32.index.angular.dtype == core.index.angular.dtype

    stacked = stack((core, core))
    assert type(stacked) is LibcintIntegralSetup
    assert stacked.alphas[0].shape[0] == 2
    assert stacked.coeffs[0].shape[0] == 2
    assert stacked.index.shells_to_atom.shape[0] == 2

    # The Node owns precomputed CGTO tensors rather than aliasing Basis.slater.
    old_alphas = tuple(a.clone() for a in core.alphas)
    old_coeffs = tuple(c.clone() for c in core.coeffs)
    with torch.no_grad():
        basis.slater.add_(0.5)
        basis.pqn.add_(1)
    for cur_a, ref_a in zip(core.alphas, old_alphas):
        assert torch.equal(cur_a, ref_a)
    for cur_c, ref_c in zip(core.coeffs, old_coeffs):
        assert torch.equal(cur_c, ref_c)


@pytest.mark.parametrize("method", ["gfn0", "gfn1", "gfn2"])
@pytest.mark.parametrize("force_cpu", [True, False])
def test_libcint_system_node_history_and_force_cpu_parity(
    method: str,
    force_cpu: bool,
) -> None:
    """Node-backed libcint setup preserves matrices, gradients, and history."""
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
    system_outer = calc.system.integral_setup
    assert system_outer is not None and system_outer.libcint is not None
    assert type(system_outer.libcint) is LibcintIntegralSetup
    node_ihelp = IndexHelper.from_numbers(numbers, param_module)
    core_outer = setup_integral_evaluation(
        numbers,
        param_module,
        node_ihelp,
        driver_type=labels.INTDRIVER_LIBCINT,
        intlevel=labels.INTLEVEL_QUADRUPOLE,
        force_cpu_for_libcint=force_cpu,
    )
    assert core_outer is not None and core_outer.libcint is not None
    assert type(core_outer.libcint) is LibcintIntegralSetup
    assert core_outer.libcint.force_cpu is force_cpu
    core = core_outer.libcint

    a = positions.clone().requires_grad_()
    b = positions.clone()
    b[1, 0] += 0.07
    core_a, refocc_a = build_integral_matrices(
        core_outer, calc.system.h0_setup, a, 0.0
    )
    sys_a, sys_refocc = build_integral_matrices(
        system_outer, calc.system.h0_setup, a, 0.0
    )
    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        torch.testing.assert_close(getattr(core_a, name), getattr(sys_a, name))
    torch.testing.assert_close(refocc_a, sys_refocc)

    build_integral_matrices(core_outer, calc.system.h0_setup, b, 0.0)
    repeated, refocc_repeated = build_integral_matrices(
        core_outer, calc.system.h0_setup, a, 0.0
    )
    for name in ("overlap", "dipole", "quadrupole", "hcore"):
        torch.testing.assert_close(
            getattr(repeated, name), getattr(core_a, name)
        )
    torch.testing.assert_close(refocc_repeated, refocc_a)

    assert core_a.dipole is not None and sys_a.dipole is not None
    core_loss = core_a.overlap.square().sum() + core_a.dipole.square().sum()
    sys_loss = sys_a.overlap.square().sum() + sys_a.dipole.square().sum()
    core_gradient = torch.autograd.grad(core_loss, a, retain_graph=True)[0]
    sys_gradient = torch.autograd.grad(sys_loss, a)[0]
    torch.testing.assert_close(core_gradient, sys_gradient)
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
    basis, setup = _basis_and_setup(numbers, par, force_cpu=True)
    objective = sum(alpha.square().sum() for alpha in setup.alphas) + sum(
        coeff.square().sum() for coeff in setup.coeffs
    )
    gradient = torch.autograd.grad(objective, parameter, retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0

    positions = samples["H2O"]["positions"].to(dtype=torch.float64)
    core_overlap = build_overlap_libcint(setup, positions)
    direct_overlap = libcint.overlap(*basis.create_libcint(positions))
    core_gradient = torch.autograd.grad(
        core_overlap.square().sum(), parameter, retain_graph=True
    )[0]
    direct_gradient = torch.autograd.grad(
        direct_overlap.square().sum(), parameter
    )[0]
    assert torch.isfinite(core_gradient).all()
    assert core_gradient.abs().sum() > 0
    torch.testing.assert_close(core_gradient, direct_gradient)


def test_libcint_node_preserves_higher_angular_shell_order() -> None:
    """The wrapper's derived AO order matches explicit Basis construction."""
    numbers = samples["SiH4"]["numbers"]
    positions = samples["SiH4"]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    basis, core = _basis_and_setup(numbers, par, force_cpu=True)
    core_overlap = build_overlap_libcint(core, positions)
    direct_overlap = libcint.overlap(*basis.create_libcint(positions))
    assert core_overlap.shape == direct_overlap.shape
    torch.testing.assert_close(core_overlap, direct_overlap)


@pytest.mark.parametrize("name", ["H2", "LiH", "H2O", "SiH4", "CH4"])
def test_libcint_node_wrapper_maps_match_index_helper(name: str) -> None:
    """LibcintIntegralSetup layout preserves shell and AO structural order."""
    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    _, core = _basis_and_setup(numbers, par, force_cpu=True)
    libcint_basis, structure = core.create_libcint(positions)
    layout = libcint_basis.layout(structure.numbers)
    np.testing.assert_array_equal(
        layout.shell_atom, core.index.shells_to_atom.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(
        layout.shell_angmom, core.index.angular.detach().cpu().numpy()
    )
    np.testing.assert_array_equal(
        layout.ao_loc[1:] - layout.ao_loc[:-1],
        core.index.orbitals_per_shell.detach().cpu().numpy(),
    )
    assert layout.nao == core.index.nao


def test_setup_libcint_rejects_non_cpu_force_cpu_source() -> None:
    """Direct construction and setup reject non-CPU sources when force_cpu."""
    numbers = samples["H2"]["numbers"]
    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    basis, core = _basis_and_setup(numbers, par, force_cpu=False)
    basis.slater = basis.slater.to(device="meta")
    with pytest.raises(ValueError, match="CPU-resident libcint setup tensors"):
        setup_libcint(basis, force_cpu=True)
    meta_core = core.to(device="meta")
    with pytest.raises(ValueError, match="CPU-resident libcint setup tensors"):
        meta_core.replace(force_cpu=True)


def test_libcint_setup_node_is_used_by_system_and_call_data_is_local() -> None:
    """Single-System setup owns a Node while libcint objects are call-local."""
    numbers = samples["H2"]["numbers"]
    positions = samples["H2"]["positions"].to(dtype=torch.float64)
    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_LIBCINT},
    )
    setup = calc.system.integral_setup
    assert setup is not None and setup.libcint is not None
    assert type(setup.libcint) is LibcintIntegralSetup
    basis1, struct1 = setup.libcint.create_libcint(positions)
    basis2, struct2 = setup.libcint.create_libcint(positions)
    assert basis1 is not basis2
    assert struct1 is not struct2

    converted = setup.to(device="meta", dtype=torch.float32)
    assert converted.libcint is not None
    assert type(converted.libcint) is LibcintIntegralSetup
    assert all(leaf.device.type == "cpu" for leaf in _leaves(converted.libcint))
    assert all(a.dtype == torch.float32 for a in converted.libcint.alphas)
    assert all(c.dtype == torch.float32 for c in converted.libcint.coeffs)


def test_libcint_eval_does_not_reexpand_cgtos_and_builds_basis_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CGTO expansion runs once in setup and libcint.Basis once per moment eval."""
    import dxtb._src.basis.bas as bas_mod

    numbers = samples["SiH4"]["numbers"]
    positions = samples["SiH4"]["positions"].to(dtype=torch.float64)
    par = ParamModule(GFN2_XTB, dtype=positions.dtype)
    ihelp = IndexHelper.from_numbers(numbers, par)

    s2g_calls = 0
    lc_basis_calls = 0
    orig_s2g = bas_mod.slater_to_gauss
    orig_lc_basis = libcint.Basis

    def wrapped_s2g(*args: Any, **kwargs: Any) -> Any:
        nonlocal s2g_calls
        s2g_calls += 1
        return orig_s2g(*args, **kwargs)

    def wrapped_lc_basis(*args: Any, **kwargs: Any) -> Any:
        nonlocal lc_basis_calls
        lc_basis_calls += 1
        return orig_lc_basis(*args, **kwargs)

    monkeypatch.setattr(bas_mod, "slater_to_gauss", wrapped_s2g)
    monkeypatch.setattr(libcint, "Basis", wrapped_lc_basis)

    isetup = setup_integral_evaluation(
        numbers,
        par,
        ihelp,
        driver_type=labels.INTDRIVER_LIBCINT,
        intlevel=labels.INTLEVEL_QUADRUPOLE,
    )
    assert isetup is not None and isetup.libcint is not None
    assert s2g_calls == len(isetup.libcint.alphas)
    assert lc_basis_calls == 0

    calc = Calculator(
        numbers,
        GFN2_XTB,
        dtype=positions.dtype,
        opts={"verbosity": 0, "int_driver": labels.INTDRIVER_LIBCINT},
    )
    s2g_calls = 0
    lc_basis_calls = 0
    build_integral_matrices(isetup, calc.system.h0_setup, positions, 0.0)
    assert s2g_calls == 0
    assert lc_basis_calls == 1


@pytest.mark.parametrize("name", ["H2O", "SiH4"])
def test_libcint_native_ket_moments_match_pytorch(name: str) -> None:
    """New j/jj values match the established shifted PyTorch moments."""
    numbers = samples[name]["numbers"]
    positions = samples[name]["positions"].to(dtype=torch.float64)
    outputs = []
    gradients = []
    for driver in (labels.INTDRIVER_PYTORCH, labels.INTDRIVER_LIBCINT):
        calc = Calculator(
            numbers,
            GFN1_XTB,
            dtype=positions.dtype,
            opts={
                "verbosity": 0,
                "int_driver": driver,
                "int_level": labels.INTLEVEL_QUADRUPOLE,
            },
        )
        assert calc.system.integral_setup is not None
        assert calc.system.h0_setup is not None
        outputs.append(
            build_integral_matrices(
                calc.system.integral_setup,
                calc.system.h0_setup,
                positions,
                0.0,
            )[0]
        )
        variable_positions = positions.clone().requires_grad_(True)
        matrices = build_integral_matrices(
            calc.system.integral_setup,
            calc.system.h0_setup,
            variable_positions,
            0.0,
        )[0]
        assert matrices.dipole is not None
        assert matrices.quadrupole is not None
        dipole_probe = torch.linspace(
            -0.3, 0.4, matrices.dipole.numel(), dtype=positions.dtype
        ).reshape_as(matrices.dipole)
        quad_probe = torch.linspace(
            -0.2, 0.5, matrices.quadrupole.numel(), dtype=positions.dtype
        ).reshape_as(matrices.quadrupole)
        scalar = (matrices.dipole * dipole_probe).sum() + (
            matrices.quadrupole * quad_probe
        ).sum()
        gradients.append(torch.autograd.grad(scalar, variable_positions)[0])

    torch_reference, libcint_result = outputs
    for field in ("overlap", "dipole", "quadrupole", "hcore"):
        torch.testing.assert_close(
            getattr(libcint_result, field),
            getattr(torch_reference, field),
            atol=2.0e-11,
            rtol=2.0e-11,
        )
    torch.testing.assert_close(
        gradients[1], gradients[0], atol=2.0e-9, rtol=2.0e-9
    )


@pytest.mark.parametrize(
    ("method", "name"),
    [(GFN1_XTB, "H2O"), (GFN2_XTB, "H2"), (GFN2_XTB, "H2O")],
    ids=["gfn1-h2o", "gfn2-h2", "gfn2-h2o"],
)
def test_libcint_full_singlepoint_matches_pytorch(
    method: Any, name: str
) -> None:
    """New libcint preserves full functional energies and forces."""
    sample = samples[name]
    numbers = sample["numbers"]
    positions = sample["positions"].to(dtype=torch.float64)
    results = []

    for driver in (labels.INTDRIVER_PYTORCH, labels.INTDRIVER_LIBCINT):
        calculator = Calculator(
            numbers,
            method,
            dtype=positions.dtype,
            opts={
                "verbosity": 0,
                "int_driver": driver,
                "int_level": labels.INTLEVEL_QUADRUPOLE,
            },
        )
        variable_positions = positions.clone().requires_grad_(True)
        result = calculator.system.singlepoint(variable_positions)
        forces = torch.autograd.grad(result.energy.sum(), variable_positions)[0]
        results.append((result, forces))

    torch_reference, libcint_result = results
    for field in ("energy", "scf", "density", "hcore", "overlap"):
        torch.testing.assert_close(
            getattr(libcint_result[0], field),
            getattr(torch_reference[0], field),
            atol=2.0e-9,
            rtol=2.0e-9,
        )
    torch.testing.assert_close(
        libcint_result[0].charges.mono,
        torch_reference[0].charges.mono,
        atol=2.0e-9,
        rtol=2.0e-9,
    )
    torch.testing.assert_close(
        libcint_result[1],
        torch_reference[1],
        atol=2.0e-8,
        rtol=2.0e-8,
    )


def test_zero_padded_single_system_moments_match_unpadded_on_both_drivers() -> (
    None
):
    """Zero-padded Z=0 slots evaluate cleanly and match unpadded moments."""
    from dxtb._src.integral.driver.libcint import (
        evaluate_moments as evaluate_libcint_moments,
    )
    from dxtb._src.integral.driver.pytorch import (
        evaluate_moments as evaluate_pytorch_moments,
    )
    from dxtb._src.integral.driver.pytorch import (
        setup_integrals as setup_pytorch,
    )

    par = ParamModule(GFN2_XTB, dtype=torch.float64)
    nums_ref = torch.tensor([8, 1])
    pos_ref = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8]], dtype=torch.float64
    )
    ih_ref = IndexHelper.from_numbers(nums_ref, par)
    bas_ref = Basis(nums_ref, par, ih_ref, dtype=torch.float64)
    ovlp_ref, dip_ref, quad_ref = evaluate_pytorch_moments(
        setup_pytorch(bas_ref), pos_ref, labels.INTLEVEL_QUADRUPOLE
    )

    nums_pad = torch.tensor([8, 1, 0])
    pos_pad = torch.tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 1.8], [9.0, -4.0, 2.0]],
        dtype=torch.float64,
    )
    ih_pad = IndexHelper.from_numbers(nums_pad, par)
    bas_pad = Basis(nums_pad, par, ih_pad, dtype=torch.float64)

    ovlp_pt, dip_pt, quad_pt = evaluate_pytorch_moments(
        setup_pytorch(bas_pad), pos_pad, labels.INTLEVEL_QUADRUPOLE
    )
    lc_setup = setup_libcint(bas_pad, force_cpu=True)
    assert lc_setup.active_atoms is not None
    ovlp_lc, dip_lc, quad_lc = evaluate_libcint_moments(
        lc_setup, pos_pad, labels.INTLEVEL_QUADRUPOLE
    )

    for actual_o, actual_d, actual_q in (
        (ovlp_pt, dip_pt, quad_pt),
        (ovlp_lc, dip_lc, quad_lc),
    ):
        assert actual_d is not None and dip_ref is not None
        assert actual_q is not None and quad_ref is not None
        torch.testing.assert_close(actual_o, ovlp_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_d, dip_ref, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(actual_q, quad_ref, atol=1e-12, rtol=1e-12)


def test_independent_backward_passes_and_functional_slater_jacrev() -> None:
    """Reusable setups support independent backward passes and jacrev."""
    from torch.func import jacrev

    numbers = samples["H2O"]["numbers"]
    pos1 = samples["H2O"]["positions"].to(dtype=torch.float64)
    pos2 = pos1.clone()
    pos2[1, 2] += 0.11

    for driver in (labels.INTDRIVER_PYTORCH, labels.INTDRIVER_LIBCINT):
        calc = Calculator(
            numbers,
            GFN2_XTB,
            dtype=torch.float64,
            opts={"verbosity": 0, "int_driver": driver},
        )
        isetup = calc.system.integral_setup
        h0_setup = calc.system.h0_setup
        assert isetup is not None and h0_setup is not None

        p1 = pos1.clone().requires_grad_(True)
        m1, _ = build_integral_matrices(isetup, h0_setup, p1, 0.0)
        assert m1.quadrupole is not None
        (m1.overlap.square().sum() + m1.quadrupole.square().sum()).backward()
        assert p1.grad is not None and torch.isfinite(p1.grad).all()

        p2 = pos2.clone().requires_grad_(True)
        m2, _ = build_integral_matrices(isetup, h0_setup, p2, 0.0)
        assert m2.quadrupole is not None
        (m2.overlap.square().sum() + m2.quadrupole.square().sum()).backward()
        assert p2.grad is not None and torch.isfinite(p2.grad).all()

    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    param_key = "parameter_tree.element.H.slater.param"
    h_slater = dict(par.named_parameters())[param_key].detach().clone()

    def cgtos_for_slater(slater_val: torch.Tensor) -> torch.Tensor:
        """Sum of squared H/O CGTO tensors under functional parameter call."""
        return _functional_cgto_loss(par, param_key, slater_val, numbers)

    jac = jacrev(cgtos_for_slater)(h_slater)
    assert jac.shape == h_slater.shape
    assert torch.isfinite(jac).all()
    assert jac.abs().sum() > 0


def _functional_cgto_loss(
    par: ParamModule,
    param_key: str,
    slater_val: torch.Tensor,
    numbers: torch.Tensor,
) -> torch.Tensor:
    """Functional CGTO objective wrt ``param_key`` via ``torch.nn.utils.stateless``."""
    from torch.func import functional_call

    class _Runner(torch.nn.Module):
        def __init__(self, mod: ParamModule) -> None:
            super().__init__()
            self.mod = mod

        def forward(self) -> torch.Tensor:
            ih = IndexHelper.from_numbers(numbers, self.mod)
            bas = Basis(numbers, self.mod, ih, dtype=torch.float64)
            lc = setup_libcint(bas, force_cpu=True)
            return sum(a.square().sum() for a in lc.alphas) + sum(
                c.square().sum() for c in lc.coeffs
            )

    runner = _Runner(par)
    return functional_call(runner, {f"mod.{param_key}": slater_val}, ())


def test_setup_libcint_rejects_mismatched_numbers_and_index_helper() -> None:
    """setup_libcint rejects Basis with unique-only numbers for multi-atom ihelp."""
    numbers = torch.tensor([8, 1, 1])
    par = ParamModule(GFN1_XTB, dtype=torch.float64)
    ihelp = IndexHelper.from_numbers(numbers, par)
    mismatched_basis = Basis(torch.unique(numbers), par, ihelp)

    with pytest.raises(ValueError, match="matching IndexHelper"):
        setup_libcint(mismatched_basis, force_cpu=True)
