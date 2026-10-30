"""Test helpers for exercising Anderson through the shared controller."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from dxtb._src.calculators.config.scf import ConfigSCF
from dxtb._src.scf.functional.anderson import (
    AndersonMethod,
    AndersonMethodSettings,
    setup_anderson_method_settings,
)
from dxtb._src.scf.functional.data import SCFMapResult
from dxtb._src.scf.functional.problem import SCFProblem
from dxtb._src.scf.functional.solver import (
    DynamicDampingSettings,
    SCFSolveResult,
    SCFSolverSettings,
    _solve_map,
    setup_solver_settings,
)
from dxtb._src.scf.functional.strategy import SCFStage, SCFStrategy
from dxtb._src.typing import Tensor


@dataclass(frozen=True)
class AndersonTestSettings:
    """Compact test-only bundle kept convenient for legacy test cases."""

    maxiter: int = 8
    mix_guess: bool = False
    damp: float = 0.4
    damp_init: float = 0.1
    generations: int = 5
    soft_start: bool = True
    diagonal_offset: float | None = 0.01
    x_atol: float = 0.0
    x_atol_max: float = 0.0
    dtype: torch.dtype = torch.float64
    solver_settings: SCFSolverSettings | None = None
    method_settings: AndersonMethodSettings | None = None
    dynamic_damping: DynamicDampingSettings = DynamicDampingSettings()

    def solver(self) -> SCFSolverSettings:
        if self.solver_settings is not None:
            base = self.solver_settings
            return SCFSolverSettings(
                maxiter=self.maxiter,
                mix_guess=self.mix_guess,
                damp=self.damp,
                x_atol=_effective(self.x_atol, self.dtype),
                x_atol_max=_effective(self.x_atol_max, self.dtype),
                dynamic_damping=self.dynamic_damping,
                requested_x_atol=self.x_atol,
                requested_x_atol_max=self.x_atol_max,
            )
        return SCFSolverSettings(
            maxiter=self.maxiter,
            mix_guess=self.mix_guess,
            damp=self.damp,
            x_atol=_effective(self.x_atol, self.dtype),
            x_atol_max=_effective(self.x_atol_max, self.dtype),
            dynamic_damping=self.dynamic_damping,
        )

    def method(self) -> AndersonMethodSettings:
        return AndersonMethodSettings(
            self.damp_init,
            self.generations,
            self.soft_start,
            self.diagonal_offset,
        )


def _effective(value: float, dtype: torch.dtype) -> float:
    resolution = torch.finfo(dtype).resolution
    return resolution * 50 if resolution > value else value


def from_config(config: ConfigSCF, dtype: torch.dtype) -> AndersonTestSettings:
    solver = setup_solver_settings(config, dtype)
    method = setup_anderson_method_settings(config)
    return AndersonTestSettings(
        maxiter=solver.maxiter,
        mix_guess=solver.mix_guess,
        damp=solver.damp,
        damp_init=method.damp_init,
        generations=method.generations,
        soft_start=method.soft_start,
        diagonal_offset=method.diagonal_offset,
        x_atol=solver.x_atol,
        x_atol_max=solver.x_atol_max,
        dtype=dtype,
        solver_settings=solver,
        method_settings=method,
        dynamic_damping=solver.dynamic_damping,
    )


def solve_common(
    guess: Tensor,
    settings: AndersonTestSettings,
    map_fn,
    *,
    problem: SCFProblem | None = None,
    eager: bool = False,
    **kwargs,
):
    """Exercise a method through the production shared controller."""
    return _solve_map(
        guess,
        settings.solver(),
        map_fn,
        problem=problem,
        strategy=SCFStrategy((SCFStage(AndersonMethod(settings.method())),)),
        eager=eager,
        **kwargs,
    )


def solve_physical(
    guess: Tensor,
    inputs,
    settings: AndersonTestSettings,
    *,
    representation=None,
    layout=None,
    eager: bool = False,
) -> SCFSolveResult:
    """Run a physical Anderson case through the common representation API."""
    from dxtb._src.scf.functional.problem import SCFRepresentation
    from dxtb._src.scf.functional.solver import solve_scf_problem

    selected = SCFRepresentation.CHARGE if representation is None else representation
    problem = SCFProblem(inputs, selected, layout)
    return solve_scf_problem(
        guess,
        problem,
        settings.solver(),
        SCFStrategy((SCFStage(AndersonMethod(settings.method())),)),
        eager=eager,
    )


def solve_potential(guess, inputs, layout, settings, *, eager=False):
    """Run a potential-representation Anderson test case."""
    from dxtb._src.scf.functional.problem import SCFRepresentation
    from dxtb._src.scf.functional.solver import solve_scf_problem

    return solve_scf_problem(
        guess,
        SCFProblem(inputs, SCFRepresentation.POTENTIAL, layout),
        settings.solver(),
        SCFStrategy((SCFStage(AndersonMethod(settings.method())),)),
        eager=eager,
    )


def solve_fock(guess, inputs, settings, *, eager=False):
    """Run a Fock-representation Anderson test case."""
    from dxtb._src.scf.functional.problem import SCFRepresentation
    from dxtb._src.scf.functional.solver import solve_scf_problem

    return solve_scf_problem(
        guess,
        SCFProblem(inputs, SCFRepresentation.FOCK),
        settings.solver(),
        SCFStrategy((SCFStage(AndersonMethod(settings.method())),)),
        eager=eager,
    )
