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
Config: Main
============

Main configuration of the calculation.

    The configuration is immutable. A changed setting is a new object, created
    with :func:`dataclasses.replace`, e.g., to change the SCF strategy::

    from dxtb.scf import strategy_preset

    cfg = dataclasses.replace(cfg, scf=dataclasses.replace(
        cfg.scf, strategy=strategy_preset("anderson")
    ))

User input (strings, lists) is converted by :meth:`Config.create`, the
constructor only accepts the final values. The configuration holds neither the
device and data type (they follow the tensors) nor the batch mode (it follows
the shape of the atomic numbers).
"""

from __future__ import annotations

import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path

import torch

from dxtb._src.constants import defaults, labels
from dxtb._src.typing import Any, PathLike, Self

from .integral import ConfigIntegrals
from .scf import ConfigSCF

__all__ = ["Config"]


@dataclass(frozen=True, kw_only=True)
class Config:
    """
    Configuration of the calculation.
    """

    file: PathLike | None = None
    """The input file or directory."""

    exclude: tuple[str, ...] = ()
    """The tight-binding components to exclude from the calculation."""

    method: int = defaults.METHOD
    """The xTB method to use."""

    grad: bool = False
    """Whether to compute the gradient."""

    max_element: int = defaults.MAX_ELEMENT
    """The maximum element number in the system."""

    # PyTorch

    anomaly: bool = False
    """Whether to run PyTorch in anomaly detection mode."""

    # configs

    ints: ConfigIntegrals = field(default_factory=ConfigIntegrals)
    """The integral configuration."""

    scf: ConfigSCF = field(default_factory=ConfigSCF)
    """The SCF configuration."""

    def __post_init__(self) -> None:
        if self.method not in (
            labels.GFN0_XTB,
            labels.GFN1_XTB,
            labels.GFN2_XTB,
        ):
            raise ValueError(f"Unknown xtb method '{self.method}'.")

        if not isinstance(self.exclude, tuple):
            raise TypeError(
                "The excluded components must be given as a tuple (use "
                "`Config.create` for strings and lists), but "
                f"'{type(self.exclude)}' was given."
            )

    @classmethod
    def create(
        cls,
        *,
        file: PathLike | None = None,
        exclude: str | list[str] | tuple[str, ...] = defaults.EXCLUDE,
        method: str | int = defaults.METHOD,
        grad: bool = False,
        # integrals
        int_cutoff: float = defaults.INTCUTOFF,
        int_driver: str | int = defaults.INTDRIVER,
        int_level: int = defaults.INTLEVEL,
        int_uplo: str = defaults.INTUPLO,
        int_algorithm: str | None = None,
        # PyTorch
        anomaly: bool = False,
        # SCF
        maxiter: int = defaults.MAXITER,
        strategy: str | object | None = None,
        differentiation: str | None = None,
        convergence: str = "standard",
        response: object | None = None,
        # Transitional input adapters are parsed into the strategy and
        # discarded; they are not fields of ConfigSCF.
        mixer: str | int | None = None,
        mix_guess: bool = defaults.MIX_GUESS,
        damp: float = defaults.DAMP,
        damp_init: float | None = None,
        damp_dynamic: bool = defaults.DAMP_DYNAMIC,
        damp_dynamic_factor: float = defaults.DAMP_DYNAMIC_FACTOR,
        damp_soft_start: bool | None = None,
        damp_generations: int | None = None,
        damp_diagonal_offset: float | None = None,
        guess: str | int = defaults.GUESS,
        scf_mode: str | int | None = None,
        scp_mode: str | int | None = None,
        x_atol: float | None = None,
        x_atol_max: float | None = None,
        f_atol: float | None = None,
        force_convergence: bool = False,
        fermi_etemp: float = defaults.FERMI_ETEMP,
        fermi_maxiter: int = defaults.FERMI_MAXITER,
        fermi_thresh: float | int | None = defaults.FERMI_THRESH,
        fermi_diff_order: int = defaults.FERMI_DIFF_ORDER,
        fermi_partition: str | int = defaults.FERMI_PARTITION,
        # misc
        max_element: int = defaults.MAX_ELEMENT,
    ) -> Self:
        """
        Create the configuration from user input (flat list of options).

        Strings are converted to integer labels, a single excluded component
        or a list of them to a tuple.

        Raises
        ------
        ValueError
            An option is unknown.
        TypeError
            An option has the wrong type.
        """
        if isinstance(method, str):
            if method.casefold() in labels.GFN0_XTB_STRS:
                method = labels.GFN0_XTB
            elif method.casefold() in labels.GFN1_XTB_STRS:
                method = labels.GFN1_XTB
            elif method.casefold() in labels.GFN2_XTB_STRS:
                method = labels.GFN2_XTB
            else:
                raise ValueError(f"Unknown xtb method '{method}'.")
        elif not isinstance(method, int):
            raise TypeError(
                "The method must be of type 'int' or 'str', but "
                f"'{type(method)}' was given."
            )

        if isinstance(file, list):
            file = tuple(file)
        if isinstance(exclude, str):
            exclude = (exclude,)

        scf = ConfigSCF.create(
            method=method,
            guess=guess,
            maxiter=maxiter,
            strategy=strategy,
            differentiation=differentiation,
            convergence=convergence,
            response=response,
            mixer=mixer,
            mix_guess=mix_guess,
            damp=damp,
            damp_init=damp_init,
            damp_dynamic=damp_dynamic,
            damp_dynamic_factor=damp_dynamic_factor,
            damp_soft_start=damp_soft_start,
            damp_generations=damp_generations,
            damp_diagonal_offset=damp_diagonal_offset,
            scf_mode=scf_mode,
            scp_mode=scp_mode,
            x_atol=x_atol,
            x_atol_max=x_atol_max,
            force_convergence=force_convergence,
            fermi_etemp=fermi_etemp,
            fermi_maxiter=fermi_maxiter,
            fermi_thresh=fermi_thresh,
            fermi_diff_order=fermi_diff_order,
            fermi_partition=fermi_partition,
        )

        if f_atol is not None:
            raise ValueError(
                "f_atol is retired; SCF convergence uses physical fixed-point "
                "and stationarity certification."
            )

        return cls(
            file=file,
            exclude=tuple(exclude),
            method=method,
            grad=grad,
            max_element=max_element,
            anomaly=anomaly,
            ints=ConfigIntegrals.create(
                level=int_level,
                cutoff=int_cutoff,
                driver=int_driver,
                uplo=int_uplo,
                algorithm=int_algorithm,
            ),
            scf=scf,
        )

    @classmethod
    def from_args(cls, args: Namespace) -> Self:
        """
        Create a configuration from command-line arguments.

        Parameters
        ----------
        args : Namespace
            The parsed command-line arguments.

        Returns
        -------
        Self
            The configuration object.
        """
        return cls.create(
            # general
            file=args.file,
            exclude=args.exclude,
            method=args.method,
            grad=args.grad,
            # integrals
            int_cutoff=args.int_cutoff,
            int_driver=args.int_driver,
            int_level=args.int_level,
            int_uplo=args.int_uplo,
            int_algorithm=getattr(args, "int_algorithm", None),
            # PyTorch
            anomaly=args.detect_anomaly,
            # SCF
            maxiter=args.maxiter,
            strategy=getattr(args, "strategy", "anderson"),
            differentiation=getattr(args, "differentiation", "unrolled"),
            convergence=getattr(args, "convergence", "standard"),
            damp=args.damp,
            guess=args.guess,
            x_atol=args.xtol,
            x_atol_max=args.xtol_max,
            force_convergence=args.force_convergence,
            # SCF: Fermi
            fermi_etemp=args.fermi_etemp,
            fermi_maxiter=args.fermi_maxiter,
            fermi_thresh=args.fermi_thresh,
            fermi_diff_order=args.fermi_diff_order,
            fermi_partition=args.fermi_partition,
        )

    @classmethod
    def from_json(cls, path: PathLike) -> Self:
        """
        Create a configuration from a JSON file.

        Parameters
        ----------
        path : PathLike
            The path to the JSON file.

        Returns
        -------
        Self
            The configuration object.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File '{path}' does not exist.")

        # pylint: disable=import-outside-toplevel
        import json

        with open(path, encoding="utf-8") as json_file:
            cfg = json.loads(json_file.read())

        return cls.from_dict(cfg)

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> Self:
        """
        Create a configuration from a dictionary.

        Parameters
        ----------
        cfg : dict[str, Any]
            The configuration dictionary.

        Returns
        -------
        Self
            The configuration object.
        """
        # TODO: More sophisticated validation
        return cls.create(**cfg)

    def info(self) -> dict[str, dict[str, Any]]:
        """
        Return a dictionary with the configuration information.

        Returns
        -------
        dict[str, dict[str, Any]]
            The configuration information.
        """
        return {
            "Calculation Configuration": {
                "Program Call": " ".join(sys.argv),
                "Input File(s)": self.file,
                "Method": labels.GFN_XTB_MAP[self.method],
                "Excluded": False if len(self.exclude) == 0 else self.exclude,
                "Gradient": self.grad,
                "Integral driver": labels.INTDRIVER_MAP[self.ints.driver],
            },
            **self.scf.info(),
        }

    def to_json(self, path: PathLike | None = None) -> str:
        """
        Serialize the configuration to a JSON-formatted string.

        Returns:
            str: A JSON-formatted string representing the configuration.
        """
        # pylint: disable=import-outside-toplevel
        import json

        config_info = self.info()

        def serialize(value):
            if isinstance(value, torch.device) or isinstance(
                value, torch.dtype
            ):
                return str(value)
            elif isinstance(value, list):
                # Recursively serialize lists
                return [serialize(v) for v in value]
            elif isinstance(value, dict):
                # Recursively serialize dicts
                return {k: serialize(v) for k, v in value.items()}
            else:
                return value

        # Serialize the entire configuration info to JSON
        serialized_info = {k: serialize(v) for k, v in config_info.items()}

        # Convert the dictionary to a JSON string
        json_string = json.dumps(serialized_info, indent=4)

        if path is not None:
            path = Path(path)
            if path.exists():
                path.unlink()

            with open(path, "w", encoding="utf-8") as json_file:
                json_file.write(json_string)

        return json_string

    def __str__(self) -> str:  # pragma: no cover
        info = self.info()["SCF Options"]
        info_str = ", ".join(f"{key}={value}" for key, value in info.items())
        return f"{self.__class__.__name__}({info_str})"

    def __repr__(self) -> str:  # pragma: no cover
        return str(self)
