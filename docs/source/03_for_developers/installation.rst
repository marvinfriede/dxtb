.. _dev_installation:

Installation
============

Obtain the source code from GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__ by cloning the repository with

.. code-block:: shell

    git clone https://github.com/grimme-lab/dxtb
    cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `mambaforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code-block:: shell

    mamba env create -n torch -f environment.yaml
    mamba activate torch

For development, additionally install the development tools in your environment, and use the ``-e`` option to install the package in editable mode.

.. code-block:: shell

    pip install -e .[dev]

The refactor development stack uses ``tad-mctc==0.11.3``,
``tad-multicharge==0.8.0``, and (when testing libcint) ``tad-libcint==0.4.0``.
Install the official D3/D4 wheels after dxtb because their currently published
metadata pins older, incompatible utility packages. The local dxtb compatibility
shim must load before importing either dispersion package.

.. code-block:: shell

    python -m pip install -e ".[dev,libcint]"
    python -m pip install --no-deps --index-url https://pypi.org/simple tad-dftd3==0.7.0 tad-dftd4==0.8.0
    python -c "import dxtb; import tad_dftd3, tad_dftd4; print(dxtb.__version__)"

This is a controlled development/test installation only. The published D3/D4
metadata conflicts are expected in ``pip check``; no other dependency conflict
is accepted. A resolver-clean public release remains blocked until compatible
official dispersion releases are available. The documented command order is
required: importing D3/D4 before dxtb bypasses the temporary compatibility
shim.

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code-block:: shell

    pre-commit install
