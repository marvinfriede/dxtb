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

The development stack uses ``tad-mctc==0.11.3``, ``tad-multicharge==0.8.0``,
and (when testing libcint) ``tad-libcint==0.4.0``. Unified DFT-D dispersion
(``tad_dftd``) is bundled directly in the repository under ``src/tad_dftd/``.

.. code-block:: shell

    python -m pip install -e ".[dev,libcint]"
    python -c "import dxtb, tad_dftd; print(dxtb.__version__)"

The pre-commit hooks are initialized by running the following command in the root of the repository.

.. code-block:: shell

    pre-commit install
