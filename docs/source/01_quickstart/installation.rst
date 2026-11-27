.. _quickstart-installation:

Installation
============

pip
---

.. image:: https://img.shields.io/pypi/v/dxtb
    :target: https://pypi.org/project/dxtb/
    :alt: PyPI Version

.. image:: https://img.shields.io/pypi/dm/dxtb?color=orange
    :target: https://pypi.org/project/dxtb/
    :alt: PyPI Downloads

*dxtb* can easily be installed with ``pip``.

.. code-block:: shell

    pip install dxtb[libcint]

Installing the ``libcint`` interface (``tad-libcint``) is recommended on Linux, as it is faster than the pure PyTorch implementation on CPU; ``tad-libcint`` is currently only available on Linux.
The pure PyTorch integral driver works on all platforms across CPU and CUDA and provides overlap, dipole, and quadrupole integrals (supporting GFN0-xTB, GFN1-xTB, and GFN2-xTB) with derivatives of any order.


conda
-----

.. image:: https://img.shields.io/conda/vn/conda-forge/dxtb.svg
   :target: https://anaconda.org/conda-forge/dxtb
   :alt: Conda Version

.. image:: https://img.shields.io/conda/dn/conda-forge/dxtb?style=flat&color=orange
   :target: https://anaconda.org/conda-forge/dxtb
   :alt: Conda Downloads

*dxtb* is also available on `conda <https://conda.io/>`__ from the *conda-forge* channel.

.. code-block:: shell

    mamba install dxtb


The optional libcint interface (not on conda) can be installed via ``pip install tad-libcint``.

For Windows, *dxtb* is not available via conda, because PyTorch itself is not registered in the conda-forge channel.


From source
-----------

This project is hosted on GitHub at `grimme-lab/dxtb <https://github.com/grimme-lab/dxtb>`__.
Obtain the source by cloning the repository with

.. code-block:: shell

    git clone https://github.com/grimme-lab/dxtb
    cd dxtb

We recommend using a `conda <https://conda.io/>`__ environment to install the package.
You can setup the environment manager using a `miniforge <https://github.com/conda-forge/miniforge>`__ installer.
Install the required dependencies from the conda-forge channel.

.. code-block:: shell

    mamba env create -n torch -f environment.yaml
    mamba activate torch

Install this project with ``pip`` in the environment

.. code-block:: shell

    pip install .


Without pip
-----------

If you want to install the package without pip, start by cloning the repository.

.. code-block:: shell

    DEST=/opt/software
    git clone https://github.com/grimme-lab/dxtb $DEST/dxtb

Next, add ``<path to dxtb>/dxtb/src`` to your ``$PYTHONPATH`` environment variable.
For the command line interface, add ``<path to dxtb>/dxtb/bin`` to your ``$PATH`` environment variable.

.. code-block:: shell

    export PYTHONPATH=$PYTHONPATH:$DEST/dxtb/src
    export PATH=$PATH:$DEST/dxtb/bin


Dependencies
------------

The following runtime dependencies are required:

- `h5py <https://www.h5py.org/>`__
- `numpy <https://numpy.org/>`__
- `pydantic <https://docs.pydantic.dev/>`__ (``>=2.0.0``)
- `scipy <https://www.scipy.org/>`__
- `tad-mctc <https://github.com/tad-mctc/tad-mctc>`__
- `tad-multicharge <https://github.com/tad-mctc/tad-multicharge>`__
- `tomli <https://github.com/hukkin/tomli>`__ / `tomli-w <https://github.com/hukkin/tomli-w>`__
- `torch <https://pytorch.org/>`__ (``>=2.7.0,<3``)
- `typing-extensions <https://typing-extensions.readthedocs.io/>`__

Dispersion corrections (D3/D4/D4SC) are provided by the bundled ``tad_dftd`` package. Optional extras include:

- `tad-libcint <https://github.com/tad-mctc/tad-libcint>`__ (``dxtb[libcint]``, Linux only)
- `ase <https://wiki.fysik.dtu.dk/ase/>`__ (``dxtb[ase]``)

For running the test suite, install ``pip install -e ".[dev]"`` (including `pytest <https://docs.pytest.org/>`__, `pytest-xdist <https://pytest-xdist.readthedocs.io/>`__, `pyscf <https://pyscf.org/>`__, `tblite <https://tblite.readthedocs.io/>`__, and `tox <https://tox.wiki/>`__).
