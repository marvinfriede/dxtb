.. _about-related:

Related Works
=============

xTB
---

The main xTB Fortran code can be found at

`xtb <https://github.com/grimme-lab/xtb>`__:
   Semiempirical Extended Tight-Binding Program Package

For an alternative implementation with a strong focus on library use, interfacing and interoperability, also check out

`tblite <https://tblite.readthedocs.io>`__:
  Light-weight tight-binding framework implemented in Fortran with Python bindings


PyTorch Autodiff Projects
-------------------------

Related PyTorch-based libraries in the `tad-mctc <https://github.com/tad-mctc>`__ ecosystem (note that DFT-D3/D4 dispersion in *dxtb* is now provided by the bundled ``tad_dftd`` package, and ``tad-libcint`` is an optional Linux integral backend):

`tad-mctc <https://github.com/tad-mctc/tad-mctc>`__:
  PyTorch Autodiff Utility Library

`tad-multicharge <https://github.com/tad-mctc/tad-multicharge>`__:
  PyTorch Autodiff Implementation of Classical Charge Models.

`tad-dftd3 <https://github.com/dftd3/tad-dftd3>`__:
  PyTorch Autodiff Implementation of DFT-D3

`tad-dftd4 <https://github.com/dftd4/tad-dftd4>`__:
  PyTorch Autodiff Implementation of DFT-D4

`tad-libcint <https://github.com/tad-mctc/tad-libcint>`__:
  PyTorch-based C-Interface for Integrals.
