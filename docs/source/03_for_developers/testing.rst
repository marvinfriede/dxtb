.. _dev_testing:

Testing
=======

pytest
------

For rapid testing, simply run `pytest`, but exclude large tests.

.. code-block:: shell

    pytest -vv -m "not large"


tox
---

For testing all Python environments, use `tox`.

.. code-block:: shell

    tox

Note that this randomizes the order of tests but skips "large" tests. To modify this behavior, `tox` has to skip the optional *posargs*.

.. code-block:: shell

    tox -- test

To test specific environments, use the `-e` flag.

.. code-block:: shell

     tox -e py310-torch2141

Development dependency setup
----------------------------

The development/test environments pin ``tad-mctc==0.11.3``,
``tad-multicharge==0.8.0``, and ``tad-libcint==0.4.0``. The official
``tad-dftd3==0.7.0`` release
declares ``tad-mctc==0.8.0``; ``tad-dftd4==0.8.0`` also requires older
packages. After installing dxtb's pinned dependencies, install them from
PyPI with ``python -m pip install --no-deps --index-url
https://pypi.org/simple tad-dftd3==0.7.0 tad-dftd4==0.8.0``. The local
``dxtb._src.mctc_shim`` restores the removed names used by those releases.
This controlled environment setup does not resolve the published metadata
conflicts: ``pip check`` reports their declared older mctc/multicharge and
NumPy constraints; these specific development conflicts are accepted.
The shim keeps the dxtb 0.11.3 API authoritative and must remain until
compatible dispersion releases are available.
