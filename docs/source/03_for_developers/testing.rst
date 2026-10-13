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

The restructuring branch currently pins ``tad-mctc==0.9.1`` and
``tad-multicharge==0.7.0``. The compatible ``tad-dftd3`` and ``tad-dftd4``
releases are installed by tox with ``pip install --no-deps`` because their
published dependency pins conflict with that ``tad-mctc`` version. Tox then
smoke-imports dxtb and both dispersion packages before running tests. This is
an isolated development/test setup, not a supported ordinary-install or
release dependency set. Do not publish while an ordinary installation still
depends on this external no-deps workaround; restore normal runtime
dependencies and remove ``mctc_shim.py``, ``ncoord/legacy.py`` and the
workaround after compatible upstream releases are available.
