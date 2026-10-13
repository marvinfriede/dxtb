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
``tad-multicharge==0.7.0``. The currently required legacy releases
``tad-dftd3==0.6.0`` and ``tad-dftd4==0.8.0`` are installed with
``pip install --no-deps`` in the restructuring development/test environments,
including tox and the pinned Track-0 baseline CI environment, so
``tad-mctc==0.9.1`` remains authoritative despite conflicting published
dependency metadata. Tox smoke-imports dxtb and both dispersion packages
before running tests. This is a controlled development/baseline-testing
workaround, not the ordinary installation or supported release dependency
model. Do not publish while an ordinary installation still depends on this
external no-deps workaround; restore normal runtime dependencies and remove
``mctc_shim.py``, ``ncoord/legacy.py`` and the workaround after compatible
upstream releases are available.
