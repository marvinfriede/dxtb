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
``tad-multicharge==0.8.0``, and ``tad-libcint==0.4.0``. Dispersion (D3, D4,
and D4SC) is provided by the in-tree unified ``tad_dftd`` package
(``src/tad_dftd/``), retiring the external ``tad-dftd3`` / ``tad-dftd4``
wheels and the legacy compatibility shim.
