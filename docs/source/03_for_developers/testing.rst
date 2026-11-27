.. _dev_testing:

Testing
=======

pytest
------

For rapid parallel testing, run ``pytest`` with ``pytest-xdist`` and exclude large and slow tests (setting ``OMP_NUM_THREADS=1`` per worker):

.. code-block:: shell

    OMP_NUM_THREADS=1 pytest -vv -n logical --dist worksteal -m "not large and not slow"


tox
---

For testing across Python and PyTorch environments, use ``tox``.

.. code-block:: shell

    tox

By default, ``tox`` runs tests in parallel with randomized order and skips ``large``, ``slow``, and ``baseline`` markers. Pass arguments after ``--`` to override the default ``pytest`` invocation:

.. code-block:: shell

    tox -- test

To test specific environments, use the ``-e`` flag (including the ``-libcint`` or ``-nolibcint`` factor):

.. code-block:: shell

    tox -e py310-torch2141-libcint

Development dependency setup
----------------------------

The development/test environments pin ``tad-mctc==0.11.3``,
``tad-multicharge==0.8.0``, and ``tad-libcint==0.4.0``. Dispersion (D3, D4,
and D4SC) is provided by the in-tree unified ``tad_dftd`` package
(``src/tad_dftd/``), retiring the external ``tad-dftd3`` / ``tad-dftd4``
wheels and the legacy compatibility shim.
