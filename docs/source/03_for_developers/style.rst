.. _dev_style:

Style
=====

This project uses the following style guidelines:

- Formatting: ``black`` and ``isort`` (line length 80)
- Type annotations & linting: ``mypy`` and ``pylint``
- Docstrings: ``numpydoc``

These checks are enforced automatically with ``pre-commit`` (and in the CI pipeline).

If you add new code to the project, please make sure to respect the high
modularity of the codebase and document your code properly.
