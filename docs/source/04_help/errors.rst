.. _help_errors:

Common Errors
=============

The following comprises a list of common errors that might occur during
development and their solutions. If an error occurs frequently, raise a
custom error in the respective code.


Non-finite SCF method updates
-----------------------------

The functional Broyden method checks secant updates before storing them. A
zero or ill-conditioned secant denominator clears the bounded correction
history and restarts from the scalar initial inverse approximation. The SCF
controller then continues with its ordinary physical map and certification
rules. This is a method-history restart, not an SCF convergence or Fermi
failure.
