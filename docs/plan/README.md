# Restructuring plan

Planning documents for the dxtb restructuring (Track 0 item T0.1, step 4).
They live outside the Sphinx sources on purpose: they describe work, not the
API.

| File | Content |
| --- | --- |
| `00-overview.md` | Goals, principles, track map, dependencies, releases |
| `01-T0-baseline.md` | Track 0: baseline before any refactor |
| `T0-baseline-report.md` | Track 0 deliverable (T0.11): the recorded baseline |

The track files `02-TB-evaluation-api.md` to `06-TD-performance.md` and the
design review are not in the repository yet.

The code references in the plan point to commit `46af7bc`. `main` has moved
on since (#268 CCA orbital ordering, #269 Fermi smearing, #270 PyTorch
multipole integrals); the baseline report records the commit it was taken at.
