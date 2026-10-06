# Restructuring plan

Planning documents for the dxtb restructuring (Track 0 item T0.1, step 4).
They live outside the Sphinx sources on purpose: they describe work, not the
API.

| File | Content |
| --- | --- |
| `00-overview.md` | Goals, principles, track map, dependencies, releases |
| `01-T0-baseline.md` | Track 0: baseline before any refactor |
| `02-TB-evaluation-api.md` | Track B: model/system/result, results instead of caches, fields as inputs |
| `03-TC-node-migration.md` | Track C: `Node` containers, retiring `TensorLike` |
| `04-TE-differentiable-core.md` | Track E: higher-order and forward-mode derivatives, `vmap` batching |
| `05-TF-machine-learning.md` | Track F: parameter training, ML terms |
| `06-TD-performance.md` | Track D: compiling and performance |
| `T0-baseline-report.md` | Track 0 deliverable (T0.11): the recorded baseline |

The design review referenced by the plan is not in the repository.

The code references in the plan point to commit `46af7bc`. `main` has moved
on since (#268 CCA orbital ordering, #269 Fermi smearing, #270 PyTorch
multipole integrals); the baseline report records the commit it was taken at and lists the
starting points that have changed (section 9a).

## Track 0 artefacts

| Path | Package | Content |
| --- | --- | --- |
| `constraints/baseline.txt` | T0.1 | Versions the baseline was recorded with |
| `.github/workflows/baseline.yaml` | T0.1 | CI job installing from the constraints (`slow`/`large` nightly) |
| `test/test_baseline/molecules.py`, `compute.py`, `refdata.py`, `generate*.py`, `reference/` | T0.2 | Reference data (dxtb and tblite) and comparison utility |
| `test/test_baseline/test_reference.py`, `tolerances.py` | T0.2 | Reproduction test and tolerances |
| `test/test_baseline/matrix.py`, `matrix_refs.py`, `test_derivatives.py`, `scans.py` | T0.3 | Derivative status matrix |
| `test/test_baseline/components.py`, `test_components.py` | T0.4 | Component-level derivative checks |
| `test/test_calculator/test_cache/test_stale.py` | T0.5 | Cache findings |
| `docs/source/04_help/known_issues.rst` | T0.6 | Known issues page |
| `test/test_baseline/transforms.py` | T0.7 | Transform status |
| `benchmarks/baseline/workloads.py` | T0.8 | Performance workloads W1-W5 |
| `test/test_baseline/params.py` | T0.9 | Parameter-gradient coverage |
| `test/conftest.py` (`LAYER_MARKERS`), `test/test_baseline/inventory.py` | T0.10 | Layer markers and inventory |
| `test/test_baseline/status/` | T0.3-T0.9 | Recorded status (`python -m test.test_baseline.status ...`) |
| `test/test_baseline/tables.py` | T0.11 | Tables of the report |

Regenerating (repository root, environment from the constraints file; tad-dftd3
and tad-dftd4 are installed with `pip install --no-deps` until they are
released for tad-mctc 0.9, see `dxtb/_src/mctc_shim.py`). Use one thread per
process (`OMP_NUM_THREADS=1`) when several run in parallel (`pytest -n`,
the parallel status runs): the default of four threads each oversubscribes four
cores and makes the runs about ten times slower.

```sh
python -m test.test_baseline.generate            # T0.2 dxtb references
python -m test.test_baseline.generate_tblite     # T0.2 tblite references (pip install tblite)
python -m test.test_baseline.matrix_refs         # T0.3 finite difference references
python -m test.test_baseline.status matrix       # T0.3 status (about 35 min, 4 jobs)
python -m test.test_baseline.status components   # T0.4 status
python -m test.test_baseline.scans               # T0.3 step size and convergence scans
python -m test.test_baseline.transforms          # T0.7
python -m test.test_baseline.params              # T0.9
python benchmarks/baseline/workloads.py W3 --nsys 12 --threads 1   # T0.8, one workload per call;
                                                                   # W2: --nconf 8, W3 GFN2: --nsys 10
python -m test.test_baseline.tables              # tables for the report
```
