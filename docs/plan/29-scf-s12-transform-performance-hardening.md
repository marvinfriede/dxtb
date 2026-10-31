# SCF S12 — transform and performance hardening

## Scope and synchronization

S12 starts at accepted S11 `eded7ed07ec8a680788511d8d1ac62d79baca36c`.
The coordination record initially named W2c11 `abdb4562a0607475e775072277b63f82501363a0`.
During S12, the live general-refactor remote had advanced to
`9653ecdca42ea7c276a44f447497ef4bd53329fd`. I inspected its four intervening
commits and the handoff marked the boundary safe, then synchronized once. The
range-diff mapped all 69 accepted S1–S11 commits; 63 were identical. The six
non-identical patch resolutions preserved existing target-base energy helper
and no-worker behavior, applied the common-controller migration, retained
S8.1b's independent entropy oracle, and propagated S11 mixer diagnostics to
the serial batch root path. No unexplained SCF change remains. The
general-refactor worktree was not changed.
The S11 acceptance record is retained as-is. S12 does not reopen response
coordinates, certification, finite-temperature policy, or the S10 research
decision.

The post-S11 gate was rerun before S12 changes using the recorded selection:

```bash
OMP_NUM_THREADS=1 python -m pytest -q -n logical --dist worksteal \
  test/test_scf/test_functional_*.py \
  test/test_calculator/test_e6a0_system_vmap.py --tb=short
```

Before S12 edits, the accepted-S11 selection collected 658 tests: 657 passed
and the only failure was the accepted Anderson FULL gradient discrepancy
(`2.5007065862503453e-11` versus `2e-13`). No test tolerance or Anderson
arithmetic was changed in S12. After synchronization and the solver
merge-boundary corrections, the same selection collected 666 tests and all
666 passed. This includes the published Anderson correction. A focused
potential/Fock/temperature/S12/S1 rerun passed 115 tests. The W2c11 batch,
mapped-implicit, invariant-spectral, and W2c6 acceptance files passed 186
tests. Exact post-sync results are also recorded in the handoff.

The PyTorch 2.8 compatibility slice covering the S12 checks, Broyden/CDIIS
vmap/JVP/fixed-trip paths, and mapped implicit response passed 14 tests.

## Result and residual semantics

`SCFSolveResult.residual`, `iterations`, and `diagnostics.strategy` describe
the primal SCF controller and trajectory. In implicit mode, an internal
alternate response representation prepares a separate attached root. Its
certification belongs to the response preparation and does not redefine the
reported primal residual or iteration count. The finalizer now restores those
primal diagnostics after attaching the response root; a regression compares
them against the original primal outcome.

## Fixed-trip and transform support

The retained capability contract is:

| Method | Fixed-trip | Representation scope | S12 evidence |
|---|---:|---|---|
| Linear | yes | charge, potential, Fock | Existing solver and physical-map transform tests |
| Anderson | yes | charge, potential, Fock | Existing fixed-trip, mixed-inactivity, JVP/jacfwd/vmap tests |
| Broyden | yes | charge, potential, Fock | Existing fixed-trip, state-freeze, JVP/jacfwd/vmap tests |
| CDIIS | yes | Fock only | Existing Fock-controller vmap/JVP/jacfwd and state-freeze tests |
| SOSCF | no | Fock only, eager | Fixed-trip rejection remains before physical work |
| Dynamic multistage strategy | no | eager | Tensor-dependent switching remains host controlled |

The single-stage fixed-trip tests retain lane-local active masks and exact
inactive-state freezing. In particular, Anderson histories, Broyden secant
and diagnostic tensors, and CDIIS histories/diagnostics are fixed-size PyTrees
with no Python list history. Their leaves are registered with the project's
`tad_mctc.tree.Node` mechanism and round-trip through PyTorch's tree flatten /
unflatten machinery with dtype, shape, device, and values preserved. Static
method settings and the Python loop index remain static; transform-capable
method code does not extract tensor values with `.item()` or `.tolist()`.

CDIIS remains Fock-only. SOSCF retains its deliberate eager-only contract for
occupation-sector analysis, PCG stopping, energy line search, and block
spectral checks. Dynamic stage switching is also eager-only. A compact-to-Fock
switch changes state shape, so it cannot be represented by a lane-wise tensor
selection without a separate padded representation design. No attempt was
made to disguise either boundary as transform-safe.

Existing controller and public-System tests cover fixed-trip/eager behavior,
serial/vmap lane values, per-lane iterations/status, position gradients,
mixed convergence, and inactive transitions. S12 adds direct state PyTree and
primal-residual contract tests. The H2 conformer benchmark below checks
distinct geometries and serial/vmap gradients at N=1, 2, 4, and 8. Padded
mixed-system safety continues to use the private E6b padded-controller tests;
this package does not add public mixed-System routing.

## W2c11 response and worker boundary

W2c11 mapped-root, response-failure, streaming, shared-Model attribution, and
chunk-size behavior remain owned by the W2c11 tests. S12 acceptance uses one
worker where the API exposes worker count. No Python thread, process, executor,
or worker-pool implementation was added. The W2c11 worker tests remain
regressions, but process scheduling is not used as an S12 performance claim.

## Compile characterization

The reproducible probe is `benchmarks/scf_s12_compile.py`. It uses
`torch.compile(..., backend="eager")`, which checks Dynamo capture and graph
breaks, not optimized backend performance. The H2/GFN1 raw charge, potential,
and Fock maps preserve values and their tested ordinary reverse and JVP
derivatives. They split into 19/21/19 graphs with 18/20/18 graph breaks. The
breaks localize to existing scatter output-size inference at `idx.max().item()`.
The S12 compile probe does not alter this raw-map implementation.

Linear, Anderson, Broyden, and CDIIS fixed-shape Fock transition kernels each
capture as one graph with no graph breaks and preserve proposal/state values.
A tensor-only affine controller loop captures into one graph for static trip
counts 4, 8, and 16. Capture/first-call time grows with the unrolled trip count;
this is not evidence that compiling the complete physical SCF is beneficial.
The added whole-solver probe compiled actual H2/GFN1 fixed-trip Linear solves
at budgets 4, 8, and 16, and Broyden/CDIIS at budget 4, with value parity.
Each produced 29 graphs and 28 graph breaks; explain plus first-call latency
was approximately 24–28 seconds per run. Anderson's eager four-trip
reference itself failed with a singular DIIS solve, so compilation was not
attempted for that case. The probe records graph counts and latency
separately. It does not treat the `backend="eager"` result as an
optimized-compiler speedup.
The whole-solver compile check is value-only; derivative parity was tested
for compiled raw maps, not for the complete compiled solver trajectory.

An active-smearing compile regression explicitly disables W2c11's detached
adaptive `_matrix_fermi_eager` path and checks that the compiled raw Fock map
still matches the static transform-safe result. Detached eager adaptive or
spectral Fermi evaluation remains separate from fixed-trip/compiled paths.
Compiled `torch.func.grad` over the raw-map wrapper failed in the installed
PyTorch composition at scatter helper FakeTensor metadata handling; this is
recorded as an unsupported composition. Ordinary reverse autograd and JVP of
the compiled callable passed the recorded probe. No private Dynamo API or
custom control-flow operator was added to production.

## Performance characterization

Benchmark scripts pin CPU PyTorch intra-op and inter-op threads to one and
document `OMP_NUM_THREADS=1`; all use one worker. JSON stores Python, PyTorch,
platform, dtype through the case, settings, map counts, timings, RSS, and source
revision. `scf_s12_methods.py` compares Linear/Anderson/Broyden on the potential
representation and CDIIS on Fock for H2O, NO2, and C4H5NCS (GFN2, 0 K,
float64, 300-map budget, `1e-7` target). H2O certified in 11–22 physical maps;
NO2 in 16–23 for the convergent methods; C4H5NCS in 22–26. Linear did not
certify NO2 or C4H5NCS within 300 maps. The other methods reached matching
physical energies to reported precision. At these settings CDIIS used more
wall time than Anderson/Broyden despite comparable map counts. This is
characterization, not a default selection.

For H2O, fixed-trip execution used 301 physical maps for a 300-map budget,
versus 11–22 maps for eager early termination, and took 3.4–4.5 s versus
0.09–0.24 s eager. The static budget tax is visible and no tolerance was
changed to reduce it.

`scf_s12_vmap.py` evaluates distinct H2/GFN1 conformers through the same
single-System fixed-trip function, then compares serial and `vmap` values and
position gradients. At N=1/2/4/8, maximum energy differences were at most
`3.4e-16` Eh and maximum gradient differences at most `8.1e-16` in the
recorded PyTorch 2.14.1 run. In this tiny workload, vmap time stayed near
0.10–0.13 s while serial time grew from about 0.06 to 0.51 s. This result is
specific to H2/GFN1 and should not be extrapolated to larger systems or memory
scaling. W2c11 streaming remains the memory-bounded derivative route.

## Limitations and S13 handoff

S12 makes no default method, strategy, switch threshold, response
representation, history-size, or compile-mode decision. It does not add
SOSCF fixed-trip support, dynamic multistage vmap, representation-changing
per-lane switching, public batch routing, or compiled implicit higher-order
support. The pinned implicit primitive's compile/forward/higher-order limits
remain the S11/W2c11 boundaries. CPU transform compatibility is validated on
PyTorch 2.14.1 and the focused S12 state/capability/active-smearing compile
checks also run on PyTorch 2.8.0; the wider S11/SOSCF regression results are
recorded with the package handoff.

## Independent review

Requested read-only review: Luna / high (runtime model identity is not
exposed). The review found no blocker or major finding. Its minor evidence note
is reflected above: full physical solver compilation has value parity only,
while compiled raw maps carry the reverse/JVP checks. Anderson's budget-4
whole-solver compile attempt is skipped because its eager reference hits a
singular DIIS solve.

S12 is evidence for S13 benchmark-driven choices, not a selection of those
choices. No spectral mathematics, S9/S10 decision, S2 certification, public
configuration, or Python-level parallelism was changed.
