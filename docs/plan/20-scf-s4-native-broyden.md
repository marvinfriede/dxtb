# SCF S4 — native functional limited-memory Broyden

## Scope and synchronization

S4 starts from S3 commit `56b6fdaee04614290e52670cd89f69cff9ca869b` and
synchronizes once to published general-refactor commit
`7d71e930e5bdbd52a252bf23d21803b94d2695f4`. The S3 history range-diff matched
all 27 prior commits one-for-one. Published changes after `124a464` were W2c8
benchmark/results/documentation only. The active uncommitted W2c9 worktree was
not inspected or modified.

S4 implements Broyden's first inverse method as an `SCFMethod`. It does not add
method switching, CDIIS, SOSCF, public mixer settings, or broad xitorch removal.
The common S3 controller continues to own the iteration budget, damping,
convergence, physical certification, and finalization.

## Sign convention and update

The functional map is `G(x)` and the controller residual is
`r(x) = G(x) - x`. The rootfinder convention is `F(x) = x - G(x) = -r(x)`.
The legacy Broyden configuration used `alpha=-0.5`; its inverse initialization
is `H0=-alpha I=0.5 I`. Therefore the initial undamped proposal is
`-H0 F = 0.5 r`, and controller damping `d` gives
`x_next = x + d * 0.5 * (G(x)-x)`. A regression checks the default `d=0.5`
factor of `0.25` and the synthetic multi-step update is compared with the
vendored reference `BroydenFirst` only in tests.

For later observations, with `s=x_k-x_(k-1)` and `y=F_k-F_(k-1)`, the method
uses:

```text
v = H.T @ s
c = s - H @ y
d = v / (y.T @ v)
H_new = H + c @ d.T
```

The inverse is represented only as `h0 I + sum(c_i d_i.T)`. Production never
allocates an inverse in flattened SCF-variable space. `BroydenState` is frozen,
contains tensor leaves only, and stores fixed shapes `(rank, n)` for both
correction arrays plus a tensor validity mask/count and restart diagnostics.

The default internal rank is 8. The correction buffer restarts when full and
then uses the current secant pair against `H0`. A secant is rejected when its
vectors or intermediate actions are zero/non-finite, the denominator is zero or
non-finite, or `abs(y.T v)/(||y|| ||v||)` is below the dtype-aware default
`sqrt(eps)`. Division is performed only after selecting a safe denominator.
Invalid updates clear correction history and retain the current observation.
Non-finite corrected steps clear corrections and retry with `H0`; physical
terminal failures remain controller-owned and do not advance method state.
There is no line search, trust-radius clipping, Broyden-specific damping, or
`damp_init` interpretation. Only controller-supplied `context.damp` is used.

## Integration and supported routes

`_prepare_solver()` now creates `BroydenMethod` and a one-stage `SCFStrategy`,
using the same `SCFSolverSettings` and shared controller as Linear and
Anderson. The method uses `context.proposal_x`, so structural projection is
applied before secant history is recorded. `mix_guess` and zero-budget behavior
remain common controller behavior. Broyden performs no map, certification,
energy, Fermi, or finalization work itself.

Eligible single-System FULL and implicit routes now honor configured Broyden
for charge, potential, and Fock representations. Explicit legacy engine
arguments, unsupported interaction extensions, and batched routes retain the
legacy boundary. The CLI no longer silently rewrites ordinary FULL Broyden to
Anderson; its explicit Anderson override for analytical IR/Raman/dipole
calculations remains. `strict` does not change the native update. `f_atol`
remains a legacy option and does not override S2 physical certification.

Fock was kept enabled after direct rank-8 GFN2 checks converged for H2O, NO2,
and C4H5NCS. Their compact Fock dimensions were 36, 144, and 1444; the two
history arrays use 4,608, 18,432, and 184,832 bytes respectively in float64.
This characterizes the tested domain, not an upper bound for very large
systems.

The charge and potential compact dimensions for those same systems were 54,
54, and 216, giving rank-8 two-array history storage of 6,912, 6,912, and
27,648 bytes in float64. In all representations storage is exactly
`2 * max_rank * n * element_size` and grows linearly with rank and flattened
state dimension.

| Fock case | Iterations | Certified | Residual L2 / max | Energy + free energy (Eh) |
| --- | ---: | --- | --- | ---: |
| H2O / GFN2, 300 K | 19 | yes | `6.11e-7 / 2.12e-7` | -5.104158799410 |
| NO2 / GFN2, doublet, 300 K | 20 | yes | `6.36e-7 / 2.81e-7` | -10.953075803272 |
| C4H5NCS / GFN2, 300 K | 23 | yes | `7.88e-7 / 2.10e-7` | -19.694059708018 |

Implicit attachment receives only the certified physical root and the exact
raw-map revalidation. Broyden history is absent from the prepared root and
response. A regression compares charge, potential, and Fock implicit responses
against Anderson and a field finite difference. A guard also replaces the
legacy implicit root class with a raising stub for the direct native prepared
route.

## Rank characterization

All measurements used GFN2, potential representation, 300 K where applicable,
`maxiter=120`, `x_atol=x_atol_max=1e-8`, and independent runs for each rank.
The rank-32 run did not hit capacity in the recorded solves and serves as the
effectively unrestricted reference for these trajectories.

| System | rank 4 iterations | rank 8 | rank 16 | rank 32 | Certified energy (Eh) |
| --- | ---: | ---: | ---: | ---: | ---: |
| H2O | 25 | 25 | 25 | 25 | -5.10415879940965 |
| NO2, doublet | 27 | 27 | 27 | 27 | -10.95307580327207 |
| C4H5NCS | 33 | 30 | 29 | 29 | -19.69405970801917 |

All ranks certified each case. Rank 8 is selected internally because it matched
the rank-32 iterations on H2O and NO2 and was only one iteration slower on
C4H5NCS; rank 4 was three to four iterations slower on the larger system. Rank
16 did not improve on rank 8 for the small cases and improved C4H5NCS by one
iteration. These tests do not establish a public rank choice.

The same rank-8 Broyden cases took 25/27/30 iterations for H2O/NO2/C4H5NCS;
Anderson took 12/17/25. Broyden reached the same certified roots and energies
but did not accelerate these samples. This is characterization, not a default
mixer recommendation.

## Physical cases and references

The rank sweep above covers finite-temperature GFN2 water, open-shell GFN2 NO2,
and the larger difficult GFN2 C4H5NCS case. Additional charge-representation
checks at `1e-8` gave:

| Case | Iterations | Certified | Residual L2 / max | Energy + free energy (Eh) |
| --- | ---: | --- | --- | ---: |
| H2 / GFN1, 0 K | 8 | yes | `3.38e-11 / 1.67e-11` | -1.058598356724 |
| H2O / GFN1, 0 K | 10 | yes | `4.53e-9 / 2.28e-9` | -5.805248705084 |
| H2O / GFN2, 300 K | 13 | yes | `7.58e-9 / 5.65e-9` | -5.104158799410 |
| NO2 / GFN2, doublet, 300 K | 17 | yes | `2.35e-9 / 1.16e-9` | -10.953075803272 |
| H2O+ / GFN2, doublet, 1000 K | 14 | yes | `5.96e-10 / 3.86e-10` | -4.431226924541 |
| H2 / GFN1, stretched to 3.0 Å, 0 K | 29 | yes | `5.72e-9 / 3.01e-9` | -0.924088449850 |

Direct full-mode comparisons against the legacy Broyden reference at tight
root settings passed for GFN1/GFN2 water. Broyden fixed-point convergence and
S2 physical certification remain separate: all accepted production cases use
the unchanged method-independent certification result.

## Validation notes

The synthetic method tests cover initial sign/factor, multi-step low-rank
parity, capacity restart, zero/non-finite secant rejection, inactive history
freeze, projected proposal use, fixed history shapes, fixed-trip transforms,
reverse/JVP/jacfwd/nested derivative, vmap with different convergence times,
and trace flattening. Trace materialization pads pre-method-state rows with
zero leaves when `mix_guess=False`, preserving fixed shapes.

The CLI driver suite currently fails during `ExactSystemCore` construction
because `Config` is unhashable in the installed `tad-mctc` Node context. The
same failure was reproduced at the clean S4 start commit, before S4 changes; it
is therefore recorded as inherited environment/baseline debt. This is separate
from the CLI edit that stops rewriting ordinary Broyden to Anderson.

The focused Broyden suite passed on Python 3.14/PyTorch 2.14.1, Python
3.12/PyTorch 2.11.0, Python 3.12/PyTorch 2.13.0, and Python 3.11/PyTorch
2.8.0 (24 tests in each environment). The configured Python 3.10/PyTorch 2.6.0
environment cannot collect repository tests: its installed `tad-mctc`
`autograd` module does not export `attach_gradient`, which is imported by the
already-synchronized W2 spectral module. This is an environment dependency
compatibility limit, not a Broyden failure.

The accepted primary-environment regression results were:

| Selection | Result |
| --- | --- |
| Broyden focused suite | 24 passed |
| Certification | 61 passed |
| Lifecycle and E6a.2 | 262 passed |
| Anderson | 32 passed, 1 inherited strict gradient-parity failure |
| Implicit/root, non-slow | 58 passed |
| Slow NO2 potential implicit response | Anderson and Broyden each passed |
| Spectral, free-energy, W2c5b and W2c6 | 156 passed |
| Phase-A Group 1 | 228 passed, 1 inherited Anderson failure |
| Phase-A Group 2 | 58 passed |
| Phase-A Group 3 | 128 passed |

The inherited Anderson gradient mismatch remained `2.5007065862503453e-11`
against the unchanged `2e-13` tolerance. No Anderson arithmetic or tolerance
was changed. A separate read-only source review found no blocking S4 findings.

An additional review of `test_singlepoint_core.py` identified old FULL-route
expectations left behind by the S3 method migration and the S4 route change.
The test expecting a warning for default Broyden was updated to expect the
native route with no fallback warning, and the map-count probe now patches the
shared `SCFProblem` map boundary rather than the retired Anderson module
namespace. Its map-count assertion checks functional iterations against
functional raw maps, not against legacy Broyden iterations. Those focused
route/count tests pass. The separate 12-case, two-step legacy parity parameter
selection still fails on the S4 starting commit `e33fcf1` in all 12 cases, so
it is pre-existing; it compares short-trajectory orbital outputs from the
functional and legacy solvers and is not a native-Broyden regression.

S4 does not delete the legacy Broyden mixer or vendored xitorch. They remain
available to explicit legacy routes and reference-only tests until S7.
