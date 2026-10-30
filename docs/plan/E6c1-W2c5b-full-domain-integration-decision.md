# E6c1-W2c5b — Full-Domain Integration Decision

## Decision

**Outcome B — production integration rejected.** The matrix-shift formula is valid algebraically and useful in its guarded operating range. A density-only shift cannot preserve the accepted matrix-Fermi domain: finite-precision saturation can erase an occupation that a later large chemical-potential shift would make physically nonzero. No transform-compatible lane-local fallback has been established that avoids evaluating the direct matrix-Fermi path for ordinary lanes. Production `spectral.py` therefore remains unchanged.

The experimental `diff_steps` handling is corrected and tested. This fixes a diagnostic contract defect; it does not make the shift candidate a safe replacement. The free-energy experiment remains limited to the orthonormal trace path and uses a scoped diagnostic substitution, so it is not a production implementation.

## Source and coordination

| Field | Value |
|---|---|
| Required starting SHA | `72234b537dfd41e91375cc42fa3ba7cd95596419` |
| Starting local SHA | `72234b537dfd41e91375cc42fa3ba7cd95596419` |
| Branch | `claude/tender-dijkstra-cda42s` |
| Remote URL | `git@github.com:marvinfriede/dxtb.git` |
| `origin/claude/tender-dijkstra-cda42s` tracking SHA at start | `72234b537dfd41e91375cc42fa3ba7cd95596419` |
| Live remote verification | `git fetch origin --prune` succeeded with `GIT_SSH_COMMAND='ssh -F /dev/null'`; remote branch was `72234b537dfd41e91375cc42fa3ba7cd95596419` at package start |
| Production spectral changes | None |
| Public Calculator routing | Unchanged |
| Dependency stack | Unchanged from W2c5a |

The shared coordination record identifies SCF S2a.2 as active and does not assign `spectral.py` to another package. A W2c5b `CLAIM` was appended there. No SCF-owned production file was edited. The package stays diagnostic-only; no merge or production integration was attempted.

## `diff_steps` contract

`filling._diff_steps(order)` returns `order.bit_length()`. Thus derivative orders map to correction counts as follows:

| Derivative order | Newton correction steps |
|---:|---:|
| 0 | 0 |
| 1 | 1 |
| 2–3 | 2 |
| 4–7 | 3 |
| 8–15 | 4 |

The production matrix-density interface runs exactly `diff_steps` Newton corrections, then evaluates the final matrix Fermi function. The experimental candidate previously hard-coded two corrections, and the W2c5a response adapter discarded the caller's `diff_steps`. The candidate now accepts `diff_steps`, performs exactly that many shift updates, and performs no shift update at zero. The response adapter forwards the caller's count. The experimental trace free-energy wrapper also forwards the requested count instead of fixing it at two.

Focused tests compare candidate and production density values, electron counts, Hamiltonian gradients, temperature derivatives, electron-count derivatives, and supported second/third directional derivatives for counts 0 through 4. They also assert that the zero-step candidate calls no shift solve. The corresponding orthonormal trace free-energy values are compared for counts 0 through 4. These checks establish schedule forwarding on the tested fixtures only; they do not qualify a production fallback or general free-energy reuse.

## Full-domain counterexample

The shift guard is `abs(delta / kT) <= 9.0109133473` for float64. The direct matrix-Fermi implementation has a wider accepted exponent range, through its S0.1b scaling-power limit. A concrete one-orbital example demonstrates the information loss:

| Quantity | Value |
|---|---:|
| Dimensionless Hamiltonian | 1000 |
| `kT` | 1 |
| Initial occupation from accepted `_matrix_fermi` | exactly 0 |
| Chemical-potential shift | 990 |
| Direct accepted `_matrix_fermi` after shift | `4.5397868702390376e-05` |
| Density-only guarded shift result | NaN (outside candidate guard) |

The initial and shifted direct matrix-Fermi evaluations are finite and within the accepted scaling range. Once the initial occupation rounds to zero, the stored density has no information from which to reconstruct the later occupation. The issue is not a defect in the exact identity; it is finite-precision information loss in its density-only input representation.

To recover the accepted result outside the guard, a fallback needs the original orthogonalized spectral matrix and the accumulated chemical-potential shift (or an equivalently informative nonsaturated representation) so it can reevaluate `_matrix_fermi`. A generic transformed per-lane branch has not been shown to work for the required `vmap` and nested AD path. A tensor `where` after computing both methods preserves values but executes the expensive direct method on all lanes, removing the shortcut's intended gain. No data-dependent Python branch, unchecked host synchronization, or unsupported control-flow primitive was introduced.

This is a production-integration blocker. The observed W2c5a physical corrections were small, but those samples do not prove that all accepted inputs and derivative paths remain inside the guard. No widened guard or special-case sequence can restore information already lost to saturation.

The saturated example establishes a limit for density-only reuse over the
direct matrix-Fermi kernel's accepted input domain. It does not show that the
particular shift of 990 arises from a converged production filling state. The
production correction range remains unbounded by available evidence, so the
full-domain contract is not established; the example is not presented as a
physical tmpda failure.

## Free-energy qualification

The existing experimental trace probe still intercepts `_matrix_fermi` with a scoped module-level substitution and restores it in `finally`. It forwards `diff_steps` after this package's correction. The probe covers only an orthonormal trace fixture. It does not provide an explicit call-local free-energy implementation and does not qualify generalized overlap, Cholesky orthogonalization, noncontiguous padding, atomic partitioning, multiple spin/temperature lanes, or free-energy derivatives over those cases. No free-energy production integration is accepted.

Because density full-domain preservation already fails, the broader free-energy and physical benchmark gates were not pursued. No tmpda response or two-conformer performance run was repeated.

## Test evidence

On the finalized tests, use `-n 1 --dist worksteal`: the environment reports `nproc=1`, while `-n logical` creates 128 workers. A previous launch with `-n logical` was invalidated after editing the test parametrization during collection; it is not counted as a test result.

The finalized focused module `test/test_scf/test_e6c1_w2c5b_shift_contract.py`
passed with 12 tests on Python 3.11/PyTorch 2.8 and 12 tests on Python
3.14/PyTorch 2.14.1. Both runs include the strengthened saturation
counterexample, correction-count schedules 0–4 through the response adapter,
density/gradient checks, and orthonormal trace free-energy schedule checks.
The Python 3.14 run emitted one PyTorch JIT future warning during a
third-order case; the tests passed. Captured summaries are in
`benchmarks/results/e6c1-w2c5b/`.

No production source changed. E7a and E6bcompat suites were not rerun. The existing W2c5a results remain historical evidence and are not reclassified as repeated performance acceptance.

Independent read-only review found no blocker. It confirmed correction-count
forwarding and the density-representation counterexample, and requested the
scope clarification above: the large shift is a direct-kernel domain example,
not a demonstrated converged-SCF correction. No review test rerun was
performed. Reviewer model identity was not exposed by runtime.

Black and isort checks pass for the touched Python files in the Python 3.14
environment; Python byte compilation and `git diff --check` pass. Ruff is
unavailable in the configured environments.

## Production decisions and remaining gates

- Matrix-shift density integration: **rejected** for the accepted full domain.
- Trace-squared integration: **deferred**. Its identity is valid and its isolated kernel result was promising, but W2c5a showed little change to the complete density reverse/JVP operation and no response-level improvement was established. It should be reconsidered independently only if an end-to-end benefit is justified and the SCF owner agrees.
- Generalized-overlap/atomic free-energy reuse: not qualified.
- Model parameter gradient and implicit-response parity for the corrected candidate: not retested in this package.
- W2 and W3: **OPEN**.
- E6broute: **BLOCKED**.
- Public Calculator routing: **UNCHANGED**.

The next optimization should preserve access to the original spectral representation or target redundant work elsewhere in the SCF/response path. That is a separate package; no alternative was implemented here.

## Required verdicts

| Gate | Verdict | Scope |
|---|---|---|
| W2c5a diagnostic preserved | PASS | Historical W2c5a evidence retained unchanged |
| `diff_steps` contract | PASS | Candidate and response adapter honor counts 0–4; zero makes no shift solve |
| Full S0.1b numerical range | NOT ESTABLISHED | Density-only candidate cannot recover the saturation counterexample |
| Large-shift stability | FAIL | Candidate rejects; direct accepted kernel remains finite |
| Full-domain fallback | NOT IMPLEMENTED | No transform-compatible fallback with useful cost was established |
| Mixed-lane fallback behavior | NOT TESTED | No fallback implementation exists |
| Density values | FAIL | Values pass guarded fixtures, but full domain differs as shown |
| First derivatives | PASS | Guarded candidate schedule fixtures only |
| Second derivatives | PASS | Guarded candidate schedule fixtures only |
| Third derivatives | PASS | Guarded candidate schedule fixtures only |
| Generalized-overlap free energy | NOT TESTED | Not pursued after density-domain blocker |
| Atomic free-energy partition | NOT TESTED | Not pursued after density-domain blocker |
| Noncontiguous padding | NOT TESTED | Historical W2c5 fixture remains diagnostic evidence, not rerun here |
| Temperature derivatives | PASS | Guarded candidate schedule fixtures only |
| Electron-count derivatives | PASS | Guarded candidate schedule fixtures only |
| Model parameter gradients | NOT TESTED | Not pursued after density-domain blocker |
| Isolated differentiated performance | IMPROVED | Guarded W2c5a candidate evidence; not full-domain/fallback-capable |
| Implicit-response parity | PASS | Prior single-conformer W2c5a boundary only; no new response run |
| Implicit-response performance | IMPROVED | Prior one-run W2c5a evidence only; not repeated in W2c5b |
| Repeated tmpda performance | INCOMPLETE | Prior candidate comparison was one run per implementation |
| Two-conformer mapped validation | NOT TESTED | Not reached |
| Production matrix-shift integration | REJECTED | Full accepted domain not preserved |
| Production trace-squared integration | DEFERRED | No response-level benefit established |
| S1 coordination | VERIFIED | S2a.2 active; no production SCF overlap |
| E7a | NOT RERUN | No production numerical source changed |
| E6bcompat | NOT RERUN | No production numerical source changed |
| W2 clearance | OPEN | Frozen eight-conformer acceptance not met |
| W3 clearance | OPEN | Separate gate remains open |
| E6broute | BLOCKED | Required E6c1 gates remain open |
| Public Calculator routing | UNCHANGED | No dispatch/API changes |
| Remote publication | See package handoff | Final publication SHA is reported after push verification |
