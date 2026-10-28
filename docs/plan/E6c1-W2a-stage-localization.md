# E6c1-W2a — Mapped conformer stage localization

## Decision

**W2a diagnosis is complete; E6c1-W2 remains OPEN.** The starting code had a
wide-spectrum matrix-Fermi scaling defect that made the one-trip tmpda Result
nonfinite. An adaptive, detached scaling schedule fixes that numerical defect
and preserves the matrix Fermi function. After the fix, serial and mapped
one- and two-trip values and position gradients agree. The instrumented
300-trip mapped run still did not finish within its 300-second bound and
reached 106.4 GiB RSS before Result construction or backward. No full-workload
parity or performance clearance is claimed.

## Repository and environment

- Starting SHA: `5dc157bd12e854d2a8df9bee084d768eaa270a35`.
- Final SHA: recorded by the W2a commit.
- Branch: `claude/tender-dijkstra-cda42s`.
- Remote: `origin` (`git@github.com:marvinfriede/dxtb.git`); fetched before
  work, remote branch matched the starting SHA. This result is local and was
  not pushed.
- `AGENTS.md` remains untouched and untracked.
- Dependencies: mctc 0.11.3, multicharge 0.8.0, libcint 0.4.0, D3 0.7.0,
  D4 0.8.0.
- Diagnostic CPU: AMD EPYC 7763, 128 logical CPUs, 64 physical cores, Linux
  x86_64, about 492 GiB available RAM at inspection. Diagnostic runs used one
  PyTorch intra-op thread, one inter-op thread, and `OMP_NUM_THREADS=1`.
- Main diagnostic profile: Python 3.11.15, PyTorch 2.8.0+cpu. Additional
  one-trip control: Python 3.14, PyTorch 2.14.1+cpu.

The historical `benchmarks/results/E6c1-W2-cpu.json` was not changed. It
continues to record the legacy packed eight-conformer measurements (36.215 s
at one thread, 16.998 s at four threads, about 8.16 GiB process peak RSS).
Those measurements are from a different recorded software/hardware profile
and are not used as a direct speedup comparison here.

## Diagnostic harness

Added `benchmarks/e6c1_w2_diagnostic.py`. It calls the actual
`System.singlepoint()` or `torch.func.vmap(System.singlepoint, in_dims=(0,))`
path, with the production GFN1 model, PyTorch integrals, D3 and all enabled
classical terms. It emits flushed JSON Lines with monotonically increasing
sequence IDs, start/completion/failure events, process elapsed time, RSS,
dimensions, dependency versions, route, and backward-request metadata. Dense
solves, matrix exponentials and matrix-Fermi calls accumulate counts and wall
time in memory and emit one aggregate record, avoiding per-operation output
and RSS sampling in the hot loop. Stage instrumentation is process-local and
restored after the call.

The small control geometry is H2O and is explicitly not treated as W2
performance evidence. The tmpda inputs come from the unchanged
`benchmarks.e6c1_w2.conformers()` generator: 69 atoms, seed 0 and 0.1-bohr
displacements. It uses the accepted one-trip diagnostic boundary. Raw JSONL
records are under [`benchmarks/results/e6c1-w2a/`](../../benchmarks/results/e6c1-w2a/).

Representative command:

```bash
timeout 120s env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0/bin/python \
  benchmarks/e6c1_w2_diagnostic.py \
  --route mapped --nconformers 1 --molecule tmpda --threads 1 \
  --maxiter 1 --backward
```

The full-setting attempt used the same command with `timeout 300s` and
`--maxiter 300`. Partial JSONL output was retained when timeout terminated
the subprocess.

## Root cause and correction

Before the change, the existing regression
`test_fermi_matrix_handles_large_positive_core_spectrum` failed on the
starting SHA: all 1,024 entries differed from the spectral oracle, with maximum
absolute error `5.2714674690836025` and maximum relative error `1118.1991`.
The tmpda one-trip trace showed a 255-by-255 matrix-Fermi channel with
electron count 0 against target 139. Its response slope was
`-7.407441654840916e-251`; because this magnitude exceeded
`torch.finfo(float64).tiny`, the old guard used it as a Newton denominator.
The resulting shift was `-1.8764913242233995e252`. The next matrix-Fermi
output contained NaNs, and `_trace_logcosh()` then failed in
`torch.linalg.eigvalsh` (error code 253). Thus the eigensolver was the
downstream exception site; the first invalid operation was the unstable
matrix-Fermi scaling/occupation followed by the chemical-potential update.

`_matrix_fermi()` now selects a lane-local power-of-two scaling schedule from
a detached maximum absolute active row-sum bound. It uses the existing
temperature-dependent minimum scale, raising it only when the spectrum
requires more scaling. The detached scale affects only the numerically
equivalent evaluation schedule, not the physical derivative. No solver
tolerance, electron count, physical term, or public route changed.
The scaling power is capped at 22, matching the existing low-temperature
maximum schedule; this is a finite supported range, not an unbounded guarantee
for arbitrarily large matrix spectra. The added oracle tests cover spectra up
to 70.1 Eh and temperatures down to 2.0e-4 Hartree.

Regression coverage extends the existing large-spectrum test through third
derivatives against the diagonal spectral oracle, adds value checks across
moderate and wide spectra at ordinary and low temperatures, and checks
lane-local scaling under `vmap`. The old adaptive-scale implementation was
present on a separate development branch but was not part of this starting
commit; the correction is implemented and validated here against the failure
reproduced on the stated baseline.

## Measured stage results

Single-process diagnostic runs use identical physical inputs and fixed-trip
settings for serial and mapped calls. Timings are individual measurements,
not medians. They include small scoped instrumentation overhead; hot numerical
operator logging is aggregated.

| Profile / trips | Route | Forward | Backward | Peak RSS | Energy (Eh) | Free energy (Eh) | Position-gradient norm (Eh/bohr) |
|---|---|---:|---:|---:|---:|---:|---:|
| Py311/Torch2.8, 1 | Serial | 5.087 s | 8.936 s | 2.41 GiB | -26.593551333776613 | -2.434717061234313e-4 | 45.75374846750 |
| Py311/Torch2.8, 1 | Mapped | 5.170 s | 9.070 s | 2.31 GiB | -26.593551333776613 | -2.434717061234313e-4 | 45.75374846750 |
| Py311/Torch2.8, 2 | Serial | 6.929 s | 12.279 s | 3.04 GiB | -28.784196279158287 | -3.637056983535258e-4 | 42.01318653383 |
| Py311/Torch2.8, 2 | Mapped | 7.105 s | 12.226 s | 2.89 GiB | -28.784196279158287 | -3.637056983535258e-4 | 42.01318653383 |
| Py314/Torch2.14.1, 1 | Serial | 5.033 s | 8.837 s | 2.46 GiB | -26.593551333776613 | -2.434717061234313e-4 | 45.75374846750 |
| Py314/Torch2.14.1, 1 | Mapped | 5.358 s | 9.332 s | 2.33 GiB | -26.593551333776613 | -2.434717061234313e-4 | 45.75374846750 |

At Py311/Torch2.8, the maximum absolute serial/mapped position-gradient
difference was `8.22e-15 Eh/bohr` for one trip and `7.11e-15 Eh/bohr` for two
trips. Energies, free energies, iteration metadata and residuals matched
exactly at the printed precision. These are deliberately nonconverged
fixed-trip results (`converged=False`; reported iterations 2 and 3). They are
not comparable to the legacy packed 205-iteration result. The Py314 one-trip
energy and free energy matched exactly at printed precision; gradients were
finite, with the same norm to the displayed precision.

The one-trip path performs two 255-state spectral evaluations. The two-trip
path performs three. In the one-trip Py311 trace, `_matrix_fermi()` was called
9 times; 198 dense solves accumulated 1.87 s and 15 matrix exponentials
accumulated 0.61 s. The final trace-based Fermi free-energy stage took about
1.56 s. The repeated `fermi_density_matrix()` and
`gap_electron_response()` calls during spectral evaluation are the dominant
per-trip cost. A single lane has almost the same forward and backward time in
serial and `vmap`, so one-lane transform overhead is small relative to the
shared functional work.

The corrected scaling makes the one- and two-trip Results finite. It does not
make fixed-trip conformer evaluation fast enough for W2 clearance. The 300-
trip mapped run completed 179 spectral evaluations and started the next
`fermi_density_matrix()` at 299.8 s; peak RSS was 106.4 GiB. It did not reach
final Result construction or backward. No full 300-trip energy, force,
convergence status, process peak after completion, or eight-conformer result
exists. This is the measured performance and memory blocker. The prior
reported 300-second one-trip timeout was not reproduced: current one-trip
serial and mapped runs complete after this numerical correction. The historic
W2 report's 512-point entropy-quadrature bottleneck describes its earlier
implementation state; the current equal-partition route uses the trace-based
free-energy path.

## Validation

All commands below used `OMP_NUM_THREADS=1`, `-n logical --dist worksteal` for
pytest, and the repository's accepted environments.

| Environment | Selection | Result |
|---|---|---:|
| Python 3.11 / Torch 2.8 | invariant spectral density + invariant free energy | 90 passed |
| Python 3.11 / Torch 2.8 | E7a padding, padded GFN1/GFN2 spectral/SCF, implicit-root controls | 147 passed, 10 warnings |
| Python 3.11 / Torch 2.8 | E1.3/E6a D3 and transform acceptance selection | 146 passed, 6 warnings |
| Python 3.11 / Torch 2.8 | E6bcompat differential and policy files | 40 passed |
| Python 3.14 / Torch 2.14.1 | invariant spectral/free energy, E7a padding, padded GFN1/GFN2, implicit-root controls | 237 passed, 63 warnings |
| Python 3.14 / Torch 2.14.1 | E1.3/E6a D3 and transform acceptance selection | 146 passed, 15 warnings |
| Python 3.14 / Torch 2.14.1 | E6bcompat differential and policy files | 40 passed |

Selections overlap; counts are not additive. The PyTorch 2.14.1 warnings
include its Python 3.14 `torch.jit.script` future warning. The functional
solver emits the established `count_nonzero` vmap performance warning in
some mapped tests. No test/reference tolerance was changed. No unrestricted
suite was run.

Formatting and checks: Black, isort, `py_compile`, and `git diff --check`
pass. Ruff and mypy were not run. CUDA was not tested. No public Calculator
routing or E6bcompat implementation was modified.

## Review and remaining work

The independent review initially identified per-operation flushed logging and
misnamed backward metadata. The harness now aggregates hot operator timings
and reports `backward_requested` separately from actual `requires_grad`
properties. Final review found no blocking issue in the scaling math or
transform connectivity. It confirmed the scale ceiling is finite and that
the new wide-spectrum schedule uses more doubling steps than the prior
normal-temperature schedule; the report does not claim that the numerical fix
is a performance improvement. The review model identity is not exposed by the
runtime.

The adaptive scaling correctness fix and tests are implemented; no separate
performance optimization has been accepted. At the tmpda conditions the
adaptive schedule selects more doubling steps than the old fixed normal-
temperature schedule; no W2 speed improvement is inferred from the fix.
Avoid caching graph-bearing
spectral values or changing the fixed-trip controller. The next performance
work should profile the repeated `fermi_density_matrix()` and
`gap_electron_response()` calculation and the memory cost per fixed trip.
E6c1-W2 remains OPEN pending completed, parity-checked full workload and an
explicit performance disposition. W2a does not close W2 or W3. E6bcompat
remains accepted, E6broute remains blocked, and public Calculator dispatch is
unchanged.
