# Agent execution policy

These settings apply to all remaining restructuring packages unless explicitly overridden by the user.

## Models

### Main implementation agent

Use:

```text
model: Luna
reasoning effort: low
```

The implementation agent owns repository inspection, implementation, focused debugging, test execution, fixing review findings, and package reports.

### Independent reviewer

After each major logical package, request a separate read-only reviewer using:

```text
model: Luna
reasoning effort: high
```

The reviewer must inspect the actual diff and relevant surrounding source, not just the implementation summary.

Review for architecture compliance, hidden mutable state or caching, autograd/transform risks, scope creep, missing tests, and compatibility with later plan packages.

Do not poll a reviewer after every small edit.

A review is required after major logical packages including B4.5, B4.6/final B4 integration, B5, substantial B6 packages, E4 subpackages, Node migration packages, and E6.

Reviewer findings are:

```text
BLOCKER
MAJOR
MINOR
NO FINDING
```

BLOCKER and MAJOR findings must be resolved before proceeding unless explicitly assigned to a later package by the architecture owner.

If the runtime does not expose the actual selected model, report:

```text
Requested reviewer: Luna / high
Actual model identity: not exposed by runtime
```

Do not claim a model identity that cannot be verified.

## Pytest execution policy

Set:

```bash
export OMP_NUM_THREADS=1
```

Normal pytest execution must use:

```text
-n logical --dist worksteal
```

This applies to focused package tests, affected subsystem tests, component baselines, reference baselines, broad CPU suites, and compatibility-matrix pytest runs.

Examples:

```bash
python -m pytest -vv \
  -n logical --dist worksteal \
  test/test_integrals
```

```bash
python -m pytest -vv \
  -n logical --dist worksteal \
  test/test_baseline/test_components.py
```

```bash
python -m pytest -vv \
  -n logical --dist worksteal \
  test/test_baseline/test_reference.py
```

```bash
python -m pytest -vv \
  -n logical --dist worksteal \
  -m "not large and not slow" \
  --random-order-bucket=global \
  test
```

Do not replace `-n logical` with a manually selected worker count for normal validation.

Serial or reduced-worker pytest is permitted only as a diagnostic step for native crashes, xdist-specific failures, or failures that require isolation.

A serial diagnostic run does not replace the required parallel validation.

Before declaring a package complete, its required validation must again run with:

```text
-n logical --dist worksteal
```

If that parallel validation still crashes, report it as unresolved.

## CPU-only policy

Current restructuring work is CPU-only.

Do not run CUDA tests, GPU benchmarks, GPU performance investigations, or GPU-specific implementation work.

The T0.8 GPU baseline remains deferred until explicitly requested.

## Test environments

Use only:

```text
/tmp1/friede/dxtb-agentic/envs/py3.10-torch2.6.0
/tmp1/friede/dxtb-agentic/envs/py3.11-torch2.8.0
/tmp1/friede/dxtb-agentic/envs/py3.12-torch2.11.0
/tmp1/friede/dxtb-agentic/envs/py3.12-torch2.13.0
/tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1
```

Never use the user's interactive Python, Conda, or shell environment.

Use explicit interpreter paths.

Primary environment:

```text
/tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1
```

Run broad validation there.

For low-level PyTorch/core changes, run the affected focused suite across all five environments.

## Package workflow

For each major package:

```text
inspect current code and plan
→ implement with Luna / low
→ focused parallel pytest
→ affected subsystem parallel pytest
→ component/reference gates as applicable
→ broad primary CPU suite
→ inspect final diff
→ independent Luna / high review
→ fix BLOCKER/MAJOR findings
→ rerun affected parallel tests
→ package report
→ stop before next package
```

Do not automatically continue through multiple major packages unless explicitly instructed.

## Repository and plan authority

The authoritative implementation plan is:

```text
docs/plan/07-current-state-and-execution-plan.md
```

It supersedes older package status/order where they conflict.

Before each package:

1. read the relevant section of `docs/plan/07-current-state-and-execution-plan.md`;
2. inspect the current source rather than relying on historical line references;
3. inspect `git status`, local HEAD, and the remote branch;
4. ensure work is not based on stale assumptions.

Current restructuring branch:

```text
claude/tender-dijkstra-cda42s
```

Do not assume the SHA recorded here remains current. Refresh the branch head at the start of each new task.

## General constraints

Do not:

- weaken tolerances to make refactors pass;
- regenerate golden references without an approved physics change;
- add hidden caches;
- detach differentiable inputs or parameters;
- introduce Python scalar conversions in transform-critical paths;
- introduce new custom autograd Functions without full transform justification;
- mix unrelated later packages into the current package;
- claim tests passed when they were not run;
- claim a reviewer model identity that runtime metadata does not expose.
