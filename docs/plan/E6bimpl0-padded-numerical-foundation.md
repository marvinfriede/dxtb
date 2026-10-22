# E6bimpl.0 — Padded PyTorch integral and H0 foundation

Status: implementation slice complete; overall E6bimpl remains open.

Starting commit: `9908721ea248aa7c318a7679090d3629754061d1`.

This package builds the first numerical consumer of the registered C8b
canonical padded core. It does not activate mixed-System public batching,
change Calculator routing, or claim mixed-System SCF support.

## Ownership and source interfaces

`Model.setup(numbers)` independently produces a `System` whose authoritative
registered setup is `ExactSystemCore`. Its dynamic children include the
PyTorch integral backend Node and H0 setup Node. The accepted
`bucket_canonical_exact_cores`, `pad_canonical_exact_core`, and
`canonicalize_positions` functions provide static canonical mappings and
validity-bearing `PaddedExactSystemCore` values. `tad_mctc.tree.stack` stacks
those registered values without constructing a mutable System per lane.

The numerical call is the private
`dxtb._src.calculators.padded_integrals.evaluate_padded_integrals` function:

```python
torch.func.vmap(evaluate_padded_integrals)(
    stacked_padded_cores,
    canonical_positions,
    charges,
)
```

It returns a registered `PaddedIntegralEvaluation` Node with H0, overlap,
optional dipole/quadrupole matrices, reference occupations, and overlap
normalization. It validates the selected backend, integral level, setup,
position and charge shapes/dtypes/devices, validity dimensions, and pair-plan
structure. It does not retain an `IndexHelper`, System façade, mutable
component list, or parameter table.

The evaluator wraps the existing `IntegralSetup` view around the captured
backend Node and calls the production `build_integral_matrices` path. That
continues to call the existing PyTorch overlap/multipole pair kernel and
`build_hcore`; no integral or Hamiltonian formula was copied. The existing
kernel accepts the canonical PairPlan plus the per-lane pair validity masks.
Invalid pair geometry is made benign before the Gaussian block calculation,
invalid blocks are removed before AO scattering, and inactive AO rows and
columns, occupations, and overlap normalization are zeroed in the output.
Inactive atom coordinates are replaced before integral and H0 evaluation.

The mapped H0 probe exposed a transform-hostile `torch.all(mask)` Python
branch in the two-axis gather wrapper when the registered index itself was a
mapped input. The gather path now keeps masking in tensor operations. The
scatter-reduction path retains its established behavior: its result shape
cannot be masked with the source-index mask, and the analytical libcint force
regression is retained.

## Numerical and derivative evidence

The focused test module independently creates production GFN1 Systems and
checks the following cases at core, dipole, and quadrupole integral levels:

| Sources | Canonical coverage |
| --- | --- |
| H2/H3 | Different atom counts, AO counts, and pair-plan sizes. |
| H2/H2O | Different atom and element sets, shell structures, AO sizes. |
| H2O/H2 and H2/H2O | Reversed input ordering yields the same per-source results. |

For each mapped lane, physical AO entries are selected using the explicit
source-to-canonical orbital map and compared entrywise with independent
`build_integral_matrices` outputs from the unpadded Systems. H0, overlap,
reference occupations, overlap normalization, and optional multipoles are
checked. Inactive AO rows/columns are zero; zero/coincident canonical
placeholders remain finite and do not affect valid entries.

The mapped H0 scalar's position Jacobian has one lane axis for outputs and one
for input conformers. Off-diagonal lane blocks and gradients with respect to
inactive padded coordinates are zero. Diagonal physical-coordinate blocks
match independent serial gradients. `jacrev`, `jacfwd`, and a JVP agree. The
H0 `kpol` parameter is a differentiable leaf of the one Model-owned
`ParamModule`; its mapped gradient equals the sum of serial H0 gradients.
Padding does not copy the complete parameter table into a System.

The new suite passed 16 tests in each available approved CPU environment:

| Python | PyTorch | Result |
| --- | --- | --- |
| 3.10.22 | 2.6.0+cpu | 16 passed |
| 3.11.15 | 2.8.0+cpu | 16 passed |
| 3.12.10 | 2.11.0+cpu | 16 passed |
| 3.12.10 | 2.13.0+cpu | 16 passed |
| 3.14.8 | 2.14.1+cpu | 16 passed |

All five environments report `tad-mctc==0.9.1` and the official
`tad-dftd3==0.7.0`. CUDA was not used. The focused tests do not require
libcint.

Command in each environment:

```bash
python -m pytest -q test/test_calculator/test_e6bimpl0_padded_integrals.py
```

The combined primary-environment regression command completed with **235
passed, 1 skipped** in 396.08 seconds. It included the new evaluator, pure
integral evaluation and setup Node controls, C7 schema controls, C8b schema,
parameter-tree and padding controls, E6a.0–E6a.4, E6bcompat-policy, packed
Calculator acceptance, and E1.3 D3 forward controls. The skipped test is the
existing PyTorch analytical-force case, which is explicitly skipped because
that backend uses autograd for forces. The separately isolated libcint
analytical-force node passed.

The combined command was:

```bash
python -m pytest -q \
  test/test_calculator/test_e6bimpl0_padded_integrals.py \
  test/test_integrals/test_pure_evaluation.py \
  test/test_integrals/test_pytorch_setup_node.py \
  test/test_integrals/test_c7_schema_proof.py \
  test/test_calculator/test_c8b3a_mixed_size_padding.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b_schema_preflight.py \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e6a4_transform_acceptance.py \
  test/test_scf/test_e6bcompat_policy.py \
  test/test_calculator/test_c6_legacy_packed_acceptance.py \
  test/test_calculator/test_e13_d3_forward.py
```

The first broad attempt exposed an error from an over-broad change to the
two-axis scatter/gather wrapper. It failed only the existing
`test_analytical_properties_after_pure_singlepoint[forces-libcint]` case
because a source-index mask was applied to a smaller scatter-reduction
result. The wrapper was narrowed to keep the mapped gather tensor-only while
preserving scatter-reduction behavior. The exact node then passed in
isolation, and the complete combined selection passed. This was a transient
regression during this implementation, not an inherited failure.

After the scatter/gather correction, the existing PyTorch screening controls
and the final padded evaluator tests completed with **63 passed** and **16
passed**, respectively. The five-environment 16-test padded-evaluator matrix
was repeated after the final source and test edits.

Formatting and compilation checks: Black and isort pass on changed Python
files, `compileall` passes, and `git diff --check` passes. Ruff and mypy are
not installed in the primary environment. CUDA is unavailable; no no-libcint
profile was available. The primary environment has `tad-libcint==0.3.0` and
PySCF 2.14.0 installed.

## Scope and next stage

Canonical numerical padding is currently GFN1/PyTorch only. A real GFN2
H2/H2O exact-core pair is explicitly rejected by the canonical bucket with
`TypeError: Canonical mixed-element padding currently supports GFN1 only.`
GFN2 mixed integral/H0 execution is therefore not accepted by this package;
it requires a narrow extension of the canonical setup mapper while reusing
the same integral/H0 kernels, followed by real numerical tests.

The mapped evaluator supports the existing PyTorch dipole and quadrupole
levels and preserves `None` when those fields are not configured. This is
single-System setup data only. No padded SCF, interaction, D3, D4, or
mixed-System energy contribution is accepted here.

The next E6bimpl stage must consume this mapped H0/overlap result and define
the padded-orbital convention needed by the fixed-trip SCF eigensolver and
Fermi filling. This package intentionally does not invent that eigensolver
rule. It must then add mapped interaction/classical consumers and compare
complete mixed GFN1 results, derivatives, status, and failure behavior
against independent serial Systems. The public Calculator and its
`batch_mode=1/2`, legacy FULL route, unknown-keyword behavior, custom
components, and libcint fallback are unchanged.
