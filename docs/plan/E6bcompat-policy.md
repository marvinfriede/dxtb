# E6bcompat-policy: batch and legacy FULL compatibility

Version: 1.0, characterized at `6e23c388f1e8ab1ad9372c36069ea2aa85bbaf21`.

## Purpose and ownership

This document freezes the compatibility boundary that E6bimpl must replace
and E6bcompat must validate. It does not implement a new route. C8b remains
complete at its accepted P9 structural/pure-kernel boundary. The C8b padded
core is not the legacy packed Calculator input and is not connected to
Calculator batch dispatch.

Ownership is deliberately split:

| Package | Responsibility |
| --- | --- |
| E6bcompat-policy | This inventory and the retain/replace/retire decisions. |
| E6bimpl | Implement the selected mixed-System replacements; no route activation. |
| E6bcompat | Compare replacements to this policy and run the required gates. |
| E6broute | Activate the authoritative route and retire production packed FULL after independent gates. |
| E4.6 | Remove legacy cull/restore and compatibility machinery after its consumers retire. |
| E6a | Separately define and validate public whole-System transforms. |

No E6a, E6bimpl, E6bcompat, E6broute, E7, or E4.6 completion is implied here.

## Entry-point and argument contracts

### Call graph and ownership

```text
Calculator(numbers, batch_mode, ...)
  -> BaseCalculator.__init__ -> Model.setup(numbers, batch_mode, dd)
  -> packed legacy adapters retained on Calculator (modes 1/2)

Calculator.singlepoint(positions, chrg, spin, field, field_grad, **kwargs)
  -> shape/content and charge/spin/field validation
  -> nonbatch: System.singlepoint -> pure single-System setup/evaluation
  -> batch: ClassicalList -> Integral builders -> scf.solve(... no extras)
         -> FULL legacy SelfConsistentFieldFull

Calculator.energy / calculate / get_property / force-property calls
  -> singlepoint, then Result field projection/reduction/autograd

direct scf.iterator.solve(..., *args, **kwargs)
  -> get_refocc -> non-SCF early return OR get_guess -> mode/route gate
  -> eligible FULL single-System with no overrides: functional FULL
  -> FULL batch/unsupported/override: SelfConsistentFieldFull(... *args, **kwargs)

direct legacy FULL construction
  -> BaseSCF consumes config/fwd_options/bck_options/eigen_options
  -> remaining args/kwargs -> BaseSCF._Data -> SCF numerical loop
```

Calculator construction accepts setup controls (including `batch_mode`,
method/options, `dtype`, `device`, and `auto_int_level`); `cache` and removed
cache/store options are rejected. Public per-call `singlepoint` accepts
position, charge, spin, field, field gradient, and Calculator output/timer
kwargs. It does not forward an engine argument because its packed call site
invokes `scf.solve` with only the documented ten prepared positional values.
By contrast, direct `scf.solve` accepts arbitrary engine `*args` and
`**kwargs` as a legacy constructor boundary. The route gate does not discard
these extras.

### Public Calculator

`Calculator(numbers, ..., batch_mode=...)` constructs a Model/System through
the packed compatibility setup. The public modes are 0 (single System), 1
(padded independent Systems), and 2 (conformers sharing one composition).
Numbers have shape `(nat,)` for mode 0 and `(batch, nat)` for packed modes;
positions have `(nat, 3)` or `(batch, nat, 3)`. Mode 1 uses padded atom
entries and the packed IndexHelper masks. Mode 2 describes same-composition
geometries. The C8b `PaddedExactSystemCore` is a separate structural adapter
and does not define either public mode.

`EnergyCalculator.singlepoint()` accepts `positions`, `chrg`, `spin`,
`field`, `field_grad`, and `**kwargs`. It validates positions against the
Calculator's numbers and converts charge/spin to the configured dtype/device.
For batches, scalar non-default charge is rejected; per-System charge and spin
tensors are reshaped from `(batch,)` to `(batch, 1)`. The current default
scalar charge path constructs two zeros, so the characterized public batch
contract is the existing two-entry compatibility behavior, not a new
arbitrary-batch scalar-broadcast promise. Fields use the Calculator's
`_call_tensor` shape/device/dtype checks; electric field requires dipole
integrals and field gradient requires quadrupole integrals. Explicit fields
are appended as call-local built-in interactions; a same-label custom
interaction conflicts with explicit field input.

On a packed `singlepoint`, `**kwargs` are used for the documented Calculator
options (`write_overlap`, `write_dipole`, `write_quadrupole`, `write_hcore`,
`cuda_sync_in_scf`) and are not passed to `scf.solve`. Other unknown kwargs
are currently accepted and ignored on this packed path. Removed result-store
kwargs are explicitly rejected. This ignored-extra behavior is not a supported
engine-option API: E6broute must reject unsupported public extras with a
clear `TypeError` rather than silently forwarding them or implying that the
SCF engine consumed them. No new positional engine-extra public API is added.

For a single System, the same kwargs remain Calculator-side output/timer
options and the call delegates to `System.singlepoint`; they are not an SCF
engine forwarding surface. `energy()` calls `singlepoint()` and sums the atom
axis (with `keepdim` controlling the reduced dimension). `calculate()` and
`get_property()` delegate to the same result boundary. Packed Result energy
retains per-atom contributions with shape `(batch, nat)`; `energy()` produces
one value per System. Packed forces are obtained by differentiating the sum
over per-atom/per-System energy as existing tests do. Legacy packed iteration
and convergence metadata follow the legacy SCF result shape/type; the
functional single-System tensor-status schema does not change that contract.

### Direct low-level SCF

`scf.iterator.solve(numbers, positions, chrg, spin, interactions, cache,
ihelp, config, integrals, refocc, *args, core_index=None, **kwargs)` is a
different API. For FULL SCF, a batch, any nonempty `args`, or any nonempty
`kwargs` selects the legacy `SelfConsistentFieldFull` constructor. The routing
layer forwards extras unchanged. `core_index` is the existing reserved
internal keyword; the internal trusted-map wrapper is separate. Non-SCF
routes retain their existing early return and kwargs path.

`BaseSCF.__init__` consumes `config`, `fwd_options`, `bck_options`, and
`eigen_options`; every other argument is forwarded to `_Data`. `_Data` checks
missing matrices and then Python reports unsupported constructor arguments.
This is why direct low-level unknown options raise rather than disappear,
while an unknown public packed Calculator kwarg is currently ignored.

## Observed batch and compatibility matrix

“Evidence” names the existing or added regression that owns the observation.
The public policy column is binding for E6bimpl/E6bcompat. A replacement may
change internals, but not an established output, error, or contribution
without the explicitly stated migration rule.

| Case | Observed current behavior | Decision and replacement owner | Required future check / migration |
| --- | --- | --- | --- |
| GFN1, mode 1, equal-size PyTorch systems | Packed Calculator setup and legacy FULL; returned energy is per-atom `(B,N)`. | REPLACE in E6bimpl, activate only in E6broute. | Compare every System to independent serial calls for energy, classical terms, forces, charges/spin, status and ordering. |
| GFN1, mode 1, different-size padded systems | Public mode 1 is packed padded-number setup and legacy FULL. This is not C8b canonical padding. | REPLACE; preserve existing mode-1 façade until E6broute. | Include real/padded masks, no fictitious contributions, mapped serial values/gradients, and convergence metadata. |
| GFN1, mode 2, equal-composition conformers | Legacy packed FULL; existing mode-2 gradients agree with serial in `test_c6_legacy_packed_acceptance.py`. | REPLACE in E6bimpl, then E6broute. | Retain conformer ordering, per-System energy, forces and metadata. |
| GFN1, `batch_mode=1` and `2` | Both remain legacy packed production routes; packed integral adapters remain active. | REPLACE; no mode is removed by this policy. | Exercise both modes with PyTorch integrals and classical contributions. |
| GFN2, packed batch | Legacy FULL. `BaseTSCF` warns and switches non-Fock SCP to Fock; strict configuration raises `NotImplementedError`. | REPLACE only where method semantics are supported; preserve this legacy behavior until parity gate. | Test Fock/non-Fock and strict/non-strict route, warning, energy, charge and failure behavior. Do not generalize from GFN1. |
| C8b H2/H3 and H2/H2O padded cores | Registered structural stacks and pure Repulsion map are accepted; ordinary Calculator does not dispatch them. | RETAIN as structural/P9 API; no E6broute substitution without a separate numerical gate. | Keep C8b tests; test Calculator packed route independently. |
| PyTorch batch integrals | Packed mutable integral builders form batch overlap/H0; the calculator calls `build_overlap`, optional multipoles, then `build_hcore`. | REPLACE in E6bimpl; packed builder retirement belongs to E6broute/E2-legacy. | Validate H0/integral values, masks, dtype/device, gradients, and errors against independent systems. |
| Standard libcint batch setup/eager evaluation | Existing libcint driver manager supports packed construction where its available tests/fixture support it; CPU residency/force-CPU behavior is owned by the libcint wrapper. No whole-System vmap is promised. | RETAIN an outer Python loop over independent single-System libcint calls as compatibility composition if packed libcint replacement is unavailable. E6bimpl owns only that compatibility composition; E6broute owns whether it becomes the authoritative route. | Compare each result/gradient to serial libcint; retain input order, force-CPU policy and errors. |
| Libcint mixed logical/physical device | `force_cpu` may keep backend leaves on CPU while the System's logical tensors have another device; it is not permission to move libcint data. CUDA/libcint numerical behavior is not accepted here. | RETAIN existing driver policy; no device-policy change in E6bcompat-policy. | CPU force-CPU controls, and actual hardware validation before any CUDA claim. |
| Built-in supported interactions | Exact eligible built-ins may use functional single-System FULL. Packed FULL remains legacy-owned. | REPLACE packed evaluation while preserving each contribution and its order. | Check per-family contribution and total result against serial legacy calls. |
| Custom `Interaction` subclass | Functional eligibility rejects unsupported subclasses before setup; the legacy engine receives the original extension and executes its methods. | RETAIN legacy single-System composition and preserve it in batch via independent legacy calls where valid. Do not claim arbitrary transformability. | Custom subclass test proves its contribution executes and is not dropped; otherwise raise a documented explicit unsupported error. |
| Additional custom classical component | Calculator's `ClassicalList` evaluates it on the public packed path before SCF and records named classical energies. | RETAIN contribution semantics through legacy composition; explicit unsupported error is allowed only at a documented retirement milestone. | Assert custom component identity/order, classical result fields, and total energy. |
| Legacy-cached interaction extension | Its current legacy cache lifecycle is consumed by legacy FULL. | RETAIN until its extension/cache owner retires; E4.6 after batch routing. | Independent calls must receive fresh appropriate caches and preserve contributions. |
| Low-level `unexpected_option` | Forces legacy FULL; unchanged kw reaches `_Data`, exact `TypeError` says unexpected keyword argument. | RETAIN low-level error behavior. E6bimpl must never filter it. | `test_e6bcompat_policy.py` batch direct-call test; preserve exception family and option name. |
| Low-level `solver_options` | Same legacy forwarding; `_Data` rejects it with `TypeError`. | RETAIN error behavior. | Batch direct-call test. |
| Other unknown low-level keyword | Same forwarding and constructor rejection. | RETAIN error behavior. | Batch direct-call test with a distinct unknown name. |
| Low-level positional engine extra | Forces legacy; extra follows the constructor's three required positional data arguments and is passed to `_Data`; unsupported arity raises Python `TypeError`. | RETAIN constructor semantics; no public Calculator expansion. | Batch capture test proves positional identity reaches legacy constructor; actual malformed arity must remain an error. |
| `fwd_options` | Consumed by BaseSCF, merged over defaults (e.g. `damp`), and used to configure the legacy forward mixer. | RETAIN at the low-level legacy boundary; E6bimpl must forward unchanged to any retained composition, not claim functional equivalence. | Batch identity forwarding plus existing test of configured forward/eigen values. |
| `bck_options` | Consumed and merged over `{"posdef": True}`. BaseSCF stores the result; FULL's forward loop does not otherwise read it. | RETAIN constructor semantics; document that acceptance does not prove a FULL forward numerical effect. | Assert merged value at legacy construction; preserve unchanged on retained low-level calls. |
| `eigen_options` | Consumed and merged over `{"method": "exacteig"}`; attached to legacy integral/eigensolver data and can select the solver method. | RETAIN at low-level legacy calls. | Batch identity forwarding and existing direct legacy constructor control. |
| Malformed named option dictionary | Dictionary merge requires mapping-compatible values; malformed values fail during BaseSCF option setup with Python `TypeError`. | RETAIN constructor error family; no silent conversion. | E6bcompat tests a representative malformed value for each option if a replacement begins consuming it. |
| Negative electronic temperature | Config accepts the numeric value. E4.5b routes FULL to legacy; legacy Fermi path preserves its warning/error/result behavior. | RETAIN legacy ownership; global temperature policy remains separate. | Existing negative-temperature route/eager comparison; batch test records legacy route. |
| Negative Fermi `maxiter` | ConfigFermi accepts it; functional FULL eligibility rejects it and selects legacy. No conversion to zero/positive. | RETAIN legacy behavior until a global config decision. | Existing negative-Fermi route regression; compare result/error to direct legacy. |
| Negative outer SCF `maxiter`, otherwise eligible single FULL | Functional route rejects it with the E0 configuration `ValueError` after common preparation. | RETAIN this functional-only error. | Existing E0/E4.5b test. |
| Negative outer `maxiter`, batch/custom/negative-temperature/negative-Fermi legacy-owned FULL | Route remains legacy before functional-only validation; legacy loop's own result/error is preserved. | RETAIN route-specific behavior; do not add global validation. | Test representative batch and extension cases plus direct legacy constructor. |
| Missing hcore / overlap at direct SCF boundary | `IntegralMatrices` itself does not permit absent required matrices (its constructor accesses the hcore/overlap shapes). At the directly callable low-level boundary, a duck-typed integral container with a missing member routes legacy; BaseSCF `_Data` raises `ValueError("No core Hamiltonian provided.")` or `ValueError("No Overlap provided.")`. | RETAIN error category and ownership. | Existing single-System `test_missing_integrals_remain_legacy_owned` and packed direct-SCF characterization. |
| Malformed integral shapes / nonfinite matrices | `IntegralMatrices` rejects an hcore/overlap shape mismatch at construction with `ValueError("Tensor 'overlap' must match the hcore shape.")`. If both matrices have the same wrong AO shape, batch legacy SCF reaches its numerical operation and raises `RuntimeError` from batched matrix shape validation. A nonfinite hcore in the H2 fixture reaches eigensolving and raises Torch `_LinAlgError` (`linalg.eigh` failed to converge). Route selection does not normalize either input. | RETAIN the current failure layer/category; E6bimpl must not synthesize zero matrices or turn errors into success. | When replacing this boundary, compare mismatched container shape, same-but-wrong AO shape, and nonfinite-matrix node IDs/errors against this direct-SCF baseline. |
| Shared charge/spin scalar for a batch | Default scalar charge zero is expanded by current two-entry public compatibility path. Non-default scalar charge errors; spin scalar is converted and left as a single value for legacy downstream behavior. | RETAIN characterized accepted forms; do not broaden scalar broadcasting. | Test default, nonzero scalar rejection, and per-System tensors. |
| Per-System charge/spin tensors | `(B,)` non-singletons reshape to `(B,1)` before packed SCF; missing spin remains `None`. | REPLACE with same physical electron counts/ordering and validation. | Compare alpha/beta occupations, charges and energies against serial systems. |
| Electric field / field gradient | `_call_tensor` requires exactly `(3,)` and `(3,3)` respectively, even when positions are packed; these are shared perturbations, not `(B,3)` or `(B,3,3)` per-System fields. Device/dtype must match positions. Required dipole/quadrupole integral levels are checked before evaluation. | REPLACE only with equivalent call-local interaction semantics; do not broaden broadcasting silently. | Existing field validation plus `test_public_packed_field_is_shared_not_per_system`; test conflicts with custom same-label terms and response derivatives. |
| SCF excluded / classical-only | Packed path returns classical energy plus zero `scf`/`fenergy`, zero iterations, no electronic result fields. | RETAIN exact result shape and optional-field semantics. | Existing Calculator result controls plus packed mode tests. |
| Non-self-consistent method | `solve()` returns through `solve_nonselfconsistent` before FULL route selection; its kwargs behavior remains separate. | RETAIN as a separate method route, not swallowed into new FULL batching. | Test supported non-SCF route, values, optional fields and error forwarding. |

### Scope of the low-level option tests

The dedicated tests construct a real `Calculator` mode-2 System, build its
packed overlap/H0 and interaction cache, and call `scf.iterator.solve()` with
that actual batched state. They verify each named/unknown keyword reaches a
spy legacy engine unchanged; unknown-key tests also execute the real legacy
constructor and verify `TypeError`. The positional test verifies object
identity at the constructor. Separately, a real public packed calculation
shows that `unexpected_option` and `solver_options` do not reach SCF and do
not alter the result. This is evidence of current public behavior, not support
for those options at the Calculator API.

## Retained legacy composition policy

Where an old batch behavior must remain compatible but the new stack/vmap
route cannot yet support it, E6bimpl may compose independent single-System
legacy FULL calls in an outer Python loop. That policy is not implemented by
this package and is not the authoritative PyTorch route. Each call must use
its own System-derived setup and parameter-gathered tensors; calls must not
share mutable SCF state. Results must be composed in original input order
with the established result/energy/classical/status shapes. Preserve custom
contributions and each System's source parameter owner. Forward low-level
`*args`/`**kwargs` unchanged when the low-level retained contract is used;
unsupported options must raise their established constructor errors, not be
filtered. Do not turn a failed call into a zero contribution. No physical
formula is added by the composition.

The eventual authoritative PyTorch route is separately gated: it must use the
supported stack/vmap design and pass every E6bimpl gate below. Outer-loop
compatibility is not evidence that mixed-System vmap or production packed
SCF is complete.

## E6bimpl and E6bcompat acceptance gates

Before any behavior moves to mapped production execution, E6bimpl must
demonstrate, per supported method/backend/family: (1) values within established
tolerances; (2) energy, SCF, classical, and optional Result fields; (3) charge
and spin; (4) masks and structural padding; (5) atom/shell/AO mappings; (6)
integral and H0 semantics; (7) no padded physical contribution; (8) no
cross-System derivative blocks; (9) position gradients; (10) supported
parameter gradients; (11) convergence/failure metadata; (12) explicit custom
component policy; (13) no lost options; (14) deterministic ordering; (15) no
mutation of independent source Systems; and (16) no duplicated full Model
parameter tables. These are future requirements, not claims that they
already pass.

E6bcompat then runs the retained/replaced cases in this matrix against their
serial or legacy references, asserts errors and migration behavior, checks
source-state preservation, and verifies optional-backend behavior. It does
not activate the route. E6broute activates only after E6bcompat, batch-critical
E7, and E6c1 W2/W3 gates. E4.6 removes compatibility code only after those
legacy consumers have retired.

## Validation record

Environment at policy characterization: Python 3.14.8, PyTorch 2.14.1+cpu,
tad-mctc 0.9.1, tad-libcint 0.3.0, PySCF 2.14.0, CUDA unavailable. The
combined focused selection covering this policy, E4.5b routing, packed
Calculator controls, integral managers, fields, and C8b regressions completed
with 312 passed and 1 skipped. The final command was:

```bash
OMP_NUM_THREADS=1 /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/python \\
  -m pytest -q -rs -n logical --dist worksteal \\
  test/test_scf/test_e6bcompat_policy.py \\
  test/test_scf/test_functional_eager_compatibility.py \\
  test/test_calculator/test_c6_legacy_packed_acceptance.py \\
  test/test_calculator/test_cache/test_properties.py \\
  test/test_integrals/test_pure_evaluation.py \\
  test/test_integrals/test_driver/test_manager.py \\
  test/test_integrals/test_driver/test_pytorch.py \\
  test/test_calculator/test_field_inputs.py \\
  test/test_calculator/test_c8b3b_canonical_padding.py \\
  test/test_calculator/test_c8b3a_mixed_size_padding.py \\
  test/test_calculator/test_c8b3_mixed_size_preflight.py \\
  test/test_calculator/test_c8b_schema_preflight.py \\
  test/test_calculator/test_c8b_model_parameter_tree.py
```

The skip is the existing PyTorch-overlap
analytical-gradient test (`test_pure_evaluation.py:604`), which directs users
to its autograd path. The new policy module separately passed its final
focused run after all policy cases, including field and missing-integral
shapes, was added: 22 passed. The relevant packed/routing/cache subset also
passed separately: 139 passed. Other Python/PyTorch combinations, CUDA, and a
no-libcint profile are not claimed by this run.

The known ten GFN1 production/legacy parity failures and five matched-base
GFN2 force/Hessian failures remain separate inherited numerical issues; this
policy changes no references, physics, tolerances, solver policy, or XFAILs.
