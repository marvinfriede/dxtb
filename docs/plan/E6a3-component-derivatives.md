# E6a.3 — Public conformer component and derivative acceptance

Status: complete for the documented GFN1/GFN2 PyTorch built-in family slice;
overall E6a remains open.

Starting SHA: `91e20bcfc951e7b7311e63558488762ad6bf726a`.

## Public integration boundary

Every accepted conformer result below comes from the public call
`torch.func.vmap(system.singlepoint, in_dims=(0,))(positions)` on one
independently constructed production `Model.setup()` System using the PyTorch
integral driver. The serial references use independent calls to that same
public `System.singlepoint` method. Tests compare the registered `Result`
pytree specification, all tensor leaves, their single leading conformer
dimension, dtype, device, values, classical labels and ordering, and public
SCF status. Standalone kernel tests are supporting evidence only; they are not
counted as public integration.

## Verified component matrix

| Family | Production availability in tested methods | Public mapped evidence | Derivatives and limits |
| --- | --- | --- | --- |
| Repulsion | GFN1 and GFN2 default classical lists | Active in GFN1/H2O and GFN2/H2O aggregate calls; full Result matches serial | Position gradients and GFN1 Model-owned `element.H.arep` parameter gradient match serial sums |
| DFT-D3 | GFN1 default classical list | Active, geometry-dependent contribution in GFN1/H2O; public serial/vmap Result parity and repeat-call stability | Public reverse position gradients and a reverse-over-reverse Hessian are finite; full-method JVP/jacfwd remains blocked by upstream `tad-dftd3.AtomicC6` lacking JVP |
| Halogen | GFN1 default classical list | Active nonzero energy on two asymmetric Br2/NH3 conformers; complete public Results match serial | Mapped position gradients match serial |
| Classical DFT-D4 | GFN2 default classical list | Active in GFN2/H2O; included with all default GFN2 interactions in complete public Result parity | Covered by GFN2 aggregate position-gradient and conformer-Jacobian checks |
| ES2 | GFN1 and GFN2 default interaction lists | Present in the GFN1 and GFN2 default aggregate System; mapped full Result and potential/charge fields match serial | GFN1 `charge.effective.gexp` Model-owned parameter gradient matches serial; GFN2 aggregate position derivative is covered |
| ES3 | GFN1 and GFN2 default interaction lists | Present in both production aggregates and retained in complete Result parity | Included in GFN1 reverse-over-reverse and GFN2 aggregate position derivatives |
| AES2 | GFN2 default interaction list | Present in GFN2/H2O aggregate public mapping; mapped charges, potentials, density and status match serial | Included in GFN2 aggregate position derivatives |
| D4SC | GFN2 default interaction list | Present in GFN2/H2O aggregate public mapping; no interaction child is dropped | Included in GFN2 aggregate position derivatives; D4SC setup-authority/gradient tests also run in the broad controls |
| ALPB / GeneralizedBorn | Explicit exact built-in interaction accepted through `Model.interaction` | GFN1/H2O System constructed with `GeneralizedBorn` maps and matches serial; ablation against the same Model without ALPB changes the public energy | Mapped position gradients match serial |
| ElectricField | Explicit `field=` System input | Existing E6a.0 shared and per-conformer field public mapping tests remain passing | Existing field-response derivative controls remain in place |
| ElectricFieldGrad | Explicit `field_grad=` System input | Existing E6a.0 quadrupole-level public mapping tests remain passing | Existing field-gradient derivative controls remain in place |
| IES | Auto-installed for GFN0, not for tested GFN1/GFN2 defaults | Not accepted for public conformer mapping: valid GFN0/H2 serial setup exists, but mapping reaches `torch.allclose` in `xitorch` `LinearOperator.m` via `symeig_padded` and raises a vmap `RuntimeError` | GFN0 transform acceptance is outside this package's selected GFN1/GFN2 scope |
| ShortRangeBond | Auto-installed for GFN0, not for tested GFN1/GFN2 defaults | Same GFN0 non-SCF transform blocker as IES; not counted as a mapped pass | No GFN0 vmap acceptance is claimed |

The tested defaults install GFN1 classical terms in the order Halogen, D3,
Repulsion, with ES2 and ES3 interactions. On H2O, D3 and Repulsion contribute
nonzero geometry-dependent energies; Halogen is zero because no halogen atom
is present. The separate Br2/NH3 test activates Halogen and checks it remains
nonzero after mapping. GFN2 installs classical D4 and Repulsion with ES2,
AES2, ES3, and D4SC interactions; the complete production aggregate maps as
one System. The ALPB row adds the exact built-in GeneralizedBorn interaction
through Model setup; it does not replace or manually evaluate the other
production interactions.

SRB and IES are available through the GFN0 parametrization, while GFN0's
public mapped non-SCF eigensolver path is not accepted. This failure is
explicitly classified as an unsupported transform row, not a component pass.
No arbitrary custom interaction, custom classical extension, libcint mapping,
or mixed-System batching is claimed.

## Derivative, ordering, and ownership evidence

The new public tests verify:

- Complete registered Result parity for GFN1/H2O, GFN2/H2O, active-halogen
  GFN1 Br2/NH3, and GFN1/H2O with ALPB.
- Classical contribution labels and order, nonzero active contributions,
  geometry dependence, and repeated evaluation at position 1 after position 2.
- An ES2-free/ES3-free GFN1 ablation, a GFN2 interaction-family ablation,
  and an ALPB-free GFN1 ablation each change the public result, so the tested
  contributions are active in the public path.
- Position-gradient parity for GFN1 aggregate, active Halogen, GFN2
  aggregate, and ALPB; the GFN2 mapped energy-vector Jacobian has shape
  `(B, B, nat, 3)`, with serial gradients on its diagonal and zero
  cross-conformer blocks.
- A GFN1 aggregate reverse-over-reverse Hessian under conformer `vmap` is
  finite and symmetric within float64 numerical precision.
- Mapped total-energy gradients reach the original Model-owned H repulsion
  and ES2 `gexp` parameter leaves and equal the sum of independent serial
  gradients. Parameter values and captured exact-core tensor leaves remain
  unchanged through repeated calls.

The existing E6a.0 tests continue to cover electric fields, field gradients,
optional multipole Result fields, and Calculator eager behavior. Existing D3
standalone position/parameter derivative tests remain passing. No DFT-D3
JVP/jacfwd acceptance is claimed; that row remains owned by E1.3.

## Inherited broad-suite failures

The primary non-slow broad selection below produced 1,366 passed, 12 skipped,
and 12 failed. It revealed no new E6a.3 regression:

- Ten established `test_production_singlepoint_matches_legacy_full` GFN1
  parity cases (`zero-k-charge-simple`, `zero-k-charge-anderson`,
  `zero-k-fock-simple`, `zero-k-fock-anderson`,
  `zero-k-potential-anderson`, `zero-k-potential-simple`,
  `300-k-charge-simple`, `300-k-charge-anderson`,
  `300-k-potential-anderson`, `300-k-fock-anderson`).
- The established Anderson position/parameter/field gradient discrepancy
  (`test_anderson_solver_real_position_parameter_and_field_gradients`).
- `test_implicit_d4sc_uses_system_setup_and_preserves_gradients` expects a
  public status for its implicit route. It fails at the same assertion on the
  exact starting SHA and current head: that implicit legacy result carries
  `status=None`. Its D4SC setup and parameter-gradient assertions are not
  evidence for public conformer mapping and this package does not alter that
  route.

The five previously tracked GFN2 force/Hessian failures were not part of this
selection; their prior matched-base classification remains unchanged.

## Environment and validation

Primary environment: Python 3.14.8, PyTorch 2.14.1+cpu, tad-mctc 0.9.1,
tad-dftd3 0.6.0, tad-libcint 0.3.0, PySCF 2.14.0. CUDA was unavailable.
The registered tad-dftd3 package cannot be imported directly with tad-mctc
0.9.1 in this environment, but the repository's compatibility shim is used
by the normal dxtb tests. No no-libcint environment was available.

The focused selection passed 47 tests in each approved CPU environment:

```bash
OMP_NUM_THREADS=1 python -m pytest -q \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a0_system_vmap.py
```

Environments were Python 3.10.22 / Torch 2.6.0+cpu, Python 3.11.15 / Torch
2.8.0+cpu, Python 3.12.10 / Torch 2.11.0+cpu, Python 3.12.10 / Torch
2.13.0+cpu, and Python 3.14.8 / Torch 2.14.1+cpu. The new component module
also passed independently (7 tests) in all five.

The broader primary-environment command was:

```bash
OMP_NUM_THREADS=1 /tmp1/friede/dxtb-agentic/envs/py3.14-torch2.14.1/bin/python -m pytest -q -m 'not slow' -n logical --dist worksteal --tb=short \
  test/test_calculator/test_e6a3_component_vmap.py \
  test/test_calculator/test_e6a2_public_status.py \
  test/test_calculator/test_e6a1a_gfn0_spin_validation.py \
  test/test_calculator/test_e6a1_conformer.py \
  test/test_calculator/test_e6a0_system_vmap.py \
  test/test_calculator/test_result_pytree.py \
  test/test_calculator/test_c8b_model_parameter_tree.py \
  test/test_calculator/test_c8b3b_canonical_padding.py \
  test/test_calculator/test_c8b_schema_preflight.py \
  test/test_scf/test_public_scf_status.py \
  test/test_scf/test_functional_trajectory_status.py \
  test/test_scf/test_e0_iteration_compat.py \
  test/test_scf/test_functional_solver.py \
  test/test_scf/test_functional_anderson.py \
  test/test_scf/test_functional_eager_compatibility.py \
  test/test_scf/test_e6bcompat_policy.py \
  test/test_calculator/test_field_inputs.py \
  test/test_calculator/test_singlepoint_core.py \
  test/test_integrals/test_pytorch_setup_node.py \
  test/test_integrals/test_multipole_options.py \
  test/test_classical/test_repulsion \
  test/test_classical/test_shortrangebond \
  test/test_classical/test_ies \
  test/test_classical/test_halogen \
  test/test_classical/test_dispersion/test_d3.py \
  test/test_classical/test_dispersion/test_d3_core_node.py \
  test/test_classical/test_dispersion/test_grad_pos.py \
  test/test_classical/test_dispersion/test_grad_param.py \
  test/test_classical/test_dispersion/test_d4_setup.py \
  test/test_classical/test_dispersion/test_d4_core_node.py \
  test/test_classical/test_dispersion/test_d4sc.py \
  test/test_classical/test_dispersion/test_d4sc_core_node.py \
  test/test_classical/test_dispersion/test_d4sc_grad.py \
  test/test_classical/test_dispersion/test_d4sc_setup.py \
  test/test_classical/test_dispersion/test_d4sc_setup_authority.py \
  test/test_coulomb \
  test/test_solvation
```

Black, isort, compileall, and `git diff --check` passed. Ruff and mypy were
unavailable.

## Remaining E6a obligations

E6a remains open for the complete method and family matrix, public transform
coverage for any newly accepted families, further derivative configurations,
and full-method forward AD blocked by E1.3. E6bimpl, E6bcompat validation,
E6broute, E7, and E4.6 retain their separate ownership. C8b and
E6bcompat-policy remain complete.
