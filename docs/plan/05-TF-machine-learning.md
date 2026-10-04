# Track F: Machine learning

**Purpose.** Goals G3 (parameter learning, including new elements) and G4 (ML contributions in the SCF).

**Starting point (at `46af7bc`).**

- `ParamModule` exists; its parameters default to `requires_grad=False` (`param/module/types.py:97`).
- `MAX_ELEMENT = 86` (`constants/defaults.py:97`), matching the published GFN1/GFN2 coverage.
- The McMurchie–Davidson integrals support angular momentum up to f (l = 3).
- Interactions take charges and multipoles and return potentials; the SCF state is charges (or potential).
- Gradient coverage of parameters is recorded in T0.9.

---

## F1 Parameter training basics

**Goal.** Train existing parameters reliably before adding new elements.

**Steps.**

1. Fix every bug-class entry from the T0.9 coverage table.
2. Reparametrize positive-only parameters (exponents, scalings) through `log` or `softplus`, so an optimiser can't push them out of range. Provide the transform in the model, not in user code.
3. Minimal training loop with `torch.func` (`partition`/`combine` once C1/C8 land; plain `ParamModule` before): energies of a small set against reference energies.
4. Recovery test: perturb the parameters of one element, train against values computed with the original parameters, and check the original values are recovered.

**Done when.** Coverage table has no bug entries; the recovery test passes in CI.

**Needs.** T0.9, B3. **Unblocks.** F2a–F2e. **Size.** M.

---

## F2a Structure and element range (decision note plus implementation)

**Goal.** Fix the discrete structure of the new elements before any training. These choices set array shapes and can't be trained.

**Decisions (written as a short note).**

1. Shell layout for Z = 89–103 in GFN1 and GFN2: which shells (s, p, d, f), whether f shells are explicit, principal quantum numbers.
2. Number of primitives per shell (STO-nG expansion).
3. Reference occupations.
4. Which parameters are trainable for the new elements; which are fixed or shared.
5. Element coverage of the dependencies:
   - D4 reference data (tad-dftd4) and D3 references (tad-dftd3, for GFN1);
   - EEQ parameters (tad-multicharge);
   - any other element-indexed tables.

   Where coverage stops at 86, decide: provide defaults and make them trainable, or extend the dependency.

**Implementation.**

1. Raise `MAX_ELEMENT` to 103; extend every element-indexed array, filling new entries from F2b.
2. Extend the dependency tables as decided.
3. Tests: an actinide-containing molecule runs through `setup` and `singlepoint` with default parameters (values not yet meaningful).

**Done when.** Note agreed; an actinide system runs end to end.

**Needs.** F1. **Size.** M.

---

## F2b Default values and constraints

**Goal.** Reasonable starting values and safe ranges for every trainable parameter of the new elements.

**Steps.**

1. Defaults from neighbouring elements or lanthanide analogues (documented per parameter), or from literature where available.
2. A parameter file format for new elements (same format as the existing parameter files), loadable by the model.
3. Bounds and reparametrizations (from F1) per parameter.
4. Sanity checks: SCF converges with the defaults for a small actinide test set.

**Done when.** Default file committed; SCF converges for the test set.

**Needs.** F2a. **Size.** S.

---

## F2c Training only the new elements

**Goal.** Train actinide entries without changing any parameter of elements 1–86.

**Problem.** Parameters are element-indexed arrays; `partition` works on whole tensors, not on rows.

**Steps.**

1. Store each element-indexed array as a fixed base plus a trainable block for the selected elements, combined (scattered) inside `setup`.
2. The model exposes the trainable blocks as leaves; `partition` selects them.
3. Test: after training, elements 1–86 are bit-identical to the input file.

**Done when.** The bit-identical test passes; gradients reach only the selected rows.

**Needs.** B3 (setup), C1 (partition; a plain mask-based variant works before C1). **Size.** M.

---

## F2d Training workflow

**Goal.** From a dataset and a default file to trained parameters.

**Steps.**

1. **Dataset format.** Per structure: numbers, positions, charge, spin; reference energy, forces, optionally charges, dipoles. Loader that groups by size bucket (E6) or iterates per molecule.
2. **Loss.** Weighted sum of energy, force (second-order derivatives through the SCF, needs E3), charge and dipole terms; regularisation towards the defaults.
3. **Optimiser loop.** `torch.func.grad` over the loss; per-step `setup` for every molecule (parameters change every step).
4. **Batching.** Mixed compositions through stacked padded systems and `vmap` (E6, C8), or a plain loop over molecules until E6 lands.
5. **Export.** Trained values written back to a parameter file (reverse of loading; `_revert` in `param/module/param.py` exists for this direction).
6. **Reproducibility.** Seeds, configuration and dataset hash written with the output.

**Done when.** A documented script trains actinide parameters on a reference set and writes a parameter file.

**Needs.** F2b, F2c, E3 (force matching), B8. **Unblocks.** D3. **Size.** L.

---

## F2e Validation

**Goal.** Show the trained parameters generalise and nothing else changed.

**Steps.**

1. Held-out structures: energies, forces, geometries (optimisation), charges against reference.
2. Elements 1–86: T0.2 reference reproduced exactly with the new parameter file.
3. Report with error statistics per property and element.

**Done when.** Report written; regression tests for the new elements added to the reference data.

**Needs.** F2d. **Size.** M.

---

## F3 ML interaction interface

**Goal.** A pure ML contribution to the Fock matrix, consistent with an energy, usable in batching and higher-order derivatives (G4).

**Design.**

1. **Energy first (P6).** An ML term defines `E_ml(inputs, system, positions)`. Its potential (Fock contribution) is the derivative of `E_ml` by autograd inside the SCF step. Forces and all higher derivatives are then consistent.
2. **Inputs.** Start with charges, shell populations and multipoles: they fit the existing interaction interface and charge mixing. The density-matrix variant is F4.
3. **Module handling.** The network is held with `ModuleNode` (C1b): architecture as context, weights as children, called through `functional_call`.
4. **Precision.** Float64 inside the SCF; a network trained in float32 is converted, or casts internally and returns float64.
5. **Determinism.** No dropout or randomness inside the SCF; evaluation mode enforced.
6. **Smoothness.** Activation functions must be smooth for higher derivatives (ReLU has zero second derivative); document this as a requirement.

**Steps.**

1. Define the interface (an interaction subclass or protocol) and register ML terms like other interactions.
2. Example term: a small MLP on atomic charges and a local environment descriptor.
3. Tests: SCF converges; forces match finite differences; `vmap` over conformers; `jacrev` with respect to network weights; joint training of network weights and physical parameters via `partition`/`combine`.

**Done when.** The example term passes all tests and trains in a short example script.

**Needs.** B6 (per-call interaction data), C1b, E3 (consistent SCF derivatives), E4 (pure step). **Unblocks.** F4. **Size.** L.

---

## F4 Density-matrix ML term (conditional)

**Goal.** Allow ML terms that depend on the full density matrix, if F3's charge-based inputs prove too limited.

**Steps.**

1. Interface: `E_ml(P, system, positions)`; Fock contribution `∂E_ml/∂P` by autograd.
2. SCF state extended to density mixing for runs that include such a term.
3. Stability and convergence study against F3-type terms.

**Done when.** A density-matrix example term converges, has consistent forces, and trains.

**Needs.** F3, E4. **Size.** L. **Start only if** F3 results show the need.

---

## Summary

| ID | Package | Needs | Size |
| --- | --- | --- | --- |
| F1 | Training basics | T0.9, B3 | M |
| F2a | Structure and element range | F1 | M |
| F2b | Defaults and constraints | F2a | S |
| F2c | Train only new elements | B3, C1 | M |
| F2d | Training workflow | F2b, F2c, E3, B8 | L |
| F2e | Validation | F2d | M |
| F3 | ML interaction interface | B6, C1b, E3, E4 | L |
| F4 | Density-matrix ML term (conditional) | F3, E4 | L |
