# dxtb

A fully differentiable implementation of extended tight-binding (xTB) Methods, which compute energies and properties of Structures.

## Language

### Setup and evaluation

**Method**:
A published xTB Hamiltonian form, such as GFN1-xTB or GFN2-xTB.
_Avoid_: model (for the method)

**Parametrization**:
The parameter values used with a Method: either the published reference set or a retrained one.
_Avoid_: parameter set, params

**Model**:
A Parametrization together with its configuration and any additional Components, independent of any Structure; it builds Systems. Retraining parameters or adding a Component gives a different Model of the same Method.
_Avoid_: method, xTB model (for the method)

**Component**:
An energy term of a Model other than the core Hamiltonian. Every Component is either a Classical or an Interaction.
_Avoid_: term, contribution (unqualified)

**Classical**:
A Component whose energy does not depend on the SCF charges, so it is evaluated once, outside the SCF. It may compute charges of its own, as electronegativity equilibration does.
_Avoid_: charge-independent term, non-SCF term

**Interaction**:
A Component whose energy depends on the SCF charges, so it enters the SCF. The same physics can appear as either kind: dispersion exists as a Classical and as a self-consistent Interaction.
_Avoid_: interaction (for the interaction energy between fragments), self-consistent term

**System**:
Everything about an xTB evaluation that depends only on the atomic numbers (no positions), built once by the Model for a given Composition.
_Avoid_: system (for the physical molecule)

**Composition**:
The ordered sequence of atomic numbers of a Structure, excluding Padding. Water written O,H,H and H,O,H are two different Compositions with the same formula; each Composition has its own System.
_Avoid_: formula, stoichiometry

**Structure**:
The physical thing being evaluated: atomic numbers, positions, total charge and spin. Neutral and cationic water at the same geometry are two different Structures.
_Avoid_: molecule, geometry, system

**External field**:
An electric field and/or field gradient applied to an evaluation from outside. It is not part of the Structure: one Structure under two different External fields is still one Structure.
_Avoid_: perturbation (unqualified)

**Calculator**:
A Model bound to the atomic numbers of one Composition or one Batch, which evaluates whatever positions, charge, spin and External field each call supplies. One Calculator therefore evaluates many Structures.
_Avoid_: driver, engine

**Result**:
The outputs of one evaluation. For a Batch, every quantity that belongs to a Lane, including its Iterations and whether it is Converged, is reported per Lane.
_Avoid_: output, properties (for the whole Result)

### Batching

**Batch**:
An ordered collection of Lanes evaluated together in one call.

**Lane**:
One position in a Batch: one Structure together with its External field. Several Lanes may share a Structure, and Lanes with the same Composition share one System.
_Avoid_: row, entry, sample

**Padded batch**:
A Batch whose Lanes may have different Compositions; shorter Lanes are filled with Padding to a common atom count.
_Avoid_: packed batch, heterogeneous batch

**Conformer batch**:
A Batch whose Lanes all share one Composition and carry no Padding.
_Avoid_: homogeneous batch

**Padding**:
Placeholder atom slots (atomic number zero) in a Padded batch that belong to no Structure. Padding may sit anywhere in a Lane, not only at its end.
_Avoid_: dummy atoms, ghost atoms, inactive atoms

**Lane group**:
Lanes of one Batch that share a System and have the same presence of spin and External field, so they can be evaluated together.
_Avoid_: bucket, cluster, group (unqualified)

**Mapped evaluation**:
Evaluating a whole Lane group at once by mapping the single-System calculation over its Lanes.
_Avoid_: vectorized evaluation, batched evaluation, mapped route, grouped route

**Independent evaluation**:
Evaluating one Lane on its own with its System, used for Lane groups of one and for Lanes that cannot be mapped.
_Avoid_: eager evaluation, lane-by-lane fallback, independent route

### SCF

**SCF variable**:
The quantity an SCF map takes and returns: the charges, the potential, or the Fock matrix.
_Avoid_: representation, response coordinates, working representation, scp_mode

**SCF solution**:
The self-consistent electronic state of a Lane. It can be expressed in any SCF variable; converting between them changes coordinates, not the solution.
_Avoid_: root, physical root, fixed point (in prose)

**Converged**:
Said of an SCF solution that is stationary: it reproduces itself under the SCF map. A residual that has merely dropped below a tolerance does not make a solution Converged.
_Avoid_: certified (as a separate state)

**Certification**:
The check that establishes whether a candidate SCF solution is Converged. It is neither an SCF map nor an iteration.
_Avoid_: convergence check (unqualified)

**SCF map**:
One application of the self-consistency map to an SCF variable, for example Fock matrix → orbitals and occupations → charges → potential → new Fock matrix.
_Avoid_: SCF step

**SCF converger**:
The algorithm that proposes the next SCF variable from the previous ones: Linear, Anderson, Broyden, CDIIS or SOSCF. "Converger" alone is fine when the SCF context is clear.
_Avoid_: method, mixer, accelerator, solver (for this)

**Stage**:
One SCF converger working on one SCF variable until its switch condition is met.

**Strategy**:
An ordered sequence of Stages that share one Iteration budget. A Strategy decides how the SCF solution is reached, never which derivative is taken.
_Avoid_: mixer (for the whole Strategy)

**Iteration**:
One SCF map applied to one Lane. A Lane's reported iteration count is the initial map plus every Trip during which the Lane was Active.
_Avoid_: cycle, trip

**Trip**:
One scheduled pass of the SCF loop over a whole Lane group. Lanes that are no longer Active take part unchanged, so a Lane group can run more Trips than any of its Lanes has Iterations.
_Avoid_: step, iteration

**Active**:
Said of a Lane that is still iterating: neither Converged nor failed terminally. This is the only meaning of active/inactive.
_Avoid_: active/inactive for non-Padding atoms or for nonzero electronic temperature

**Frozen**:
Said of a Lane under the Fixed-trip controller that is no longer Active; its state is carried through the remaining Trips unchanged.

**SCF controller**:
The part of an evaluation that decides when the SCF loop stops for each Lane. There are two: the Eager controller and the Fixed-trip controller.
_Avoid_: route, SCF route, functional route, legacy route

**Eager controller**:
An SCF controller that checks each Lane on the host after every Iteration and stops as soon as the Lane is no longer Active.
_Avoid_: eager route, functional loop

**Fixed-trip controller**:
An SCF controller that runs Trips scheduled in advance, with no data-dependent branching inside them; Lanes that stop being Active are Frozen rather than removed. It is the only controller that works under function transforms.
_Avoid_: masked controller, transform route

**Static schedule**:
A Fixed-trip controller schedule that always runs the configured maximum number of Trips. It is used inside function transforms.

**Adaptive schedule**:
A Fixed-trip controller schedule that runs Trips in Chunks and stops after the first Chunk at whose end no Lane is Active.

**Chunk**:
A run of consecutive Trips between two host checks in an Adaptive schedule. Its length affects only performance, never results.

### Differentiation

**Differentiation mode**:
How derivatives of SCF quantities are obtained: Unrolled or Implicit. Both approximate the same exact derivative and agree as the convergence tolerance goes to zero.
_Avoid_: scf_mode, full (for unrolled)

**Unrolled**:
The Differentiation mode that differentiates through every Iteration that ran, so the derivative depends on the path the SCF loop took.
_Avoid_: full, backprop through the SCF

**Implicit**:
The Differentiation mode that differentiates the stationarity condition at the Converged solution, independent of the path taken to reach it.
_Avoid_: adjoint (unqualified), analytic

**CPSCF equation**:
The linear system that Implicit differentiation solves for the first-order change of the SCF solution under a change of any input (positions, External field, Parametrization). "CPSCF solve" covers it and its adjoint when the direction does not matter. It has its own convergence, separate from the SCF being Converged.
_Avoid_: response solve, response equations, response (unqualified)

**Z-vector equation**:
The adjoint of the CPSCF equation, solved once per derivative target instead of once per input; it is what nuclear forces use under Implicit differentiation.
_Avoid_: adjoint response, backward solve
