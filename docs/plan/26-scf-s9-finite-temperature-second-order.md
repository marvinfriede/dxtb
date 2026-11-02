# SCF S9 — finite-temperature second-order policy

## Decision C

No active-smearing second-order stage is adopted. Active-smearing SCF remains
on the accepted Broyden, Anderson, and CDIIS methods. Fixed-occupation SOSCF
remains the S8 method and continues to reject active Fermi smearing.

A matrix-free physical Newton–Krylov prototype was implemented and measured.
It solved the correct projected fixed-point equation, included the complete
thermal occupation and chemical-potential response, and reproduced implicit
geometry derivatives. Across the measured cases it reduced map counts and
often tightened the residual substantially, but wall time was consistently
higher than standalone first-order methods. At the current S2 stopping
profiles, that extra work does not provide a demonstrated practical benefit.
The prototype is retained under `benchmarks/prototypes/` for reproducibility;
it is not imported by production SCF and is not a supported method.

This is an evidence-based interim policy, not a claim that a coupled thermal
free-energy Hessian is impossible. No thermal orbital SOSCF or frozen thermal
occupation approximation was implemented.

## Synchronization and baseline

S9 started from SCF SHA `ef4f3ae4d91e17f8618892c42c6a6dc16b9b56bd` and
synchronized once to W2c10 at
`8ed902f762804a9da7f4537e8d40bab80822ed1d`. The resulting pre-S9 commit was
`be0e0c7679fa20c9c9e4096a0b30acf96ca6edaa`.

The accepted SCF range-diff mapped all 54 pre-S9 SCF commits one-for-one,
apart from unavoidable commit identities, except S6 commit `9d50e41` which
needed a manual hunk reconciliation in `batch_implicit.py`: W2c10's prepared
input refresh and result assembly were retained while the accepted S6
strategy handoff and attached-root ownership were restored. No other SCF
commit had unexplained changes. The general-refactor worktree was not edited.

Before S9 production/prototype work, the focused post-sync set passed:

```text
test_invariant_spectral_density.py
test_e6c1_w2c6_tanh_solve.py
test_e6c1_w2c9_batch_implicit.py
test_functional_temperature_cutoff.py
test_functional_soscf.py
212 passed
```

The reproducible functional selection initially collected 603 tests: 601
passed, with the inherited Anderson strict gradient-parity discrepancy and a
source-scan mismatch for W2c10's eager adaptive matrix-Fermi path. The latter
guard was scoped to transform-capable source; the behavioral schedule test
still verifies that the Newton JVP uses the static path. The Anderson
discrepancy remains `2.5007065862503453e-11` versus `2e-13`.

## Why fixed-occupation orbital SOSCF is insufficient

At active smearing, the density has the form

```text
P = C f C.T
```

and both the orbitals `C` and fractional occupation matrix `f` respond to a
Hamiltonian perturbation. A unitary orbital rotation preserves the spectrum
of `f`, so an orbital-only commutator cannot represent diagonal thermal
response.

The executable counterexample uses a diagonal three-level Hamiltonian and a
diagonal perturbation at finite temperature. It observes:

```text
off-diagonal orbital commutator: 0
Fermi density JVP norm:          1.10082495785
trace of density JVP:            -5.55e-17
```

The occupations are materially fractional. The density response is nonzero
while the total electron-number response remains zero. This directly
demonstrates why simply relaxing S8's smearing rejection would optimize an
incomplete problem. The regression is in
`test_functional_finite_temperature_newton.py`.

## Newton–Krylov prototype and sign convention

For the authoritative raw physical map `G(x)`, the prototype used

```text
r(x) = G(x) - x
A(v) = v - J_G(x) v
A(delta) = r(x)
x_new = x + delta
```

This is the Newton equation for `r(x)=0`. The operator was computed as a JVP
of the raw mapped value, with the same structural padding projection as the
SCF residual. Fock tangent/result values were symmetrized on the active block.
The solve used matrix-free BiCGSTAB with explicit finite and breakdown
handling. Unconverged Krylov iterates were not returned as Newton steps; the
prototype recorded and used a bounded simple fixed-point fallback instead.

The operator `I - J_G` is the same physical linearization used by the
implicit-response equation. The primal solver and implicit attachment did not
share iteration histories. On implicit finalization the prototype-generated
root was discarded and the existing exact raw-map revalidation and implicit
root attachment were used.

The dense Jacobian was formed only in tests for small state spaces. Full
projected operator parity passed for charge, potential, and Fock coordinates;
the Fock tangent was restricted to symmetric matrices. A diagonal thermal
direction also matched centered finite differences and preserved the
electron-count constraint.

## W2c10 matrix-Fermi schedule

W2c10's detached eager primal schedule and transform-safe static schedule
remain separate. The prototype wrapped every differentiated raw map in
`eager_primal_fermi_scope(False)`, including when an outer detached eager
scope was enabled. A regression monkeypatched `_matrix_fermi_eager` to raise
and verified that Newton JVP evaluation succeeded through the static path;
ordinary detached primal evaluation remained eligible for the adaptive path.
No production spectral mathematics was changed.

## Conditioning and benchmark evidence

Measurements below used Python 3.14.8, PyTorch 2.14.1+cpu, float64, one CPU
thread, and the same physical inputs for each method. `benchmarks/results/scf-s9/`
contains raw JSON. Dense conditioning numbers are singular-value condition
estimates for `A=I-J_G` at independently certified roots.

| Case | Fractional channels | Broyden maps / s | Broyden→Newton maps / s | Newton JVPs | `cond(A)` charge / potential / Fock |
|---|---:|---:|---:|---:|---:|
| H2O/GFN2, 300 K | 0 | 22 / 1.56 | 7 / 2.81 | 11 | 3.14 / 3.14 / 3.95 |
| NO2/GFN2, 2000 K | 5 | 25 / 1.75 | 10 / 4.29 | 17 | 8.28 / 8.28 / 13.06 |
| NO2/GFN2, 5000 K | 13 | 25 / 1.77 | 10 / 4.73 | 17 | 8.13 / 8.13 / 12.64 |
| H2O/GFN1, charge +0.25, 2000 K | 6 | 21 / 1.41 | 7 / 1.72 | 7 | 2.66 / 2.53 / 3.27 |

The state dimensions for H2O/GFN2 at 300 K were 54/54/36 (charge/potential/Fock),
for NO2/GFN2 they were 54/54/144, and for fractionally charged H2O/GFN1 they
were 54/8/64. The compact potential representation was the smallest for the
fractional-charge case and was no worse conditioned than charge. Fock was
larger and somewhat more poorly conditioned in these samples.

For NO2/GFN2 at 2000 K, Anderson alone converged in 16 maps / 1.13 s and
Anderson→Newton in 10 maps / 5.07 s. CDIIS alone converged in 25 maps / 1.82 s
and CDIIS→Newton in 9 maps / 5.44 s. All results were S2-certified. The
Newton-stage threshold study at fixed-point L2 values `1e-2`, `1e-3`, and
`1e-4` produced respectively `(10 maps, 4.47 s)`, `(13 maps, 5.19 s)`, and
`(16 maps, 3.52 s)`. The threshold moved the cost but did not establish a
robust wall-time advantage.

H2O/GFN2 at 300 K had `smear=True` but no materially fractional orbitals; it
is retained as the requested low-temperature active-branch control, not as
evidence of strong thermal response. NO2 at 2000/5000 K and fractionally charged
H2O provide the material fractionality evidence. The accepted first-order
methods already met the S2 tolerance. Newton consistently spent extra JVP
work to obtain residuals below the controller's required tolerance. This
supports Decision C for the current portfolio.

## Implicit geometry and second derivative

For NO2/GFN2 at 2000 K, the geometry-dependent `System.singlepoint()` implicit
route was run with a selected Cartesian coordinate of atom 0 as the variable.
Standalone CDIIS and Broyden→Newton both certified. Results were:

| Strategy | Maps | Energy | Fixed-point L2 | `dE/dx` | `dE/d(field_y)` | `d²E/dx²` |
|---|---:|---:|---:|---:|---:|---:|
| CDIIS | 26 | -10.906375813027564 | 8.12e-9 | -4.320804188990286e-5 | -0.027329684837228668 | 0.1616226223729646 |
| Broyden→Newton | 9 | -10.906375813027566 | 8.30e-12 | -4.320804189380859e-5 | -0.027329684837248736 | 0.16162262237300512 |

The geometry first-derivative difference was `3.9e-15`, the field first-
derivative difference was `2.0e-14`, and the second-derivative difference
was `4.0e-14`. This confirms that implicit derivatives depend on
the certified physical root, not the prototype's switch or Krylov history.
The hybrid's tighter final residual did not translate into a useful reduction
in total wall time in the broader timing matrix.

## Scope and limitations

- Active-smearing SOSCF remains unsupported; no occupations were frozen and
  no temperature was changed.
- Newton–Krylov is a root solver, not a free-energy descent method.
- The prototype was eager-only and rejected fixed-trip execution.
- No public mixer, strategy, or default was added.
- No dense Jacobian exists in production; the prototype was moved out of
  `src/dxtb` after Decision C.
- The requested eight-conformer benchmark and large-system thermal scaling
  were not pursued because the small physical matrix showed no wall-time
  advantage over the first-order methods.
- S2 certification, W2c10 spectral behavior, S8 SOSCF, and implicit
  differentiation definitions remain unchanged.

## Validation record

On the synchronized W2c10 tree, the required spectral, tanh-doubling, batch
implicit, temperature-cutoff, and S8 SOSCF selection passed **212 tests**.
The S9 finite-temperature prototype/operator file passed **19 tests** under
Python 3.14.8/PyTorch 2.14.1+cpu and Python 3.11/PyTorch 2.8.0. The narrowed
legacy-state and transform-host-control architecture guards passed on both
environments. The reproducible functional/calculator selection collected 622
tests: 621 passed and the sole failure was the inherited Anderson strict
gradient parity discrepancy (`2.5007065862503453e-11` versus `2e-13`).

An independent read-only review confirmed the thermal counterexample,
projected JVP checks, AST-scoped spectral guard, benchmark provenance, and
absence of production Newton imports. The review's initial tracked-artifact
blocker was resolved by force-adding the benchmark prototype, runners, JSON
results, and provenance file. No production SCF or spectral file was changed
for S9.

## Final policy

**Decision C:** No second-order active-smearing stage is justified for the
current solver portfolio. Active-smearing calculations remain on Broyden,
Anderson, or CDIIS. Fixed/Aufbau calculations continue to use the S8 SOSCF
policy. A future method would need to show a wall-time or robustness benefit
at the accepted S2 tolerances while preserving the same thermal response and
implicit derivative contracts.
