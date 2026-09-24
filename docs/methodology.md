# Methodology

## 1. Problem statement
ReaxFF parametrization = find parameter values p (a subset of the ~600
numbers in an ffield file) that minimize the discrepancy between ReaxFF and
reference (normally DFT) energies and forces over a training set. The loss
surface is high-dimensional, multimodal and *not smooth* (bond-order cutoffs,
tapering, QEq), and some parameter sets crash the MD engine. These are the
conditions under which a derivative-free global optimizer (GA) followed by
a local polish is appropriate.

## 2. Genome
Each optimized parameter i has bounds [l_i, u_i] (default: start value ±30 %,
minimum half-width 0.05). The genome is g in [0,1]^d with p_i = l_i + g_i (u_i - l_i).
Parameters are identified by explicit keys (e.g. `bond:C-O:De_sigma`), so
every log line names the physical quantity.

## 3. Training set (graphene oxide test case)
Model: C32 graphene sheet (orthogonal 9.8 x 8.5 A cell, 20 A vacuum) with one
epoxide (top), one hydroxyl (top) and one hydroxyl (bottom): C32O3H2. It is
built at the equilibrium lattice constant of the force field used to relax it
(2.502 A for Chenoweth 2008), then relaxed at fixed cell.

Families (energies are always compared *relative to the family's reference
configuration*, which removes the arbitrary energy zero):

| family | probes | train points |
|---|---|---|
| strain | in-plane elasticity of the functionalized sheet | ±3 % isotropic, 7 |
| epoxide_z | C-O-C bridge height: C-O bond, C-O-C angle | -0.15..+0.45 A, 8 |
| hydroxyl_stretch | C-O(H) bond | -0.10..+0.40 A, 7 |
| hydroxyl_bend | C-O-H angle | H moved ±0.30 A, 5 |
| rattle | forces near equilibrium (all terms) | 8 x sigma = 0.04 A |

A held-out validation set uses different amplitudes and rattle seeds.

## 4. Objective
L = <w_e ((dE_ff - dE_ref)/sigma_E)^2>_E + lambda_F <w_f |F_ff - F_ref|^2 / (3 sigma_F^2)>_F,
sigma_E = 1 kcal/mol, sigma_F = 1 kcal/mol/A, lambda_F = 1. Parameter sets
that crash LAMMPS receive a fixed penalty (1e6).

## 5. Optimizer
Real-coded GA: Latin-hypercube initial population with the starting force
field injected (so the result is never worse than the start), tournament
selection (k = 3), SBX crossover (p = 0.9, eta_c = 15), bounded polynomial
mutation (p = 1/d, eta_m = 5), 2 elites, early stop after 15 generations
without improvement. Then bounded Nelder-Mead from the GA best.
The operator defaults were tuned on a 5-D Rastrigin function
(see development_log.md, stage 6).

## 6. Engine
LAMMPS `pair_style reaxff` + `fix qeq/reaxff` (tolerance 1e-6), units real.
One persistent LAMMPS instance per configuration; each candidate force field
is loaded with `pair_coeff` (~6.5 ms per single point for 37 atoms).

## 7. Validation protocol
* **Synthetic recovery benchmark** (this release): reference data generated
  with the unperturbed force field. Success = the training loss collapses,
  the validation loss collapses too (no overfitting), and the recovered
  parameters approach the true values. Parameters that the data cannot
  determine (low sensitivity) may *not* return to the truth even at zero
  loss; this identifiability information is itself a result.
* **Real parametrization** (next): replace the synthetic labels by DFT.

## 8. Moving to DFT data
1. Generate the same families of geometries (`pipeline.go_configurations`
   writes them; use `TrainingSet.write`).
2. Compute single points with VASP / Quantum ESPRESSO / SIESTA.
3. Read the outputs with ASE (`ase.io.read`), convert energies to kcal/mol
   (1 eV = 23.0605 kcal/mol) and forces to kcal/mol/A, store them in
   `info["ref_energy"]` / `arrays["ref_forces"]`, keep `family` / `is_ref`.
4. Set `reference.mode = "file"` with `train_file` / `validation_file`.
Important: use the DFT lattice constant for the DFT geometries, and a
consistent dispersion treatment (ReaxFF's vdW term vs DFT-D).

## 9. Known limitations
* Single-objective weighted sum (a multi-objective NSGA-II variant is the
  natural extension).
* Serial evaluation (the objective is embarrassingly parallel across
  individuals; a process pool is the next performance step).
* Orthogonal cells only.
* Small ReaxFF non-conservative force offsets (~0.17 kcal/mol/A on the OH
  group, see development log) set a floor for force agreement.
