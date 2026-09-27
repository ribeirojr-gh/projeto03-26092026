# First fits of the Zn + C/H/N/O force field

Stage 17. Goal: fit the Zn + C/H/N/O ReaxFF with the GA so that more Zn MOFs
pass the per-MOF scorecard of stage 14 (baseline: 96 of 228), without
breaking the molecules of the applications (water splitting, CO2).

**Reproduce.**
`python scripts/fit_zn.py build configs/zn_fit_0N.toml` (teacher labels on
the local GPU), `python scripts/fit_zn.py fit configs/zn_fit_0N.toml` (GA +
Nelder-Mead on local CPU workers sized by `ga_reaxff.resources`), then
`python scripts/baseline_validation.py --ffield runs/zn_fit_0N/ffield.best
--out docs/zn_fit_0N_validation` and `python scripts/check_molecules.py`.
Run records: `runs/zn_fit_0N/` (manifest, history, result, ffield.best);
training sets with hashes: `data/training/`.

## 1. New machinery

- **Stress** from LAMMPS (GPa, rotated back to the original frame), checked
  against finite-difference strain derivatives on a triclinic MOF: all six
  components within 0.1 %.
- **Parallel objective**: `ShardedEngine` deals the configurations over 12
  worker processes (spawned, one LAMMPS instance per configuration overall);
  results identical to the serial engine.
- **Loss with a stress term** (sigma_S = 1 GPa).
- **MOF targets** (`mofdata.py`), each from one reference source, never mixed
  point by point:

  | family | configurations | reference | targets |
  |---|---|---|---|
  | equilibrium | QMOF structure of each training MOF | QMOF (VASP PBE-D3(BJ)) | forces = 0, stress = 0 |
  | strain | cell and atoms scaled by -3 ... +3 % | MACE teacher | energy differences |
  | bond scan | Zn moved along one Zn-N and one Zn-O bond, -0.15 ... +0.30 A | MACE teacher | energy differences |
  | molecules | CO2, H2O, CH4, H2, CO, NH3, HCOOH: minimum + stretches/bends | SIESTA (local DFT) | energy differences and forces |

- **Training MOFs** (30, at most 60 atoms) are drawn from the MOFs *not* in
  the 228-MOF validation set: 20 with Zn-N contacts, 10 carboxylate-only.

## 2. Fits

| | fit 01 | fit 02 | fit 03 |
|---|---|---|---|
| training configurations | 80 | 213 | 405 |
| added | equilibrium + strain | + 14 bond scans, + 49 SIESTA molecule configurations, + over-coordination, O-Zn-O angle, C-O pi terms | + atom-level over-coordination / lone-pair terms (O, N, Zn), bond scans in all 30 MOFs (weight 3), from the start force field again |
| parameters | 24 | 32 | 38 |
| wall time (12 workers) | 12 min | 30 min | 80 min |
| objective evaluations (failed) | 2717 (40) | 3411 (18) | 5004 (108) |
| loss start -> final | 2.67e+05 -> 4348 | 9.63e+04 -> 2207 | 6.31e+04 -> 1540 |
| equilibrium force rms, final (kcal/mol/A) | 78.8 | 65.6 | 60.3 |
| equilibrium stress rms, final (GPa) | 8.3 | 9.1 | 9.6 |
| energy-difference rms, final (kcal/mol) | 52.5 | 30.0 | 21.6 |
| parameters at a bound | 4 | 5 | 3 |

## 3. Per-MOF validation (228 MOFs not used in training) and molecules

| | initial | fit 01 | fit 02 | fit 03 |
|---|---|---|---|---|
| **MOFs passed (of 228)** | **96** (42 %) | **45** (20 %) | **85** (37 %) | **128** (56 %) |
| Zn-N bonded (182) | 94 | 25 | 60 | 105 |
| carboxylate-only (46) | 2 | 20 | 25 | 23 |
| median volume change | +5.5 % | -12.3 % | -7.0 % | -3.8 % |
| MOFs losing a Zn ligand | 7 | 11 | 5 | 4 |
| Zn-N bond, median change (Zn-N bonded MOFs) | -1.8 % | -34.8 % | -22.0 % | -4.4 % |
| Zn-O bond, median change (carboxylate-only MOFs) | +4.0 % | -0.6 % | -1.8 % | -1.7 % |
| C-O bond, median change (Zn-N bonded MOFs) | +6.2 % | +2.4 % | +2.6 % | +0.9 % |
| N-H bond, median change (Zn-N bonded MOFs) | +13.4 % | +6.1 % | +5.3 % | +1.5 % |

Molecules relaxed with each force field (bond between atoms 0 and 1, A):

| molecule | exp. | SIESTA | initial | fit 01 | fit 02 | fit 03 |
|---|---|---|---|---|---|---|
| CO2 C-O | 1.160 | 1.183 | 1.253 (+8 %) | 1.079 (-7 %) | 1.112 (-4 %) | 1.217 (+5 %) |
| H2O O-H | 0.958 | 0.975 | 0.969 (+1 %) | 0.969 (+1 %) | 0.969 (+1 %) | 0.971 (+1 %) |
| CH4 C-H | 1.087 | 1.106 | 1.069 (-2 %) | 1.069 (-2 %) | 1.069 (-2 %) | 1.069 (-2 %) |
| CO O-C | 1.128 | 1.150 | 1.126 (-0 %) | 0.905 (-20 %) | 0.970 (-14 %) | 1.238 (+10 %) |
| H2 H-H | 0.741 | 0.771 | 0.738 (-0 %) | 0.738 (-0 %) | 0.738 (-0 %) | 0.738 (-0 %) |
| NH3 N-H | 1.012 | 1.028 | 1.146 (+13 %) | 0.993 (-2 %) | 1.000 (-1 %) | 0.997 (-1 %) |

## 4. What the first two fits taught

1. **Equilibrium forces alone do not place the minima.** Fit 01 lowered the
   equilibrium force error from 329 to 79 kcal/mol/A by shortening every
   C-O and Zn-N bond: CO2 C=O -7 %, CO -20 %, Zn-N in the validation MOFs
   -35 %. The carboxylate-only MOFs improved (2 -> 20 pass) but the Zn-N
   MOFs collapsed (94 -> 25), so the total fell from 96 to 45. The
   validation set, disjoint from the training set, caught it.
2. **Molecules and bond scans constrain the curves.** Fit 02 recovered to
   85 passing MOFs, kept the carboxylate gain (25 of 46) and moved CO2 back
   to -4 %, but Zn-N still shrank by 22 % and CO stayed 14 % short.
3. **The dominant equilibrium force is the atom over/under-coordination
   energy.** Finite differences of each ReaxFF energy term on the atom with
   the largest force (8 training MOFs): over/under-coordination 55 % of the
   total at the start, 30 % after fit 02; bond energy 15 % -> 29 %. That term
   is governed by atom-level parameters (O and N from FC, Zn from the ZnOH
   donor), which fits 01-02 did not optimize, so the GA compensated with the
   C-O and Zn-N bond radii.

## 5. Fit 03: first improvement over the initial force field

Adding the atom-level over-coordination and lone-pair parameters of O, N and
Zn, and bond scans in all 30 training MOFs, gives **128 of 228 MOFs passing
(56 %, from 42 %)**: Zn-N is back to -4 % (from -22 % in fit 02), the
carboxylate C-O to +0.9 % (from +6.2 % initially), and the number of MOFs
losing a Zn ligand drops from 7 to 4. Remaining problems:

- **CO2 and CO are too long** (+4.9 % and +9.8 % vs experiment): the same
  C-O parameters serve the carboxylate bond (order ~1.5), CO2 (2) and CO (3),
  and the MOF data outweigh the molecules.
- **Frameworks now contract slightly** (median -3.8 %, from +5.5 %).
- **Equilibrium forces are still large** (60 kcal/mol/A rms): the functional
  form with these parameters cannot yet make every QMOF structure a minimum.
- **Bounds limit the search**: 24 of 38 parameters moved by more than 15 %
  and three stopped at the +/-30 % bound (C-O De_pi and p_ovun1, O p_ovun2).

Next fit (stage 18): start from fit 03 with wider bounds (+/-50 %), give the
CO/CO2 families more weight, work on the C-O bond-order parameters that
separate single, double and triple bonds, and add H2O dissociation near a Zn
site (SIESTA) for the water-splitting application.

