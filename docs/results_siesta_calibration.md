# SIESTA calibration against QMOF (PBE-D3(BJ))

Stage 16. Goal: a local DFT reference (project rule: DFT only on the local
workstation) that reproduces the QMOF level of theory, PBE-D3(BJ) (VASP,
PAW, plane waves), closely enough to label configurations for the ReaxFF
fit and for fine-tuning the MLIP teacher.

**Reproduce.** `python scripts/siesta_calibration.py sweep | check | relax`
(runs locally, sized with `ga_reaxff.resources`; 8 MPI ranks). Records in
[`siesta_calibration/`](siesta_calibration/): `sweep.jsonl`, `check.jsonl`,
`relax_bonds.jsonl`. Full SIESTA directories in `runs/siesta_calibration/`
(not versioned). Files named `*_d3-generic.jsonl` were produced before the
D3 fix below and are kept only as a record of it.

## 1. Final settings

| setting | value |
|---|---|
| code | SIESTA 5.4.2 (MPI, s-dftd3, libxc) |
| functional | PBE + D3, Becke-Johnson damping, **explicit** PBE parameters (s6 1, a1 0.4289, s8 0.7875, a2 4.4407 Bohr), **no three-body term** |
| pseudopotentials | H, C, N, O: SIESTA ATOM-TABLE (Troullier-Martins, PBE, `.psf`); Zn: PseudoDojo (PBE, 3s3p semicore, `.psml`) |
| basis | DZP, PAO.EnergyShift 0.01 Ry |
| real-space mesh | 300 Ry |
| k-points | kgrid.Cutoff 10 A |
| smearing | 300 K |
| cost | 0.6-0.9 s per atom for a single point on 8 ranks (32-98 atoms); fixed-cell relaxation of a 26-atom MOF about 11 min |

## 2. Two configuration problems found

1. **`DFTD3.UseXCDefaults true` did not apply the PBE parameters.** On the
   test MOF (qmof-69d6aa4) SIESTA gave E(D3) = -1.768 eV; QMOF reports
   -2.143 eV, and an independent D3(BJ) implementation (torch-dftd, PBE)
   gives -2.1433 eV on the same structure. With explicit PBE parameters
   SIESTA gave -2.021 eV.
2. **SIESTA adds the three-body (Axilrod-Teller-Muto) D3 term**, which VASP
   IVDW=12 (QMOF) does not: torch-dftd with the three-body term gives
   -2.0215 eV, matching SIESTA. With a vanishing three-body cutoff SIESTA
   gives **-2.1429 eV** (QMOF -2.1433 eV).

## 3. Convergence (test MOF, 26 atoms, at the QMOF geometry)

At the QMOF geometry VASP's forces and stress are ~0, so SIESTA's residual
forces and pressure there measure the disagreement between the two setups.

| case | force rms (eV/A) | pressure (GPa) | time (s) |
|---|---|---|---|
| **final settings** | **0.230** | **+2.49** | 17 |
| mesh 200 / 500 Ry | 0.254 / 0.232 | +2.52 / +2.52 | 16 / 21 |
| k-grid cutoff 15 A | 0.230 | +2.49 | 35 |
| TZP basis | 0.197 | +2.04 | 25 |
| EnergyShift 0.005 / 0.002 Ry | 0.266 / 0.291 | +4.59 / +5.09 | 21 / 29 |
| all PseudoDojo | 0.314 | +3.60 | 25 |
| all ATOM-TABLE | 0.215 | +4.45 | 22 |

Mesh and k-points are converged. TZP helps little at 1.5x the cost. More
extended orbitals (smaller EnergyShift) *increase* the disagreement, which
points at the pseudopotentials rather than basis incompleteness; the
pseudopotential choice has the largest effect.

## 4. Where the disagreement comes from

Relaxing only the atomic positions with SIESTA (cell fixed at QMOF) removes
almost all of the pressure: the QMOF cell is fine for SIESTA, and the
forces come from bonds that SIESTA prefers slightly longer.

Fixed-cell relaxation of qmof-69d6aa4 from the QMOF structure; mean length of the same bonds:

| pseudopotentials | Zn-O | C-O | C-C | C-H | pressure after (GPa) | time (s) |
|---|---|---|---|---|---|---|
| VASP (QMOF), A | 1.975 | 1.278 | 1.512 | 1.099 | ~0 | - |
| **mixed (ATOM-TABLE + Zn DOJO)** (final) | +0.38 % | +0.66 % | +0.67 % | +1.23 % | -0.08 | 682 |
| DOJO-PSML | +0.38 % | +0.79 % | +0.98 % | +1.82 % | +0.09 | 546 |
| ATOM-TABLE | +2.44 % | +0.55 % | +0.51 % | +1.19 % | +0.85 | 465 |

Gas molecules relaxed with the final settings (vs. experimental geometries, NIST CCCBDB, as in stage 12): CH4 C-H 1.106 A (+1.7 %), H2O O-H 0.975 A (+1.8 %) and 104.1 deg, CO2 C-O 1.183 A (+2.0 %); part of the excess is PBE itself. PseudoDojo for all elements: CH4 1.113 A, CO2 1.186 A. Records: `siesta_calibration/molecules.jsonl`.

## 5. Six Zn MOFs with the final settings (QMOF geometries)

| MOF | atoms | force rms (eV/A) | pressure (GPa) | Hirshfeld vs DDEC6 charges, correlation | Zn charge, Hirshfeld / DDEC6 |
|---|---|---|---|---|---|
| qmof-377ea80 | 96 | 0.203 | +3.35 | 0.87 | 0.80 / 0.87 |
| qmof-5a6f521 | 98 | 0.196 | +4.40 | 0.85 | 0.80 / 0.82 |
| qmof-647f18b | 44 | 0.226 | +2.77 | 0.86 | 0.84 / 1.00 |
| qmof-91482b3 | 32 | 0.403 | +0.55 | 0.88 | 0.80 / 0.87 |
| qmof-998553b | 42 | 0.234 | +2.18 | 0.85 | 0.83 / 0.92 |
| qmof-bf69fe8 | 72 | 0.197 | +2.18 | 0.87 | 0.79 / 0.84 |

## 6. Conclusions

1. **SIESTA reproduces the QMOF level of theory closely but not exactly.**
   Its bonds are 0.4-1.2 % longer (DZP basis, norm-conserving
   pseudopotentials vs VASP PAW), which appears as 0.2 eV/A residual forces
   (median 0.21 eV/A over six MOFs) and +0.5 to +4.4 GPa at the QMOF
   geometries. After relaxing the positions the
   pressure vanishes, so SIESTA's cells agree with QMOF.
2. **These differences are small compared with what the fit must correct**
   (initial ReaxFF: C-O +6 %, Zn-O +3-4 %, N-H +13 %), and comparable to the
   teacher's force error (0.15 eV/A), with SIESTA being a true DFT reference
   for the reactive configurations the teacher has never seen.
3. **Consistency rule for the fit:** SIESTA energies and forces must be
   combined with SIESTA geometries. QMOF structures enter as starting points
   (and their cells as targets); configurations labelled with SIESTA are
   built from SIESTA-relaxed structures, so the fit never mixes the two
   potential-energy surfaces point by point.
4. **Hirshfeld charges** correlate with the DDEC6 charges (r = 0.85-0.88)
   but are smaller for Zn (+0.8 vs +0.8-1.0); the QEq target stays DDEC6
   from QMOF.
