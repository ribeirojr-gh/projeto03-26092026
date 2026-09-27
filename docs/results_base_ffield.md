# Base force field for the Zn + C/H/N/O MOF family

Stage 12. Goal: a starting ReaxFF for the pilot family (section 4 of
[`results_qmof_survey.md`](results_qmof_survey.md)), built from published
parameter sets, with an explicit list of what is missing.

**Applications that drive the design** (defined by the project owner,
2026-09-25):
1. green-hydrogen production by water splitting (catalysis: H2O
   dissociation, O-H cleavage, H2 and O2 formation at the framework);
2. CO2 capture, CO2/CH4 selectivity and CO2 splitting in CH4/CO2 mixtures
   (CCUS).

**Reproduce.** `python scripts/build_base_ffields.py` (about 3 s). Sources and
their hashes: [`data/ffields/sources/`](../data/ffields/sources/README.md).
Merged files: `data/ffields/base/ffield.Zn-<base>`. Every number below is in
[`base_ffield/screen.json`](base_ffield/screen.json); outputs are identical
across runs.

## 1. Construction

The Zn atom block and every Zn term of `ffield.reax.ZnOH` (Raymand *et al.*
2010, water/ZnO) were added to three organic C/H/N/O force fields shipped
with LAMMPS (`ffield.merge_elements`): FC (Singh 2013), budzien (PETN, 2009)
and mattsson (hydrocarbons, 2010). General parameters and the O and H
parameters stay those of the organic base; the differences are recorded.

The donor provides Zn-O, Zn-H and Zn-Zn bonds, Zn-O and Zn-H off-diagonal
terms and five Zn-containing angles. **No file provides Zn-N or Zn-C** (and,
for FC, Zn-S, Zn-F, Zn-Cl): without them Zn does not bond to imidazolate N
or to the carboxylate carbon framework. These are the first fit targets.

## 2. Screening

Relaxed gas-phase geometries against experimental equilibrium geometries
(NIST CCCBDB), and a Zn(OH)2(H2O)2 cluster:

| molecule | quantity | exp. | FC | budzien | mattsson |
|---|---|---|---|---|---|
| H2 | H-H (A) | 0.741 | **0.738** | 0.787 | 0.807 |
| N2 | N-N | 1.098 | 1.145 | 1.132 | 1.152 |
| O2 | O-O | 1.208 | 1.263 | 1.202 | 1.214 |
| CO | C-O | 1.128 | **1.126** | 1.069 | 1.076 |
| H2O | O-H | 0.958 | 0.969 | 0.965 | 1.042 |
| H2O | angle (deg) | 104.5 | **105.0** | 102.5 | 106.3 |
| CO2 | C-O | 1.160 | 1.253 | 1.221 | 1.222 |
| CH4 | C-H | 1.087 | 1.069 | 1.110 | 1.113 |
| NH3 | N-H | 1.012 | 1.146 | **1.023** | 1.103 |
| NH3 | angle | 106.7 | 112.3 | 120.0 | 114.0 |
| C6H6 | C-C | 1.397 | **1.416** | 1.423 | 1.424 |
| C6H6 | C-H | 1.084 | 1.058 | 1.107 | 1.093 |

| | FC | budzien | mattsson |
|---|---|---|---|
| Zn(OH)2(H2O)2: Zn-O after relaxation (A) | 1.99, 1.99, 2.27, 2.27 | 1.93, 1.93, 3.56, 3.62 | 1.99, 1.99, 2.17, 2.17 |
| O within 2.5 A of Zn | **4** | 2 (water lost) | 4 |
| general parameters differing from ZnOH (of 39) | **2** | 18 | 12 |
| O / H atom parameters differing from ZnOH | 9 / 10 | 16 / 12 | 14 / 13 |

## 3. Decision: FC + Zn

1. **Consistency with the Zn donor.** The Zn terms were fitted with the
   general parameters of ZnOH. FC differs in 2 of 39 (budzien 18, mattsson
   12), so the donor terms keep their meaning.
2. **Water and H2**, central to water splitting, are the best described
   (H2 0.4 %, O-H 1.1 %, H-O-H 0.5 deg).
3. **The Zn cluster keeps its four ligands** (Zn-OH 1.99 A, Zn-OH2 2.27 A);
   budzien loses both water ligands.
4. **Element coverage.** FC also contains S, F and Cl, the next elements of
   the survey's organic base (section 3 of the survey), and Ni.

**Known weaknesses of the choice, to be addressed by the fit:**
- **CO2 is poor in all three** (C=O 1.25 A in FC, +8 %). CO2 is one of the
  two target molecules, so the C-O double-bond terms enter the optimization
  together with the Zn terms, with CO2 bending, stretching and dissociation
  (CO2 -> CO + O) in the training set.
- **N-H and NH3** (+13 % in FC) matter for amine-functionalized linkers and
  N-H...O hydrogen bonds; lower priority for the Zn-imidazolate/carboxylate
  frameworks.
- **O2** is 4.6 % long; ReaxFF has no spin, so the triplet ground state is
  only mimicked. O2 evolution energetics need explicit reference data.
- O and H parameters of FC differ from those the Zn-O terms were fitted with
  (9 and 10 parameters), so Zn-O is re-optimized as well.

## 4. Consequences of the applications for the next stages

- **Reference data must include reactions, not only near-equilibrium
  configurations**: H2O dissociation at Zn-O(carboxylate) sites, O-H
  scans, H2 recombination, CO2 bending and C=O stretching up to
  dissociation, CO2 and CH4 adsorption at nodes and linkers.
- **Adsorption selectivity** depends on dispersion and electrostatics, where
  ReaxFF is weakest. The QMOF GCMC results (CO2 and CH4 Henry coefficients
  and adsorption energies, dropped from the stage-11 snapshot) can serve as
  an independent validation target.
- **Zn(II) is d10 and redox-inactive**: Zn MOFs are good for CO2 capture
  and as a method pilot, but water-splitting catalysis mostly involves
  open-shell metals (Co, Ni, Fe, Cu, Mn), the spin-polarized families of
  the survey. They are the second milestone, once the workflow is validated
  on Zn.

## 5. Parameters to optimize first (Zn + C/H/N/O)

| group | terms | why |
|---|---|---|
| Zn-N, Zn-C | bond, off-diagonal (new) | missing: Zn-imidazolate, Zn-carboxylate framework |
| Zn-O | bond, off-diagonal, O-Zn-O and Zn-O-C angles | donor terms fitted with other O parameters; carboxylate and water coordination |
| N-Zn-N, O-Zn-N, Zn-N-C angles | (new) | tetrahedral Zn nodes |
| C-O | bond (pi and double-bond part) | CO2 geometry and dissociation |

The full list of missing Zn terms is in `base_ffield/screen.json`
(`FC.missing`: 7 bonds, 7 off-diagonals, 140 angles, most of them for
elements absent from this family).
