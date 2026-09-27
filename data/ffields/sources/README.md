# Source ReaxFF force fields

Copied unchanged from the LAMMPS distribution, `potentials/` directory,
LAMMPS `stable_22Jul2025_update6` (git 9c5ab448c78a14fd534619622162ba418d6a1fb1).
LAMMPS is distributed under the GPL v2; the force-field files keep their
original headers. SHA-256 of each file in `SHA256SUMS`.

| File | Elements | Reference (as given in the file header) | Role here |
|---|---|---|---|
| `ffield.reax.ZnOH` | H O Zn (+ dummy X) | D. Raymand, A. C. T. van Duin, D. Spångberg, W. A. Goddard III, K. Hermansson, *Surf. Sci.* **604**, 741-752 (2010) - water/zinc oxide | Donor of the Zn atom block and the Zn-O, Zn-H, Zn-Zn terms |
| `ffield.reax.FC` | C H O N S F Pt Cl Ni (+ dummy X) | Singh, *Phys. Rev. B* **87**, 104114 (2013) | Organic base candidate |
| `ffield.reax.budzien` | C H O N | J. Budzien, A. P. Thompson, S. V. Zybin, *J. Phys. Chem.* **113**, 13142 (2009) - PETN | Organic base candidate |
| `ffield.reax.mattsson` | C H O N S | T. R. Mattsson *et al.*, *Phys. Rev. B* **81**, 054103 (2010) - general-purpose hydrocarbon parameterization | Organic base candidate |

Not used: `ffield.reax.lg` (low-gradient dispersion variant, extra columns
not supported by `ffield.py`), `ffield.reax.rdx` (older format), and the HNS
example file (no citation in its header).
