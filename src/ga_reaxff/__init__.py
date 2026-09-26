"""ga-reaxff: genetic-algorithm parametrization of ReaxFF force fields.

Pipeline (one module per stage):
    ffield      -> read / write / address ReaxFF parameter files
    parameters  -> choose which parameters are optimized and their bounds
    structures  -> build atomistic models (graphene oxide test case)
    dataset     -> training sets (reference energies / forces, extxyz I/O)
    engine      -> LAMMPS single-point evaluator (pair_style reaxff + QEq)
    fitness     -> objective function comparing ReaxFF vs reference
    ga          -> real-coded genetic algorithm + local refinement
    audit       -> provenance records (hashes, seeds, versions, history)
    qmof        -> QMOF database snapshot (Materials Project MOF Explorer)
    families    -> MOF chemical families and force-field coverage
    teacher     -> machine-learning teacher potentials checked against QMOF DFT
    validate    -> per-MOF scorecard against the DFT-relaxed structure
    resources   -> free cores / memory / GPU before local runs
"""
__version__ = "0.2.0"
