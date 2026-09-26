"""Stage 16 tests: SIESTA input writing and output parsing (no SIESTA run in CI)."""
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from ga_reaxff.siesta import SiestaSettings, parse_output, write_fdf

OUT = """\
Version         : 5.4.2
siesta: Atomic forces (eV/Ang):
     1    0.100000    0.000000    1.000000
     2   -0.100000    0.500000   -0.500000
----------------------------------------
SCF Convergence by DM+H criterion
siesta: Stress tensor (static) (eV/Ang**3):
siesta:     0.001000    0.000000    0.000000
siesta:     0.000000    0.002000    0.000000
siesta:     0.000000    0.000000    0.003000

siesta: Final energy (eV):
siesta: D3 dispersion =      -0.500000
siesta:         Total =    -100.250000
siesta: FreeEng =      -100.260000
Hirshfeld Atomic Populations:
Atom #   charge [q] valence [e]  Species
     1    -0.300000    6.300000  O
     2     0.300000    0.700000  H
--------------------
Job completed
"""
FA = "     2\n     1   0.1E+00   0.0E+00   0.2E+01\n     2  -0.1E+00   0.5E+00  -0.2E+01\n"


def test_parse_output():
    r = parse_output(OUT)
    assert r["energy_total"] == -100.25 and r["free_energy"] == -100.26
    assert r["energy_d3"] == -0.5
    assert r["forces"] == [[0.1, 0.0, 1.0], [-0.1, 0.5, -0.5]]
    assert np.allclose(np.diag(r["stress"]), [0.001, 0.002, 0.003])
    assert r["hirshfeld_charges"] == [-0.3, 0.3]
    assert r["scf_converged"] and r["normal_exit"]


def test_fa_file_takes_precedence():
    r = parse_output(OUT, FA)
    assert r["forces"][0] == [0.1, 0.0, 2.0]


def test_convergence_flag_ignores_option_echo():
    text = "redata: SCF convergence failure will abort job\n" + OUT.replace(
        "SCF Convergence by DM+H criterion\n", "")
    assert not parse_output(text)["scf_converged"]


def test_write_fdf_triclinic_species_and_d3():
    a = Atoms("ZnOH", positions=[[0, 0, 0], [1.9, 0, 0], [2.3, 0.9, 0]],
              cell=[[8, 0, 0], [2, 7, 0], [1, 1, 9]], pbc=True)
    txt = write_fdf(a, "t", SiestaSettings())
    # species ordered by atomic number, indices consistent with the coordinates
    assert " 1 1 H\n 2 8 O\n 3 30 Zn" in txt
    assert txt.count("%block LatticeVectors") == 1 and " 2.0000000000 7.0000000000 0.0000000000" in txt
    coords = txt.split("%block AtomicCoordinatesAndAtomicSpecies\n")[1].split("%endblock")[0]
    assert [ln.split()[-1] for ln in coords.strip().splitlines()] == ["3", "2", "1"]
    for key in ("XC.authors PBE", "PAO.BasisSize DZP", "DFTD3 true", "DFTD3.BJdamping true",
                "DFTD3.UseXCDefaults false", "DFTD3.a1 0.4289", "DFTD3.s8 0.7875", "DFTD3.a2 4.4407", "DFTD3.3BodyCutOff 0.01 Bohr",
                "MeshCutoff 300.0 Ry", "kgrid.Cutoff 10.0 Ang", "WriteHirshfeldPop true"):
        assert key in txt
    assert "MD.TypeOfRun" not in txt


def test_write_fdf_relax_and_extra():
    txt = write_fdf(molecule("H2O", vacuum=5.0, pbc=True), "w",
                    SiestaSettings(d3=False, extra={"SCF.DM.Converge": "T"}),
                    relax={"cell": True, "fmax": 0.01})
    assert "DFTD3" not in txt and "SCF.DM.Converge T" in txt
    assert "MD.VariableCell true" in txt and "MD.MaxForceTol 0.01 eV/Ang" in txt


def test_pseudo_selection_per_element(tmp_path):
    from ga_reaxff.siesta import pseudo_file, pseudo_sha256
    (tmp_path / "A").mkdir()
    (tmp_path / "B").mkdir()
    (tmp_path / "A" / "C.psf").write_text("c-psf")
    (tmp_path / "A" / "Zn.psf").write_text("zn-psf")
    (tmp_path / "B" / "Zn.psml").write_text("zn-psml")
    m = {"default": tmp_path / "A", "Zn": tmp_path / "B"}
    assert pseudo_file("C", m).name == "C.psf"
    assert pseudo_file("Zn", m) == tmp_path / "B" / "Zn.psml"
    assert pseudo_file("Zn", tmp_path / "A").name == "Zn.psf"
    assert len(pseudo_sha256("Zn", m)) == 64
    with pytest.raises(FileNotFoundError):
        pseudo_file("N", m)


def test_default_pseudo_families():
    s = SiestaSettings()
    assert s.pseudo_families == {"default": "ATOM-TABLE", "Zn": "DOJO-PSML"}
    assert s.pseudo_dirs()["Zn"].name == "DOJO-PSML"
