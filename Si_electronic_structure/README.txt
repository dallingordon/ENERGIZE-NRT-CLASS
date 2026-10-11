Si DFT-to-Experiment: Band Structure, DOS, and Band Gap
============================================================

PURPOSE
-------
This is a small live VASP exercise for a graduate materials-science course.
One scheduler submission performs three sequential calculations on primitive
diamond Si:

  1. Self-consistent calculation on a regular k-point mesh
  2. Fixed-charge electronic band structure along an FCC high-symmetry path
  3. Fixed-charge density of states on a denser regular k-point mesh

The instructor must supply a licensed Si POTCAR.

DIRECTORY STRUCTURE
-------------------
POSCAR
POTCAR                  <-- instructor adds this file
submit.sh
01_scf/
    INCAR
    KPOINTS
02_bands/
    INCAR
    KPOINTS
03_dos/
    INCAR
    KPOINTS

RUNNING
-------
From the top-level directory:

    qsub submit.sh

Monitor with:

    qstat -u $USER

The job requests 2 CPU cores and 5 minutes under project me500jc.

WHAT EACH STAGE DOES
--------------------
01_scf
    12x12x12 Gamma-centered mesh
    Produces the converged CHGCAR used by the next two stages.
    The regular mesh also provides a useful numerical estimate of the
    indirect band gap from sampled occupied/unoccupied eigenvalues.

02_bands
    ICHARG = 11 fixed-charge calculation
    25 points per line segment along:

    Gamma-X-W-K-Gamma-L-U-W-L-K

    This produces the conventional E(k) electronic-band plot.

03_dos
    ICHARG = 11 fixed-charge calculation
    16x16x16 Gamma-centered mesh
    Tetrahedron method (ISMEAR = -5)
    NEDOS = 2000

    This produces the total DOS and orbital/site-resolved DOS information.

KEY OUTPUTS FOR THE PYTHON DEMO
-------------------------------
01_scf/vasprun.xml
01_scf/EIGENVAL

02_bands/vasprun.xml
02_bands/EIGENVAL

03_dos/vasprun.xml
03_dos/DOSCAR

TEACHING POINTS
---------------
Students can use the results to:

  * Plot an electronic band structure.
  * Plot the electronic density of states.
  * Locate the valence-band maximum (VBM).
  * Locate the conduction-band minimum (CBM).
  * Determine whether Si has a direct or indirect gap.
  * Calculate the PBE Kohn-Sham band gap.
  * Compare that result with the experimental Si gap (~1.1 eV at room
    temperature) and discuss why semilocal DFT underestimates band gaps.

NOTES
-----
The supplied lattice parameter is 5.43 Angstrom for conventional diamond Si.
The POSCAR uses the two-atom primitive cell.

The high-symmetry coordinates in 02_bands/KPOINTS are expressed in reciprocal
coordinates corresponding to this primitive FCC cell.
