Part D - VASP geometry optimization of Al2

Where these files came from:
  POSCAR    - written for this problem; format/layout copied from
              homework/homework2/problem5/aluminum/scf/POSCAR.
              Geometry matches the ASE starting dimer in
              fit_lj_and_optimize_ase_template.py (Part B):
                20 A cubic cell, atoms at (8.25,10,10) and (11.75,10,10)
                -> Al-Al = 3.50 A, centered, >8 A from every face.
              Cartesian coordinates with scale factor 1.0 (positions in A).
  submit.sh - copied from homework/homework2/problem5/aluminum/scf/submit.sh
              (job name + walltime changed; see comments inside).
  INCAR, KPOINTS, POTCAR - unchanged copies of the supplied files in ../vasp/.

(POSCAR comments live in line 1 only: VASP parses the other lines
 positionally, so extra text there can break the read.)

After the run: compare_lj_vasp.py expects ../vasp/CONTCAR and ../vasp/OUTCAR,
so copy CONTCAR and OUTCAR from here into ../vasp/ before running it.
