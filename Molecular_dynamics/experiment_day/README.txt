FCC Al classroom workflow: structure -> LAMMPS -> local Python analysis
======================================================================

FILES
-----
01_make_al_fcc.py
    Generates a 3x3x3 conventional FCC Al crystal (108 atoms).

in.al_ensembles
    Reads the generated structure and runs:
        NPT: 1000 K, 0 bar, 20 ps
        NVT: 1000 K,        20 ps
        NVE:                20 ps
    It writes al_trajectory.dump.
    Your batch script should pipe LAMMPS stdout to a file named: out

02_analyze_md.py
    Students run this locally after copying down:
        out
        al_trajectory.dump
    It creates the five-panel analysis figure and processed CSV files.

SCC WORKFLOW
------------
1. python 01_make_al_fcc.py
2. Submit your existing batch script so LAMMPS runs in.al_ensembles and stdout
   is redirected to "out".
3. After completion, copy these two files to the laptop:
       out
       al_trajectory.dump
4. Put them in the same folder as 02_analyze_md.py.
5. On the laptop:
       python 02_analyze_md.py

PYTHON REQUIREMENTS FOR LOCAL ANALYSIS
--------------------------------------
numpy
pandas
matplotlib

ANALYSIS OUTPUT
---------------
al_md_analysis.png
velocity_force_statistics.csv
energy_vs_timestep.csv
msd_vs_timestep.csv
rdf_by_ensemble.csv
al_md_summary.txt
