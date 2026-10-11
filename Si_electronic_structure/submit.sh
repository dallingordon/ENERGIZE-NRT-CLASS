#!/bin/bash -l

#$ -P me500jc
#$ -N Si_DFT_exp
#$ -j y
#$ -o vasp_job.out
#$ -l h_rt=00:05:00
#$ -pe omp 2

set -euo pipefail

ROOT_DIR="${SGE_O_WORKDIR:-$PWD}"
cd "${ROOT_DIR}"

if [[ ! -f POTCAR ]]; then
    echo "ERROR: POTCAR is missing."
    echo "Place the Si POTCAR in: ${ROOT_DIR}/POTCAR"
    exit 1
fi

if [[ ! -f POSCAR ]]; then
    echo "ERROR: POSCAR is missing from ${ROOT_DIR}"
    exit 1
fi

module purge
module use /projectnb/me500jc/materials/software/modules
module load intel/2021.1
module load vasp/6.3.2
export OPAL_PREFIX=/projectnb/me500jc/materials/software/vasp/6.3.2/install/third-party

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G

echo "============================================================"
echo " Si: SCF -> BAND STRUCTURE -> DOS"
echo "============================================================"
echo "Job ID:   ${JOB_ID:-unknown}"
echo "Host:     $(hostname)"
echo "Slots:    ${NSLOTS:-2}"
echo "VASP:     $(which vasp_std)"
echo "MPI:      $(which mpirun)"
echo "Started:  $(date)"
echo "============================================================"

run_vasp () {
    local stage="$1"
    echo
    echo "------------------------------------------------------------"
    echo "Starting ${stage}: $(date)"
    echo "------------------------------------------------------------"
    mpirun -np "${NSLOTS:-2}" vasp_std > out
    echo "Finished ${stage}: $(date)"
}

# ------------------------------
# 1. Self-consistent calculation
# ------------------------------
cd "${ROOT_DIR}/01_scf"
cp "${ROOT_DIR}/POSCAR" POSCAR
cp "${ROOT_DIR}/POTCAR" POTCAR
rm -f CHGCAR WAVECAR EIGENVAL DOSCAR OUTCAR vasprun.xml OSZICAR out
run_vasp "01_scf"

if [[ ! -s CHGCAR ]]; then
    echo "ERROR: 01_scf did not produce a usable CHGCAR."
    exit 2
fi

# ------------------------------
# 2. High-symmetry band structure
# ------------------------------
cd "${ROOT_DIR}/02_bands"
cp "${ROOT_DIR}/POSCAR" POSCAR
cp "${ROOT_DIR}/POTCAR" POTCAR
cp "${ROOT_DIR}/01_scf/CHGCAR" CHGCAR
rm -f WAVECAR EIGENVAL DOSCAR OUTCAR vasprun.xml OSZICAR out
run_vasp "02_bands"

# ------------------------------
# 3. Density of states
# ------------------------------
cd "${ROOT_DIR}/03_dos"
cp "${ROOT_DIR}/POSCAR" POSCAR
cp "${ROOT_DIR}/POTCAR" POTCAR
cp "${ROOT_DIR}/01_scf/CHGCAR" CHGCAR
rm -f WAVECAR EIGENVAL DOSCAR OUTCAR vasprun.xml OSZICAR out
run_vasp "03_dos"

cd "${ROOT_DIR}"

echo
echo "============================================================"
echo " All three VASP stages completed successfully."
echo " Finished: $(date)"
echo
echo " Key outputs:"
echo "   01_scf/vasprun.xml    dense-mesh eigenvalues / gap check"
echo "   02_bands/EIGENVAL     high-symmetry electronic bands"
echo "   02_bands/vasprun.xml  high-symmetry electronic bands"
echo "   03_dos/DOSCAR         density of states"
echo "   03_dos/vasprun.xml    DOS + Fermi-level information"
echo "============================================================"
