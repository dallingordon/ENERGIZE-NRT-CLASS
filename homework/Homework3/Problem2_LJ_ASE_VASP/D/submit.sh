#!/bin/bash -l
# Started from: homework/homework2/problem5/aluminum/scf/submit.sh
# Changes vs. that file:
#   - job name al_scf -> al2_relax
#   - walltime 00:30:00 -> 02:00:00 (20 A mostly-vacuum box, PREC=Accurate,
#     spin-polarized, up to NSW=100 ionic steps -> much bigger than a
#     1-atom FCC primitive cell)
# Everything else (project, modules, MPI launch, vasp_std) is unchanged.

#$ -P me500jc
#$ -N al2_relax
#$ -j y
#$ -o vasp_job.out
#$ -l h_rt=02:00:00
#$ -pe omp 4

module purge
module use /projectnb/me500jc/materials/software/modules
module load intel/2021.1
module load vasp/6.3.2
# bundled Open MPI was relocated; point it at its real install dir
export OPAL_PREFIX=$SCC_VASP_DIR/third-party

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G

mpirun -np "${NSLOTS:-4}" vasp_std > out
