#!/bin/bash -l

#$ -P me500jc
#$ -N homework_2
#$ -j y
#$ -o vasp_job.out
#$ -l h_rt=00:30:00
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
