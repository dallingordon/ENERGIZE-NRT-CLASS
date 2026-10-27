#!/bin/bash -l

#$ -P me500jc
#$ -N LAMMPS_MD
#$ -j y
#$ -o lammps_job.out
#$ -l h_rt=00:30:00
#$ -pe omp 4

module purge
module load gcc/11.2.0
module load openmpi/3.1.4_gnu-11.2.0

lammps_bin_dir=/projectnb/me500jc/materials/software/lammps/src
lammps_bin=lmp_mpi
lammps_input=in.thermal_expansion
outfile=out

export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G

mpirun -np "${NSLOTS:-4}" "${lammps_bin_dir}/${lammps_bin}" < "${lammps_input}" > "${outfile}"