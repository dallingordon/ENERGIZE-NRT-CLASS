#!/bin/bash -l

#$ -P me500jc
#$ -N LAMMPS_MD
#$ -j y
#$ -o lammps_job.out
#$ -l h_rt=00:30:00
#$ -pe omp 12

module load openmpi/3.1.4_gnu-11.2.0
module load gcc/11.2.0

lammps_bin_dir=/projectnb/me500jc/materials/software/lammps/src
lammps_bin=lmp_mpi
lammps_input=in.thermal_expansion
outfile=out

# VASP recommends increasing the OpenMP stack size.
export OMP_STACKSIZE=1G
#  Explanation is below
#mpirun -np 12 ${lammps_bin_dir}/${lammps_bin} < ${lammps_input} > ${outfile}
#the frenchman says do this: 
mpirun -np 12 ${lammps_bin_dir}/${lammps_bin} -in ${lammps_input} > ${outfile}