#!/bin/bash -l
# Si Raman spectrum from VASP: relax -> DFPT Gamma phonons -> eps_inf at +/- displacements -> spectrum.
# Submit from this directory:   qsub run_pipeline.sh
# Re-submitting is safe: finished stages are skipped.

#$ -P me500jc
#$ -N si_raman
#$ -j y
#$ -o si_raman.log
#$ -l h_rt=04:00:00
#$ -pe omp 8

set -eo pipefail
cd "${SGE_O_WORKDIR:-$(dirname "$0")}"
HERE=$(pwd)
RAMAN="python3 $HERE/../raman.py"

module purge
module use /projectnb/me500jc/materials/software/modules
module load python3/3.10.12
module load intel/2021.1
module load vasp/6.3.2
export OPAL_PREFIX=$SCC_VASP_DIR/third-party
export OMP_NUM_THREADS=1
export OMP_STACKSIZE=1G
NP=${NSLOTS:-4}

# --- POTCAR (gitignored): use ./POTCAR or fall back to the homework copies
if [[ ! -f POTCAR ]]; then
  for p in ../../homework/homework2/problem5/POTCAR_Si/POTCAR ../../Si_electronic_structure/POTCAR; do
    [[ -f $p ]] && { cp "$p" POTCAR; echo "POTCAR <- $p"; break; }
  done
fi
[[ -f POTCAR ]] || { echo "ERROR: no Si POTCAR found; put one in $HERE/POTCAR"; exit 1; }
grep -m1 TITEL POTCAR

finished() { [[ -f $1/OUTCAR ]] && grep -q "Voluntary context switches" "$1/OUTCAR"; }
run_vasp() {  # dir incar poscar
  local d=$1
  if finished "$d"; then echo "[skip] $d already done"; return; fi
  mkdir -p "$d"; cp "$2" "$d/INCAR"; cp "$3" "$d/POSCAR" 2>/dev/null || true
  cp KPOINTS POTCAR "$d/"
  echo "[run ] $d  ($(date +%T))"
  (cd "$d" && mpirun -np "$NP" vasp_std > out)
  finished "$d" || { echo "ERROR: $d did not finish - see $d/out"; exit 1; }
}

# 1. relax cell + ions
run_vasp 01_relax INCAR.relax POSCAR
grep -E "reached required accuracy" 01_relax/OUTCAR >/dev/null || echo "[warn] relax did not report 'reached required accuracy'"

# 2. Gamma phonons by DFPT on the relaxed structure
run_vasp 02_phonon INCAR.phonon 01_relax/CONTCAR
grep -E "f(/i)? *=.*cm-1" 02_phonon/OUTCAR | tail -6

# 3. displaced structures along each optical mode
[[ -f 03_raman/modes.json ]] || $RAMAN displace --outcar 02_phonon/OUTCAR --poscar 02_phonon/POSCAR --outdir 03_raman

# 4. eps_inf at every displaced structure
for d in 03_raman/mode_*; do
  run_vasp "$d" INCAR.eps "$d/POSCAR"
done

# 5. Raman activities -> spectrum
$RAMAN collect --outdir 03_raman --out si_raman --laser 532
echo "done $(date)"
