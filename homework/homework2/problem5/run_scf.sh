#!/bin/bash
# Step 1 (run on SCC from problem5/): put POTCARs + Si POSCAR in place, submit both SCF jobs.
set -e
cd "$(dirname "$0")"

# POTCARs are gitignored, so git pull doesn't bring them. They must be in POTCAR_Al/ and POTCAR_Si/.
for el in Al Si; do
  [ -f POTCAR_$el/POTCAR ] || { echo "Missing POTCAR_$el/POTCAR -- scp it over first"; exit 1; }
done
for s in scf dos; do
  cp POTCAR_Al/POTCAR aluminum/$s/POTCAR
  cp POTCAR_Si/POTCAR silicon/$s/POTCAR
done

# Si structure = relaxed CONTCAR from problem 4, renamed to POSCAR
[ -f ../problem4/CONTCAR ] || { echo "Missing ../problem4/CONTCAR"; exit 1; }
cp ../problem4/CONTCAR silicon/scf/POSCAR
cp ../problem4/CONTCAR silicon/dos/POSCAR

(cd aluminum/scf && qsub submit.sh)
(cd silicon/scf  && qsub submit.sh)
