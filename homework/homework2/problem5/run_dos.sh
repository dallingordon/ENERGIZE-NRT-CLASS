#!/bin/bash
# Step 2 (run on SCC after BOTH SCF jobs finish): copy each converged CHGCAR into dos/ and submit.
set -e
cd "$(dirname "$0")"
for m in aluminum silicon; do
  [ -s $m/scf/CHGCAR ] || { echo "$m/scf/CHGCAR missing or empty -- SCF not finished?"; exit 1; }
  cp $m/scf/CHGCAR $m/dos/CHGCAR
  (cd $m/dos && qsub submit.sh)
done
