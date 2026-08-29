#!/usr/bin/env python3
"""DEMO 2: Toy geometry optimization built on repeated SCF calculations.

TEACHING GOAL
-------------
Show the nested workflow:
    geometry -> SCF -> energy/forces -> move atoms -> new geometry -> SCF ...

INPUT STRUCTURE
---------------
By default this script reads ../01_SCF/POSCAR, so Demo 2 begins from the exact
same structure used in Demo 1.  If a local POSCAR exists, it takes priority.

WHAT TO CHANGE LIVE
-------------------
Edit INCAR:
  POTIM  : toy ionic step size; 0.10 slow, 0.25 good, 1.00 unstable
  EDIFFG : force stopping criterion
  NSW    : maximum ionic steps
  EDIFF/AMIX/NELM : electronic convergence inside every ionic step
"""
from pathlib import Path
import sys, csv, shutil
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from toy_dft_engine import (parse_incar, parse_poscar, pair_vector_and_distance, get_float, get_int,
    total_energy_for_distance, radial_force, structure_with_distance, write_poscar, write_toy_outcar)

RESULTS = HERE / "results"; RESULTS.mkdir(exist_ok=True)
source_poscar = HERE / "POSCAR"
if not source_poscar.exists():
    source_poscar = ROOT / "01_SCF" / "POSCAR"
structure = parse_poscar(source_poscar)
_, R = pair_vector_and_distance(structure)
incar = parse_incar(HERE / "INCAR")

potim = get_float(incar, "POTIM", 0.25)
ediffg = abs(get_float(incar, "EDIFFG", -0.02))
nsw = get_int(incar, "NSW", 40)
scf_kw = dict(
    amix=get_float(incar, "AMIX", 0.30),
    ediff=get_float(incar, "EDIFF", 1e-7),
    nelm=get_int(incar, "NELM", 100),
    initial_charge_a=1.0,
)

history=[]
print("\n" + "="*78)
print("TOY GEOMETRY OPTIMIZATION -- NOT VASP")
print("="*78)
for step in range(1, nsw+1):
    E, scf = total_energy_for_distance(R, scf_kw)
    F = radial_force(R, delta=1e-3, scf_kwargs=scf_kw)
    history.append({"ionic_step":step, "distance_A":R, "total_energy_eV":E,
                    "radial_force_eV_A":F, "scf_iterations":scf["iterations"],
                    "scf_converged":scf["converged"]})
    print(f"{step:3d} F= {E: .8f}  |F|max= {abs(F):.5e}  R= {R:.6f} Å  SCF={scf['iterations']:d}")
    if abs(F) < ediffg:
        print("IONIC RELAXATION CONVERGED")
        break
    R = R + potim * F
    if R < 0.8 or R > 6.0 or not np.isfinite(R):
        print("IONIC UPDATE BECAME UNPHYSICAL -- reduce POTIM")
        break

final_structure = structure_with_distance(structure, R)
write_poscar(final_structure, HERE / "CONTCAR", comment="Toy relaxed structure -- educational, not VASP")

with (HERE / "OSZICAR").open("w") as f:
    f.write("TOY EDUCATIONAL OUTPUT -- NOT A VASP OSZICAR\n")
    for row in history:
        f.write(f"{row['ionic_step']:4d} F={row['total_energy_eV']: .10f} R={row['distance_A']:.8f} "
                f"MAX_FORCE={abs(row['radial_force_eV_A']):.6e}\n")

with (RESULTS / "relaxation_history.csv").open("w", newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

write_toy_outcar(HERE / "OUTCAR", "Toy geometry optimization", [
    f"Starting POSCAR = {source_poscar}", f"POTIM = {potim}", f"EDIFFG = {ediffg}", f"NSW = {nsw}",
    f"Initial distance = {history[0]['distance_A']:.8f} Angstrom",
    f"Final distance = {R:.8f} Angstrom", f"Ionic steps = {len(history)}",
])

steps=np.array([x['ionic_step'] for x in history]); Rs=np.array([x['distance_A'] for x in history])
Es=np.array([x['total_energy_eV'] for x in history]); Fs=np.abs([x['radial_force_eV_A'] for x in history])
scanR=np.linspace(1.55,3.35,160)
scanE=np.array([total_energy_for_distance(r, scf_kw)[0] for r in scanR])

fig,ax=plt.subplots(2,2,figsize=(12,8))
ax[0,0].plot(steps,Es,'o-'); ax[0,0].set(xlabel='ionic step',ylabel='total toy energy (eV)',title='Energy decreases during relaxation')
ax[0,1].semilogy(steps,np.maximum(Fs,1e-12),'o-'); ax[0,1].axhline(ediffg,ls='--',label='EDIFFG'); ax[0,1].set(xlabel='ionic step',ylabel='|force| (eV/Å)',title='Force convergence'); ax[0,1].legend()
ax[1,0].plot(steps,Rs,'o-'); ax[1,0].set(xlabel='ionic step',ylabel='A-B separation (Å)',title='Atomic geometry changes')
ax[1,1].plot(scanR,scanE,label='toy potential-energy surface'); ax[1,1].plot(Rs,Es,'o-',label='optimizer path'); ax[1,1].set(xlabel='A-B separation (Å)',ylabel='total toy energy (eV)',title='Walking downhill on the PES'); ax[1,1].legend()
fig.suptitle(f'Toy geometry optimization: POTIM={potim}')
fig.tight_layout(); fig.savefig(RESULTS/'02_geometry_optimization.png',dpi=180); plt.close(fig)

print(f"\nFinal structure written to {HERE/'CONTCAR'}")
print(f"Saved: {RESULTS/'02_geometry_optimization.png'}")
print("Demo 3 will automatically use this CONTCAR if it exists.")
