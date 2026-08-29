#!/usr/bin/env python3
"""DEMO 3: Toy electronic density of states (DOS).

TEACHING GOAL
-------------
Show that a DOS is constructed from electronic eigenvalues sampled over many
k-points, then broadened/smoothed into a continuous-looking spectrum.

INPUT STRUCTURE
---------------
The script automatically uses ../02_geometry_optimization/CONTCAR when it
exists, so the DOS is computed at the relaxed geometry from Demo 2.

WHAT TO CHANGE LIVE
-------------------
  KPOINTS mesh : 2x2x2 -> sparse/jagged, 6x6x6 -> better, 12x12x12 -> converged-looking
  SIGMA        : controls Gaussian broadening of each discrete eigenvalue
  NEDOS        : number of energy-grid points used to write/plot DOSCAR
"""
from pathlib import Path
import sys, csv
import numpy as np
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent; ROOT=HERE.parent
sys.path.insert(0,str(ROOT))
from toy_dft_engine import (parse_incar,parse_poscar,parse_kpoints_mesh,pair_vector_and_distance,get_float,get_int,
    scf_solve,kmesh_points,bloch_bands,gaussian_dos,write_toy_outcar)

RESULTS=HERE/'results'; RESULTS.mkdir(exist_ok=True)
source_poscar=HERE/'POSCAR'
if not source_poscar.exists():
    relaxed=ROOT/'02_geometry_optimization'/'CONTCAR'
    source_poscar=relaxed if relaxed.exists() else ROOT/'01_SCF'/'POSCAR'
structure=parse_poscar(source_poscar); _,R=pair_vector_and_distance(structure)
incar=parse_incar(HERE/'INCAR'); mesh=parse_kpoints_mesh(HERE/'KPOINTS')

sigma=get_float(incar,'SIGMA',0.08); nedos=get_int(incar,'NEDOS',1000)
scf=scf_solve(R,amix=get_float(incar,'AMIX',0.30),ediff=get_float(incar,'EDIFF',1e-8),nelm=get_int(incar,'NELM',100),initial_charge_a=1.0)
kpts=kmesh_points(mesh); bands,weights_a=bloch_bands(R,scf['charges'],kpts)
dos=gaussian_dos(bands,weights_a,nedos=nedos,sigma=sigma)

valence_max=float(bands[:,0].max()); conduction_min=float(bands[:,1].min()); gap=conduction_min-valence_max
fermi=0.5*(valence_max+conduction_min)

print("\n"+'='*78); print('TOY DOS CALCULATION -- NOT VASP'); print('='*78)
print(f'Structure source: {source_poscar}')
print(f'Relaxed pair distance: {R:.6f} Å')
print(f'KPOINTS mesh: {mesh[0]} x {mesh[1]} x {mesh[2]} = {len(kpts)} points')
print(f'SIGMA: {sigma} eV; NEDOS: {nedos}')
print(f'Valence-band maximum: {valence_max:.6f} eV')
print(f'Conduction-band minimum: {conduction_min:.6f} eV')
print(f'Toy band gap: {gap:.6f} eV')

# VASP-like DOSCAR analogue
with (HERE/'DOSCAR').open('w') as f:
    f.write('TOY EDUCATIONAL OUTPUT -- NOT A VASP DOSCAR\n')
    f.write(f'# mesh={mesh} sigma={sigma} eV fermi={fermi:.10f} eV\n')
    f.write('# E_minus_Ef  total_DOS  integrated_DOS  PDOS_A  PDOS_B\n')
    for e,td,integ,pa,pb in zip(dos['energy']-fermi,dos['total'],dos['integrated'],dos['pdos_a'],dos['pdos_b']):
        f.write(f'{e: .10f} {td: .10e} {integ: .10e} {pa: .10e} {pb: .10e}\n')

# Also save all discrete eigenvalues so students can see the raw ingredients.
np.savetxt(RESULTS/'discrete_eigenvalues.csv',np.column_stack([np.arange(len(kpts)),kpts,bands]),delimiter=',',
           header='k_index,kx,ky,kz,band_1_eV,band_2_eV',comments='')

write_toy_outcar(HERE/'OUTCAR','Toy DOS calculation',[
    f'Structure source = {source_poscar}',f'Pair distance = {R:.8f} Angstrom',f'KPOINT mesh = {mesh}',
    f'Number of k points = {len(kpts)}',f'SIGMA = {sigma}',f'NEDOS = {nedos}',
    f'Valence maximum = {valence_max:.8f} eV',f'Conduction minimum = {conduction_min:.8f} eV',f'Toy gap = {gap:.8f} eV'])

# For a compact band-view, sort k-points by radius from Gamma. This is NOT a high-symmetry band path,
# just a visualization of the sampled eigenvalue cloud.
kdist=np.linalg.norm(kpts,axis=1); order=np.argsort(kdist)
fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].scatter(kdist[order],bands[order,0]-fermi,s=10,label='valence band')
ax[0].scatter(kdist[order],bands[order,1]-fermi,s=10,label='conduction band')
ax[0].axhline(0,ls='--'); ax[0].set(xlabel='distance from Γ in toy k-space',ylabel='energy - $E_F$ (eV)',title='Discrete eigenvalues over k-points'); ax[0].legend()

allE=(bands-fermi).reshape(-1)
ax[1].hist(allE,bins=max(12,int(np.sqrt(len(allE)))),density=True,alpha=.7)
ax[1].axvspan(valence_max-fermi,conduction_min-fermi,alpha=.12)
ax[1].set(xlabel='energy - $E_F$ (eV)',ylabel='raw eigenvalue histogram',title='DOS starts as a distribution of eigenvalues')

ax[2].plot(dos['total'],dos['energy']-fermi,label='total DOS')
ax[2].plot(dos['pdos_a'],dos['energy']-fermi,ls='--',label='A/site projected')
ax[2].plot(dos['pdos_b'],dos['energy']-fermi,ls='--',label='B/site projected')
ax[2].axhline(0,ls='--'); ax[2].axhspan(valence_max-fermi,conduction_min-fermi,alpha=.12,label='toy gap')
ax[2].set(xlabel='DOS (states / toy eV)',ylabel='energy - $E_F$ (eV)',title=f'Gaussian-broadened DOS (σ={sigma})'); ax[2].legend()
fig.suptitle(f'Toy DOS from a {mesh[0]}×{mesh[1]}×{mesh[2]} k-point mesh')
fig.tight_layout(); fig.savefig(RESULTS/'03_dos_workflow.png',dpi=180); plt.close(fig)

print(f"\nSaved: {HERE/'DOSCAR'}")
print(f"Saved: {RESULTS/'03_dos_workflow.png'}")
