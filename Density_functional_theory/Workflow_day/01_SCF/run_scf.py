#!/usr/bin/env python3
"""DEMO 1: A toy self-consistent-field (SCF) calculation.

TEACHING GOAL
-------------
Show that an electronic Hamiltonian depends on the electron density/charge,
while the Hamiltonian's eigenvectors generate a new density.  SCF repeatedly
updates the density until input and output agree.

WHAT TO CHANGE LIVE
-------------------
Edit INCAR rather than this code whenever possible:
  AMIX  : 0.05 -> slow; 0.30 -> efficient; 0.70/0.90 -> oscillatory/unstable
  EDIFF : tighter/looser convergence tolerance
  NELM  : maximum number of electronic iterations
  TOY_INITIAL_CHARGE_A : changes the initial density guess

ENCUT is read only for VASP familiarity; this tiny backend does not use a
plane-wave basis.
"""
from pathlib import Path
import sys, csv
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from toy_dft_engine import parse_incar, parse_poscar, pair_vector_and_distance, get_float, get_int, scf_solve, write_toy_outcar, electronic_energy, TOTAL_ELECTRONS

RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
incar = parse_incar(HERE / "INCAR")
structure = parse_poscar(HERE / "POSCAR")
_, distance = pair_vector_and_distance(structure)

amix = get_float(incar, "AMIX", 0.30)
ediff = get_float(incar, "EDIFF", 1e-6)
nelm = get_int(incar, "NELM", 100)
q0a = get_float(incar, "TOY_INITIAL_CHARGE_A", 1.80)
q0b = TOTAL_ELECTRONS - q0a

# Also store the initial single-particle orbital so students can compare the
# starting and converged wavefunctions/orbitals.  In this toy model, the
# occupied orbital is represented in a two-site basis (A and B) and then
# expanded onto simple Gaussian-like atom-centered basis functions for plotting.
initial_energy, initial_eigvals, initial_eigvecs, initial_qout = electronic_energy(distance, np.array([q0a, q0b], dtype=float))

res = scf_solve(distance, amix=amix, ediff=ediff, nelm=nelm, initial_charge_a=q0a)
h = res["history"]

print("\n" + "=" * 78)
print("TOY SCF DEMO -- NOT VASP")
print("=" * 78)
for row in h:
    de = row["dE"]
    de_txt = "   --------" if not np.isfinite(de) else f"{de:12.5e}"
    print(f"DAV: {row['iteration']:3d}   E = {row['energy']:12.7f}   dE = {de_txt}   "
          f"rms(q) = {row['residual']:10.3e}")
print("-" * 78)
print("SCF CONVERGED" if res["converged"] else "SCF DID NOT CONVERGE WITHIN NELM")
print(f"Iterations: {res['iterations']}")
print(f"Final charges: A={res['charges'][0]:.6f}, B={res['charges'][1]:.6f}")
print(f"Final toy electronic energy: {res['energy']:.8f} eV")

# Save VASP-like OSZICAR
with (HERE / "OSZICAR").open("w") as f:
    f.write("TOY EDUCATIONAL OUTPUT -- NOT A VASP OSZICAR\n")
    for row in h:
        de = row["dE"]
        f.write(f"DAV: {row['iteration']:3d} E={row['energy']: .10f} dE={de: .5e} rms={row['residual']: .5e}\n")

# Save history CSV
with (RESULTS / "scf_history.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(h[0].keys()))
    writer.writeheader(); writer.writerows(h)

write_toy_outcar(HERE / "OUTCAR", "Toy SCF calculation", [
    f"Pair distance = {distance:.6f} Angstrom",
    f"AMIX = {amix}", f"EDIFF = {ediff}", f"NELM = {nelm}",
    f"Converged = {res['converged']}", f"Iterations = {res['iterations']}",
    f"Final charges = {res['charges']}", f"Final eigenvalues = {res['eigvals']}",
    f"Final electronic energy = {res['energy']:.10f} toy eV",
])

its = np.array([r["iteration"] for r in h])
energies = np.array([r["energy"] for r in h])
qa = np.array([r["charge_a"] for r in h])
qb = np.array([r["charge_b"] for r in h])
resid = np.array([max(r["residual"], 1e-16) for r in h])
qouta = np.array([r["output_charge_a"] for r in h])

# Build a simple real-space visualization of the occupied orbital / wavefunction
# from the initial and converged Hamiltonians.  This is not a real Kohn-Sham
# orbital in a plane-wave basis; it is a pedagogical real-space plot obtained by
# expanding the occupied two-site eigenvector onto atom-centered Gaussians.
initial_occ = initial_eigvecs[:, 0]
final_occ = res["eigvecs"][:, 0]
center_a = -0.5 * distance
center_b = +0.5 * distance
x = np.linspace(center_a - 2.5, center_b + 2.5, 800)
basis_width = max(0.35, 0.22 * distance)
phi_a = np.exp(-0.5 * ((x - center_a) / basis_width) ** 2)
phi_b = np.exp(-0.5 * ((x - center_b) / basis_width) ** 2)
psi_initial = initial_occ[0] * phi_a + initial_occ[1] * phi_b
psi_final = final_occ[0] * phi_a + final_occ[1] * phi_b
# Normalize for easier visual comparison
for arr_name in ["psi_initial", "psi_final"]:
    arr = locals()[arr_name]
    norm = np.sqrt(np.trapezoid(arr**2, x))
    if norm > 1e-14:
        locals()[arr_name] = arr / norm
psi_initial = locals()["psi_initial"]
psi_final = locals()["psi_final"]

fig, axes = plt.subplots(3, 2, figsize=(13, 11))
axes[0,0].plot(its, qa, "o-", label="input charge A")
axes[0,0].plot(its, qb, "o-", label="input charge B")
axes[0,0].axhline(res["charges"][0], ls="--", alpha=.5)
axes[0,0].axhline(res["charges"][1], ls="--", alpha=.5)
axes[0,0].set(xlabel="SCF iteration", ylabel="site electron population", title="Density/charge convergence")
axes[0,0].legend()

axes[0,1].plot(its, energies, "o-")
axes[0,1].set(xlabel="SCF iteration", ylabel="toy electronic energy (eV)", title="Energy convergence")

axes[1,0].semilogy(its, resid, "o-")
axes[1,0].axhline(ediff, ls="--", label="EDIFF-like target")
axes[1,0].set(xlabel="SCF iteration", ylabel="density residual", title="Self-consistency residual")
axes[1,0].legend()

axes[1,1].plot(qa, qouta, "o-")
lo = min(qa.min(), qouta.min()); hi = max(qa.max(), qouta.max())
axes[1,1].plot([lo,hi],[lo,hi], "--")
axes[1,1].set(xlabel="input charge A", ylabel="Hamiltonian output charge A", title="Self-consistency map")

axes[2,0].plot(x, psi_initial, label="initial occupied orbital", lw=2)
axes[2,0].plot(x, psi_final, label="converged occupied orbital", lw=2)
axes[2,0].axvline(center_a, ls="--", alpha=0.4)
axes[2,0].axvline(center_b, ls="--", alpha=0.4)
axes[2,0].set(xlabel="position along bond (Å)", ylabel="toy orbital amplitude", title="Initial vs converged wavefunction")
axes[2,0].legend()

axes[2,1].bar(["A", "B"], np.abs(initial_occ)**2, width=0.35, label="initial |c|²", alpha=0.7)
axes[2,1].bar(["A", "B"], np.abs(final_occ)**2, width=0.22, label="converged |c|²", alpha=0.9)
axes[2,1].set(ylabel="orbital weight on site", title="Occupied-orbital composition")
axes[2,1].legend()

fig.suptitle(f"Toy SCF loop: AMIX={amix}, pair distance={distance:.2f} Å")
fig.tight_layout()
fig.savefig(RESULTS / "01_scf_convergence.png", dpi=180)
plt.close(fig)

print(f"\nSaved: {RESULTS / '01_scf_convergence.png'}")
print(f"Saved: {HERE / 'OSZICAR'}")
print(f"Saved: {HERE / 'OUTCAR'}")
