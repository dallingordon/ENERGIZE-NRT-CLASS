# Homework 3 — Problems 4 and 5 (written responses)
Dallin Gordon

---

## Problem 4 — Thermal expansion of FCC Al (EAM, NPT)

### B. Equilibration and fluctuations
The instantaneous pressure responds to all the other changes happening in the system. At each time step the volume and the temperature are updated, and the "inertia" of the whole system will cause it to fluctuate, but with enough time those fluctuations settle about a mean — the mean that we picked. The degree of the fluctuation has much to do with the properties of the material, in this case the stiffness of aluminum. The second half of the simulation has a very steady mean about our preset pressure and temperature, as does our volume.

### C/D. Lattice parameter and thermal expansion coefficient
Plot: `al_thermal_expansion.png` (averaged data, SEM error bars, linear fit, residuals).

| Target T (K) | Mean lattice parameter a (Å) | SEM (Å) |
|---:|---:|---:|
| 100 | 4.090351 | 0.000125 |
| 300 | 4.113028 | 0.000175 |
| 500 | 4.141542 | 0.000363 |
| 700 | 4.175988 | 0.000404 |

Linear fit a(T) = a_ref + (da/dT)·T:

- da/dT = 1.4271 × 10⁻⁴ Å/K
- a_ref = 4.07314 Å
- α = (1/a_ref)(da/dT) = 3.50 × 10⁻⁵ K⁻¹

### E. Ensemble choice
Studying thermal expansion means we are interested in how a material expands — how its volume changes — as temperature increases. We care about volume explicitly at a particular temperature, so we let it vary and pick our temperatures. NVT would mean we selected both temperature and volume, which goes against the question we are asking. In NVT we could still pick our temperature, but we would also have to pick our volume, and pressure would be the property we would measure. We would basically be constraining the thermal expansion to whatever volume we chose, so it wouldn't give us the output we need to answer our question.

### F. Interpretation
The average lattice parameter does increase monotonically with temperature, from 4.0904 Å at 100 K to 4.1760 Å at 700 K, but not linearly: it grows faster than the linear fit, as the residuals show (above the line at 100 and 700 K, below it at 300 and 500 K). Time averaging is required because there are fluctuations happening all the time; single points can be high or low and are less likely to capture the property we care about, while an average over the production data overcomes those fluctuations. NPT permits thermal expansion to occur naturally because pressure and temperature are held constant, so volume is the variable that has to change to keep them constant. Increasing temperature increases the amplitude at which the atoms vibrate, and since it is easier to move atoms apart than closer together, the crystal expands. If the same heating study were run in NVT, the volume would be held constant, so the pressure would increase with temperature instead; we would be constraining the thermal expansion to whatever volume we chose. Finally, fitting a single linear α over 100–700 K assumes the lattice parameter grows linearly with temperature. It appears not to, so one α (3.50 × 10⁻⁵ K⁻¹) underestimates the expansion rate at high temperature and overestimates it at low temperature.

---

## Problem 5 — Fitted LJ vs. ASE vs. VASP for Al₂

### A. Lennard-Jones fit
| Quantity | Value |
|---|---:|
| ε | 0.950584 eV |
| σ | 2.439971 Å |
| RMSE | 0.009027 eV |
| Analytical r_eq = 2^(1/6) σ | 2.738775 Å |

It is a great fit: the RMSE is two orders of magnitude smaller than ε, which is a very small error (see `results/lj_fit.png`).

### B. Al₂ dimer construction
Atoms placed at (8.25, 10, 10) Å and (11.75, 10, 10) Å in a 20 Å cubic cell with pbc=False, giving an initial separation of 3.500 Å centered in the box.

### C. ASE geometry optimization
BFGS optimizer, trajectory and log saved to `results/`, run to fmax ≤ 0.01 eV/Å. Converged in 8 steps: final d = 2.738827 Å, max force = 0.00048 eV/Å.

In both cases the underlying parameterization is the same. In Part A we solved for force = 0 (dU/dr = 0) in closed form, whereas Part C was performed iteratively, with BFGS (a quasi-Newton method) stepping against the gradient — along the force — until the maximum force fell below our threshold of 0.01 eV/Å. The only difference is the method of solving, not the modelling framework, so you would expect them to return the same result as long as the solving methods are sound. Both found equilibrium at 2.7388 Å (analytical 2.738775 Å vs. ASE 2.738827 Å); the ~5 × 10⁻⁵ Å difference is only in the less significant digits, because the optimizer stops at the force threshold rather than at exactly zero force.

### D. VASP geometry optimization
POSCAR (same starting geometry: 20 Å cubic cell, 3.50 Å separation, centered) and submit.sh written by me; supplied INCAR/KPOINTS/POTCAR used. Converged ("reached required accuracy") in 6 ionic steps; CONTCAR and OUTCAR generated. Spin-polarized result: magnetic moment 2.0 (triplet).

### E. Comparison
| Method | Al–Al distance (Å) |
|---|---:|
| Analytical LJ | 2.738775 |
| ASE + fitted LJ | 2.738827 |
| VASP / PBE | 2.756252 |

ASE-LJ vs. VASP: −0.017425 Å (0.632 %). Plots: `results/lj_vs_vasp_comparison.png`, `vasp_energy_convergence.png`, `vasp_force_convergence.png`.

### F. Interpretation
The ASE-LJ optimized geometry should agree closely with the analytical LJ minimum because the math is the same; the difference is only how the optimum is solved for — in closed form for the analytical minimum, and iteratively for ASE, stopping once we reach a threshold that we decide on (fmax ≤ 0.01 eV/Å). That is why they agree to within ~5 × 10⁻⁵ Å (2.73878 vs. 2.73883 Å). The fitted LJ geometry can still disagree with VASP (2.756 Å) even though the fit itself is excellent. LJ is a much simpler model, and being well-fit only means our LJ model passes through our synthetic data points, which were not generated by VASP. VASP takes into account the electronic physics and models much more complexity than what we used to fit our LJ model. In particular, a two-body LJ description of metallic Al is missing electronic-level physics: orbitals and spin (VASP found a magnetic dimer). Bonds are only expressed as distances between atoms, and LJ also doesn't take into account neighbors, which has a large effect on metallic bonds. That our synthetic-data-trained model worked on synthetic data makes sense, but on data that wasn't used to fit our model we have no guarantee — for example a surface, or larger groups of atoms. A potential therefore has to be validated on configurations and properties outside its fitting procedure before it can be trusted.
