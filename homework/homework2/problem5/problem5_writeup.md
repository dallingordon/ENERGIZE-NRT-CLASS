# HW2 Problem 5 — write-up draft

## Submit checklist
- [ ] INCAR + KPOINTS for both materials (`aluminum/{scf,dos}/`, `silicon/{scf,dos}/`)
- [ ] Python DOS-analysis script (`problem5_analysis_template.py`)
- [ ] `problem5_Al_vs_Si_DOS.png`
- [ ] Comparison table (below / `problem5_comparison_table.csv`)
- [ ] 150–250 word interpretation (below)

## Settings used
| | Al | Si |
|---|---|---|
| Structure | 1-atom primitive FCC, a = 4.05 Å | relaxed CONTCAR from problem 4 (8 atoms) |
| SCF k-mesh | 11×11×11 Gamma | 5×5×5 Gamma |
| DOS k-mesh | 21×21×21 Gamma | 9×9×9 Gamma |
| SCF INCAR | ENCUT=400, EDIFF=1E-6, PREC=Accurate, LREAL=Auto, LWAVE/LCHARG=.TRUE. | same |
| DOS INCAR | + ICHARG=11, NEDOS=2000, ISMEAR=0, SIGMA=0.05 eV | same |

## A4. SCF convergence
- Al: converged in 10 electronic steps, F = -3.754 eV
- Si: converged in 11 electronic steps, F = -43.377 eV (matches problem 4 final energy)

## C4. Fermi energies (before shifting)
- Al: 7.949 eV (DOS run; SCF run gave 7.935 eV)
- Si: 6.034 eV (DOS run; SCF run gave 5.932 eV)

## E. Comparison table
| Material | Fermi energy (eV) | DOS at E_F (states/eV/atom) | Calculated gap (eV) | Electronic classification |
|---|---|---|---|---|
| Al | 7.949 | 0.344 | 0 | metal |
| Si | 6.034 | 0.000 | 0.51 | semiconductor |

Si band edges: VBM = -0.18 eV, CBM = +0.33 eV (relative to E_F). Experimental gap ≈ 1.1 eV.

## F. Interpretation (150–250 words)
1. What feature of the DOS near the Fermi level distinguishes aluminum from silicon?
    The DOS at E_F for aluminum has a positive value while there is a gap between positive values around E_F for silicon where the DOS is zero.

2. Why can aluminum conduct electrically without electrons first being excited across an energy gap?
    Aluminum has unoccupied states contiguous to E_F requiring very little energy of an electron to enter.  The gap has no states.  Electrons have nothing to occupy.

3. How does the calculated silicon band gap compare with experiment?
    We calculated just about half the experimental gap (0.508 eV).  This boils down to the model we chose (PAW_PBE in the POTCAR).  DFT is calculating ground states and doesn't take into consideration excited states, which as seen here results in underestimation.

4. Would the DOS alone be sufficient to determine whether an unknown material is metallic or semiconducting? Explain what feature you would examine.
    the DOS at E_F; I imagine there would be plenty of false negatives, semiconducting materials with smaller gaps could appear as metallic.
