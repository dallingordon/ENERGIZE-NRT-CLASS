# Homework 4 

# Problem 4 — Monovacancy migration in FCC Al (NEB → KMC)

## A. Vacancy migration endpoints

- Perfect supercell atoms: **108**; defect atoms: **107**
- Nearest-neighbor jump distance: **2.8638 Å** (expected a/√2 = 2.8638 Å; matches)

**Why should the relaxed initial and final states have nearly the same energy? (2–4 sentences)**

The initial state is a perfect crystal with a single vacancy, as is the final state.  They are strucuturally nearly identical, a single vacancy, just at a different position.  

## B. NEB run

- Files generated: `results/neb_energies.csv`, `results/neb_final_images.traj`, `results/vacancy_neb.traj`, `results/vacancy_neb.log`
- Relaxed endpoint energy difference: **9.7 × 10⁻¹³ eV**

The difference is rounding, calculating iteratively will be off even if they are symmetrical/equivalent. 

## C. NEB migration barrier

- Saddle image index: **3** (of 7 images)
- E_a^NEB = **0.359 eV**
- Figure: `results/al_vacancy_neb.png`

## D. KMC temperatures

Selected temperatures: **550, 600, 650, 700, 750 K**

## E. Vacancy MSD and diffusivity

Figure: `results/vacancy_msd.png`

## F. Arrhenius activation energy and comparison with experiment

Figure: `results/vacancy_arrhenius.png`

### Activation energies

| Quantity | Value (eV) |
|---|---|
| E_a^NEB | 0.3595 |
| E_a^KMC (Arrhenius fit) | 0.3599 |
| E_a^exp (Khellaf, Seeger & Emrick) | 0.61 ± 0.02 |
| E_a^KMC − E_a^NEB | 0.0004 |
| E_a^NEB − E_a^exp | −0.251 |

### Diffusivities

| T (K) | D_v^KMC (m²/s) | D_v^exp (m²/s) | D_KMC / D_exp |
|---|---|---|---|
| 550 | 3.32 × 10⁻⁹ | 1.54 × 10⁻¹¹ | 215 |
| 600 | 5.52 × 10⁻⁹ | 4.51 × 10⁻¹¹ | 122 |
| 650 | 8.55 × 10⁻⁹ | 1.12 × 10⁻¹⁰ | 76 |
| 700 | 1.76 × 10⁻⁸ | 2.43 × 10⁻¹⁰ | 72 |
| 750 | 2.36 × 10⁻⁸ | 4.78 × 10⁻¹⁰ | 49 |

**Over- or under-prediction, and by roughly what factor:**

Over prediction, and it seems to over predict lower temperatures proportionally higher (>200x at 550, and ~50x at 750).

## G. Interpretation (150–250 words)

KMC and NEB agree on V down to 3 decimal points (0.3599 eV for KMC and 0.3595 eV for NEB barrier). Both are below the experimental migration energy of 0.61 eV. This is likely an inaccuracy propagated in from the selected potential. The slope of the Arrhenius is just that, a slope, and has in its Numerator the error from the potential. Thus the multiplicative over estimation that scales with temperature.   MD has to deal on the scale of femto seconds, whereas we skip through time to the interesting events in KMC.  The Temperature for KMC only depends on the E_a it started with, and it just gives that back.  Its own barrier reflects that.  This had one fixed barrier, a perfect lattice, a single vacancy, no vibrations, and an experimental v_0.  

---

# Problem 5 — Divacancy rearrangements in FCC Al (Dimer vs NEB)

## A. Divacancy construction

- Perfect atoms: **108**; defect atoms: **106**
- Removed sites: **53, 54** (1NN pair)
- Initial vacancy–vacancy separation: **2.864 Å** (1NN)
- Relaxed energy: **−377.9507 eV**

**Why is a nearest-neighbor divacancy a useful starting point for exploring nearby rearrangements?**

It allows for a lot of exploration.  Since there are many vacancies that can be filled there are many physically distinct outcomes possible.    

## B. Dimer searches

| Search | Atom → vacancy | Converged | Curvature (eV/Å²) | Barrier from start (eV) |
|---|---|---|---|---|
| S01 | 53 → 1 | yes | −0.207 | 0.044 |
| S02 | 54 → 2 | yes | −0.207 | 0.044 |
| S03 | 30 → 1 | yes | −1.026 | 0.390 |
| S04 | 19 → 1 | yes | −0.918 | 0.397 |
| S05 | 64 → 1 | yes | −1.061 | 0.413 |
| S06 | 90 → 2 | yes | −0.920 | 0.398 |

Figure: `results/dimer/dimer_barriers.png`; RMSD matrix: `results/dimer/saddle_rmsd_matrix_A.csv`

All six searches converged to first-order saddles (negative curvature). S01 and S02 are mirror-equivalent searches with a very small barrier (0.044 eV) and weak curvature; relaxing downhill from them showed they do not connect two distinct minima (one side returns to the starting divacancy, the other sits on a nearly flat shoulder), so they were not used. S04 and S06 are the same mechanism reached from the two different vacancies (0.397 vs 0.398 eV; curvature −0.918 vs −0.920 eV/Å²).


## C. Selected saddles

| Mechanism name | Search |
|---|---|
| alt_1 | S03 |
| offaxis_high | S04 |

**Why these two are physically distinct (use structure AND energy):**

Structurally its a different atom that hops (the indexes are distinct). The curvature at the saddles are significantly different as are the final energies.  

## D. Connected minima (2–4 sentences per mechanism)

| Mechanism | State A vacancies | State B vacancies | A sep (Å) | B sep (Å) | E_B − E_A (eV) |
|---|---|---|---|---|---|
| alt_1 | 53, 54 | 30, 54 | 2.864 (1NN) | 5.728 (4NN) | +0.063 |
| offaxis_high | 53, 54 | 19, 54 | 2.864 (1NN) | 4.960 (3NN) | +0.053 |

**alt_1:**

site 30 hops to site 53, resulting in vacancies at 30 and 54.  

**offaxis_high:**

atom at site 19 hops to site 53, so vacancies at 19 and 54.  

## E. NEB validation

- `neb_results.json` and `neb_final_images.traj` present for both mechanisms: ☑ alt_1 ☑ offaxis_high (both converged)
- Highest-energy NEB image — alt_1: **image 3** (E = −377.5544 eV); offaxis_high: **image 3** (E = −377.5462 eV) — the middle (climbing) image of 7 (0–6)

## F. Dimer vs NEB

Figure: `results/mechanisms/dimer_vs_neb.png`; CSV: `results/mechanisms/dimer_vs_neb_comparison.csv`

| Mechanism | E_a,Dimer (eV) | E_a,NEB (eV) | \|Δ\| (eV) |
|---|---|---|---|
| alt_1 | 0.3902 | 0.3964 | 0.0062 |
| offaxis_high | 0.3979 | 0.4051 | 0.0071 |

NEB recovers the Dimer barriers to within 6–7 meV (< 2 %) for both mechanisms, confirming that both methods located the same saddle points. NEB is slightly higher in both cases, consistent with the looser NEB force tolerance (0.04 vs 0.03 eV/Å for Dimer) and the discrete seven-image band.

## G. Interpretation (150–250 words)

Although all six Dimer searches began from the same relaxed divacancy, they converged to different saddles because each started from a different local perturbation. The Dimer method follows the lowest-curvature mode from wherever it starts, so pushing different atoms toward different vacancies leads uphill along different escape routes. Dimer needs no final state because it uses only local information (forces and curvature at the dimer's current position), whereas NEB optimizes an entire path with fixed endpoints. To find what each saddle connects, the saddle was displaced by ±δ along its unstable eigenmode and relaxed downhill: each side fell into a different basin, giving state A (the original divacancy) and state B (a partially dissociated pair). The same check showed that S01, although a converged saddle, did not link two distinct minima. NEB reproduced both Dimer barriers within about 7 meV, with the small offset attributable to convergence tolerances and the discrete band. NEB additionally provides the full minimum-energy path and the reverse barrier; Dimer, in turn, can discover transitions and product states that were never guessed. Dimer is preferable when exploring an unfamiliar defect, surface, or grain boundary to build an event catalog for off-lattice KMC, where final states are unknown. NEB is preferable when the initial and final states are already known and an accurate barrier and pathway are needed, such as the single vacancy hop in Problem 4.
