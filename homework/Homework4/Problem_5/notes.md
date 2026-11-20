# HW4 Problem 5 notes

## Part A
- Divacancy: removed perfect-lattice sites 53 and 54 (1NN pair)
- Initial vacancy separation: 2.8638 A
- 106 atoms, relaxed E = -377.950742 eV

## Why S01 was dropped (keep for discussion questions)
S01 (atom 53 -> vacancy 1; mirror twin S02, atom 54 -> vacancy 2) converged to a
first-order saddle: curvature -0.207 eV/A^2, barrier 0.0443 eV. Downhill
relaxation from it (04_relax_from_saddles.py) showed it does NOT connect two
distinct minima:

- State A = the original divacancy (E = -377.95157 eV, vacancy sites 53, 54)
- "State B" E = -377.90633 eV, slightly ABOVE the saddle (-377.90647 eV)
  -> reverse barrier = -0.00014 eV (unphysical for a real minimum)
- State B vacancy sites = 53, 54 (same as A); endpoint RMSD only 0.107 A
- Plus-side FIRE relaxation stopped after 5 steps: fmax already < 0.02 eV/A
  because the region is nearly flat, not because it reached a minimum

Interpretation: S01/S02 sit on a very flat shoulder of the energy surface. One
side relaxes back to the starting divacancy; the other side has no real second
minimum. A NEB between these endpoints would be meaningless.

Discussion point: a converged Dimer search with negative curvature only proves
a first-order saddle exists. It doesn't guarantee the saddle links two distinct
minima or matters for kinetics. You have to relax downhill along +/- e_TS to
check.

## S04 (offaxis_high) - kept
- Atom at site 19 hops into vacancy at site 53: vacancy sites (53, 54) -> (19, 54)
- Separation 2.8638 A (1NN) -> 4.9602 A (3NN): partial dissociation of the divacancy
- E_A = -377.95121, E_B = -377.89871 (B is 0.0525 eV higher: lost binding)
- Dimer forward barrier 0.3979 eV, reverse 0.3454 eV
- Duplicate: S06 (atom 90 -> vacancy 2) has the same barrier (0.3976) and
  curvature (-0.920), i.e. the same mechanism from the other vacancy

## S03 (alt_1) - kept (replaced S01)
- Atom at site 30 (far end of the divacancy axis) hops into vacancy at site 53:
  vacancy sites (53, 54) -> (30, 54)
- Separation 2.8638 A (1NN) -> 5.7276 A (4NN, = 2 x 1NN, collinear along <110>):
  the vacancies end up on one straight line, with site 53 (now refilled by the
  hopping atom) between them
- E_A = -377.95073, E_B = -377.88765 (B is 0.0631 eV higher than A)
- Dimer forward barrier 0.3902 eV, reverse 0.3271 eV
- Distinct from S04: different final vacancy site (30 vs 19), different final
  separation (4NN vs 3NN), different B energy (+0.063 vs +0.053 eV),
  different curvature (-1.026 vs -0.918 eV/A^2)

## Part F - Dimer vs NEB
| mechanism | search | Ea Dimer (eV) | Ea NEB (eV) | abs diff (eV) | A sep (A) | B sep (A) |
|---|---|---|---|---|---|---|
| alt_1 | S03 | 0.3902 | 0.3964 | 0.0062 | 2.864 (1NN) | 5.728 (4NN) |
| offaxis_high | S04 | 0.3979 | 0.4051 | 0.0071 | 2.864 (1NN) | 4.960 (3NN) |
- NEB is ~6-7 meV higher for both; agreement < 2%, so the same saddle was found.
- Possible reasons for the small offset: NEB fmax 0.04 vs Dimer 0.03 eV/A; only 7 images.
- Figure: results/mechanisms/dimer_vs_neb.png; table: results/mechanisms/dimer_vs_neb_comparison.csv
