# HW2 Problem 4 — write-up draft

## Summary table
```python problem4_analysis_template.py;```
Output:
Per-atom displacement (A):
  atom 1 (Si): 0.0249
  atom 2 (Si): 0.1009
  atom 3 (Si): 0.1254
  atom 4 (Si): 0.1123
  atom 5 (Si): 0.1882
  atom 6 (Si): 0.1560
  atom 7 (Si): 0.1752
  atom 8 (Si): 0.1801
Ionic steps: 13
Initial energy: -41.989828 eV
Final energy:   -43.376836 eV
Final max force:0.015692 eV/A
Largest displacement: 0.188185 A (atom 5)
SCF iterations per ionic step: [13, 9, 7, 5, 7, 4, 6, 6, 5, 6, 6, 3, 4]

## A4. Did it reach the force criterion?
(evidence: `reached required accuracy - stopping structural energy minimisation`, 13 of NSW=30 steps)

## B. Starting vs final structure
(which atoms moved most, RDF comparison)

## E. Electronic vs ionic loops
(SCF iterations per ionic step: 13, 9, 7, 5, 7, 4, 6, 6, 5, 6, 6, 3, 4)

## F. Interpretation (150–250 words)
1. Why must the electronic structure be converged before VASP can update atomic positions?
    If the electronic structure (the inner loop) hasn't converged, the gradients used to update atomic positions would be incorrect, and future iterations would lead to compounding errors.  It seems like Expectation Maximization to me, but one of the EM steps is fully converging.  

2. Why does the total energy generally decrease during the optimization?
This is a result of our modelling, particles want to go to lower energy states. So we set our optimizer to minimize energy.  It will tend toward LOCAL minima based on the gradient of the loss function (energy). Sufficiently large gradient steps can over shoot and end up in higher energy configurations, so the loss can be noisy and not strictly decrease during relaxation.   

3. Why is a force criterion (EDIFFG) more useful than stopping when energy changes only slightly?
A small change in energy doesn't mean we have minimized we may be at a plateau.  We are more interested in a configuration where the force (the gradient of the loss, energy) on each atom is low, in our case below 0.02 per the EDIFFG parameter.

4. Based on final forces and RDF, is the final structure relaxed to the crystalline ground state?
Our final max atomic force is 0.016, below our EDIFFG value, and so we are at least at a local minima.  However our RDF png looks a whole lot like what we expect Silicon to look like, which suggests we are at a useful one.