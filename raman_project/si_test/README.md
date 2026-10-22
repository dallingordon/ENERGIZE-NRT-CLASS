# Si Raman test (VASP)

End-to-end test of the simulated-Raman pipeline on crystalline Si before quartz.
Si has one Raman-active mode (triply degenerate optical phonon, 520.7 cm-1 experimentally)
and is in the contrastive network's RRUFF library (R040145).

## Run (on the SCC)
    cd raman_project/si_test
    qsub run_pipeline.sh          # log: si_raman.log

Stages (each in its own folder, skipped on resubmit if finished):
1. `01_relax`   - cell + ions relax (PBE, ENCUT 600, 10x10x10 Gamma k-mesh)
2. `02_phonon`  - Gamma phonons by DFPT (IBRION=8, NWRITE=3)
3. `03_raman/`  - `../raman.py displace`: +/-0.01 A along each optical mode
4. `03_raman/mode_*` - LEPSILON (eps_inf) at each displaced structure (6 runs for Si)
5. `../raman.py collect` -> `si_raman_modes.csv`, `si_raman_spectrum.csv`, `si_raman.png`

Expected: optical mode ~505-515 cm-1 (PBE runs slightly soft), single Lorentzian peak.

## Then: through the network
From the raman_spectra_matching repo root:
    python infer.py /path/to/si_raman_spectrum.csv --device cpu --topk 5

## For quartz
Same pipeline: swap POSCAR (from Quartz__0006362.cif), POTCAR = cat Si O, adjust k-mesh
(~6x6x5), and the job name. 24 optical modes -> 48 LEPSILON runs.
