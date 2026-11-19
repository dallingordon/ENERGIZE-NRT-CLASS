#!/usr/bin/env python3
"""Shared utilities for Homework Question 5: Al divacancy Dimer + NEB."""

from pathlib import Path
import json
import os
import numpy as np


def make_calculator():
    """Return the calculator selected by environment variables.

    Student/default mode:
        AL_CALCULATOR_MODE=EAM
        AL_EAM_FILE=Al.eam.alloy

    Instructor smoke-test mode:
        AL_CALCULATOR_MODE=EMT
    """
    mode = os.environ.get("AL_CALCULATOR_MODE", "EAM").strip().upper()
    if mode == "EMT":
        from ase.calculators.emt import EMT
        return EMT()
    if mode != "EAM":
        raise ValueError(f"Unknown AL_CALCULATOR_MODE={mode!r}; use EAM or EMT.")

    from ase.calculators.eam import EAM
    potential = Path(os.environ.get("AL_EAM_FILE", "Al_zhou.eam.alloy")).expanduser()
    if not potential.exists():
        raise FileNotFoundError(
            f"EAM potential not found: {potential}\n"
            "Put the instructor-provided potential in this directory as Al.eam.alloy, "
            "or set AL_EAM_FILE to its path."
        )
    return EAM(potential=str(potential.resolve()))


def dump_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with Path(path).open() as f:
        return json.load(f)


def mic_vector(atoms, p_from, p_to):
    """Minimum-image vector p_from -> p_to in Cartesian Angstrom."""
    from ase.geometry import find_mic
    vec = np.asarray(p_to, float) - np.asarray(p_from, float)
    v, _ = find_mic(vec, atoms.cell, pbc=atoms.pbc)
    return np.asarray(v, float)


def mic_distance(atoms, p1, p2):
    return float(np.linalg.norm(mic_vector(atoms, p1, p2)))


def pair_candidates_near_center(atoms, target_distance, tolerance=0.12, max_pairs=12):
    """Nearest-neighbor pairs sorted by how close their midpoint is to box center."""
    cell_center = 0.5 * np.sum(np.asarray(atoms.cell), axis=0)
    out = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            vec = mic_vector(atoms, atoms.positions[i], atoms.positions[j])
            d = float(np.linalg.norm(vec))
            if abs(d - target_distance) <= tolerance:
                midpoint = atoms.positions[i] + 0.5 * vec
                dc = mic_distance(atoms, midpoint, cell_center)
                out.append((dc, d, i, j, midpoint.tolist()))
    out.sort(key=lambda x: x[0])
    return out[:max_pairs]


def choose_anchor_atom(atoms, defect_center):
    """Choose one atom far from the defect to pin and remove global translation."""
    distances = [mic_distance(atoms, p, defect_center) for p in atoms.positions]
    return int(np.argmax(distances))


def nearby_atoms_for_vacancy(atoms, vacancy_position, n=8):
    rows = []
    for i, p in enumerate(atoms.positions):
        vec = mic_vector(atoms, p, vacancy_position)
        d = float(np.linalg.norm(vec))
        rows.append((d, i, vec))
    rows.sort(key=lambda x: x[0])
    return rows[:n]


def rmsd_mic(a, b):
    """RMS Cartesian displacement between same-index structures using MIC."""
    if len(a) != len(b):
        return np.inf
    dsq = []
    for pa, pb in zip(a.positions, b.positions):
        v = mic_vector(a, pa, pb)
        dsq.append(float(np.dot(v, v)))
    return float(np.sqrt(np.mean(dsq)))


def infer_vacancy_sites(perfect, defect):
    """Infer the two unoccupied ideal FCC sites using greedy nearest-site matching.

    This is intended for small relaxations around a defect, not arbitrary diffusion.
    """
    pairs = []
    for ia, p in enumerate(defect.positions):
        for isite, s in enumerate(perfect.positions):
            d = mic_distance(perfect, p, s)
            pairs.append((d, ia, isite))
    pairs.sort(key=lambda x: x[0])
    used_atoms = set()
    used_sites = set()
    for d, ia, isite in pairs:
        if ia not in used_atoms and isite not in used_sites:
            used_atoms.add(ia)
            used_sites.add(isite)
            if len(used_atoms) == len(defect):
                break
    missing = [i for i in range(len(perfect)) if i not in used_sites]
    return missing


def vacancy_separation(perfect, missing_sites):
    if len(missing_sites) != 2:
        return float('nan')
    return mic_distance(
        perfect,
        perfect.positions[missing_sites[0]],
        perfect.positions[missing_sites[1]],
    )


def normalized(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    if n == 0:
        raise ValueError("Cannot normalize a zero vector.")
    return v / n
