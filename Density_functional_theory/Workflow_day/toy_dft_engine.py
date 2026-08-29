"""Shared toy electronic-structure backend for the ME500 DFT workflow demos.

This is NOT a DFT code and does NOT call VASP.  It implements the minimum
mathematical machinery needed to make SCF, force/geometry, and DOS workflows
visible in a classroom setting.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math
import numpy as np

# -----------------------------------------------------------------------------
# TOY ELECTRONIC MODEL PARAMETERS
# -----------------------------------------------------------------------------
# These are deliberately chosen for pedagogical behavior, not physical accuracy.
EPS_A0 = -1.0          # bare onsite level, toy eV
EPS_B0 = +1.0
HUBBARD_A = 2.0        # local charge-feedback strength
HUBBARD_B = 2.0
REFERENCE_CHARGE_A = 1.0
REFERENCE_CHARGE_B = 1.0
HOPPING_AT_RREF = 2.4  # magnitude of A-B hopping at R_REF
HOPPING_DECAY = 1.15
R_REF = 2.0
TOTAL_ELECTRONS = 2.0

# Short-range ionic/core repulsion used only for the geometry/PES demo.
REPULSION_A = 175.0
REPULSION_DECAY = 2.0


@dataclass
class Structure:
    comment: str
    scale: float
    lattice: np.ndarray
    symbols: List[str]
    counts: List[int]
    coord_mode: str
    frac_positions: np.ndarray

    @property
    def cart_positions(self) -> np.ndarray:
        return self.frac_positions @ self.lattice


def parse_incar(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("#", 1)[0].split("!", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, val = line.split("=", 1)
        data[key.strip().upper()] = val.strip()
    return data


def get_float(d: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(d.get(key.upper(), default))
    except Exception:
        return float(default)


def get_int(d: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(d.get(key.upper(), default)))
    except Exception:
        return int(default)


def parse_poscar(path: Path) -> Structure:
    lines = [x.rstrip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR appears incomplete: {path}")
    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)]) * scale
    symbols = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    idx = 7
    if lines[idx].lower().startswith("s"):
        idx += 1
    mode = lines[idx].strip()
    idx += 1
    n = sum(counts)
    coords = np.array([[float(x) for x in lines[idx + i].split()[:3]] for i in range(n)])
    if mode.lower().startswith("c") or mode.lower().startswith("k"):
        coords = coords @ np.linalg.inv(lattice)
        mode = "Direct"
    return Structure(comment, scale, lattice, symbols, counts, mode, coords)


def write_poscar(structure: Structure, path: Path, comment: str | None = None) -> None:
    with Path(path).open("w") as f:
        f.write((comment or structure.comment) + "\n")
        f.write("1.0\n")
        for row in structure.lattice:
            f.write("  {:16.10f} {:16.10f} {:16.10f}\n".format(*row))
        f.write("  " + "  ".join(structure.symbols) + "\n")
        f.write("  " + "  ".join(str(x) for x in structure.counts) + "\n")
        f.write("Direct\n")
        for p in structure.frac_positions:
            f.write("  {:16.10f} {:16.10f} {:16.10f}\n".format(*p))


def parse_kpoints_mesh(path: Path) -> Tuple[int, int, int]:
    lines = [x.strip() for x in Path(path).read_text().splitlines() if x.strip()]
    if len(lines) < 4:
        return (1, 1, 1)
    vals = lines[3].split()
    return tuple(max(1, int(float(v))) for v in vals[:3])


def pair_vector_and_distance(structure: Structure) -> Tuple[np.ndarray, float]:
    cart = structure.cart_positions
    if len(cart) < 2:
        raise ValueError("Toy model requires at least two atoms.")
    vec = cart[1] - cart[0]
    # Minimum-image wrap in fractional coordinates for a general parallelepiped.
    dfrac = structure.frac_positions[1] - structure.frac_positions[0]
    dfrac -= np.round(dfrac)
    vec = dfrac @ structure.lattice
    return vec, float(np.linalg.norm(vec))


def hopping(distance: float) -> float:
    return -HOPPING_AT_RREF * math.exp(-HOPPING_DECAY * (distance - R_REF))


def hamiltonian(distance: float, charges: np.ndarray) -> np.ndarray:
    eps_a = EPS_A0 + HUBBARD_A * (charges[0] - REFERENCE_CHARGE_A)
    eps_b = EPS_B0 + HUBBARD_B * (charges[1] - REFERENCE_CHARGE_B)
    t = hopping(distance)
    return np.array([[eps_a, t], [t, eps_b]], dtype=float)


def electronic_energy(distance: float, charges: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    H = hamiltonian(distance, charges)
    eigvals, eigvecs = np.linalg.eigh(H)
    occupied = eigvecs[:, 0]
    output_charges = TOTAL_ELECTRONS * np.abs(occupied) ** 2
    eband = TOTAL_ELECTRONS * eigvals[0]
    # Toy double-counting correction, analogous in spirit only.
    edc = -0.5 * HUBBARD_A * (charges[0] - REFERENCE_CHARGE_A) ** 2
    edc += -0.5 * HUBBARD_B * (charges[1] - REFERENCE_CHARGE_B) ** 2
    return float(eband + edc), eigvals, eigvecs, output_charges


def scf_solve(
    distance: float,
    amix: float = 0.30,
    ediff: float = 1e-6,
    nelm: int = 100,
    initial_charge_a: float = 1.80,
    initial_charge_b: float | None = None,
) -> Dict[str, object]:
    if initial_charge_b is None:
        initial_charge_b = TOTAL_ELECTRONS - initial_charge_a
    charges = np.array([initial_charge_a, initial_charge_b], dtype=float)
    history = []
    converged = False
    last_energy = None

    for it in range(1, nelm + 1):
        energy, eigvals, eigvecs, q_out = electronic_energy(distance, charges)
        q_resid = float(np.max(np.abs(q_out - charges)))
        dE = np.nan if last_energy is None else float(energy - last_energy)
        history.append({
            "iteration": it,
            "energy": energy,
            "dE": dE,
            "charge_a": float(charges[0]),
            "charge_b": float(charges[1]),
            "output_charge_a": float(q_out[0]),
            "output_charge_b": float(q_out[1]),
            "residual": q_resid,
            "eig_1": float(eigvals[0]),
            "eig_2": float(eigvals[1]),
        })

        mixed = (1.0 - amix) * charges + amix * q_out
        dq = float(np.max(np.abs(mixed - charges)))
        if dq < ediff:
            charges = mixed
            converged = True
            break
        charges = mixed
        last_energy = energy

    energy, eigvals, eigvecs, q_out = electronic_energy(distance, charges)
    return {
        "converged": converged,
        "iterations": len(history),
        "charges": charges,
        "energy": energy,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "history": history,
        "distance": distance,
    }


def ionic_repulsion(distance: float) -> float:
    return REPULSION_A * math.exp(-REPULSION_DECAY * distance)


def total_energy_for_distance(distance: float, scf_kwargs: Dict[str, float | int] | None = None) -> Tuple[float, Dict[str, object]]:
    kw = dict(scf_kwargs or {})
    res = scf_solve(distance, **kw)
    return float(res["energy"] + ionic_repulsion(distance)), res


def radial_force(distance: float, delta: float = 1e-3, scf_kwargs: Dict[str, float | int] | None = None) -> float:
    eplus, _ = total_energy_for_distance(distance + delta, scf_kwargs)
    eminus, _ = total_energy_for_distance(distance - delta, scf_kwargs)
    return float(-(eplus - eminus) / (2.0 * delta))


def structure_with_distance(structure: Structure, new_distance: float) -> Structure:
    # Keep pair center fixed and move both atoms symmetrically along their current bond axis.
    cart = structure.cart_positions.copy()
    vec, old_distance = pair_vector_and_distance(structure)
    if old_distance < 1e-12:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = vec / old_distance
    center = 0.5 * (cart[0] + cart[1])
    cart[0] = center - 0.5 * new_distance * direction
    cart[1] = center + 0.5 * new_distance * direction
    frac = cart @ np.linalg.inv(structure.lattice)
    frac %= 1.0
    return Structure(structure.comment, 1.0, structure.lattice.copy(), list(structure.symbols), list(structure.counts), "Direct", frac)


def kmesh_points(mesh: Tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = mesh
    axes = [np.linspace(-math.pi, math.pi, n, endpoint=False) for n in (nx, ny, nz)]
    pts = np.array([(x, y, z) for x in axes[0] for y in axes[1] for z in axes[2]], dtype=float)
    return pts


def bloch_bands(distance: float, charges: np.ndarray, kpts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eps_a = EPS_A0 + HUBBARD_A * (charges[0] - REFERENCE_CHARGE_A)
    eps_b = EPS_B0 + HUBBARD_B * (charges[1] - REFERENCE_CHARGE_B)
    t1 = hopping(distance)
    t2 = 0.18 * t1
    bands = []
    weights_a = []
    for kx, ky, kz in kpts:
        off = t1 + t2 * (math.cos(kx) + math.cos(ky) + math.cos(kz))
        H = np.array([[eps_a, off], [off, eps_b]], dtype=float)
        vals, vecs = np.linalg.eigh(H)
        bands.append(vals)
        weights_a.append(np.abs(vecs[0, :]) ** 2)
    return np.asarray(bands), np.asarray(weights_a)


def gaussian_dos(energies: np.ndarray, weights_a: np.ndarray, nedos: int, sigma: float) -> Dict[str, np.ndarray]:
    flat = energies.reshape(-1)
    pad = max(0.8, 6.0 * sigma)
    grid = np.linspace(flat.min() - pad, flat.max() + pad, nedos)
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    total = np.zeros_like(grid)
    pa = np.zeros_like(grid)
    pb = np.zeros_like(grid)
    nk = energies.shape[0]
    for ik in range(nk):
        for ib in range(energies.shape[1]):
            e = energies[ik, ib]
            g = norm * np.exp(-0.5 * ((grid - e) / sigma) ** 2) / nk
            total += g
            wa = weights_a[ik, ib]
            pa += wa * g
            pb += (1.0 - wa) * g
    integ = np.zeros_like(grid)
    if len(grid) > 1:
        de = grid[1] - grid[0]
        integ = np.cumsum(total) * de
    return {"energy": grid, "total": total, "pdos_a": pa, "pdos_b": pb, "integrated": integ}


def write_toy_outcar(path: Path, title: str, lines: List[str]) -> None:
    with Path(path).open("w") as f:
        f.write("TOY EDUCATIONAL OUTPUT -- NOT A VASP OUTCAR\n")
        f.write(title + "\n")
        f.write("=" * 78 + "\n")
        for line in lines:
            f.write(str(line) + "\n")
