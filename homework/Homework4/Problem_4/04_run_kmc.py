#!/usr/bin/env python3
"""
Question 4D: On-lattice KMC simulation of a single vacancy in FCC Al.

The migration barrier is read directly from the student's NEB result:
    results/neb_results.json

STUDENT TASK IN THIS FILE
-------------------------
Choose FIVE temperatures between 550 K and 800 K by editing TEMPERATURES_K.

The KMC implementation itself is supplied. It models one vacancy performing
nearest-neighbor jumps on an FCC lattice. For pure Al all 12 nearest-neighbor
vacancy jumps are equivalent and use the same NEB migration barrier.

The script records the unwrapped vacancy position and physical KMC time for
multiple independent replicas. Students will calculate MSD and diffusivity
from these trajectories in 05_analyze_kmc_template.py.
"""

from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
KMC_DIR = ROOT / "kmc"
KMC_DIR.mkdir(exist_ok=True)

KB_EV_PER_K = 8.617333262145e-5
LATTICE_CONSTANT_A = 4.05

# Experimental monovacancy attempt frequency reported by Khellaf et al.
# The barrier is NOT taken from experiment: it comes from the student's NEB.
ATTEMPT_FREQUENCY_S = 3.6e13

N_REPLICAS = 128
N_EVENTS = 50000
SAMPLE_EVERY = 250
BASE_SEED = 20261108


# ============================================================================
# STUDENT TODO — CHOOSE FIVE TEMPERATURES
# ============================================================================
# Choose five different temperatures between 550 K and 800 K.
# Example syntax only:
#     TEMPERATURES_K = [550, 600, 650, 700, 750]
#
# Your values do not need to match the example.
# ============================================================================
TEMPERATURES_K = [550, 600, 650, 700, 750]


if len(TEMPERATURES_K) != 5:
    raise ValueError("Choose exactly five KMC temperatures in TEMPERATURES_K.")
if len(set(TEMPERATURES_K)) != 5:
    raise ValueError("The five KMC temperatures must be different.")
if min(TEMPERATURES_K) < 550 or max(TEMPERATURES_K) > 800:
    raise ValueError("Choose all five temperatures between 550 K and 800 K.")

neb_file = RESULTS_DIR / "neb_results.json"
if not neb_file.exists():
    raise FileNotFoundError(
        "results/neb_results.json was not found. Complete the NEB analysis first."
    )

with open(neb_file) as f:
    neb_results = json.load(f)

EA_NEB_EV = float(neb_results["migration_barrier_eV"])
if EA_NEB_EV <= 0:
    raise ValueError("The NEB migration barrier must be positive.")

# Twelve FCC nearest-neighbor vacancy jumps.
# Each vector has length a/sqrt(2).
base_vectors = np.array([
    [0, +1, +1], [0, +1, -1], [0, -1, +1], [0, -1, -1],
    [+1, 0, +1], [+1, 0, -1], [-1, 0, +1], [-1, 0, -1],
    [+1, +1, 0], [+1, -1, 0], [-1, +1, 0], [-1, -1, 0],
], dtype=float)

jump_vectors_m = base_vectors * (LATTICE_CONSTANT_A * 1.0e-10 / 2.0)
N_EVENTS_AVAILABLE = len(jump_vectors_m)
jump_distance_m = float(np.linalg.norm(jump_vectors_m[0]))

sample_event_indices = np.arange(0, N_EVENTS + 1, SAMPLE_EVERY, dtype=int)
if sample_event_indices[-1] != N_EVENTS:
    sample_event_indices = np.append(sample_event_indices, N_EVENTS)


def run_one_replica(temperature_K, rng):
    """Run one exact residence-time KMC vacancy random walk."""
    per_jump_rate = ATTEMPT_FREQUENCY_S * np.exp(
        -EA_NEB_EV / (KB_EV_PER_K * temperature_K)
    )
    total_rate = N_EVENTS_AVAILABLE * per_jump_rate

    # With 12 symmetry-equivalent events, event selection is uniform.
    event_choices = rng.integers(0, N_EVENTS_AVAILABLE, size=N_EVENTS)

    # Residence times are exponentially distributed with mean 1/R_total.
    waiting_times_s = rng.exponential(scale=1.0 / total_rate, size=N_EVENTS)

    increments_m = jump_vectors_m[event_choices]
    full_positions_m = np.vstack([
        np.zeros((1, 3)),
        np.cumsum(increments_m, axis=0),
    ])
    full_times_s = np.concatenate([
        np.array([0.0]),
        np.cumsum(waiting_times_s),
    ])

    return (
        full_times_s[sample_event_indices],
        full_positions_m[sample_event_indices],
        per_jump_rate,
        total_rate,
    )


print("KMC vacancy diffusion")
print("=====================")
print(f"NEB migration barrier: {EA_NEB_EV:.6f} eV")
print(f"Attempt frequency:     {ATTEMPT_FREQUENCY_S:.3e} s^-1")
print(f"FCC jump distance:     {jump_distance_m:.4e} m")
print(f"Replicas per T:        {N_REPLICAS}")
print(f"Events per replica:    {N_EVENTS}")

for temperature_K in sorted(TEMPERATURES_K):
    rng = np.random.default_rng(BASE_SEED + int(round(temperature_K * 10)))

    times = []
    positions = []
    per_jump_rate = None
    total_rate = None

    for _ in range(N_REPLICAS):
        t, r, per_jump_rate, total_rate = run_one_replica(temperature_K, rng)
        times.append(t)
        positions.append(r)

    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)

    outfile = KMC_DIR / f"kmc_T{int(round(temperature_K)):04d}K.npz"
    np.savez_compressed(
        outfile,
        times_s=times,
        positions_m=positions,
        temperature_K=float(temperature_K),
        migration_barrier_eV=EA_NEB_EV,
        attempt_frequency_s=ATTEMPT_FREQUENCY_S,
        lattice_constant_A=LATTICE_CONSTANT_A,
        jump_distance_m=jump_distance_m,
        per_jump_rate_s=float(per_jump_rate),
        total_rate_s=float(total_rate),
        n_replicas=N_REPLICAS,
        n_events=N_EVENTS,
        sample_every=SAMPLE_EVERY,
    )

    print(
        f"T = {temperature_K:7.2f} K | "
        f"per-jump rate = {per_jump_rate:.3e} s^-1 | "
        f"saved {outfile.name}"
    )

print("\nKMC trajectories complete.")
print("Next: complete 05_analyze_kmc_template.py")
