import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ---------------- Morse potential ----------------
def morse_potential(r, D, a, r_e):
    return D * (1 - np.exp(-a * (r - r_e)))**2 - D

def morse_force(r, D, a, r_e):
    exp_term = np.exp(-a * (r - r_e))
    return -2 * a * D * (1 - exp_term) * exp_term

# ---------------- Periodic boundary ----------------
def minimum_image(rij, box_size):
    return rij - box_size * np.round(rij / box_size)

# ---------------- Initialize dimers ----------------
def initialize_dimers(num_dimers, box_size, r0, temperature=0.01, seed=42):
    rng = np.random.default_rng(seed)
    positions = []
    velocities = []

    for d in range(num_dimers):
        center = rng.uniform(0.2*box_size, 0.8*box_size, size=2)
        theta = rng.uniform(0, 2*np.pi)
        bond_vec = np.array([np.cos(theta), np.sin(theta)]) * (r0/2)
        pos1 = (center - bond_vec) % box_size
        pos2 = (center + bond_vec) % box_size
        positions.append(pos1)
        positions.append(pos2)

        v1 = rng.normal(0, np.sqrt(temperature), size=2)
        v2 = rng.normal(0, np.sqrt(temperature), size=2)
        velocities.append(v1)
        velocities.append(v2)

    positions = np.array(positions)
    velocities = np.array(velocities)
    velocities -= np.mean(velocities, axis=0)  # remove net drift
    return positions, velocities

# ---------------- Forces ----------------
def compute_forces(positions, box_size, D, a, r_e):
    N = len(positions)
    forces = np.zeros_like(positions)
    potential_energy = 0.0
    for i in range(N):
        for j in range(i+1,N):
            rij = minimum_image(positions[i]-positions[j], box_size)
            r = np.linalg.norm(rij)
            if r < box_size/2:
                f = morse_force(r, D, a, r_e)
                fij = f * rij / r
                forces[i] += fij
                forces[j] -= fij
                potential_energy += morse_potential(r, D, a, r_e)
    return forces, potential_energy

# ---------------- Velocity-Verlet ----------------
def velocity_verlet(positions, velocities, box_size, D, a, r_e, dt, steps):
    N = len(positions)
    traj = np.zeros((steps, N, 2))
    energies = np.zeros((steps, 3))

    # Create your own velocity-verlet algorithm
    '''
    
    
    TO-DO
    
    
    '''

    return traj, energies

# ---------------- Simulation parameters ----------------
params = {"D": 1.0, "a": 1.0, "r_e": 1.0} #add your optimized potential parameters here
num_dimers = 4
N = num_dimers*2
box_size = 20.0
dt = 0.01
steps = 20000

# Initialize positions and velocities
positions, velocities = initialize_dimers(num_dimers, box_size, r0=params["r_e"])
traj, energies = velocity_verlet(positions, velocities, box_size,
                                 params["D"], params["a"], params["r_e"],
                                 dt, steps)

# ---------------- Energy plot ----------------
plt.figure(figsize=(6,4))
plt.plot(energies[:,0], label="Kinetic")
plt.plot(energies[:,1], label="Potential")
plt.plot(energies[:,2], label="Total", lw=2)
plt.title("Energy vs Time")
plt.xlabel("Step")
plt.ylabel("Energy (eV)")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------- Animation ----------------
r_break = 4.0  # bond break distance
r_form = 2.0   # bond formation distance

# Initialize bond list with starting dimers
bond_list = [(2*d, 2*d+1) for d in range(num_dimers)]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")

scat = ax.scatter(traj[0,:,0], traj[0,:,1], s=60, c='dodgerblue')
bond_lines = []
bond_texts = []

def update(frame):
    global bond_list
    scat.set_offsets(traj[frame])

    # Clear old visuals
    for txt in bond_texts:
        txt.remove()
    bond_texts.clear()
    for line in bond_lines:
        line.remove()
    bond_lines.clear()

    # Compute pairwise distances
    N_atoms = len(traj[frame])
    distances = np.zeros((N_atoms,N_atoms))
    for i in range(N_atoms):
        for j in range(i+1,N_atoms):
            rij = minimum_image(traj[frame,i] - traj[frame,j], box_size)
            distances[i,j] = distances[j,i] = np.linalg.norm(rij)

    # Step 1: Remove bonds longer than r_break
    bond_list = [ (i,j) for (i,j) in bond_list if distances[i,j] <= r_break ]

    # Step 2: Track bonded atoms
    bonded_atoms = set([atom for bond in bond_list for atom in bond])

    # Step 3: Form new bonds if possible
    potential_pairs = []
    for i in range(N_atoms):
        if i in bonded_atoms: continue
        for j in range(i+1, N_atoms):
            if j in bonded_atoms: continue
            if distances[i,j] <= r_form:
                potential_pairs.append((distances[i,j], i, j))
    potential_pairs.sort()  # closest first
    for d, i, j in potential_pairs:
        if i not in bonded_atoms and j not in bonded_atoms:
            bond_list.append((i,j))
            bonded_atoms.update([i,j])

    # Step 4: Draw bonds
    for (i,j) in bond_list:
        ri = traj[frame,i]
        rj = traj[frame,j]
        rij = minimum_image(ri - rj, box_size)
        r = np.linalg.norm(rij)
        cross = np.abs(rij) > box_size/2
        color = 'b' if np.any(cross) else 'k'

        start = ri
        end = (ri - rij) % box_size
        line, = ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2,alpha=0.25)
        bond_lines.append(line)

        mid = (ri - 0.5*rij) % box_size
        txt = ax.text(mid[0], mid[1], f"{r:.2f}", color='red', fontsize=8,
                      ha='center', va='center')
        bond_texts.append(txt)

    ax.set_title(f"Step {frame}")
    return [scat, *bond_lines, *bond_texts]

anim = FuncAnimation(fig, update, frames=range(0, steps, 10),
                     interval=30, blit=True)
plt.show()

# To save:
# anim.save("al_dimers_dynamic_bonds.mp4", writer="ffmpeg", fps=30)
