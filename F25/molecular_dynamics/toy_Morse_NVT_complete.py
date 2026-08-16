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

# ---------------- Velocity-Verlet NVT (Nosé–Hoover corrected) ----------------
def velocity_verlet_nose_hoover_corrected(positions, velocities, box_size, D, a, r_e,
                                          dt, steps, T_target=0.1, Q=10.0):
    """
    Simple symmetric Nosé–Hoover integrator that advances:
     - particle positions & velocities
     - thermostat variables xi and eta

    Units: reduced (m=1, kB=1).
    """
    N = len(positions)
    dof = 2 * N
    traj = np.zeros((steps, N, 2))
    energies = np.zeros((steps, 5))  # KE, PE, KE+PE, thermostat_energy, E_ext (extended/Hamiltonian)
    temperatures = np.zeros(steps)

    # thermostat variables
    xi = 0.0    # friction / thermostat momentum-like
    eta = 0.0   # thermostat position-like

    # initial forces
    forces, pe = compute_forces(positions, box_size, D, a, r_e)

    for step in range(steps):
        if step % 100 == 0:
            print('Step ',step,' of ',steps)
        # store positions
        traj[step] = positions

        # kinetic energy (m=1)
        ke = 0.5 * np.sum(velocities**2)
        temperatures[step] = (2.0 * ke) / dof  # kB = 1

        # thermostat energy (full expression): 0.5 Q xi^2 + dof * T_target * eta
        e_therm = 0.5 * Q * xi**2 + dof * T_target * eta

        # extended (conserved) energy
        e_ext = ke + pe + e_therm

        energies[step] = [ke, pe, ke + pe, e_therm, e_ext]

        # --- Integrator steps (symmetric) ---
        # 1) half-step velocity scaling by xi
        #    v <- v * exp(-xi * dt/2)
        velocities *= np.exp(-0.5 * xi * dt)

        # 2) position update using current velocities and forces (standard VV)
        positions = positions + velocities * dt + 0.5 * forces * dt**2
        positions %= box_size

        # 3) compute new forces at updated positions
        new_forces, pe = compute_forces(positions, box_size, D, a, r_e)

        # 4) velocity update by half-step with forces
        velocities += 0.5 * (forces + new_forces) * dt

        # 5) another half-step velocity scaling by xi
        velocities *= np.exp(-0.5 * xi * dt)

        # 6) update thermostat variables
        #    xi_dot = (2K - dof * T_target) / Q
        ke_now = 0.5 * np.sum(velocities**2)
        xi_dot = (2.0 * ke_now - dof * T_target) / Q
        xi += xi_dot * dt

        #    eta_dot = xi
        eta += xi * dt

        # advance forces reference
        forces = new_forces

    return traj, energies, temperatures

# ---------------- Simulation parameters ----------------
params = {"D": 2.40, "a": 1.0, "r_e": 2.7}
num_dimers = 4
N = num_dimers*2
box_size = 20.0
dt = 0.01
steps = 10000 # adjustable (keeps runtime reasonable)

# Initialize positions and velocities
positions, velocities = initialize_dimers(num_dimers, box_size, r0=params["r_e"])

# Run NVT with corrected Nose-Hoover
T_target = 0.05
Q = 50.0
traj, energies, temperatures = velocity_verlet_nose_hoover_corrected(
    positions.copy(), velocities.copy(), box_size,
    params["D"], params["a"], params["r_e"],
    dt, steps, T_target=T_target, Q=Q
)

# ---------------- Energy + Temperature plots ----------------
fig, axes = plt.subplots(2, 1, figsize=(7,8), sharex=True)
ax1, ax2 = axes

ax1.plot(energies[:,0], label="Kinetic")
ax1.plot(energies[:,1], label="Potential")
ax1.plot(energies[:,2], label="Kinetic + Potential", lw=1.5)
ax1.plot(energies[:,3], label="Thermostat Energy (0.5 Q xi^2 + g T eta)", ls="--")
ax1.plot(energies[:,4], label="Extended Energy (conserved)", lw=2, color="k")
ax1.set_ylabel("Energy (reduced units)")
ax1.set_title("Energy vs Time (NVT Nose–Hoover, corrected)")
ax1.legend(loc="upper right")

ax2.plot(temperatures, label="Instantaneous T (reduced)")
ax2.axhline(T_target, color="red", ls="--", label=f"Target T = {T_target}")
ax2.set_xlabel("Step")
ax2.set_ylabel("Temperature (reduced units)")
ax2.set_title("Temperature vs Time")
ax2.legend()

plt.tight_layout()
plt.show()

# ---------------- Animation with dynamic bonds (same as before) ----------------
r_break = 4.0
r_form = 2.0
bond_list = [(2*d, 2*d+1) for d in range(num_dimers)]

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, box_size)
ax.set_ylim(0, box_size)
ax.set_aspect("equal")

scat = ax.scatter(traj[0,:,0], traj[0,:,1], s=60, c='dodgerblue')
bond_lines, bond_texts = [], []

def update(frame):
    global bond_list
    scat.set_offsets(traj[frame])
    for txt in bond_texts: txt.remove()
    for line in bond_lines: line.remove()
    bond_texts.clear(); bond_lines.clear()

    N_atoms = len(traj[frame])
    distances = np.zeros((N_atoms,N_atoms))
    for i in range(N_atoms):
        for j in range(i+1,N_atoms):
            rij = minimum_image(traj[frame,i]-traj[frame,j], box_size)
            distances[i,j] = distances[j,i] = np.linalg.norm(rij)

    # Remove long bonds
    bond_list = [(i,j) for (i,j) in bond_list if distances[i,j] <= r_break]
    bonded_atoms = set([atom for bond in bond_list for atom in bond])

    # Form new bonds
    potential_pairs = []
    for i in range(N_atoms):
        if i in bonded_atoms: continue
        for j in range(i+1,N_atoms):
            if j in bonded_atoms: continue
            if distances[i,j] <= r_form:
                potential_pairs.append((distances[i,j], i, j))
    potential_pairs.sort()
    for d, i, j in potential_pairs:
        if i not in bonded_atoms and j not in bonded_atoms:
            bond_list.append((i,j))
            bonded_atoms.update([i,j])

    # Draw bonds
    for (i,j) in bond_list:
        ri, rj = traj[frame,i], traj[frame,j]
        rij = minimum_image(ri-rj, box_size)
        r = np.linalg.norm(rij)
        cross = np.abs(rij) > box_size/2
        color = 'b' if np.any(cross) else 'k'
        start, end = ri, (ri - rij) % box_size
        line, = ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=2, alpha=0.25)
        bond_lines.append(line)
        mid = (ri - 0.5*rij) % box_size
        txt = ax.text(mid[0], mid[1], f"{r:.2f}", color='red', fontsize=8,
                      ha='center', va='center')
        bond_texts.append(txt)

    ax.set_title(f"Step {frame}")
    return [scat, *bond_lines, *bond_texts]

anim = FuncAnimation(fig, update, frames=range(0, steps, 10), interval=30, blit=True)
plt.show()
