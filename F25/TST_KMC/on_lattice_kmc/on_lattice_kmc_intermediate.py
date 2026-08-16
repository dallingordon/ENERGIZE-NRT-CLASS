import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
L = 100             # lattice size
N_steps = 5000      # total KMC steps
kT = 0.025         # thermal energy (eV) - increase temperature
steps_per_frame = 50
n_frames = N_steps // steps_per_frame

# Initialize lattice with 1% vacancies randomly
lattice = np.ones((L, L), dtype=int)
initial_vac_frac = 0.01
num_initial_vac = int(L*L*initial_vac_frac)
vac_indices = np.random.choice(L*L, size=num_initial_vac, replace=False)
for index in vac_indices:
    x, y = divmod(index, L)
    lattice[x, y] = 0

# Random vacancy formation energies (0.025 to 0.1 eV)
vac_formation_energies = np.random.uniform(0.025, 0.1, size=(L, L))

vacancy_counts = []
lattice_snapshots = []

def get_vacancy_energy(x, y):
    """Return the vacancy formation energy for site (x, y)."""
    return vac_formation_energies[x, y]


def attempt_vacancy_change(x, y):
    """
    Attempt to flip occupancy of voxel (x, y):
    - if occupied (atom=1), attempt vacancy creation
    - if vacant (vacancy=0), attempt filling vacancy with atom
    Acceptance follows Metropolis criterion.

    Returns:
        True if move accepted (lattice flipped), False if rejected.
    """

    # 1. Get current occupancy state at site (x,y): 1 = atom, 0 = vacancy
    # TO-DO
    # 2. Retrieve local vacancy formation energy at site (energy cost/gain for creating/removing vacancy)
    # TO-DO
    # 3. Compute energy change (ΔE) associated with flipping this site:
    #    If currently atom (1), trying to form vacancy costs E_form (positive ΔE)
    #    If currently vacancy (0), trying to fill vacancy gains energy (-E_form, negative ΔE)
    if current_state == 1:
        # Attempt vacancy creation -> energy increases by E_form
    # TO-DO
    else:
        # Attempt filling vacancy -> energy decreases by E_form
    # TO-DO
    # 4. Determine if proposed move is accepted using Metropolis criterion:
    #    - If ΔE ≤ 0: accept move unconditionally (energy lowered or unchanged)
    #    - Otherwise accept with probability exp(-ΔE / kT)
    if delta_E <= 0:
    # TO-DO
    else:
    # TO-DO
    # 5. If accepted, flip occupancy state:
    #    Change atom to vacancy or vacancy to atom by toggling 1 <-> 0
    if accept:
    # TO-DO
    # 6. Return whether the move was accepted for tracking or debugging
    return accept


def perform_kmc_steps(n):
    """
    Perform `n` KMC on-lattice moves:
    - For each move, randomly select a site
    - Attempt vacancy flip move (create or fill)
    - Updates lattice if accepted
    This simulates evolution of vacancy concentrations over time.
    """

    for _ in range(n):
        # 1. Choose random voxel coordinates uniformly over lattice
# TO-DO
        # 2. Attempt vacancy flip move at that site
# TO-DO
        # (Optional: here you could tally acceptance stats or update time)

# ==== Perform full KMC simulation first and save snapshots ====

print("Running full KMC simulation...")
for frame in range(n_frames):
    perform_kmc_steps(steps_per_frame)
    vac_count = np.sum(lattice == 0)
    vacancy_counts.append(vac_count)
    # Make a copy of lattice for animation snapshot
    lattice_snapshots.append(lattice.copy())

print("KMC simulation done. Total frames:", len(lattice_snapshots))

# ==== Animate saved snapshots ====

fig, ax = plt.subplots(figsize=(6,6))
cmap = plt.cm.get_cmap('Greys_r', 2)  # Vacancy=white, atom=black
im = ax.imshow(lattice_snapshots[0], cmap=cmap, vmin=0, vmax=1)
ax.axis('off')

def update(frame):
    im.set_data(lattice_snapshots[frame])
    ax.set_title(f'KMC Step {(frame+1)*steps_per_frame}\nVacancies: {vacancy_counts[frame]}')
    return [im]

anim = animation.FuncAnimation(fig, update, frames=n_frames, blit=True, interval=50)

#anim.save('vacancy_kmc_separate.mp4', writer='ffmpeg', fps=20)
#plt.close()

# ==== Plot vacancy counts ====

plt.figure(figsize=(8,5))
steps_array = np.arange(len(vacancy_counts)) * steps_per_frame
plt.plot(steps_array, vacancy_counts, marker='o')
plt.xlabel('KMC Steps')
plt.ylabel('Number of Vacancies')
plt.title('Vacancy concentration vs. KMC Step')
plt.grid(True)
plt.tight_layout()
plt.savefig('vacancy_count_separate.png')
plt.show()