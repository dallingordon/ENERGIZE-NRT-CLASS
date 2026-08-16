import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

def make_fcc_lattice(a, nx, ny, nz):
    basis = np.array([[0, 0, 0],
                      [0.5, 0.5, 0],
                      [0.5, 0, 0.5],
                      [0, 0.5, 0.5]])
    coords = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                cell_origin = np.array([i, j, k])
                for b in basis:
                    coords.append(a * (cell_origin + b))
    return np.array(coords)

def minimum_image_distances(positions, box_length):
    delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    delta -= box_length * np.round(delta / box_length)
    dist_matrix = np.linalg.norm(delta, axis=-1)
    return dist_matrix

def gaussian_kernel(r, centers, sigma):
    diff = r[:, np.newaxis] - centers[np.newaxis, :]
    return np.sum(np.exp(-0.5 * (diff / sigma)**2), axis=1) / (centers.size * sigma * np.sqrt(2 * np.pi))

def identify_shells(distances, tol=0.05):
    shells = []
    current_shell = [distances[0]]
    shell_indices = np.zeros_like(distances, dtype=int)
    shell_idx = 1
    shell_indices[0] = shell_idx
    for i in range(1, len(distances)):
        if abs(distances[i] - np.mean(current_shell)) < tol:
            current_shell.append(distances[i])
            shell_indices[i] = shell_idx
        else:
            shell_idx += 1
            current_shell = [distances[i]]
            shell_indices[i] = shell_idx
    return shell_indices

def perturb_positions(positions, max_displacement, box_length):
    """
    Apply random displacement in [-max_displacement, max_displacement] (Å) to each atom's position,
    then wrap positions back into the periodic box [0, box_length).
    """
    displacement = np.random.uniform(-max_displacement, max_displacement, positions.shape)
    new_positions = positions + displacement
    # Wrap inside box (periodic boundary)
    new_positions = np.mod(new_positions, box_length)
    return new_positions

# Parameters
a_Al = 4.05
nx = ny = nz = 4
num_shells = 8
sigma = 0.02
max_perturbation = 0.75  # max displacement in Angstroms (set as you like)

# Generate lattice & box size
coords = make_fcc_lattice(a_Al, nx, ny, nz)
box_length = a_Al * nx

# Perturb positions randomly
coords_perturbed = perturb_positions(coords, max_perturbation, box_length)

# Compute distances with PBC
dist_matrix = minimum_image_distances(coords_perturbed, box_length)
np.fill_diagonal(dist_matrix, np.inf)

# set nearest neighbor shells anf fill them
'''


TO-DO


'''




# Prepare r grid for KDE
all_dists_flat = np.hstack(list(shell_distances.values()))
r_min = max(0.0, all_dists_flat.min() - 0.1)
r_max = all_dists_flat.max() + 0.1
r_grid = np.linspace(r_min, r_max, 1000)

cmap = get_cmap('coolwarm')
colors = [cmap(i / (num_shells - 1)) for i in range(num_shells)]

plt.figure(figsize=(8,6))
for shell in range(1, num_shells + 1):
    if len(shell_distances[shell]) > 0:
        g_n = gaussian_kernel(r_grid, shell_distances[shell], sigma)
        plt.plot(r_grid, g_n, color=colors[shell-1], linewidth=3, label=f'Shell {shell}')

plt.xlabel(r'Distance $r$ (\AA)')
plt.ylabel(r'$g_n(r)$ - local RDF per shell')
plt.title(f'Exclusive neighbor shell RDFs (Al FCC, 256 atoms) with max perturbation {max_perturbation:.3f} \AA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()