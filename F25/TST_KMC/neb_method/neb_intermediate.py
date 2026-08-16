import numpy as np
import matplotlib.pyplot as plt
from ase.build import fcc111
from ase import Atom
from ase.calculators.eam import EAM
from ase.constraints import FixAtoms
from ase.optimize import BFGS

# --- Setup slab and EMT potential (replace with DFT calculator if available) ---
slab = fcc111('Al', size=(3, 3, 4), vacuum=10.0)
calc = EAM(potential=r'C:\Users\jc112358\venv\Lib\site-packages\ENERGIZE-NRT-CLASS\TST_KMC\dimer_method\Al_zhou.eam.alloy')
slab.set_calculator(calc)

# --- Identify surface atoms to define fcc hollow, adjacent hollow, bridge sites ---
top_layer_z = slab.positions[:, 2].max()
top_layer_atoms = [atom for atom in slab if abs(atom.position[2] - top_layer_z) < 0.1]
top_positions = np.array([atom.position for atom in top_layer_atoms])

# Pick three atoms forming an fcc hollow site triangle (example indices)
triangle_atoms = top_positions[[0, 1, 3]]

# Calculate fcc hollow site as centroid of triangle
fcc_hollow = np.mean(triangle_atoms, axis=0)

# Set adatom height above surface layer (adjust if needed)
adatom_height = 1.2
fcc_hollow[2] += adatom_height

# Adjacent hollow site: translate along a1/3
a1 = slab.get_cell()[0]
adjacent_hollow = fcc_hollow + a1 / 3
adjacent_hollow[2] = fcc_hollow[2]

# Bridge site: midpoint between two adjacent surface atoms (atoms 0 and 1)
bridge_site = (top_positions[0] + top_positions[1]) / 2
bridge_site[2] = fcc_hollow[2]

# --- Helper function to create Images with adatom ---
def create_image(base_slab, adatom_pos):
    img = base_slab.copy()
    img += Atom('Al', position=adatom_pos)
    img.set_calculator(calc)
    return img

# --- Create and relax initial image ---
initial_image = create_image(slab, fcc_hollow)
mask_init = [atom.index != len(initial_image) - 1 for atom in initial_image]
initial_image.set_constraint(FixAtoms(mask=mask_init))
print("Relaxing initial state...")
dyn_init = BFGS(initial_image, logfile=None)
dyn_init.run(fmax=0.05)

# --- Create and relax final image ---
final_image = create_image(slab, adjacent_hollow)
mask_final = [atom.index != len(final_image) - 1 for atom in final_image]
final_image.set_constraint(FixAtoms(mask=mask_final))
print("Relaxing final state...")
dyn_final = BFGS(final_image, logfile=None)
dyn_final.run(fmax=0.05)

# --- Generate NEB images by interpolation between relaxed fcc sites ---
n_images = 11
images = []

for i in range(n_images):
    interp_pos = initial_image[-1].position + (final_image[-1].position - initial_image[-1].position) * i / (n_images - 1)
    img = create_image(slab, interp_pos)
    mask = [atom.index != len(img) - 1 for atom in img]
    img.set_constraint(FixAtoms(mask=mask))
    images.append(img)

# --- NEB optimization parameters ---
max_steps = 200
force_tol = 0.005  # eV/Å
k_spring = 1.0

# --- Estimate tangent vector at each intermediate image ---
def get_tangent(prev, curr, next_):
    # Extract positions of the adatom (or relevant atom) in the previous image
    # TO-DO
    # Extract position of the adatom in the current image
    # TO-DO
    # Extract position of the adatom in the next image
    # TO-DO

    # Get potential energy of the previous image
    # TO-DO
    # Get potential energy of the current image
    # TO-DO
    # Get potential energy of the next image
    # TO-DO

    # Compute vector difference from current to next image (forward segment)
    # TO-DO
    # Compute vector difference from previous to current image (backward segment)
    # TO-DO

    # Determine tangent direction - key step in NEB to avoid kinks and correctly define path direction:
    # If energy increases going forward (next) and backward (previous) is lower,
    # Use forward difference as tangent (moving uphill in energy)
    if E_next > E_curr > E_prev:
        # TO-DO
    # If energy decreases going forward and backward is higher,
    # Use backward difference as tangent (moving downhill in energy)
    elif E_next < E_curr < E_prev:
        # TO-DO
    else:
        # Otherwise (i.e., current is a local minimum or maximum),
        # Use the average of forward and backward differences
        # TO-DO
    # Normalize the tangent vector to unit length: tangent must be a unit vector along the band
    # TO-DO
    if norm == 0:
        # If tangent has zero length (degenerate case), return zero vector to avoid NaNs
        return np.zeros_like(tangent)
    return tangent / norm

# --- Calculate NEB forces ---
def calc_neb_forces(images):
    # Number of images in the NEB chain (including initial and final states)
    n = len(images)

    forces = []
    energies = []
    # Loop over all images to collect adatom forces and energies
    for image in images:
        # Extract the force on the adatom in this image
        # TO-DO
        # Extract the total potential energy of this image
        # TO-DO

    # Initialize NEB force list for all images as zero vectors (array of 3D vectors)
    # TO-DO

    # Loop over intermediate images only (exclude fixed endpoints)
    for i in range(1, n - 1):
        # True force on the adatom from the potential energy surface at image i
        # TO-DO

        # Positions of the adatom for current, previous, and next images
        # TO-DO
        # TO-DO  # R_{i-1}
        # TO-DO  # R_{i+1}

        # Get the tangent to the path at image i, needed to separate force components
        # TO-DO

        # Calculate spring force along the band (restoring force between adjacent images)
        # k_spring is the spring constant (assumed defined globally or provided elsewhere)
        # The spring force is proportional to the difference in distances between image i and its neighbors,
        # projected along the tangent direction tau (along the path)
        # TO-DO

        # Project true force perpendicular to the tangent:
        # NEB only lets potential forces relax perpendicular to the path,
        # removing any component along the path to prevent sliding down the path
        # TO-DO

        # Total NEB force is sum of perpendicular true force and spring force along tangent
        # TO-DO

    # Boundary images (initial and final) are fixed, so their NEB forces are zero
    # TO-DO
    # TO-DO

    # Return the forces to move images according to NEB and also the energies for convergence checking
    return neb_forces, energies

# --- Update images with NEB forces ---
def update_images(images, neb_forces, step_size=0.1):
    for i in range(1, len(images) - 1):
        pos = images[i][-1].position
        pos += step_size * neb_forces[i]
        images[i][-1].position = pos

# --- Run NEB optimization ---
print("Starting NEB optimization...")
for step in range(max_steps):
    neb_forces, energies = calc_neb_forces(images)
    max_force = max(np.linalg.norm(f) for f in neb_forces)
    if step % 10 == 0 or max_force < force_tol:
        print(f"Step {step}: max NEB force = {max_force:.3f} eV/Å")
    if max_force < force_tol:
        print("Converged!")
        break
    update_images(images, neb_forces, step_size=0.1)

# Find transition state image
energies = np.array([img.get_potential_energy() for img in images])
ts_index = np.argmax(energies)
print(f"Transition state is image {ts_index} with energy {energies[ts_index]:.3f} eV")

# --- Plotting ---

initial_xy = images[0][-1].position[:2]
path_coords = np.array([img[-1].position[:2] for img in images])
path_rel = path_coords - initial_xy

dist_stepwise = np.linalg.norm(np.diff(path_coords, axis=0), axis=1)
reaction_coord = np.insert(np.cumsum(dist_stepwise), 0, 0)

grid_points = 50
x_range = np.linspace(-5, 5, grid_points)
y_range = np.linspace(-5, 5, grid_points)
Z = np.zeros((grid_points, grid_points))

slab_atoms = slab.copy()
z_adatom = images[0][-1].position[2]

for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        test_slab = slab_atoms.copy()
        test_slab += Atom('Al', position=[initial_xy[0] + x, initial_xy[1] + y, z_adatom])
        test_slab.set_calculator(calc)
        Z[j, i] = test_slab.get_potential_energy()

plt.figure(figsize=(12, 5))

# Energy surface contour and path
plt.subplot(1, 2, 1)
contours = plt.contourf(x_range, y_range, Z, levels=30, cmap='viridis')
plt.colorbar(contours, label='Energy (eV)')
plt.plot(path_rel[:, 0], path_rel[:, 1], 'o-', color='red', label='NEB Path')
plt.scatter(path_rel[0, 0], path_rel[0, 1], marker='*', s=200, color='green', label='Start (fcc hollow)')
plt.scatter(path_rel[-1, 0], path_rel[-1, 1], marker='s', s=200, color='blue', label='End (adjacent fcc hollow)')
plt.scatter(path_rel[ts_index, 0], path_rel[ts_index, 1], marker='X', s=200, color='black', label='TS (bridge approx)')
plt.xlabel('X (Å) relative to initial adatom')
plt.ylabel('Y (Å) relative to initial adatom')
plt.legend()
plt.title('Energy Surface with NEB Path')

# Relative energy vs reaction coordinate
relative_energies = energies - energies[0]
plt.subplot(1, 2, 2)
plt.plot(reaction_coord, relative_energies, 'o-', color='navy', label='Relative Energy')
plt.plot(reaction_coord[ts_index], relative_energies[ts_index], 'X', color='black', markersize=12, label='TS')
plt.xlabel('Reaction coordinate (Å)')
plt.ylabel('Energy (eV)')
plt.title('Relative Energy Profile Along Path')
plt.legend()

plt.tight_layout()
plt.show()