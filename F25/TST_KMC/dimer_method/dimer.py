import numpy as np
import matplotlib.pyplot as plt
from ase.build import fcc111
from ase import Atom
from ase.calculators.eam import EAM
from ase.io import write
from ase.visualize import view
from ase.neighborlist import NeighborList

def normalize(v):
    return v / np.linalg.norm(v)


def energy_and_forces(atoms):
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    return energy, forces


def finite_difference_torque(atoms, R, N, dimer_sep):
    """Calculate torque on dimer direction via finite difference of forces."""
    R_plus = R + (dimer_sep / 2) * N
    R_minus = R - (dimer_sep / 2) * N

    atoms.set_positions(R_plus.reshape((-1, 3)))
    f_plus = atoms.get_forces().flatten()

    atoms.set_positions(R_minus.reshape((-1, 3)))
    f_minus = atoms.get_forces().flatten()

    delta_f = f_plus - f_minus
    tau = delta_f - np.dot(delta_f, N) * N
    return tau


def rotate_dimer(N, tau, angle_step=0.1):
    """Rotate dimer direction by small angle step along torque vector."""


    '''
    
    TO-DO
    
    '''


    return normalize(N_new)


def dimer_translation_step(R, F, N, step_size=0.01):
    """Translate dimer midpoint along effective force climbing up along N."""

    '''

    TO-DO

    '''


    return R_new


def dimer_curvature(atoms, R, N, dimer_sep):
    '''

    TO-DO

    '''

    return curvature

def create_al_slab_and_adatom_bridge():
    slab = fcc111('Al', size=(4,4,4), vacuum=10.0, orthogonal=True)
    positions = slab.get_positions()
    zmax = positions[:,2].max()
    tol = 0.1
    top_layer_indices = [i for i, pos in enumerate(positions) if pos[2] > zmax - tol]

    # Build neighborlist for top layer atoms only
    # Use cutoff ~3.0 Angstrom suitable for Al nearest neighbors
    cutoff = 3.0
    cutoffs = np.full(len(slab), cutoff)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(slab)

    # Find first neighbor pair within top layer
    top_layer_set = set(top_layer_indices)
    first_bridge = None
    for i in top_layer_indices:
        indices, offsets = nl.get_neighbors(i)
        for j in indices:
            if j in top_layer_set and i < j:
                first_bridge = (i, j)
                break
        if first_bridge is not None:
            break

    if first_bridge is None:
        raise RuntimeError("Failed to find neighbor pair in top layer for bridge site.")

    i_atom, j_atom = first_bridge
    pos_i = positions[i_atom]
    pos_j = positions[j_atom]

    bridge_xy = (pos_i[:2] + pos_j[:2]) / 2
    z_bridge = max(positions[top_layer_indices,2]) + 1.8
    adatom_pos = np.array([bridge_xy[0], bridge_xy[1], z_bridge])

    slab += Atom('Al', position=adatom_pos)

    # Use EAM calculator, must exist on your system!
    calc = EAM(potential=r'C:\Users\jc112358\venv\Lib\site-packages\ENERGIZE-NRT-CLASS\TST_KMC\dimer_method\Al_zhou.eam.alloy')
    slab.set_calculator(calc)

    return slab

def create_al_slab_and_adatom():
    # Create Al(111) slab
    slab = fcc111('Al', size=(4, 4, 4), vacuum=10.0, orthogonal=True)

    # Calculate the height of the topmost atom layer
    positions = slab.get_positions()
    max_z = positions[:, 2].max()

    # Define adatom position 1.8 Angstrom above the highest surface atom,
    # placed roughly at the center of the slab in x and y
    cell = slab.get_cell()
    x_center = cell[0, 0] / 2
    y_center = cell[1, 1] / 2
    adatom_pos = [x_center, y_center, max_z + 1.8]

    # Add a new Al atom at the adatom_pos
    slab += Atom('Al', position=adatom_pos)

    # Use proper EAM potential
    calc = EAM(potential=r'C:\Users\jc112358\venv\Lib\site-packages\ENERGIZE-NRT-CLASS\TST_KMC\dimer_method\Al_zhou.eam.alloy')
    slab.calc = calc
    return slab

def plot_energy_contour_with_path_centered(slab, positions, energies):
    import matplotlib.pyplot as plt
    import numpy as np
    from ase import Atom

    slab_atoms = slab[:-1]  # slab excluding the adatom
    calc = slab.calc

    cell = slab.get_cell()
    x_max = cell[0, 0]
    y_max = cell[1, 1]

    num_points = 40  # finer grid for better resolution
    # Extract initial adatom position (the center)
    initial_adatom_pos = positions[0].reshape((-1, 3))[-1][:2]

    # Define grid around zero after centering at initial adatom xy pos
    padding = 5.0  # angstroms around center to show
    x_min_shifted = -padding
    x_max_shifted = +padding
    y_min_shifted = -padding
    y_max_shifted = +padding

    X_shifted = np.linspace(x_min_shifted, x_max_shifted, num_points)
    Y_shifted = np.linspace(y_min_shifted, y_max_shifted, num_points)

    Z = np.zeros((num_points, num_points))
    z_fixed = positions[0].reshape((-1, 3))[-1][2]  # fixed height same as adatom initial height

    print(f"Calculating energy grid at z = {z_fixed:.2f} Å centered on adatom xy = {initial_adatom_pos}")

    for i, dx in enumerate(X_shifted):
        for j, dy in enumerate(Y_shifted):
            # Convert shifted (dx, dy) back to absolute (x, y)
            x_abs = initial_adatom_pos[0] + dx
            y_abs = initial_adatom_pos[1] + dy

            # Map positions inside periodic box if desired (using modulo)
            x_box = x_abs % x_max
            y_box = y_abs % y_max

            new_slab = slab_atoms.copy()
            new_slab += Atom('Al', position=[x_box, y_box, z_fixed])
            new_slab.set_calculator(calc)
            try:
                energy = new_slab.get_potential_energy()
            except Exception as e:
                print(f"Error calculating energy at ({x_box:.2f}, {y_box:.2f}): {e}")
                energy = np.nan
            Z[j, i] = energy

    # Extract adatom path xy and shift them relative to initial adatom pos
    path_coords_abs = np.array([R.reshape((-1, 3))[-1][:2] for R in positions])
    path_energies = np.array(energies)
    path_coords = path_coords_abs - initial_adatom_pos  # center path coords

    # Plotting
    plt.figure(figsize=(8, 6))
    cp = plt.contourf(X_shifted, Y_shifted, Z, levels=25, cmap='viridis')
    cbar = plt.colorbar(cp)
    cbar.set_label('Energy (eV)')
    plt.xlabel('X (Å) relative to initial adatom')
    plt.ylabel('Y (Å) relative to initial adatom')
    plt.title('Al Adatom Energy Landscape on Al(111) at fixed height – centered')

    # Plot dimer path as a line + scatter, centered coords
    plt.plot(path_coords[:, 0], path_coords[:, 1], 'r-', alpha=0.7, label='Dimer Path')
    plt.scatter(path_coords[:, 0], path_coords[:, 1], c=path_energies,
                cmap='inferno', s=60, edgecolors='k', label='Path Points')

    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    dimer_sep = 0.1  # Dimer separation distance (Angstrom)
    max_steps = 1000
    angle_step = 0.2
    translation_step = 0.1

    slab = create_al_slab_and_adatom_bridge()
    R = slab.get_positions().flatten()

    # Initialize dimer direction randomly
    N = np.random.randn(len(R))
    N = normalize(N)

    energies = []
    positions = []

    for step in range(max_steps):
        slab.set_positions(R.reshape((-1, 3)))
        energy, F = energy_and_forces(slab)
        energies.append(energy)
        positions.append(R.copy())

        curvature = dimer_curvature(slab, R, N, dimer_sep)
        tau = finite_difference_torque(slab, R, N, dimer_sep)
        N = rotate_dimer(N, tau, angle_step)
        R_new = dimer_translation_step(R, F.flatten(), N, translation_step)

        F_eff = F.flatten() - 2 * np.dot(F.flatten(), N) * N
        force_norm = np.linalg.norm(F_eff)

        print(f"Step {step:3d}: Energy={energy:.5f} eV, Curv={curvature:.5f}, |F_eff|={force_norm:.5f}")

        if force_norm < 0.05:
            print("Converged (approximate transition state reached).")
            break

        R = R_new

    slab.set_positions(R.reshape((-1, 3)))
    write('TS_al_adatom.xyz', slab)

    plot_energy_contour_with_path_centered(slab, positions, energies)


if __name__ == "__main__":
    main()