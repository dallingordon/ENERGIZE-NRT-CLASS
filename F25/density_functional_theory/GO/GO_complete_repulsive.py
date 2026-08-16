import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

def kinetic_operator_2d(Nx, Ny, dx, dy):
    N = Nx * Ny
    diag = np.ones(N) * 2 * (1 / dx**2 + 1 / dy**2)
    off_x = -1 / dx**2
    off_y = -1 / dy**2

    H = np.zeros((N, N))

    def idx(ix, iy):
        return iy * Nx + ix

    for iy in range(Ny):
        for ix in range(Nx):
            i = idx(ix, iy)
            H[i, i] = diag[i]

            if ix > 0:
                H[i, idx(ix - 1, iy)] = off_x
            if ix < Nx - 1:
                H[i, idx(ix + 1, iy)] = off_x
            if iy > 0:
                H[i, idx(ix, iy - 1)] = off_y
            if iy < Ny - 1:
                H[i, idx(ix, iy + 1)] = off_y

    return -0.5 * H

def harmonic_potential_on_grid(x_grid, y_grid, particles, k=100.0):
    """
    Construct a harmonic potential for each particle.
    V = sum_i 0.5 * k * ((x - x_i)^2 + (y - y_i)^2)
    """
    V = np.zeros_like(x_grid)
    for (px, py) in particles:
        dx = x_grid - px
        dy = y_grid - py
        V += 0.5 * k * (dx**2 + dy**2)
    return V

def harmonic_bond_force(particles, k=100.0, r0=2.5):
    """
    Force on each atom due to harmonic bond: F = - k * (r - r0) * (unit vector rij)
    Returns forces array of shape (2,2) for dimer.
    """
    rij = particles[1] - particles[0]
    r = np.linalg.norm(rij) + 1e-8
    force_magnitude = -k * (r - r0)
    force_vector = force_magnitude * (rij / r)
    forces = np.zeros_like(particles)
    forces[0] = force_vector
    forces[1] = -force_vector
    return forces

def confining_force(particles, Lx, Ly, k_wall=50.0, margin=0.3):
    """Soft harmonic wall forces near edges to confine particles."""
    forces = np.zeros_like(particles)
    for i, (x, y) in enumerate(particles):
        if x < margin:
            forces[i, 0] += k_wall * (margin - x)
        if x > Lx - margin:
            forces[i, 0] -= k_wall * (x - (Lx - margin))
        if y < margin:
            forces[i, 1] += k_wall * (margin - y)
        if y > Ly - margin:
            forces[i, 1] -= k_wall * (y - (Ly - margin))
    return forces

def scf_step_2d(H):
    eigvals, eigvecs = eigh(H)
    psi = eigvecs[:, 0]
    psi /= np.linalg.norm(psi)
    energy = eigvals[0]
    return energy, psi

def repulsive_energy_and_force(particles, A=1e-5):
    """
    Short-range repulsive potential and forces:
    E_rep = A / r^12,
    F_rep = -dE/dr = 12 A / r^13 (repulsive force pushing ions apart)
    """
    rij = particles[1] - particles[0]
    r = np.linalg.norm(rij) + 1e-12  # to avoid division by zero
    E_rep = A / r**12
    F_rep_magnitude = 12 * A / r**13
    F_rep_vector = F_rep_magnitude * (rij / r)
    forces = np.zeros_like(particles)
    forces[0] -= F_rep_vector  # repulsive force pushes ions away from each other
    forces[1] += F_rep_vector
    return E_rep, forces

def energy_and_forces(particles, x_grid, y_grid, kinetic_op, k_pot, r0, dx, dy):
    V_2d = harmonic_potential_on_grid(x_grid, y_grid, particles, k_pot)
    Nx, Ny = x_grid.shape
    V_diag = V_2d.flatten()
    H = kinetic_op + np.diag(V_diag)

    energy_scf, psi = scf_step_2d(H)

    forces_scf = np.zeros_like(particles)
    h = 5e-3  # finite difference step for force calculations

    for i, (x0, y0) in enumerate(particles):
        for dim in [0, 1]:
            shift = np.zeros_like(particles)
            shift[i, dim] = h

            V_plus = harmonic_potential_on_grid(x_grid, y_grid, particles + shift, k_pot).flatten()
            H_plus = kinetic_op + np.diag(V_plus)
            e_plus, _ = scf_step_2d(H_plus)

            V_minus = harmonic_potential_on_grid(x_grid, y_grid, particles - shift, k_pot).flatten()
            H_minus = kinetic_op + np.diag(V_minus)
            e_minus, _ = scf_step_2d(H_minus)

            forces_scf[i, dim] = -(e_plus - e_minus) / (2 * h)

    forces_bond = harmonic_bond_force(particles, k_pot, r0)
    forces_confine = confining_force(particles, x_grid.max(), y_grid.max())

    E_rep, forces_rep = repulsive_energy_and_force(particles, A=1000.0)

    energy_total = energy_scf + E_rep
    forces_total = forces_scf + forces_bond + forces_confine + forces_rep

    return energy_total, forces_total, psi

def geometry_optimization(particles_init, x_grid, y_grid, kinetic_op,
                          k_pot, r0, dx, dy,
                          max_steps=50, tol=1e-4, step_size=0.005):
    particles = particles_init.copy()
    energies = []
    forces_norm = []
    positions = [particles.copy()]

    for step in range(max_steps):
        energy, forces, psi = energy_and_forces(particles, x_grid, y_grid,
                                                kinetic_op, k_pot, r0, dx, dy)
        energies.append(energy)

        # Clip forces to max norm 1.0
        max_force_norm = 1.0
        for i in range(len(forces)):
            norm = np.linalg.norm(forces[i])
            if norm > max_force_norm:
                forces[i] = (forces[i] / norm) * max_force_norm

        total_force = np.linalg.norm(forces, axis=1).sum()
        forces_norm.append(total_force)

        print(f"Step {step + 1}: Energy = {energy:.6f}, Sum force = {total_force:.6f}",
              f" bond length: {np.linalg.norm(particles[0] - particles[1]):.4f}")

        if total_force < tol:
            print("Converged geometry optimization!")
            break

        particles += step_size * forces
        particles[:, 0] = np.clip(particles[:, 0], 0, x_grid.max())
        particles[:, 1] = np.clip(particles[:, 1], 0, y_grid.max())

        positions.append(particles.copy())

    return particles, energies, forces_norm, positions, psi

def main():
    Nx, Ny = 50, 50
    Lx, Ly = 5.0, 5.0
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    x_grid, y_grid = np.meshgrid(x, y, indexing='ij')

    kinetic_op = kinetic_operator_2d(Nx, Ny, dx, dy)

    # Initial dimer positions
    initial_bond_length = 2.0  # start stretched relative to equilibrium
    center_y = Ly / 2
    particle1 = np.array([Lx / 2 - initial_bond_length / 2, center_y])
    particle2 = np.array([Lx / 2 + initial_bond_length / 2, center_y])
    particles_init = np.array([particle1, particle2])
    print("Initial dimer positions:\n", particles_init)

    k_pot = 1.0  # spring constant for potential and bond force
    r0 = 2.5       # approximate equilibrium bond length

    particles_opt, energies, forces_norm, positions, psi = geometry_optimization(
        particles_init, x_grid, y_grid, kinetic_op,
        k_pot, r0, dx, dy,
        max_steps=300, tol=1e-3, step_size=0.005)

    print("Optimized dimer positions:\n", particles_opt)
    print(f"Optimized bond length: {np.linalg.norm(particles_opt[1] - particles_opt[0]):.4f}")

    # Bond distances for all iterations
    bond_distances = [np.linalg.norm(pos[1] - pos[0]) for pos in positions]

    # Plot total energy, sum forces, and bond distance over GO iterations
    fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    axs[0].plot(energies, '-o')
    axs[0].set_ylabel('Total Energy')
    axs[0].set_title('Energy vs GO iteration')
    axs[0].grid(True)

    axs[1].plot(forces_norm, '-o', color='r')
    axs[1].set_ylabel('Sum of forces')
    axs[1].set_title('Sum of forces vs GO iteration')
    axs[1].grid(True)

    axs[2].plot(bond_distances, '-o', color='g')
    axs[2].set_ylabel('Bond distance')
    axs[2].set_xlabel('GO iteration')
    axs[2].set_title('Bond distance vs GO iteration')
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()