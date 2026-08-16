import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh

def kinetic_operator(N, dx):
    """Construct 1D finite-difference (-1/2) kinetic operator with Dirichlet BCs"""
    diag = np.ones(N) * 2
    offdiag = np.ones(N-1) * -1
    laplacian = (np.diag(diag) + np.diag(offdiag, -1) + np.diag(offdiag, 1)) / (dx**2)
    T = -0.5 * laplacian
    return T

def potential_linear(x, a=10.0):
    """Linear potential V(x) = a*x"""
    return a * x

def potential_spring(x, k=30.0):
    """Harmonic oscillator V(x) = 0.5 * k * x^2"""
    return 0.5 * k * x**2

def potential_lennard_jones(x, epsilon=1.0, sigma=0.5):
    """
    1D Lennard-Jones like potential centered at midpoint.
    Avoid division by zero by adding a small number to denominator.
    """
    x0 = np.mean(x)
    r = np.abs(x - x0) + 1e-6  # small offset preventing division by zero
    V = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    # Shift potential so minimum is near zero to help SCF convergence
    V -= np.min(V)
    return V

def scf_loop(N, dx, V, max_iter=100, tol=1e-8):
    T = kinetic_operator(N, dx)

    # Initialize wavefunction randomly and normalize
    psi = np.random.rand(N)
    psi /= np.linalg.norm(psi)
    psi_init = psi.copy()

    energy_old = 0.0

    '''
    STUDENT PORTION
    '''

    for iteration in range(max_iter):
        # Hamiltonian = kinetic + potential


        # Solve eigenvalue problem (calculate energy and new wavefunctions)


        # print convergence info
        # Commented out for clean output, but you can enable if you want
        # print(f"Iter {iteration+1}: Energy = {energy:.6f}, deltaE = {delta_energy:.2e}, deltaPsi = {delta_psi:.2e}")

        # check for convergence


    return psi_init, psi_new, energy

def main():
    N = 200
    dx = 0.05
    x = np.arange(N) * dx

    # Define potentials
    potentials = {
        'Linear': potential_linear(x, a=10.0),
        'Spring': potential_spring(x - np.mean(x), k=30.0),  # center the spring potential
        'Lennard-Jones': potential_lennard_jones(x, epsilon=1.0, sigma=0.1)
    }

    results = {}

    # Run SCF for each potential
    for name, V in potentials.items():
        print(f'SCF for {name} potential')
        psi_init, psi_final, E = scf_loop(N, dx, V)
        results[name] = (psi_init, psi_final, V, E)

    # Prepare 2x3 plot: top row wavefunctions, bottom row potentials
    fig, axs = plt.subplots(2, 3, figsize=(18, 8), sharex=True)

    for col, (name, (psi_init, psi_final, V, E)) in enumerate(results.items()):
        # Top row: wavefunctions
        axs[0, col].plot(x, psi_init, label='Initial $\psi$', alpha=0.7)
        axs[0, col].plot(x, psi_final, label='Final $\psi$', linewidth=2)
        axs[0, col].set_title(f'{name} potential\nFinal Energy = {E:.3f}')
        axs[0, col].set_ylabel(r'Wavefunction $\psi(x)$')
        axs[0, col].grid(True)
        axs[0, col].legend()

        # Bottom row: potential
        axs[1, col].plot(x, V, 'k-', label=f'{name} $V(x)$')
        axs[1, col].set_xlabel('x')
        axs[1, col].set_ylabel('Potential $V(x)$')
        axs[1, col].grid(True)
        axs[1, col].legend()

    plt.suptitle('SCF Wavefunctions and Potentials for Different Potentials', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()