import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.sparse import lil_matrix

# --- Parameters ---
L = 1.0               # Length of the 1D rod in meters
nx = 50               # Number of spatial discretization points
dx = L / (nx - 1)     # Spatial grid spacing (uniform)
alpha = 1e-4          # Thermal diffusivity (m^2/s); material property controlling conduction rate
T_left = 100.0        # Dirichlet boundary condition: fixed temperature at the left boundary (deg C)
T_initial = 20.0      # Initial temperature for the entire domain (deg C)
dt = 0.5              # Time step size in seconds (chosen based on stability / accuracy considerations)
nt = 1000            # Number of time steps for temporal integration

x = np.linspace(0, L, nx)  # Create spatial grid array from 0 to L

# --- Initial condition setup ---
# Initialize temperature arrays for the three numerical methods: FDM, FEM, FVM
T_fdm = np.ones(nx) * T_initial  # Initial temperature profile for Finite Difference Method
T_fem = np.ones(nx) * T_initial  # Initial temperature profile for Finite Element Method
T_fvm = np.ones(nx) * T_initial  # Initial temperature profile for Finite Volume Method

# Enforce Dirichlet boundary condition at left boundary: temperature fixed at T_left
T_fdm[0] = T_left
T_fem[0] = T_left
T_fvm[0] = T_left

# --- Finite Difference Method (Explicit Scheme) ---
def update_fdm(T, alpha, dx, dt):
    """
    Explicit time integration using central difference for spatial second derivative.
    Solves one time step of 1D heat conduction equation:
    T_new[i] = T[i] + (alpha*dt/dx^2)*(T[i+1] - 2*T[i] + T[i-1])
    """
    T_new = T.copy()  # Work on a copy to update simultaneously

    # Loop over interior nodes (excluding boundaries)
    for i in range(1, nx - 1):
        T_new[i] = T[i] + alpha * dt / dx**2 * (T[i+1] - 2 * T[i] + T[i-1])

    # Right boundary: Neumann BC (insulated) → zero gradient dT/dx=0 implemented as T_new at last point equals second last
    T_new[-1] = T_new[-2]

    # Left boundary: Dirichlet BC enforced explicitly (fixed temperature)
    T_new[0] = T_left

    return T_new

# --- Finite Element Method (Implicit backward Euler) ---

def fem_setup(nx, dx, alpha):
    """
    Assemble 1D linear finite element mass matrix and stiffness matrix
    for heat conduction problem.
    Using linear basis functions, mass matrix M approximate integral of N_i N_j,
    stiffness matrix K approximate integral of dN_i/dx dN_j/dx.
    These matrices are scaled by dx and alpha accordingly.
    Returns sparse matrices M and K (csc format).
    """

    # Stiffness matrix diagonals: 2 on diagonal, -1 on off-diagonals (from second derivative discretization)
    main_diag = np.full(nx, 2)
    off_diag = np.full(nx - 1, -1)
    stiffness = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
    stiffness = stiffness * alpha / dx  # scale with diffusivity and mesh size

    # Mass matrix typical values for linear elements: 4/6 on diagonal, 1/6 on off-diagonal
    mass_main = np.full(nx, 4 / 6)
    mass_off = np.full(nx - 1, 1 / 6)
    mass = diags([mass_off, mass_main, mass_off], [-1, 0, 1]).tocsc()
    mass = mass * dx  # scale with element size

    # Note: Dirichlet BC handled later by modifying system matrix and RHS vector

    return mass, stiffness

# Assemble FEM matrices once since mesh and parameters are fixed
mass, stiffness = fem_setup(nx, dx, alpha)
A = (mass + dt * stiffness).tocsc()  # Left-hand-side matrix for implicit Euler: (M + dt K)

def fem_step(T_old):
    """
    Perform one implicit time step in FEM:
    Solve (M + dt K) T_new = M T_old + b
    with Dirichlet BC enforced by row modifications.
    """
    A_bc = A.copy().tolil()       # Convert to LIL format for easy row modification
    b = mass @ T_old              # Compute right-hand side M T_old

    # Apply Dirichlet BC at left node (index 0):
    # Zero out first row of matrix and set diagonal to 1
    A_bc[0, :] = 0
    A_bc[0, 0] = 1
    # Set first value of RHS vector to prescribed temperature
    b[0] = T_left

    A_bc = A_bc.tocsc()          # Convert back to sparse CSC format for efficient solver

    T_new = spsolve(A_bc, b)     # Solve sparse linear system

    return T_new

# --- Finite Volume Method (Implicit backward Euler) ---

def fvm_setup(nx, dx, alpha, dt):
    """
    Setup coefficient matrix A corresponding to implicit FVM discretization:
    (I + alpha*dt/dx^2 * Laplacian) structure.
    Main diagonal is 1 + 2*coeff; off diagonals are -coeff.
    """
    coeff = alpha * dt / dx**2
    main_diag = np.full(nx, 1 + 2 * coeff)
    off_diag = np.full(nx -1, -coeff)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()

    return A

# Assemble FVM system matrix once
A_fvm = fvm_setup(nx, dx, alpha, dt)

def fvm_step(T_old):
    """
    Perform one implicit FVM time step by solving A T_new = T_old with BC enforcement.
    Dirichlet BC at left enforced by modifying first row of matrix and RHS.
    """
    b = T_old.copy()  # Right hand side vector

    # Enforce Dirichlet BC: Left node temperature fixed at T_left
    b[0] = T_left

    # Modify matrix first row to enforce Dirichlet BC explicitly
    A_mod = A_fvm.copy().tolil()
    A_mod[0, :] = 0
    A_mod[0, 0] = 1
    A_mod = A_mod.tocsc()

    T_new = spsolve(A_mod, b)  # Solve linear system for new temperature

    return T_new

# --- Prepare animation setup for visualization of temperature over time ---

fig, axs = plt.subplots(1,3, figsize=(15,4))  # Create a figure with 3 side-by-side subplots
lines = []
titles = ["FDM (Explicit)", "FEM (Implicit)", "FVM (Implicit)"]
datasets = [T_fdm, T_fem, T_fvm]

# Initialize plots with initial temperature
for ax, title in zip(axs, titles):
    line, = ax.plot(x, np.ones_like(x)*T_initial, lw=2)  # plot initial uniform temp
    lines.append(line)
    ax.set_ylim(0, 110)               # fixed y-axis limits for consistent comparison
    ax.set_title(title)                # subplot title indicating method
    ax.set_xlabel("Position (m)")     # x-axis label (spatial coordinate)
    ax.set_ylabel("Temperature (°C)") # y-axis label
    ax.grid(True)                     # add grid for better visualization

time_text = fig.suptitle("")  # Title to show current time during animation

def animate(i):
    """
    Animation update function called sequentially to update temperature profiles.
    Advances simulation by one time step for each method.
    """
    global T_fdm, T_fem, T_fvm

    # Update temperature profiles advancing one time step
    T_fdm = update_fdm(T_fdm, alpha, dx, dt)
    T_fem = fem_step(T_fem)
    T_fvm = fvm_step(T_fvm)

    # Update data for each plot line
    for line, T in zip(lines, [T_fdm, T_fem, T_fvm]):
        line.set_ydata(T)

    time_text.set_text(f"Time = {i*dt:.1f} s")  # Update title with current time

    return lines + [time_text]

# Create animation object, calling animate function for nt frames
ani = FuncAnimation(fig, animate, frames=nt, interval=50, blit=False)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()