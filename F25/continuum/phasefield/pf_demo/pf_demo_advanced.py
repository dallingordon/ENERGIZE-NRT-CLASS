import numpy as np
import matplotlib.pyplot as plt

# Define spatial domain size (nx, ny grid points)
nx, ny = 100, 100
# Define spatial step size (assuming uniform grid)
dx = 1.0
# Define time step size for the simulation
dt = 1e-3
# Number of time steps to evolve the phase field
nsteps = 1000
# Number of phases: 2 bubbles + 1 liquid phase = 3 total
N = 3

# Model parameters:
epsilon = 1   # Interface thickness parameter (controls width of interface)
M = 1         # Mobility parameter (controls interface velocity)

# Initialize the phase field array with zeros
# Shape: (N phases, nx, ny)
phi = np.zeros((N, nx, ny))

# Function to create a circle (used for initial bubbles)
def circle(cx, cy, r):
    # Create open grid coordinates for Y and X (for vectorized distance calc)
    Y, X = np.ogrid[:nx, :ny]
    # Compute distance from center (cx, cy) at each grid point
    d = np.sqrt((X - cx)**2 + (Y - cy)**2)
    # Return binary circle mask (1 inside radius r, else 0)
    return (d <= r).astype(float)

# Initialize bubbles as circles in phase 0 and 1
phi[0] = circle(30, 50, 10)  # bubble 1 centered at (30,50)
phi[1] = circle(70, 50, 10)  # bubble 2 centered at (70,50)
# Liquid phase occupies the complementary region (sum of bubbles subtracted from 1)

phi[2] = # TO-DO

# Ensure values are between 0 and 1 and sum to unity at each grid point
phi = np.clip(phi, 0, 1)
phi /= np.sum(phi, axis=0)  # normalize so sum of phases == 1

# Define a function to compute the discrete Laplacian using periodic boundary conditions
def laplacian(f):
    '''

    TO-DO

    '''

    return l

# Copy initial phi to phi_old for comparison later
phi_old = phi.copy()

# Time evolution loop
for step in range(nsteps):
    # Update each phase field independently
    for i in range(N):
    '''
    
    TO-DO
    
    '''



# Plot results: initial and evolved sum of gas phases (bubble phase 0 + 1)
fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# Initial gas fraction plot
im0 = axs[0].imshow(phi_old[0] + phi_old[1], cmap='Blues', vmin=0, vmax=1)
axs[0].set_title("Initial")
axs[0].axis('off')
cbar0 = plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
cbar0.set_label('Gas phase fraction', rotation=270, labelpad=15)

# Evolved gas fraction plot
im1 = axs[1].imshow(phi[0] + phi[1], cmap='Blues', vmin=0, vmax=1)
axs[1].set_title("After evolution")
axs[1].axis('off')
cbar1 = plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
cbar1.set_label('Gas phase fraction', rotation=270, labelpad=15)

# Improve subplot layout
plt.tight_layout()
# Show the plot
plt.show()