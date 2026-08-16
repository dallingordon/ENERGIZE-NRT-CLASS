# plot_results.py

import matplotlib.pyplot as plt

# Example data from multiple runs (students fill this)
# Each tuple: (mean_rmse, speedup)
results = [
    (0.05, 3.2),
    (0.10, 5.0),
    (0.15, 7.1),
    (0.25, 12.5),
    (0.40, 20.0),
    # Add more points as you like...
]

rmses, speedups = zip(*results)

plt.figure(figsize=(6,4))
plt.scatter(rmses, speedups, s=80, color='blue')
plt.xlabel('Mean RMSE (Accuracy)')
plt.ylabel('Speedup (PDE / ML time)')
plt.title('Speedup vs Accuracy of ML-Accelerated Phase Field Simulation')
plt.grid(True)
plt.tight_layout()
plt.show()