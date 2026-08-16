import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- True PES data (from Morse-type approximation above) ---
distances = np.array([1.60, 1.80, 2.00, 2.20, 2.40,
                      2.60, 2.70, 2.80, 3.00, 3.50,
                      4.00, 5.00])
energies_true = np.array([15.7422, 6.7177, 1.7786, -0.7702, -1.9482,
                          -2.3631, -2.4030, -2.3728, -2.1846, -1.4892,
                          -0.9048, -0.2949])

# --- Morse potential ---
def morse(r, D, a, r_e):
    return D * (1 - np.exp(-a * (r - r_e)))**2 - D

# --- Fit Morse parameters ---
p0 = [2.0, 1.0, 1.0]
params, cov = curve_fit(morse, distances, energies_true, p0=p0)
D_fit, a_fit, re_fit = params

print(f"Fitted Morse parameters:")
print(f"  D_e = {D_fit:.4f} eV")
print(f"  a   = {a_fit:.4f} Å⁻¹")
print(f"  r_e = {re_fit:.4f} Å")

# Compute fitted curve
energies_morse = morse(distances, *params)

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7,9), sharex=True)

# Full PES
ax1.plot(distances, energies_true, 'o-', label="True PES (data)", color="black")
ax1.plot(distances, energies_morse, 's--',
         label=f"Morse Fit (D={D_fit:.2f} eV, a={a_fit:.2f} Å⁻¹, rₑ={re_fit:.2f} Å)",
         color="blue")
ax1.axhline(0, color="gray", linestyle=":")
ax1.set_ylabel("Energy (eV)", fontsize=12)
ax1.set_title("Al₂ Dimer PES: True vs. Morse Fit")
ax1.legend()

# Zoomed-in view on well region
ax2.plot(distances, energies_true, 'o-', color="black")
ax2.plot(distances, energies_morse, 's--', color="blue")
ax2.axhline(0, color="gray", linestyle=":")
ax2.set_xlabel("Bond distance r (Å)", fontsize=12)
ax2.set_ylabel("Energy (eV)", fontsize=12)
ax2.set_ylim(min(energies_true) - 0.5, max(energies_true) + 0.5)

plt.tight_layout()
plt.show()
