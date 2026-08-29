"""
ME500 In-Class DFT Demo 2
==========================
Toy PAW partial-wave fitting / pseudization demo

PURPOSE
-------
This script is NOT a PAW dataset generator and does not perform DFT. It creates a
synthetic all-electron-like radial partial wave with rapid core oscillations,
then replaces the core region with a smooth pseudo partial wave.

The demonstration is designed to reinforce three ideas:

    1. All-electron functions can vary rapidly near a nucleus.
    2. A smooth pseudo representation is much easier to represent numerically.
    3. PAW keeps information that lets the all-electron character be restored
       locally inside an augmentation region.

The script also computes a simple Fourier-space complexity proxy to show why a
smooth pseudo function requires fewer high-frequency basis components.

RUNNING
-------
    python paw_partial_wave_demo.py

All figures and a CSV summary are written to a 'results' directory next to this
script, regardless of the directory from which you run it.

WHAT TO CHANGE IN CLASS
-----------------------
The block labeled "IN-CLASS PARAMETERS" contains the useful knobs:

    AUGMENTATION_RADIUS
        Radius inside which the all-electron function is replaced by a smooth
        pseudo function. This is the most important parameter to vary live.

    CORE_OSCILLATION_FREQUENCY
        Controls how rapidly the synthetic all-electron wave oscillates near
        the nucleus. Higher values make the all-electron representation harder.

    CORE_OSCILLATION_STRENGTH
        Controls how much core structure must be removed by pseudization.

    PSEUDO_CENTER_FRACTION
        Controls the finite smooth value imposed at r = 0.

    RUN_RADIUS_SWEEP / SWEEP_RADII
        Compares several augmentation radii automatically.

Suggested questions for students:
    1. Where are phi_AE and phi_tilde identical?
    2. Where is the PAW correction phi_AE - phi_tilde non-zero?
    3. How does changing r_c alter smoothness?
    4. Why does the pseudo function contain less high-k Fourier content?
    5. What accuracy/cost tradeoff appears as r_c is changed?
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# IN-CLASS PARAMETERS: CHANGE THESE AND RERUN
# =============================================================================

# Radial grid
R_MAX = 8.0
GRID_POINTS = 5001

# Synthetic all-electron partial-wave shape
VALENCE_DECAY = 0.55
VALENCE_NODE_SCALE = 4.5
CORE_OSCILLATION_FREQUENCY = 15.0
CORE_OSCILLATION_STRENGTH = 0.55
CORE_LOCALIZATION_WIDTH = 1.15

# Pseudization / augmentation region
AUGMENTATION_RADIUS = 1.30
PSEUDO_CENTER_FRACTION = 0.65

# Optional sweep to show the cutoff-radius tradeoff
RUN_RADIUS_SWEEP = True
SWEEP_RADII = [0.80, 1.00, 1.30, 1.60, 2.00]

# Fourier complexity metric
SPECTRAL_POWER_FRACTION = 0.99

# Figures
SHOW_FIGURES = False
FIGURE_DPI = 180


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# SYNTHETIC ALL-ELECTRON AND PSEUDO PARTIAL WAVES
# =============================================================================

def all_electron_partial_wave(r: np.ndarray) -> np.ndarray:
    """
    Synthetic radial partial wave.

    The slowly varying factor represents valence-scale behavior. A localized
    oscillatory core factor adds deliberately expensive high-frequency content
    close to the nucleus.
    """
    valence = np.exp(-VALENCE_DECAY * r) * (1.0 - r / VALENCE_NODE_SCALE)
    core_envelope = np.exp(-(r / CORE_LOCALIZATION_WIDTH) ** 2)
    core_structure = 1.0 + CORE_OSCILLATION_STRENGTH * core_envelope * np.cos(
        CORE_OSCILLATION_FREQUENCY * r
    )
    return valence * core_structure


def derivatives_at(r: np.ndarray, y: np.ndarray, r_c: float):
    """Interpolate value, first derivative, and second derivative at r_c."""
    dy = np.gradient(y, r)
    d2y = np.gradient(dy, r)
    y_c = float(np.interp(r_c, r, y))
    dy_c = float(np.interp(r_c, r, dy))
    d2y_c = float(np.interp(r_c, r, d2y))
    return y_c, dy_c, d2y_c


def fit_smooth_core(r: np.ndarray, phi_ae: np.ndarray, r_c: float) -> np.ndarray:
    """
    Fit an even sixth-order polynomial inside r_c:

        p(r) = a0 + a2 r^2 + a4 r^4 + a6 r^6

    Four constraints are imposed:
        p(0)    = chosen smooth center value
        p(r_c)  = phi_AE(r_c)
        p'(r_c) = phi_AE'(r_c)
        p''(r_c)= phi_AE''(r_c)

    Outside r_c, the pseudo and all-electron functions are identical.
    """
    y_c, dy_c, d2y_c = derivatives_at(r, phi_ae, r_c)
    center_value = PSEUDO_CENTER_FRACTION * y_c

    rc = float(r_c)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, rc**2, rc**4, rc**6],
            [0.0, 2.0 * rc, 4.0 * rc**3, 6.0 * rc**5],
            [0.0, 2.0, 12.0 * rc**2, 30.0 * rc**4],
        ],
        dtype=float,
    )
    rhs = np.array([center_value, y_c, dy_c, d2y_c], dtype=float)
    a0, a2, a4, a6 = np.linalg.solve(matrix, rhs)

    pseudo = phi_ae.copy()
    inside = r <= rc
    rin = r[inside]
    pseudo[inside] = a0 + a2 * rin**2 + a4 * rin**4 + a6 * rin**6
    return pseudo


# =============================================================================
# COMPLEXITY / FOURIER PROXY
# =============================================================================

def spectral_power_curve(r: np.ndarray, phi: np.ndarray):
    """
    Return a simple Fourier-space complexity proxy.

    We create an even extension from r >= 0 to x in [-R_MAX, R_MAX], then use a
    real-space FFT. This is not a spherical-Bessel transform; it is intentionally
    a transparent proxy for the amount of high-frequency content.
    """
    dr = r[1] - r[0]
    extended = np.concatenate((phi[:0:-1], phi))
    fft = np.fft.rfft(extended)
    power = np.abs(fft) ** 2
    k = 2.0 * np.pi * np.fft.rfftfreq(len(extended), d=dr)
    if power.sum() > 0:
        power = power / power.sum()
    return k, power


def k_for_power_fraction(k: np.ndarray, power: np.ndarray, fraction: float) -> float:
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, fraction, side="left"))
    idx = min(idx, len(k) - 1)
    return float(k[idx])


def smoothness_metric(r: np.ndarray, phi: np.ndarray, r_c: float) -> float:
    """Integral of squared curvature inside the augmentation region."""
    d1 = np.gradient(phi, r)
    d2 = np.gradient(d1, r)
    mask = r <= r_c
    return float(np.trapezoid(d2[mask] ** 2, r[mask]))


def core_rms_difference(r: np.ndarray, a: np.ndarray, b: np.ndarray, r_c: float) -> float:
    mask = r <= r_c
    if not np.any(mask):
        return 0.0
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


# =============================================================================
# EVALUATION AND PLOTTING
# =============================================================================

def evaluate_radius(r: np.ndarray, phi_ae: np.ndarray, r_c: float) -> dict:
    pseudo = fit_smooth_core(r, phi_ae, r_c)
    correction = phi_ae - pseudo

    k_ae, p_ae = spectral_power_curve(r, phi_ae)
    k_ps, p_ps = spectral_power_curve(r, pseudo)

    return {
        "r_c": float(r_c),
        "pseudo": pseudo,
        "correction": correction,
        "k_ae": k_ae,
        "p_ae": p_ae,
        "k_ps": k_ps,
        "p_ps": p_ps,
        "k99_ae": k_for_power_fraction(k_ae, p_ae, SPECTRAL_POWER_FRACTION),
        "k99_pseudo": k_for_power_fraction(k_ps, p_ps, SPECTRAL_POWER_FRACTION),
        "smoothness_ae": smoothness_metric(r, phi_ae, r_c),
        "smoothness_pseudo": smoothness_metric(r, pseudo, r_c),
        "core_rms_difference": core_rms_difference(r, phi_ae, pseudo, r_c),
    }


def save_main_figure(r: np.ndarray, phi_ae: np.ndarray, result: dict) -> None:
    rc = result["r_c"]
    pseudo = result["pseudo"]
    correction = result["correction"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(r, phi_ae, label="all-electron-like $\\phi$", linewidth=2)
    ax.plot(r, pseudo, label="pseudo $\\tilde{\\phi}$", linewidth=2)
    ax.axvspan(0.0, rc, alpha=0.12, label="augmentation region")
    ax.axvline(rc, linestyle="--", linewidth=1)
    ax.set_xlim(0, min(R_MAX, 4.5))
    ax.set_title("All-electron vs pseudo partial wave")
    ax.set_xlabel("Radius r")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(r, correction, linewidth=2)
    ax.axvline(rc, linestyle="--", linewidth=1)
    ax.axhline(0.0, linewidth=1)
    ax.set_xlim(0, min(R_MAX, 4.5))
    ax.set_title(r"Local PAW correction: $\phi-\tilde{\phi}$")
    ax.set_xlabel("Radius r")

    ax = axes[1, 0]
    ax.semilogy(result["k_ae"], result["p_ae"] + 1e-18, label="all-electron-like")
    ax.semilogy(result["k_ps"], result["p_ps"] + 1e-18, label="pseudo")
    ax.axvline(result["k99_ae"], linestyle="--", linewidth=1)
    ax.axvline(result["k99_pseudo"], linestyle="--", linewidth=1)
    ax.set_xlim(0, max(40.0, 1.2 * result["k99_ae"]))
    ax.set_title("Fourier-space complexity proxy")
    ax.set_xlabel("Spatial frequency k")
    ax.set_ylabel("Normalized spectral power")
    ax.legend()

    ax = axes[1, 1]
    labels = ["AE-like", "Pseudo"]
    smooth = [result["smoothness_ae"], result["smoothness_pseudo"]]
    k99 = [result["k99_ae"], result["k99_pseudo"]]
    x_pos = np.arange(2)
    width = 0.36
    ax.bar(x_pos - width / 2, smooth, width, label="core curvature metric")
    ax2 = ax.twinx()
    ax2.bar(x_pos + width / 2, k99, width, alpha=0.55, label="k at 99% power")
    ax.set_xticks(x_pos, labels)
    ax.set_ylabel("Curvature / smoothness metric")
    ax2.set_ylabel("Fourier cutoff proxy")
    ax.set_title("Why smoothing lowers numerical cost")

    # Combine legends from both axes.
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.suptitle(f"Toy PAW pseudization demo: augmentation radius = {rc:.2f}", fontsize=15)
    fig.savefig(RESULTS_DIR / "01_paw_partial_wave_demo.png", dpi=FIGURE_DPI)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def run_radius_sweep(r: np.ndarray, phi_ae: np.ndarray) -> list[dict]:
    rows = []
    for rc in SWEEP_RADII:
        result = evaluate_radius(r, phi_ae, float(rc))
        rows.append(
            {
                "augmentation_radius": float(rc),
                "core_rms_difference": result["core_rms_difference"],
                "smoothness_ae": result["smoothness_ae"],
                "smoothness_pseudo": result["smoothness_pseudo"],
                "k99_ae": result["k99_ae"],
                "k99_pseudo": result["k99_pseudo"],
                "k99_reduction_percent": 100.0
                * (result["k99_ae"] - result["k99_pseudo"])
                / max(result["k99_ae"], 1e-12),
            }
        )

    rc = np.array([row["augmentation_radius"] for row in rows])
    smooth = np.array([row["smoothness_pseudo"] for row in rows])
    rms = np.array([row["core_rms_difference"] for row in rows])
    k99 = np.array([row["k99_pseudo"] for row in rows])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    axes[0].plot(rc, smooth, marker="o")
    axes[0].set_xlabel("Augmentation radius")
    axes[0].set_ylabel("Pseudo curvature metric")
    axes[0].set_title("Smoothness")

    axes[1].plot(rc, rms, marker="o")
    axes[1].set_xlabel("Augmentation radius")
    axes[1].set_ylabel("Core RMS difference")
    axes[1].set_title("Amount of core replacement")

    axes[2].plot(rc, k99, marker="o")
    axes[2].set_xlabel("Augmentation radius")
    axes[2].set_ylabel("k at 99% spectral power")
    axes[2].set_title("Fourier complexity proxy")

    fig.suptitle("Toy augmentation-radius tradeoff")
    fig.savefig(RESULTS_DIR / "02_augmentation_radius_sweep.png", dpi=FIGURE_DPI)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return rows


def save_csv(rows: list[dict]) -> None:
    if not rows:
        return
    path = RESULTS_DIR / "augmentation_radius_sweep_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    r = np.linspace(0.0, R_MAX, GRID_POINTS)
    phi_ae = all_electron_partial_wave(r)
    result = evaluate_radius(r, phi_ae, AUGMENTATION_RADIUS)

    save_main_figure(r, phi_ae, result)

    print("\n" + "=" * 72)
    print("TOY PAW PARTIAL-WAVE FITTING DEMO")
    print("=" * 72)
    print(f"Augmentation radius r_c      : {AUGMENTATION_RADIUS:.3f}")
    print(f"AE-like core curvature       : {result['smoothness_ae']:.6e}")
    print(f"Pseudo core curvature        : {result['smoothness_pseudo']:.6e}")
    print(f"AE-like k@99% spectral power : {result['k99_ae']:.3f}")
    print(f"Pseudo k@99% spectral power  : {result['k99_pseudo']:.3f}")
    print(f"Core RMS replacement         : {result['core_rms_difference']:.6f}")

    rows = []
    if RUN_RADIUS_SWEEP:
        rows = run_radius_sweep(r, phi_ae)
        save_csv(rows)
        print("\nAugmentation-radius sweep:")
        for row in rows:
            print(
                f"  r_c={row['augmentation_radius']:4.2f}  "
                f"pseudo curvature={row['smoothness_pseudo']:10.3e}  "
                f"k99={row['k99_pseudo']:7.3f}  "
                f"core RMS={row['core_rms_difference']:7.4f}"
            )

    print(f"\nResults written to: {RESULTS_DIR}")
    print("\nTry changing AUGMENTATION_RADIUS or CORE_OSCILLATION_FREQUENCY at the top.")


if __name__ == "__main__":
    main()
