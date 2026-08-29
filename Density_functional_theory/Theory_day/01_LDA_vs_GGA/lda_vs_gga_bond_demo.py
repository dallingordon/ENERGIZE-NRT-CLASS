"""
ME500 In-Class DFT Demo 1
==========================
LDA vs GGA for a toy two-atom bond

PURPOSE
-------
This script is NOT a DFT code. It builds a simple one-dimensional model electron
 density for two atoms and uses that density to illustrate the conceptual
 difference between LDA and GGA:

    LDA: local exchange-correlation response depends only on n(x)
    GGA: local response also depends on how rapidly n(x) changes

The exchange piece uses the familiar PBE-style enhancement factor F_x(s).
The correlation-gradient correction is deliberately simplified and bounded so
that the script remains transparent enough for an introductory lecture.

RUNNING
-------
    python lda_vs_gga_bond_demo.py

All figures and a CSV summary are written to a 'results' directory next to this
script. Nothing depends on the current working directory.

WHAT TO CHANGE IN CLASS
-----------------------
The block labeled "IN-CLASS PARAMETERS" is designed for live experimentation.
The most useful knobs are:

    ATOM_SEPARATION
        Move the two atoms closer together or farther apart.

    ATOM_WIDTH
        Change how diffuse the model atomic electron densities are.

    RUN_SEPARATION_SWEEP / SWEEP_SEPARATIONS
        Compare several bond separations automatically.

    KAPPA and MU
        Change the PBE-like exchange enhancement factor.

    CORRELATION_GRADIENT_STRENGTH
        Change the size of the toy GGA correlation correction.

Suggested questions for students:
    1. Where in space is |dn/dx| largest?
    2. Where does the reduced gradient s become large?
    3. Where do LDA and GGA differ most?
    4. What changes when the atoms are moved farther apart?
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# IN-CLASS PARAMETERS: CHANGE THESE AND RERUN
# =============================================================================

# Geometry / density model
ATOM_SEPARATION = 3.0          # distance between atom centers (toy length units)
ATOM_WIDTH = 0.90              # Gaussian width; larger = more diffuse density
ATOM_AMPLITUDE = 1.00          # peak density scale for each atom
DOMAIN_HALF_WIDTH = 8.0        # simulation domain is [-L, +L]
GRID_POINTS = 4001             # spatial resolution

# GGA exchange parameters (PBE-like form)
KAPPA = 0.804
MU = 0.21951

# Toy correlation model
# This is intentionally pedagogical, not a production correlation functional.
CORRELATION_GRADIENT_STRENGTH = 0.25
CORRELATION_A = 0.030
CORRELATION_B = 1.50

# Numerical safeguards
DENSITY_FLOOR = 1.0e-10
PLOT_DENSITY_THRESHOLD = 0.01  # hide meaningless huge s values in near-vacuum tails

# Optional live comparison across several bond lengths
RUN_SEPARATION_SWEEP = True
SWEEP_SEPARATIONS = [1.8, 2.4, 3.0, 4.0, 5.0]

# Figure behavior
SHOW_FIGURES = False           # set True for interactive classroom display
FIGURE_DPI = 180


# =============================================================================
# FILE MANAGEMENT
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TOY PHYSICS FUNCTIONS
# =============================================================================

def atomic_density(x: np.ndarray, center: float) -> np.ndarray:
    """Simple Gaussian model of an atom-centered valence electron density."""
    return ATOM_AMPLITUDE * np.exp(-0.5 * ((x - center) / ATOM_WIDTH) ** 2)


def build_density(x: np.ndarray, separation: float) -> np.ndarray:
    """Two identical atoms centered at +/- separation/2."""
    left = atomic_density(x, -0.5 * separation)
    right = atomic_density(x, +0.5 * separation)
    return left + right


def lda_exchange_per_electron(n: np.ndarray) -> np.ndarray:
    """
    Dirac exchange energy per electron in atomic-unit form:
        epsilon_x^LDA = -C_x n^(1/3)
    """
    c_x = 0.75 * (3.0 / np.pi) ** (1.0 / 3.0)
    return -c_x * np.power(np.maximum(n, DENSITY_FLOOR), 1.0 / 3.0)


def toy_lda_correlation_per_electron(n: np.ndarray) -> np.ndarray:
    """
    Smooth, negative local correlation model used only for teaching.

    It behaves qualitatively like a density-dependent LDA correlation energy,
    but it is NOT intended to reproduce a named production functional.
    """
    n_safe = np.maximum(n, DENSITY_FLOOR)
    rs = np.power(3.0 / (4.0 * np.pi * n_safe), 1.0 / 3.0)
    return -CORRELATION_A * np.log1p(CORRELATION_B / np.maximum(rs, 1.0e-12))


def reduced_gradient(x: np.ndarray, n: np.ndarray):
    """Return dn/dx, local Fermi wavevector k_F, and reduced gradient s."""
    n_safe = np.maximum(n, DENSITY_FLOOR)
    dn_dx = np.gradient(n, x)
    k_f = np.power(3.0 * np.pi**2 * n_safe, 1.0 / 3.0)
    s = np.abs(dn_dx) / (2.0 * k_f * n_safe)
    return dn_dx, k_f, s


def pbe_like_exchange_enhancement(s: np.ndarray) -> np.ndarray:
    """PBE-style exchange enhancement factor F_x(s)."""
    return 1.0 + KAPPA - KAPPA / (1.0 + MU * s**2 / KAPPA)


def toy_correlation_gradient_correction(
    eps_c_lda: np.ndarray, s: np.ndarray
) -> np.ndarray:
    """
    Bounded positive correction to the negative LDA correlation energy.

    This mimics the high-level GGA idea that correlation changes when density
    is inhomogeneous. It is deliberately simple for an introductory demo.
    """
    bounded_gradient_measure = s**2 / (1.0 + s**2)
    return (
        CORRELATION_GRADIENT_STRENGTH
        * np.abs(eps_c_lda)
        * bounded_gradient_measure
    )


def evaluate_model(x: np.ndarray, separation: float) -> dict:
    """Evaluate the complete toy LDA and GGA model for one bond separation."""
    n = build_density(x, separation)
    dn_dx, k_f, s = reduced_gradient(x, n)
    f_x = pbe_like_exchange_enhancement(s)

    eps_x_lda = lda_exchange_per_electron(n)
    eps_c_lda = toy_lda_correlation_per_electron(n)

    eps_x_gga = eps_x_lda * f_x
    delta_eps_c_grad = toy_correlation_gradient_correction(eps_c_lda, s)
    eps_c_gga = eps_c_lda + delta_eps_c_grad

    # Local energy density = n(x) * energy per electron.
    e_lda = n * (eps_x_lda + eps_c_lda)
    e_gga = n * (eps_x_gga + eps_c_gga)

    e_x_lda = n * eps_x_lda
    e_x_gga = n * eps_x_gga
    e_c_lda = n * eps_c_lda
    e_c_gga = n * eps_c_gga

    total_lda = np.trapezoid(e_lda, x)
    total_gga = np.trapezoid(e_gga, x)

    return {
        "x": x,
        "n": n,
        "dn_dx": dn_dx,
        "k_f": k_f,
        "s": s,
        "f_x": f_x,
        "eps_x_lda": eps_x_lda,
        "eps_x_gga": eps_x_gga,
        "eps_c_lda": eps_c_lda,
        "eps_c_gga": eps_c_gga,
        "delta_eps_c_grad": delta_eps_c_grad,
        "e_x_lda": e_x_lda,
        "e_x_gga": e_x_gga,
        "e_c_lda": e_c_lda,
        "e_c_gga": e_c_gga,
        "e_lda": e_lda,
        "e_gga": e_gga,
        "total_lda": total_lda,
        "total_gga": total_gga,
    }


# =============================================================================
# PLOTTING
# =============================================================================

def save_main_figure(result: dict) -> None:
    x = result["x"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(x, result["n"], linewidth=2)
    ax.axvline(-0.5 * ATOM_SEPARATION, linestyle="--", linewidth=1)
    ax.axvline(+0.5 * ATOM_SEPARATION, linestyle="--", linewidth=1)
    ax.set_title("Toy two-atom electron density")
    ax.set_xlabel("Position x")
    ax.set_ylabel("n(x)")

    ax = axes[0, 1]
    s_display = np.where(result["n"] >= PLOT_DENSITY_THRESHOLD, result["s"], np.nan)
    ax.plot(x, np.abs(result["dn_dx"]), label="|dn/dx|", linewidth=2)
    ax.plot(x, s_display, label="reduced gradient s", linewidth=2)
    ax.set_title("Density inhomogeneity")
    ax.set_xlabel("Position x")
    ax.legend()

    ax = axes[1, 0]
    ax.plot(x, result["f_x"], linewidth=2)
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.set_title("PBE-like exchange enhancement")
    ax.set_xlabel("Position x")
    ax.set_ylabel("F_x(s)")

    ax = axes[1, 1]
    ax.plot(x, result["e_lda"], label="LDA", linewidth=2)
    ax.plot(x, result["e_gga"], label="GGA", linewidth=2)
    ax.plot(
        x,
        result["e_gga"] - result["e_lda"],
        label="GGA - LDA",
        linestyle="--",
        linewidth=2,
    )
    ax.set_title("Local toy exchange-correlation energy density")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Energy density (toy units)")
    ax.legend()

    fig.suptitle(
        f"LDA vs GGA toy bond demo: separation = {ATOM_SEPARATION:.2f}",
        fontsize=15,
    )
    fig.savefig(RESULTS_DIR / "01_lda_vs_gga_bond_demo.png", dpi=FIGURE_DPI)

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def save_exchange_correlation_breakdown(result: dict) -> None:
    x = result["x"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].plot(x, result["e_x_lda"], label="LDA exchange", linewidth=2)
    axes[0].plot(x, result["e_x_gga"], label="GGA exchange", linewidth=2)
    axes[0].set_title("Exchange contribution")
    axes[0].set_xlabel("Position x")
    axes[0].set_ylabel("Energy density (toy units)")
    axes[0].legend()

    axes[1].plot(x, result["e_c_lda"], label="LDA correlation", linewidth=2)
    axes[1].plot(x, result["e_c_gga"], label="GGA correlation", linewidth=2)
    axes[1].set_title("Correlation contribution")
    axes[1].set_xlabel("Position x")
    axes[1].legend()

    fig.savefig(RESULTS_DIR / "02_exchange_correlation_breakdown.png", dpi=FIGURE_DPI)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)


def run_and_plot_separation_sweep(x: np.ndarray) -> list[dict]:
    rows = []
    for separation in SWEEP_SEPARATIONS:
        r = evaluate_model(x, float(separation))
        rows.append(
            {
                "separation": float(separation),
                "lda_total": float(r["total_lda"]),
                "gga_total": float(r["total_gga"]),
                "gga_minus_lda": float(r["total_gga"] - r["total_lda"]),
                "max_s": float(np.max(r["s"][r["n"] >= PLOT_DENSITY_THRESHOLD])),
            }
        )

    sep = np.array([row["separation"] for row in rows])
    lda = np.array([row["lda_total"] for row in rows])
    gga = np.array([row["gga_total"] for row in rows])
    diff = np.array([row["gga_minus_lda"] for row in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].plot(sep, lda, marker="o", label="LDA")
    axes[0].plot(sep, gga, marker="o", label="GGA")
    axes[0].set_xlabel("Atom separation")
    axes[0].set_ylabel("Integrated toy XC energy")
    axes[0].set_title("Integrated LDA and GGA response")
    axes[0].legend()

    axes[1].plot(sep, diff, marker="o")
    axes[1].axhline(0.0, linestyle="--", linewidth=1)
    axes[1].set_xlabel("Atom separation")
    axes[1].set_ylabel("GGA - LDA")
    axes[1].set_title("Gradient correction vs separation")

    fig.savefig(RESULTS_DIR / "03_separation_sweep.png", dpi=FIGURE_DPI)
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(fig)

    return rows


def save_csv(rows: list[dict]) -> None:
    path = RESULTS_DIR / "separation_sweep_summary.csv"
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    x = np.linspace(-DOMAIN_HALF_WIDTH, DOMAIN_HALF_WIDTH, GRID_POINTS)
    result = evaluate_model(x, ATOM_SEPARATION)

    save_main_figure(result)
    save_exchange_correlation_breakdown(result)

    print("\n" + "=" * 72)
    print("LDA vs GGA TOY BOND DEMO")
    print("=" * 72)
    print(f"Atom separation              : {ATOM_SEPARATION:.3f}")
    print(f"Integrated LDA XC energy     : {result['total_lda']:.6f}")
    print(f"Integrated GGA XC energy     : {result['total_gga']:.6f}")
    print(f"GGA - LDA                    : {result['total_gga'] - result['total_lda']:.6f}")

    correction = np.abs(result["e_gga"] - result["e_lda"])
    i_max = int(np.argmax(correction))
    print(f"Largest local correction near: x = {x[i_max]:.3f}")
    print(f"Reduced gradient there       : s = {result['s'][i_max]:.3f}")

    sweep_rows = []
    if RUN_SEPARATION_SWEEP:
        sweep_rows = run_and_plot_separation_sweep(x)
        save_csv(sweep_rows)
        print("\nSeparation sweep:")
        for row in sweep_rows:
            print(
                f"  d={row['separation']:4.1f}  "
                f"LDA={row['lda_total']: .5f}  "
                f"GGA={row['gga_total']: .5f}  "
                f"Delta={row['gga_minus_lda']: .5f}"
            )

    print(f"\nResults written to: {RESULTS_DIR}")
    print("\nTry changing ATOM_SEPARATION or ATOM_WIDTH at the top and rerun.")


if __name__ == "__main__":
    main()
