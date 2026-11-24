#!/usr/bin/env python3
"""2D multiphase phase-field classroom demo using FINITE DIFFERENCES (FDM).

Edit the USER SETTINGS section below, then run:
    python phase_field_fdm_2d.py

Physical model (from the lecture slides):
    f = sum_i g_i h(phi_i) + W sum_{i<j} phi_i**2 phi_j**2
    h(phi) = phi**2 (3 - 2 phi)
    F = integral [f + sum_i kappa_i/2 |grad phi_i|**2] dA
    d(phi_i)/dt = -L_i [delta F/delta phi_i - lambda(x,y,t)]

The Lagrange multiplier lambda ensures sum_i phi_i = 1 at each pixel.
SPATIAL METHOD: FDM, 5-point centered Laplacian on a rectangular grid.
TIME METHOD: explicit Euler, RK4, or semi-implicit Euler/FFT linear solve.
All quantities are DIMENSIONLESS; this is not a calibrated materials model.
"""

# ============================================================================
# USER SETTINGS: CHANGE THESE VALUES FOR YOUR CLASS EXPERIMENTS
# ============================================================================

# (A) SPATIAL DISCRETIZATION: finite differences on a RECTANGULAR GRID.
# FDM is the only spatial method implemented in this script.
SPATIAL_METHOD = "fdm"

# Change NX and NY to study the spatial mesh. Examples:
#   NX = NY = 32      very coarse (fast but interfaces poorly resolved)
#   NX = NY = 64      coarse
#   NX = NY = 128     default (recommended first run)
#   NX = NY = 256     fine (more computation / stricter explicit timestep)
# You can also try NX=160, NY=96 for a rectangular grid.
NX = 32
NY = 32

# KEEP THE DOMAIN LENGTHS FIXED WHEN COMPARING GRID RESOLUTIONS.
# The script calculates DX = LENGTH_X/NX and DY = LENGTH_Y/NY for you.
LENGTH_X = 128.0
LENGTH_Y = 128.0

# (B) TIME INTEGRATOR: independent of spatial method above.
# All 3 choices solve the SAME finite-difference semi-discrete equations.
# For semi_implicit_fft, FFT is the algebraic solver for the implicit FD
# Laplacian (its eigenvalues come from the five-point FD stencil).
TIME_INTEGRATOR = "semi_implicit_fft"
# Choices: "euler", "rk4", "semi_implicit_fft"

# Initial microstructure: "solidification", "melting", or "coarsening".
SCENARIO = "solidification"

# Phase order: solid alpha, solid beta, solid gamma, liquid.
# KAPPA[i] is the interface-gradient coefficient for phase i.
KAPPA = [8.0, 8.0, 8.0, 8.0]
# MOBILITY[i] is the kinetic coefficient L_i for phase i.
MOBILITY = [1.0, 1.0, 1.0, 1.0]

# Local free-energy terms:
#   h(phi) = phi^2 (3-2phi)
#   f = sum_i G_i h(phi_i) + W_OVERLAP * sum_{i<j} phi_i^2 phi_j^2.
# For solidification G_solid < G_liquid; for melting the reverse.
G_SOLIDIFICATION = [-0.70, -0.70, -0.70, 0.00]
G_MELTING        = [ 0.50,  0.50,  0.50,-0.55]
G_COARSENING     = [ 0.00,  0.00,  0.00, 2.00]
W_OVERLAP = 2.0

# Time integration. All scenarios stop at the same simulated FINAL_TIME.
# If AUTO_LIMIT_DT is True, explicit solvers automatically reduce DT if needed.
# Turn it off to investigate the consequences of using a too-large timestep.
FINAL_TIME = 14.0
DT = 0.02
AUTO_LIMIT_DT = True

# Number of sampled frames in the saved animation (including first/last).
ANIMATION_FRAMES = 32
VIDEO_FPS = 8
VIDEO_FORMAT = "auto"   # "auto" (MP4 if ffmpeg exists, else GIF), "mp4", "gif"
SAVE_ANIMATION = True

# Diagnostics are written as CSV; images + frame arrays are saved as well.
DIAGNOSTICS_EVERY = 10
OUTPUT_DIRECTORY = "results"
RANDOM_SEED = 42

# Initial diffuse interface thickness (dimensionless length units).
# At coarse meshes, this may be resolved by fewer than 2 grid cells.
INITIAL_INTERFACE_WIDTH = 2.5

# Show the actual mesh over a zoomed-in part of the initial microstructure.
# The zoom is the SAME PHYSICAL REGION even when NX and NY change.
SAVE_MESH_PLOT = True

# ============================================================================
# END OF USER SETTINGS. STUDENTS DO NOT NEED TO EDIT BELOW THIS LINE.
# ============================================================================

from datetime import datetime
from pathlib import Path
import csv
import json
import math
import shutil
import time
import warnings

import matplotlib
matplotlib.use("Agg")  # works in classrooms, terminals, and remote SSH sessions
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib.patches import Patch
import numpy as np

PHASE_NAMES = ("Solid alpha", "Solid beta", "Solid gamma", "Liquid")
PHASE_COLORS = np.array([
    [0.98, 0.79, 0.42],  # alpha (gold)
    [0.54, 0.79, 0.56],  # beta (green)
    [0.77, 0.66, 0.89],  # gamma (purple)
    [0.34, 0.59, 0.82],  # liquid (blue)
], dtype=np.float64)
LINE_COLORS = ("#c08018", "#41894a", "#9463be", "#2872b4")


def check_settings():
    if not isinstance(NX, int) or not isinstance(NY, int) or NX < 16 or NY < 16:
        raise ValueError("NX and NY must be integers >= 16.")
    if LENGTH_X <= 0 or LENGTH_Y <= 0:
        raise ValueError("Domain lengths must be positive.")
    if SPATIAL_METHOD.lower() != "fdm":
        raise ValueError("This teaching script implements spatial method 'fdm' only.")
    if TIME_INTEGRATOR not in ("euler", "rk4", "semi_implicit_fft"):
        raise ValueError("TIME_INTEGRATOR must be 'euler', 'rk4', or 'semi_implicit_fft'.")
    if SCENARIO not in ("solidification", "melting", "coarsening"):
        raise ValueError("Choose solidification, melting, or coarsening.")
    if VIDEO_FORMAT not in ("auto", "mp4", "gif"):
        raise ValueError("VIDEO_FORMAT must be auto, mp4, or gif.")
    if len(KAPPA) != 4 or len(MOBILITY) != 4:
        raise ValueError("Provide four KAPPA and four MOBILITY values (three solids + liquid).")
    if min(KAPPA) < 0 or min(MOBILITY) <= 0 or W_OVERLAP <= 0:
        raise ValueError("KAPPA must be nonnegative, MOBILITY and W_OVERLAP positive.")
    if FINAL_TIME <= 0 or DT <= 0 or ANIMATION_FRAMES < 2 or VIDEO_FPS < 1:
        raise ValueError("FINAL_TIME, DT, VIDEO_FPS must be positive; use >=2 frames.")
    if DIAGNOSTICS_EVERY < 1 or INITIAL_INTERFACE_WIDTH <= 0:
        raise ValueError("DIAGNOSTICS_EVERY and INITIAL_INTERFACE_WIDTH must be positive.")


def settings_dict():
    names = (
        "SPATIAL_METHOD", "NX", "NY", "LENGTH_X", "LENGTH_Y", "TIME_INTEGRATOR", "SCENARIO", "KAPPA",
        "MOBILITY", "G_SOLIDIFICATION", "G_MELTING", "G_COARSENING",
        "W_OVERLAP", "FINAL_TIME", "DT", "AUTO_LIMIT_DT", "ANIMATION_FRAMES",
        "VIDEO_FPS", "VIDEO_FORMAT", "SAVE_ANIMATION", "DIAGNOSTICS_EVERY",
        "RANDOM_SEED", "INITIAL_INTERFACE_WIDTH", "SAVE_MESH_PLOT",
    )
    return {key: globals()[key] for key in names}


class Model:
    def __init__(self):
        self.dx = LENGTH_X / NX
        self.dy = LENGTH_Y / NY
        self.shape = (4, NY, NX)
        self.kappa = np.asarray(KAPPA, dtype=np.float64)[:, None, None]
        self.mobility = np.asarray(MOBILITY, dtype=np.float64)[:, None, None]
        self.sum_mobility = self.mobility.sum()
        self.g = np.asarray({
            "solidification": G_SOLIDIFICATION,
            "melting": G_MELTING,
            "coarsening": G_COARSENING,
        }[SCENARIO], dtype=np.float64)[:, None, None]
        if self.g.shape[0] != 4:
            raise ValueError("G array needs four entries (one per phase).")
        # Eigenvalues of -1 times the SAME periodic, centered finite-difference
        # Laplacian used by Euler and RK4 (rather than a different spectral PDE).
        kx = np.fft.fftfreq(NX)  # grid frequencies m/NX
        ky = np.fft.fftfreq(NY)
        self.laplace_eigenvalue = (
            4.0 * np.sin(np.pi * ky[:, None]) ** 2 / self.dy ** 2
            + 4.0 * np.sin(np.pi * kx[None, :]) ** 2 / self.dx ** 2
        )[None, :, :]

    def laplacian(self, phi):
        """2D FDM: (right-2*center+left)/dx**2 + (up-2*center+down)/dy**2.
        np.roll applies periodic x/y boundary conditions.
        Note: dx and dy need NOT be equal.
        """
        return (
            (np.roll(phi, 1, axis=2) + np.roll(phi, -1, axis=2) - 2 * phi)
            / self.dx**2
            + (np.roll(phi, 1, axis=1) + np.roll(phi, -1, axis=1) - 2 * phi)
            / self.dy**2
        )

    def df_dphi(self, phi):
        # derivative of g_i phi_i^2 (3-2phi_i) is 6*g_i*phi_i*(1-phi_i)
        p2 = phi * phi
        # derivative of W sum_(i<j) phi_i^2 phi_j^2
        return (6 * self.g * phi * (1 - phi)
                + 2 * W_OVERLAP * phi * (p2.sum(axis=0, keepdims=True) - p2))

    def rhs(self, phi):
        chemical_potential = self.df_dphi(phi) - self.kappa * self.laplacian(phi)
        # Orthogonal descent with a pointwise sum(phi_i) = 1 constraint.
        lam = (self.mobility * chemical_potential).sum(axis=0, keepdims=True) / self.sum_mobility
        return -self.mobility * (chemical_potential - lam)

    def step_euler(self, phi, dt):
        return phi + dt * self.rhs(phi)

    def step_rk4(self, phi, dt):
        k1 = self.rhs(phi)
        k2 = self.rhs(phi + 0.5 * dt * k1)
        k3 = self.rhs(phi + 0.5 * dt * k2)
        k4 = self.rhs(phi + dt * k3)
        return phi + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def step_semi_implicit_fft(self, phi, dt):
        """Semi-implicit Euler: implicit FD Laplacian; explicit local potential.

        Solve for a pointwise Lagrange multiplier in Fourier space so unequal
        kappa_i and L_i still satisfy sum_i phi_i^(n+1) = 1 at each grid cell.
        For each Fourier mode q:
             a_i phi_i(new) = b_i + dt L_i lambda(new)
             a_i = 1 + dt L_i kappa_i q^2
             b_i = fft(phi_i(old) - dt L_i df/dphi_i(old))
        """
        df = self.df_dphi(phi)
        b_hat = np.fft.fft2(phi - dt * self.mobility * df, axes=(-2, -1))
        a = 1.0 + dt * self.mobility * self.kappa * self.laplace_eigenvalue
        target_sum_hat = np.zeros((1, NY, NX), dtype=np.complex128)
        target_sum_hat[0, 0, 0] = NX * NY  # FFT of the constant field 1
        lam_hat = (target_sum_hat - (b_hat / a).sum(axis=0, keepdims=True)) / (
            dt * (self.mobility / a).sum(axis=0, keepdims=True))
        new_hat = (b_hat + dt * self.mobility * lam_hat) / a
        return np.fft.ifft2(new_hat, axes=(-2, -1)).real

    def energy(self, phi):
        p2 = phi * phi
        # Sum over i<j: ( (sum phi_i^2)^2 - sum phi_i^4 ) / 2
        bulk = np.sum(self.g * p2 * (3 - 2*phi))
        overlap = 0.5 * W_OVERLAP * np.sum(p2.sum(axis=0)**2 - (p2*p2).sum(axis=0))
        d_dx = (np.roll(phi, -1, axis=2) - phi) / self.dx
        d_dy = (np.roll(phi, -1, axis=1) - phi) / self.dy
        gradient = 0.5 * np.sum(self.kappa * (d_dx*d_dx + d_dy*d_dy))
        area = self.dx * self.dy
        return (bulk + overlap)*area, gradient*area


def initial_condition(model):
    y, x = np.meshgrid((np.arange(NY) + 0.5) * model.dy,
                       (np.arange(NX) + 0.5) * model.dx, indexing="ij")
    phi = np.zeros(model.shape, dtype=np.float64)

    rng = np.random.default_rng(RANDOM_SEED)

    if SCENARIO == "coarsening":
        # Three differently oriented solid grains that fill the whole domain.
        # Each is a different phase-field variable, not a different chemistry.
        seeds = [(0.19, 0.25), (0.78, 0.25), (0.47, 0.80)]
        distances = []
        for sx, sy in seeds:
            sx += rng.uniform(-0.008, 0.008)
            sy += rng.uniform(-0.008, 0.008)
            dx = np.abs(x - sx * LENGTH_X)
            dy = np.abs(y - sy * LENGTH_Y)
            dx, dy = np.minimum(dx, LENGTH_X-dx), np.minimum(dy, LENGTH_Y-dy)
            distances.append(np.sqrt(dx*dx + dy*dy))
        distance = np.asarray(distances)
        # Smooth nearest-seed (Voronoi-like) phase partition.
        logits = -distance / INITIAL_INTERFACE_WIDTH
        logits -= logits.max(axis=0, keepdims=True)
        exps = np.exp(logits)
        phi[:3] = exps / exps.sum(axis=0, keepdims=True)
        phi[3] = 0.0
    else:
        # Circular solid nuclei in a liquid matrix. Coordinates are fractions
        # of the fixed physical box dimensions; radii scale with box size.
        seeds = [
            (0.23, 0.27, 0.105, 0),
            (0.70, 0.28, 0.115, 1),
            (0.46, 0.72, 0.130, 2),
            (0.81, 0.76, 0.085, 0),
        ]
        unit = min(LENGTH_X, LENGTH_Y)
        for sx, sy, radius, phase in seeds:
            sx += rng.uniform(-0.008, 0.008)
            sy += rng.uniform(-0.008, 0.008)
            radius += rng.uniform(-0.003, 0.003)
            dx = np.abs(x - sx * LENGTH_X)
            dy = np.abs(y - sy * LENGTH_Y)
            dx, dy = np.minimum(dx, LENGTH_X-dx), np.minimum(dy, LENGTH_Y-dy)
            dist = np.sqrt(dx*dx + dy*dy)
            mask = 0.5 * (1.0 - np.tanh((dist - radius*unit) / INITIAL_INTERFACE_WIDTH))
            phi[phase] += mask
        solids = phi[:3].sum(axis=0)
        # Only needed if multiple diffuse profiles happen to overlap.
        phi[:3] /= np.maximum(1.0, solids)[None, :, :]
        phi[3] = 1.0 - phi[:3].sum(axis=0)

    return phi


def project_if_needed(phi):
    """Project onto {phi_i >= 0, sum_i phi_i = 1}, only when necessary.

    Returns corrected fields, fraction of affected pixels, and max overshoot.
    This enforces nonnegative phase fractions at their obstacle boundaries;
    a finite-step correction is not the original smooth unconstrained PDE.
    Very large overshoots abort.
    """
    if not np.all(np.isfinite(phi)):
        raise RuntimeError("Nonfinite phase-field values: reduce DT or change solver.")
    sum_error = np.abs(phi.sum(axis=0) - 1.0)
    needs_fix = ((phi < -1e-10) | (phi > 1.0 + 1e-10)).any(axis=0) | (sum_error > 1e-10)
    if not needs_fix.any():
        return phi, 0.0, 0.0
    max_violation = max(float(np.max(-phi)), float(np.max(phi - 1.0)))
    if max_violation > 0.15:
        raise RuntimeError(
            f"Fields overshot [0,1] by {max_violation:.3g}; likely an unstable "
            "time step. Reduce DT (or enable AUTO_LIMIT_DT)."
        )
    # Euclidean projection onto simplex, pixel by pixel (vectorized).
    v = phi.reshape(4, -1)
    u = np.sort(v, axis=0)[::-1]
    cssv = np.cumsum(u, axis=0) - 1.0
    rank = np.arange(1, 5)[:, None]
    rho = np.sum(u - cssv/rank > 0, axis=0) - 1
    theta = cssv[rho, np.arange(v.shape[1])] / (rho + 1)
    projected = np.maximum(v - theta[None, :], 0.0).reshape(phi.shape)
    return projected, float(needs_fix.mean()), max(0.0, max_violation)


def effective_timestep(model):
    dt = DT
    if TIME_INTEGRATOR in ("euler", "rk4"):
        diffusion_scale = float(np.max(model.mobility * model.kappa))
        # Conservative explicit diffusion guideline on a 2D FD mesh:
        # dt <= 1 / (2 D [1/dx^2 + 1/dy^2]), with safety factor 0.72.
        # Coupling through the constraint modifies the exact linearized operator,
        # so this is a guideline, not a guarantee of energy stability.
        # This is a conservative diffusion guideline, not a guarantee for the
        # coupled nonlinear reaction term or for arbitrarily large DT.
        if diffusion_scale > 0:
            diffusion_dt = 0.72 / (
                2.0 * diffusion_scale * (1/model.dx**2 + 1/model.dy**2))
            if AUTO_LIMIT_DT and dt > diffusion_dt:
                print(f"[timestep] Reducing requested DT {dt:.5g} to {diffusion_dt:.5g} "
                      "for an explicit-solver diffusion stability guideline.")
                dt = diffusion_dt
            elif not AUTO_LIMIT_DT and dt > diffusion_dt:
                warnings.warn(f"DT={dt:g} exceeds conservative diffusion limit "
                              f"{diffusion_dt:g}. Results may become unstable.")
    nsteps = max(1, math.ceil(FINAL_TIME / dt))
    dt_eff = FINAL_TIME / nsteps
    return nsteps, dt_eff


def draw_rgb(phi):
    return np.clip(np.einsum("iyx,ic->yxc", phi, PHASE_COLORS), 0, 1)


def choose_video_writer():
    if VIDEO_FORMAT == "auto":
        fmt = "mp4" if shutil.which("ffmpeg") else "gif"
    else:
        fmt = VIDEO_FORMAT
    if fmt == "mp4" and not shutil.which("ffmpeg"):
        raise RuntimeError("MP4 needs ffmpeg. Install it or use VIDEO_FORMAT='gif'.")
    return fmt, FFMpegWriter(fps=VIDEO_FPS, bitrate=1500) if fmt == "mp4" else PillowWriter(fps=VIDEO_FPS)


def plot_animation(frames, frame_times, records, outdir):
    fmt, writer = choose_video_writer()
    path = outdir / f"phase_field_evolution.{fmt}"
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 7.3), constrained_layout=True)
    ax_map, ax_cross, ax_energy, ax_fraction = axs.flat
    fig.suptitle(f"2D FDM phase-field | {SCENARIO} | {TIME_INTEGRATOR} | {NX} x {NY}", fontsize=14)

    image = ax_map.imshow(draw_rgb(frames[0]), origin="lower", interpolation="nearest",
                          extent=[0, LENGTH_X, 0, LENGTH_Y], vmin=0, vmax=1)
    cross_y_index = int(round(0.29 * (NY-1)))
    cross_y = (cross_y_index + .5) * (LENGTH_Y/NY)
    ax_map.axhline(cross_y, color="black", linewidth=1.3, linestyle="--", alpha=0.8)
    ax_map.set_title("2D phase morphology")
    ax_map.set_xlabel("x (dimensionless)")
    ax_map.set_ylabel("y (dimensionless)")
    ax_map.legend(handles=[Patch(facecolor=PHASE_COLORS[i], label=PHASE_NAMES[i])
                           for i in range(4)], loc="upper center", ncol=2,
                  fontsize=7, framealpha=0.9)
    clock = ax_map.text(0.02, 0.03, "", transform=ax_map.transAxes,
                        bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"), fontsize=10)

    xs = (np.arange(NX) + .5) * (LENGTH_X/NX)
    profile_lines = []
    for i in range(4):
        (line,) = ax_cross.plot(xs, frames[0][i, cross_y_index], color=LINE_COLORS[i],
                                linewidth=2, label=PHASE_NAMES[i])
        profile_lines.append(line)
    ax_cross.set(xlim=(0, LENGTH_X), ylim=(-0.03, 1.05),
                 xlabel="x along dashed line", ylabel="Local phase-field value")
    ax_cross.set_title("Phase-field profiles along horizontal line")
    ax_cross.legend(fontsize=7, ncol=2, loc="upper right")
    ax_cross.grid(alpha=.18)

    rt = np.asarray([r["time"] for r in records])
    eb = np.asarray([r["bulk_energy"] for r in records])
    eg = np.asarray([r["gradient_energy"] for r in records])
    et = eb + eg
    lo = float(min(eb.min(), eg.min(), et.min()))
    hi = float(max(eb.max(), eg.max(), et.max()))
    pad = max(0.03*(hi-lo), 1.0)
    ax_energy.set(xlim=(0, FINAL_TIME), ylim=(lo-pad, hi+pad),
                  xlabel="Simulation time", ylabel="Total energy (dimensionless)")
    ax_energy.set_title("Energy decreases as fields evolve")
    energy_lines = []
    for color, label in (("#27272a", "F total"), ("#9a6a3a", "F local"),
                         ("#387b9b", "F gradient")):
        (line,) = ax_energy.plot([], [], label=label, color=color, linewidth=1.9)
        energy_lines.append(line)
    ax_energy.legend(fontsize=8)
    ax_energy.grid(alpha=.18)

    ax_fraction.set(xlim=(0, FINAL_TIME), ylim=(0, 1),
                    xlabel="Simulation time", ylabel="Mean phase fraction")
    ax_fraction.set_title("Global phase fractions")
    fraction_lines = []
    for i in range(4):
        (line,) = ax_fraction.plot([], [], label=PHASE_NAMES[i], color=LINE_COLORS[i], linewidth=2)
        fraction_lines.append(line)
    ax_fraction.legend(fontsize=8, ncol=2)
    ax_fraction.grid(alpha=.18)
    fraction_values = np.array([[r[f"fraction_{i}"] for i in range(4)] for r in records])

    def update(frame_index):
        phi = frames[frame_index]
        t = frame_times[frame_index]
        image.set_data(draw_rgb(phi))
        clock.set_text(f"t = {t:.2f}")
        for i, line in enumerate(profile_lines):
            line.set_ydata(phi[i, cross_y_index])
        pos = np.searchsorted(rt, t, side="right")
        for line, values in zip(energy_lines, (et, eb, eg)):
            line.set_data(rt[:pos], values[:pos])
        for i, line in enumerate(fraction_lines):
            line.set_data(rt[:pos], fraction_values[:pos, i])
        return [image, clock, *profile_lines, *energy_lines, *fraction_lines]

    try:
        animation = FuncAnimation(fig, update, frames=len(frames), interval=1000/VIDEO_FPS,
                                  blit=False, repeat=False)
        print(f"[video] Saving {len(frames)} frames to {path.name} ...")
        animation.save(str(path), writer=writer, dpi=90)
    finally:
        plt.close(fig)
    return path


def plot_mesh_discretization(initial, model, outdir):
    """Show a fixed physical zoom with EVERY FDM cell boundary visible.

    This image makes NX, NY, dx, dy tangible. Larger NX and NY genuinely
    refine the same physical domain rather than expanding the box.
    """
    fig, axs = plt.subplots(1, 2, figsize=(11.2, 4.8), layout="constrained")
    extent = [0, LENGTH_X, 0, LENGTH_Y]
    image = draw_rgb(initial)
    for ax in axs:
        ax.imshow(image, origin="lower", interpolation="nearest", extent=extent)
        ax.set(xlabel="x (dimensionless)", ylabel="y (dimensionless)")
        ax.set_aspect("equal")
    axs[0].set_title(f"Whole domain: {NX} x {NY} FDM cells")
    # Fixed real-space rectangle placed across the interface of the first
    # nucleus. This same physical region is shown for every mesh size.
    xlo, xhi = 0.265*LENGTH_X, 0.395*LENGTH_X
    ylo, yhi = 0.19*LENGTH_Y, 0.33*LENGTH_Y
    axs[0].add_patch(plt.Rectangle((xlo, ylo), xhi-xlo, yhi-ylo,
                                   edgecolor="black", facecolor="none", linewidth=1.5))
    axs[1].set(xlim=(xlo, xhi), ylim=(ylo, yhi))
    # Grid boundaries at j*dx and k*dy (cell centers are (j+0.5)*dx, etc.)
    xfirst, xlast = math.ceil(xlo/model.dx), math.floor(xhi/model.dx)
    yfirst, ylast = math.ceil(ylo/model.dy), math.floor(yhi/model.dy)
    for j in range(xfirst, xlast+1):
        axs[1].axvline(j*model.dx, color="black", lw=.55, alpha=.55)
    for k in range(yfirst, ylast+1):
        axs[1].axhline(k*model.dy, color="black", lw=.55, alpha=.55)
    axs[1].set_title(f"Same-area zoom with mesh lines\n"
                     f"dx={model.dx:.3g}, dy={model.dy:.3g}")
    fig.suptitle("Finite-difference spatial discretization (periodic domain)")
    fig.savefig(outdir/"fdm_spatial_grid.png", dpi=150)
    plt.close(fig)


def plot_static(initial, final, records, outdir):
    for label, frame in (("initial", initial), ("final", final)):
        fig, ax = plt.subplots(figsize=(6.2, 5.4), layout="constrained")
        ax.imshow(draw_rgb(frame), origin="lower", extent=[0,LENGTH_X,0,LENGTH_Y])
        ax.set(xlabel="x (dimensionless)", ylabel="y (dimensionless)",
               title=f"{label.title()} morphology | {SCENARIO} | {NX}x{NY}")
        ax.legend(handles=[Patch(facecolor=PHASE_COLORS[i], label=PHASE_NAMES[i])
                           for i in range(4)], fontsize=8, loc="upper right")
        fig.savefig(outdir / f"{label}_microstructure.png", dpi=180)
        plt.close(fig)

    t = np.asarray([r["time"] for r in records])
    bulk = np.asarray([r["bulk_energy"] for r in records])
    grad = np.asarray([r["gradient_energy"] for r in records])
    fig, axs = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    axs[0,0].plot(t, bulk+grad, color="black", label="F total", lw=2)
    axs[0,0].plot(t, bulk, color="#9a6a3a", label="F local", lw=1.5)
    axs[0,0].plot(t, grad, color="#387b9b", label="F gradient", lw=1.5)
    axs[0,0].set(xlabel="Time", ylabel="Energy", title="Free energy")
    axs[0,0].legend(fontsize=8)
    for i, name in enumerate(PHASE_NAMES):
        axs[0,1].plot(t, [r[f"fraction_{i}"] for r in records],
                      color=LINE_COLORS[i], label=name)
    axs[0,1].set(xlabel="Time", ylabel="Mean phase fraction", title="Phase fractions", ylim=(0,1))
    axs[0,1].legend(fontsize=8)
    axs[1,0].semilogy(t, np.maximum([r["sum_constraint_error"] for r in records], 1e-16))
    axs[1,0].set(xlabel="Time", ylabel="Maximum |sum(phi)-1|", title="Constraint accuracy")
    axs[1,1].plot(t, [r["min_phi"] for r in records], label="minimum phi")
    axs[1,1].plot(t, [r["max_phi"] for r in records], label="maximum phi")
    axs[1,1].set(xlabel="Time", ylabel="Field value", title="Field bounds")
    axs[1,1].legend(fontsize=8)
    for ax in axs.flat:
        ax.grid(alpha=.2)
    fig.suptitle(f"FDM simulation diagnostics | {SCENARIO} | {TIME_INTEGRATOR}")
    fig.savefig(outdir / "diagnostics.png", dpi=170)
    plt.close(fig)


def main():
    check_settings()
    model = Model()
    nsteps, actual_dt = effective_timestep(model)
    outdir = Path(__file__).resolve().parent / OUTPUT_DIRECTORY / (
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_{SCENARIO}_fdm_{TIME_INTEGRATOR}_{NX}x{NY}"
    )
    outdir.mkdir(parents=True, exist_ok=False)
    config = settings_dict()
    config.update({"actual_dt": actual_dt, "nsteps": nsteps,
                   "dx": model.dx, "dy": model.dy,
                   "number_of_grid_cells": NX*NY,
                   "interface_width_in_x_cells": INITIAL_INTERFACE_WIDTH / model.dx,
                   "interface_width_in_y_cells": INITIAL_INTERFACE_WIDTH / model.dy,
                   "G_USED": model.g[:,0,0].tolist(),
                   "BOUNDARY_CONDITIONS": "periodic in x and y"})
    (outdir/"settings.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    phi = initial_condition(model)
    phi_initial = phi.copy()
    frame_steps = set(np.linspace(0, nsteps, min(ANIMATION_FRAMES, nsteps+1),
                                  dtype=int).tolist())
    frames = [phi.copy()]
    frame_times = [0.0]
    records = []
    projection_events = 0
    maximum_projected_fraction = 0.0
    maximum_projection_overshoot = 0.0
    print(f"[start] Spatial method={SPATIAL_METHOD.upper()} (5-point Laplacian)")
    print(f"[start] Scenario={SCENARIO}, time integrator={TIME_INTEGRATOR}")
    print(f"[start] Grid={NX}x{NY} cells; fixed domain={LENGTH_X:g}x{LENGTH_Y:g}; "
          f"dx={model.dx:g}, dy={model.dy:g}")
    if min(INITIAL_INTERFACE_WIDTH / model.dx,
           INITIAL_INTERFACE_WIDTH / model.dy) < 1.5:
        print("[mesh warning] The initialized diffuse-interface width is "
              "resolved by fewer than 1.5 cells in at least one direction; "
              "try a finer grid for reliable interfaces.")
    print(f"[start] Steps={nsteps}, effective DT={actual_dt:g}, "
          f"final time={FINAL_TIME:g}")
    print(f"[start] Output directory: {outdir}")
    start_clock = time.perf_counter()

    def record(step, current):
        local, gradient = model.energy(current)
        row = {"step": step, "time": step * actual_dt,
               "bulk_energy": local, "gradient_energy": gradient,
               "total_energy": local+gradient,
               "sum_constraint_error": float(np.max(np.abs(current.sum(axis=0)-1.0))),
               "min_phi": float(current.min()), "max_phi": float(current.max())}
        for i in range(4):
            row[f"fraction_{i}"] = float(np.mean(current[i]))
        records.append(row)

    record(0, phi)
    stepper = {"euler": model.step_euler, "rk4": model.step_rk4,
               "semi_implicit_fft": model.step_semi_implicit_fft}[TIME_INTEGRATOR]
    for step in range(1, nsteps+1):
        new_phi = stepper(phi, actual_dt)
        try:
            phi, projected_fraction, overshoot = project_if_needed(new_phi)
        except RuntimeError as exc:
            (outdir/"FAILED_RUN.txt").write_text(f"Step {step}: {exc}\n")
            raise RuntimeError(f"Step {step}: {exc} | output directory: {outdir}") from exc
        if projected_fraction > 0:
            projection_events += 1
            maximum_projected_fraction = max(maximum_projected_fraction, projected_fraction)
            maximum_projection_overshoot = max(maximum_projection_overshoot, overshoot)
        if step % DIAGNOSTICS_EVERY == 0 or step == nsteps:
            record(step, phi)
        if step in frame_steps:
            frames.append(phi.copy())
            frame_times.append(step * actual_dt)
        if step % max(1, nsteps // 5) == 0:
            print(f"[running] {step}/{nsteps} steps ({100*step/nsteps:.0f}%)")

    elapsed = time.perf_counter() - start_clock
    plot_static(phi_initial, phi, records, outdir)
    if SAVE_MESH_PLOT:
        plot_mesh_discretization(phi_initial, model, outdir)
    np.savez_compressed(outdir/"fields_and_frames.npz", initial=phi_initial, final=phi,
                        frames=np.asarray(frames), frame_times=np.asarray(frame_times),
                        phase_names=np.array(PHASE_NAMES))
    with (outdir/"history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    video_path = None
    if SAVE_ANIMATION:
        video_path = plot_animation(frames, frame_times, records, outdir)

    energies = np.array([r["total_energy"] for r in records])
    increments = np.diff(energies)
    tolerance = max(1e-8, 1e-9 * max(1.0, abs(energies[0])))
    energy_increase_count = int((increments > tolerance).sum())
    sum_error = max(r["sum_constraint_error"] for r in records)
    lines = [
        f"Scenario: {SCENARIO}", f"Spatial method: {SPATIAL_METHOD.upper()} (5-point finite differences)",
        f"Time integrator: {TIME_INTEGRATOR}", f"Grid: {NX} x {NY}",
        f"Total mesh cells: {NX*NY}", f"Physical domain: {LENGTH_X:g} x {LENGTH_Y:g}",
        f"dx, dy: {model.dx:.8g}, {model.dy:.8g}",
        f"Time steps: {nsteps}; dt: {actual_dt:.8g}; final time: {FINAL_TIME}",
        f"Initial total energy: {energies[0]:.8g}",
        f"Final total energy: {energies[-1]:.8g}",
        f"Sampled intervals with energy increase: {energy_increase_count}",
        f"Maximum sum(phi)=1 error: {sum_error:.4e}",
        f"Steps requiring bounds/simplex correction: {projection_events}",
        f"Maximum affected pixel fraction per step: {maximum_projected_fraction:.4%}",
        f"Maximum simplex-bound overshoot before projection: {maximum_projection_overshoot:.5g}",
        f"Simulation runtime (video not included): {elapsed:.2f} seconds",
        f"Animation: {video_path.name if video_path else '(disabled)'}",
        "NOTE: Projected corrections enforce nonnegative phase fractions.",
        "Even small corrections can occur when phases are absent locally.",
        "Large overshoots (>0.02) or increases in energy warrant reducing DT.",
        "All numerical units are arbitrary/dimensionless; periodic boundaries.",
    ]
    (outdir/"summary.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("\n".join(f"[summary] {line}" for line in lines[:18]))
    print(f"[done] Files saved in {outdir}")
    return outdir


if __name__ == "__main__":
    main()
