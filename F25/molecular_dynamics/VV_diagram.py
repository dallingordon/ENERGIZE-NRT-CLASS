import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def harmonic_acceleration(pos):
    return -0.5 * pos


def velocity_verlet_path(dt, n_steps, x0, v0, a_func):
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)

    for _ in range(n_steps):
        a = a_func(x)
        x_new = x + v * dt + 0.5 * a * dt ** 2
        a_new = a_func(x_new)
        v_new = v + 0.5 * (a + a_new) * dt
        x, v = x_new, v_new

    return x


def leapfrog_path(dt, n_steps, x0, v0, a_func):
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    a = a_func(x)
    v_half = v - 0.5 * dt * a

    for _ in range(n_steps):
        x_new = x + v_half * dt
        a_new = a_func(x_new)
        v_half_new = v_half + a_new * dt
        x, v_half = x_new, v_half_new

    return x


def velocity_verlet_full_path(dt, n_steps, x0, v0, a_func):
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    positions = [x.copy()]
    for _ in range(n_steps):
        a = a_func(x)
        x_new = x + v * dt + 0.5 * a * dt ** 2
        a_new = a_func(x_new)
        v_new = v + 0.5 * (a + a_new) * dt
        positions.append(x_new.copy())
        x, v = x_new, v_new
    return np.array(positions)


def leapfrog_full_path(dt, n_steps, x0, v0, a_func):
    x = np.array(x0, dtype=float)
    v = np.array(v0, dtype=float)
    a = a_func(x)
    v_half = v - 0.5 * dt * a
    positions = [x.copy()]
    for _ in range(n_steps):
        x_new = x + v_half * dt
        a_new = a_func(x_new)
        v_half_new = v_half + a_new * dt
        positions.append(x_new.copy())
        x, v_half = x_new, v_half_new
    return np.array(positions)


def optimize_vv_init_vel(v0_guess, dt, n_steps, x0, x_target, a_func):
    final_pos = velocity_verlet_path(dt, n_steps, x0, v0_guess, a_func)
    return np.linalg.norm(final_pos - x_target)


def optimize_lf_init_vel(v0_guess, dt, n_steps, x0, x_target, a_func):
    final_pos = leapfrog_path(dt, n_steps, x0, v0_guess, a_func)
    return np.linalg.norm(final_pos - x_target)


def main():
    # Initial conditions for the harmonic oscillator
    x0 = np.array([-1.5, 0.0])
    v0_true = np.array([0.0, 1.0])

    # Generate "true" path via very fine Velocity-Verlet integration
    true_dt = 0.001
    true_steps = 6000  # total sim time ~6 units

    x_true_full = velocity_verlet_full_path(true_dt, true_steps, x0, v0_true, harmonic_acceleration)
    x_true = x_true_full[:, 0]
    y_true = x_true_full[:, 1]
    x_target = x_true_full[-1]

    # Coarser timestep and fewer steps for integrators
    dt = .1
    n_steps = 100

    v0_guess = np.array([-10, 10.0])  # initial velocity guess

    # Optimize initial velocity to hit endpoint for Velocity-Verlet
    res_vv = minimize(optimize_vv_init_vel, v0_guess, args=(dt, n_steps, x0, x_target, harmonic_acceleration), tol=1e-8)
    v0_vv_opt = res_vv.x

    # Optimize initial velocity to hit endpoint for Leapfrog
    res_lf = minimize(optimize_lf_init_vel, v0_guess, args=(dt, n_steps, x0, x_target, harmonic_acceleration), tol=1e-8)
    v0_lf_opt = res_lf.x

    # Generate integrator paths
    vv_path = velocity_verlet_full_path(dt, n_steps, x0, v0_vv_opt, harmonic_acceleration)
    lf_path = leapfrog_full_path(dt, n_steps, x0, v0_lf_opt, harmonic_acceleration)

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    # True path plot
    axs[0].plot(x_true, y_true, lw=2, color='tab:blue', label='True Path')
    axs[0].scatter(x0[0], x0[1], s=120, facecolors='none', edgecolors='r', linewidth=3, label='Start')
    axs[0].scatter(x_true[-1], y_true[-1], s=180, marker='>', color='g', label='End')
    axs[0].set_title('True path')
    axs[0].set_xlabel('Position X')
    axs[0].set_ylabel('Position Y')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Leapfrog plot
    axs[1].plot(x_true, y_true, linestyle='--', color='lightgrey', label='True path (ref)')
    axs[1].plot(lf_path[:, 0], lf_path[:, 1], marker='o', color='k', lw=2, label='Leapfrog Steps')
    axs[1].scatter(lf_path[0, 0], lf_path[0, 1], s=120, facecolors='none', edgecolors='r', linewidth=3, label='Start')
    axs[1].scatter(lf_path[-1, 0], lf_path[-1, 1], s=180, marker='>', color='g', label='End')
    axs[1].set_title('Leapfrog Integrator')
    axs[1].set_xlabel('Position X')
    axs[1].set_ylabel('Position Y')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # Velocity-Verlet plot
    axs[2].plot(x_true, y_true, linestyle='--', color='lightgrey', label='True path (ref)')
    axs[2].plot(vv_path[:, 0], vv_path[:, 1], marker='o', color='k', lw=2, label='Velocity-Verlet Steps')
    axs[2].scatter(vv_path[0, 0], vv_path[0, 1], s=120, facecolors='none', edgecolors='r', linewidth=3, label='Start')
    axs[2].scatter(vv_path[-1, 0], vv_path[-1, 1], s=180, marker='>', color='g', label='End')
    axs[2].set_title('Velocity-Verlet Integrator')
    axs[2].set_xlabel('Position X')
    axs[2].set_ylabel('Position Y')
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    for ax in axs:
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()