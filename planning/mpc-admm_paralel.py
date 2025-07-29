import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
import time
import concurrent.futures
import multiprocessing
import scenarios

scenario = scenarios.USHAPE_5

ell = scenarios.ell
savefile =  scenario["savefile"]
obstacles =  scenario["obstacles"]
x_ref =  scenario["x_ref"]

N_drones =  scenario["N_drones"]
K_admm =  scenario["K_admm"]
T_sim =  scenario["T_sim"]

# Parameters
horizon = 6
dim = 3
rho = 15.0
gamma = 50.0
dt = 0.1
u_max = 1.0
eps_pri = 1e-3
eps_dual = 1e-3
tau = 1.0
safety_margin = 0.2

def dynamics(x, u):
    return x + dt * u

def solve_drone_optimization(i, x_current_i, x_ref_i, x_global_i, alpha_i, x_pred, z, lambd):
    x = cp.Variable((horizon + 1, dim))
    u = cp.Variable((horizon, dim))
    cost = 0
    constraints = [x[0] == x_current_i]

    for t_h in range(horizon):
        cost += cp.sum_squares(x[t_h] - x_ref_i) + 0.1 * cp.sum_squares(u[t_h])
        cost += (rho / 2) * cp.sum_squares(x[t_h] - x_global_i[t_h] + alpha_i[t_h])
        constraints += [
            x[t_h + 1] == x[t_h] + dt * u[t_h],
            cp.norm(u[t_h], 'inf') <= u_max
        ]

        if i < N_drones - 1:
            j = i + 1
            cost += (rho / 2) * cp.sum_squares(x[t_h] -
                                                x_pred[j, t_h] - z[i, t_h] + lambd[i, t_h]/rho)
            # cost += cp.sum_squares(x_pred[i + 1, t_h] - x_ref[i + 1])

        if i > 0:
            j = i - 1
            cost += (rho / 2) * cp.sum_squares(x[t_h] -
                                                x_pred[j, t_h] + z[j, t_h] - lambd[j, t_h] / rho)
            # cost += cp.sum_squares(x_pred[i - 1, t_h] - x_ref[i - 1])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return i, x.value, u.value, prob.value

def main():
    x_current = scenario["x_current"]

    x_hist = [x_current.copy()]
    u_hist = []

    # Initial prediction
    x_pred_next = np.repeat(x_current[:, None, :], horizon + 1, axis=1)
    u_pred_next = np.zeros((N_drones, horizon, dim))

    residual_log = []  # List of lists: one per timestep, containing residuals per ADMM iteration
    costs_log = []

    with concurrent.futures.ProcessPoolExecutor() as executor:  # moved outside the k loop
        for t in range(T_sim):
            print("Loop iteration:", t)
            x_pred = x_pred_next.copy()
            u_pred = u_pred_next.copy()
            x_global = x_pred.copy()
            alpha = np.zeros_like(x_pred)
            z = np.zeros((N_drones - 1, horizon, dim))
            lambd = np.zeros_like(z)

            for k in range(K_admm):
                x_prev = x_pred.copy()
                iteration_cost = 0

                args_list = [
                    (i, x_current[i], x_ref[i], x_global[i], alpha[i], x_pred, z, lambd)
                    for i in range(N_drones)
                ]
                futures = [executor.submit(solve_drone_optimization, *args) for args in args_list]
                for future in concurrent.futures.as_completed(futures):
                    i, x_val, u_val, cost_val = future.result()
                    x_pred[i] = x_val
                    u_pred[i] = u_val

                    iteration_cost += cost_val # accumulate cost over drones

                # After all drones are solved for this ADMM iteration:
                if k == 0:
                    costs_log.append([])  # start cost log for this timestep
                costs_log[-1].append(iteration_cost)

                # Projection step (Obstacle avoidance)
                for i in range(N_drones):
                    for t_h in range(horizon):
                        xg = x_pred[i, t_h] + alpha[i, t_h]
                        for obs in obstacles:
                            p = obs["center"]
                            r = obs["radius"] + safety_margin
                            vec = xg - p
                            dist = np.linalg.norm(vec)
                            if dist < r:
                                xg = p + r * vec / (dist + 1e-6)
                        x_global[i, t_h] = xg

                # Dual update for local-global constraint
                alpha += x_pred - x_global

                # Enforce fixed spacing: project z to be ell-distance vector
                for i in range(N_drones - 1):
                    for t_h in range(horizon):
                        diff = x_pred[i, t_h] - x_pred[i + 1, t_h]
                        norm = np.linalg.norm(diff)
                        z[i, t_h] = ell * diff / (norm + 1e-6)

                # ADMM dual update
                for i in range(N_drones - 1):
                    for t_h in range(horizon):
                        lambd[i, t_h] += tau * (x_pred[i, t_h] -
                                        x_pred[i + 1, t_h] - z[i, t_h])

                # Residual check
                r_pri = np.linalg.norm(x_pred - x_global)
                r_dual = np.linalg.norm(x_pred - x_prev)

                # Log the residuals for this ADMM iteration
                if k == 0:
                    residual_log.append([])  # Start new time step
                residual_log[-1].append((r_dual, r_pri))

                if r_pri < eps_pri and r_dual < eps_dual:
                    print(f"[ADMM] Converged at iteration {k+1} at time step {t}")
                    break

            x_pred_next[:, :-1] = x_pred[:, 1:]
            x_pred_next[:, -1] = x_pred[:, -1]
            u_pred_next[:, :-1] = u_pred[:, 1:]
            u_pred_next[:, -1] = 0

            # Apply control and simulate
            u_apply = np.array([u_pred[i, 0] for i in range(N_drones)])
            x_next = np.array([dynamics(x_current[i], u_apply[i])
                            for i in range(N_drones)])
            x_hist.append(x_next.copy())
            u_hist.append(u_apply.copy())
            x_current = x_next.copy()
            np.save(f"saves/{savefile}.npy", np.array(x_hist))

    x_hist = np.array(x_hist)

    # ---------------------
    # Plot residuals over ADMM iterations for each timestep
    # ---------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    for t, residuals in enumerate(residual_log):
        dual_res = [r[0] for r in residuals]
        primal_res = [r[1] for r in residuals]
        ax[0].plot(dual_res, label=f"t={t}")
        ax[1].plot(primal_res, label=f"t={t}")


    ax[0].set_title("Dual Residuals")
    ax[0].set_xlabel("ADMM Iteration")
    ax[0].set_ylabel("Residual")
    ax[0].set_yscale("log")
    ax[0].grid(True)

    ax[1].set_title("Primal Residuals")
    ax[1].set_xlabel("ADMM Iteration")
    ax[1].set_ylabel("Residual")
    ax[1].set_yscale("log")
    ax[1].grid(True)

    plt.tight_layout()
    plt.savefig(f"saves/{savefile}_admm_residuals.png")
    plt.show()

    # ---------------------
    # Plot costs over ADMM iterations for each timestep
    # ---------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))

    for t, cost_list in enumerate(costs_log):
        ax2.plot(cost_list, label=f"t={t}")

    ax2.set_title("ADMM Cost Across Iterations per Timestep")
    ax2.set_xlabel("ADMM Iteration")
    ax2.set_ylabel("Objective Cost")
    ax2.grid(True)
    # ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{savefile}_admm_costs.png")
    plt.show()


    # ---------------------
    # 3D Animation
    # ---------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    def update(frame):
        ax.cla()
        ax.set_title(f"3D Drone Chain (Step {frame})")
        ax.set_xlim(-1, 5)
        ax.set_ylim(-1, 5)
        ax.set_zlim(-1, 2)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Plot drone paths
        for i in range(N_drones):
            ax.plot(x_hist[:frame+1, i, 0],
                    x_hist[:frame+1, i, 1],
                    x_hist[:frame+1, i, 2])
            ax.scatter(*x_hist[frame, i], s=40)

        # Plot bars and distances
        for i in range(N_drones - 1):
            p1, p2 = x_hist[frame, i], x_hist[frame, i + 1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', lw=2)
            midpoint = (p1 + p2) / 2
            dist = np.linalg.norm(p1 - p2)
            ax.text(midpoint[0], midpoint[1], midpoint[2] +
                    0.1, f"{dist:.2f}", color='red', fontsize=8)

        # Plot spherical obstacles
        for obs in obstacles:
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            cx, cy, cz = obs["center"]
            r = obs["radius"]
            xs = r * np.cos(u) * np.sin(v) + cx
            ys = r * np.sin(u) * np.sin(v) + cy
            zs = r * np.cos(v) + cz
            ax.plot_surface(xs, ys, zs, color='gray', alpha=0.3)


    ani = animation.FuncAnimation(fig, update, frames=T_sim, interval=100)
    ani.save(f"saves/{savefile}_admm_output.gif",
            writer="pillow", fps=int(1/dt))
    plt.show()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
