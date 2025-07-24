import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp
import time
import concurrent.futures
import multiprocessing

# Parameters
N_drones = 9
horizon = 6
dim = 3
rho = 15.0
gamma = 50.0
ell = 0.35
K_admm = 30
T_sim = 50
dt = 0.1
u_max = 1.0
eps_pri = 1e-3
eps_dual = 1e-3
eps_neighbor = 1e-3
use_neighbor_consensus = True

obstacles = [{"center": np.array([1, 1.5, -0.5]), "radius": 0.8}]
safety_margin = 0.2

def dynamics(x, u):
    return x + dt * u

def solve_drone_optimization(i, x_current_i, x_ref_i, x_global_i, alpha_i, x_pred, z, lambd, w, mu):
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

        if t_h > 0:
            cost += gamma * cp.sum_squares(x[t_h] - x[t_h - 1])

        if i < N_drones - 1:
            j = i + 1
            cost += (rho / 2) * cp.sum_squares(x[t_h] - x_pred[j, t_h] - z[i, t_h] + lambd[i, t_h]/rho)

        if i > 0:
            j = i - 1
            cost += (rho / 2) * cp.sum_squares(x[t_h] - x_pred[j, t_h] + z[j, t_h] - lambd[j, t_h] / rho)

        if use_neighbor_consensus and 0 < i < N_drones - 1:
            j_prev = i - 1
            j_next = i + 1
            second_order = x_pred[j_prev, t_h] - 2 * x[t_h] + x_pred[j_next, t_h]
            cost += (rho / 2) * cp.sum_squares(second_order - w[i, t_h] + mu[i, t_h])

    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return i, x.value, u.value, prob.value

def main():
        # Initial setup
    x_ref = np.array([[i * ell, 0.0, 0.0] for i in range(N_drones)])


    x_current = np.array([[0, i * ell, 0] for i in range(N_drones)])
    x_hist = [x_current.copy()]
    u_hist = []
    x_pred_next = np.repeat(x_current[:, None, :], horizon + 1, axis=1)
    u_pred_next = np.zeros((N_drones, horizon, dim))
    residual_log = []
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
            w = np.zeros((N_drones, horizon, dim))
            mu = np.zeros_like(w)

            for k in range(K_admm):
                x_prev = x_pred.copy()
                iteration_cost = 0

                args_list = [
                    (i, x_current[i], x_ref[i], x_global[i], alpha[i], x_pred, z, lambd, w, mu)
                    for i in range(N_drones)
                ]
                futures = [executor.submit(solve_drone_optimization, *args) for args in args_list]
                for future in concurrent.futures.as_completed(futures):
                    i, x_val, u_val, cost_val = future.result()
                    x_pred[i] = x_val
                    u_pred[i] = u_val
                    iteration_cost += cost_val

                if k == 0:
                    costs_log.append([])
                costs_log[-1].append(iteration_cost)

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

                alpha += x_pred - x_global

                for i in range(N_drones - 1):
                    for t_h in range(horizon):
                        diff = x_pred[i, t_h] - x_pred[i + 1, t_h]
                        norm = np.linalg.norm(diff)
                        z[i, t_h] = ell * diff / (norm + 1e-6)

                for i in range(N_drones - 1):
                    for t_h in range(horizon):
                        lambd[i, t_h] += (x_pred[i, t_h] - x_pred[i + 1, t_h] - z[i, t_h])

                if use_neighbor_consensus:
                    for i in range(1, N_drones - 1):
                        for t_h in range(horizon):
                            diff = x_pred[i - 1, t_h] - 2 * x_pred[i, t_h] + x_pred[i + 1, t_h] + mu[i, t_h]
                            w[i, t_h] = diff

                    for i in range(1, N_drones - 1):
                        for t_h in range(horizon):
                            mu[i, t_h] += (
                                x_pred[i - 1, t_h] - 2 * x_pred[i, t_h] + x_pred[i + 1, t_h] - w[i, t_h]
                            )

                r_pri = np.linalg.norm(x_pred - x_global)
                r_dual = np.linalg.norm(x_pred - x_prev)

                if use_neighbor_consensus:
                    r_second_order = np.linalg.norm([
                        x_pred[i - 1, t_h] - 2 * x_pred[i, t_h] + x_pred[i + 1, t_h] - w[i, t_h]
                        for i in range(1, N_drones - 1)
                        for t_h in range(horizon)
                    ])
                else:
                    r_second_order = 0.0

                if k == 0:
                    residual_log.append([])
                residual_log[-1].append((r_second_order, r_dual, r_pri))

                if r_pri < eps_pri and r_dual < eps_dual and r_second_order < eps_neighbor:
                    print(f"[ADMM] Converged at iteration {k+1} at time step {t}")
                    break

            x_pred_next[:, :-1] = x_pred[:, 1:]
            x_pred_next[:, -1] = x_pred[:, -1]
            u_pred_next[:, :-1] = u_pred[:, 1:]
            u_pred_next[:, -1] = 0

            u_apply = np.array([u_pred[i, 0] for i in range(N_drones)])
            x_next = np.array([dynamics(x_current[i], u_apply[i]) for i in range(N_drones)])
            x_hist.append(x_next.copy())
            u_hist.append(u_apply.copy())
            x_current = x_next.copy()
            np.save("mpc_solution_x.npy", np.array(x_hist))
            np.save("mpc_solution_u.npy", np.array(u_hist))

    x_hist = np.array(x_hist)

    # ---------------------
    # Plot residuals over ADMM iterations for each timestep
    # ---------------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for t, residuals in enumerate(residual_log):
        neighbor_res = [r[0] for r in residuals]
        dual_res = [r[1] for r in residuals]
        primal_res = [r[2] for r in residuals]
        ax[0].plot(neighbor_res, label=f"t={t}")
        ax[1].plot(dual_res, label=f"t={t}")
        ax[2].plot(primal_res, label=f"t={t}")

    ax[0].set_title("Neighbor Consensus Residuals")
    ax[0].set_xlabel("ADMM Iteration")
    ax[0].set_ylabel("Residual")
    ax[0].set_yscale("log")
    ax[0].grid(True)

    ax[1].set_title("Dual Residuals")
    ax[1].set_xlabel("ADMM Iteration")
    ax[1].set_ylabel("Residual")
    ax[1].set_yscale("log")
    ax[1].grid(True)

    ax[2].set_title("Primal Residuals")
    ax[2].set_xlabel("ADMM Iteration")
    ax[2].set_ylabel("Residual")
    ax[2].set_yscale("log")
    ax[2].grid(True)

    plt.tight_layout()
    plt.savefig("admm_residuals_plot.png")
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
    plt.savefig("admm_costs_plot.png")
    plt.show()


    # ---------------------
    # 3D Animation
    # ---------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')


    def update(frame):
        ax.cla()
        ax.set_title(f"3D Drone Chain (Step {frame})")
        ax.set_xlim(-1, 2)
        ax.set_ylim(-1, 2)
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
    ani.save("drone_chain_admm_augmented_obstacles.gif",
            writer="pillow", fps=int(1/dt))
    # plt.show()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
