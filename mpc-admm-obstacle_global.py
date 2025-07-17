import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cvxpy as cp

# Parameters
N_drones = 8
horizon = 6
dim = 3
rho = 15.0
ell = 1.0
K_admm = 10
T_sim = 50
dt = 0.1
u_max = 1.0
eps_pri = 1e-4
eps_dual = 1e-4

# Obstacles
obstacles = [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8}]
safety_margin = 0.2

# Dynamics
def dynamics(x, u):
    return x + dt * u

# Initial setup
x_ref = np.array([[i * ell, 0.0, 0.0] for i in range(N_drones)])
x_current = np.array([[0, i * ell, 0] for i in range(N_drones)])
x_hist = [x_current.copy()]
u_hist = []

# Initial prediction
x_pred_next = np.repeat(x_current[:, None, :], horizon + 1, axis=1)
u_pred_next = np.zeros((N_drones, horizon, dim))

# ADMM-MPC loop
for t in range(T_sim):
    x_pred = x_pred_next.copy()
    u_pred = u_pred_next.copy()
    x_global = x_pred.copy()
    alpha = np.zeros_like(x_pred)
    z = np.zeros((N_drones - 1, horizon, dim))
    lambd = np.zeros_like(z)

    for k in range(K_admm):
        x_prev = x_pred.copy()

        # Local optimization for each drone
        for i in range(N_drones):
            x = cp.Variable((horizon + 1, dim))
            u = cp.Variable((horizon, dim))
            cost = 0
            constraints = [x[0] == x_current[i]]

            for t_h in range(horizon):
                cost += cp.sum_squares(x[t_h] - x_ref[i]) + 0.1 * cp.sum_squares(u[t_h])
                cost += (rho / 2) * cp.sum_squares(x[t_h] - x_global[i, t_h] + alpha[i, t_h])
                constraints += [
                    x[t_h + 1] == x[t_h] + dt * u[t_h],
                    cp.norm(u[t_h], 'inf') <= u_max
                ]

                if i < N_drones - 1:
                    j = i + 1
                    cost += (rho / 2) * cp.sum_squares(x[t_h] - x_pred[j, t_h] - z[i, t_h] + lambd[i, t_h])
                if i > 0:
                    j = i - 1
                    cost += (rho / 2) * cp.sum_squares(x[t_h] - x_pred[j, t_h] + z[j, t_h] - lambd[j, t_h])

            cp.Problem(cp.Minimize(cost), constraints).solve(solver=cp.SCS, verbose=False)
            x_pred[i] = x.value
            u_pred[i] = u.value

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

        # Dual updates
        alpha += x_pred - x_global
        for i in range(N_drones - 1):
            for t_h in range(horizon):
                diff = x_pred[i, t_h] - x_pred[i + 1, t_h]
                norm = np.linalg.norm(diff)
                z[i, t_h] = ell * diff / norm if norm > 1e-6 else np.array([ell, 0, 0])
                lambd[i, t_h] += x_pred[i, t_h] - x_pred[i + 1, t_h] - z[i, t_h]

        # Residual check
        r_pri = np.linalg.norm(x_pred - x_global)
        r_dual = np.linalg.norm(x_pred - x_prev)
        if r_pri < eps_pri and r_dual < eps_dual:
            print(f"[ADMM] Converged at iteration {k+1} at time step {t}")
            break

    # Shift predictions
    x_pred_next[:, :-1] = x_pred[:, 1:]
    x_pred_next[:, -1] = x_pred[:, -1]
    u_pred_next[:, :-1] = u_pred[:, 1:]
    u_pred_next[:, -1] = 0

    # Apply control and simulate
    u_apply = np.array([u_pred[i, 0] for i in range(N_drones)])
    x_next = np.array([dynamics(x_current[i], u_apply[i]) for i in range(N_drones)])
    x_hist.append(x_next.copy())
    u_hist.append(u_apply.copy())
    x_current = x_next.copy()

x_hist = np.array(x_hist)

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
        ax.text(midpoint[0], midpoint[1], midpoint[2] + 0.1, f"{dist:.2f}", color='red', fontsize=8)

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
ani.save("drone_chain_admm_obstacle_terminated.gif", writer="pillow", fps=int(1/dt))
# plt.show()
