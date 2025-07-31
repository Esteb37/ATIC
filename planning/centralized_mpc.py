import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time



# ===============================
# Parameters
# ===============================
N_drones = 5
H = 10
dt = 0.1
ell = 1.0
u_max = 1.0
dim = 3
T_sim = 60
warm_start_steps = 5  # Initial steps without obstacles

# Obstacles
obstacles = []
safety_margin = 0.0

# Initial/reference positions
x_ref = np.array([[i * ell, 0, 0] for i in range(N_drones)])
x_current = np.array([[0, i * ell, 0] for i in range(N_drones)])
x_hist = [x_current.copy()]

def dynamics(x, u):
    return x + dt * u

# Initial feasible trajectory
x_prev_traj = np.zeros((N_drones, H+1, dim))
for i in range(N_drones):
    for k in range(H+1):
        alpha = k / H
        x_prev_traj[i, k] = (1 - alpha) * x_current[i] + alpha * x_ref[i]

# ===============================
# MPC Loop
# ===============================
total = 0
for t in range(T_sim):
    start = time.time()
    X = cp.Variable((N_drones, H+1, dim))
    U = cp.Variable((N_drones, H, dim))
    slack_rigid = cp.Variable((N_drones-1, H), nonneg=True)
    slack_obs = cp.Variable((N_drones, H), nonneg=True)

    cost = 0
    constraints = []

    # Initial condition
    for i in range(N_drones):
        constraints += [X[i, 0, :] == x_current[i]]

    # Build MPC
    for k in range(H):
        for i in range(N_drones):
            cost += 10 * cp.sum_squares(X[i, k] - x_ref[i]) + 0.01 * cp.sum_squares(U[i, k])
            constraints += [X[i, k+1] == X[i, k] + dt * U[i, k]]
            constraints += [cp.norm(U[i, k], 'inf') <= u_max]

            # Obstacle constraints only after warm start
            if t >= warm_start_steps:
                for obs in obstacles:
                    p = obs["center"]
                    r = obs["radius"] + safety_margin
                    x_bar = x_prev_traj[i, k]
                    n_o = (x_bar - p) / (np.linalg.norm(x_bar - p) + 1e-6)
                    constraints += [n_o @ (X[i, k] - p) >= r - slack_obs[i, k]]

        # Rigid-link constraints (with slack)
        for i in range(N_drones - 1):
            d_bar = x_prev_traj[i+1, k] - x_prev_traj[i, k]
            n_link = d_bar / (np.linalg.norm(d_bar) + 1e-6)
            constraints += [
                n_link @ (X[i+1, k] - X[i, k]) >= ell - slack_rigid[i, k],
                n_link @ (X[i+1, k] - X[i, k]) <= ell + slack_rigid[i, k]
            ]

    # Penalise slack variables more strongly after warm start
    if t >= warm_start_steps:
        cost += 1000 * cp.sum(slack_rigid) + 1000 * cp.sum(slack_obs)
    else:
        cost += 10 * cp.sum(slack_rigid)

    # Solve MPC
    prob = cp.Problem(cp.Minimize(cost), constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if prob.status != cp.OPTIMAL:
        print(f"[Warning] MPC infeasible at timestep {t}")
        break

    # Apply control
    u_apply = np.array([U[i, 0].value for i in range(N_drones)])
    x_next = np.array([dynamics(x_current[i], u_apply[i]) for i in range(N_drones)])
    x_hist.append(x_next.copy())
    x_current = x_next.copy()

    # Update linearisation trajectory
    for i in range(N_drones):
        for k in range(H+1):
            alpha = k / H
            x_prev_traj[i, k] = (1 - alpha) * x_current[i] + alpha * x_ref[i]

    end = time.time()
    total += end - start
    print(f"{total / (t+1)}")

print("Linearised Centralised MPC simulation complete.")
x_hist = np.array(x_hist)

# ===============================
# Animation
# ===============================
frames = len(x_hist)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.cla()
    ax.set_title(f"Linearised Centralised MPC (Step {frame})")
    ax.set_xlim(-1, 5)
    ax.set_ylim(-1, 5)
    ax.set_zlim(-1, 2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot paths
    for i in range(N_drones):
        ax.plot(x_hist[:frame+1, i, 0], x_hist[:frame+1, i, 1], x_hist[:frame+1, i, 2])
        ax.scatter(*x_hist[frame, i], s=40)

    # Rigid links + distances
    for i in range(N_drones - 1):
        p1, p2 = x_hist[frame, i], x_hist[frame, i+1]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k-', lw=2)

        # Show distance as text
        dist = np.linalg.norm(p2 - p1)
        midpoint = (p1 + p2) / 2
        ax.text(midpoint[0], midpoint[1], midpoint[2] + 0.05, f"{dist:.2f}", color="red", fontsize=8)

    # Obstacles
    for obs in obstacles:
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        cx, cy, cz = obs["center"]
        r = obs["radius"]
        xs = r * np.cos(u) * np.sin(v) + cx
        ys = r * np.sin(u) * np.sin(v) + cy
        zs = r * np.cos(v) + cz
        ax.plot_surface(xs, ys, zs, color='gray', alpha=0.3)

ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
ani.save("linearised_mpc_with_distances.gif", writer="pillow", fps=5)
plt.show()