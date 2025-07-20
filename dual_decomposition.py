import numpy as np
import cvxpy as cp
import Dragon as Dragon
import pybullet as p
import threading
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

dragon = Dragon.Dragon()
dragon.reset_joint_pos("joint1_pitch",-1.5)
dragon.reset_joint_pos("joint2_pitch", 1.5)
dragon.reset_joint_pos("joint3_pitch", 1.5)
dragon.hover()
dragon.step()

# Constants

e_z = np.array([0, 0, 1])

# Desired total wrench change
W_star = np.array([2, 0, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz

real_dW = W_star - dragon.wrench()

Adj = np.array([
                  [2/3, 1/3, 0,   0  ],
                  [1/3, 1/3, 1/3, 0  ],
                  [0,   1/3, 1/3, 1/3],
                  [0,   0,   1/3, 2/3]
              ])


N = dragon.num_modules

W = [dragon.module_wrench(i) for i in range(1, N + 1)]

for i in range(100):
  W = Adj @ W


alpha = 0.6
beta = 1

dW_pred = np.zeros((N, 6))
dW_pred_hist = []

V = W.copy()

Q = np.diag([1, 1, 1, 10, 20, 10]) * alpha

MAX_ITERS = 100

epsilon = 0.001

prev_dW = None
for k in range(MAX_ITERS):
  V_new = V.copy()
  dus = []
  for i in range(N):
    dW = W_star / N - W[i]

    du = cp.Variable(3)
    phi = dragon.module_phi(i + 1)
    theta = dragon.module_theta(i + 1)
    thrust = dragon.module_thrust(i + 1)

    _, J = dragon.linearize_module(i + 1, phi, theta, thrust)

    f = cp.sum_squares(du)

    lagrange = V[i] @ (J @ du)

    constraints = []
    constraints.append(phi + du[0] >= -np.pi / 2)  # phi >= -90 degrees
    constraints.append(phi + du[0] <= np.pi / 2)   # phi <= 90 degrees
    constraints.append(theta + du[1] >= -np.pi / 2)  # theta >= -90 degrees
    constraints.append(theta + du[1] <= np.pi / 2)   # theta <= 90 degrees
    constraints.append(thrust + du[2] >= 0)  # lambda >= 0
    constraints.append(thrust + du[2] <= 10)  # lambda <= 10 N
    constraints.append(cp.abs(du[0]) <= 0.2)
    constraints.append(cp.abs(du[1]) <= 0.2)

    cp.Problem(cp.Minimize(f + lagrange), constraints).solve(verbose = False, polish = False)
    du_value = du.value

    cons = 0
    for j in range(N):
      cons += Adj[i, j] * (V[j] - V[i])

    V_new[i] += Q @ (J @ du_value - dW) + beta * cons

    dW_pred[i] = J @ du_value

  dW_pred_hist.append(dW_pred.copy())
  V = V_new.copy()

  if prev_dW is not None and np.all(np.abs(np.sum(dW_pred, axis = 0) - prev_dW) < epsilon):
    print(f"Converged at iteration {k}")
    break

  prev_dW = np.sum(dW_pred, axis=0)

names = ["FX", "FY", "FZ", "TX", "TY", "TZ"]
fig = plt.figure()
x = np.arange(len(dW_pred_hist))
for i, name in enumerate(names):
  ax = fig.add_subplot(2, 3, i + 1)
  ax.plot(x, [h[0][i] for h in dW_pred_hist], label=f'1', color='blue')
  ax.plot(x, [h[1][i] for h in dW_pred_hist], label=f'2', color='green')
  ax.plot(x, [h[2][i] for h in dW_pred_hist], label=f'3', color='red')
  ax.plot(x, [h[3][i] for h in dW_pred_hist], label=f'4', color='orange')
  ax.plot(x, [np.sum(h, axis = 0)[i] for h in dW_pred_hist], label="Sum", color='purple')
  ax.set_xlabel('Iteration')
  ax.set_ylabel(f'{name} magnitude')
  ax.axhline(real_dW[i], color='black', linestyle='--', label=f'Targ. {name}')

plt.show()