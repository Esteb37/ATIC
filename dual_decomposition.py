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
W_star = np.array([2, 1, 9.81 * dragon.total_mass, 1, 1, 1])  # fx, fy, fz, tx, ty, tz

real_dW = W_star - dragon.wrench()

Adj = np.array([
                  [2/3, 1/3, 0,   0  ],
                  [1/3, 1/3, 1/3, 0  ],
                  [0,   1/3, 1/3, 1/3],
                  [0,   0,   1/3, 2/3]
              ])

ITERS = 1000

N = dragon.num_modules

W = [dragon.module_wrench(i) for i in range(1, N + 1)]

for i in range(100):
  W = Adj @ W

W *= N

alpha = 0.8
beta = 0.9

dW_pred = np.zeros((N, 6))
dW_pred_hist = []

V = np.zeros((N, 6))
for k in range(ITERS):
  V_new = V.copy()
  for i in range(N):
    dW = (W_star - W[i]) / N

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

    cp.Problem(cp.Minimize(f + lagrange), constraints).solve(verbose = False, polish = False)
    du_value = du.value

    cons = 0
    for j in range(N):
      cons += Adj[i, j] * (V[j] - V[i])

    V_new[i] += alpha * (J @ du_value - dW) + beta * cons

    dW_pred[i] = J @ du_value

  dW_pred_hist.append(dW_pred.copy())

  V = V_new.copy()

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