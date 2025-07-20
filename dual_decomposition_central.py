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
W_star = np.array([2, 2, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz


Adj = np.array([
                  [2/3, 1/3, 0,   0  ],
                  [1/3, 1/3, 1/3, 0  ],
                  [0,   1/3, 1/3, 1/3],
                  [0,   0,   1/3, 2/3]
              ])

N = dragon.num_modules

W = dragon.wrench()

alpha = 0.8
beta = 0.9

dW_pred = np.zeros((N, 6))
dW_pred_hist = []

for k in range(3):

  real_dW = W_star - dragon.wrench()
  sum_du = np.zeros(6)
  cost = 0
  constraints = []
  dus = []
  Js = []
  for i in range(N):
    du = cp.Variable(3)

    phi = dragon.module_phi(i + 1)
    theta = dragon.module_theta(i + 1)
    thrust = dragon.module_thrust(i + 1)

    _, J = dragon.linearize_module(i + 1, phi, theta, thrust)

    f = cp.sum_squares(du)

    sum_du = sum_du + J @ du

    cost += f

    constraints.append(phi + du[0] >= -np.pi / 2)  # phi >= -90 degrees
    constraints.append(phi + du[0] <= np.pi / 2)   # phi <= 90 degrees
    constraints.append(theta + du[1] >= -np.pi / 2)  # theta >= -90 degrees
    constraints.append(theta + du[1] <= np.pi / 2)   # theta <= 90 degrees
    constraints.append(thrust + du[2] >= 0)  # lambda >= 0
    constraints.append(thrust + du[2] <= 10)  # lambda <= 10 N

    dus.append(du)
    Js.append(J)

  residual = sum_du - real_dW
  cost += 1000 * cp.sum_squares(residual)

  cp.Problem(cp.Minimize(cost), constraints).solve(verbose = False, polish = False)

  thrusts = []
  pred_dW = np.zeros(6)
  for i in range(N):
    phi = dragon.module_phi(i + 1) + dus[i][0].value
    theta = dragon.module_theta(i + 1) + dus[i][1].value
    thrust = dragon.module_thrust(i + 1) + dus[i][2].value

    dragon.reset_joint_pos("G" + str(i + 1),  phi)
    dragon.reset_joint_pos("F" + str(i + 1), theta)
    thrusts.append(thrust / 2)
    thrusts.append(thrust / 2)

    pred_dW += Js[i] @ dus[i].value

  dragon.step()
  dragon.thrust(thrusts)

print(dragon.wrench() - W_star)