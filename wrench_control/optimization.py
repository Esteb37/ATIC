import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import cvxpy as cp
from sim.Dragon import Dragon
import pybullet as p
import threading
import matplotlib.pyplot as plt
import time

np.set_printoptions(precision=4, suppress=True)

dragon = Dragon()
dragon.set_joint_pos("joint1_pitch", -1.57)
dragon.set_joint_pos("joint2_pitch", 1.57)
dragon.set_joint_pos("joint3_pitch", 1.57)
dragon.hover()
dragon.step()

# Constants
F_G_z = np.linalg.norm(dragon.link_position("F1") - dragon.link_position("G1"))
e_z = np.array([0, 0, 1])

# Desired total wrench change
savefile = "forward_snake"
W_star = np.array([2, 0, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz
W_hist = []  # History of wrenches

fps = 15

def sim_loop(dragon: Dragon):
  k = 0
  while True:
    if k % 5 == 0:

      phi = []
      theta = []
      lam = []

      W = np.zeros(6)
      dx = []

      suma = np.zeros(6)

      constraints = []

      for MODULE in range(1, dragon.num_modules + 1):
        phi_i = dragon.get_joint_pos(f"G{MODULE}")
        theta_i = dragon.get_joint_pos(f"F{MODULE}")
        lambda_i = dragon.module_thrust(MODULE)

        W_i, J_i = dragon.linearize_module(MODULE, phi_i, theta_i, lambda_i)

        # === CVXPY problem ===

        # Variables: delta phi, delta theta, delta lambda
        dx_i = cp.Variable(3)

        W += W_i  # Accumulate wrench

        suma = J_i @ dx_i + suma  # Sum of all contributions

        constraints.append(phi_i + dx_i[0] >= -np.pi / 2)  # phi >= -90 degrees
        constraints.append(phi_i + dx_i[0] <= np.pi / 2)   # phi <= 90 degrees
        constraints.append(theta_i + dx_i[1] >= -np.pi / 2)  # theta >= -90 degrees
        constraints.append(theta_i + dx_i[1] <= np.pi / 2)   # theta <= 90 degrees
        constraints.append(lambda_i + dx_i[2] >= 0)  # lambda >= 0
        constraints.append(lambda_i + dx_i[2] <= 10)  # lambda <= 10 N
        constraints.append(cp.abs(dx_i[0]) <= 0.1)
        constraints.append(cp.abs(dx_i[1]) <= 0.1)
        constraints.append(cp.abs(dx_i[2]) <= 0.1)

        phi.append(phi_i)
        theta.append(theta_i)
        lam.append(lambda_i)
        dx.append(dx_i)

      residual = (dragon.wrench() + suma) - W_star

      cost = cp.sum_squares(residual)

      prob = cp.Problem(cp.Minimize(cost), constraints)
      prob.solve()

      for i in range(dragon.num_modules):
        # Update angles and thrusts
        phi[i] += dx[i][0].value
        theta[i] += dx[i][1].value
        lam[i] += dx[i][2].value

      W_hist.append(W.copy())

    dragon.reset_joint_pos("G1", phi[0])
    dragon.reset_joint_pos("G2", phi[1])
    dragon.reset_joint_pos("G3", phi[2])
    dragon.reset_joint_pos("G4", phi[3])
    dragon.reset_joint_pos("F1", theta[0])
    dragon.reset_joint_pos("F2", theta[1])
    dragon.reset_joint_pos("F3", theta[2])
    dragon.reset_joint_pos("F4", theta[3])

    k += 1

    dragon.step()
    dragon.thrust([lam[0] / 2, lam[0] / 2, lam[1] / 2, lam[1] / 2,
                  lam[2] / 2, lam[2] / 2, lam[3] / 2, lam[3] / 2])

    # Print seconds
    print(f"{k / 240:.2f}s", end="\r")

def main():
  threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
  ani = dragon.animate()
  ani.save(f"saves/{savefile}_wrench.gif", writer='pillow', fps=fps, dpi=100)

if __name__ == "__main__":
    main()
