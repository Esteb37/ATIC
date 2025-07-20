import numpy as np
import cvxpy as cp
import Dragon as Dragon
import pybullet as p
import threading

np.set_printoptions(precision=4, suppress=True)

dragon = Dragon.Dragon()
dragon.set_joint_pos("joint1_pitch",-1.5)
dragon.set_joint_pos("joint2_pitch", 1.5)
dragon.set_joint_pos("joint3_pitch", 1.5)
dragon.hover()
dragon.step()

# Constants
F_G_z = np.linalg.norm(dragon.link_position("F1") - dragon.link_position("G1"))
e_z = np.array([0, 0, 1])

# Desired total wrench change
W_star = np.array([2, 1, 9.81 * dragon.total_mass, 1, 1, 1])  # fx, fy, fz, tx, ty, tz
W_hist = []  # History of wrenches


def sim_loop(dragon: Dragon):

  k = 0

  while True:

    if k % 5 == 0:
      print("Iteration:", k)

      phi = []
      theta = []
      lam = []


      A = []
      W = np.zeros(6)
      dx = []

      suma = np.zeros(6)

      constraints = []

      for MODULE in range(1, dragon.num_modules + 1):
        phi_i = dragon.get_joint_pos(f"G{MODULE}")
        theta_i = dragon.get_joint_pos(f"F{MODULE}")
        lambda_i = dragon.module_thrust(MODULE)

        W_i, A_i = dragon.linearize_module(MODULE, phi_i, theta_i, lambda_i)

        # === CVXPY problem ===

        # Variables: delta phi, delta theta, delta lambda
        dx_i = cp.Variable(3)

        W += W_i  # Accumulate wrench

        suma = A_i @ dx_i + suma  # Sum of all contributions

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
        print(f"Module {i+1}: dx = {dx[i].value}, W = {W[i]}")

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
                  lam[2] / 2, lam[2] / 2, lam[3] / 2, lam[3] / 2,])

def main():
  threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
  dragon.animate()

if __name__ == "__main__":
    main()
