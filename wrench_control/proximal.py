import numpy as np
import cvxpy as cp
import sim.Dragon as Dragon
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
W_star = np.array([1, 0, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz
W_hist = []  # History of wrenches

alpha = 10000
beta = 1
rho = 1

Adj = np.array([
                  [2/3, 1/3, 0,   0  ],
                  [1/3, 1/3, 1/3, 0  ],
                  [0,   1/3, 1/3, 1/3],
                  [0,   0,   1/3, 2/3]
              ])

ADMM_ITERATIONS = 20

def module_problem(MODULE, phi, theta, lamb, dual_W, z_W):

    # CVX Problem Setup

    W, A, _= dragon.linearize_module(MODULE, phi, theta, lamb)

    dx = cp.Variable(3)

    track_cost = alpha / 2 * cp.sum_squares((W + A @ dx) - W_star / dragon.num_modules)
    effort_cost = beta / 2 * cp.sum_squares(dx)
    cons_cost = rho / 2 * cp.sum_squares((W + A @ dx) - z_W + dual_W)

    cost = cons_cost + track_cost

    constraints = []
    constraints.append(phi + dx[0] >= -np.pi / 2)  # phi >= -90 degrees
    constraints.append(phi + dx[0] <= np.pi / 2)   # phi <= 90 degrees
    constraints.append(theta + dx[1] >= -np.pi / 2)  # theta >= -90 degrees
    constraints.append(theta + dx[1] <= np.pi / 2)   # theta <= 90 degrees
    constraints.append(lamb + dx[2] >= 0)  # lambda >= 0
    constraints.append(lamb + dx[2] <= 10)  # lambda <= 10 N

    prob = cp.Problem(cp.Minimize(cost), constraints)

    return prob, W, A, dx, track_cost, cons_cost


def solve_admm(dragon : Dragon):
  N = dragon.num_modules  # Number of modules
  dual_W = np.zeros((N, 6))  # Dual variables for wrenches
  z_W = [dragon.module_wrench(i + 1) for i in range(N)]  # Initial wrenches from modules

  updated_W = np.zeros((N, 6))  # Updated wrenches

  variables = []

  for MODULE in range(1, N + 1):
    phi = dragon.get_joint_pos("G"+str(MODULE))  # roll
    theta = dragon.get_joint_pos("F"+str(MODULE))  # pitch
    lamb = dragon.module_thrust(MODULE)  # thrust force

    variables.append((phi, theta, lamb))

  f_history = []
  for _ in range(ADMM_ITERATIONS):
    probs = []

    for i in range(N):
      phi, theta, lamb = variables[i]
      probs.append(module_problem(i + 1, phi, theta, lamb, dual_W[i], z_W[i]))

    for i in range(N):
      problem, current_W, A, dx, track_cost, cons_cost = probs[i]
      problem.solve()

      updated_W[i] = current_W + A @ dx.value

      variables[i] = (variables[i][0] + dx.value[0],
                      variables[i][1] + dx.value[1],
                      variables[i][2] + dx.value[2])

    z_W = Adj @ (updated_W + dual_W)  # Consensus step

    f_history.append(z_W.copy())

    dual_W += (updated_W - z_W)


  phi = np.array([variables[i][0] for i in range(N)])
  theta = np.array([variables[i][1] for i in range(N)])

  dragon.reset_joint_pos("G1", phi[0])
  dragon.reset_joint_pos("G2", phi[1])
  dragon.reset_joint_pos("G3", phi[2])
  dragon.reset_joint_pos("G4", phi[3])
  dragon.reset_joint_pos("F1", theta[0])
  dragon.reset_joint_pos("F2", theta[1])
  dragon.reset_joint_pos("F3", theta[2])
  dragon.reset_joint_pos("F4", theta[3])

  dragon.step()

  lambs = [variables[i][2] / 2 for i in range(N)]

  dragon.thrust([lambs[0], lambs[0], lambs[1], lambs[1],
                 lambs[2], lambs[2], lambs[3], lambs[3]])

  fig = plt.figure()
  ax2 = fig.add_subplot(222)
  x = np.arange(len(f_history))
  ax2.plot(x, [h[0][0] for h in f_history], label='F1 X', color='blue')
  ax2.plot(x, [h[1][0] for h in f_history], label='F2 X', color='green')
  ax2.plot(x, [h[2][0] for h in f_history], label='F3 X', color='red')
  ax2.plot(x, [h[3][0] for h in f_history], label='F4 X', color='orange')
  ax2.set_xlabel('Iteration')
  ax2.set_ylabel('Fx magnitude')
  ax2.axhline(W_star[0], color='black', linestyle='--', label='Real CoG X')
  ax2.legend()
  ax3 = fig.add_subplot(223)
  ax3.plot(x, [h[0][1] for h in f_history], label='F1 Y', color='blue')
  ax3.plot(x, [h[1][1] for h in f_history], label='F2 Y', color='green')
  ax3.plot(x, [h[2][1] for h in f_history], label='F3 Y', color='red')
  ax3.plot(x, [h[3][1] for h in f_history], label='F4 Y', color='orange')
  ax3.set_xlabel('Iteration')
  ax3.set_ylabel('Fy magnitude')
  ax3.axhline(W_star[1], color='black', linestyle='--', label='Real CoG Y')
  ax3.legend()
  ax4 = fig.add_subplot(224)
  ax4.plot(x, [h[0][2] for h in f_history], label='F1 Z', color='blue')
  ax4.plot(x, [h[1][2] for h in f_history], label='F2 Z', color='green')
  ax4.plot(x, [h[2][2] for h in f_history], label='F3 Z', color='red')
  ax4.plot(x, [h[3][2] for h in f_history], label='F4 Z', color='orange')
  ax4.set_xlabel('Iteration')
  ax4.set_ylabel('Fz magnitude')
  ax4.axhline(W_star[2], color='black', linestyle='--', label='Real CoG Z')
  ax4.legend()
  plt.savefig("admm_wrench_convergence.png")
  print("here")


def sim_loop(dragon: Dragon):
  while True:
    dragon.step()
    solve_admm(dragon)

def main():
  threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
  dragon.animate()

if __name__ == "__main__":
    main()
