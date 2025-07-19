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


alpha = 100
beta = 1
rho = 100

Adj = np.array([
                  [2/3, 1/3, 0,   0  ],
                  [1/3, 1/3, 1/3, 0  ],
                  [0,   1/3, 1/3, 1/3],
                  [0,   0,   1/3, 2/3]
              ])

ADMM_ITERATIONS = 20
