from Dragon import Dragon, GRAVITY
from Module import Module
import numpy as np
import pybullet as p
import cvxpy as cp

dragon = Dragon()

thrust_range = [0, 2.0 * GRAVITY]
yaw_range = [-np.pi / 2, np.pi / 2]
pitch_range = [-np.pi / 2, np.pi / 2]
phi_range = [-np.pi, np.pi]
theta_range = [-np.pi, np.pi]

alpha_scale = 1 # In meters
beta_scale = 1 / thrust_range[1] * dragon.total_mass # In Newtons
gamma_scale = 1 / thrust_range[1] * dragon.total_mass # In Newtons
rho_scale = 10

alpha = alpha_scale * 1
beta = beta_scale * 1
gamma = gamma_scale * 1
rho =  rho_scale * 1