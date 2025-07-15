import numpy as np
import cvxpy as cp
import Dragon as Dragon
import pybullet as p


dragon = Dragon.Dragon()
dragon.reset_joint_pos("joint1_pitch",-1)
dragon.reset_joint_pos("joint2_yaw", 1)
dragon.reset_joint_pos("joint3_pitch",-1)
dragon.reset_joint_pos("joint2_pitch", -1)
dragon.hover()
dragon.step()

MODULE = 4

# Constants
F_G_z = np.linalg.norm(dragon.link_position("F"+str(MODULE)) - dragon.link_position("G"+str(MODULE)))
e_z = np.array([0, 0, 1])

R_ri = np.array(p.getMatrixFromQuaternion(dragon.module_orientation(MODULE))).reshape(3, 3)  # Rotation matrix of the module
r_ri = dragon.module_position(MODULE)  # Position of the module in inertial frame

W_hist = []  # History of wrenches

# Current state
phi_i = dragon.get_joint_pos("G"+str(MODULE))  # roll
theta_i = dragon.get_joint_pos("F"+str(MODULE))  # pitch
lambda_i = dragon.module_thrust(MODULE)  # thrust force

# Desired total wrench change
W_star = np.array([0, 0, 9.81 * dragon.total_mass / dragon.num_modules, 0, 0, 0])  # fx, fy, fz, tx, ty, tz

for i in range(100):
  cp_phi, sp_phi = np.cos(phi_i), np.sin(phi_i)
  cp_theta, sp_theta = np.cos(theta_i), np.sin(theta_i)

  # Rotation matrices
  R_phi = np.array([[1, 0, 0],
                    [0, cp_phi, -sp_phi],
                    [0, sp_phi, cp_phi]])

  R_theta = np.array([[cp_theta, 0, sp_theta],
                      [0, 1, 0],
                      [-sp_theta, 0, cp_theta]])

  # u_i: thrust direction
  u_i = R_ri @ R_phi @ R_theta @ e_z

  vec_transform = np.eye(4)
  vec_transform[2, 3] = F_G_z

  imu_transform = np.eye(4)
  imu_transform[:3, 3] = r_ri
  imu_transform[:3, :3] = R_ri

  roll_transform = np.eye(4)
  roll_transform[:3, :3] = R_phi

  # p_i: from CoG to thrust point
  p_i = imu_transform @ roll_transform @ vec_transform @ np.array([0, 0, 0, 1])
  p_i = p_i[:3]
  v_i = np.cross(p_i, u_i)

  # Force and torque
  f_i = lambda_i * u_i
  tau_i = lambda_i * v_i

  W_i =  np.concatenate([f_i, tau_i])  # shape (6,)

  # === Jacobians ===

  # dR_phi/dphi
  dR_phi_dphi = np.array([[0, 0, 0],
                          [0, -sp_phi, -cp_phi],
                          [0, cp_phi, -sp_phi]])

  # dR_theta/dtheta
  dR_theta_dtheta = np.array([[-sp_theta, 0, cp_theta],
                              [0, 0, 0],
                              [-cp_theta, 0, -sp_theta]])

  # df/dphi
  du_dphi = R_ri @ dR_phi_dphi @ R_theta @ e_z
  df_dphi = lambda_i * du_dphi

  # df/dtheta
  du_dtheta = R_ri @ R_phi @ dR_theta_dtheta @ e_z
  df_dtheta = lambda_i * du_dtheta

  # df/dlambda
  df_dlambda = u_i

  # dτ/dphi
  dτ_dphi = np.cross(p_i, df_dphi)

  # dτ/dtheta
  dτ_dtheta = np.cross(p_i, df_dtheta)

  # dτ/dlambda
  dτ_dlambda = np.cross(p_i, df_dlambda)

  # === Assemble A_i ===

  A_i = np.block([
      [df_dphi.reshape(3,1), df_dtheta.reshape(3,1), df_dlambda.reshape(3,1)],
      [dτ_dphi.reshape(3,1), dτ_dtheta.reshape(3,1), dτ_dlambda.reshape(3,1)]
  ])  # shape (6, 3)

  # === CVXPY problem ===

  # Variables: delta phi, delta theta, delta lambda
  dx = cp.Variable(3)

  # Linear model
  residual = A_i @ dx - (W_star - W_i)

  # Solve QP
  prob = cp.Problem(cp.Minimize(cp.sum_squares(residual)))
  prob.solve()

  dragon.set_joint_vel("G"+str(MODULE), dx.value[0])  # Set roll velocity
  dragon.set_joint_vel("F"+str(MODULE), dx.value[1])  # Set pitch velocity
  dragon.set_module_thrust(MODULE, lambda_i + dx.value[2])  # Set thrust force

  # Update state
  phi_i += dx.value[0]
  theta_i += dx.value[1]
  lambda_i += dx.value[2]

  W_hist.append(W_i.copy())

# Plot history of each component with target in separate plots
import matplotlib.pyplot as plt

names = ["fx", "fy", "fz", "tx", "ty", "tz"]
plt.figure(figsize=(12, 8))
for i in range(6):
    plt.subplot(3, 2, i + 1)
    plt.plot([w[i] for w in W_hist], label=names[i])
    plt.axhline(W_star[i], color='red', linestyle='--', label='Target')
    plt.title(f'Component {names[i]} over iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.legend()
plt.tight_layout()
plt.show()
