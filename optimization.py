import numpy as np
import cvxpy as cp
import Dragon as Dragon
import pybullet as p
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

dragon = Dragon.Dragon()
dragon.reset_joint_pos("joint1_pitch",-1)
dragon.reset_joint_pos("joint2_yaw", 1)
dragon.reset_joint_pos("joint3_pitch",-1)
dragon.reset_joint_pos("joint2_pitch", -1)
dragon.hover()
dragon.step()

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': '3d'})

dragon.plot_on_ax(ax)
plt.show()

# Constants
F_G_z = np.linalg.norm(dragon.link_position("F1") - dragon.link_position("G1"))
e_z = np.array([0, 0, 1])

# Desired total wrench change
W_star = np.array([0, 0, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz
W_hist = []  # History of wrenches

R_r = []
r_r = []
phi = []
theta = []
lam = []

for MODULE in range(1, dragon.num_modules + 1):
  phi_i = dragon.get_joint_pos("G"+str(MODULE))  # roll
  theta_i = dragon.get_joint_pos("F"+str(MODULE))  # pitch
  lambda_i = dragon.module_thrust(MODULE)  # thrust force

  phi.append(phi_i)
  theta.append(theta_i)
  lam.append(lambda_i)

for i in range(100):

  A = []
  W = np.zeros(6)
  dx = []

  for MODULE in range(1, dragon.num_modules + 1):

    R_ri = np.array(p.getMatrixFromQuaternion(dragon.module_orientation(MODULE))).reshape(3, 3)  # Rotation matrix of the module
    r_ri = dragon.module_position(MODULE)  # Position of the module in inertial frame

    phi_i = phi[MODULE - 1]  # Roll angle of the module
    theta_i = theta[MODULE - 1]  # Pitch angle of the module
    lambda_i = lam[MODULE - 1]  # Thrust force of the module

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
    p_i = p_i[:3] - dragon.center_of_gravity
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
    dx_i = cp.Variable(3)

    # Collect A_i and W_i
    A.append(A_i)
    dx.append(dx_i)
    W += W_i  # Accumulate wrench

  N = len(A)  # Number of modules

  suma = np.zeros(6)

  for i in range(N):
    suma = A[i] @ dx[i] + suma  # Sum of all contributions

  residual = W - W_star + suma  # Residual vector

  constraints = []
  # angle constraints
  for i in range(dragon.num_modules):
    constraints.append(phi[i] + dx[i][0] >= -np.pi / 2)  # phi >= -90 degrees
    constraints.append(phi[i] + dx[i][0] <= np.pi / 2)   # phi <= 90 degrees
    constraints.append(theta[i] + dx[i][1] >= -np.pi / 2)  # theta >= -90 degrees
    constraints.append(theta[i] + dx[i][1] <= np.pi / 2)   # theta <= 90 degrees
    constraints.append(lam[i] + dx[i][2] >= 0)  # lambda >= 0
    constraints.append(lam[i] + dx[i][2] <= 10)  # lambda <= 10 N
  # thrust constraints
  for i in range(dragon.num_modules):
    constraints.append(lam[i] + dx[i][2] >= 0)  # thrust >= 0
    constraints.append(lam[i] + dx[i][2] <= 10)  # thrust <= 10 N


  prob = cp.Problem(cp.Minimize(cp.sum_squares(residual)), constraints)
  prob.solve()

  for i in range(dragon.num_modules):
    # Update angles and thrusts
    phi[i] += dx[i][0].value
    theta[i] += dx[i][1].value
    lam[i] += dx[i][2].value

  W_hist.append(W.copy())


def R_i(module, dragon, phi, theta):
    link_orientation = dragon.module_orientation(module + 1)

    base = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(3, 3)

    # Roll with phi
    cp = np.cos(phi)
    sp = np.sin(phi)
    roll = np.array([[1, 0, 0],
                    [0, cp, -sp],
                    [0, sp, cp]])

    # Pitch with theta
    ct = np.cos(theta)
    st = np.sin(theta)
    pitch = np.array([[ct, 0, st],
                    [0, 1, 0],
                    [-st, 0, ct]])

    return  base @ roll @ pitch

def T_i(module, dragon, phi):
    module_pos = dragon.module_position(module + 1)
    module_orn = dragon.module_orientation(module + 1)

    imu_transform = np.eye(4)
    imu_transform[:3, 3] = module_pos
    imu_transform[:3, :3] = np.array(p.getMatrixFromQuaternion(module_orn)).reshape(3, 3)

    F_G_z = np.linalg.norm(dragon.link_position("F1") - dragon.link_position("G1"))

    vec_transform = np.eye(4)
    vec_transform[2, 3] = F_G_z

    cp = np.cos(phi)
    sp = np.sin(phi)

    roll_transform = np.eye(4)
    roll_transform[:3, :3] = np.array([[1, 0, 0],
                                    [0, cp, -sp],
                                    [0, sp, cp]])


    transform = imu_transform @ roll_transform @ vec_transform

    return transform

def rel_pos(module, dragon, phi):
    return (T_i(module, dragon, phi) @ np.array([0, 0, 0, 1]))[:3] - dragon.center_of_gravity

R = np.array([R_i(i, dragon, phi[i], theta[i]) for i in range(dragon.num_modules)])

u = np.array([rot @ e_z for rot in R])

rel_positions = [rel_pos(i, dragon, phi[i]) for i in range(dragon.num_modules)]

v = np.array([np.cross(rel_positions[0], u[0]),
            np.cross(rel_positions[1], u[1]),
            np.cross(rel_positions[2], u[2]),
            np.cross(rel_positions[3], u[3])])

F = [lam[i] * u[i] for i in range(dragon.num_modules)]
T = [lam[i] * v[i] for i in range(dragon.num_modules)]

sum_F = np.sum(F, axis=0)
sum_T = np.sum(T, axis=0)

W_final = np.concatenate([sum_F, sum_T])
print("Final Wrench:", W_final)
