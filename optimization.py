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
W_star = np.array([0.2, 0, 9.81 * dragon.total_mass, 0, 0, 0])  # fx, fy, fz, tx, ty, tz
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
        phi_i = dragon.get_joint_pos("G"+str(MODULE))  # roll
        theta_i = dragon.get_joint_pos("F"+str(MODULE))  # pitch
        lambda_i = dragon.module_thrust(MODULE)  # thrust force

        R_ri = np.array(p.getMatrixFromQuaternion(dragon.module_orientation(MODULE))).reshape(3, 3)  # Rotation matrix of the module
        r_ri = dragon.module_position(MODULE)  # Position of the module in inertial frame

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

        W += W_i  # Accumulate wrench

        suma = A_i @ dx_i + suma  # Sum of all contributions

        constraints.append(phi_i + dx_i[0] >= -np.pi / 2)  # phi >= -90 degrees
        constraints.append(phi_i + dx_i[0] <= np.pi / 2)   # phi <= 90 degrees
        constraints.append(theta_i + dx_i[1] >= -np.pi / 2)  # theta >= -90 degrees
        constraints.append(theta_i + dx_i[1] <= np.pi / 2)   # theta <= 90 degrees
        constraints.append(lambda_i + dx_i[2] >= 0)  # lambda >= 0
        constraints.append(lambda_i + dx_i[2] <= 10)  # lambda <= 10 N
        constraints.append(lambda_i + dx_i[2] >= 0)  # thrust >= 0
        constraints.append(lambda_i + dx_i[2] <= 10)  # thrust <= 10 N

        phi.append(phi_i)
        theta.append(theta_i)
        lam.append(lambda_i)
        dx.append(dx_i)

      residual = W - W_star + suma  # Residual vector

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
    dragon.thrust([lam[3] / 2, lam[3] / 2, lam[2] / 2, lam[2] / 2,
                  lam[1] / 2, lam[1] / 2, lam[0] / 2, lam[0] / 2,])

def main():
  threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
  dragon.animate()

if __name__ == "__main__":
    main()
