from Dragon import Dragon
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

def R_i(module, dragon):
    link_orientation = dragon.module_orientation(module + 1)

    base = np.array(p.getMatrixFromQuaternion(link_orientation)).reshape(3, 3)

    phi = dragon.get_joint_pos("G" + str(module + 1))
    theta = dragon.get_joint_pos("F" + str(module + 1))

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

def T_i(module, dragon):
    module_pos = dragon.module_position(module + 1)
    module_orn = dragon.module_orientation(module + 1)

    imu_transform = np.eye(4)
    imu_transform[:3, 3] = module_pos
    imu_transform[:3, :3] = np.array(p.getMatrixFromQuaternion(module_orn)).reshape(3, 3)

    F_G_z = np.linalg.norm(dragon.link_position("F1") - dragon.link_position("G1"))

    vec_transform = np.eye(4)
    vec_transform[2, 3] = F_G_z

    # Roll with phi
    phi = dragon.get_joint_pos("G" + str(module + 1))
    cp = np.cos(phi)
    sp = np.sin(phi)

    roll_transform = np.eye(4)
    roll_transform[:3, :3] = np.array([[1, 0, 0],
                                    [0, cp, -sp],
                                    [0, sp, cp]])


    transform = imu_transform @ roll_transform @ vec_transform

    return transform

def rel_pos(module, dragon):
    return (T_i(module, dragon) @ np.array([0, 0, 0, 1]))[:3] - dragon.center_of_gravity

if __name__ == "__main__":

    dragon = Dragon()

    dragon.reset_joint_pos("joint1_pitch",-1)
    dragon.reset_joint_pos("joint2_yaw", 1)
    dragon.reset_joint_pos("joint3_pitch",-1)
    dragon.reset_joint_pos("joint2_pitch", -1)
    dragon.step()
    dragon.hover()

    A = np.array([
                    [2/3, 1/3, 0,   0  ],
                    [1/3, 1/3, 1/3, 0  ],
                    [0,   1/3, 1/3, 1/3],
                    [0,   0,   1/3, 2/3]
                ])

    w = np.ones(dragon.num_modules)

    ez = np.array([0, 0, 1])

    lamb = np.array([dragon.module_thrust(1),
                    dragon.module_thrust(2),
                    dragon.module_thrust(3),
                    dragon.module_thrust(4)])

    R = np.array([R_i(i, dragon) for i in range(dragon.num_modules)])

    u = np.array([rot @ ez for rot in R])


    rel_positions = [rel_pos(i, dragon) for i in range(dragon.num_modules)]

    v = np.array([np.cross(rel_positions[0], u[0]),
                np.cross(rel_positions[1], u[1]),
                np.cross(rel_positions[2], u[2]),
                np.cross(rel_positions[3], u[3])])


    F = [lamb[i] * u[i] for i in range(dragon.num_modules)]
    T = [lamb[i] * v[i] for i in range(dragon.num_modules)]

    F_real = dragon.sum_of_forces()
    T_real = dragon.sum_of_torques()


    f_history = []
    t_history = []
    for i in range(20):
        F = A @ F
        T = A @ T
        f_history.append(F.copy() * dragon.num_modules)
        t_history.append(T.copy() * dragon.num_modules)

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')

    ax2 = fig.add_subplot(222)
    x = np.arange(len(f_history))
    ax2.plot(x, [h[0][0] for h in f_history], label='F1 X', color='blue')
    ax2.plot(x, [h[1][0] for h in f_history], label='F2 X', color='green')
    ax2.plot(x, [h[2][0] for h in f_history], label='F3 X', color='red')
    ax2.plot(x, [h[3][0] for h in f_history], label='F4 X', color='orange')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Fx magnitude')
    ax2.axhline(F_real[0], color='black', linestyle='--', label='Real CoG X')
    ax2.legend()
    ax3 = fig.add_subplot(223)
    ax3.plot(x, [h[0][1] for h in f_history], label='F1 Y', color='blue')
    ax3.plot(x, [h[1][1] for h in f_history], label='F2 Y', color='green')
    ax3.plot(x, [h[2][1] for h in f_history], label='F3 Y', color='red')
    ax3.plot(x, [h[3][1] for h in f_history], label='F4 Y', color='orange')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Fy magnitude')
    ax3.axhline(F_real[1], color='black', linestyle='--', label='Real CoG Y')
    ax3.legend()
    ax4 = fig.add_subplot(224)
    ax4.plot(x, [h[0][2] for h in f_history], label='F1 Z', color='blue')
    ax4.plot(x, [h[1][2] for h in f_history], label='F2 Z', color='green')
    ax4.plot(x, [h[2][2] for h in f_history], label='F3 Z', color='red')
    ax4.plot(x, [h[3][2] for h in f_history], label='F4 Z', color='orange')
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Fz magnitude')
    ax4.axhline(F_real[2], color='black', linestyle='--', label='Real CoG Z')
    ax4.legend()

    F_real /= 9.81
    f_history = [[f / 9.81  for f in h] for h in f_history]

    cog = dragon.center_of_gravity
    # plot history
    for i in range(len(f_history)):
        ax.quiver(cog[0], cog[1], cog[2], f_history[i][0][0], f_history[i][0][1], f_history[i][0][2], color='blue', alpha=0.5, label ="F1" if i == 0 else None)
        ax.quiver(cog[0], cog[1], cog[2], f_history[i][1][0], f_history[i][1][1], f_history[i][1][2], color='green', alpha=0.5, label ="F2" if i == 0 else None)
        ax.quiver(cog[0], cog[1], cog[2], f_history[i][2][0], f_history[i][2][1], f_history[i][2][2], color='red', alpha=0.5, label ="F3" if i == 0 else None)
        ax.quiver(cog[0], cog[1], cog[2], f_history[i][3][0], f_history[i][3][1], f_history[i][3][2], color='orange', alpha=0.5, label ="F4" if i == 0 else None)

    ax.quiver(cog[0], cog[1], cog[2], F_real[0], F_real[1], F_real[2], color='black', label='Real Force', linewidth=10)

    ax.legend()

    dragon.plot_on_ax(ax, CoG=False, forces=False)
    plt.show()

    # Plot torques
    fig2 = plt.figure()
    ax5 = fig2.add_subplot(221, projection='3d')

    ax6 = fig2.add_subplot(222)
    x = np.arange(len(t_history))
    ax6.plot(x, [h[0][0] for h in t_history], label='T1 X', color='blue')
    ax6.plot(x, [h[1][0] for h in t_history], label='T2 X', color='green')
    ax6.plot(x, [h[2][0] for h in t_history], label='T3 X', color='red')
    ax6.plot(x, [h[3][0] for h in t_history], label='T4 X', color='orange')
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Tx magnitude')
    ax6.axhline(T_real[0], color='black', linestyle='--', label='Real CoG X')
    ax6.legend()
    ax7 = fig2.add_subplot(223)
    ax7.plot(x, [h[0][1] for h in t_history], label='T1 Y', color='blue')
    ax7.plot(x, [h[1][1] for h in t_history], label='T2 Y', color='green')
    ax7.plot(x, [h[2][1] for h in t_history], label='T3 Y', color='red')
    ax7.plot(x, [h[3][1] for h in t_history], label='T4 Y', color='orange')
    ax7.set_xlabel('Iteration')
    ax7.set_ylabel('Ty magnitude')
    ax7.axhline(T_real[1], color='black', linestyle='--', label='Real CoG Y')
    ax7.legend()
    ax8 = fig2.add_subplot(224)
    ax8.plot(x, [h[0][2] for h in t_history], label='T1 Z', color='blue')
    ax8.plot(x, [h[1][2] for h in t_history], label='T2 Z', color='green')
    ax8.plot(x, [h[2][2] for h in t_history], label='T3 Z', color='red')
    ax8.plot(x, [h[3][2] for h in t_history], label='T4 Z', color='orange')
    ax8.set_xlabel('Iteration')
    ax8.set_ylabel('Tz magnitude')
    ax8.axhline(T_real[2], color='black', linestyle='--', label='Real CoG Z')
    ax8.legend()
    # plot history
    for i in range(len(t_history)):
        ax5.quiver(cog[0], cog[1], cog[2], t_history[i][0][0], t_history[i][0][1], t_history[i][0][2], color='blue', alpha=0.5, label ="T1" if i == 0 else None)
        ax5.quiver(cog[0], cog[1], cog[2], t_history[i][1][0], t_history[i][1][1], t_history[i][1][2], color='green', alpha=0.5, label ="T2" if i == 0 else None)
        ax5.quiver(cog[0], cog[1], cog[2], t_history[i][2][0], t_history[i][2][1], t_history[i][2][2], color='red', alpha=0.5, label ="T3" if i == 0 else None)
        ax5.quiver(cog[0], cog[1], cog[2], t_history[i][3][0], t_history[i][3][1], t_history[i][3][2], color='orange', alpha=0.5, label ="T4" if i == 0 else None)
    ax5.quiver(cog[0], cog[1], cog[2], T_real[0], T_real[1], T_real[2], color='black', label='Real Torque', linewidth=10)
    ax5.legend()
    dragon.plot_on_ax(ax5, CoG=False, forces=False)
    plt.show()
