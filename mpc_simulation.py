from Dragon import Dragon
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import pybullet as p
from scipy.spatial.transform import Rotation as R

# Load MPC solution
mpc_solution_x = np.load("mpc_solution_x.npy", allow_pickle=True)
dragon = Dragon("dragon_long.urdf", gravity=0.0)
dt = 0.1

obstacles = [{"center": np.array([1, 1.5, -0.5]), "radius": 0.8}]

dragon.obstacles = obstacles

def angle_diff(a, b):
    """Return the difference a−b wrapped to [−π, π]."""
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi

def get_yaw_pitch(x_1, x_2):
    """
    Calculate yaw and pitch from two positions.
    :param x_1: Position of the first prism [x,y,z].
    :param x_2: Position of the second prism [x,y,z].
    :return: (yaw, pitch) in radians.
    """
    dx = x_2[0] - x_1[0]
    dy = x_2[1] - x_1[1]
    dz = x_2[2] - x_1[2]
    yaw = np.arctan2(dy, dx)
    pitch = np.arctan2(dz, np.hypot(dx, dy))
    return yaw, pitch

def align(p1, p2):
    v = np.array(p2) - np.array(p1)
    v_norm = v / np.linalg.norm(v)

    # Original direction of prism's z-axis
    z_axis = np.array([1, 0, 0])

    # Axis-angle to rotate z_axis → v_norm
    if np.allclose(v_norm, z_axis):
        quat = R.from_rotvec([0, 0, 0]).as_quat()
    elif np.allclose(v_norm, -z_axis):
        quat = R.from_rotvec(np.pi * np.array([1, 0, 0])).as_quat()  # 180° flip
    else:
        axis = np.cross(z_axis, v_norm)
        angle = np.arccos(np.dot(z_axis, v_norm))
        quat = R.from_rotvec(angle * axis / np.linalg.norm(axis)).as_quat()

    return quat.tolist()

def sim_loop(dragon: Dragon):
    for pos in mpc_solution_x:
        dragon.set_pos_ref(pos)
        abs_orients = []

        # Absolute orientation of first link
        yaw0, pitch0 = get_yaw_pitch(pos[0], pos[1])
        abs_orients.append((yaw0, pitch0))

        # Place base link
        dragon.reset_start_pos_orn(pos[0], align(pos[0], pos[1]))
        dragon.reset_joints()

        # Compute absolute for each subsequent module
        for i in range(1, dragon.num_modules):
            yaw_i, pitch_i = get_yaw_pitch(pos[i], pos[i+1])
            abs_orients.append((yaw_i, pitch_i))

        # Set relative joints
        for i in range(1, dragon.num_modules):
            yaw_prev, pitch_prev = abs_orients[i-1]
            yaw_curr, pitch_curr = abs_orients[i]

            rel_yaw   = angle_diff(yaw_curr,  yaw_prev)
            rel_pitch = angle_diff(pitch_curr, pitch_prev)

            dragon.reset_joint_pos(f"joint{i}_yaw",   rel_yaw)
            dragon.reset_joint_pos(f"joint{i}_pitch", -rel_pitch)

        dragon.step()
        time.sleep(dt)

    print("Simulation completed.")

def main():
    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    ani = dragon.animate()
    # plt.show()
    ani.save("mpc_simulation.gif", writer="pillow", fps=1/dt)

if __name__ == "__main__":
    main()
