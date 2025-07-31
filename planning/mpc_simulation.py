import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.Dragon import Dragon
import numpy as np
import matplotlib.pyplot as plt
import threading
import time
import pybullet as p
from scipy.spatial.transform import Rotation as R
import scenarios

scenario = scenarios.USHAPE_5

savefile = scenario["savefile"]
obstacles =  scenario["obstacles"]

# Load MPC solution
mpc_path = np.load(f"saves/{savefile}.npy", allow_pickle=True)

if len(mpc_path[0]) == 5:
    urdf = "assets/dragon.urdf"
elif len(mpc_path[0]) == 9:
    urdf = "assets/dragon_long.urdf"
else:
    raise ValueError("Invalid MPC solution format. Expected 5 or 9 elements per position.")


dt = 0.1


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

    # Make sure roll is zero
    quat = R.from_quat(quat)
    quat = quat.as_euler('xyz', degrees=False)
    quat[0] = 0  # Set roll to zero
    quat = R.from_euler('xyz', quat, degrees=False).as_quat()

    # Convert to quaternion
    quat = R.from_quat(quat)
    quat = quat.as_quat()  # Convert to quaternion format (x, y, z, w)

    return quat.tolist()

def sim_loop(dragon: Dragon):
    dist_hist = []
    sim_dist_hist = []


    for t, pos in enumerate(mpc_path[1:]):
        dragon.set_pos_ref(mpc_path[:t+1])

        abs_orients = []

        # Absolute orientation of first link
        yaw0, pitch0 = get_yaw_pitch(pos[0], pos[1])

        abs_orients.append((yaw0, pitch0))

        # Place base link
        dragon.reset_start_pos_orn(pos[0], align(pos[0], pos[1]))

        # Compute absolute for each subsequent module
        dists = []

        for i in range(1, dragon.num_modules + 1):
            dist = np.linalg.norm(np.array(pos[i]) - np.array(pos[i-1]))
            dists.append(dist)

        for i in range(1, dragon.num_modules):
            yaw_i, pitch_i = get_yaw_pitch(pos[i], pos[i+1])
            abs_orients.append((yaw_i, pitch_i))

        # Set relative joints
        for i in range(dragon.num_modules - 1):
            yaw_curr, pitch_curr = abs_orients[i]
            yaw_next, pitch_next = abs_orients[i + 1]

            rel_yaw   = angle_diff(yaw_next, yaw_curr)
            rel_pitch = angle_diff(pitch_next, pitch_curr)

            dragon.reset_joint_pos(f"joint{i + 1}_yaw",   rel_yaw)
            dragon.reset_joint_pos(f"joint{i + 1}_pitch", -rel_pitch)

        dragon.step()
        time.sleep(dt)

        sim_dists = []
        for i in range(0, dragon.num_modules):
            dist = np.linalg.norm(pos[i] - dragon.link_position(f"joint{i}_yaw"))
            sim_dists.append(dist)

        dist_hist.append(dists)
        sim_dist_hist.append(sim_dists)
        print(f"Step {t+1}/{len(mpc_path)}")



    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].plot(dist_hist, label = [f"{i}->{i+1}" for i in range(dragon.num_modules)])

    # Horizontal line at ell
    ell = dragon.MODULE_DISTANCE
    ax[0].axhline(y=ell, color='r', linestyle='--', label='Module distance (ell)')
    ymin = np.min(dist_hist)
    ymax = np.max(dist_hist)
    margin = 0.05 * (ymax - ymin)  # optional small buffer
    ax[0].set_ylim(ymin - margin, ymax + margin)
    ax[0].legend()
    ax[0].set_title("Rigid constraint violation")
    ax[0].set_xlabel("Time step")
    ax[0].set_ylabel("Distance (m)")
    ax[0].grid(True)
    ax[1].plot(sim_dist_hist, label = [f"Module {i}" for i in range(dragon.num_modules)])
    ax[1].set_title("Simulation tracking error")
    ax[1].set_xlabel("Time step")
    ax[1].set_ylabel("Distance (m)")
    ax[1].grid(True)
    ax[1].legend()
    plt.savefig(f"saves/{savefile}_distances_plot.png")

    print("Distances plot saved")


def main():
    dragon = Dragon(urdf, gravity=0.0)
    dragon.obstacles = obstacles

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    ani = dragon.animate()
    ani.save(f"saves/{savefile}_simulation.gif", writer='pillow', fps=1 / dt)



if __name__ == "__main__":
    main()
