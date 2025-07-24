from Dragon import Dragon
import numpy as np
import matplotlib.pyplot as plt
import threading
import time

mpc_solution_x = np.load("mpc_solution_x.npy", allow_pickle=True)
dragon = Dragon("dragon_long.urdf", gravity=0.0)

dt = 0.1

def get_yaw_pitch(x_1, x_2):
    dx = x_2[0] - x_1[0]
    dy = x_2[1] - x_1[1]
    dz = x_2[2] - x_1[2]

    yaw = np.arctan2(dy, dx) - np.pi / 2
    pitch = np.arctan2(dz, np.sqrt(dx**2 + dy**2))
    return yaw, pitch


def sim_loop(dragon: Dragon):
    for pos in mpc_solution_x:
        # Store absolute orientations
        abs_orientations = []

        # First prism orientation
        yaw, pitch = get_yaw_pitch(pos[0], pos[1])
        abs_orientations.append((yaw, pitch))

        # Set base
        dragon.reset_start_pos_orn(pos[0], [0, pitch, yaw + 1.57])

        dragon.reset_joints()

        # Compute absolute orientations
        for i in range(1, dragon.num_modules):
            yaw, pitch = get_yaw_pitch(pos[i], pos[i + 1])
            abs_orientations.append((yaw, pitch))

        # Set relative orientations (joints)
        for i in range(1, dragon.num_modules):
            yaw_prev, pitch_prev = abs_orientations[i - 1]
            yaw_curr, pitch_curr = abs_orientations[i]

            rel_yaw = yaw_curr - yaw_prev
            rel_pitch = pitch_curr - pitch_prev

            dragon.reset_joint_pos(f"joint{i}_yaw", rel_yaw)
            dragon.reset_joint_pos(f"joint{i}_pitch", -rel_pitch)

        dragon.step()

        time.sleep(0.1)

        dragon.set_pos_ref(pos)

    print("Simulation completed.")

def main():
    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    ani = dragon.animate()
    plt.show()
    # ani.save("mpc_simulation.gif", writer="pillow", fps=1 / dt)


if __name__ == "__main__":
    main()
