import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep
import threading
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Dragon:

    def __init__(self, urdf_path="dragon.urdf"):
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        start_pos = [0, 0, 5]
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=False)
        p.setGravity(0, 0, -9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")



        self.num_links = p.getNumJoints(self.robot_id)
        self.center_of_gravity = [0.0, 0.0, 0.0]
        self.link_positions = []
        self.link_names = []

        self.get_com()

    def get_com(self):
        total_mass = 0.0
        weighted_com = [0.0, 0.0, 0.0]
        self.link_positions = []
        self.link_names = []

        for link_index in range(-1, self.num_links):
            dynamics = p.getDynamicsInfo(self.robot_id, link_index)
            mass = dynamics[0]

            if link_index == -1:
                com_pos, _ = p.getBasePositionAndOrientation(self.robot_id)
                link_name = p.getBodyInfo(self.robot_id)[1].decode('utf-8')
            else:
                com_pos, _ = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[0:2]
                link_name = p.getJointInfo(self.robot_id, link_index)[12].decode('utf-8')

            if "origin" in link_name or "modular" in link_name:
                continue

            self.link_positions.append(com_pos)
            self.link_names.append(link_name)

            if mass > 0:
                weighted_com = [w + mass * c for w, c in zip(weighted_com, com_pos)]
                total_mass += mass

        if total_mass > 0:
            self.center_of_gravity = [w / total_mass for w in weighted_com]
        return self.center_of_gravity

    def set_pos(self, name, value):
        name = name.lower()
        for i in range(self.num_links):
            joint_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()
            if name in joint_name:
                p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                        targetPosition=value,
                                        force=1.0,
                                        maxVelocity=1.0)
                return
        raise ValueError(f"Joint '{name}' not found.")

    def step(self):
        p.stepSimulation()
        self.get_com()
        sleep(1/240)

    def _get_color(self, link):
        link = link.lower()
        if "rotor" in link or link[0] == "f":
            return "black"
        if link[0] == "l" or link[0] == "g":
            return "red"
        return "blue"

    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

        self.scatter_cog = self.ax.scatter([], [], [], color='g', s=100, label='Center of Gravity')
        self.scatter_links = [self.ax.scatter([], [], [], color=self._get_color(name), s=20)
                              for name in self.link_names]


        self.plot_plane(self.ax)  # Add this call to plot the plane
        self.ax.set_xlim([-2, 2])
        self.ax.set_ylim([-2, 2])
        self.ax.set_zlim([0, 5])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_title('Dragon URDF Center of Mass Visualization')
        self.ax.legend()

    def update_plot(self, frame):
        cog = self.center_of_gravity
        self.scatter_cog._offsets3d = ([cog[0]], [cog[1]], [cog[2]])

        for scatter, pos in zip(self.scatter_links, self.link_positions):
            scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        return [self.scatter_cog] + self.scatter_links

    def plot_plane(self, ax, size=2.0, z=0.0, color='gray', alpha=0.3):
        # Define the corners of the plane square
        corners = np.array([
            [-size, -size, z],
            [ size, -size, z],
            [ size,  size, z],
            [-size,  size, z]
        ])

        # Create a polygon and add to plot
        plane = Poly3DCollection([corners], color=color, alpha=alpha)
        ax.add_collection3d(plane)

    def animate(self):
        self.init_plot()
        ani = FuncAnimation(self.fig, self.update_plot, interval=1000/60)
        plt.show()


def sim_loop(dragon):
    while True:
        dragon.step()


def main():
    dragon = Dragon()
    dragon.set_pos("joint2_yaw", 0.5)
    dragon.set_pos("joint3_pitch", 0.5)

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    dragon.animate()

if __name__ == "__main__":
    main()
