import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

class Dragon:

    def __init__(self, urdf_path="dragon.urdf",
                 start_pos = [0, 0, 3],
                 start_orientation = [0, 0, 0]):

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        start_orientation = p.getQuaternionFromEuler(start_orientation)
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=False)
        p.setGravity(0, 0, -9.81)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf")

        self.num_links = p.getNumJoints(self.robot_id)
        self.num_rotors = len([i for i in range(self.num_links) if "rotor" in p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()])
        self.center_of_gravity = [0.0, 0.0, 0.0]
        self.link_positions = []
        self.link_orientations = []
        self.link_dimensions = []
        self.link_names = []
        self.total_mass = 0.0

        self._drawn_artists = []

        self.update_kinematics()

    def update_kinematics(self):
        self.total_mass = 0.0
        weighted_com = [0.0, 0.0, 0.0]
        self.link_positions = []
        self.link_orientations = []
        self.link_dimensions = []
        self.link_names = []

        for link_index in range(-1, self.num_links):
            dynamics = p.getDynamicsInfo(self.robot_id, link_index)
            mass = dynamics[0]

            if link_index == -1:
                pos, orn = p.getBasePositionAndOrientation(self.robot_id)
                name = p.getBodyInfo(self.robot_id)[1].decode('utf-8')
                shape_data = []
            else:
                pos, orn = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[0:2]
                name = p.getJointInfo(self.robot_id, link_index)[12].decode('utf-8')
                shape_data = p.getCollisionShapeData(self.robot_id, link_index)

            self.link_positions.append(pos)
            self.link_orientations.append(orn)
            self.link_names.append(name)
            if shape_data and len(shape_data) > 0:
                dims = shape_data[0][3]
                self.link_dimensions.append([d for d in dims])
            else:
                self.link_dimensions.append([0.05, 0.05, 0.05])

            if mass > 0:
                weighted_com = [w + mass * c for w, c in zip(weighted_com, pos)]
                self.total_mass += mass

        if self.total_mass > 0:
            self.center_of_gravity = [w / self.total_mass for w in weighted_com]

    def set_joint_pos(self, name_or_id, value):

        if isinstance(name_or_id, int):
            p.setJointMotorControl2(self.robot_id, name_or_id, p.POSITION_CONTROL,
                                    targetPosition=value,
                                    force=1.0,
                                    maxVelocity=1.0)
        else:
            name = name_or_id.lower()
            for i in range(self.num_links):
                joint_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()
                if name in joint_name:
                    p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL,
                                            targetPosition=value,
                                            force=1.0,
                                            maxVelocity=5.0)
                    return
            raise ValueError(f"Joint '{name}' not found.")

    def hover(self):
        self.thrust([9.82 * self.total_mass / self.num_rotors] * self.num_rotors)

    ######################### SIMULATION #############################

    def thrust(self, forces):
        rotor_indices = [i for i in range(self.num_links) if "rotor" in p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()]
        if len(rotor_indices) != len(forces):
            raise ValueError("Number of forces must match number of rotors.")

        for i, force in zip(rotor_indices, forces):
            p.applyExternalForce(self.robot_id, i, [0, 0, force], self.link_positions[i], p.WORLD_FRAME)

    def lock_joints(self):
        # Lock all joints to their current positions
        for i in range(self.num_links):
            self.set_joint_pos(i, p.getJointState(self.robot_id, i)[0])


    def step(self):
        p.stepSimulation()
        self.update_kinematics()
        sleep(1/240)

    def _get_color(self, link):
        link = link.lower()
        if "rotor" in link or link[0] == "f":
            return "black"
        if link[0] == "l" or link[0] == "g":
            return "red"
        return "blue"

    def draw_box(self, ax, pos, orn, dims, color):
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        l, w, h = dims
        corners = np.array([
            [-l/2, -w/2, -h/2], [ l/2, -w/2, -h/2], [ l/2,  w/2, -h/2], [-l/2,  w/2, -h/2],
            [-l/2, -w/2,  h/2], [ l/2, -w/2,  h/2], [ l/2,  w/2,  h/2], [-l/2,  w/2,  h/2]
        ])
        rotated = (rot_matrix @ corners.T).T + np.array(pos)
        faces = [
            [rotated[i] for i in [0, 1, 2, 3]], [rotated[i] for i in [4, 5, 6, 7]],
            [rotated[i] for i in [0, 1, 5, 4]], [rotated[i] for i in [2, 3, 7, 6]],
            [rotated[i] for i in [1, 2, 6, 5]], [rotated[i] for i in [0, 3, 7, 4]]
        ]
        box = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=0.8)
        ax.add_collection3d(box)
        self._drawn_artists.append(box)


    def init_plot(self):
        self.fig = plt.figure(figsize=(20, 10))
        self.ax_world = self.fig.add_subplot(121, projection='3d')
        self.ax_robot = self.fig.add_subplot(122, projection='3d')
        self.scatter_cog_world = self.ax_world.scatter([], [], [], color='g', s=100, label='Center of Gravity')
        self.scatter_cog_robot = self.ax_robot.scatter([], [], [], color='g', s=100, label='Center of Gravity')
        self.plot_plane(self.ax_world)  # Add this call to plot the plane
        self.ax_world.set_xlim([-2, 2])
        self.ax_world.set_ylim([-2, 2])
        self.ax_world.set_zlim([0, 5])
        self.ax_world.set_xlabel('X')
        self.ax_world.set_ylabel('Y')
        self.ax_world.set_zlabel('Z')
        self.ax_world.set_title('Flight Simulation')

        self.ax_robot.set_xlim([-1, 1])
        self.ax_robot.set_ylim([-1, 1])
        self.ax_robot.set_zlim([-1, 1])
        self.ax_robot.set_xlabel('X')
        self.ax_robot.set_ylabel('Y')
        self.ax_robot.set_zlabel('Z')
        self.ax_robot.set_title('Forces')

        self.box_artists = []

    def update_plot(self, frame):
        for artist in self._drawn_artists:
            artist.remove()
        self._drawn_artists.clear()

        for pos, orn, dims, name in zip(self.link_positions, self.link_orientations, self.link_dimensions, self.link_names):

            if name[0] == "G" or name[0] == "F":
                color = self._get_color(name)
                self.draw_box(self.ax_world, pos, orn, dims, color)

                rel_pos = np.array(pos) - np.array(self.center_of_gravity)
                self.draw_box(self.ax_robot, rel_pos, orn, dims, color)

        cog = self.center_of_gravity
        self.scatter_cog_world._offsets3d = ([cog[0]], [cog[1]], [cog[2]])
        self.scatter_cog_robot._offsets3d = ([0], [0], [0])

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
