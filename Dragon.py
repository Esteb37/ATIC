import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from time import sleep
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

GRAVITY = 9.81

class Dragon:

    def __init__(self, urdf_path="dragon.urdf",
                 start_pos = np.array([0, 0, 3]),
                 start_orientation = np.array([0, 0, 0]), gravity = None):

        if gravity is None:
            self.GRAVITY = GRAVITY
        else:
            self.GRAVITY = gravity

        ####### Setup PyBullet #######
        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -self.GRAVITY)
        p.setPhysicsEngineParameter(numSolverIterations=200)
        # p.setPhysicsEngineParameter(enableConeFriction=1)
        p.setPhysicsEngineParameter(numSubSteps=5)
        p.setPhysicsEngineParameter(solverResidualThreshold=1e-6)
        p.setPhysicsEngineParameter(contactBreakingThreshold=0.05)
        p.setTimeStep(1/240)

        # p.loadURDF("plane.urdf")


        ####### Load Robot #######
        start_orientation = p.getQuaternionFromEuler(start_orientation)
        self.robot_id = p.loadURDF(urdf_path, start_pos, start_orientation, useFixedBase=False)

        ####### Kinematics Information #######
        self.rotor_names = [p.getJointInfo(self.robot_id, i)[12].decode('utf-8') for i in range(p.getNumJoints(self.robot_id)) if "rotor" in p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()]

        self.rotor_names = self.rotor_names[::-1]

        self.num_links = p.getNumJoints(self.robot_id)
        self.num_rotors = len(self.rotor_names)
        self.num_modules = int(self.num_rotors / 2)
        self.kinematics = {}
        ####### Dynamics Variables #######

        self.rotor_thrusts = []
        self.external_forces = []
        self.center_of_gravity = np.zeros(3)
        self.total_mass = 0.0

        self._drawn_artists = [] # For plotting

        self._pos_ref = []

        self.load_body_info()
        self.update_kinematics()

        self.thrust([0] * self.num_rotors)

        self.F_G_z = np.linalg.norm(self.link_position("F1") - self.link_position("G1"))
        self.MODULE_DISTANCE = np.linalg.norm(self.link_position("G2") - self.link_position("G1"))

    ####### Kinematics and Dynamics Methods #######
    def load_body_info(self):
        self.link_dimensions = []
        self.total_mass = 0.0

        for link_index in range(-1, self.num_links):
            dynamics = p.getDynamicsInfo(self.robot_id, link_index)
            mass = dynamics[0]

            if link_index == -1:
                name = p.getBodyInfo(self.robot_id)[1].decode('utf-8')
                shape_data = []
            else:
                name = p.getJointInfo(self.robot_id, link_index)[12].decode('utf-8')
                shape_data = p.getCollisionShapeData(self.robot_id, link_index)

            self.kinematics[name] = {"index": link_index, "mass": mass}

            if shape_data and len(shape_data) > 0:
                dims = shape_data[0][3]
                self.kinematics[name]["dimensions"] = [d for d in dims]
            else:
                self.kinematics[name]["dimensions"] = np.array([0.3, 0.05, 0.05])

            if mass > 0:
                self.total_mass += mass

    def link_pos_orn(self, name_or_id):
        idx = self._get_id(name_or_id)
        if idx == -1:
            pos, orn = p.getBasePositionAndOrientation(self.robot_id)
            return np.array(pos), np.array(orn)
        else:
            pos, orn = p.getLinkState(self.robot_id, idx, computeForwardKinematics=True)[0:2]
            return np.array(pos), np.array(orn)

    def update_kinematics(self):
        weighted_com = np.array([0.0, 0.0, 0.0])

        for link_name, link_info in self.kinematics.items():
            pos, orn = self.link_pos_orn(link_name)
            self.kinematics[link_name]["position"] = np.array(pos)
            self.kinematics[link_name]["orientation"] = np.array(orn)

            mass = link_info["mass"]
            weighted_com += mass * np.array(pos)

        if self.total_mass > 0:
            self.center_of_gravity = weighted_com / self.total_mass

    def module_thrust(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")

        rotor_index_r = (module_index - 1) * 2
        rotor_index_l = rotor_index_r + 1

        thrust_right = self.rotor_thrusts[rotor_index_r]
        thrust_left = self.rotor_thrusts[rotor_index_l]

        thrust = thrust_right + thrust_left

        return thrust

    def module_force(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")

        rotor_index_r = (module_index - 1) * 2
        rotor_index_l = rotor_index_r + 1

        force_right = np.array(self.external_forces[rotor_index_r][1])
        force_left = np.array(self.external_forces[rotor_index_l][1])

        force = force_right + force_left

        return force

    def module_torque(self, module_index):
        force = self.module_force(module_index)
        position = self.link_position("F" + str(module_index)) - self.center_of_gravity
        torque = np.cross(position, force)
        return torque

    def module_wrench(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")

        force = self.module_force(module_index)
        torque = self.module_torque(module_index)

        return np.concatenate((force, torque))

    def module_phi(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")
        return self.get_joint_pos("G" + str(module_index))

    def module_theta(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")
        return self.get_joint_pos("F" + str(module_index))

    def sum_of_forces(self):
        forces = np.zeros(3)
        for _, force in self.external_forces:
            forces += np.array(force)
        return forces

    def sum_of_torques(self):
        torque = np.zeros(3)
        for position, force in self.external_forces:
            rel_pos = np.array(position) - self.center_of_gravity
            torque += np.cross(rel_pos, force)
        return torque

    def wrench(self):
        forces = self.sum_of_forces()
        torques = self.sum_of_torques()

        return np.concatenate((forces, torques))

    def link_position(self, name_or_id):
        pos, _ = self.link_pos_orn(name_or_id)
        return np.array(pos)

    def link_orientation(self, name_or_id):
        _, orn = self.link_pos_orn(name_or_id)
        return np.array(orn)

    def module_com(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")
        total_mass = 0
        com = np.array([0.0, 0.0, 0.0])
        for link in self.kinematics:
            if str(module_index) in link:
                link_info = self.kinematics[link]
                total_mass += link_info["mass"]
                com += link_info["mass"] * link_info["position"]
        if total_mass > 0:

            com /= total_mass

        return com

    def module_position(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")
        return self.link_position("G" + str(module_index))

    def module_orientation(self, module_index):
        if module_index < 1 or module_index > self.num_modules:
            raise ValueError(f"Module index must be between 1 and {self.num_modules}.")

        link_name = "L" + str(module_index)
        if link_name not in self.kinematics:
            raise ValueError(f"Module {module_index} does not have a corresponding link.")
        orn = self.kinematics[link_name]["orientation"]

        return np.array(orn)

    def pred_module_wrench(self, module_index, phi, theta, lamb, cog = None):
        R_ri = np.array(p.getMatrixFromQuaternion(self.module_orientation(module_index))).reshape(3, 3)  # Rotation matrix of the module
        r_ri = self.module_position(module_index)  # Position of the module in inertial frame

        if cog is None:
            cog = self.center_of_gravity

        cp_phi, sp_phi = np.cos(phi), np.sin(phi)
        cp_theta, sp_theta = np.cos(theta), np.sin(theta)

        # Rotation matrices
        R_phi = np.array([[1, 0, 0],
                        [0, cp_phi, -sp_phi],
                        [0, sp_phi, cp_phi]])

        R_theta = np.array([[cp_theta, 0, sp_theta],
                            [0, 1, 0],
                            [-sp_theta, 0, cp_theta]])

        e_z = np.array([0, 0, 1])  # Unit vector in z direction

        u = R_ri @ R_phi @ R_theta @ e_z

        vec_transform = np.eye(4)
        vec_transform[2, 3] = self.F_G_z

        imu_transform = np.eye(4)
        imu_transform[:3, 3] = r_ri
        imu_transform[:3, :3] = R_ri

        roll_transform = np.eye(4)
        roll_transform[:3, :3] = R_phi

        # pos: from CoG to thrust point
        pos = imu_transform @ roll_transform @ vec_transform @ np.array([0, 0, 0, 1])
        pos = pos[:3] - cog
        v = np.cross(pos, u)

        # Force and torque
        f = lamb * u
        tau = lamb * v

        W =  np.concatenate([f, tau])  # shape (6,)

        return W, (R_ri, R_phi, R_theta, u, pos)

    def linearize_module(self, module_index, phi, theta, lamb, cog = None):

        if cog is None:
            cog = self.center_of_gravity

        W, (R_ri, R_phi, R_theta, u, pos) = self.pred_module_wrench(module_index, phi, theta, lamb, cog)

        cp_phi, sp_phi = np.cos(phi), np.sin(phi)
        cp_theta, sp_theta = np.cos(theta), np.sin(theta)

        # dR_phi/dphi
        dR_phi_dphi = np.array([[0, 0, 0],
                                [0, -sp_phi, -cp_phi],
                                [0, cp_phi, -sp_phi]])

        # dR_theta/dtheta
        dR_theta_dtheta = np.array([[-sp_theta, 0, cp_theta],
                                    [0, 0, 0],
                                    [-cp_theta, 0, -sp_theta]])

        e_z = np.array([0, 0, 1])  # Unit vector in z direction

        # df/dphi
        du_dphi = R_ri @ dR_phi_dphi @ R_theta @ e_z
        df_dphi = lamb * du_dphi

        # df/dtheta
        du_dtheta = R_ri @ R_phi @ dR_theta_dtheta @ e_z
        df_dtheta = lamb * du_dtheta

        # df/dlambda
        df_dlambda = u

        # dtau/dphi
        dtau_dphi = np.cross(pos, df_dphi)

        # dtau/dtheta
        dtau_dtheta = np.cross(pos, df_dtheta)

        # dtau/dlambda
        dtau_dlambda = np.cross(pos, df_dlambda)

        # === Assemble A ===

        A = np.block([
            [df_dphi.reshape(3,1), df_dtheta.reshape(3,1), df_dlambda.reshape(3,1)],
            [dtau_dphi.reshape(3,1), dtau_dtheta.reshape(3,1), dtau_dlambda.reshape(3,1)]
        ])

        return W, A

    ######## Control Methods #######
    def thrust(self, forces):

        self.update_kinematics()

        if len(self.rotor_names) != len(forces):
            raise ValueError("Number of forces must match number of rotors.")

        self.external_forces.clear()

        self.rotor_thrusts = forces.copy()

        for rotor_name, force in zip(self.rotor_names, forces):
            position = self.kinematics[rotor_name]["position"]
            orientation = self.kinematics[rotor_name]["orientation"]
            idx = self.kinematics[rotor_name]["index"]
            local_force = [0, 0, force]
            world_force = np.array(p.rotateVector(orientation, local_force))

            p.applyExternalForce(self.robot_id, idx, world_force, position, p.WORLD_FRAME)

            self.external_forces.append((position, world_force))


    def set_joint_pos(self, name_or_id, value):
        idx = self._get_id(name_or_id)
        p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL,
                                targetPosition=value,
                                force=100.0,
                                maxVelocity=1.0)

    def set_joint_vel(self, name_or_id, value):
        idx = self._get_id(name_or_id)
        p.setJointMotorControl2(self.robot_id, idx, p.VELOCITY_CONTROL,
                                targetVelocity=value,
                                force=10.0)

    def get_joint_pos(self, name_or_id):
        idx = self._get_id(name_or_id)
        return p.getJointState(self.robot_id, idx)[0]

    def reset_joint_pos(self, name_or_id, value=0.0):
        idx = self._get_id(name_or_id)
        p.resetJointState(self.robot_id, idx, value)

    def hover(self):
        self.thrust([self.GRAVITY * self.total_mass / self.num_rotors] * self.num_rotors)

    def lock_joints(self):
        # Lock all joints to their current positions
        for i in range(self.num_links):
            self.set_joint_pos(i, p.getJointState(self.robot_id, i)[0])

    def reset_joints(self):
        # Lock all joints to their current positions
        for i in range(self.num_links):
            self.set_joint_pos(i, 0)

    def set_pos_ref(self, pos_ref):
        if len(pos_ref) != self.num_modules + 1:
            raise ValueError(f"Position reference must have {self.num_modules + 1} elements.")
        self._pos_ref = pos_ref

    def step(self):
        p.stepSimulation()
        self.update_kinematics()
        sleep(1/240)

    def animate(self):
        self._init_plot()
        ani = FuncAnimation(self._fig, self._update_plot, interval=50)
        return ani

    def reset_start_pos_orn(self, pos, orn):
        if len(pos) != 3:
            raise ValueError("Position must be a 3-element vector.")

        p.resetBasePositionAndOrientation(self.robot_id, pos, p.getQuaternionFromEuler(orn))
        self.update_kinematics()


    def plot_on_ax(self, ax, CoG = True, forces = True):

        self.update_kinematics()

        for name, info in self.kinematics.items():
            pos = info["position"]
            orn = info["orientation"]
            dims = info["dimensions"]

            if len(name) != 2:
                continue

            if name[0] == "L":
                offset = [0.15, 0, 0]
                offset = p.rotateVector(orn, offset)
                pos = [pos[i] + offset[i] for i in range(3)]

            if name[0] == "L" or name[0] == "F":
                color = self._get_color(name)
                self._draw_box(ax, pos, orn, dims, color)

        cog = self.center_of_gravity

        if CoG:
            ax.scatter([cog[0]], [cog[1]], [cog[2]], color='g', s=100, label='Center of Gravity')

        if forces:
            for pos, force in self.external_forces:
                magnitude = np.linalg.norm(force)
                ax.quiver(
                    pos[0], pos[1], pos[2],
                    force[0], force[1], force[2],
                    color='green', length=magnitude/ (self.GRAVITY or 1) * self.num_rotors, normalize=True
                )

            total_force = self.sum_of_forces()
            force_magnitude = np.linalg.norm(total_force) / self.total_mass
            if force_magnitude > 0:
                ax.quiver(
                    self.center_of_gravity[0], self.center_of_gravity[1], self.center_of_gravity[2],
                    total_force[0], total_force[1], total_force[2],
                    color='black', length=force_magnitude/ (self.GRAVITY or 1), normalize=True
                )

            total_torque = self.sum_of_torques()
            torque_magnitude = np.linalg.norm(total_torque)
            if torque_magnitude > 0:
                ax.quiver(
                    self.center_of_gravity[0], self.center_of_gravity[1], self.center_of_gravity[2],
                    total_torque[0], total_torque[1], total_torque[2],
                    color='red', length=torque_magnitude, normalize=True
                )

            for i in range(self.num_modules):
                wrench = self.module_wrench(i + 1)
                module_pos = self.module_position(i + 1)

                ax.quiver(
                    module_pos[0], module_pos[1], module_pos[2],
                    wrench[0], wrench[1], wrench[2],
                    color='blue', length=np.linalg.norm(wrench[:3]) / 2, normalize=True
                )

                ax.quiver(
                    module_pos[0], module_pos[1], module_pos[2],
                    wrench[3], wrench[4], wrench[5],
                    color='orange', length=np.linalg.norm(wrench[:3]) / 2, normalize=True
                )

    ##### Private Methods #####
    def _draw_box(self, ax, pos, orn, dims, color):
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

    def _init_plot(self):
        self._fig = plt.figure(figsize=(10, 10))
        self._ax_world = self._fig.add_subplot(121, projection='3d')
        self._ax_robot = self._fig.add_subplot(122, projection='3d')
        self._scatter_cog_world = self._ax_world.scatter([], [], [], color='g', s=100, label='Center of Gravity')
        self._scatter_cog_robot = self._ax_robot.scatter([], [], [], color='g', s=100, label='Center of Gravity')
        self._scatter_pos_ref = [self._ax_world.scatter([], [], [], color='b', s=50) for _ in range(self.num_modules + 1)]

        # self._plot_plane(self._ax_world)
        self._ax_world.set_xlim([-0.5, 3])
        self._ax_world.set_ylim([-0.5, 3])
        self._ax_world.set_zlim([-1, 1])
        self._ax_world.set_xlabel('X')
        self._ax_world.set_ylabel('Y')
        self._ax_world.set_zlabel('Z')
        self._ax_world.set_title('Flight Simulation')

        self._ax_robot.set_xlim([-1, 1])
        self._ax_robot.set_ylim([-1, 1])
        self._ax_robot.set_zlim([-1, 1])
        self._ax_robot.set_xlabel('X')
        self._ax_robot.set_ylabel('Y')
        self._ax_robot.set_zlabel('Z')
        self._ax_robot.set_title('Forces')

        self._box_artists = []

    def _update_plot(self, frame):
        for artist in self._drawn_artists:
            artist.remove()
        self._drawn_artists.clear()

        for name, info in self.kinematics.items():
            pos = info["position"]
            orn = info["orientation"]
            dims = info["dimensions"]

            if len(name) != 2:
                continue

            if name[0] == "L":
                offset = [0.15, 0, 0]
                offset = p.rotateVector(orn, offset)
                pos = [pos[i] + offset[i] for i in range(3)]

            if name[0] == "L" or name[0] == "F":
                color = self._get_color(name)
                self._draw_box(self._ax_world, pos, orn, dims, color)

                rel_pos = pos - self.center_of_gravity
                self._draw_box(self._ax_robot, rel_pos, orn, dims, color)

        cog = self.center_of_gravity
        self._scatter_cog_world._offsets3d = ([cog[0]], [cog[1]], [cog[2]])
        self._scatter_cog_robot._offsets3d = ([0], [0], [0])

        for pos, force in self.external_forces:
            rel_pos = pos - self.center_of_gravity
            magnitude = np.linalg.norm(force)
            arrow = self._ax_robot.quiver(
                rel_pos[0], rel_pos[1], rel_pos[2],
                force[0], force[1], force[2],
                color='green', length=magnitude/ (self.GRAVITY or 1) * self.num_rotors, normalize=True
            )
            self._drawn_artists.append(arrow)

        total_force = self.sum_of_forces()
        force_magnitude = np.linalg.norm(total_force) / self.total_mass
        if force_magnitude > 0:
            arrow = self._ax_robot.quiver(
                0, 0, 0,
                total_force[0], total_force[1], total_force[2],
                color='black', length=force_magnitude/ (self.GRAVITY or 1), normalize=True
            )

            self._drawn_artists.append(arrow)

        total_torque = self.sum_of_torques()
        torque_magnitude = np.linalg.norm(total_torque)
        if torque_magnitude > 0:
            arrow = self._ax_robot.quiver(
                0, 0, 0,
                total_torque[0], total_torque[1], total_torque[2],
                color='red', length=torque_magnitude, normalize=True
            )
            self._drawn_artists.append(arrow)

        # Plot pos ref
        if self._pos_ref is not None and len(self._pos_ref) > 0:
            for i, pos_ref in enumerate(self._pos_ref):
                if len(pos_ref) == 3:
                    self._scatter_pos_ref[i]._offsets3d = ([pos_ref[0]], [pos_ref[1]], [pos_ref[2]])
                else:
                    self._scatter_pos_ref[i]._offsets3d = ([0], [0], [0])

    def _plot_plane(self, ax, size=2.0, z=0.0, color='gray', alpha=0.3):
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

    def _get_color(self, link):
        link = link.lower()
        if "rotor" in link or link[0] == "f":
            return "black"
                # rotate the offset vector to match the orientation of the link
        if link[0] == "l" or link[0] == "g":
            return "red"
        return "blue"

    def _get_id(self, name_or_id):
        if isinstance(name_or_id, int):
            return name_or_id
        else:
            try:
                return self.kinematics[name_or_id]["index"]
            except KeyError:

                for name, info in self.kinematics.items():
                    if name_or_id.lower() in name.lower():
                        return info["index"]

                raise ValueError(f"Joint '{name_or_id}' not found.")